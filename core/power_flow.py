"""
Power Flow Module
Wrapper for pandapower AC power flow calculations, parameter extraction,
topology connectivity analysis, and electrical constraint violation detection.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx

# Suppress verbose numba and convergence warnings
warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)


@dataclass
class PowerFlowResults:
    """Encapsulates comprehensive electrical state results from power flow analysis."""
    converged: bool
    status_message: str
    total_load_mw: float
    served_load_mw: float
    unserved_load_mw: float
    restoration_percentage: float
    total_loss_mw: float
    total_loss_kw: float
    total_loss_pct: float
    min_vm_pu: float
    min_vm_bus: int
    max_vm_pu: float
    max_vm_bus: int
    avg_vm_pu: float
    max_line_loading: float
    max_loading_line: int
    overloaded_lines: List[int] = field(default_factory=list)
    undervoltage_buses: List[int] = field(default_factory=list)
    overvoltage_buses: List[int] = field(default_factory=list)
    isolated_buses: List[int] = field(default_factory=list)
    bus_voltages: Dict[int, float] = field(default_factory=dict)
    line_loadings: Dict[int, float] = field(default_factory=dict)
    line_currents_ka: Dict[int, float] = field(default_factory=dict)


def analyze_network_connectivity(net: pp.pandapowerNet) -> Tuple[Set[int], Set[int], nx.Graph]:
    """
    Constructs active graph topology and determines:
      1. Energized buses (connected to an active ext_grid or DG in service)
      2. De-energized / isolated buses
    """
    G = nx.Graph()
    for b in net.bus.index:
        G.add_node(int(b))

    # Add active lines
    active_lines = net.line[net.line.in_service == True]
    for idx, row in active_lines.iterrows():
        fb = int(row['from_bus'])
        tb = int(row['to_bus'])
        G.add_edge(fb, tb, line_id=idx)

    # Infeed buses (external grids)
    source_buses = set(net.ext_grid[net.ext_grid.in_service == True]['bus'].astype(int))
    
    # Also include online generators / storages if capable of islanding
    # (In standard distribution, ext_grid is the slack/infeed source)
    energized_buses = set()
    for src in source_buses:
        if src in G:
            connected = nx.node_connected_component(G, src)
            energized_buses.update(connected)

    all_buses = set(net.bus.index.astype(int))
    isolated_buses = all_buses - energized_buses

    return energized_buses, isolated_buses, G


def solve_power_flow(net: pp.pandapowerNet, v_min_limit: float = 0.95, v_max_limit: float = 1.05,
                     loading_limit: float = 100.0) -> PowerFlowResults:
    """
    Runs AC Newton-Raphson power flow on the pandapower network,
    accurately accounting for any isolated islands and extracting state variables.
    """
    energized_buses, isolated_buses, G = analyze_network_connectivity(net)
    
    # Calculate total nominal active load
    active_loads = net.load[net.load.in_service == True]
    total_load_mw = float(active_loads['p_mw'].sum()) if len(active_loads) > 0 else 0.0

    # Calculate unserved load from isolated buses
    isolated_loads = net.load[net.load.bus.isin(isolated_buses) & (net.load.in_service == True)]
    unserved_load_mw = float(isolated_loads['p_mw'].sum()) if len(isolated_loads) > 0 else 0.0
    served_load_mw = max(0.0, total_load_mw - unserved_load_mw)
    restoration_pct = (served_load_mw / total_load_mw * 100.0) if total_load_mw > 0 else 100.0

    # Temporarily disconnect loads in isolated buses to allow pandapower power flow to solve the energized island
    orig_load_status = net.load['in_service'].copy()
    if isolated_buses:
        net.load.loc[net.load.bus.isin(isolated_buses), 'in_service'] = False

    converged = False
    try:
        # Run pandapower AC power flow without numba
        pp.runpp(net, algorithm='nr', numba=False, check_connectivity=False, enforce_q_lims=False)
        converged = True
        status_msg = "AC Power Flow Converged"
    except Exception as e:
        # Fallback to DC power flow or state estimation if AC diverged
        try:
            pp.rundcpp(net, numba=False)
            converged = True
            status_msg = "AC Power Flow Diverged; DC Power Flow Fallback Converged"
        except Exception as e2:
            converged = False
            status_msg = f"Power Flow Diverged: {str(e)[:100]}"

    # Restore original load status
    net.load['in_service'] = orig_load_status

    # Extract metrics
    bus_voltages = {}
    line_loadings = {}
    line_currents = {}
    overloaded_lines = []
    undervoltage_buses = []
    overvoltage_buses = []

    for b in net.bus.index:
        b_int = int(b)
        if b_int in isolated_buses:
            vm = 0.0
        elif converged and hasattr(net, 'res_bus') and b in net.res_bus.index and not np.isnan(net.res_bus.at[b, 'vm_pu']):
            vm = float(net.res_bus.at[b, 'vm_pu'])
            if vm < v_min_limit:
                undervoltage_buses.append(b_int)
            elif vm > v_max_limit:
                overvoltage_buses.append(b_int)
        else:
            vm = 0.0
            undervoltage_buses.append(b_int)
        bus_voltages[b_int] = vm

    for l_idx in net.line.index:
        l_int = int(l_idx)
        if not net.line.at[l_idx, 'in_service']:
            loading = 0.0
            i_ka = 0.0
        elif converged and hasattr(net, 'res_line') and l_idx in net.res_line.index and not np.isnan(net.res_line.at[l_idx, 'loading_percent']):
            loading = float(net.res_line.at[l_idx, 'loading_percent'])
            i_ka = float(net.res_line.at[l_idx, 'i_ka']) if 'i_ka' in net.res_line.columns else 0.0
            if loading > loading_limit:
                overloaded_lines.append(l_int)
        else:
            loading = 0.0
            i_ka = 0.0
        line_loadings[l_int] = loading
        line_currents[l_int] = i_ka

    # Losses
    if converged and hasattr(net, 'res_line') and 'pl_mw' in net.res_line.columns:
        total_loss_mw = float(net.res_line['pl_mw'].sum())
    else:
        total_loss_mw = 0.0

    total_loss_kw = total_loss_mw * 1000.0
    total_loss_pct = (total_loss_mw / total_load_mw * 100.0) if total_load_mw > 0 else 0.0

    # Energized bus voltages statistics
    energized_vols = [v for b, v in bus_voltages.items() if b in energized_buses and v > 0.1]
    if energized_vols:
        min_v = float(np.min(energized_vols))
        min_b = [b for b, v in bus_voltages.items() if abs(v - min_v) < 1e-4 and b in energized_buses][0]
        max_v = float(np.max(energized_vols))
        max_b = [b for b, v in bus_voltages.items() if abs(v - max_v) < 1e-4 and b in energized_buses][0]
        avg_v = float(np.mean(energized_vols))
    else:
        min_v, min_b, max_v, max_b, avg_v = 0.0, 0, 0.0, 0, 0.0

    # Max line loading
    active_loadings = [ld for l_id, ld in line_loadings.items() if net.line.at[l_id, 'in_service']]
    if active_loadings:
        max_load = float(np.max(active_loadings))
        max_load_l = [l_id for l_id, ld in line_loadings.items() if abs(ld - max_load) < 1e-4 and net.line.at[l_id, 'in_service']][0]
    else:
        max_load, max_load_l = 0.0, 0

    return PowerFlowResults(
        converged=converged,
        status_message=status_msg,
        total_load_mw=total_load_mw,
        served_load_mw=served_load_mw,
        unserved_load_mw=unserved_load_mw,
        restoration_percentage=restoration_pct,
        total_loss_mw=total_loss_mw,
        total_loss_kw=total_loss_kw,
        total_loss_pct=total_loss_pct,
        min_vm_pu=min_v,
        min_vm_bus=min_b,
        max_vm_pu=max_v,
        max_vm_bus=max_b,
        avg_vm_pu=avg_v,
        max_line_loading=max_load,
        max_loading_line=max_load_l,
        overloaded_lines=overloaded_lines,
        undervoltage_buses=undervoltage_buses,
        overvoltage_buses=overvoltage_buses,
        isolated_buses=list(isolated_buses),
        bus_voltages=bus_voltages,
        line_loadings=line_loadings,
        line_currents_ka=line_currents
    )
