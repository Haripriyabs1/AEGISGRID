"""
Network Builder Module
Builds standard IEEE 33-Bus benchmark and 16-Bus Multi-Substation Urban Microgrid
with realistic electrical parameters, tie-switches, load priorities, and renewable DGs.
"""

import copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn


def build_ieee33_network(include_dg: bool = True, load_scaling: float = 1.0) -> pp.pandapowerNet:
    """
    Constructs the IEEE 33-Bus Radial Distribution Network (Baran & Wu benchmark)
    augmented with:
      - 5 Normally Open (NO) Tie-Switches for self-healing reconfiguration
      - Realistic branch ampacity ratings (max_i_ka)
      - Bus coordinates for graphical visualization
      - Load priority classifications (Critical, Commercial, Residential)
      - Renewable Distributed Generation (Solar PV and Battery Energy Storage)
    """
    net = pn.case33bw()
    
    # Scale loads
    if load_scaling != 1.0:
        net.load['p_mw'] *= load_scaling
        net.load['q_mvar'] *= load_scaling

    # Assign realistic ampacity ratings (0.35 kA for main feeder trunk, 0.22 kA for laterals)
    # Baran & Wu standard base voltage is 12.66 kV
    # 0.35 kA at 12.66 kV corresponds to ~7.67 MVA thermal limit
    net.line['max_i_ka'] = 0.22  # default for lateral branches & tie lines
    # Main trunk lines 0 to 16
    net.line.loc[0:16, 'max_i_ka'] = 0.38

    # Tie lines indices in pandapower case33bw are 32, 33, 34, 35, 36
    tie_line_indices = [32, 33, 34, 35, 36]
    for tl in tie_line_indices:
        net.line.at[tl, 'in_service'] = False
        net.line.at[tl, 'name'] = f"Tie-Line {tl} (Bus {net.line.at[tl, 'from_bus']}-{net.line.at[tl, 'to_bus']})"

    # Clean line naming
    for i in range(32):
        fb = net.line.at[i, 'from_bus']
        tb = net.line.at[i, 'to_bus']
        net.line.at[i, 'name'] = f"Line {i} (Bus {fb}-{tb})"

    # Define clean, orthogonal distribution grid bus coordinates (X, Y)
    bus_coords = {}
    # Feeder Main Trunk (0 to 17) along horizontal line Y = 3.0
    for b in range(18):
        bus_coords[b] = (float(b), 3.0)
    
    # Lateral Feeder 1: Buses 18 to 21 tapped off Bus 1, going upwards Y = 4.5
    for idx, b in enumerate(range(18, 22)):
        bus_coords[b] = (float(idx + 1) * 1.5, 4.5)
        
    # Lateral Feeder 2: Buses 22 to 24 tapped off Bus 2, going downwards Y = 1.8
    for idx, b in enumerate(range(22, 25)):
        bus_coords[b] = (float(idx + 2) * 1.2, 1.8)
        
    # Lateral Feeder 3: Buses 25 to 32 tapped off Bus 5, going downwards Y = 0.5
    for idx, b in enumerate(range(25, 33)):
        bus_coords[b] = (float(idx + 5) * 1.3, 0.5)

    # Attach coordinates to net
    net.bus_coords = bus_coords

    # Classify Load Priorities
    # Priority 1: Critical (Hospital, Emergency center, Data center, Pumping station)
    # Priority 2: Commercial (Commercial complexes, Offices, Light industry)
    # Priority 3: Residential (General domestic residential consumers)
    load_priorities = {}
    critical_buses = [7, 13, 24, 29]
    commercial_buses = [3, 8, 17, 21, 31]
    
    for idx, row in net.load.iterrows():
        bus = int(row['bus'])
        if bus in critical_buses:
            priority = 1
            cat = "Critical"
        elif bus in commercial_buses:
            priority = 2
            cat = "Commercial"
        else:
            priority = 3
            cat = "Residential"
        load_priorities[bus] = {"priority": priority, "category": cat, "p_mw": row['p_mw'], "q_mvar": row['q_mvar']}
    
    net.load_priorities = load_priorities

    # Add Distributed Generation (Solar PV & BESS)
    dg_info = []
    if include_dg:
        # Solar PV at Bus 17 (Trunk end - voltage support)
        s1 = pp.create_sgen(net, bus=17, p_mw=0.45, q_mvar=0.05, name="Solar PV Alpha (0.45 MW)", type="PV", in_service=True)
        dg_info.append({"id": s1, "bus": 17, "name": "Solar PV Alpha", "p_mw": 0.45, "type": "Solar PV"})

        # Battery Energy Storage System (BESS) at Bus 32 (Lateral 3 end)
        s2 = pp.create_storage(net, bus=32, p_mw=0.35, max_e_mwh=1.2, soc_percent=85.0, name="BESS Beta (0.35 MW / 1.2 MWh)", in_service=True)
        dg_info.append({"id": s2, "bus": 32, "name": "BESS Beta", "p_mw": 0.35, "type": "BESS Storage"})

        # Solar PV at Bus 24 (Lateral 2 end)
        s3 = pp.create_sgen(net, bus=24, p_mw=0.30, q_mvar=0.02, name="Solar PV Gamma (0.30 MW)", type="PV", in_service=True)
        dg_info.append({"id": s3, "bus": 24, "name": "Solar PV Gamma", "p_mw": 0.30, "type": "Solar PV"})

    net.dg_info = dg_info
    net.tie_lines = tie_line_indices
    net.meta_network_name = "IEEE 33-Bus Distribution Network"
    
    return net


def build_urban16_network(include_dg: bool = True, load_scaling: float = 1.0) -> pp.pandapowerNet:
    """
    Constructs a 16-Bus Multi-Substation Urban Smart Microgrid.
    Features:
      - 2 Grid Infeeds / Substations (Substation Alpha at Bus 0, Substation Beta at Bus 15)
      - Dual Radial Feeders (Feeder 1: Buses 0-7, Feeder 2: Buses 8-15)
      - 3 Inter-Feeder Tie-Lines with NO tie-switches
      - High resilience testbed for loop self-healing and microgrid restoration
    """
    net = pp.create_empty_network(name="Urban 16-Bus Smart Distribution Microgrid")
    vn_kv = 11.0  # 11 kV standard urban distribution

    # Create 16 Buses
    for i in range(16):
        pp.create_bus(net, vn_kv=vn_kv, name=f"Bus {i}")

    # Substations / External Grids
    pp.create_ext_grid(net, bus=0, vm_pu=1.02, name="Substation Alpha (110kV/11kV Infeed)")
    pp.create_ext_grid(net, bus=15, vm_pu=1.02, name="Substation Beta (110kV/11kV Infeed)")

    # Standard distribution cable type
    std_r = 0.32  # ohm/km
    std_x = 0.25  # ohm/km
    trunk_max_i = 0.35  # kA (~6.6 MVA)
    tie_max_i = 0.25  # kA

    # Feeder 1 branches: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7
    f1_lines = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for idx, (fb, tb) in enumerate(f1_lines):
        pp.create_line_from_parameters(net, from_bus=fb, to_bus=tb, length_km=0.8,
                                       r_ohm_per_km=std_r, x_ohm_per_km=std_x, c_nf_per_km=10.0,
                                       max_i_ka=trunk_max_i, name=f"F1 Line {fb}-{tb}", in_service=True)

    # Feeder 2 branches: 15-14, 14-13, 13-12, 12-11, 11-10, 10-9, 9-8
    f2_lines = [(15, 14), (14, 13), (13, 12), (12, 11), (11, 10), (10, 9), (9, 8)]
    for idx, (fb, tb) in enumerate(f2_lines):
        pp.create_line_from_parameters(net, from_bus=fb, to_bus=tb, length_km=0.8,
                                       r_ohm_per_km=std_r, x_ohm_per_km=std_x, c_nf_per_km=10.0,
                                       max_i_ka=trunk_max_i, name=f"F2 Line {fb}-{tb}", in_service=True)

    # Tie Lines (Normally Open):
    # Tie 1: Bus 3 to Bus 12
    # Tie 2: Bus 5 to Bus 10
    # Tie 3: Bus 7 to Bus 8
    tie_pairs = [(3, 12), (5, 10), (7, 8)]
    tie_line_indices = []
    for fb, tb in tie_pairs:
        l_idx = pp.create_line_from_parameters(net, from_bus=fb, to_bus=tb, length_km=1.2,
                                               r_ohm_per_km=std_r * 1.1, x_ohm_per_km=std_x * 1.1, c_nf_per_km=10.0,
                                               max_i_ka=tie_max_i, name=f"Tie-Line {fb}-{tb}", in_service=False)
        tie_line_indices.append(l_idx)

    # Loads across buses (typical urban load 0.2 MW to 0.6 MW per bus)
    load_data = {
        1: (0.35, 0.12), 2: (0.45, 0.15), 3: (0.50, 0.18), 4: (0.30, 0.10),
        5: (0.40, 0.14), 6: (0.55, 0.20), 7: (0.30, 0.11), 8: (0.35, 0.12),
        9: (0.45, 0.16), 10: (0.50, 0.17), 11: (0.30, 0.10), 12: (0.60, 0.22),
        13: (0.40, 0.15), 14: (0.35, 0.12)
    }

    load_priorities = {}
    critical_buses = [3, 6, 10, 12]
    commercial_buses = [2, 5, 9, 13]

    for bus_id, (p_val, q_val) in load_data.items():
        p_val *= load_scaling
        q_val *= load_scaling
        pp.create_load(net, bus=bus_id, p_mw=p_val, q_mvar=q_val, name=f"Load Bus {bus_id}")
        
        if bus_id in critical_buses:
            cat = "Critical"
            priority = 1
        elif bus_id in commercial_buses:
            cat = "Commercial"
            priority = 2
        else:
            cat = "Residential"
            priority = 3
        load_priorities[bus_id] = {"priority": priority, "category": cat, "p_mw": p_val, "q_mvar": q_val}

    net.load_priorities = load_priorities

    # Bus coordinates for clean side-by-side feeder visualization
    bus_coords = {}
    # Feeder 1 at Y = 4.0 (buses 0 to 7)
    for b in range(8):
        bus_coords[b] = (float(b) * 2.0, 4.0)
    # Feeder 2 at Y = 1.0 (buses 8 to 15, bus 15 at right or left)
    for idx, b in enumerate(range(8, 16)):
        bus_coords[b] = (float(idx) * 2.0, 1.0)

    net.bus_coords = bus_coords

    # Distributed Generation
    dg_info = []
    if include_dg:
        s1 = pp.create_sgen(net, bus=6, p_mw=0.40, q_mvar=0.05, name="Urban Solar Park 1 (0.4 MW)", type="PV", in_service=True)
        dg_info.append({"id": s1, "bus": 6, "name": "Urban Solar Park 1", "p_mw": 0.40, "type": "Solar PV"})

        s2 = pp.create_storage(net, bus=11, p_mw=0.35, max_e_mwh=1.0, soc_percent=90.0, name="City BESS 2 (0.35 MW)", in_service=True)
        dg_info.append({"id": s2, "bus": 11, "name": "City BESS 2", "p_mw": 0.35, "type": "BESS Storage"})

    net.dg_info = dg_info
    net.tie_lines = tie_line_indices
    net.meta_network_name = "16-Bus Urban Smart Microgrid"

    return net


def get_network(network_type: str = "ieee33", include_dg: bool = True, load_scaling: float = 1.0) -> pp.pandapowerNet:
    """
    Factory function to retrieve initialized distribution network model.
    """
    if "16" in str(network_type).lower():
        return build_urban16_network(include_dg=include_dg, load_scaling=load_scaling)
    else:
        return build_ieee33_network(include_dg=include_dg, load_scaling=load_scaling)
