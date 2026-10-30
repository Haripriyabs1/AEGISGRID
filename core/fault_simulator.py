"""
Fault Simulator Module
Simulates realistic physical electrical distribution faults:
  - Three-Phase Bolted Short-Circuit
  - Single Line-to-Ground (SLG) Fault
  - Line-to-Line (L-L) Fault
  - High Impedance Fault (HIF)
with variable fault impedances, fault locations along the line, and sensor telemetry noise.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np
import pandapower as pp
from core.power_flow import solve_power_flow, PowerFlowResults


class FaultType(str, Enum):
    NORMAL = "Normal (No Fault)"
    THREE_PHASE_BOLTED = "Three-Phase Bolted Short-Circuit"
    SINGLE_LINE_TO_GROUND = "Single Line-to-Ground (SLG) Fault"
    LINE_TO_LINE = "Line-to-Line (L-L) Fault"
    HIGH_IMPEDANCE = "High Impedance Fault (HIF)"


@dataclass
class FaultEvent:
    fault_type: FaultType
    line_id: int
    from_bus: int
    to_bus: int
    location_percent: float  # 0% to 100% distance from from_bus
    fault_impedance_ohm: float
    simulated_noise_percent: float
    description: str


@dataclass
class TelemetryData:
    prefault_voltages: Dict[int, float]
    fault_voltages: Dict[int, float]
    voltage_sags: Dict[int, float]
    prefault_currents: Dict[int, float]
    fault_currents: Dict[int, float]
    current_surges: Dict[int, float]
    fault_line_id: int
    fault_type_name: str
    feature_vector: np.ndarray


def simulate_fault(
    net: pp.pandapowerNet,
    fault_type: FaultType,
    line_id: int,
    location_percent: float = 50.0,
    fault_impedance_ohm: float = 1.0,
    sensor_noise_percent: float = 0.5,
    base_res: Optional[PowerFlowResults] = None
) -> Tuple[pp.pandapowerNet, PowerFlowResults, TelemetryData, FaultEvent]:
    """
    Simulates a physical fault event on a specified line in the distribution network.
    Returns:
      - Modified network during the fault
      - Power flow results during the fault
      - Raw & processed PMU/meter sensor telemetry data
      - Fault event metadata
    """
    fault_net = copy.deepcopy(net)

    if fault_type == FaultType.NORMAL or line_id not in fault_net.line.index:
        # Normal baseline operation
        res = base_res if base_res is not None else solve_power_flow(fault_net)
        
        # Build consistent feature vector [V, Sag, I, Surge]
        v_arr = np.array([res.bus_voltages.get(b, 1.0) for b in sorted(res.bus_voltages.keys())])
        sag_arr = np.zeros_like(v_arr)
        i_arr = np.array([res.line_currents_ka.get(l, 0.0) for l in sorted(res.line_currents_ka.keys())])
        surge_arr = np.zeros_like(i_arr)
        feat_vec = np.concatenate([v_arr, sag_arr, i_arr, surge_arr])

        telemetry = TelemetryData(
            prefault_voltages=res.bus_voltages,
            fault_voltages=res.bus_voltages,
            voltage_sags={b: 0.0 for b in res.bus_voltages},
            prefault_currents=res.line_currents_ka,
            fault_currents=res.line_currents_ka,
            current_surges={l: 0.0 for l in res.line_currents_ka},
            fault_line_id=-1,
            fault_type_name=FaultType.NORMAL.value,
            feature_vector=feat_vec
        )
        event = FaultEvent(
            fault_type=FaultType.NORMAL,
            line_id=-1,
            from_bus=-1,
            to_bus=-1,
            location_percent=0.0,
            fault_impedance_ohm=0.0,
            simulated_noise_percent=0.0,
            description="System operating in healthy, steady-state condition."
        )
        return fault_net, res, telemetry, event

    # Baseline pre-fault power flow
    if base_res is None:
        base_res = solve_power_flow(net)


    from_bus = int(fault_net.line.at[line_id, 'from_bus'])
    to_bus = int(fault_net.line.at[line_id, 'to_bus'])
    vn_kv = float(fault_net.bus.at[from_bus, 'vn_kv'])

    # Determine electrical characteristics of the fault
    # Fault impedance: lower Rf produces more severe fault currents and deeper sags
    if fault_type == FaultType.THREE_PHASE_BOLTED:
        rf = max(0.05, fault_impedance_ohm * 0.2)
        xf = rf * 0.5
        severity_factor = 8.0
        desc = f"Severe 3-Phase Bolted Fault on Line {line_id} (Bus {from_bus}->{to_bus}) at {location_percent:.1f}% length. High symmetrical short-circuit current."
    elif fault_type == FaultType.SINGLE_LINE_TO_GROUND:
        rf = max(0.8, fault_impedance_ohm * 1.5)
        xf = rf * 0.8
        severity_factor = 4.5
        desc = f"Single Line-to-Ground (SLG) Fault on Line {line_id} (Bus {from_bus}->{to_bus}) at {location_percent:.1f}% length. Phase-to-earth flashover."
    elif fault_type == FaultType.LINE_TO_LINE:
        rf = max(0.4, fault_impedance_ohm * 0.8)
        xf = rf * 0.6
        severity_factor = 6.0
        desc = f"Line-to-Line (L-L) Fault on Line {line_id} (Bus {from_bus}->{to_bus}) at {location_percent:.1f}% length. Cross-phase conductor contact."
    elif fault_type == FaultType.HIGH_IMPEDANCE:
        rf = max(25.0, fault_impedance_ohm * 30.0)
        xf = rf * 0.3
        severity_factor = 1.4
        desc = f"High Impedance Fault (HIF) on Line {line_id} (Bus {from_bus}->{to_bus}) at {location_percent:.1f}% length. Downed conductor arcing with dry ground."
    else:
        rf = 5.0
        xf = 2.0
        severity_factor = 3.0
        desc = f"General Fault on Line {line_id}"

    # Target bus nearest to the fault location along line
    target_bus = from_bus if location_percent < 50.0 else to_bus
    
    # Calculate equivalent fault power injection
    # S = V^2 / Z_fault
    z_mag_sq = rf**2 + xf**2
    p_fault_mw = (vn_kv**2) * rf / z_mag_sq
    q_fault_mvar = (vn_kv**2) * xf / z_mag_sq

    # Limit to physically realistic simulation bounds so Newton-Raphson solver converges
    p_fault_mw = min(p_fault_mw, 18.0)
    q_fault_mvar = min(q_fault_mvar, 12.0)

    # Inject fault load into target bus
    f_load_id = pp.create_load(
        fault_net,
        bus=target_bus,
        p_mw=p_fault_mw,
        q_mvar=q_fault_mvar,
        name=f"Fault Shunt ({fault_type.name})"
    )

    # Solve power flow during fault
    fault_res = solve_power_flow(fault_net)

    # If power flow diverged due to extreme fault severity, synthesize realistic short-circuit profile
    if not fault_res.converged or fault_res.min_vm_pu < 0.1:
        # Analytical voltage sag profile based on electrical distance
        synthesized_voltages = {}
        for b in net.bus.index:
            b_int = int(b)
            dist = abs(b_int - target_bus)
            # exponential sag decay away from fault
            sag = max(0.05, 1.0 - (0.85 / (1.0 + 0.35 * dist)))
            synthesized_voltages[b_int] = sag
        fault_res.bus_voltages = synthesized_voltages
        fault_res.min_vm_pu = synthesized_voltages[target_bus]
        fault_res.min_vm_bus = target_bus
        fault_res.status_message = "Fault Short-Circuit State (Analytically Solved)"

    # Add realistic sensor telemetry noise (smart meter / PMU telemetry inaccuracies)
    noise_sigma = sensor_noise_percent / 100.0
    noisy_voltages = {}
    voltage_sags = {}
    for b_id, v_val in fault_res.bus_voltages.items():
        noise = np.random.normal(0, noise_sigma * max(v_val, 0.1))
        noisy_v = float(np.clip(v_val + noise, 0.0, 1.25))
        noisy_voltages[b_id] = noisy_v
        prefault_v = base_res.bus_voltages.get(b_id, 1.0)
        voltage_sags[b_id] = float(prefault_v - noisy_v)

    noisy_currents = {}
    current_surges = {}
    for l_id, i_val in fault_res.line_currents_ka.items():
        # High surge on lines feeding into the fault
        is_feed_line = (l_id == line_id)
        surge_mult = severity_factor if is_feed_line else 1.0
        val = i_val * surge_mult
        noise = np.random.normal(0, noise_sigma * max(val, 0.01))
        noisy_i = float(max(0.0, val + noise))
        noisy_currents[l_id] = noisy_i
        prefault_i = base_res.line_currents_ka.get(l_id, 0.05)
        current_surges[l_id] = float(noisy_i - prefault_i)

    fault_res.bus_voltages = noisy_voltages
    fault_res.line_currents_ka = noisy_currents

    # Construct feature vector for AI Engine:
    # [v_1, ..., v_N, sag_1, ..., sag_N, i_1, ..., i_M, surge_1, ..., surge_M]
    v_arr = np.array([noisy_voltages[b] for b in sorted(noisy_voltages.keys())])
    sag_arr = np.array([voltage_sags[b] for b in sorted(voltage_sags.keys())])
    i_arr = np.array([noisy_currents[l] for l in sorted(noisy_currents.keys())])
    surge_arr = np.array([current_surges[l] for l in sorted(current_surges.keys())])
    feature_vector = np.concatenate([v_arr, sag_arr, i_arr, surge_arr])

    telemetry = TelemetryData(
        prefault_voltages=base_res.bus_voltages,
        fault_voltages=noisy_voltages,
        voltage_sags=voltage_sags,
        prefault_currents=base_res.line_currents_ka,
        fault_currents=noisy_currents,
        current_surges=current_surges,
        fault_line_id=line_id,
        fault_type_name=fault_type.value,
        feature_vector=feature_vector
    )

    event = FaultEvent(
        fault_type=fault_type,
        line_id=line_id,
        from_bus=from_bus,
        to_bus=to_bus,
        location_percent=location_percent,
        fault_impedance_ohm=fault_impedance_ohm,
        simulated_noise_percent=sensor_noise_percent,
        description=desc
    )

    return fault_net, fault_res, telemetry, event
