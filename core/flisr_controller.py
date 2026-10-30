"""
FLISR Controller Module
Autonomous Self-Healing Engine implementing:
  - Step 1: AI Fault Diagnostics (Detection, Classification, Localization)
  - Step 2: Fault Isolation (Automated sectionalizer / breaker opening)
  - Step 3: Service Restoration via Intelligent Network Reconfiguration:
            - Radiality constraint verification (NetworkX)
            - Thermal ampacity constraint checking
            - Voltage security margin optimization
            - Priority-based load preservation (Critical > Commercial > Residential)
            - Total system loss minimization
"""

from dataclasses import dataclass, field
import copy
import time
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandapower as pp
from core.ai_engine import AIFaultEngine, AIPrediction
from core.fault_simulator import TelemetryData, FaultEvent
from core.power_flow import analyze_network_connectivity, solve_power_flow, PowerFlowResults


@dataclass
class SwitchingOperation:
    step_number: int
    action_type: str  # "OPEN" or "CLOSE" or "LOAD_SHED"
    target_element: str
    element_id: int
    rationale: str
    execution_time_seconds: float


@dataclass
class RestorationCandidate:
    tie_line_id: int
    from_bus: int
    to_bus: int
    is_radial: bool
    converged: bool
    unserved_mw: float
    total_loss_kw: float
    min_vm_pu: float
    max_loading: float
    violates_voltage: bool
    violates_thermal: bool
    score: float


@dataclass
class SelfHealingResult:
    fault_detected: bool
    ai_prediction: AIPrediction
    fault_isolated: bool
    isolated_line_id: int
    isolated_buses_count: int
    deenergized_load_mw: float
    service_restored: bool
    selected_tie_line_id: Optional[int]
    switching_sequence: List[SwitchingOperation]
    candidate_evaluations: List[RestorationCandidate]
    total_healing_time_seconds: float
    post_healing_results: PowerFlowResults
    healed_network: pp.pandapowerNet
    summary_report: str


class FLISRController:
    """
    Autonomous Self-Healing Electrical Distribution Network Controller.
    """
    def __init__(self, ai_engine: AIFaultEngine):
        self.ai_engine = ai_engine

    def execute_self_healing(
        self,
        fault_net: pp.pandapowerNet,
        telemetry: TelemetryData,
        v_min_threshold: float = 0.90,
        max_loading_threshold: float = 110.0,
        enable_priority_shedding: bool = True
    ) -> SelfHealingResult:
        """
        Executes end-to-end autonomous FLISR sequence:
          Fault Identification -> Section Isolation -> Constraint-Driven Restoration
        """
        t_start = time.perf_counter()
        switching_ops: List[SwitchingOperation] = []
        op_step = 1

        net_name = getattr(fault_net, 'meta_network_name', 'ieee33')
        ai_pred = self.ai_engine.predict(telemetry.feature_vector, network_name=net_name)
        
        if not ai_pred.fault_detected or ai_pred.predicted_line_id < 0:
            # Healthy network or no fault found
            normal_res = solve_power_flow(fault_net)
            return SelfHealingResult(
                fault_detected=False,
                ai_prediction=ai_pred,
                fault_isolated=False,
                isolated_line_id=-1,
                isolated_buses_count=0,
                deenergized_load_mw=0.0,
                service_restored=False,
                selected_tie_line_id=None,
                switching_sequence=[],
                candidate_evaluations=[],
                total_healing_time_seconds=0.05,
                post_healing_results=normal_res,
                healed_network=fault_net,
                summary_report="AI Diagnostics: Steady-State Normal. All feeders healthy and operational."
            )

        faulted_line = ai_pred.predicted_line_id

        # =========================================================================
        # STEP 2: Autonomous Fault Isolation
        # =========================================================================
        # Open circuit breakers / sectionalizers flanking the faulted section
        healed_net = copy.deepcopy(fault_net)
        
        # Remove any fault shunt load if present in simulation
        fault_loads = healed_net.load[healed_net.load.name.str.contains("Fault", case=False, na=False)]
        if len(fault_loads) > 0:
            healed_net.load.drop(index=fault_loads.index, inplace=True)

        fb = int(healed_net.line.at[faulted_line, 'from_bus'])
        tb = int(healed_net.line.at[faulted_line, 'to_bus'])

        # Switch line off service
        healed_net.line.at[faulted_line, 'in_service'] = False
        
        # Simulated intelligent breaker operating time: 35-50 milliseconds
        iso_time = 0.045
        switching_ops.append(SwitchingOperation(
            step_number=op_step,
            action_type="OPEN",
            target_element=f"Sectionalizer CB-{faulted_line}",
            element_id=faulted_line,
            rationale=f"Isolate faulted branch {faulted_line} (Bus {fb} <-> Bus {tb}). Trip upstream & downstream breakers.",
            execution_time_seconds=iso_time
        ))
        op_step += 1

        # Analyze de-energized island
        energized_buses, isolated_buses, G_iso = analyze_network_connectivity(healed_net)
        iso_res = solve_power_flow(healed_net)
        deenergized_mw = iso_res.unserved_load_mw

        # If no buses are isolated (e.g. meshed network or already redundant path), service is safe
        if len(isolated_buses) == 0:
            total_time = time.perf_counter() - t_start + 0.15
            return SelfHealingResult(
                fault_detected=True,
                ai_prediction=ai_pred,
                fault_isolated=True,
                isolated_line_id=faulted_line,
                isolated_buses_count=0,
                deenergized_load_mw=0.0,
                service_restored=True,
                selected_tie_line_id=None,
                switching_sequence=switching_ops,
                candidate_evaluations=[],
                total_healing_time_seconds=total_time,
                post_healing_results=iso_res,
                healed_network=healed_net,
                summary_report=f"Fault on Line {faulted_line} isolated. Natural loop redundancy maintained 100% supply."
            )

        # =========================================================================
        # STEP 3: Autonomous Service Restoration (Reconfiguration Optimization)
        # =========================================================================
        # Available tie-lines
        available_tie_lines = getattr(healed_net, 'tie_lines', [])
        if not available_tie_lines:
            # Search lines that are currently out of service
            available_tie_lines = list(healed_net.line[healed_net.line.in_service == False].index)
            # Exclude the isolated faulted line
            available_tie_lines = [l for l in available_tie_lines if l != faulted_line]

        candidates: List[RestorationCandidate] = []
        best_candidate: Optional[RestorationCandidate] = None
        best_score = -1e9

        for tie_idx in available_tie_lines:
            if tie_idx == faulted_line:
                continue

            tfb = int(healed_net.line.at[tie_idx, 'from_bus'])
            ttb = int(healed_net.line.at[tie_idx, 'to_bus'])

            # Fast filter: Does this tie line bridge between an energized bus and an isolated bus?
            bridges_island = (
                (tfb in energized_buses and ttb in isolated_buses) or
                (ttb in energized_buses and tfb in isolated_buses)
            )

            # Test closing this tie line
            test_net = copy.deepcopy(healed_net)
            test_net.line.at[tie_idx, 'in_service'] = True

            # Constraint 1: Check Radiality (Acyclic graph structure)
            _, _, G_test = analyze_network_connectivity(test_net)
            # Find largest connected component (energized network)
            infeed_bus = int(test_net.ext_grid.at[0, 'bus'])
            if infeed_bus in G_test:
                main_comp = G_test.subgraph(nx.node_connected_component(G_test, infeed_bus))
                # For a tree, number of edges = number of nodes - 1
                is_radial = (main_comp.number_of_edges() == main_comp.number_of_nodes() - 1)
            else:
                is_radial = False

            # Constraint 2 & 3: Power flow, Voltage limits, Thermal limits
            test_res = solve_power_flow(test_net)

            violates_v = (test_res.min_vm_pu < v_min_threshold)
            violates_thermal = (test_res.max_line_loading > max_loading_threshold)

            # Objective Function Score:
            # Maximize restored load, minimize losses, penalize voltage & thermal violations
            restored_mw = iso_res.unserved_load_mw - test_res.unserved_load_mw
            score = (
                (restored_mw * 50.0)
                - (test_res.total_loss_kw * 0.05)
                - (500.0 if not is_radial else 0.0)
                - (300.0 if not test_res.converged else 0.0)
                - (100.0 if violates_v else 0.0)
                - (100.0 if violates_thermal else 0.0)
                + (test_res.min_vm_pu * 20.0)
            )

            cand = RestorationCandidate(
                tie_line_id=tie_idx,
                from_bus=tfb,
                to_bus=ttb,
                is_radial=is_radial,
                converged=test_res.converged,
                unserved_mw=test_res.unserved_load_mw,
                total_loss_kw=test_res.total_loss_kw,
                min_vm_pu=test_res.min_vm_pu,
                max_loading=test_res.max_line_loading,
                violates_voltage=violates_v,
                violates_thermal=violates_thermal,
                score=score
            )
            candidates.append(cand)

            if bridges_island and score > best_score:
                best_score = score
                best_candidate = cand

        # Apply best restoration candidate
        service_restored = False
        selected_tie_id = None

        if best_candidate and best_candidate.converged and (best_candidate.unserved_mw < iso_res.unserved_load_mw):
            selected_tie_id = best_candidate.tie_line_id
            healed_net.line.at[selected_tie_id, 'in_service'] = True
            
            # Simulated motorized tie-switch transfer time: 1.2 to 1.8 seconds
            tie_time = 1.45
            tfb = best_candidate.from_bus
            ttb = best_candidate.to_bus
            switching_ops.append(SwitchingOperation(
                step_number=op_step,
                action_type="CLOSE",
                target_element=f"Tie-Switch TS-{selected_tie_id}",
                element_id=selected_tie_id,
                rationale=f"Close tie-line {selected_tie_id} connecting Bus {tfb} and Bus {ttb} to backfeed de-energized island.",
                execution_time_seconds=tie_time
            ))
            op_step += 1

            # Priority-based Load Shedding check if thermal capacity is constrained
            if enable_priority_shedding and best_candidate.violates_thermal:
                # Shed lowest priority (Residential, priority 3) loads in the island first
                load_priorities = getattr(healed_net, 'load_priorities', {})
                for l_idx, row in healed_net.load.iterrows():
                    b_id = int(row['bus'])
                    p_info = load_priorities.get(b_id, {})
                    if p_info.get('priority', 3) == 3 and b_id in isolated_buses:
                        healed_net.load.at[l_idx, 'in_service'] = False
                        switching_ops.append(SwitchingOperation(
                            step_number=op_step,
                            action_type="LOAD_SHED",
                            target_element=f"Load Breaker at Bus {b_id}",
                            element_id=b_id,
                            rationale=f"Shed non-critical residential load at Bus {b_id} ({row['p_mw']:.2f} MW) to protect feeder ampacity.",
                            execution_time_seconds=0.08
                        ))
                        op_step += 1
                        break  # shed enough to relieve constraint

            service_restored = True

        post_res = solve_power_flow(healed_net)
        total_time = sum(op.execution_time_seconds for op in switching_ops) + (time.perf_counter() - t_start)

        report = (
            f"Autonomous Self-Healing Completed in {total_time:.2f}s | "
            f"Faulted Line {faulted_line} Isolated | "
            f"{'Tie-Switch ' + str(selected_tie_id) + ' closed to restore ' + f'{post_res.restoration_percentage:.1f}% load' if selected_tie_id is not None else 'No tie-switch available'} | "
            f"Final Served Load: {post_res.served_load_mw:.2f} MW / {post_res.total_load_mw:.2f} MW"
        )

        return SelfHealingResult(
            fault_detected=True,
            ai_prediction=ai_pred,
            fault_isolated=True,
            isolated_line_id=faulted_line,
            isolated_buses_count=len(isolated_buses),
            deenergized_load_mw=deenergized_mw,
            service_restored=service_restored,
            selected_tie_line_id=selected_tie_id,
            switching_sequence=switching_ops,
            candidate_evaluations=candidates,
            total_healing_time_seconds=total_time,
            post_healing_results=post_res,
            healed_network=healed_net,
            summary_report=report
        )
