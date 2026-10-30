"""
Verification script for end-to-end AegisGrid AI platform
"""
from core.network_builder import get_network
from core.power_flow import solve_power_flow
from core.fault_simulator import FaultType, simulate_fault
from core.ai_engine import AIFaultEngine
from core.flisr_controller import FLISRController
from core.reliability_metrics import compute_reliability_indices
from visualization.network_graph import create_network_graph
from visualization.dashboard_charts import plot_voltage_profile_comparison, plot_line_loading_comparison

print("[1/6] Testing IEEE 33-Bus Network Building & Power Flow...")
net = get_network("ieee33", include_dg=True)
base_res = solve_power_flow(net)
assert base_res.converged, "Baseline power flow failed!"
print(f"  Converged! Total load: {base_res.total_load_mw:.2f} MW, Min Vm: {base_res.min_vm_pu:.4f} p.u.")

print("[2/6] Testing Physical Fault Simulation...")
f_net, f_res, tel, evt = simulate_fault(net, FaultType.THREE_PHASE_BOLTED, line_id=13, location_percent=50.0)
print(f"  Fault injected on Line {evt.line_id}! Min Vm: {f_res.min_vm_pu:.4f} p.u.")

print("[3/6] Testing AI Diagnostics Inference...")
ai_engine = AIFaultEngine(model_dir="models")
loaded = ai_engine.load_models("ieee33")
assert loaded, "Failed to load pre-trained models!"
pred = ai_engine.predict(tel.feature_vector)
print(f"  AI Prediction: Fault={pred.fault_detected}, Type={pred.predicted_fault_type}, Predicted Line={pred.predicted_line_id} (Confidence: {pred.line_confidence*100:.1f}%) in {pred.inference_time_ms:.2f}ms")

print("[4/6] Testing Autonomous FLISR Self-Healing...")
flisr = FLISRController(ai_engine)
healing_res = flisr.execute_self_healing(f_net, tel)
print(f"  Self-Healing result: Restored={healing_res.service_restored}, Selected Tie Line={healing_res.selected_tie_line_id}")
print(f"  Post-healing served load: {healing_res.post_healing_results.served_load_mw:.2f} MW ({healing_res.post_healing_results.restoration_percentage:.1f}%)")
print(f"  Total operations: {len(healing_res.switching_sequence)}, Total time: {healing_res.total_healing_time_seconds:.2f}s")

print("[5/6] Testing IEEE 1366 Reliability Indices...")
rel = compute_reliability_indices(
    total_load_mw=base_res.total_load_mw,
    deenergized_load_mw=healing_res.deenergized_load_mw,
    restored_load_mw=healing_res.post_healing_results.served_load_mw,
    unserved_load_mw=healing_res.post_healing_results.unserved_load_mw
)
print(f"  SAIDI Manual: {rel.saidi_manual_min:.1f} min vs AI: {rel.saidi_ai_min:.2f} min ({rel.saidi_reduction_percent:.1f}% reduction)")
print(f"  Avoided Outage Economic Savings: ${rel.economic_savings_usd:,.2f}")

print("[6/6] Testing Visualization Figures Generation...")
fig_net = create_network_graph(net, base_res)
fig_volt = plot_voltage_profile_comparison(base_res, f_res, healing_res.post_healing_results)
fig_load = plot_line_loading_comparison(base_res, healing_res.post_healing_results)
assert fig_net is not None and fig_volt is not None and fig_load is not None
print("  All visual figures generated successfully!")

print("\nSUCCESS: All components passed verification tests!")
