"""
AegisGrid AI: Autonomous Self-Healing Electrical Distribution Network Simulation Platform
========================================================================================
Interactive FLISR (Fault Location, Isolation, and Service Restoration) Dashboard
Built with Python, pandapower, Scikit-Learn, NetworkX, and Streamlit.
"""

import copy
import io
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Core imports
from core.network_builder import get_network, build_ieee33_network, build_urban16_network
from core.power_flow import solve_power_flow, PowerFlowResults
from core.fault_simulator import FaultType, FaultEvent, TelemetryData, simulate_fault
from core.ai_engine import AIFaultEngine, AIPrediction, ModelMetrics
from core.flisr_controller import FLISRController, SelfHealingResult, SwitchingOperation
from core.reliability_metrics import compute_reliability_indices, ReliabilityComparison

# Visualization imports
from visualization.network_graph import create_network_graph
from visualization.dashboard_charts import (
    plot_voltage_profile_comparison,
    plot_line_loading_comparison,
    plot_power_losses_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_reliability_comparison
)

# =============================================================================
# PAGE CONFIGURATION & THEME STYLING
# =============================================================================
st.set_page_config(
    page_title="AegisGrid AI | Self-Healing Distribution Network",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Engineering CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #0d1b2a 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: 0.5px;
    }
    .main-subtitle {
        color: #8B949E;
        font-size: 13.5px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-normal {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
    }
    .badge-fault {
        background: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid #FF1744;
    }
    .badge-isolated {
        background: rgba(255, 145, 0, 0.15);
        color: #FF9100;
        border: 1px solid #FF9100;
    }
    .badge-healed {
        background: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid #00E5FF;
    }
    
    /* Glassmorphic Metrics Card */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 11.5px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 11.5px;
        margin-top: 3px;
    }
    
    /* Step Sequence Timeline */
    .timeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
    }
    .timeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .timeline-step.active {
        color: #00E5FF;
    }
    .timeline-step.completed {
        color: #00E676;
    }
    .timeline-step.pending {
        color: #6E7681;
    }
    .timeline-arrow {
        color: #484F58;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE CACHED AI ENGINE & SESSION STATE
# Initialize AI Engine in session state
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = AIFaultEngine(model_dir="models")
ai_engine = st.session_state.ai_engine
flisr_ctrl = FLISRController(ai_engine)

# Initialize Session State
if "network_name" not in st.session_state:
    st.session_state.network_name = "IEEE 33-Bus Distribution Network"
if "flisr_stage" not in st.session_state:
    st.session_state.flisr_stage = "NORMAL"  # "NORMAL", "FAULT", "ISOLATED", "HEALED"
if "active_net" not in st.session_state:
    st.session_state.active_net = get_network("ieee33", include_dg=True)
if "base_results" not in st.session_state:
    st.session_state.base_results = solve_power_flow(st.session_state.active_net)
if "fault_net" not in st.session_state:
    st.session_state.fault_net = None
if "fault_results" not in st.session_state:
    st.session_state.fault_results = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
if "fault_event" not in st.session_state:
    st.session_state.fault_event = None
if "flisr_result" not in st.session_state:
    st.session_state.flisr_result = None
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# Load models if available
if not ai_engine.is_trained(st.session_state.active_net.meta_network_name):
    with st.spinner("Initializing and training baseline AI diagnostic models..."):
        ai_engine.train_models(st.session_state.active_net, num_scenarios_per_line=4)
else:
    ai_engine.load_models(st.session_state.active_net.meta_network_name)


# =============================================================================
# HEADER SECTION
# =============================================================================
status_map = {
    "NORMAL": ("badge-normal", "🟢 GRID HEALTHY (STEADY-STATE)"),
    "FAULT": ("badge-fault", "🔴 CRITICAL FAULT DETECTED"),
    "ISOLATED": ("badge-isolated", "🟠 FAULT SECTION ISOLATED"),
    "HEALED": ("badge-healed", "⚡ AUTONOMOUS SERVICE RESTORED")
}
badge_class, badge_text = status_map.get(st.session_state.flisr_stage, ("badge-normal", "NORMAL"))

st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="main-title">
                <span>⚡ AegisGrid AI</span>
                <span style="font-size: 15px; font-weight: 400; color: #8B949E; margin-left: 10px;">| Autonomous Self-Healing Distribution Network</span>
            </div>
            <div class="main-subtitle">
                Fully Software-Based Simulation &bull; pandapower AC Power Flow &bull; Scikit-Learn AI Diagnostics &bull; Automated FLISR Reconfiguration
            </div>
        </div>
        <div>
            <span class="status-badge {badge_class}">{badge_text}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR CONTROLS & GRID CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Network & Simulation Setup")
    
    net_choice = st.selectbox(
        "Select Distribution Network Model",
        ["IEEE 33-Bus Benchmark (Baran & Wu)", "16-Bus Urban Smart Microgrid"],
        index=0 if "33" in st.session_state.network_name else 1
    )
    
    chosen_code = "ieee33" if "33" in net_choice else "urban16"
    
    # Check if network changed
    if chosen_code != ("ieee33" if "33" in st.session_state.network_name else "urban16"):
        st.session_state.network_name = net_choice
        st.session_state.active_net = get_network(chosen_code, include_dg=True)
        st.session_state.base_results = solve_power_flow(st.session_state.active_net)
        st.session_state.flisr_stage = "NORMAL"
        st.session_state.fault_net = None
        st.session_state.fault_results = None
        st.session_state.flisr_result = None
        if not ai_engine.is_trained(st.session_state.active_net.meta_network_name):
            with st.spinner("Training models for chosen network..."):
                ai_engine.train_models(st.session_state.active_net, num_scenarios_per_line=4)
        else:
            ai_engine.load_models(st.session_state.active_net.meta_network_name)
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Grid Parameters")
    
    load_scaling = st.slider("Global Load Scaling Factor", min_value=0.50, max_value=1.50, value=1.0, step=0.05,
                             help="Scale active and reactive power consumption across all grid nodes.")
    include_dg = st.checkbox("Enable Distributed Generation (Solar PV & BESS)", value=True,
                             help="Injects renewable solar and battery storage systems for enhanced grid resilience.")
    
    st.markdown("---")
    st.markdown("### ⚠️ Fault Injection Parameters")
    
    # Available active lines for fault injection
    active_line_indices = [int(l) for l in st.session_state.active_net.line[st.session_state.active_net.line.in_service == True].index]
    default_fault_line = 13 if 13 in active_line_indices else (active_line_indices[len(active_line_indices)//2] if active_line_indices else 0)
    
    fault_line_id = st.selectbox(
        "Fault Target Branch (Line ID)",
        active_line_indices,
        index=active_line_indices.index(default_fault_line) if default_fault_line in active_line_indices else 0,
        format_func=lambda l: f"Line {l}: Bus {st.session_state.active_net.line.at[l, 'from_bus']} -> Bus {st.session_state.active_net.line.at[l, 'to_bus']}"
    )

    fault_type_str = st.selectbox(
        "Fault Physical Type",
        [
            "Three-Phase Bolted Short-Circuit",
            "Single Line-to-Ground (SLG) Fault",
            "Line-to-Line (L-L) Fault",
            "High Impedance Fault (HIF)"
        ],
        index=0
    )

    fault_type_map = {
        "Three-Phase Bolted Short-Circuit": FaultType.THREE_PHASE_BOLTED,
        "Single Line-to-Ground (SLG) Fault": FaultType.SINGLE_LINE_TO_GROUND,
        "Line-to-Line (L-L) Fault": FaultType.LINE_TO_LINE,
        "High Impedance Fault (HIF)": FaultType.HIGH_IMPEDANCE
    }
    selected_fault_type = fault_type_map[fault_type_str]

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        fault_location_pct = st.slider("Location (% of Line)", 10.0, 90.0, 50.0, 5.0)
    with col_sb2:
        fault_impedance = st.number_input("Impedance Rf (Ω)", 0.05, 50.0, 1.0, 0.5)

    telemetry_noise = st.slider("Sensor Telemetry Noise (%)", 0.0, 2.5, 0.5, 0.1,
                                help="Simulates measurement inaccuracies in PMU and smart meter telemetry.")

    st.markdown("---")
    st.markdown("### 🛡️ FLISR Automation Constraints")
    v_min_threshold = st.slider("Under-Voltage Sag Threshold (p.u.)", 0.85, 0.95, 0.90, 0.01)
    max_loading_limit = st.slider("Thermal Ampacity Limit (%)", 90.0, 130.0, 110.0, 5.0)
    enable_priority_shed = st.checkbox("Enable Priority Load Shedding", value=True,
                                       help="If alternative feeder is overloaded, shed residential loads first to protect critical hospitals.")


# =============================================================================
# HELPER ACTIONS
# =============================================================================
def trigger_fault_injection():
    # Build baseline with current settings
    base_net = get_network(chosen_code, include_dg=include_dg, load_scaling=load_scaling)
    st.session_state.active_net = base_net
    st.session_state.base_results = solve_power_flow(base_net)

    # Simulate physical fault
    f_net, f_res, tel, evt = simulate_fault(
        net=base_net,
        fault_type=selected_fault_type,
        line_id=fault_line_id,
        location_percent=fault_location_pct,
        fault_impedance_ohm=fault_impedance,
        sensor_noise_percent=telemetry_noise
    )
    st.session_state.fault_net = f_net
    st.session_state.fault_results = f_res
    st.session_state.telemetry = tel
    st.session_state.fault_event = evt
    st.session_state.flisr_stage = "FAULT"
    st.session_state.flisr_result = None

    # Log event
    st.session_state.history_log.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "event": "FAULT_INJECTED",
        "details": evt.description,
        "line": fault_line_id,
        "type": fault_type_str
    })


def trigger_self_healing():
    if st.session_state.fault_net is None or st.session_state.telemetry is None:
        trigger_fault_injection()

    res = flisr_ctrl.execute_self_healing(
        fault_net=st.session_state.fault_net,
        telemetry=st.session_state.telemetry,
        v_min_threshold=v_min_threshold,
        max_loading_threshold=max_loading_limit,
        enable_priority_shedding=enable_priority_shed
    )
    st.session_state.flisr_result = res
    st.session_state.flisr_stage = "HEALED"

    # Log event
    st.session_state.history_log.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "event": "SELF_HEALING_COMPLETED",
        "details": res.summary_report,
        "line": res.isolated_line_id,
        "tie_switch": res.selected_tie_line_id
    })


def reset_grid():
    base_net = get_network(chosen_code, include_dg=include_dg, load_scaling=load_scaling)
    st.session_state.active_net = base_net
    st.session_state.base_results = solve_power_flow(base_net)
    st.session_state.flisr_stage = "NORMAL"
    st.session_state.fault_net = None
    st.session_state.fault_results = None
    st.session_state.telemetry = None
    st.session_state.fault_event = None
    st.session_state.flisr_result = None


# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live FLISR Console",
    "📈 Electrical Parameter Analytics",
    "🧠 AI Diagnostics & Model Inspector",
    "🛡️ IEEE 1366 Reliability & Economics"
])


# =============================================================================
# TAB 1: LIVE FLISR SELF-HEALING CONSOLE
# =============================================================================
with tab1:
    # 1. TIMELINE PROGRESS INDICATOR
    stage_idx = {"NORMAL": 1, "FAULT": 2, "ISOLATED": 3, "HEALED": 4}.get(st.session_state.flisr_stage, 1)
    
    st.markdown(f"""
    <div class="timeline-container">
        <div class="timeline-step {'completed' if stage_idx > 1 else ('active' if stage_idx == 1 else 'pending')}">
            <span>{'✔️' if stage_idx > 1 else '1.'}</span> Normal Baseline
        </div>
        <span class="timeline-arrow">➔</span>
        <div class="timeline-step {'completed' if stage_idx > 2 else ('active' if stage_idx == 2 else 'pending')}">
            <span>{'✔️' if stage_idx > 2 else '2.'}</span> Fault Injected
        </div>
        <span class="timeline-arrow">➔</span>
        <div class="timeline-step {'completed' if stage_idx > 3 else ('active' if stage_idx == 3 else 'pending')}">
            <span>{'✔️' if stage_idx > 3 else '3.'}</span> AI Detection & Isolation
        </div>
        <span class="timeline-arrow">➔</span>
        <div class="timeline-step {'completed' if stage_idx == 4 else 'pending'}">
            <span>{'✔️' if stage_idx == 4 else '4.'}</span> Reconfigured & Healed
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. CONTROL BUTTONS
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1.5, 1.8, 1.2, 1.2])
    with c_btn1:
        if st.button("🚨 Inject Fault & Outage", use_container_width=True, type="secondary"):
            trigger_fault_injection()
            st.rerun()

    with c_btn2:
        if st.button("🤖 Run Autonomous FLISR", use_container_width=True, type="primary"):
            trigger_self_healing()
            st.rerun()

    with c_btn3:
        # Step-by-Step Walkthrough
        if st.button("⏯️ Step Walkthrough", use_container_width=True):
            if st.session_state.flisr_stage == "NORMAL":
                trigger_fault_injection()
            elif st.session_state.flisr_stage == "FAULT":
                trigger_self_healing()
            else:
                reset_grid()
            st.rerun()

    with c_btn4:
        if st.button("🔄 Reset Grid", use_container_width=True):
            reset_grid()
            st.rerun()

    # Determine current display state
    if st.session_state.flisr_stage == "NORMAL":
        disp_net = st.session_state.active_net
        disp_res = st.session_state.base_results
        f_line, iso_line, rest_tie = None, None, None
    elif st.session_state.flisr_stage == "FAULT":
        disp_net = st.session_state.fault_net
        disp_res = st.session_state.fault_results
        f_line = fault_line_id
        iso_line, rest_tie = None, None
    else:  # HEALED
        disp_net = st.session_state.flisr_result.healed_network
        disp_res = st.session_state.flisr_result.post_healing_results
        f_line = None
        iso_line = st.session_state.flisr_result.isolated_line_id
        rest_tie = st.session_state.flisr_result.selected_tie_line_id

    # 3. METRIC CARDS ROW
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        srv_pct = disp_res.restoration_percentage
        srv_mw = disp_res.served_load_mw
        tot_mw = disp_res.total_load_mw
        sub_col = "#00E676" if srv_pct >= 99.0 else ("#FF1744" if srv_pct < 60.0 else "#FF9100")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Served Power Load</div>
            <div class="metric-val">{srv_pct:.1f}%</div>
            <div class="metric-sub" style="color: {sub_col};">{srv_mw:.2f} MW / {tot_mw:.2f} MW</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        uns_mw = disp_res.unserved_load_mw
        uns_col = "#00E676" if uns_mw == 0.0 else "#FF1744"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unserved (Outage) Load</div>
            <div class="metric-val" style="color: {uns_col};">{uns_mw:.2f} MW</div>
            <div class="metric-sub" style="color: #8B949E;">{len(disp_res.isolated_buses)} Islanded Buses</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        min_v = disp_res.min_vm_pu
        min_b = disp_res.min_vm_bus
        v_col = "#00E676" if min_v >= 0.95 else ("#FF9100" if min_v >= 0.90 else "#FF1744")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Minimum Bus Voltage</div>
            <div class="metric-val" style="color: {v_col};">{min_v:.3f} p.u.</div>
            <div class="metric-sub" style="color: #8B949E;">Observed at Bus {min_b}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        max_ld = disp_res.max_line_loading
        max_l = disp_res.max_loading_line
        ld_col = "#00E676" if max_ld <= 90.0 else ("#FF9100" if max_ld <= 100.0 else "#FF1744")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Branch Loading</div>
            <div class="metric-val" style="color: {ld_col};">{max_ld:.1f}%</div>
            <div class="metric-sub" style="color: #8B949E;">Line {max_l} Critical Path</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        loss_kw = disp_res.total_loss_kw
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Power Losses</div>
            <div class="metric-val">{loss_kw:.1f} kW</div>
            <div class="metric-sub" style="color: #8B949E;">{disp_res.total_loss_pct:.2f}% of Total Load</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. INTERACTIVE NETWORK TOPOLOGY VISUALIZER
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    graph_title = f"{st.session_state.network_name} — {'🟢 Steady-State Normal' if st.session_state.flisr_stage == 'NORMAL' else ('🔴 Fault Injected' if st.session_state.flisr_stage == 'FAULT' else '⚡ Autonomous Reconfiguration')}"
    
    fig_net = create_network_graph(
        net=disp_net,
        results=disp_res,
        fault_line_id=f_line,
        isolated_line_id=iso_line,
        restored_tie_id=rest_tie,
        fault_location_pct=fault_location_pct,
        title=graph_title
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # 5. SIDE-BY-SIDE PERFORMANCE MATRIX & SWITCHING OPERATIONS
    st.markdown("### 📊 Performance Comparison: Before vs. During vs. After Recovery")

    col_tbl, col_ops = st.columns([1.1, 0.9])
    
    with col_tbl:
        base = st.session_state.base_results
        flt = st.session_state.fault_results
        heal = st.session_state.flisr_result.post_healing_results if st.session_state.flisr_result else None

        comp_data = {
            "Parameter Metric": [
                "Served Power Load (MW)",
                "Restoration Percentage (%)",
                "Unserved Outage Load (MW)",
                "Minimum Bus Voltage (p.u.)",
                "Peak Branch Loading (%)",
                "Active System Losses (kW)",
                "Isolated / Blackout Buses"
            ],
            "Normal Baseline": [
                f"{base.served_load_mw:.2f} MW",
                "100.0%",
                "0.00 MW",
                f"{base.min_vm_pu:.4f} p.u.",
                f"{base.max_line_loading:.1f}%",
                f"{base.total_loss_kw:.1f} kW",
                "0"
            ],
            "During Fault": [
                f"{flt.served_load_mw:.2f} MW" if flt else "—",
                f"{flt.restoration_percentage:.1f}%" if flt else "—",
                f"{flt.unserved_load_mw:.2f} MW" if flt else "—",
                f"{flt.min_vm_pu:.4f} p.u." if flt else "—",
                f"{flt.max_line_loading:.1f}%" if flt else "—",
                f"{flt.total_loss_kw:.1f} kW" if flt else "—",
                f"{len(flt.isolated_buses)}" if flt else "—"
            ],
            "Post-Healing (Restored)": [
                f"{heal.served_load_mw:.2f} MW" if heal else "—",
                f"{heal.restoration_percentage:.1f}%" if heal else "—",
                f"{heal.unserved_load_mw:.2f} MW" if heal else "—",
                f"{heal.min_vm_pu:.4f} p.u." if heal else "—",
                f"{heal.max_line_loading:.1f}%" if heal else "—",
                f"{heal.total_loss_kw:.1f} kW" if heal else "—",
                f"{len(heal.isolated_buses)}" if heal else "—"
            ]
        }
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with col_ops:
        st.markdown("**Autonomous Switching Operation Log**")
        if st.session_state.flisr_result and st.session_state.flisr_result.switching_sequence:
            ops_records = []
            for op in st.session_state.flisr_result.switching_sequence:
                ops_records.append({
                    "Step": f"#{op.step_number}",
                    "Action": op.action_type,
                    "Target Element": op.target_element,
                    "Duration": f"{op.execution_time_seconds*1000:.0f} ms" if op.execution_time_seconds < 1.0 else f"{op.execution_time_seconds:.2f} s",
                    "Rationale": op.rationale
                })
            st.dataframe(pd.DataFrame(ops_records), use_container_width=True, hide_index=True)
            st.success(f"⏱️ Total Autonomous Healing Time: **{st.session_state.flisr_result.total_healing_time_seconds:.2f} seconds**")
        else:
            st.info("No active switching operations. Trigger a fault and run Autonomous FLISR to observe the breaker and tie-switch sequence.")


# =============================================================================
# TAB 2: ELECTRICAL PARAMETER ANALYTICS
# =============================================================================
with tab2:
    st.markdown("### 📈 In-Depth Electrical Parameter Profiles")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        # Voltage Profile Curves
        fig_volt = plot_voltage_profile_comparison(
            base_res=st.session_state.base_results,
            fault_res=st.session_state.fault_results,
            post_res=st.session_state.flisr_result.post_healing_results if st.session_state.flisr_result else None,
            v_min_limit=v_min_threshold
        )
        st.plotly_chart(fig_volt, use_container_width=True)

    with col_c2:
        # Branch Loading Comparison
        fig_load = plot_line_loading_comparison(
            base_res=st.session_state.base_results,
            post_res=st.session_state.flisr_result.post_healing_results if st.session_state.flisr_result else None
        )
        st.plotly_chart(fig_load, use_container_width=True)

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        # Active Power Losses
        fig_loss = plot_power_losses_comparison(
            base_res=st.session_state.base_results,
            fault_res=st.session_state.fault_results,
            post_res=st.session_state.flisr_result.post_healing_results if st.session_state.flisr_result else None
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_c4:
        # Tie-Switch Candidate Optimization Table
        st.markdown("**Tie-Switch Candidate Optimization Matrix**")
        if st.session_state.flisr_result and st.session_state.flisr_result.candidate_evaluations:
            cands = []
            for c in st.session_state.flisr_result.candidate_evaluations:
                cands.append({
                    "Tie Line ID": f"Tie-{c.tie_line_id}",
                    "Connects": f"Bus {c.from_bus} ➔ {c.to_bus}",
                    "Radially Valid": "✅ Yes" if c.is_radial else "❌ Loop",
                    "Min V (p.u.)": f"{c.min_vm_pu:.3f}",
                    "Peak Load": f"{c.max_loading:.1f}%",
                    "Loss (kW)": f"{c.total_loss_kw:.1f}",
                    "Optimization Score": f"{c.score:.1f}"
                })
            st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)
        else:
            st.info("Execute Autonomous FLISR to evaluate all candidate tie-switches under radiality and thermal constraints.")


# =============================================================================
# TAB 3: AI DIAGNOSTICS & MODEL INSPECTOR
# =============================================================================
with tab3:
    st.markdown("### 🧠 Scikit-Learn AI Diagnostics Hub")
    
    # 1. Live Inference Panel
    st.markdown("#### ⚡ Real-Time Diagnostic Inference")
    col_ai1, col_ai2, col_ai3 = st.columns(3)

    if st.session_state.telemetry:
        ai_pred = ai_engine.predict(
            st.session_state.telemetry.feature_vector,
            network_name=st.session_state.active_net.meta_network_name
        )
        
        with col_ai1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Fault Type</div>
                <div class="metric-val" style="color: #FF1744; font-size: 18px;">{ai_pred.predicted_fault_type}</div>
                <div class="metric-sub">Confidence: <b>{ai_pred.fault_type_confidence*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_ai2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pinpointed Faulted Branch</div>
                <div class="metric-val" style="color: #00E5FF; font-size: 20px;">Line {ai_pred.predicted_line_id}</div>
                <div class="metric-sub">Section Confidence: <b>{ai_pred.line_confidence*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_ai3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Fault Distance</div>
                <div class="metric-val" style="color: #FFD600; font-size: 20px;">{ai_pred.predicted_distance_percent:.1f}%</div>
                <div class="metric-sub">AI Inference Time: <b>{ai_pred.inference_time_ms:.2f} ms</b></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Inject a fault or run simulation to view real-time AI diagnostic inference and confidence probabilities.")

    st.markdown("---")
    
    # 2. Confusion Matrix & Feature Importance
    if ai_engine.metrics:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            fig_cm = plot_confusion_matrix(ai_engine.metrics)
            st.plotly_chart(fig_cm, use_container_width=True)
        with m_col2:
            fig_fi = plot_feature_importance(ai_engine.metrics)
            st.plotly_chart(fig_fi, use_container_width=True)
    
    # 3. Model Retraining Section
    st.markdown("#### 🔄 On-Demand Synthetic Model Retraining")
    st.caption("Generate fresh Monte-Carlo power flow scenarios with varied load profiles, renewable fluctuations, and fault impedances to retrain the Scikit-learn AI models on the fly.")
    
    col_ret1, col_ret2 = st.columns([1, 3])
    with col_ret1:
        num_scenarios = st.slider("Scenarios per Branch", min_value=3, max_value=12, value=6, step=1)
        if st.button("🚀 Retrain AI Models Now", use_container_width=True):
            with st.spinner("Generating Monte-Carlo scenarios and training Random Forest & ExtraTrees classifiers..."):
                t_train_start = time.time()
                metrics = ai_engine.train_models(st.session_state.active_net, num_scenarios_per_line=num_scenarios)
                st.success(f"Models retrained in {time.time()-t_train_start:.1f}s! Classifier Accuracy: {metrics.classifier_accuracy*100:.1f}%, Section Locator: {metrics.locator_accuracy*100:.1f}%")
                st.rerun()

    with col_ret2:
        if ai_engine.metrics:
            st.markdown(f"""
            - **Classifier Algorithm:** Random Forest Ensemble (100 estimators, max depth 16)
            - **Section Locator Algorithm:** ExtraTrees Classifier (100 estimators, max depth 18)
            - **Fault Distance Regressor:** Gradient Boosting Regressor (80 estimators, learning rate 0.08)
            - **Total Training Scenarios:** `{ai_engine.metrics.num_training_samples}` physical power flow states
            - **Overall F1-Score:** `{ai_engine.metrics.classifier_f1*100:.2f}%`
            """)


# =============================================================================
# TAB 4: IEEE 1366 RELIABILITY & GRID ECONOMICS
# =============================================================================
with tab4:
    st.markdown("### 🛡️ IEEE 1366 Distribution Reliability Standards & Outage Economics")
    st.caption("Quantifying the resilience improvements of AI Autonomous FLISR vs. Traditional Utility Crew Field Dispatch.")

    # Calculate metrics
    tot_p = st.session_state.base_results.total_load_mw
    deenergized_p = st.session_state.flisr_result.deenergized_load_mw if st.session_state.flisr_result else (st.session_state.fault_results.unserved_load_mw if st.session_state.fault_results else 1.2)
    unserved_p = st.session_state.flisr_result.post_healing_results.unserved_load_mw if st.session_state.flisr_result else 0.0
    restored_p = max(0.0, deenergized_p - unserved_p)
    healing_time = st.session_state.flisr_result.total_healing_time_seconds if st.session_state.flisr_result else 2.5

    col_voll, col_time = st.columns(2)
    with col_voll:
        voll_val = st.slider("Value of Lost Load (VoLL in $/MWh)", 2000.0, 25000.0, 9500.0, 500.0,
                             help="Standard regulatory economic cost per unsupplied MWh of electricity.")
    with col_time:
        manual_time_h = st.slider("Manual Crew Dispatch Duration (Hours)", 0.5, 4.0, 2.0, 0.25,
                                  help="Average time required for line patrol, manual isolation, and field switching.")

    rel_comp = compute_reliability_indices(
        total_load_mw=tot_p,
        deenergized_load_mw=deenergized_p,
        restored_load_mw=restored_p,
        unserved_load_mw=unserved_p,
        manual_repair_time_hours=manual_time_h,
        ai_restoration_time_seconds=healing_time,
        value_of_lost_load_per_mwh=voll_val
    )

    # Cards Row
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SAIDI (Outage Duration)</div>
            <div class="metric-val" style="color: #00E676;">{rel_comp.saidi_ai_min:.2f} min</div>
            <div class="metric-sub">Manual Crew: {rel_comp.saidi_manual_min:.1f} min ({rel_comp.saidi_reduction_percent:.1f}% reduction)</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customers Saved from Outage</div>
            <div class="metric-val" style="color: #00E5FF;">{rel_comp.customers_saved_by_ai:,}</div>
            <div class="metric-sub">Out of {rel_comp.interrupted_customers_manual:,} initially downstream</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Energy Not Supplied (ENS)</div>
            <div class="metric-val" style="color: #FFD600;">{rel_comp.ens_ai_mwh:.2f} MWh</div>
            <div class="metric-sub">Saved: {rel_comp.ens_saved_mwh:.2f} MWh</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Outage Economic Savings</div>
            <div class="metric-val" style="color: #00E676;">${rel_comp.economic_savings_usd:,.0f}</div>
            <div class="metric-sub">Direct Customer & Utility Benefit</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    col_rel_chart, col_rel_tbl = st.columns([1.1, 0.9])
    with col_rel_chart:
        fig_rel = plot_reliability_comparison(rel_comp)
        st.plotly_chart(fig_rel, use_container_width=True)

    with col_rel_tbl:
        st.markdown("**IEEE 1366 Reliability Indices Comparison Table**")
        rel_table_data = {
            "IEEE Reliability Index": [
                "SAIDI (System Avg Interruption Duration)",
                "SAIFI (System Avg Interruption Frequency)",
                "CAIDI (Customer Avg Interruption Duration)",
                "ENS (Energy Not Supplied in MWh)",
                "ASAI (Average Service Availability Index)"
            ],
            "Manual Restoration": [
                f"{rel_comp.saidi_manual_min:.2f} min/cust",
                f"{rel_comp.saifi_manual:.4f} outages/cust",
                f"{rel_comp.caidi_manual_min:.1f} min",
                f"{rel_comp.ens_manual_mwh:.2f} MWh",
                f"{rel_comp.asai_manual_pct:.4f}%"
            ],
            "AegisGrid AI Self-Healing": [
                f"{rel_comp.saidi_ai_min:.2f} min/cust",
                f"{rel_comp.saifi_ai:.4f} outages/cust",
                f"{rel_comp.caidi_ai_min:.1f} min",
                f"{rel_comp.ens_ai_mwh:.2f} MWh",
                f"{rel_comp.asai_ai_pct:.4f}%"
            ],
            "Resilience Improvement": [
                f"🔻 {rel_comp.saidi_reduction_percent:.1f}% Shorter",
                "🔻 Reduced Sustained Events",
                "🔻 Instant Transfer",
                f"🔻 {rel_comp.ens_saved_mwh:.2f} MWh Saved",
                "🔺 Near 'Five Nines' (99.999%)"
            ]
        }
        st.dataframe(pd.DataFrame(rel_table_data), use_container_width=True, hide_index=True)

    # 4. DOWNLOADABLE AUDIT REPORT
    st.markdown("---")
    st.markdown("#### 📥 System Event Audit Log & Telemetry Export")
    
    if st.session_state.history_log:
        df_hist = pd.DataFrame(st.session_state.history_log)
        csv_buffer = io.StringIO()
        df_hist.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Event Audit Trail (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"aegisgrid_flisr_audit_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Event log will populate as you inject faults and execute self-healing reconfigurations.")

if __name__ == "__main__":
    import sys
    # If launched directly via `python app.py`, invoke streamlit runner cleanly
    try:
        from streamlit.web import cli as stcli
        import streamlit as st
        # Only invoke if not already running inside a live Streamlit runtime
        if not (hasattr(st, "runtime") and st.runtime.exists()):
            sys.argv = ["streamlit", "run", __file__]
            sys.exit(stcli.main())
    except Exception:
        pass


