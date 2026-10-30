"""
Dashboard Charts Module
Generates high-contrast, executive-level Plotly charts for:
  - Voltage Profile Curve Comparison (Pre-Fault, Fault Sag, Post-Healing)
  - Branch Loading & Thermal Limit Monitoring
  - Active Power Loss Comparison
  - AI Model Confusion Matrix Heatmap
  - Telemetry Feature Importance for Fault Location
  - IEEE 1366 Reliability Metrics & Savings
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from core.ai_engine import ModelMetrics
from core.power_flow import PowerFlowResults
from core.reliability_metrics import ReliabilityComparison


CHART_THEME = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#161B22",
    font=dict(color="#E0E0E0", family="Segoe UI"),
    margin=dict(l=40, r=20, t=45, b=40)
)


def plot_voltage_profile_comparison(
    base_res: PowerFlowResults,
    fault_res: Optional[PowerFlowResults] = None,
    post_res: Optional[PowerFlowResults] = None,
    v_min_limit: float = 0.95,
    v_max_limit: float = 1.05
) -> go.Figure:
    """
    Renders bus voltage profile curves across all buses before, during, and after healing.
    """
    fig = go.Figure()
    buses = sorted(base_res.bus_voltages.keys())

    # Baseline curve
    base_v = [base_res.bus_voltages[b] for b in buses]
    fig.add_trace(go.Scatter(
        x=buses, y=base_v,
        mode='lines+markers',
        name='Normal Baseline',
        line=dict(color='#00E676', width=2.5),
        marker=dict(size=5, color='#00E676')
    ))

    # Fault curve
    if fault_res:
        fault_v = [fault_res.bus_voltages.get(b, 0.0) for b in buses]
        fig.add_trace(go.Scatter(
            x=buses, y=fault_v,
            mode='lines+markers',
            name='During Fault (Sag/Outage)',
            line=dict(color='#FF1744', width=2.5, dash='dash'),
            marker=dict(size=5, color='#FF1744')
        ))

    # Restored curve
    if post_res:
        post_v = [post_res.bus_voltages.get(b, 0.0) for b in buses]
        fig.add_trace(go.Scatter(
            x=buses, y=post_v,
            mode='lines+markers',
            name='Post-Healing (Restored)',
            line=dict(color='#00E5FF', width=3),
            marker=dict(size=6, color='#00E5FF')
        ))

    # Threshold lines
    fig.add_hline(y=v_min_limit, line_dash="dot", line_color="#FF9100",
                  annotation_text=f"Under-Voltage Limit ({v_min_limit:.2f} p.u.)",
                  annotation_position="bottom right", annotation_font_color="#FF9100")
    fig.add_hline(y=v_max_limit, line_dash="dot", line_color="#FF9100",
                  annotation_text=f"Over-Voltage Limit ({v_max_limit:.2f} p.u.)",
                  annotation_position="top right", annotation_font_color="#FF9100")

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="<b>Bus Voltage Profiles (p.u.) Across Network</b>", font=dict(size=15)),
        xaxis=dict(title="Bus ID Number", dtick=2, gridcolor="#21262D"),
        yaxis=dict(title="Voltage Magnitude (p.u.)", range=[0.0, 1.12], gridcolor="#21262D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380
    )
    return fig


def plot_line_loading_comparison(
    base_res: PowerFlowResults,
    post_res: Optional[PowerFlowResults] = None,
    top_n: int = 15
) -> go.Figure:
    """
    Compares branch loading percentages between pre-fault baseline and post-healing configuration.
    """
    fig = go.Figure()
    lines = sorted(base_res.line_loadings.keys())[:top_n]
    base_loads = [base_res.line_loadings.get(l, 0.0) for l in lines]

    fig.add_trace(go.Bar(
        x=[f"Line {l}" for l in lines],
        y=base_loads,
        name="Baseline Loading",
        marker_color="#29B6F6"
    ))

    if post_res:
        post_loads = [post_res.line_loadings.get(l, 0.0) for l in lines]
        fig.add_trace(go.Bar(
            x=[f"Line {l}" for l in lines],
            y=post_loads,
            name="Post-Healing Loading",
            marker_color="#00E676"
        ))

    # 100% Thermal ampacity limit line
    fig.add_hline(y=100.0, line_dash="dash", line_color="#FF1744",
                  annotation_text="100% Ampacity Limit", annotation_font_color="#FF1744")

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="<b>Feeder Branch Thermal Loading Comparison (%)</b>", font=dict(size=15)),
        barmode='group',
        xaxis=dict(title="Distribution Branch", gridcolor="#21262D"),
        yaxis=dict(title="Loading (%)", gridcolor="#21262D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380
    )
    return fig


def plot_power_losses_comparison(
    base_res: PowerFlowResults,
    fault_res: Optional[PowerFlowResults] = None,
    post_res: Optional[PowerFlowResults] = None
) -> go.Figure:
    """
    Displays real power loss (kW) comparison across system states.
    """
    labels = ["Normal Baseline"]
    values = [base_res.total_loss_kw]
    colors = ["#00E676"]

    if fault_res:
        labels.append("During Fault")
        values.append(fault_res.total_loss_kw)
        colors.append("#FF1744")

    if post_res:
        labels.append("Post-Healing")
        values.append(post_res.total_loss_kw)
        colors.append("#00E5FF")

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=[f"{v:.1f} kW" for v in values],
        textposition="auto"
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="<b>Active Power Losses (kW)</b>", font=dict(size=15)),
        xaxis=dict(gridcolor="#21262D"),
        yaxis=dict(title="Real Power Loss (kW)", gridcolor="#21262D"),
        height=350
    )
    return fig


def plot_confusion_matrix(metrics: ModelMetrics) -> go.Figure:
    """
    Generates an annotated heatmap of the AI Fault Classifier's confusion matrix.
    """
    labels = metrics.class_labels
    z = metrics.confusion_mat

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=z,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"}
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(
            text=f"<b>AI Fault Classifier Confusion Matrix (Acc: {metrics.classifier_accuracy*100:.1f}%)</b>",
            font=dict(size=14)
        ),
        xaxis=dict(title="Predicted Class", gridcolor="#21262D"),
        yaxis=dict(title="True Ground Truth Class", gridcolor="#21262D"),
        height=400
    )
    return fig


def plot_feature_importance(metrics: ModelMetrics) -> go.Figure:
    """
    Renders top telemetry features contributing to fault section localization.
    """
    features = [f[0] for f in metrics.feature_importances][::-1]
    scores = [f[1] for f in metrics.feature_importances][::-1]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker=dict(color="#00E5FF")
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="<b>Top Telemetry Signals Driving AI Localization</b>", font=dict(size=14)),
        xaxis=dict(title="Relative Gini Importance Score", gridcolor="#21262D"),
        yaxis=dict(gridcolor="#21262D"),
        height=380
    )
    return fig


def plot_reliability_comparison(rel: ReliabilityComparison) -> go.Figure:
    """
    Compares IEEE 1366 SAIDI (outage duration in minutes) and Energy Not Supplied.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Traditional Crew Dispatch",
        x=["SAIDI (Outage Min / Customer)", "ENS (Energy Not Supplied MWh)"],
        y=[rel.saidi_manual_min, rel.ens_manual_mwh],
        marker_color="#FF5252",
        text=[f"{rel.saidi_manual_min:.1f} min", f"{rel.ens_manual_mwh:.2f} MWh"],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="AI Autonomous Self-Healing",
        x=["SAIDI (Outage Min / Customer)", "ENS (Energy Not Supplied MWh)"],
        y=[rel.saidi_ai_min, rel.ens_ai_mwh],
        marker_color="#00E676",
        text=[f"{rel.saidi_ai_min:.2f} min", f"{rel.ens_ai_mwh:.2f} MWh"],
        textposition="auto"
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text=f"<b>Reliability Impact: {rel.saidi_reduction_percent:.1f}% Outage Duration Reduction</b>", font=dict(size=14)),
        barmode='group',
        yaxis=dict(title="Index Value", gridcolor="#21262D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380
    )
    return fig
