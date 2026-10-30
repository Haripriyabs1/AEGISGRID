"""
Network Graph Visualizer Module
Generates rich, interactive 2D distribution network single-line diagrams using Plotly.
Visualizes buses, lines, switches, distributed generators, fault location markers,
and service restoration tie-switch paths.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pandapower as pp
import plotly.graph_objects as go
from core.power_flow import PowerFlowResults


def create_network_graph(
    net: pp.pandapowerNet,
    results: Optional[PowerFlowResults] = None,
    fault_line_id: Optional[int] = None,
    isolated_line_id: Optional[int] = None,
    restored_tie_id: Optional[int] = None,
    fault_location_pct: float = 50.0,
    title: str = "Distribution Network Topology"
) -> go.Figure:
    """
    Renders an interactive distribution network single-line topology graph.
    """
    fig = go.Figure()
    coords = getattr(net, 'bus_coords', {})
    if not coords:
        # Fallback coordinates if not set
        coords = {int(b): (float(b), 2.0) for b in net.bus.index}

    # Extract results if available
    bus_vols = results.bus_voltages if results else {int(b): 1.0 for b in net.bus.index}
    line_loads = results.line_loadings if results else {int(l): 0.0 for l in net.line.index}
    isolated_buses = set(results.isolated_buses) if results else set()

    # Infeed buses (Substations)
    substation_buses = set(net.ext_grid['bus'].astype(int))
    # DG buses
    dg_buses = set()
    if hasattr(net, 'sgen') and len(net.sgen) > 0:
        dg_buses.update(net.sgen['bus'].astype(int))
    if hasattr(net, 'storage') and len(net.storage) > 0:
        dg_buses.update(net.storage['bus'].astype(int))

    load_priorities = getattr(net, 'load_priorities', {})
    tie_lines = set(getattr(net, 'tie_lines', []))

    # 1. DRAW LINES (EDGES)
    # Group lines by category for clean visual representation:
    # - Normal active lines
    # - Overloaded lines
    # - Faulted line
    # - Isolated line
    # - Restored tie line
    # - Open tie lines (inactive)
    
    for l_idx, row in net.line.iterrows():
        l_id = int(l_idx)
        fb = int(row['from_bus'])
        tb = int(row['to_bus'])
        
        if fb not in coords or tb not in coords:
            continue
            
        x0, y0 = coords[fb]
        x1, y1 = coords[tb]
        in_service = bool(row['in_service'])
        loading = line_loads.get(l_id, 0.0)
        is_tie = (l_id in tie_lines)

        # Determine line style and color
        if l_id == fault_line_id:
            line_color = "#FF1744"  # Neon Red (Fault)
            line_width = 5.5
            line_dash = "solid"
            status_desc = "FAULTED ZONE (Active Short-Circuit)"
        elif l_id == isolated_line_id or (not in_service and not is_tie):
            line_color = "#616161"  # Dark Charcoal (Isolated)
            line_width = 3.0
            line_dash = "dash"
            status_desc = "ISOLATED (Sectionalizers Open)"
        elif l_id == restored_tie_id:
            line_color = "#00E5FF"  # Electric Cyan (Restored tie line)
            line_width = 5.0
            line_dash = "solid"
            status_desc = "RESTORED TIE-PATH (Tie-Switch Closed)"
        elif is_tie and not in_service:
            line_color = "#455A64"  # Translucent Slate (Normally Open tie switch)
            line_width = 2.0
            line_dash = "dot"
            status_desc = "Normally Open Tie-Line"
        elif loading > 100.0:
            line_color = "#FF9100"  # Orange (Overloaded)
            line_width = 4.5
            line_dash = "solid"
            status_desc = f"OVERLOADED ({loading:.1f}%)"
        else:
            line_color = "#00E676"  # Emerald Green (Normal Active)
            line_width = 3.5
            line_dash = "solid"
            status_desc = f"Normal Active ({loading:.1f}%)"

        hover_text = (
            f"<b>{row.get('name', f'Line {l_id}')}</b><br>"
            f"Status: {status_desc}<br>"
            f"From Bus: {fb} -> To Bus: {tb}<br>"
            f"Loading: {loading:.1f}%<br>"
            f"R: {row['r_ohm_per_km']:.3f} Ω/km | X: {row['x_ohm_per_km']:.3f} Ω/km"
        )

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(color=line_color, width=line_width, dash=line_dash),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))

        # Add middle marker for switches / tie-lines
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0
        if is_tie or l_id == restored_tie_id:
            switch_sym = "circle" if in_service else "circle-open"
            switch_col = "#00E5FF" if in_service else "#78909C"
            switch_text = f"Tie-Switch {l_id}: {'CLOSED' if in_service else 'OPEN'}"
            fig.add_trace(go.Scatter(
                x=[xm], y=[ym],
                mode='markers',
                marker=dict(symbol=switch_sym, size=11, color=switch_col, line=dict(color="#FFFFFF", width=1.5)),
                hoverinfo='text',
                hovertext=switch_text,
                showlegend=False
            ))

    # 2. DRAW FAULT FLASH MARKER (if fault present)
    if fault_line_id is not None and fault_line_id in net.line.index:
        fb = int(net.line.at[fault_line_id, 'from_bus'])
        tb = int(net.line.at[fault_line_id, 'to_bus'])
        if fb in coords and tb in coords:
            xf0, yf0 = coords[fb]
            xf1, yf1 = coords[tb]
            pct = fault_location_pct / 100.0
            xf = xf0 + pct * (xf1 - xf0)
            yf = yf0 + pct * (yf1 - yf0)
            
            fig.add_trace(go.Scatter(
                x=[xf], y=[yf],
                mode='markers+text',
                marker=dict(symbol='star-triangle-up', size=24, color='#FF1744',
                            line=dict(color='#FFFFFF', width=2)),
                text=["⚠️ FAULT"],
                textposition="top center",
                textfont=dict(size=12, color='#FF1744', family="Arial Black"),
                hoverinfo='text',
                hovertext=f"<b>FAULT LOCATION</b><br>Line {fault_line_id} at {fault_location_pct:.1f}%",
                name="Active Fault Location",
                showlegend=True
            ))

    # 3. DRAW BUSES (NODES)
    bus_x, bus_y = [], []
    bus_colors = []
    bus_sizes = []
    bus_symbols = []
    bus_hover = []
    bus_labels = []

    for b_idx in net.bus.index:
        b_id = int(b_idx)
        if b_id not in coords:
            continue
        x, y = coords[b_id]
        bus_x.append(x)
        bus_y.append(y)
        bus_labels.append(str(b_id))

        v_pu = bus_vols.get(b_id, 1.0)
        is_isolated = (b_id in isolated_buses)
        is_sub = (b_id in substation_buses)
        is_dg = (b_id in dg_buses)

        p_info = load_priorities.get(b_id, {})
        cat = p_info.get('category', 'Residential')
        p_val = p_info.get('p_mw', 0.0)

        # Node styling based on state
        if is_sub:
            sym = "square"
            size = 20
            color = "#FFD600"  # Gold Substation
            node_type = "Main Substation Infeed (Slack Bus)"
        elif is_isolated:
            sym = "circle"
            size = 14
            color = "#D50000"  # Crimson Blackout
            node_type = "DE-ENERGIZED ISLAND (Outage)"
        elif v_pu < 0.95:
            sym = "circle"
            size = 16
            color = "#FF9100"  # Amber Sag
            node_type = f"Under-Voltage Sag ({v_pu:.3f} p.u.)"
        elif is_dg:
            sym = "diamond"
            size = 18
            color = "#00E5FF"  # Cyan DG
            node_type = f"Distributed Generator / BESS ({cat})"
        elif b_id in [14, 15, 16, 17, 20, 21, 24, 32] and restored_tie_id is not None:
            sym = "circle"
            size = 16
            color = "#00B0FF"  # Bright Blue Restored
            node_type = f"Self-Healed Restored Bus ({v_pu:.3f} p.u.)"
        else:
            sym = "circle"
            size = 15
            color = "#00E676"  # Healthy Emerald
            node_type = f"Energized Healthy ({v_pu:.3f} p.u.)"

        bus_symbols.append(sym)
        bus_sizes.append(size)
        bus_colors.append(color)

        info_text = (
            f"<b>Bus {b_id}</b><br>"
            f"Type: {node_type}<br>"
            f"Voltage: {v_pu:.4f} p.u. ({v_pu * float(net.bus.at[b_id, 'vn_kv']):.2f} kV)<br>"
            f"Load: {p_val:.3f} MW ({cat} Priority)<br>"
            f"Status: {'🔴 BLACKOUT' if is_isolated else '🟢 ENERGIZED'}"
        )
        bus_hover.append(info_text)

    fig.add_trace(go.Scatter(
        x=bus_x,
        y=bus_y,
        mode='markers+text',
        marker=dict(
            symbol=bus_symbols,
            size=bus_sizes,
            color=bus_colors,
            line=dict(color='#FFFFFF', width=1.5)
        ),
        text=bus_labels,
        textposition="bottom center",
        textfont=dict(color='#EEEEEE', size=10, family="Segoe UI"),
        hoverinfo='text',
        hovertext=bus_hover,
        name="Network Buses",
        showlegend=False
    ))

    # Add legend items for clarity
    legend_items = [
        ("Substation Infeed", "#FFD600", "square"),
        ("Healthy Energized", "#00E676", "circle"),
        ("Under-Voltage Sag", "#FF9100", "circle"),
        ("De-Energized (Outage)", "#D50000", "circle"),
        ("Self-Healed Path", "#00E5FF", "circle"),
        ("Tie-Switch (Open)", "#78909C", "circle-open")
    ]
    for label, col, sym in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=col, symbol=sym, line=dict(color="#FFFFFF", width=1)),
            name=label,
            showlegend=True
        ))

    # Figure Layout Aesthetics
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color="#FFFFFF", family="Segoe UI"),
            x=0.03, y=0.96
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(color="#E0E0E0", size=11),
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363D",
            borderwidth=1
        ),
        hovermode="closest",
        height=520
    )

    return fig
