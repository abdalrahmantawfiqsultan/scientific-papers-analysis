import plotly.graph_objects as go
import streamlit as st

def render_stepper(current_phase: int):
    """Renders a 4-phase visual execution stepper."""
    phases = ["1. Grounding", "2. Discovery", "3. Validation", "4. Synthesis"]
    cols = st.columns(4)
    
    for idx, (col, phase) in enumerate(zip(cols, phases), start=1):
        with col:
            if idx < current_phase:
                st.markdown(f"✅ **{phase}**")
            elif idx == current_phase:
                st.markdown(f"⏳ **:blue[{phase}]**")
            else:
                st.markdown(f"⚪ {phase}")

def render_temporal_chart(trend_data: dict, method_name: str, domain_name: str):
    """Renders a Plotly line chart tracking method adoption over years."""
    years = list(trend_data.keys())
    counts = list(trend_data.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=counts,
        mode="lines+markers+text",
        text=counts,
        textposition="top center",
        textfont=dict(size=11, color="#A5B4FC"),
        line=dict(color="#818CF8", width=3),
        marker=dict(size=8, color="#6366F1", line=dict(width=2, color="#A5B4FC")),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.08)"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{method_name}</b> in <b>{domain_name}</b> — Publication Trend",
            font=dict(size=14, color="#CBD5E1"),
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#0C1222",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="#1E293B", title="Year", title_font=dict(size=12)),
        yaxis=dict(gridcolor="#1E293B", title="Papers", title_font=dict(size=12)),
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)
