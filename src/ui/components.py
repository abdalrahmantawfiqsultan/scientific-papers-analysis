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
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#0284C7")
    ))

    fig.update_layout(
        title=f"Adoption Velocity: <b>{method_name}</b> in <b>{domain_name}</b>",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#334155", title="Publication Year"),
        yaxis=dict(gridcolor="#334155", title="Paper Count"),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
