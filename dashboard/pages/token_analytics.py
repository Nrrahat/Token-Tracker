"""Persisted token attribution and usage analytics."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices
from exports.csv_exporter import token_usage_csv


def render(services: AppServices) -> None:
    page_header("Token analytics", "Provider-reported usage and local component estimates are shown separately.")
    analytics = services.analytics_service.token_analytics()
    totals = analytics.totals
    metrics = st.columns(4)
    metrics[0].metric("Reported input", f"{totals.reported_input_tokens:,}")
    metrics[1].metric("Reported output", f"{totals.reported_output_tokens:,}")
    metrics[2].metric("Cached input", f"{totals.cached_input_tokens:,}")
    metrics[3].metric("Reported calls", f"{totals.reported_calls:,}", help=f"{totals.llm_calls:,} total calls")

    estimated = st.columns(3)
    estimated[0].metric("Component estimate", f"{totals.estimated_component_tokens:,}", help="Local tokenizer estimates")
    estimated[1].metric("Attribution gap", f"{totals.attribution_gap_tokens:,}", help="Reported input minus estimated components; not evidence of hidden prompts")
    cost_text = " · ".join(f"{currency} {amount:.6f}" for currency, amount in sorted(totals.estimated_costs.items())) or "—"
    estimated[2].metric("Estimated cost", cost_text, help="Estimated costs grouped by currency")
    st.download_button(
        "Export per-call token usage CSV",
        data=token_usage_csv(services.session_factory),
        file_name="token-usage.csv",
        mime="text/csv",
        icon="⬇",
    )

    if not totals.llm_calls:
        empty_state("No LLM call data is available yet.")
        return

    left, right = st.columns(2)
    if analytics.by_model:
        model_frame = pd.DataFrame(
            [
                {"Model": row.key, "Reported input": row.input_tokens, "Reported output": row.output_tokens}
                for row in analytics.by_model
            ]
        )
        chart = px.bar(
            model_frame,
            x="Model",
            y=["Reported input", "Reported output"],
            barmode="group",
            color_discrete_sequence=["#176b58", "#e59645"],
        )
        chart.update_layout(height=320, margin={"t": 16, "r": 8, "b": 8, "l": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        left.plotly_chart(chart, use_container_width=True)
    if analytics.by_day:
        day_frame = pd.DataFrame(
            [
                {"Day": row.day, "Reported input": row.input_tokens, "Reported output": row.output_tokens}
                for row in analytics.by_day
            ]
        )
        chart = px.line(
            day_frame,
            x="Day",
            y=["Reported input", "Reported output"],
            markers=True,
            color_discrete_sequence=["#176b58", "#e59645"],
        )
        chart.update_layout(height=320, margin={"t": 16, "r": 8, "b": 8, "l": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        right.plotly_chart(chart, use_container_width=True)

    st.subheader("Prompt component estimates")
    if analytics.by_component:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Component": row.component,
                        "Source": row.source_id,
                        "Estimated tokens": row.estimated_tokens,
                    }
                    for row in analytics.by_component
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        empty_state("No component-level estimates were stored for these calls.")