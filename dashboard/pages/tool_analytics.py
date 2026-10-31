"""Persisted tool use, latency, failures, duplicate calls, and result attribution."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices
from exports.csv_exporter import tool_analytics_csv


def render(services: AppServices) -> None:
    page_header("Tool analytics", "Tool execution is separate from LLM usage; result tokens count only when attributed to a later request.")
    rows = services.analytics_service.tool_analytics()
    if not rows:
        empty_state("No persisted tool executions are available yet.")
        return

    calls = sum(row.calls for row in rows)
    failures = sum(row.failures for row in rows)
    duplicates = sum(row.duplicate_calls for row in rows)
    tokens_included = sum(row.included_result_tokens for row in rows)
    metrics = st.columns(4)
    metrics[0].metric("Tool calls", f"{calls:,}")
    metrics[1].metric("Failures", f"{failures:,}", f"{failures / calls:.0%}" if calls else "0%")
    metrics[2].metric("Repeated arguments", f"{duplicates:,}")
    metrics[3].metric("Result tokens included", f"{tokens_included:,}", help="Estimated result tokens found in subsequent LLM-call component records")

    frame = pd.DataFrame(
        [
            {
                "Tool": row.tool_name,
                "Calls": row.calls,
                "Failures": row.failures,
                "Duration ms": row.total_duration_ms,
                "Output bytes": row.total_output_size_bytes,
                "Output token estimate": row.estimated_output_tokens,
                "Result tokens included": row.included_result_tokens,
            }
            for row in rows
        ]
    )
    duration_chart = px.bar(
        frame,
        x="Tool",
        y="Duration ms",
        color_discrete_sequence=["#176b58"],
    )
    token_chart = px.bar(
        frame,
        x="Tool",
        y=["Output token estimate", "Result tokens included"],
        barmode="group",
        color_discrete_sequence=["#e59645", "#4d8ca8"],
    )
    for chart in (duration_chart, token_chart):
        chart.update_layout(height=320, margin={"t": 16, "r": 8, "b": 8, "l": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    charts = st.columns(2)
    charts[0].plotly_chart(duration_chart, use_container_width=True)
    charts[1].plotly_chart(token_chart, use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Export tool analytics CSV",
        data=tool_analytics_csv(rows),
        file_name="tool-analytics.csv",
        mime="text/csv",
        icon="⬇",
    )