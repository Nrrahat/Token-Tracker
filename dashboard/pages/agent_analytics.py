"""Per-agent execution, call, token, and estimated-cost analytics."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices


def render(services: AppServices) -> None:
    page_header("Agent analytics", "Compare agent success, reported tokens, call volume, tool use, and estimated costs.")
    analytics = services.analytics_service.agent_analytics()
    if not analytics.agents:
        empty_state("Create and run an agent to start collecting agent-level analytics.")
        return

    frame = pd.DataFrame(
        [
            {
                "Agent": row.agent_name,
                "Executions": row.executions,
                "Success rate": row.success_rate,
                "LLM calls / task": row.average_llm_calls_per_execution,
                "Tokens / task": row.average_reported_tokens_per_execution,
                "Tool calls / task": row.average_tool_calls_per_execution,
                "Estimated costs": json.dumps(
                    {currency: str(amount) for currency, amount in row.estimated_costs.items()},
                    sort_keys=True,
                ),
            }
            for row in analytics.agents
        ]
    )
    chart = px.bar(
        frame,
        x="Agent",
        y="Success rate",
        color="Success rate",
        range_y=[0, 1],
        color_continuous_scale=["#e59645", "#176b58"],
    )
    chart.update_layout(height=300, margin={"t": 16, "r": 8, "b": 8, "l": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)