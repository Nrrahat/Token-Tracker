"""Execution and token overview dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select

from core.enums import ExecutionStatus
from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices
from database.models import AgentRecord, ExecutionRunRecord, LLMCallRecord, TaskRecord
from exports.report_generator import summary_report_markdown


def render(services: AppServices) -> None:
    page_header("Overview", "A live read on agent activity, token use, and estimated spend.")
    with services.session_factory() as session:
        agent_count = session.scalar(select(func.count()).select_from(AgentRecord)) or 0
        run_count = session.scalar(select(func.count()).select_from(ExecutionRunRecord)) or 0
        success_count = session.scalar(
            select(func.count()).select_from(ExecutionRunRecord).where(
                ExecutionRunRecord.status == ExecutionStatus.SUCCEEDED.value
            )
        ) or 0
        call_count = session.scalar(select(func.count()).select_from(LLMCallRecord)) or 0
        calls = session.scalars(select(LLMCallRecord).order_by(LLMCallRecord.requested_at)).all()

    input_tokens = sum(call.input_tokens or 0 for call in calls)
    output_tokens = sum(call.output_tokens or 0 for call in calls)
    reasoning_tokens = sum(call.reasoning_tokens or 0 for call in calls)
    cached_tokens = sum(call.cached_input_tokens or 0 for call in calls)
    known_costs = [call for call in calls if call.estimated_cost is not None]
    total_cost = sum((call.estimated_cost for call in known_costs), start=0)
    currencies = {call.cost_currency for call in known_costs if call.cost_currency}

    metrics = st.columns(4)
    metrics[0].metric("Executions", f"{run_count:,}", f"{agent_count} agents")
    metrics[1].metric("LLM calls", f"{call_count:,}", help=f"{input_tokens + output_tokens:,} reported tokens")
    metrics[2].metric("Successful runs", f"{success_count:,}", f"{_success_rate(success_count, run_count):.0%}")
    cost_label = f"{next(iter(currencies))} {total_cost:.4f}" if len(currencies) == 1 else "—"
    metrics[3].metric("Estimated cost", cost_label, help="Configured models only; otherwise not available.")
    st.download_button(
        "Download summary report",
        data=summary_report_markdown(
            services.analytics_service.token_analytics(),
            services.evaluation_manager.summary(),
        ),
        file_name="agent-observatory-summary.md",
        mime="text/markdown",
        icon="⬇",
    )

    st.subheader("Token volume")
    token_columns = st.columns(4)
    token_columns[0].metric("Input", f"{input_tokens:,}")
    token_columns[1].metric("Output", f"{output_tokens:,}")
    token_columns[2].metric("Reasoning", f"{reasoning_tokens:,}")
    token_columns[3].metric("Cached input", f"{cached_tokens:,}")

    if not calls:
        empty_state("No LLM calls recorded yet. Run an agent task to populate the dashboard.")
        return

    chart_data = pd.DataFrame(
        [
            {"Model": call.model, "Input": call.input_tokens or 0, "Output": call.output_tokens or 0}
            for call in calls
        ]
    ).groupby("Model", as_index=False)[["Input", "Output"]].sum()
    chart = px.bar(
        chart_data,
        x="Model",
        y=["Input", "Output"],
        barmode="group",
        color_discrete_sequence=["#176b58", "#e59645"],
    )
    chart.update_layout(
        height=330,
        margin={"t": 16, "r": 12, "b": 12, "l": 12},
        legend_title_text="Token type",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, use_container_width=True)


def _success_rate(success_count: int, run_count: int) -> float:
    return success_count / run_count if run_count else 0.0