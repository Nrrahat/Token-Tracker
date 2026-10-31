"""Browse persisted execution traces and normalized LLM calls."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices
from database.models import ExecutionRunRecord
from exports.json_exporter import execution_trace_json


def render(services: AppServices) -> None:
    page_header("Execution traces", "Inspect the ordered events and LLM-call records for each run.")
    with services.session_factory() as session:
        runs = session.scalars(
            select(ExecutionRunRecord).order_by(ExecutionRunRecord.started_at.desc())
        ).all()
        options = [
            {
                "id": run.id,
                "status": run.status,
                "started_at": run.started_at,
                "agent_id": run.agent_id,
                "task_id": run.task_id,
            }
            for run in runs
        ]
    if not options:
        empty_state("No runs recorded yet.")
        return

    default_run_id = st.session_state.get("last_run_id")
    default_index = next(
        (index for index, run in enumerate(options) if run["id"] == default_run_id),
        0,
    )
    run_id = st.selectbox(
        "Execution run",
        options=[run["id"] for run in options],
        index=default_index,
        format_func=lambda value: _format_run(next(run for run in options if run["id"] == value)),
    )
    trace = services.trace_manager.get_trace(run_id)
    if trace is None:
        empty_state("This run no longer exists.")
        return
    st.download_button(
        "Export trace JSON",
        data=execution_trace_json(trace),
        file_name=f"trace-{run_id}.json",
        mime="application/json",
        icon="⬇",
    )

    st.subheader("Event sequence")
    if trace.events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": event.sequence,
                        "Event": event.event_type,
                        "Status": event.status,
                        "Time": event.occurred_at,
                        "Duration ms": event.duration_ms,
                        "Details": event.metadata,
                        "Error": event.error_details,
                    }
                    for event in trace.events
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.subheader("LLM calls")
    if trace.llm_calls:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Provider": call.provider,
                        "Model": call.model,
                        "Status": call.status,
                        "Latency ms": round(call.latency_ms, 2),
                        "Input": call.input_tokens,
                        "Output": call.output_tokens,
                        "Reasoning": call.reasoning_tokens,
                        "Cached input": call.cached_input_tokens,
                        "Component estimate": call.estimated_component_tokens,
                        "Attribution gap": call.attribution_gap_tokens,
                        "Est. cost": f"{call.cost_currency or ''} {call.estimated_cost}" if call.estimated_cost is not None else None,
                    }
                    for call in trace.llm_calls
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        empty_state("No LLM calls were recorded for this run.")


def _format_run(run: dict[str, object]) -> str:
    started = run["started_at"]
    timestamp = started.strftime("%Y-%m-%d %H:%M:%S") if started else "time unknown"
    return f"{timestamp} · {run['status']} · {str(run['id'])[:8]}"