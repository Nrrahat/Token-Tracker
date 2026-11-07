"""Automatic execution checks and human response-quality evaluations."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices
from database.models import ExecutionRunRecord
from evaluation.schemas import HumanEvaluationSubmission


def render(services: AppServices) -> None:
    page_header("Evaluation", "Execution-derived success is automatic; response quality scores are explicitly human.")
    summary = services.evaluation_manager.summary()
    columns = st.columns(4)
    columns[0].metric("Automatic evaluations", summary.automatic_evaluations)
    columns[1].metric(
        "Automatic task success",
        f"{summary.automatic_success_rate:.0%}" if summary.automatic_success_rate is not None else "—",
    )
    columns[2].metric("Human evaluations", summary.human_evaluations)
    columns[3].metric(
        "Human quality average",
        f"{summary.average_quality:.1f} / 5" if summary.average_quality is not None else "—",
    )

    with services.session_factory() as session:
        runs = session.scalars(
            select(ExecutionRunRecord)
            .where(ExecutionRunRecord.finished_at.is_not(None))
            .order_by(ExecutionRunRecord.started_at.desc())
        ).all()
        options = [
            {"id": run.id, "status": run.status, "result": run.result, "started_at": run.started_at}
            for run in runs
        ]
    if not options:
        empty_state("No completed runs are available to evaluate.")
        return

    run_id = st.selectbox(
        "Execution run",
        [run["id"] for run in options],
        format_func=lambda value: _format_run(next(run for run in options if run["id"] == value)),
    )
    selected = next(run for run in options if run["id"] == run_id)
    if selected["result"]:
        with st.expander("Agent response"):
            st.markdown(selected["result"])

    if st.button("Record automatic execution evaluation", icon=":material/check:"):
        services.evaluation_manager.evaluate_execution(run_id)
        st.success("Execution status and tool events recorded as an automatic evaluation.")
        st.rerun()

    st.subheader("Human response assessment")
    with st.form("human_evaluation"):
        evaluator_name = st.text_input("Evaluator", value="reviewer")
        scores = st.columns(3)
        relevance = scores[0].slider("Relevance", 1, 5, 3)
        completeness = scores[1].slider("Completeness", 1, 5, 3)
        quality = scores[2].slider("Overall quality", 1, 5, 3)
        notes = st.text_area("Notes", max_chars=4000, height=100)
        submitted = st.form_submit_button("Save human evaluation", type="primary")
    if submitted:
        submission = HumanEvaluationSubmission(
            run_id=run_id,
            evaluator_name=evaluator_name,
            response_relevance=relevance,
            response_completeness=completeness,
            output_quality=quality,
            notes=notes,
        )
        services.evaluation_manager.submit_human_evaluation(submission)
        st.success("Human evaluation saved separately from automatic execution metrics.")
        st.rerun()

    evaluations = services.evaluation_manager.list_evaluations(run_id)
    if evaluations:
        st.subheader("Evaluation history")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Type": item.evaluator_type.value,
                        "Evaluator": item.evaluator_name,
                        "Task success": item.task_success,
                        "Tool success": item.tool_use_success,
                        "Relevance": item.response_relevance,
                        "Completeness": item.response_completeness,
                        "Quality": item.output_quality,
                        "Notes": item.notes,
                        "Evaluated at": item.evaluated_at,
                    }
                    for item in evaluations
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _format_run(run: dict[str, object]) -> str:
    started = run["started_at"]
    timestamp = started.strftime("%Y-%m-%d %H:%M:%S") if started else "time unknown"
    return f"{timestamp} · {run['status']} · {str(run['id'])[:8]}"