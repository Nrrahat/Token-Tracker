"""Interactive task submission to configured agents."""

import streamlit as st
from sqlalchemy import select

from dashboard.components import empty_state, page_header
from dashboard.provider_configuration import create_session_provider, provider_ready
from dashboard.runtime import AppServices
from database.models import ExecutionEventRecord, ExecutionRunRecord


def render(services: AppServices) -> None:
    page_header("Task execution", "Submit a task to an active agent and inspect its completed run.")
    active_agents = [agent for agent in services.agent_manager.list_agents() if agent.is_active]
    if not active_agents:
        empty_state("No active agents yet. Create and activate one in Agent builder.")
    else:
        selected_id = st.selectbox(
            "Agent",
            options=[agent.agent_id for agent in active_agents],
            format_func=lambda value: next(agent.config.name for agent in active_agents if agent.agent_id == value),
        )
        selected_agent = next(agent for agent in active_agents if agent.agent_id == selected_id)
        if not provider_ready(selected_agent.config.provider):
            st.warning(f"Configure {selected_agent.config.provider.value} in Provider settings before running tasks.")
        else:
            with st.form("task_submission"):
                prompt = st.text_area("Task", height=180, placeholder="Describe the complete task for this agent.")
                submitted = st.form_submit_button("Run task", type="primary", icon="▶")
            if submitted:
                if not prompt.strip():
                    st.error("Enter a task before running the agent.")
                else:
                    try:
                        provider = create_session_provider(selected_agent.config.provider)
                        with st.spinner("Agent is working…"):
                            result = services.agent_executor.execute(selected_agent.agent_id, prompt, provider)
                        st.session_state["last_run_id"] = result.run_id
                    except Exception as error:
                        st.error(f"Task could not be started: {type(error).__name__}")
                        st.caption("Check provider configuration and application logs for details.")

    _render_last_result(services)


def _render_last_result(services: AppServices) -> None:
    run_id = st.session_state.get("last_run_id")
    if not run_id:
        return

    with services.session_factory() as session:
        run = session.get(ExecutionRunRecord, run_id)
        failure_events = (
            session.scalars(
                select(ExecutionEventRecord)
                .where(
                    ExecutionEventRecord.run_id == run_id,
                    ExecutionEventRecord.status == "failed",
                )
                .order_by(ExecutionEventRecord.sequence)
            ).all()
            if run is not None and run.status != "succeeded"
            else []
        )
        run_result = (
            {
                "status": run.status,
                "result": run.result,
                "error_message": run.error_message,
                "failure_events": [
                    {
                        "event_type": event.event_type,
                        "error_details": event.error_details,
                        "metadata": event.event_data,
                    }
                    for event in failure_events
                ],
            }
            if run is not None
            else None
        )

    st.divider()
    st.subheader("Latest task result")
    if run_result is None:
        st.error(f"Saved task run {run_id} could not be found.")
    else:
        if run_result["result"]:
            st.markdown(run_result["result"])
        elif run_result["error_message"]:
            st.error(f"Run failed: {run_result['error_message']}")
            with st.expander("Why did this task fail?"):
                if run_result["failure_events"]:
                    for event in run_result["failure_events"]:
                        metadata = event["metadata"]
                        context = " · ".join(
                            f"{label}: {metadata[key]}"
                            for key, label in (
                                ("provider", "Provider"),
                                ("model", "Model"),
                                ("tool_name", "Tool"),
                            )
                            if metadata.get(key)
                        )
                        st.markdown(f"**{event['event_type'].replace('_', ' ').title()}**")
                        if event["error_details"]:
                            st.write(f"Reason: {event['error_details']}")
                        if context:
                            st.caption(context)
                else:
                    st.write("No detailed failure event was recorded for this run.")
                st.caption(
                    "Provider error messages are redacted; check the provider key, model access, "
                    "network connection, and account limits if an LLM request failed."
                )
        else:
            st.info("This run did not produce a response.")

        if run_result["status"] == "succeeded":
            st.success(f"Completed · run {run_id}")
        else:
            st.error(f"Run {run_result['status']} · run {run_id}")

        st.button(
            "Open execution trace",
            icon=":material/visibility:",
            key="open_latest_execution_trace",
            on_click=_open_execution_trace,
        )

    if st.button(
        "Clear displayed result",
        icon=":material/delete:",
        key="clear_latest_task_result",
    ):
        st.session_state.pop("last_run_id", None)
        st.rerun()


def _open_execution_trace() -> None:
    st.session_state["workspace_page"] = "Execution traces"
