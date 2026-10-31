"""Interactive task submission to configured agents."""

import streamlit as st

from dashboard.components import empty_state, page_header
from dashboard.provider_configuration import create_session_provider, provider_ready
from dashboard.runtime import AppServices


def render(services: AppServices) -> None:
    page_header("Task execution", "Submit a task to an active agent and inspect its completed run.")
    active_agents = [agent for agent in services.agent_manager.list_agents() if agent.is_active]
    if not active_agents:
        empty_state("No active agents yet. Create and activate one in Agent builder.")
        return
    selected_id = st.selectbox(
        "Agent",
        options=[agent.agent_id for agent in active_agents],
        format_func=lambda value: next(agent.config.name for agent in active_agents if agent.agent_id == value),
    )
    selected_agent = next(agent for agent in active_agents if agent.agent_id == selected_id)
    if not provider_ready(selected_agent.config.provider):
        st.warning(f"Configure {selected_agent.config.provider.value} in Provider settings before running tasks.")
        return
    with st.form("task_submission"):
        prompt = st.text_area("Task", height=180, placeholder="Describe the complete task for this agent.")
        submitted = st.form_submit_button("Run task", type="primary", icon="▶")
    if submitted:
        if not prompt.strip():
            st.error("Enter a task before running the agent.")
            return
        try:
            provider = create_session_provider(selected_agent.config.provider)
            with st.spinner("Agent is working…"):
                result = services.agent_executor.execute(selected_agent.agent_id, prompt, provider)
            st.session_state["last_run_id"] = result.run_id
            if result.answer:
                st.markdown("### Response")
                st.markdown(result.answer)
            if result.status.value == "succeeded":
                st.success(f"Completed · {result.iterations} iterations · {result.tool_calls} tool calls · run {result.run_id}")
            else:
                st.error(f"Run {result.status.value}: {result.error_type or 'unknown error'} · {result.run_id}")
            if st.button("Open execution trace", icon="⌁"):
                st.session_state["workspace_page"] = "Execution traces"
                st.rerun()
        except Exception as error:
            st.error(f"Task could not be started: {type(error).__name__}")
            st.caption("Check provider configuration and application logs for details.")