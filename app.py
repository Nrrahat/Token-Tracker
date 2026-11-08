"""Streamlit entry point for Agent Observatory."""

import streamlit as st

from dashboard.components import configure_page
from dashboard.pages import (
    agent_analytics,
    agent_builder,
    evaluation,
    experiment_manager,
    execution_traces,
    overview,
    provider_settings,
    skill_manager,
    task_execution,
    token_analytics,
    tool_builder,
    tool_manager,
    tool_analytics,
)
from dashboard.provider_configuration import key_for
from dashboard.runtime import AppServices, build_services
from core.enums import ProviderName


@st.cache_resource(show_spinner="Preparing local database and tools…")
def _services() -> AppServices:
    return build_services()


configure_page()
services = _services()

with st.sidebar:
    st.markdown("## Agent Observatory")
    st.caption("Execution · tokens · traces")
    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Agent builder",
            "Skill manager",
            "Tool manager",
            "Tool Builder",
            "Task execution",
            "Execution traces",
            "Token analytics",
            "Tool analytics",
            "Agent analytics",
            "Evaluation",
            "Experiment manager",
            "Provider settings",
        ],
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.divider()
    configured = [
        provider.value.title()
        for provider in (ProviderName.OPENAI, ProviderName.GEMINI, ProviderName.OPENROUTER)
        if key_for(provider)
    ]
    st.caption("PROVIDERS")
    st.markdown(", ".join(configured) if configured else "No remote key configured")
    st.caption("Ollama endpoint: session-configured")

pages = {
    "Overview": lambda: overview.render(services),
    "Agent builder": lambda: agent_builder.render(services),
    "Skill manager": lambda: skill_manager.render(services),
    "Tool manager": lambda: tool_manager.render(services),
    "Tool Builder": lambda: tool_builder.render(
        services.custom_tool_store,
        services.tool_registry,
        services.agent_manager,
    ),
    "Task execution": lambda: task_execution.render(services),
    "Execution traces": lambda: execution_traces.render(services),
    "Token analytics": lambda: token_analytics.render(services),
    "Tool analytics": lambda: tool_analytics.render(services),
    "Agent analytics": lambda: agent_analytics.render(services),
    "Evaluation": lambda: evaluation.render(services),
    "Experiment manager": lambda: experiment_manager.render(services),
    "Provider settings": provider_settings.render,
}
pages[st.session_state["workspace_page"]]()