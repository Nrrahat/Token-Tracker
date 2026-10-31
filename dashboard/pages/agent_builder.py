"""Agent creation and configuration management."""

import streamlit as st
from pydantic import ValidationError

from core.enums import ProviderName
from core.schemas import AgentConfig
from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices


def render(services: AppServices) -> None:
    page_header("Agent builder", "Configure an independent model, instruction set, skill set, and tool allowlist.")
    agents = services.agent_manager.list_agents()
    mode = st.segmented_control("Mode", ["Create", "Edit"], default="Create")
    selected = None
    if mode == "Edit":
        if not agents:
            empty_state("Create an agent before editing configurations.")
            return
        selected_id = st.selectbox(
            "Agent",
            options=[agent.agent_id for agent in agents],
            format_func=lambda value: next(agent.config.name for agent in agents if agent.agent_id == value),
        )
        selected = next(agent for agent in agents if agent.agent_id == selected_id)

    skills = services.skill_registry.discover()
    skill_options = [skill.metadata.name for skill in skills]
    tool_options = [tool.name for tool in services.tool_registry.list_tools()]
    current = selected.config if selected else None
    with st.form("agent_configuration"):
        left, right = st.columns(2)
        name = left.text_input("Name", value=current.name if current else "")
        description = right.text_input("Description", value=current.description if current else "")
        implemented_providers = [
            ProviderName.OPENAI,
            ProviderName.GEMINI,
            ProviderName.OPENROUTER,
            ProviderName.OLLAMA,
            ProviderName.OPENAI_COMPATIBLE,
        ]
        current_provider = current.provider if current else ProviderName.OPENAI
        provider = st.selectbox(
            "Provider",
            implemented_providers,
            index=implemented_providers.index(current_provider),
            format_func=lambda item: {
                ProviderName.OPENAI: "OpenAI",
                ProviderName.GEMINI: "Google Gemini",
                ProviderName.OPENROUTER: "OpenRouter",
                ProviderName.OLLAMA: "Ollama (local)",
                ProviderName.OPENAI_COMPATIBLE: "Custom OpenAI-compatible",
            }[item],
        )
        default_model = {
            ProviderName.OPENAI: "gpt-4o-mini",
            ProviderName.GEMINI: "gemini-2.5-flash",
            ProviderName.OPENROUTER: "openai/gpt-4o-mini",
            ProviderName.OLLAMA: "llama3.2",
            ProviderName.OPENAI_COMPATIBLE: "model-name",
        }[provider]
        model = st.text_input("Model", value=current.model if current else default_model)
        system_instructions = st.text_area(
            "System instructions", value=current.system_instructions if current else "", height=120
        )
        developer_instructions = st.text_area(
            "Developer instructions", value=current.developer_instructions if current else "", height=100
        )
        temperature = st.slider("Temperature", 0.0, 2.0, current.temperature if current else 0.2, 0.1)
        limits = st.columns(3)
        max_iterations = limits[0].number_input(
            "Max iterations", min_value=1, max_value=100, value=current.max_iterations if current else 10
        )
        max_tool_calls = limits[1].number_input(
            "Max tool calls", min_value=0, max_value=500, value=current.max_tool_calls if current else 20
        )
        request_timeout_seconds = limits[2].number_input(
            "Request timeout (s)",
            min_value=1.0,
            max_value=600.0,
            value=float(current.request_timeout_seconds if current else 60),
        )
        skill_names = st.multiselect(
            "Skills",
            skill_options,
            default=current.skill_names if current else [],
        )
        tool_names = st.multiselect(
            "Tools",
            tool_options,
            default=current.tool_names if current else [],
        )
        submitted = st.form_submit_button("Save agent", type="primary", icon="💾")

    if submitted:
        try:
            config = AgentConfig(
                name=name,
                description=description,
                provider=provider,
                model=model,
                system_instructions=system_instructions,
                developer_instructions=developer_instructions,
                temperature=temperature,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                request_timeout_seconds=request_timeout_seconds,
                skill_names=skill_names,
                tool_names=tool_names,
            )
            saved = (
                services.agent_manager.update_agent(selected.agent_id, config)
                if selected
                else services.agent_manager.create_agent(config)
            )
            st.success(f"Saved {saved.config.name}")
            st.rerun()
        except (ValidationError, LookupError, ValueError) as error:
            st.error(str(error))

    if agents:
        st.divider()
        st.subheader("Existing agents")
        for agent in agents:
            with st.container(border=True):
                columns = st.columns([3, 1, 1])
                columns[0].markdown(f"**{agent.config.name}**  \n{agent.config.provider.value} · {agent.config.model}")
                desired_active = columns[1].toggle(
                    "Active",
                    value=agent.is_active,
                    key=f"active_{agent.agent_id}",
                )
                if desired_active != agent.is_active:
                    services.agent_manager.set_active(agent.agent_id, desired_active)
                    st.rerun()
                if columns[2].button("Delete", key=f"delete_{agent.agent_id}", icon="🗑️"):
                    try:
                        services.agent_manager.delete_agent(agent.agent_id)
                        st.rerun()
                    except RuntimeError as error:
                        st.error(str(error))