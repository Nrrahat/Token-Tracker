"""Inspect dynamically discovered skills and instruction estimates."""

import streamlit as st

from dashboard.components import empty_state, page_header
from dashboard.runtime import AppServices


def render(services: AppServices) -> None:
    page_header("Skill manager", "Inspect versioned instruction bundles available to agents.")
    skills = services.skill_registry.discover(include_disabled=True)
    if not skills:
        empty_state("No skills were discovered in the configured skills directory.")
        return
    for skill in skills:
        with st.container(border=True):
            heading, tokens, enabled = st.columns([4, 1, 1])
            heading.markdown(f"**{skill.metadata.name}** · v{skill.metadata.version}")
            heading.caption(skill.metadata.description)
            tokens.metric("Est. tokens", f"{skill.estimated_instruction_tokens:,}" if skill.estimated_instruction_tokens is not None else "—")
            enabled_state = enabled.toggle(
                "Enabled",
                value=skill.metadata.enabled,
                key=f"skill_enabled_{skill.metadata.name}",
            )
            if enabled_state != skill.metadata.enabled:
                services.skill_registry.set_enabled(skill.metadata.name, enabled_state)
                st.rerun()
            if skill.metadata.tags:
                st.caption(" · ".join(skill.metadata.tags))
            with st.expander("Instructions"):
                st.markdown(skill.instructions)