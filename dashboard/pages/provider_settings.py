"""Session-only provider credential configuration."""

import streamlit as st

from dashboard.components import page_header


def render() -> None:
    page_header("Provider settings", "Keys and endpoints are kept in this browser session, never in SQLite.")
    with st.expander("OpenAI", expanded=True):
        st.text_input("OpenAI API key", type="password", key="openai_api_key", placeholder="sk-…")
        _status("openai_api_key")
    with st.expander("Google Gemini"):
        st.text_input("Gemini API key", type="password", key="gemini_api_key")
        _status("gemini_api_key")
    with st.expander("OpenRouter"):
        st.text_input("OpenRouter API key", type="password", key="openrouter_api_key")
        st.text_input(
            "OpenRouter base URL",
            key="openrouter_base_url",
            value="https://openrouter.ai/api/v1",
        )
        _status("openrouter_api_key")
    with st.expander("Ollama"):
        st.text_input(
            "Ollama base URL",
            key="ollama_base_url",
            value="http://localhost:11434/v1",
            help="The local Ollama OpenAI-compatible endpoint. Ollama does not require an API key by default.",
        )
        st.text_input("Optional API key", type="password", key="ollama_api_key")
        st.caption("Make sure Ollama is running and the selected model has been pulled locally.")
    with st.expander("Custom OpenAI-compatible API"):
        st.text_input("Compatible API base URL", key="compatible_base_url", placeholder="https://api.example.com/v1")
        st.text_input("Compatible API key", type="password", key="compatible_api_key")
        configured = bool(
            st.session_state.get("compatible_base_url", "").strip()
            and st.session_state.get("compatible_api_key", "").strip()
        )
        st.caption("Configured for this session" if configured else "Base URL and API key are both required")
    st.caption("Anthropic is not implemented yet.")
    if st.button("Forget all provider credentials", icon="🗑️", on_click=_forget_keys):
        st.rerun()


def _status(session_key: str) -> None:
    value = st.session_state.get(session_key, "")
    configured = isinstance(value, str) and bool(value.strip())
    st.caption("Key configured for this session" if configured else "No key configured")


def _forget_keys() -> None:
    for key in (
        "openai_api_key",
        "gemini_api_key",
        "openrouter_api_key",
        "ollama_api_key",
        "compatible_api_key",
    ):
        st.session_state.pop(key, None)