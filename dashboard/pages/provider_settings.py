"""Session-only provider credential configuration."""

from typing import Any

import streamlit as st

from dashboard.components import page_header


def render() -> None:
    page_header("Provider settings", "Keys and endpoints are kept in this browser session, never in SQLite.")
    with st.expander("OpenAI", expanded=True):
        _text_input("OpenAI API key", "openai_api_key", type="password", placeholder="sk-…")
        _status("openai_api_key")
    with st.expander("Google Gemini"):
        _text_input("Gemini API key", "gemini_api_key", type="password")
        _status("gemini_api_key")
    with st.expander("OpenRouter"):
        _text_input("OpenRouter API key", "openrouter_api_key", type="password")
        _text_input(
            "OpenRouter base URL",
            "openrouter_base_url",
            value="https://openrouter.ai/api/v1",
        )
        _status("openrouter_api_key")
    with st.expander("Groq"):
        _text_input("Groq API key", "groq_api_key", type="password")
        st.caption("Uses Groq's OpenAI-compatible API endpoint.")
        _status("groq_api_key")
    with st.expander("Ollama"):
        _text_input(
            "Ollama base URL",
            "ollama_base_url",
            value="http://localhost:11434/v1",
            help="The local Ollama OpenAI-compatible endpoint. Ollama does not require an API key by default.",
        )
        _text_input("Optional API key", "ollama_api_key", type="password")
        st.caption("Make sure Ollama is running and the selected model has been pulled locally.")
    with st.expander("Custom OpenAI-compatible API"):
        _text_input(
            "Compatible API base URL",
            "compatible_base_url",
            placeholder="https://api.example.com/v1",
        )
        _text_input("Compatible API key", "compatible_api_key", type="password")
        configured = bool(
            st.session_state.get("compatible_base_url", "").strip()
            and st.session_state.get("compatible_api_key", "").strip()
        )
        st.caption("Configured for this session" if configured else "Base URL and API key are both required")
    st.caption("Anthropic is not implemented yet.")
    if st.button("Forget all provider credentials", icon="🗑️", on_click=_forget_keys):
        st.rerun()


def _text_input(label: str, session_key: str, **kwargs: Any) -> None:
    widget_key = f"_provider_setting_{session_key}"
    default_value = kwargs.pop("value", "")
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(session_key, default_value)
    st.text_input(
        label,
        key=widget_key,
        on_change=_persist_setting,
        args=(widget_key, session_key),
        **kwargs,
    )


def _persist_setting(widget_key: str, session_key: str) -> None:
    st.session_state[session_key] = st.session_state[widget_key]


def _status(session_key: str) -> None:
    value = st.session_state.get(session_key, "")
    configured = isinstance(value, str) and bool(value.strip())
    st.caption("Key configured for this session" if configured else "No key configured")


def _forget_keys() -> None:
    for session_key in (
        "openai_api_key",
        "gemini_api_key",
        "openrouter_api_key",
        "openrouter_base_url",
        "groq_api_key",
        "ollama_base_url",
        "ollama_api_key",
        "compatible_api_key",
        "compatible_base_url",
    ):
        st.session_state.pop(session_key, None)
        st.session_state.pop(f"_provider_setting_{session_key}", None)