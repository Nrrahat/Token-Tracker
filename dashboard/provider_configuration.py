"""Resolve session-only provider settings into transient adapter instances."""

from pydantic import SecretStr
import streamlit as st

from core.enums import ProviderName
from providers.base_provider import ProviderConfigurationError, LLMProvider
from providers.provider_factory import create_provider


_KEYS = {
    ProviderName.OPENAI: "openai_api_key",
    ProviderName.GEMINI: "gemini_api_key",
    ProviderName.OPENROUTER: "openrouter_api_key",
    ProviderName.OLLAMA: "ollama_api_key",
}


def key_for(provider: ProviderName) -> str | None:
    session_key = _KEYS.get(provider)
    if session_key is None:
        return None
    value = st.session_state.get(session_key, "")
    return value.strip() if isinstance(value, str) and value.strip() else None


def provider_ready(provider: ProviderName) -> bool:
    if provider is ProviderName.OLLAMA:
        return True
    if provider is ProviderName.OPENAI_COMPATIBLE:
        return bool(st.session_state.get("compatible_api_key", "").strip()) and bool(
            st.session_state.get("compatible_base_url", "").strip()
        )
    return key_for(provider) is not None


def create_session_provider(provider: ProviderName) -> LLMProvider:
    api_key = key_for(provider)
    base_url = None
    if provider is ProviderName.OLLAMA:
        base_url = st.session_state.get("ollama_base_url", "http://localhost:11434/v1").strip()
    elif provider is ProviderName.OPENAI_COMPATIBLE:
        api_key = st.session_state.get("compatible_api_key", "").strip() or None
        base_url = st.session_state.get("compatible_base_url", "").strip() or None
    elif provider is ProviderName.OPENROUTER:
        base_url = st.session_state.get(
            "openrouter_base_url",
            "https://openrouter.ai/api/v1",
        ).strip()

    if provider is not ProviderName.OLLAMA and not api_key:
        raise ProviderConfigurationError(provider, f"Configure a {provider.value} API key in Provider Settings")
    return create_provider(
        provider,
        SecretStr(api_key) if api_key else None,
        base_url=base_url,
    )