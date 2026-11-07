from core.enums import ProviderName
from dashboard import provider_configuration
from dashboard.pages import provider_settings


def test_session_settings_create_provider_with_provider_specific_values(monkeypatch) -> None:
    state = {
        "gemini_api_key": "gemini-session-key",
        "openrouter_api_key": "router-session-key",
        "groq_api_key": "groq-session-key",
        "ollama_base_url": "http://127.0.0.1:11435/v1",
        "compatible_api_key": "custom-session-key",
        "compatible_base_url": "https://example.test/v1",
    }
    captured = []
    monkeypatch.setattr(provider_configuration.st, "session_state", state)

    def fake_create(provider, api_key, base_url=None):
        captured.append((provider, api_key.get_secret_value() if api_key else None, base_url))
        return object()

    monkeypatch.setattr(provider_configuration, "create_provider", fake_create)

    provider_configuration.create_session_provider(ProviderName.GEMINI)
    provider_configuration.create_session_provider(ProviderName.OPENROUTER)
    provider_configuration.create_session_provider(ProviderName.GROQ)
    provider_configuration.create_session_provider(ProviderName.OLLAMA)
    provider_configuration.create_session_provider(ProviderName.OPENAI_COMPATIBLE)

    assert captured == [
        (ProviderName.GEMINI, "gemini-session-key", None),
        (ProviderName.OPENROUTER, "router-session-key", "https://openrouter.ai/api/v1"),
        (ProviderName.GROQ, "groq-session-key", "https://api.groq.com/openai/v1"),
        (ProviderName.OLLAMA, None, "http://127.0.0.1:11435/v1"),
        (ProviderName.OPENAI_COMPATIBLE, "custom-session-key", "https://example.test/v1"),
    ]


def test_groq_setting_survives_provider_settings_widget_cleanup(monkeypatch) -> None:
    state = {"_provider_setting_groq_api_key": "groq-session-key"}
    monkeypatch.setattr(provider_settings.st, "session_state", state)

    provider_settings._persist_setting("_provider_setting_groq_api_key", "groq_api_key")
    state.pop("_provider_setting_groq_api_key")

    assert provider_configuration.key_for(ProviderName.GROQ) == "groq-session-key"
    assert provider_configuration.provider_ready(ProviderName.GROQ)