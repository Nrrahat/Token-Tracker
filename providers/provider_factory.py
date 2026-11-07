"""Factory for configured provider adapters."""

from typing import Any

from pydantic import SecretStr

from core.enums import ProviderName
from providers.base_provider import LLMProvider, ProviderConfigurationError
from providers.gemini_provider import GeminiProvider
from providers.openai_compatible_provider import (
    GROQ_BASE_URL,
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    GroqProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
)
from providers.openai_provider import OpenAIProvider


def create_provider(
    provider: ProviderName,
    api_key: SecretStr | str | None = None,
    client: Any | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Construct an implemented provider without persisting its credentials."""
    if provider is ProviderName.OPENAI:
        if api_key is None:
            raise ProviderConfigurationError(provider, "An OpenAI API key is required")
        return OpenAIProvider(api_key=api_key, client=client)
    if provider is ProviderName.GEMINI:
        return GeminiProvider(api_key=api_key, client=client)
    if provider is ProviderName.OPENROUTER:
        return OpenRouterProvider(api_key=api_key, client=client, base_url=base_url or OPENROUTER_BASE_URL)
    if provider is ProviderName.GROQ:
        return GroqProvider(api_key=api_key, client=client, base_url=base_url or GROQ_BASE_URL)
    if provider is ProviderName.OLLAMA:
        return OllamaProvider(api_key=api_key, client=client, base_url=base_url or OLLAMA_BASE_URL)
    if provider is ProviderName.OPENAI_COMPATIBLE:
        if not base_url:
            raise ProviderConfigurationError(
                provider,
                "A base URL is required for a custom OpenAI-compatible provider",
            )
        return OpenAICompatibleProvider(provider, api_key, base_url, client=client)
    raise ProviderConfigurationError(provider, f"The {provider.value} adapter is not implemented yet")