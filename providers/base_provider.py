"""Provider-neutral interface and errors."""

from typing import Protocol

from core.enums import ProviderName
from core.schemas import ProviderRequest, ProviderResponse


class LLMProvider(Protocol):
    """Common synchronous interface implemented by provider adapters."""

    provider_name: ProviderName

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request and normalize the provider response."""


class ProviderError(RuntimeError):
    """Base class for safe, normalized provider errors."""

    def __init__(self, provider: ProviderName, message: str) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderConfigurationError(ProviderError):
    """Raised when adapter configuration or dependencies are missing."""


class ProviderRequestError(ProviderError):
    """Raised when a provider rejects or cannot complete a request."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response cannot be normalized safely."""