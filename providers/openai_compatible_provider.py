"""OpenAI-compatible adapters for OpenRouter, Ollama, and custom endpoints."""

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from pydantic import SecretStr, ValidationError

from core.enums import ProviderName
from core.schemas import ChatMessage, ProviderRequest, ProviderResponse, TokenUsage, ToolCall, ToolDefinition
from providers.base_provider import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OpenAICompatibleProvider:
    """Normalize a chat completion endpoint implementing the OpenAI API shape."""

    def __init__(
        self,
        provider_name: ProviderName,
        api_key: SecretStr | str | None,
        base_url: str,
        *,
        client: Any | None = None,
        require_api_key: bool = False,
    ) -> None:
        secret = _secret_value(api_key)
        if require_api_key and not secret:
            raise ProviderConfigurationError(provider_name, f"A {provider_name.value} API key is required")
        if not base_url.startswith(("https://", "http://")):
            raise ProviderConfigurationError(provider_name, "Provider base URL must use HTTP or HTTPS")
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self._client = client if client is not None else _create_openai_client(
            provider_name,
            secret or "ollama",
            self.base_url,
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
            "timeout": request.timeout_seconds,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [_serialize_tool(tool) for tool in request.tools]

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as error:
            raise ProviderRequestError(
                self.provider_name,
                f"{self.provider_name.value} request failed ({type(error).__name__})",
            ) from None
        latency_ms = (perf_counter() - started) * 1000

        choices = _get(response, "choices")
        if not choices:
            raise ProviderResponseError(self.provider_name, "Provider returned no completion choices")
        choice = choices[0]
        message = _get(choice, "message")
        if message is None:
            raise ProviderResponseError(self.provider_name, "Provider returned a choice without a message")

        tool_calls = _normalize_tool_calls(_get(message, "tool_calls"), self.provider_name)
        usage, raw_usage = _normalize_usage(_get(response, "usage"), self.provider_name)
        return ProviderResponse(
            provider=self.provider_name,
            model=str(_get(response, "model") or request.model),
            content=_get(message, "content"),
            tool_calls=tool_calls,
            usage=usage,
            raw_usage=raw_usage,
            latency_ms=latency_ms,
            finish_reason=_get(choice, "finish_reason"),
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter adapter using its OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: SecretStr | str | None,
        *,
        client: Any | None = None,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        super().__init__(
            ProviderName.OPENROUTER,
            api_key,
            base_url,
            client=client,
            require_api_key=True,
        )


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama's local OpenAI-compatible API; no API key is required by default."""

    def __init__(
        self,
        api_key: SecretStr | str | None = None,
        *,
        client: Any | None = None,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        super().__init__(
            ProviderName.OLLAMA,
            api_key,
            base_url,
            client=client,
            require_api_key=False,
        )


def _secret_value(api_key: SecretStr | str | None) -> str | None:
    if api_key is None:
        return None
    value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return value.strip() or None


def _create_openai_client(provider: ProviderName, api_key: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ProviderConfigurationError(
            provider,
            "Install the OpenAI SDK with `uv sync --extra providers`",
        ) from error
    return OpenAI(api_key=api_key, base_url=base_url)


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        result["content"] = message.content
    if message.name is not None:
        result["name"] = message.name
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, separators=(",", ":")),
                },
            }
            for tool_call in message.tool_calls
        ]
    return result


def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _normalize_tool_calls(provider_calls: Any, provider: ProviderName) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    for provider_call in provider_calls or []:
        function = _get(provider_call, "function")
        if function is None:
            raise ProviderResponseError(provider, "Provider returned an invalid tool call")
        try:
            arguments = json.loads(_get(function, "arguments") or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            raise ProviderResponseError(provider, "Provider returned malformed tool arguments") from error
        if not isinstance(arguments, dict):
            raise ProviderResponseError(provider, "Tool arguments must be a JSON object")
        try:
            normalized.append(
                ToolCall(
                    id=_get(provider_call, "id"),
                    name=_get(function, "name"),
                    arguments=arguments,
                )
            )
        except ValidationError as error:
            raise ProviderResponseError(provider, "Provider returned an invalid tool call") from error
    return normalized


def _normalize_usage(provider_usage: Any, provider: ProviderName) -> tuple[TokenUsage, dict[str, Any]]:
    if provider_usage is None:
        return TokenUsage(), {}
    prompt_details = _get(provider_usage, "prompt_tokens_details")
    completion_details = _get(provider_usage, "completion_tokens_details")
    usage = TokenUsage(
        input_tokens=_get(provider_usage, "prompt_tokens"),
        output_tokens=_get(provider_usage, "completion_tokens"),
        reasoning_tokens=_get(completion_details, "reasoning_tokens"),
        cached_input_tokens=_get(prompt_details, "cached_tokens"),
    )
    return usage, _raw_usage(provider_usage)


def _raw_usage(provider_usage: Any) -> dict[str, Any]:
    model_dump = getattr(provider_usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    raw: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _get(provider_usage, field)
        if value is not None:
            raw[field] = value
    return raw


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)