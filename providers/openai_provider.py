"""OpenAI Chat Completions adapter with normalized usage and tool calls."""

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from pydantic import SecretStr, ValidationError

from core.enums import ProviderName
from core.schemas import (
    ChatMessage,
    ProviderRequest,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from providers.base_provider import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)


class OpenAIProvider:
    """Synchronous OpenAI adapter; credentials are supplied at construction."""

    provider_name = ProviderName.OPENAI

    def __init__(self, api_key: SecretStr | str, client: Any | None = None) -> None:
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret.strip():
            raise ProviderConfigurationError(self.provider_name, "An OpenAI API key is required")
        self._client = client if client is not None else self._create_client(secret)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Call OpenAI and convert its response to the internal provider schema."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
            "timeout": request.timeout_seconds,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [_serialize_tool(tool) for tool in request.tools]

        started_at = perf_counter()
        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as error:
            raise ProviderRequestError(
                self.provider_name,
                f"OpenAI request failed ({type(error).__name__})",
            ) from None
        latency_ms = (perf_counter() - started_at) * 1000

        choices = _get(response, "choices")
        if not choices:
            raise ProviderResponseError(self.provider_name, "OpenAI returned no completion choices")

        choice = choices[0]
        message = _get(choice, "message")
        if message is None:
            raise ProviderResponseError(self.provider_name, "OpenAI returned a choice without a message")

        tool_calls = _normalize_tool_calls(_get(message, "tool_calls"))
        usage_data = _get(response, "usage")
        usage, raw_usage = _normalize_usage(usage_data)
        returned_model = _get(response, "model") or request.model

        return ProviderResponse(
            provider=self.provider_name,
            model=str(returned_model),
            content=_get(message, "content"),
            tool_calls=tool_calls,
            usage=usage,
            raw_usage=raw_usage,
            latency_ms=latency_ms,
            finish_reason=_get(choice, "finish_reason"),
        )

    @staticmethod
    def _create_client(api_key: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ProviderConfigurationError(
                ProviderName.OPENAI,
                "Install the provider dependency with `uv sync --extra providers`",
            ) from error
        return OpenAI(api_key=api_key)


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


def _normalize_tool_calls(provider_tool_calls: Any) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    for provider_tool_call in provider_tool_calls or []:
        function = _get(provider_tool_call, "function")
        if function is None:
            raise ProviderResponseError(ProviderName.OPENAI, "OpenAI returned an invalid tool call")
        try:
            arguments = json.loads(_get(function, "arguments") or "{}")
        except (json.JSONDecodeError, TypeError) as error:
            raise ProviderResponseError(
                ProviderName.OPENAI,
                "OpenAI returned malformed tool arguments",
            ) from error
        if not isinstance(arguments, dict):
            raise ProviderResponseError(
                ProviderName.OPENAI,
                "OpenAI tool arguments must be a JSON object",
            )
        try:
            normalized.append(
                ToolCall(
                    id=_get(provider_tool_call, "id"),
                    name=_get(function, "name"),
                    arguments=arguments,
                )
            )
        except ValidationError as error:
            raise ProviderResponseError(ProviderName.OPENAI, "OpenAI returned an invalid tool call") from error
    return normalized


def _normalize_usage(provider_usage: Any) -> tuple[TokenUsage, dict[str, Any]]:
    if provider_usage is None:
        return TokenUsage(), {}

    prompt_details = _get(provider_usage, "prompt_tokens_details")
    completion_details = _get(provider_usage, "completion_tokens_details")
    usage = TokenUsage(
        input_tokens=_get(provider_usage, "prompt_tokens"),
        output_tokens=_get(provider_usage, "completion_tokens"),
        reasoning_tokens=_get(completion_details, "reasoning_tokens"),
        cached_input_tokens=_get(prompt_details, "cached_tokens"),
        is_estimated=False,
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
    for field in ("prompt_tokens_details", "completion_tokens_details"):
        details = _get(provider_usage, field)
        if details is not None:
            detail_values = {
                name: _get(details, name)
                for name in ("cached_tokens", "reasoning_tokens")
                if _get(details, name) is not None
            }
            raw[field] = detail_values
    return raw


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)