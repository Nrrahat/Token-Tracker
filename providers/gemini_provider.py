"""Google Gemini adapter using the official google-genai SDK."""

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import SecretStr

from core.enums import ProviderName
from core.schemas import ChatMessage, ProviderRequest, ProviderResponse, TokenUsage, ToolCall, ToolDefinition
from providers.base_provider import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)


class GeminiProvider:
    provider_name = ProviderName.GEMINI

    def __init__(self, api_key: SecretStr | str | None, client: Any | None = None) -> None:
        secret = _secret_value(api_key)
        if not secret:
            raise ProviderConfigurationError(self.provider_name, "A Gemini API key is required")
        self._client = client if client is not None else self._create_client(secret)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        system_instruction, contents = _serialize_messages(request.messages)
        config: dict[str, Any] = {"http_options": {"timeout": int(request.timeout_seconds * 1000)}}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.tools:
            config["tools"] = [_serialize_tools(request.tools)]

        started = perf_counter()
        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise ProviderRequestError(
                self.provider_name,
                f"Gemini request failed ({type(error).__name__})",
            ) from None
        latency_ms = (perf_counter() - started) * 1000

        candidates = _get(response, "candidates") or []
        candidate = candidates[0] if candidates else None
        candidate_content = _get(candidate, "content") if candidate is not None else None
        parts = _get(candidate_content, "parts") or []
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for index, part in enumerate(parts):
            text = _get(part, "text")
            if isinstance(text, str):
                text_chunks.append(text)
            function_call = _get(part, "function_call")
            if function_call is not None:
                name = _get(function_call, "name")
                arguments = _get(function_call, "args") or {}
                if not isinstance(arguments, Mapping):
                    raise ProviderResponseError(self.provider_name, "Gemini tool arguments must be an object")
                tool_calls.append(
                    ToolCall(
                        id=f"gemini-{index}-{uuid4().hex[:8]}",
                        name=name,
                        arguments=dict(arguments),
                    )
                )

        usage_metadata = _get(response, "usage_metadata")
        usage, raw_usage = _normalize_gemini_usage(usage_metadata)
        model_version = _get(response, "model_version") or request.model
        finish_reason = _get(candidate, "finish_reason") if candidate is not None else None
        return ProviderResponse(
            provider=self.provider_name,
            model=str(model_version),
            content="".join(text_chunks) or None,
            tool_calls=tool_calls,
            usage=usage,
            raw_usage=raw_usage,
            latency_ms=latency_ms,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )

    @staticmethod
    def _create_client(api_key: str) -> Any:
        try:
            from google import genai
        except ImportError as error:
            raise ProviderConfigurationError(
                ProviderName.GEMINI,
                "Install Gemini support with `uv sync --extra providers`",
            ) from error
        return genai.Client(api_key=api_key)


def _secret_value(api_key: SecretStr | str | None) -> str | None:
    if api_key is None:
        return None
    value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return value.strip() or None


def _serialize_messages(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "developer"}:
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role == "tool":
            response: Any = message.content or ""
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except (ValueError, TypeError):
                    response = {"result": response}
            if not isinstance(response, dict):
                response = {"result": response}
            contents.append(
                {
                    "role": "user",
                    "parts": [{"function_response": {"name": message.name or "tool", "response": response}}],
                }
            )
            continue

        role = "model" if message.role == "assistant" else "user"
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        if message.tool_calls:
            parts.extend(
                {
                    "function_call": {
                        "name": tool_call.name,
                        "args": tool_call.arguments,
                    }
                }
                for tool_call in message.tool_calls
            )
        if parts:
            contents.append({"role": role, "parts": parts})
    return "\n\n".join(system_parts), contents


def _serialize_tools(tools: list[ToolDefinition]) -> dict[str, Any]:
    return {
        "function_declarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
    }


def _normalize_gemini_usage(metadata: Any) -> tuple[TokenUsage, dict[str, Any]]:
    if metadata is None:
        return TokenUsage(), {}
    raw = _model_dump(metadata)
    usage = TokenUsage(
        input_tokens=_get(metadata, "prompt_token_count"),
        output_tokens=_get(metadata, "candidates_token_count"),
        reasoning_tokens=_get(metadata, "thoughts_token_count"),
        cached_input_tokens=_get(metadata, "cached_content_token_count"),
    )
    if not raw:
        raw = {
            key: value
            for key in (
                "prompt_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "cached_content_token_count",
                "total_token_count",
            )
            if (value := _get(metadata, key)) is not None
        }
    return usage, raw


def _model_dump(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        data = dump(mode="json", exclude_none=True)
        if isinstance(data, dict):
            return data
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)