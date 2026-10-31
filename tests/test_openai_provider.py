import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from core.enums import ProviderName
from core.schemas import ChatMessage, ProviderRequest, ToolDefinition
from providers.base_provider import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)
from providers.openai_provider import OpenAIProvider
from providers.provider_factory import create_provider


def make_client(response: object | None = None) -> SimpleNamespace:
    create = Mock(return_value=response)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def test_complete_normalizes_response_usage_and_latency() -> None:
    usage = SimpleNamespace(
        prompt_tokens=21,
        completion_tokens=8,
        total_tokens=29,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    response = SimpleNamespace(
        model="gpt-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
    client = make_client(response)
    provider = OpenAIProvider(SecretStr("test-secret"), client=client)

    result = provider.complete(
        ProviderRequest(model="gpt-test", messages=[ChatMessage(role="user", content="Run task")])
    )

    assert result.provider is ProviderName.OPENAI
    assert result.model == "gpt-test"
    assert result.content == "Done"
    assert result.usage.input_tokens == 21
    assert result.usage.output_tokens == 8
    assert result.usage.reasoning_tokens == 2
    assert result.usage.cached_input_tokens == 4
    assert result.usage.is_estimated is False
    assert result.raw_usage["total_tokens"] == 29
    assert result.latency_ms >= 0
    client.chat.completions.create.assert_called_once()


def test_complete_serializes_tools_and_normalizes_tool_calls() -> None:
    response = SimpleNamespace(
        model="gpt-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="lookup", arguments='{"item":"x"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )
    client = make_client(response)
    provider = OpenAIProvider("test-secret", client=client)
    request = ProviderRequest(
        model="gpt-test",
        messages=[ChatMessage(role="user", content="Look this up")],
        tools=[
            ToolDefinition(
                name="lookup",
                description="Look up a value",
                parameters={"type": "object", "properties": {"item": {"type": "string"}}},
            )
        ],
    )

    result = provider.complete(request)

    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].arguments == {"item": "x"}
    sent_tools = client.chat.completions.create.call_args.kwargs["tools"]
    assert sent_tools[0]["function"]["name"] == "lookup"
    assert sent_tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"item": {"type": "string"}},
    }


def test_complete_rejects_malformed_tool_arguments() -> None:
    response = SimpleNamespace(
        model="gpt-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="lookup", arguments="not-json"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )
    provider = OpenAIProvider("test-secret", client=make_client(response))

    with pytest.raises(ProviderResponseError, match="malformed tool arguments"):
        provider.complete(
            ProviderRequest(model="gpt-test", messages=[ChatMessage(role="user", content="Go")])
        )


def test_request_error_does_not_expose_provider_error_text() -> None:
    client = make_client()
    client.chat.completions.create.side_effect = RuntimeError("test-secret leaked")
    provider = OpenAIProvider("test-secret", client=client)

    with pytest.raises(ProviderRequestError) as error:
        provider.complete(
            ProviderRequest(model="gpt-test", messages=[ChatMessage(role="user", content="Go")])
        )

    assert "test-secret" not in str(error.value)
    assert "RuntimeError" in str(error.value)


def test_credentials_are_required_and_factory_rejects_unimplemented_provider() -> None:
    with pytest.raises(ProviderConfigurationError, match="API key is required"):
        OpenAIProvider(SecretStr("   "), client=make_client())

    with pytest.raises(ProviderConfigurationError, match="adapter is not implemented"):
        create_provider(ProviderName.ANTHROPIC, SecretStr("test-secret"))