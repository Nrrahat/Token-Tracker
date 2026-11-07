from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from core.enums import ProviderName
from core.schemas import ChatMessage, ProviderRequest, ToolDefinition
from providers.base_provider import ProviderConfigurationError, ProviderRequestError
from providers.gemini_provider import GeminiProvider
from providers.openai_compatible_provider import GroqProvider, OllamaProvider, OpenRouterProvider
from providers.provider_factory import create_provider
from tools.available.calculator import CalculatorInput


def openai_compatible_client(response=None):
    create = Mock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def test_openrouter_normalizes_chat_usage_tools_and_passes_timeout() -> None:
    response = SimpleNamespace(
        model="provider/model-x",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="lookup", arguments='{"query":"x"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=7, total_tokens=37),
    )
    client = openai_compatible_client(response)
    provider = OpenRouterProvider(SecretStr("router-secret"), client=client)
    request = ProviderRequest(
        model="provider/model-x",
        messages=[ChatMessage(role="user", content="Look up x")],
        tools=[ToolDefinition(name="lookup", parameters={"type": "object"})],
        timeout_seconds=22,
    )

    result = provider.complete(request)

    assert result.provider is ProviderName.OPENROUTER
    assert result.tool_calls[0].arguments == {"query": "x"}
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 7
    assert result.raw_usage["total_tokens"] == 37
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["timeout"] == 22
    assert sent["tools"][0]["function"]["name"] == "lookup"


def test_openrouter_requires_key_and_ollama_defaults_to_local_endpoint() -> None:
    with pytest.raises(ProviderConfigurationError, match="API key is required"):
        OpenRouterProvider(None, client=openai_compatible_client())

    ollama_client = openai_compatible_client(
        SimpleNamespace(
            model="qwen-local",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Local answer", tool_calls=None), finish_reason="stop")],
            usage=None,
        )
    )
    ollama = OllamaProvider(client=ollama_client)
    response = ollama.complete(
        ProviderRequest(model="qwen-local", messages=[ChatMessage(role="user", content="Hi")])
    )

    assert ollama.base_url == "http://localhost:11434/v1"
    assert response.provider is ProviderName.OLLAMA
    assert response.content == "Local answer"
    assert response.usage.input_tokens is None
    assert response.usage.is_estimated is False


def test_groq_uses_openai_compatible_endpoint_and_requires_key() -> None:
    response = SimpleNamespace(
        model="llama-3.3-70b-versatile",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Groq response", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4, total_tokens=13),
    )
    client = openai_compatible_client(response)

    provider = create_provider(
        ProviderName.GROQ,
        SecretStr("groq-key"),
        client=client,
    )
    result = provider.complete(
        ProviderRequest(
            model="llama-3.3-70b-versatile",
            messages=[ChatMessage(role="user", content="Hello")],
        )
    )

    assert isinstance(provider, GroqProvider)
    assert provider.base_url == "https://api.groq.com/openai/v1"
    assert result.provider is ProviderName.GROQ
    assert result.content == "Groq response"
    assert result.usage.input_tokens == 9
    with pytest.raises(ProviderConfigurationError, match="API key is required"):
        create_provider(ProviderName.GROQ, client=openai_compatible_client())


def test_factory_constructs_each_supported_provider_and_custom_endpoint() -> None:
    router = create_provider(
        ProviderName.OPENROUTER,
        SecretStr("router-key"),
        client=openai_compatible_client(),
    )
    ollama = create_provider(ProviderName.OLLAMA, client=openai_compatible_client())
    custom = create_provider(
        ProviderName.OPENAI_COMPATIBLE,
        SecretStr("custom-key"),
        base_url="http://127.0.0.1:8080/v1",
        client=openai_compatible_client(),
    )
    gemini = create_provider(
        ProviderName.GEMINI,
        SecretStr("gemini-key"),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=Mock())),
    )

    assert router.provider_name is ProviderName.OPENROUTER
    assert ollama.provider_name is ProviderName.OLLAMA
    assert custom.provider_name is ProviderName.OPENAI_COMPATIBLE
    assert gemini.provider_name is ProviderName.GEMINI
    with pytest.raises(ProviderConfigurationError, match="base URL is required"):
        create_provider(ProviderName.OPENAI_COMPATIBLE, "key", client=openai_compatible_client())


def test_gemini_maps_messages_tools_usage_and_response() -> None:
    response = SimpleNamespace(
        model_version="gemini-test",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="Result ", function_call=None),
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(name="lookup", args={"query": "x"}),
                        ),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=40,
            candidates_token_count=12,
            thoughts_token_count=3,
            cached_content_token_count=5,
            total_token_count=52,
        ),
    )
    generate_content = Mock(return_value=response)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider(SecretStr("gemini-key"), client=client)
    request = ProviderRequest(
        model="gemini-test",
        messages=[
            ChatMessage(role="system", content="Be accurate."),
            ChatMessage(role="user", content="Look up x"),
        ],
        temperature=0.1,
        tools=[ToolDefinition(name="lookup", description="Lookup", parameters={"type": "object"})],
        timeout_seconds=35,
    )

    result = provider.complete(request)

    assert result.provider is ProviderName.GEMINI
    assert result.model == "gemini-test"
    assert result.content == "Result "
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].arguments == {"query": "x"}
    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 12
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cached_input_tokens == 5
    assert result.raw_usage["total_token_count"] == 52
    sent = generate_content.call_args.kwargs
    assert sent["config"]["system_instruction"] == "Be accurate."
    assert sent["config"]["http_options"]["timeout"] == 35_000
    assert sent["config"]["tools"][0]["function_declarations"][0]["name"] == "lookup"


def test_gemini_normalizes_pydantic_tool_schema() -> None:
    generate_content = Mock(return_value=SimpleNamespace(candidates=[], usage_metadata=None))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider("gemini-key", client=client)

    provider.complete(
        ProviderRequest(
            model="gemini-test",
            messages=[ChatMessage(role="user", content="Calculate 1 + 1")],
            tools=[
                ToolDefinition(
                    name="calculator",
                    parameters=CalculatorInput.model_json_schema(),
                )
            ],
        )
    )

    schema = generate_content.call_args.kwargs["config"]["tools"][0]["function_declarations"][0]["parameters"]
    assert schema == {
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "type": "object",
    }


def test_gemini_serializes_tool_results_and_sanitizes_errors() -> None:
    generate_content = Mock(side_effect=RuntimeError("gemini-secret leaked"))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider("gemini-secret", client=client)
    request = ProviderRequest(
        model="gemini-test",
        messages=[
            ChatMessage(role="assistant", content="", tool_calls=[]),
            ChatMessage(role="tool", name="lookup", tool_call_id="call-1", content='{"value":2}'),
        ],
    )

    with pytest.raises(ProviderRequestError) as error:
        provider.complete(request)

    assert "gemini-secret" not in str(error.value)
    contents = generate_content.call_args.kwargs["contents"]
    assert contents == [
        {"role": "user", "parts": [{"function_response": {"name": "lookup", "response": {"value": 2}}}]}
    ]


def test_gemini_requires_key() -> None:
    with pytest.raises(ProviderConfigurationError, match="Gemini API key is required"):
        GeminiProvider(None, client=SimpleNamespace())