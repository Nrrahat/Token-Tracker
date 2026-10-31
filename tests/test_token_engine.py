from types import SimpleNamespace

from core.enums import ProviderName
from core.schemas import ProviderResponse, TokenUsage
from token_engine.prompt_analyzer import PromptAnalyzer
from token_engine.schemas import PromptComponent, PromptComponentKind
from token_engine.token_attribution import TokenAttributor
from token_engine.tokenizer import TiktokenTokenizer
from token_engine.usage_extractor import extract_provider_usage


class CharacterTokenizer:
    model = "test-model"
    encoding_name = "test-encoding"

    def count_text(self, text: str) -> int:
        return len(text)


def make_response(
    usage: TokenUsage,
    raw_usage: dict[str, object] | None = None,
) -> ProviderResponse:
    return ProviderResponse(
        provider=ProviderName.OPENAI,
        model="test-model",
        content="Done",
        usage=usage,
        raw_usage=raw_usage or {},
        latency_ms=0,
    )


def test_tiktoken_tokenizer_uses_model_encoding_and_safe_text_count() -> None:
    import tiktoken

    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    assert tokenizer.encoding_name == encoding.name
    assert tokenizer.used_fallback is False
    assert tokenizer.count_text("hello world") == len(
        encoding.encode("hello world", disallowed_special=())
    )


def test_unknown_model_uses_and_reports_configured_fallback() -> None:
    tokenizer = TiktokenTokenizer("unknown-model-for-tests")

    assert tokenizer.encoding_name == "cl100k_base"
    assert tokenizer.used_fallback is True
    assert tokenizer.count_text("") == 0


def test_prompt_analyzer_keeps_component_estimates_separate() -> None:
    analyzer = PromptAnalyzer(CharacterTokenizer())
    breakdown = analyzer.analyze(
        [
            PromptComponent(kind=PromptComponentKind.SYSTEM_INSTRUCTION, text="system"),
            PromptComponent(
                kind=PromptComponentKind.SKILL_INSTRUCTION,
                source_id="research-v1",
                text="skill text",
            ),
            PromptComponent(kind=PromptComponentKind.TOOL_RESULT, text=""),
        ]
    )

    assert [item.estimated_tokens for item in breakdown.components] == [6, 10, 0]
    assert breakdown.components[1].source_id == "research-v1"
    assert breakdown.total_estimated_tokens == 16
    assert breakdown.is_estimated is True


def test_usage_extraction_preserves_actual_fields_without_filling_missing_values() -> None:
    response = make_response(
        TokenUsage(
            input_tokens=13,
            output_tokens=5,
            reasoning_tokens=2,
            cached_input_tokens=3,
            is_estimated=False,
        ),
        {"total_tokens": 18, "vendor_detail": {"kept": True}},
    )

    extracted = extract_provider_usage(response)

    assert extracted.has_reported_usage is True
    assert extracted.usage.input_tokens == 13
    assert extracted.usage.output_tokens == 5
    assert extracted.usage.reasoning_tokens == 2
    assert extracted.usage.cached_input_tokens == 3
    assert extracted.reported_total_tokens == 18
    assert extracted.raw_usage["vendor_detail"] == {"kept": True}


def test_usage_extraction_marks_missing_usage_unreported() -> None:
    extracted = extract_provider_usage(make_response(TokenUsage(), {}))

    assert extracted.has_reported_usage is False
    assert extracted.reported_total_tokens is None
    assert extracted.usage.input_tokens is None
    assert extracted.usage.output_tokens is None


def test_attribution_gap_is_signed_and_does_not_replace_reported_usage() -> None:
    attributor = TokenAttributor(PromptAnalyzer(CharacterTokenizer()))
    components = [
        PromptComponent(kind=PromptComponentKind.SYSTEM_INSTRUCTION, text="system"),
        PromptComponent(kind=PromptComponentKind.USER_PROMPT, text="task"),
    ]
    response = make_response(TokenUsage(input_tokens=15, output_tokens=4))

    report = attributor.attribute(components, response)

    assert report.prompt.total_estimated_tokens == 10
    assert report.provider_usage.usage.input_tokens == 15
    assert report.attribution_gap_tokens == 5

    lower_report = attributor.attribute(components, make_response(TokenUsage(input_tokens=7)))
    assert lower_report.attribution_gap_tokens == -3


def test_attribution_gap_is_absent_when_provider_input_usage_is_unavailable() -> None:
    attributor = TokenAttributor(PromptAnalyzer(CharacterTokenizer()))
    response = make_response(TokenUsage(output_tokens=9))

    report = attributor.attribute(
        [PromptComponent(kind=PromptComponentKind.USER_PROMPT, text="task")],
        response,
    )

    assert report.attribution_gap_tokens is None
    assert report.provider_usage.usage.output_tokens == 9