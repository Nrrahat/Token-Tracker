"""Compare local component estimates with provider-reported input usage."""

from collections.abc import Sequence

from core.schemas import ProviderResponse
from token_engine.prompt_analyzer import PromptAnalyzer
from token_engine.schemas import PromptComponent, TokenAttributionReport
from token_engine.usage_extractor import extract_provider_usage


class TokenAttributor:
    """Build per-request attribution without treating estimates as actual usage."""

    def __init__(self, prompt_analyzer: PromptAnalyzer) -> None:
        self._prompt_analyzer = prompt_analyzer

    def attribute(
        self,
        components: Sequence[PromptComponent],
        response: ProviderResponse,
    ) -> TokenAttributionReport:
        prompt = self._prompt_analyzer.analyze(components)
        provider_usage = extract_provider_usage(response)
        reported_input = provider_usage.usage.input_tokens
        gap = None
        if provider_usage.has_reported_usage and reported_input is not None:
            gap = reported_input - prompt.total_estimated_tokens

        return TokenAttributionReport(
            prompt=prompt,
            provider_usage=provider_usage,
            attribution_gap_tokens=gap,
        )