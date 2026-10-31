"""Preserve normalized provider-reported usage without estimating missing fields."""

from core.schemas import ProviderResponse
from token_engine.schemas import UsageExtraction


def extract_provider_usage(response: ProviderResponse) -> UsageExtraction:
    """Extract usage fields already normalized by the provider adapter."""
    usage = response.usage
    fields = (
        usage.input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
        usage.cached_input_tokens,
    )
    has_reported_usage = not usage.is_estimated and any(value is not None for value in fields)
    raw_total = response.raw_usage.get("total_tokens")
    total_tokens = raw_total if isinstance(raw_total, int) and not isinstance(raw_total, bool) else None

    return UsageExtraction(
        provider=response.provider,
        model=response.model,
        usage=usage,
        raw_usage=response.raw_usage,
        reported_total_tokens=total_tokens,
        has_reported_usage=has_reported_usage,
    )