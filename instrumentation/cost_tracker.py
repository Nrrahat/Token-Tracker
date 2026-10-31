"""Estimated request costs using the configured pricing catalog."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from core.enums import ProviderName
from core.schemas import TokenUsage
from instrumentation.pricing import ModelPricing, PricingCatalog


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    amount: Decimal | None = None
    currency: str | None = None
    effective_date: date | None = None
    is_estimated: bool = True


class CostTracker:
    """Calculate costs only when model pricing and both token counts exist."""

    def __init__(self, catalog: PricingCatalog | None = None) -> None:
        self._catalog = catalog or PricingCatalog.default()

    def estimate(self, provider: ProviderName, model: str, usage: TokenUsage) -> CostEstimate:
        pricing = self._catalog.get(provider, model)
        if pricing is None or usage.input_tokens is None or usage.output_tokens is None:
            return CostEstimate(
                currency=pricing.currency if pricing else None,
                effective_date=pricing.effective_date if pricing else None,
            )

        amount = self._calculate(pricing, usage)
        return CostEstimate(
            amount=amount,
            currency=pricing.currency,
            effective_date=pricing.effective_date,
        )

    @staticmethod
    def _calculate(pricing: ModelPricing, usage: TokenUsage) -> Decimal:
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cached_tokens = min(usage.cached_input_tokens or 0, input_tokens)

        if pricing.cached_input_price_per_million is None:
            input_cost = Decimal(input_tokens) * pricing.input_price_per_million
        else:
            uncached_tokens = input_tokens - cached_tokens
            input_cost = (
                Decimal(uncached_tokens) * pricing.input_price_per_million
                + Decimal(cached_tokens) * pricing.cached_input_price_per_million
            )
        output_cost = Decimal(output_tokens) * pricing.output_price_per_million
        return (input_cost + output_cost) / Decimal(1_000_000)