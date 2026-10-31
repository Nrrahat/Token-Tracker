"""Validated model pricing entries and JSON-backed pricing catalog."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.enums import ProviderName


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    model: str = Field(min_length=1)
    input_price_per_million: Decimal = Field(ge=0)
    output_price_per_million: Decimal = Field(ge=0)
    cached_input_price_per_million: Decimal | None = Field(default=None, ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=8)
    effective_date: date | None = None


class PricingCatalog:
    """Lookup pricing by provider/model without embedding rates in code."""

    def __init__(self, entries: list[ModelPricing] | None = None) -> None:
        self._entries: dict[tuple[ProviderName, str], ModelPricing] = {}
        for entry in entries or []:
            key = (entry.provider, entry.model)
            if key in self._entries:
                raise ValueError(f"Duplicate pricing entry for {entry.provider.value}/{entry.model}")
            self._entries[key] = entry

    @classmethod
    def from_json(cls, path: str | Path) -> "PricingCatalog":
        with Path(path).open("r", encoding="utf-8") as pricing_file:
            data: Any = json.load(pricing_file)
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Pricing file must contain a 'models' list")
        return cls([ModelPricing.model_validate(entry) for entry in entries])

    @classmethod
    def default(cls) -> "PricingCatalog":
        pricing_path = Path(__file__).resolve().parents[1] / "config" / "model_pricing.json"
        return cls.from_json(pricing_path)

    def get(self, provider: ProviderName, model: str) -> ModelPricing | None:
        return self._entries.get((provider, model))