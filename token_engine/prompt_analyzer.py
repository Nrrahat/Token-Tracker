"""Estimate textual prompt components independently."""

from collections.abc import Sequence
from typing import Protocol

from token_engine.schemas import (
    ComponentTokenEstimate,
    PromptComponent,
    PromptTokenBreakdown,
)


class TextTokenizer(Protocol):
    model: str
    encoding_name: str

    def count_text(self, text: str) -> int: ...


class PromptAnalyzer:
    """Produce estimates for explicitly supplied prompt components."""

    def __init__(self, tokenizer: TextTokenizer) -> None:
        self._tokenizer = tokenizer

    def analyze(self, components: Sequence[PromptComponent]) -> PromptTokenBreakdown:
        estimates = [
            ComponentTokenEstimate(
                kind=component.kind,
                source_id=component.source_id,
                estimated_tokens=self._tokenizer.count_text(component.text),
            )
            for component in components
        ]
        return PromptTokenBreakdown(
            model=self._tokenizer.model,
            encoding_name=self._tokenizer.encoding_name,
            components=estimates,
            total_estimated_tokens=sum(item.estimated_tokens for item in estimates),
        )