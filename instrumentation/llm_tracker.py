"""Instrument provider calls and persist usage, latency, cost, and trace events."""

from datetime import date, datetime, timezone
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from core.enums import EventType, ExecutionStatus
from core.schemas import ProviderRequest, ProviderResponse
from database.models import LLMCallRecord
from instrumentation.cost_tracker import CostTracker
from instrumentation.event_tracker import EventTracker
from instrumentation.latency_tracker import LatencyTracker
from providers.base_provider import LLMProvider
from token_engine.schemas import TokenAttributionReport
from token_engine.prompt_analyzer import PromptAnalyzer
from token_engine.schemas import PromptComponent
from token_engine.token_attribution import TokenAttributor
from token_engine.tokenizer import TiktokenTokenizer, TokenizerConfigurationError
from token_engine.usage_extractor import extract_provider_usage


class LLMTracker:
    """Wrap one provider call and persist a privacy-conscious trace record."""

    def __init__(
        self,
        session_factory: sessionmaker,
        event_tracker: EventTracker | None = None,
        cost_tracker: CostTracker | None = None,
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._event_tracker = event_tracker or EventTracker(session_factory)
        self._cost_tracker = cost_tracker or CostTracker()
        self._latency_tracker = latency_tracker or LatencyTracker()

    def complete(
        self,
        run_id: str,
        provider: LLMProvider,
        request: ProviderRequest,
        attribution: TokenAttributionReport | None = None,
        prompt_components: Sequence[PromptComponent] | None = None,
    ) -> ProviderResponse:
        call_id = str(uuid4())
        requested_at = datetime.now(timezone.utc)
        timer = self._latency_tracker.start()
        start_event = self._event_tracker.record_event(
            run_id,
            EventType.LLM_REQUEST_STARTED,
            status=ExecutionStatus.RUNNING,
            metadata={
                "llm_call_id": call_id,
                "provider": provider.provider_name.value,
                "model": request.model,
                "message_count": len(request.messages),
                "tool_count": len(request.tools or []),
                "temperature": request.temperature,
            },
        )

        try:
            response = provider.complete(request)
        except Exception as error:
            latency_ms = self._latency_tracker.elapsed_ms(timer)
            completed_at = datetime.now(timezone.utc)
            self._persist_call(
                call_id=call_id,
                run_id=run_id,
                provider=provider.provider_name.value,
                model=request.model,
                status=ExecutionStatus.FAILED.value,
                requested_at=requested_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                provider_latency_ms=None,
                request_metadata=_request_metadata(request),
                error_type=type(error).__name__,
            )
            self._event_tracker.record_event(
                run_id,
                EventType.LLM_REQUEST_FAILED,
                status=ExecutionStatus.FAILED,
                metadata={"llm_call_id": call_id, "error_type": type(error).__name__},
                duration_ms=latency_ms,
                parent_event_id=start_event.id,
                error_details=type(error).__name__,
            )
            raise

        latency_ms = self._latency_tracker.elapsed_ms(timer)
        completed_at = datetime.now(timezone.utc)
        if attribution is None and prompt_components is not None:
            try:
                tokenizer = TiktokenTokenizer(response.model)
                attribution = TokenAttributor(PromptAnalyzer(tokenizer)).attribute(
                    prompt_components,
                    response,
                )
            except TokenizerConfigurationError:
                attribution = None
        extracted = extract_provider_usage(response)
        cost = self._cost_tracker.estimate(response.provider, response.model, response.usage)
        self._persist_call(
            call_id=call_id,
            run_id=run_id,
            provider=response.provider.value,
            model=response.model,
            status=ExecutionStatus.SUCCEEDED.value,
            requested_at=requested_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            provider_latency_ms=response.latency_ms,
            request_metadata=_request_metadata(request),
            usage=extracted,
            cost_amount=cost.amount,
            cost_currency=cost.currency,
            pricing_effective_date=cost.effective_date,
            attribution=attribution,
        )
        self._event_tracker.record_event(
            run_id,
            EventType.LLM_RESPONSE_RECEIVED,
            status=ExecutionStatus.SUCCEEDED,
            metadata={
                "llm_call_id": call_id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "cached_input_tokens": response.usage.cached_input_tokens,
                "estimated_cost": str(cost.amount) if cost.amount is not None else None,
                "cost_currency": cost.currency,
            },
            duration_ms=latency_ms,
            parent_event_id=start_event.id,
        )
        return response

    def _persist_call(
        self,
        *,
        call_id: str,
        run_id: str,
        provider: str,
        model: str,
        status: str,
        requested_at: datetime,
        completed_at: datetime,
        latency_ms: float,
        provider_latency_ms: float | None,
        request_metadata: dict[str, Any],
        usage: Any = None,
        cost_amount: Any = None,
        cost_currency: str | None = None,
        pricing_effective_date: date | None = None,
        attribution: TokenAttributionReport | None = None,
        error_type: str | None = None,
    ) -> None:
        usage_data = usage.usage if usage is not None else None
        component_breakdown = None
        if attribution is not None:
            component_breakdown = [
                {
                    "kind": component.kind.value,
                    "source_id": component.source_id,
                    "estimated_tokens": component.estimated_tokens,
                }
                for component in attribution.prompt.components
            ]

        with self._session_factory() as session:
            record = LLMCallRecord(
                id=call_id,
                run_id=run_id,
                provider=provider,
                model=model,
                status=status,
                requested_at=requested_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                provider_latency_ms=provider_latency_ms,
                request_metadata=request_metadata,
                input_tokens=usage_data.input_tokens if usage_data else None,
                output_tokens=usage_data.output_tokens if usage_data else None,
                reasoning_tokens=usage_data.reasoning_tokens if usage_data else None,
                cached_input_tokens=usage_data.cached_input_tokens if usage_data else None,
                reported_total_tokens=usage.reported_total_tokens if usage is not None else None,
                usage_is_reported=usage.has_reported_usage if usage is not None else False,
                raw_usage=usage.raw_usage if usage is not None else {},
                estimated_component_tokens=(
                    attribution.prompt.total_estimated_tokens if attribution is not None else None
                ),
                component_breakdown=component_breakdown,
                attribution_gap_tokens=(
                    attribution.attribution_gap_tokens if attribution is not None else None
                ),
                estimated_cost=cost_amount,
                cost_currency=cost_currency,
                pricing_effective_date=pricing_effective_date,
                error_type=error_type,
            )
            session.add(record)
            session.commit()


def _request_metadata(request: ProviderRequest) -> dict[str, Any]:
    return {
        "message_count": len(request.messages),
        "tool_count": len(request.tools or []),
        "temperature": request.temperature,
    }