"""Load persisted execution events and LLM-call summaries for trace views."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from database.models import ExecutionEventRecord, ExecutionRunRecord, LLMCallRecord


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    parent_event_id: str | None
    sequence: int
    event_type: str
    status: str | None
    occurred_at: datetime
    duration_ms: int | None
    metadata: dict[str, object]
    error_details: str | None


class LLMCallTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    provider: str
    model: str
    status: str
    requested_at: datetime
    completed_at: datetime
    latency_ms: float
    provider_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    reported_total_tokens: int | None
    usage_is_reported: bool
    estimated_component_tokens: int | None
    attribution_gap_tokens: int | None
    estimated_cost: Decimal | None
    cost_currency: str | None
    error_type: str | None


class ExecutionTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    events: list[TraceEvent]
    llm_calls: list[LLMCallTrace]


class TraceManager:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_trace(self, run_id: str) -> ExecutionTrace | None:
        with self._session_factory() as session:
            if session.get(ExecutionRunRecord, run_id) is None:
                return None
            events = session.scalars(
                select(ExecutionEventRecord)
                .where(ExecutionEventRecord.run_id == run_id)
                .order_by(ExecutionEventRecord.sequence)
            ).all()
            calls = session.scalars(
                select(LLMCallRecord)
                .where(LLMCallRecord.run_id == run_id)
                .order_by(LLMCallRecord.requested_at, LLMCallRecord.id)
            ).all()
            return ExecutionTrace(
                run_id=run_id,
                events=[
                    TraceEvent(
                        id=event.id,
                        parent_event_id=event.parent_event_id,
                        sequence=event.sequence,
                        event_type=event.event_type,
                        status=event.status,
                        occurred_at=event.occurred_at,
                        duration_ms=event.duration_ms,
                        metadata=event.event_data,
                        error_details=event.error_details,
                    )
                    for event in events
                ],
                llm_calls=[
                    LLMCallTrace(
                        id=call.id,
                        provider=call.provider,
                        model=call.model,
                        status=call.status,
                        requested_at=call.requested_at,
                        completed_at=call.completed_at,
                        latency_ms=call.latency_ms,
                        provider_latency_ms=call.provider_latency_ms,
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        reasoning_tokens=call.reasoning_tokens,
                        cached_input_tokens=call.cached_input_tokens,
                        reported_total_tokens=call.reported_total_tokens,
                        usage_is_reported=call.usage_is_reported,
                        estimated_component_tokens=call.estimated_component_tokens,
                        attribution_gap_tokens=call.attribution_gap_tokens,
                        estimated_cost=call.estimated_cost,
                        cost_currency=call.cost_currency,
                        error_type=call.error_type,
                    )
                    for call in calls
                ],
            )