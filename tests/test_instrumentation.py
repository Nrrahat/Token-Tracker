from datetime import date
from decimal import Decimal

import pytest

from core.enums import EventType, ProviderName
from core.schemas import (
    ChatMessage,
    ProviderRequest,
    ProviderResponse,
    TokenUsage,
)
from database.connection import create_database_engine, create_schema, create_session_factory
from database.models import AgentRecord, ExecutionRunRecord, LLMCallRecord, TaskRecord
from instrumentation.cost_tracker import CostTracker
from instrumentation.event_tracker import EventTracker
from instrumentation.llm_tracker import LLMTracker
from instrumentation.pricing import ModelPricing, PricingCatalog
from instrumentation.trace_manager import TraceManager
from token_engine.prompt_analyzer import PromptAnalyzer
from token_engine.schemas import PromptComponent, PromptComponentKind
from token_engine.token_attribution import TokenAttributor


class CharacterTokenizer:
    model = "test-model"
    encoding_name = "test-encoding"

    def count_text(self, text: str) -> int:
        return len(text)


class FakeProvider:
    provider_name = ProviderName.OPENAI

    def __init__(self, response: ProviderResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.fixture
def execution_db() -> tuple[object, object, str]:
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        agent = AgentRecord(name="Test agent", provider="openai", model="test-model")
        session.add(agent)
        session.flush()
        task = TaskRecord(agent_id=agent.id, prompt="Private task text")
        session.add(task)
        session.flush()
        run = ExecutionRunRecord(task_id=task.id, agent_id=agent.id, status="running")
        session.add(run)
        session.commit()
        run_id = run.id
    yield engine, session_factory, run_id
    engine.dispose()


def make_response() -> ProviderResponse:
    return ProviderResponse(
        provider=ProviderName.OPENAI,
        model="test-model",
        content="Secret response text",
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=50,
            cached_input_tokens=250,
        ),
        raw_usage={"total_tokens": 1500},
        latency_ms=12.5,
    )


def make_cost_tracker() -> CostTracker:
    return CostTracker(
        PricingCatalog(
            [
                ModelPricing(
                    provider=ProviderName.OPENAI,
                    model="test-model",
                    input_price_per_million=Decimal("2"),
                    output_price_per_million=Decimal("10"),
                    cached_input_price_per_million=Decimal("1"),
                    currency="USD",
                    effective_date=date(2026, 1, 1),
                )
            ]
        )
    )


def test_cost_tracker_uses_cached_rate_and_requires_complete_usage() -> None:
    tracker = make_cost_tracker()

    estimate = tracker.estimate(
        ProviderName.OPENAI,
        "test-model",
        TokenUsage(input_tokens=1000, output_tokens=500, cached_input_tokens=250),
    )
    incomplete = tracker.estimate(
        ProviderName.OPENAI,
        "test-model",
        TokenUsage(input_tokens=1000),
    )

    assert estimate.amount == Decimal("0.00675")
    assert estimate.currency == "USD"
    assert estimate.effective_date == date(2026, 1, 1)
    assert estimate.is_estimated is True
    assert incomplete.amount is None


def test_event_tracker_persists_ordered_events_and_validates_parent(execution_db) -> None:
    _, session_factory, run_id = execution_db
    tracker = EventTracker(session_factory)

    first = tracker.record_event(run_id, EventType.AGENT_STARTED)
    second = tracker.record_event(
        run_id,
        EventType.LLM_REQUEST_STARTED,
        parent_event_id=first.id,
        metadata={"message_count": 1},
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.parent_event_id == first.id
    assert second.event_data == {"message_count": 1}
    with pytest.raises(ValueError, match="does not exist"):
        tracker.record_event("missing-run", EventType.AGENT_STARTED)


def test_llm_tracker_persists_success_usage_cost_and_trace(execution_db) -> None:
    _, session_factory, run_id = execution_db
    event_tracker = EventTracker(session_factory)
    tracker = LLMTracker(session_factory, event_tracker, make_cost_tracker())
    response = make_response()
    attribution = TokenAttributor(PromptAnalyzer(CharacterTokenizer())).attribute(
        [PromptComponent(kind=PromptComponentKind.SYSTEM_INSTRUCTION, text="sys")],
        response,
    )
    request = ProviderRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Private request text")],
    )

    result = tracker.complete(run_id, FakeProvider(response=response), request, attribution)
    trace = TraceManager(session_factory).get_trace(run_id)

    assert result is response
    assert trace is not None
    assert [event.sequence for event in trace.events] == [1, 2]
    assert [event.event_type for event in trace.events] == [
        EventType.LLM_REQUEST_STARTED.value,
        EventType.LLM_RESPONSE_RECEIVED.value,
    ]
    assert trace.events[1].parent_event_id == trace.events[0].id
    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.status == "succeeded"
    assert call.input_tokens == 1000
    assert call.output_tokens == 500
    assert call.reasoning_tokens == 50
    assert call.cached_input_tokens == 250
    assert call.reported_total_tokens == 1500
    assert call.usage_is_reported is True
    assert call.estimated_component_tokens == 3
    assert call.attribution_gap_tokens == 997
    assert call.estimated_cost == Decimal("0.00675")
    assert call.provider_latency_ms == 12.5
    assert call.latency_ms >= 0
    assert "Private request text" not in str(trace)
    assert "Secret response text" not in str(trace)


def test_llm_tracker_persists_safe_failure_metadata_and_reraises(execution_db) -> None:
    _, session_factory, run_id = execution_db
    tracker = LLMTracker(session_factory, cost_tracker=make_cost_tracker())
    request = ProviderRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Private request text")],
    )

    with pytest.raises(RuntimeError, match="credential-secret"):
        tracker.complete(
            run_id,
            FakeProvider(error=RuntimeError("credential-secret")),
            request,
        )

    trace = TraceManager(session_factory).get_trace(run_id)
    assert trace is not None
    assert [event.event_type for event in trace.events] == [
        EventType.LLM_REQUEST_STARTED.value,
        EventType.LLM_REQUEST_FAILED.value,
    ]
    assert trace.llm_calls[0].status == "failed"
    assert trace.llm_calls[0].error_type == "RuntimeError"
    assert "credential-secret" not in str(trace)
    assert "Private request text" not in str(trace)


def test_trace_manager_returns_none_for_unknown_run(execution_db) -> None:
    _, session_factory, _ = execution_db

    assert TraceManager(session_factory).get_trace("unknown") is None
