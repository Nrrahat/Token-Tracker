from datetime import datetime, timezone
from decimal import Decimal

import pytest

from analytics.service import AnalyticsService
from core.enums import EventType
from database.connection import create_database_engine, create_schema, create_session_factory
from database.models import (
    AgentRecord,
    EvaluationRecord,
    ExecutionEventRecord,
    ExecutionRunRecord,
    LLMCallRecord,
    TaskRecord,
)
from evaluation.schemas import (
    EvaluatorType,
    HumanEvaluationSubmission,
)
from evaluation.service import EvaluationManager


@pytest.fixture
def analytics_database():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        agent = AgentRecord(name="Analyst", provider="openai", model="test-model")
        session.add(agent)
        session.flush()
        task = TaskRecord(agent_id=agent.id, prompt="Compute a value")
        session.add(task)
        session.flush()
        run = ExecutionRunRecord(
            task_id=task.id,
            agent_id=agent.id,
            status="succeeded",
            started_at=now,
            finished_at=now,
            result="42",
        )
        session.add(run)
        session.flush()
        session.add_all(
            [
                LLMCallRecord(
                    run_id=run.id,
                    provider="openai",
                    model="test-model",
                    status="succeeded",
                    requested_at=now,
                    completed_at=now,
                    latency_ms=20,
                    request_metadata={},
                    input_tokens=100,
                    output_tokens=20,
                    reasoning_tokens=4,
                    cached_input_tokens=10,
                    reported_total_tokens=120,
                    usage_is_reported=True,
                    raw_usage={"total_tokens": 120},
                    estimated_component_tokens=18,
                    component_breakdown=[
                        {"kind": "system_instruction", "source_id": None, "estimated_tokens": 13},
                        {"kind": "tool_result", "source_id": "calculator", "estimated_tokens": 5},
                    ],
                    attribution_gap_tokens=82,
                    estimated_cost=Decimal("0.0004"),
                    cost_currency="USD",
                ),
                LLMCallRecord(
                    run_id=run.id,
                    provider="openai",
                    model="test-model",
                    status="succeeded",
                    requested_at=now,
                    completed_at=now,
                    latency_ms=10,
                    request_metadata={},
                    usage_is_reported=False,
                    raw_usage={},
                    estimated_component_tokens=7,
                    component_breakdown=[
                        {"kind": "tool_result", "source_id": "calculator", "estimated_tokens": 7}
                    ],
                ),
            ]
        )
        session.add_all(
            [
                ExecutionEventRecord(
                    run_id=run.id,
                    sequence=1,
                    event_type=EventType.TOOL_COMPLETED.value,
                    status="succeeded",
                    occurred_at=now,
                    duration_ms=8,
                    event_data={
                        "tool_name": "calculator",
                        "argument_fingerprint": "same-arguments",
                        "duration_ms": 8,
                        "output_size_bytes": 14,
                        "estimated_output_tokens": 5,
                        "is_empty": False,
                    },
                ),
                ExecutionEventRecord(
                    run_id=run.id,
                    sequence=2,
                    event_type=EventType.TOOL_FAILED.value,
                    status="failed",
                    occurred_at=now,
                    duration_ms=2,
                    event_data={
                        "tool_name": "calculator",
                        "argument_fingerprint": "same-arguments",
                        "duration_ms": 2,
                        "output_size_bytes": 0,
                        "estimated_output_tokens": 0,
                        "is_empty": False,
                    },
                ),
            ]
        )
        session.commit()
        run_id = run.id
        agent_id = agent.id
    yield session_factory, run_id, agent_id
    engine.dispose()


def test_token_analytics_separates_reported_usage_from_estimates(analytics_database) -> None:
    session_factory, _, _ = analytics_database
    result = AnalyticsService(session_factory).token_analytics()

    assert result.totals.llm_calls == 2
    assert result.totals.reported_calls == 1
    assert result.totals.reported_input_tokens == 100
    assert result.totals.reported_output_tokens == 20
    assert result.totals.reported_reasoning_tokens == 4
    assert result.totals.cached_input_tokens == 10
    assert result.totals.estimated_component_tokens == 25
    assert result.totals.attribution_gap_tokens == 82
    assert result.totals.estimated_costs == {"USD": Decimal("0.0004")}
    assert result.by_model[0].input_tokens == 100
    assert result.by_agent[0].key == "Analyst"
    assert len(result.by_day) == 1
    assert {(item.component, item.source_id, item.estimated_tokens) for item in result.by_component} == {
        ("system_instruction", None, 13),
        ("tool_result", "calculator", 12),
    }


def test_tool_analytics_uses_persisted_fingerprints_and_included_token_estimates(
    analytics_database,
) -> None:
    session_factory, _, _ = analytics_database
    rows = AnalyticsService(session_factory).tool_analytics()

    assert len(rows) == 1
    calculator = rows[0]
    assert calculator.tool_name == "calculator"
    assert calculator.calls == 2
    assert calculator.failures == 1
    assert calculator.duplicate_calls == 1
    assert calculator.total_duration_ms == 10
    assert calculator.total_output_size_bytes == 14
    assert calculator.estimated_output_tokens == 5
    assert calculator.included_result_tokens == 12


def test_agent_analytics_calculates_execution_and_usage_averages(analytics_database) -> None:
    session_factory, _, agent_id = analytics_database
    result = AnalyticsService(session_factory).agent_analytics()

    assert len(result.agents) == 1
    agent = result.agents[0]
    assert agent.agent_id == agent_id
    assert agent.executions == 1
    assert agent.successful_executions == 1
    assert agent.success_rate == 1.0
    assert agent.llm_calls == 2
    assert agent.average_llm_calls_per_execution == 2.0
    assert agent.reported_input_tokens == 100
    assert agent.reported_output_tokens == 20
    assert agent.tool_calls == 2
    assert agent.average_tool_calls_per_execution == 2.0
    assert agent.estimated_costs == {"USD": Decimal("0.0004")}


def test_automatic_and_human_evaluations_remain_distinct(analytics_database) -> None:
    session_factory, run_id, _ = analytics_database
    manager = EvaluationManager(session_factory)

    automatic = manager.evaluate_execution(run_id)
    repeated_automatic = manager.evaluate_execution(run_id)
    human = manager.submit_human_evaluation(
        HumanEvaluationSubmission(
            run_id=run_id,
            evaluator_name="reviewer-1",
            response_relevance=4,
            response_completeness=5,
            output_quality=4,
            notes="Correct and concise.",
        )
    )
    summary = manager.summary()

    assert automatic.evaluator_type is EvaluatorType.AUTOMATIC
    assert repeated_automatic.id == automatic.id
    assert automatic.task_success is True
    assert automatic.tool_use_success is False
    assert automatic.output_quality is None
    assert human.evaluator_type is EvaluatorType.HUMAN
    assert human.task_success is True
    assert human.output_quality == 4
    assert summary.automatic_evaluations == 1
    assert summary.automatic_success_rate == 1.0
    assert summary.human_evaluations == 1
    assert summary.average_relevance == 4.0
    assert summary.average_completeness == 5.0
    assert summary.average_quality == 4.0


def test_human_evaluation_requires_a_quality_score(analytics_database) -> None:
    _, run_id, _ = analytics_database

    with pytest.raises(ValueError, match="at least one quality score"):
        HumanEvaluationSubmission(run_id=run_id, evaluator_name="reviewer")