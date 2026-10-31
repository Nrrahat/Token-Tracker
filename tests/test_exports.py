import csv
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from io import StringIO

from analytics.schemas import (
    ComponentTokenGroup,
    TokenAnalytics,
    TokenDay,
    TokenGroup,
    TokenTotals,
    ToolAnalyticsRow,
)
from core.enums import ProviderName
from database.connection import create_database_engine, create_schema, create_session_factory
from database.models import AgentRecord, ExecutionRunRecord, LLMCallRecord, TaskRecord
from evaluation.schemas import EvaluationSummary
from experiments.schemas import (
    AgentConfigSnapshot,
    ExperimentArm,
    ExperimentArmMetrics,
    ExperimentCase,
    ExperimentComparison,
    ExperimentDataset,
    ExperimentDefinition,
    ExperimentResult,
    ExperimentStatus,
)
from exports.csv_exporter import (
    experiment_comparison_csv,
    token_usage_csv,
    tool_analytics_csv,
)
from exports.json_exporter import experiment_comparison_json, execution_trace_json
from exports.report_generator import experiment_report_markdown, summary_report_markdown
from instrumentation.trace_manager import ExecutionTrace, TraceEvent


def test_token_usage_csv_contains_per_call_metrics_but_not_task_text() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    sessions = create_session_factory(engine)
    now = datetime.now(timezone.utc)
    with sessions() as session:
        agent = AgentRecord(name="CSV agent", provider="openai", model="test-model")
        session.add(agent)
        session.flush()
        task = TaskRecord(agent_id=agent.id, prompt="PRIVATE TASK PROMPT")
        session.add(task)
        session.flush()
        run = ExecutionRunRecord(task_id=task.id, agent_id=agent.id, status="succeeded")
        session.add(run)
        session.flush()
        session.add(
            LLMCallRecord(
                run_id=run.id,
                provider="openai",
                model="test-model",
                status="succeeded",
                requested_at=now,
                completed_at=now,
                latency_ms=5,
                request_metadata={},
                input_tokens=10,
                output_tokens=3,
                reasoning_tokens=1,
                cached_input_tokens=2,
                reported_total_tokens=13,
                usage_is_reported=True,
                raw_usage={"api_key": "never-export"},
                estimated_component_tokens=8,
                attribution_gap_tokens=2,
                estimated_cost=Decimal("0.0001"),
                cost_currency="USD",
            )
        )
        session.commit()

    csv_text = token_usage_csv(sessions)
    rows = list(csv.DictReader(StringIO(csv_text)))

    assert len(rows) == 1
    assert rows[0]["input_tokens"] == "10"
    assert rows[0]["usage_is_reported"] == "True"
    assert rows[0]["cost_currency"] == "USD"
    assert "PRIVATE TASK PROMPT" not in csv_text
    assert "never-export" not in csv_text
    engine.dispose()


def make_comparison() -> ExperimentComparison:
    created = datetime(2026, 10, 30, tzinfo=timezone.utc)
    baseline_snapshot = AgentConfigSnapshot(
        agent_id="agent-base",
        name="Baseline",
        config={"model": "model-a"},
        skill_versions={},
    )
    optimized_snapshot = AgentConfigSnapshot(
        agent_id="agent-opt",
        name="Optimized",
        config={"model": "model-b"},
        skill_versions={},
    )
    experiment = ExperimentDefinition(
        experiment_id="experiment-1",
        name="A/B test",
        description="",
        status=ExperimentStatus.COMPLETED,
        baseline_snapshot=baseline_snapshot,
        optimized_snapshot=optimized_snapshot,
        dataset=ExperimentDataset(
            name="Dataset",
            cases=[ExperimentCase(case_id="case-1", name="Case 1", prompt="Task")],
        ),
        created_at=created,
        started_at=created,
        finished_at=created,
        error_type=None,
    )
    baseline = ExperimentArmMetrics(
        runs=1,
        successful_runs=1,
        task_success_rate=1.0,
        reported_input_tokens=100,
        reported_output_tokens=50,
        reported_total_tokens=150,
        reported_reasoning_tokens=2,
        cached_input_tokens=10,
        llm_calls=1,
        average_llm_calls_per_run=1.0,
        tool_calls=0,
        average_tool_calls_per_run=0.0,
        average_latency_ms=25,
        estimated_costs={"USD": Decimal("0.001")},
        human_evaluations=1,
        average_human_relevance=4,
        average_human_completeness=4,
        average_human_quality=4,
    )
    optimized = baseline.model_copy(
        update={
            "reported_input_tokens": 60,
            "reported_output_tokens": 40,
            "reported_total_tokens": 100,
            "estimated_costs": {"USD": Decimal("0.0008")},
            "average_human_quality": 3,
        }
    )
    return ExperimentComparison(
        experiment=experiment,
        baseline=baseline,
        optimized=optimized,
        token_reduction_percent=100 / 3,
        success_rate_change=0,
        human_quality_change=-1,
        cost_reduction_percent_by_currency={"USD": 20},
        results=[
            ExperimentResult(
                case_id="case-1",
                arm=ExperimentArm.BASELINE,
                run_id="run-base",
                status="succeeded",
            ),
            ExperimentResult(
                case_id="case-1",
                arm=ExperimentArm.OPTIMIZED,
                run_id="run-opt",
                status="succeeded",
            ),
        ],
    )


def test_tool_csv_and_experiment_csv_include_aggregate_and_trial_rows() -> None:
    tool_csv = tool_analytics_csv(
        [
            ToolAnalyticsRow(
                tool_name="calculator",
                calls=2,
                failures=1,
                duplicate_calls=1,
                empty_results=0,
                total_duration_ms=12,
                total_output_size_bytes=28,
                estimated_output_tokens=9,
                included_result_tokens=7,
            )
        ]
    )
    tool_rows = list(csv.DictReader(StringIO(tool_csv)))
    comparison_csv = experiment_comparison_csv(make_comparison())
    comparison_rows = list(csv.DictReader(StringIO(comparison_csv)))

    assert tool_rows[0]["duplicate_calls"] == "1"
    assert tool_rows[0]["included_result_tokens"] == "7"
    assert [row["record_type"] for row in comparison_rows] == ["arm_summary", "arm_summary", "trial", "trial"]
    assert comparison_rows[0]["reported_total_tokens"] == "150"
    assert comparison_rows[2]["case_id"] == "case-1"


def test_json_and_markdown_exports_preserve_caution_and_quality_context() -> None:
    comparison = make_comparison()
    trace = ExecutionTrace(
        run_id="run-1",
        events=[
            TraceEvent(
                id="event-1",
                parent_event_id=None,
                sequence=1,
                event_type="agent_finished",
                status="succeeded",
                occurred_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
                duration_ms=10,
                metadata={"iterations": 1},
                error_details=None,
            )
        ],
        llm_calls=[],
    )
    token_analytics = TokenAnalytics(
        totals=TokenTotals(
            llm_calls=0,
            reported_calls=0,
            reported_input_tokens=0,
            reported_output_tokens=0,
            reported_reasoning_tokens=0,
            cached_input_tokens=0,
            estimated_component_tokens=0,
            attribution_gap_tokens=0,
            estimated_costs={},
        ),
        by_model=[],
        by_provider=[],
        by_agent=[],
        by_day=[TokenDay(day=date(2026, 10, 30), input_tokens=0, output_tokens=0)],
        by_component=[ComponentTokenGroup(component="system_instruction", source_id=None, estimated_tokens=0)],
    )

    trace_payload = json.loads(execution_trace_json(trace))
    comparison_payload = json.loads(experiment_comparison_json(comparison))
    report = summary_report_markdown(
        token_analytics,
        EvaluationSummary(
            automatic_evaluations=0,
            automatic_success_rate=None,
            human_evaluations=0,
            average_relevance=None,
            average_completeness=None,
            average_quality=None,
        ),
        comparison,
    )
    experiment_report = experiment_report_markdown(comparison)

    assert trace_payload["events"][0]["sequence"] == 1
    assert comparison_payload["token_reduction_percent"] > 0
    assert "do not establish hidden prompt content" in report
    assert "Token reduction alone does not establish an optimization benefit" in experiment_report
    assert "Human-quality change: -1.00 / 5" in experiment_report