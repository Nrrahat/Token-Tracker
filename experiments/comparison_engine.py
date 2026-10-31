"""Compare paired experiment arms using persisted runs, calls, and human ratings."""

from collections import defaultdict
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from core.enums import EventType, ExecutionStatus
from database.models import (
    EvaluationRecord,
    ExecutionEventRecord,
    ExecutionRunRecord,
    ExperimentResultRecord,
    LLMCallRecord,
)
from evaluation.schemas import EvaluatorType
from experiments.experiment_manager import ExperimentManager
from experiments.schemas import (
    ExperimentArm,
    ExperimentArmMetrics,
    ExperimentComparison,
)


class ComparisonEngine:
    def __init__(self, session_factory: sessionmaker, experiment_manager: ExperimentManager) -> None:
        self._session_factory = session_factory
        self._experiment_manager = experiment_manager

    def compare(self, experiment_id: str) -> ExperimentComparison:
        definition = self._experiment_manager.get(experiment_id)
        results = self._experiment_manager.results(experiment_id)
        with self._session_factory() as session:
            run_ids = [result.run_id for result in results]
            runs = session.scalars(
                select(ExecutionRunRecord).where(ExecutionRunRecord.id.in_(run_ids))
            ).all() if run_ids else []
            calls = session.scalars(
                select(LLMCallRecord).where(LLMCallRecord.run_id.in_(run_ids))
            ).all() if run_ids else []
            events = session.scalars(
                select(ExecutionEventRecord).where(
                    ExecutionEventRecord.run_id.in_(run_ids),
                    ExecutionEventRecord.event_type.in_(
                        [EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value]
                    ),
                )
            ).all() if run_ids else []
            evaluations = session.scalars(
                select(EvaluationRecord).where(
                    EvaluationRecord.run_id.in_(run_ids),
                    EvaluationRecord.evaluator_type == EvaluatorType.HUMAN.value,
                )
            ).all() if run_ids else []

        run_by_id = {run.id: run for run in runs}
        baseline = _arm_metrics(ExperimentArm.BASELINE, results, run_by_id, calls, events, evaluations)
        optimized = _arm_metrics(ExperimentArm.OPTIMIZED, results, run_by_id, calls, events, evaluations)
        token_reduction = None
        if baseline.reported_total_tokens > 0:
            token_reduction = (
                (baseline.reported_total_tokens - optimized.reported_total_tokens)
                / baseline.reported_total_tokens
                * 100
            )
        success_change = None
        if baseline.task_success_rate is not None and optimized.task_success_rate is not None:
            success_change = optimized.task_success_rate - baseline.task_success_rate
        quality_change = None
        if baseline.average_human_quality is not None and optimized.average_human_quality is not None:
            quality_change = optimized.average_human_quality - baseline.average_human_quality
        cost_reduction: dict[str, float] = {}
        for currency in baseline.estimated_costs.keys() & optimized.estimated_costs.keys():
            baseline_cost = baseline.estimated_costs[currency]
            if baseline_cost > 0:
                cost_reduction[currency] = float(
                    (baseline_cost - optimized.estimated_costs[currency]) / baseline_cost * 100
                )
        return ExperimentComparison(
            experiment=definition,
            baseline=baseline,
            optimized=optimized,
            token_reduction_percent=token_reduction,
            success_rate_change=success_change,
            human_quality_change=quality_change,
            cost_reduction_percent_by_currency=cost_reduction,
            results=results,
        )


def _arm_metrics(arm, results, run_by_id, calls, events, evaluations):
    arm_results = [result for result in results if result.arm is arm]
    arm_run_ids = {result.run_id for result in arm_results}
    arm_runs = [run_by_id[run_id] for run_id in arm_run_ids if run_id in run_by_id]
    arm_calls = [call for call in calls if call.run_id in arm_run_ids]
    reported_calls = [call for call in arm_calls if call.usage_is_reported]
    arm_events = [event for event in events if event.run_id in arm_run_ids]
    human_evaluations = [evaluation for evaluation in evaluations if evaluation.run_id in arm_run_ids]
    successes = sum(run.status == ExecutionStatus.SUCCEEDED.value for run in arm_runs)
    run_count = len(arm_runs)
    costs: dict[str, Decimal] = defaultdict(Decimal)
    for call in arm_calls:
        if call.estimated_cost is not None and call.cost_currency:
            costs[call.cost_currency] += call.estimated_cost
    latencies = [
        (run.finished_at - run.started_at).total_seconds() * 1000
        for run in arm_runs
        if run.started_at is not None and run.finished_at is not None
    ]
    return ExperimentArmMetrics(
        runs=run_count,
        successful_runs=successes,
        task_success_rate=successes / run_count if run_count else None,
        reported_input_tokens=sum(call.input_tokens or 0 for call in reported_calls),
        reported_output_tokens=sum(call.output_tokens or 0 for call in reported_calls),
        reported_total_tokens=sum((call.input_tokens or 0) + (call.output_tokens or 0) for call in reported_calls),
        reported_reasoning_tokens=sum(call.reasoning_tokens or 0 for call in reported_calls),
        cached_input_tokens=sum(call.cached_input_tokens or 0 for call in reported_calls),
        llm_calls=len(arm_calls),
        average_llm_calls_per_run=len(arm_calls) / run_count if run_count else None,
        tool_calls=len(arm_events),
        average_tool_calls_per_run=len(arm_events) / run_count if run_count else None,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        estimated_costs=dict(costs),
        human_evaluations=len(human_evaluations),
        average_human_relevance=_mean(item.response_relevance for item in human_evaluations),
        average_human_completeness=_mean(item.response_completeness for item in human_evaluations),
        average_human_quality=_mean(item.output_quality for item in human_evaluations),
    )


def _mean(values) -> float | None:
    scores = [value for value in values if value is not None]
    return sum(scores) / len(scores) if scores else None