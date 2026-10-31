"""Privacy-conscious CSV exports for usage, tools, and experiments."""

import csv
from io import StringIO

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from analytics.schemas import ToolAnalyticsRow
from database.models import AgentRecord, ExecutionRunRecord, LLMCallRecord
from experiments.schemas import ExperimentComparison


def token_usage_csv(session_factory: sessionmaker) -> str:
    """Export per-call token facts without prompt text or provider credentials."""
    output = StringIO(newline="")
    fields = [
        "llm_call_id",
        "run_id",
        "agent_id",
        "agent_name",
        "requested_at",
        "provider",
        "model",
        "status",
        "usage_is_reported",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "reported_total_tokens",
        "estimated_component_tokens",
        "attribution_gap_tokens",
        "estimated_cost",
        "cost_currency",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    with session_factory() as session:
        rows = session.execute(
            select(LLMCallRecord, ExecutionRunRecord.agent_id, AgentRecord.name)
            .join(ExecutionRunRecord, LLMCallRecord.run_id == ExecutionRunRecord.id)
            .join(AgentRecord, ExecutionRunRecord.agent_id == AgentRecord.id)
            .order_by(LLMCallRecord.requested_at, LLMCallRecord.id)
        ).all()
        for call, agent_id, agent_name in rows:
            writer.writerow(
                {
                    "llm_call_id": call.id,
                    "run_id": call.run_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "requested_at": call.requested_at.isoformat(),
                    "provider": call.provider,
                    "model": call.model,
                    "status": call.status,
                    "usage_is_reported": call.usage_is_reported,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "reasoning_tokens": call.reasoning_tokens,
                    "cached_input_tokens": call.cached_input_tokens,
                    "reported_total_tokens": call.reported_total_tokens,
                    "estimated_component_tokens": call.estimated_component_tokens,
                    "attribution_gap_tokens": call.attribution_gap_tokens,
                    "estimated_cost": call.estimated_cost,
                    "cost_currency": call.cost_currency,
                }
            )
    return output.getvalue()


def tool_analytics_csv(rows: list[ToolAnalyticsRow]) -> str:
    output = StringIO(newline="")
    fields = [
        "tool_name",
        "calls",
        "failures",
        "duplicate_calls",
        "empty_results",
        "total_duration_ms",
        "total_output_size_bytes",
        "estimated_output_tokens",
        "included_result_tokens",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.model_dump())
    return output.getvalue()


def experiment_comparison_csv(comparison: ExperimentComparison) -> str:
    output = StringIO(newline="")
    fields = [
        "experiment_id",
        "experiment_name",
        "record_type",
        "arm",
        "case_id",
        "run_id",
        "run_status",
        "runs",
        "task_success_rate",
        "reported_input_tokens",
        "reported_output_tokens",
        "reported_total_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "llm_calls",
        "tool_calls",
        "average_latency_ms",
        "estimated_costs_by_currency",
        "human_evaluations",
        "average_human_quality",
        "token_reduction_percent",
        "success_rate_change",
        "human_quality_change",
        "cost_reduction_percent_by_currency",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for arm_name, arm in (("baseline", comparison.baseline), ("optimized", comparison.optimized)):
        writer.writerow(
            {
                "experiment_id": comparison.experiment.experiment_id,
                "experiment_name": comparison.experiment.name,
                "record_type": "arm_summary",
                "arm": arm_name,
                "runs": arm.runs,
                "task_success_rate": arm.task_success_rate,
                "reported_input_tokens": arm.reported_input_tokens,
                "reported_output_tokens": arm.reported_output_tokens,
                "reported_total_tokens": arm.reported_total_tokens,
                "reasoning_tokens": arm.reported_reasoning_tokens,
                "cached_input_tokens": arm.cached_input_tokens,
                "llm_calls": arm.llm_calls,
                "tool_calls": arm.tool_calls,
                "average_latency_ms": arm.average_latency_ms,
                "estimated_costs_by_currency": _json_value(arm.estimated_costs),
                "human_evaluations": arm.human_evaluations,
                "average_human_quality": arm.average_human_quality,
                "token_reduction_percent": comparison.token_reduction_percent,
                "success_rate_change": comparison.success_rate_change,
                "human_quality_change": comparison.human_quality_change,
                "cost_reduction_percent_by_currency": _json_value(
                    comparison.cost_reduction_percent_by_currency
                ),
            }
        )
    for result in comparison.results:
        writer.writerow(
            {
                "experiment_id": comparison.experiment.experiment_id,
                "experiment_name": comparison.experiment.name,
                "record_type": "trial",
                "arm": result.arm.value,
                "case_id": result.case_id,
                "run_id": result.run_id,
                "run_status": result.status.value,
            }
        )
    return output.getvalue()


def _json_value(value: object) -> str:
    import json

    return json.dumps(value, default=str, sort_keys=True)