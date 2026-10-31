"""Read-only analytics built from persisted runs, calls, and events."""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from analytics.schemas import (
    AgentAnalytics,
    AgentAnalyticsRow,
    ComponentTokenGroup,
    TokenAnalytics,
    TokenDay,
    TokenGroup,
    TokenTotals,
    ToolAnalyticsRow,
)
from core.enums import EventType, ExecutionStatus
from database.models import (
    AgentRecord,
    ExecutionEventRecord,
    ExecutionRunRecord,
    LLMCallRecord,
)


class AnalyticsService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def token_analytics(self) -> TokenAnalytics:
        with self._session_factory() as session:
            call_rows = session.execute(
                select(LLMCallRecord, AgentRecord.id, AgentRecord.name)
                .join(ExecutionRunRecord, LLMCallRecord.run_id == ExecutionRunRecord.id)
                .join(AgentRecord, ExecutionRunRecord.agent_id == AgentRecord.id)
                .order_by(LLMCallRecord.requested_at)
            ).all()

        by_model: dict[str, dict[str, Any]] = defaultdict(_token_bucket)
        by_provider: dict[str, dict[str, Any]] = defaultdict(_token_bucket)
        by_agent: dict[str, dict[str, Any]] = defaultdict(_token_bucket)
        by_day: dict[date, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        by_component: dict[tuple[str, str | None], int] = defaultdict(int)
        estimated_costs: dict[str, Decimal] = defaultdict(Decimal)
        reported_calls = input_tokens = output_tokens = reasoning = cached = 0
        component_total = attribution_gap = 0

        for call, agent_id, agent_name in call_rows:
            input_count = call.input_tokens or 0 if call.usage_is_reported else 0
            output_count = call.output_tokens or 0 if call.usage_is_reported else 0
            for key, bucket in (
                (call.model, by_model[call.model]),
                (call.provider, by_provider[call.provider]),
                (agent_name, by_agent[agent_name]),
            ):
                bucket["calls"] += 1
                bucket["input"] += input_count
                bucket["output"] += output_count
                if call.estimated_cost is not None and call.cost_currency:
                    bucket["costs"][call.cost_currency] += call.estimated_cost

            if call.usage_is_reported:
                reported_calls += 1
                input_tokens += input_count
                output_tokens += output_count
                reasoning += call.reasoning_tokens or 0
                cached += call.cached_input_tokens or 0
                day = call.requested_at.date()
                by_day[day]["input"] += input_count
                by_day[day]["output"] += output_count
            if call.cost_currency and call.estimated_cost is not None:
                estimated_costs[call.cost_currency] += call.estimated_cost
            component_total += call.estimated_component_tokens or 0
            attribution_gap += call.attribution_gap_tokens or 0
            for component in call.component_breakdown or []:
                kind = str(component.get("kind", "unknown"))
                source_id = component.get("source_id")
                source = str(source_id) if source_id is not None else None
                by_component[(kind, source)] += int(component.get("estimated_tokens", 0))

        return TokenAnalytics(
            totals=TokenTotals(
                llm_calls=len(call_rows),
                reported_calls=reported_calls,
                reported_input_tokens=input_tokens,
                reported_output_tokens=output_tokens,
                reported_reasoning_tokens=reasoning,
                cached_input_tokens=cached,
                estimated_component_tokens=component_total,
                attribution_gap_tokens=attribution_gap,
                estimated_costs=dict(estimated_costs),
            ),
            by_model=_token_groups(by_model),
            by_provider=_token_groups(by_provider),
            by_agent=_token_groups(by_agent),
            by_day=[
                TokenDay(day=day, input_tokens=values["input"], output_tokens=values["output"])
                for day, values in sorted(by_day.items())
            ],
            by_component=[
                ComponentTokenGroup(component=kind, source_id=source, estimated_tokens=count)
                for (kind, source), count in sorted(by_component.items())
            ],
        )

    def tool_analytics(self) -> list[ToolAnalyticsRow]:
        with self._session_factory() as session:
            events = session.scalars(
                select(ExecutionEventRecord)
                .where(
                    ExecutionEventRecord.event_type.in_(
                        [EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value]
                    )
                )
                .order_by(ExecutionEventRecord.occurred_at)
            ).all()
            calls = session.scalars(select(LLMCallRecord)).all()

        buckets: dict[str, dict[str, Any]] = defaultdict(_tool_bucket)
        seen_fingerprints: set[tuple[str, str, str]] = set()
        for event in events:
            metadata = event.event_data or {}
            name = metadata.get("tool_name")
            if not isinstance(name, str):
                continue
            bucket = buckets[name]
            bucket["calls"] += 1
            bucket["failures"] += event.event_type == EventType.TOOL_FAILED.value
            bucket["empty"] += metadata.get("is_empty") is True
            bucket["duration"] += int(metadata.get("duration_ms") or 0)
            bucket["size"] += int(metadata.get("output_size_bytes") or 0)
            bucket["estimated"] += int(metadata.get("estimated_output_tokens") or 0)
            fingerprint = metadata.get("argument_fingerprint")
            if isinstance(fingerprint, str):
                signature = (event.run_id, name, fingerprint)
                if signature in seen_fingerprints:
                    bucket["duplicates"] += 1
                seen_fingerprints.add(signature)

        for call in calls:
            for component in call.component_breakdown or []:
                if component.get("kind") != "tool_result":
                    continue
                source = component.get("source_id")
                if isinstance(source, str):
                    buckets[source]["included"] += int(component.get("estimated_tokens", 0))

        return [
            ToolAnalyticsRow(
                tool_name=name,
                calls=values["calls"],
                failures=values["failures"],
                duplicate_calls=values["duplicates"],
                empty_results=values["empty"],
                total_duration_ms=values["duration"],
                total_output_size_bytes=values["size"],
                estimated_output_tokens=values["estimated"],
                included_result_tokens=values["included"],
            )
            for name, values in sorted(buckets.items())
        ]

    def agent_analytics(self) -> AgentAnalytics:
        with self._session_factory() as session:
            agents = session.scalars(select(AgentRecord).order_by(AgentRecord.name)).all()
            runs = session.scalars(select(ExecutionRunRecord)).all()
            calls = session.scalars(select(LLMCallRecord)).all()
            tool_events = session.scalars(
                select(ExecutionEventRecord).where(
                    ExecutionEventRecord.event_type.in_(
                        [EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value]
                    )
                )
            ).all()

        runs_by_agent: dict[str, list[ExecutionRunRecord]] = defaultdict(list)
        run_agent: dict[str, str] = {}
        for run in runs:
            runs_by_agent[run.agent_id].append(run)
            run_agent[run.id] = run.agent_id
        calls_by_agent: dict[str, list[LLMCallRecord]] = defaultdict(list)
        for call in calls:
            agent_id = run_agent.get(call.run_id)
            if agent_id:
                calls_by_agent[agent_id].append(call)
        tool_counts: dict[str, int] = defaultdict(int)
        for event in tool_events:
            agent_id = run_agent.get(event.run_id)
            if agent_id:
                tool_counts[agent_id] += 1

        rows: list[AgentAnalyticsRow] = []
        for agent in agents:
            agent_runs = runs_by_agent[agent.id]
            agent_calls = calls_by_agent[agent.id]
            successes = sum(run.status == ExecutionStatus.SUCCEEDED.value for run in agent_runs)
            actual_calls = [call for call in agent_calls if call.usage_is_reported]
            input_count = sum(call.input_tokens or 0 for call in actual_calls)
            output_count = sum(call.output_tokens or 0 for call in actual_calls)
            costs: dict[str, Decimal] = defaultdict(Decimal)
            for call in agent_calls:
                if call.estimated_cost is not None and call.cost_currency:
                    costs[call.cost_currency] += call.estimated_cost
            run_count = len(agent_runs)
            rows.append(
                AgentAnalyticsRow(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    executions=run_count,
                    successful_executions=successes,
                    success_rate=successes / run_count if run_count else 0.0,
                    llm_calls=len(agent_calls),
                    average_llm_calls_per_execution=len(agent_calls) / run_count if run_count else 0.0,
                    reported_input_tokens=input_count,
                    reported_output_tokens=output_count,
                    average_reported_tokens_per_execution=(input_count + output_count) / run_count
                    if run_count
                    else 0.0,
                    tool_calls=tool_counts[agent.id],
                    average_tool_calls_per_execution=tool_counts[agent.id] / run_count
                    if run_count
                    else 0.0,
                    estimated_costs=dict(costs),
                )
            )
        return AgentAnalytics(agents=rows)


def _token_bucket() -> dict[str, Any]:
    return {"calls": 0, "input": 0, "output": 0, "costs": defaultdict(Decimal)}


def _token_groups(buckets: dict[str, dict[str, Any]]) -> list[TokenGroup]:
    return [
        TokenGroup(
            key=key,
            calls=values["calls"],
            input_tokens=values["input"],
            output_tokens=values["output"],
            estimated_costs=dict(values["costs"]),
        )
        for key, values in sorted(buckets.items())
    ]


def _tool_bucket() -> dict[str, int]:
    return {
        "calls": 0,
        "failures": 0,
        "duplicates": 0,
        "empty": 0,
        "duration": 0,
        "size": 0,
        "estimated": 0,
        "included": 0,
    }