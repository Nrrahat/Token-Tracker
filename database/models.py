"""Initial SQLAlchemy entities for agent execution persistence."""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AgentRecord(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    system_instructions: Mapped[str] = mapped_column(Text, nullable=False, default="")
    developer_instructions: Mapped[str] = mapped_column(Text, nullable=False, default="")
    temperature: Mapped[float] = mapped_column(nullable=False, default=0.2)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    max_tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    request_timeout_seconds: Mapped[float] = mapped_column(nullable=False, default=60.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    tasks: Mapped[list["TaskRecord"]] = relationship(back_populates="agent")
    runs: Mapped[list["ExecutionRunRecord"]] = relationship(back_populates="agent")
    skill_attachments: Mapped[list["AgentSkillAttachmentRecord"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )
    tool_attachments: Mapped[list["AgentToolAttachmentRecord"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )


class TaskRecord(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    agent: Mapped[AgentRecord] = relationship(back_populates="tasks")
    runs: Mapped[list["ExecutionRunRecord"]] = relationship(back_populates="task")


class ExecutionRunRecord(Base):
    __tablename__ = "execution_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    agent: Mapped[AgentRecord] = relationship(back_populates="runs")
    task: Mapped[TaskRecord] = relationship(back_populates="runs")
    events: Mapped[list["ExecutionEventRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="ExecutionEventRecord.sequence"
    )
    llm_calls: Mapped[list["LLMCallRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="LLMCallRecord.requested_at"
    )
    evaluations: Mapped[list["EvaluationRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EvaluationRecord.evaluated_at"
    )


class ExecutionEventRecord(Base):
    __tablename__ = "execution_events"
    __table_args__ = (Index("ix_execution_events_run_sequence", "run_id", "sequence", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("execution_runs.id", ondelete="CASCADE"), nullable=False)
    parent_event_id: Mapped[str | None] = mapped_column(
        ForeignKey("execution_events.id", ondelete="SET NULL")
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    status: Mapped[str | None] = mapped_column(String(24))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    event_data: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    error_details: Mapped[str | None] = mapped_column(Text)

    run: Mapped[ExecutionRunRecord] = relationship(back_populates="events")
    parent: Mapped["ExecutionEventRecord | None"] = relationship(
        remote_side="ExecutionEventRecord.id", back_populates="children"
    )
    children: Mapped[list["ExecutionEventRecord"]] = relationship(back_populates="parent")


class LLMCallRecord(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_run_requested", "run_id", "requested_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("execution_runs.id", ondelete="CASCADE"), nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latency_ms: Mapped[float] = mapped_column(nullable=False)
    provider_latency_ms: Mapped[float | None] = mapped_column()
    request_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    reported_total_tokens: Mapped[int | None] = mapped_column(Integer)
    usage_is_reported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_usage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    estimated_component_tokens: Mapped[int | None] = mapped_column(Integer)
    component_breakdown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    attribution_gap_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 9))
    cost_currency: Mapped[str | None] = mapped_column(String(8))
    pricing_effective_date: Mapped[date | None] = mapped_column(Date)
    error_type: Mapped[str | None] = mapped_column(String(200))

    run: Mapped[ExecutionRunRecord] = relationship(back_populates="llm_calls")


class AgentSkillAttachmentRecord(Base):
    __tablename__ = "agent_skills"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True
    )
    skill_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    skill_version: Mapped[str] = mapped_column(String(40), nullable=False)
    attached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    agent: Mapped[AgentRecord] = relationship(back_populates="skill_attachments")


class AgentToolAttachmentRecord(Base):
    __tablename__ = "agent_tools"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True
    )
    tool_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    attached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    agent: Mapped[AgentRecord] = relationship(back_populates="tool_attachments")


class EvaluationRecord(Base):
    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_evaluations_run_evaluated", "run_id", "evaluated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("execution_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    evaluator_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    evaluator_name: Mapped[str] = mapped_column(String(120), nullable=False)
    task_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tool_use_success: Mapped[bool | None] = mapped_column(Boolean)
    response_relevance: Mapped[int | None] = mapped_column(Integer)
    response_completeness: Mapped[int | None] = mapped_column(Integer)
    output_quality: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    run: Mapped[ExecutionRunRecord] = relationship(back_populates="evaluations")


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="draft", index=True)
    baseline_agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False
    )
    optimized_agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False
    )
    baseline_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    optimized_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(200))

    results: Mapped[list["ExperimentResultRecord"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentResultRecord.case_id"
    )


class ExperimentResultRecord(Base):
    __tablename__ = "experiment_results"
    __table_args__ = (
        Index("ix_experiment_results_experiment_case_arm", "experiment_id", "case_id", "arm", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_id: Mapped[str] = mapped_column(String(80), nullable=False)
    arm: Mapped[str] = mapped_column(String(16), nullable=False)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("execution_runs.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    experiment: Mapped[ExperimentRecord] = relationship(back_populates="results")