"""Create the current schema and add Phase 6 agent request timeout."""

from alembic import op
import sqlalchemy as sa

revision = "20261030_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "agents" not in tables:
        op.create_table(
            "agents",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("provider", sa.String(40), nullable=False),
            sa.Column("model", sa.String(200), nullable=False),
            sa.Column("system_instructions", sa.Text(), nullable=False),
            sa.Column("developer_instructions", sa.Text(), nullable=False),
            sa.Column("temperature", sa.Float(), nullable=False),
            sa.Column("max_iterations", sa.Integer(), nullable=False),
            sa.Column("max_tool_calls", sa.Integer(), nullable=False),
            sa.Column("request_timeout_seconds", sa.Float(), nullable=False, server_default="60.0"),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_agents_name", "agents", ["name"])

    agent_columns = {column["name"] for column in sa.inspect(bind).get_columns("agents")}
    if "request_timeout_seconds" not in agent_columns:
        op.add_column(
            "agents",
            sa.Column("request_timeout_seconds", sa.Float(), nullable=False, server_default="60.0"),
        )

    if "tasks" not in tables:
        op.create_table(
            "tasks",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_tasks_agent_id", "tasks", ["agent_id"])

    if "execution_runs" not in tables:
        op.create_table(
            "execution_runs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
            sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True)),
            sa.Column("finished_at", sa.DateTime(timezone=True)),
            sa.Column("result", sa.Text()),
            sa.Column("error_message", sa.Text()),
        )
        op.create_index("ix_execution_runs_task_id", "execution_runs", ["task_id"])
        op.create_index("ix_execution_runs_agent_id", "execution_runs", ["agent_id"])
        op.create_index("ix_execution_runs_status", "execution_runs", ["status"])

    if "execution_events" not in tables:
        op.create_table(
            "execution_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("execution_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("parent_event_id", sa.String(36), sa.ForeignKey("execution_events.id", ondelete="SET NULL")),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(48), nullable=False),
            sa.Column("status", sa.String(24)),
            sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("duration_ms", sa.Integer()),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("error_details", sa.Text()),
        )
        op.create_index("ix_execution_events_event_type", "execution_events", ["event_type"])
        op.create_index("ix_execution_events_run_sequence", "execution_events", ["run_id", "sequence"], unique=True)

    if "llm_calls" not in tables:
        op.create_table(
            "llm_calls",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("execution_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("provider", sa.String(40), nullable=False),
            sa.Column("model", sa.String(200), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("latency_ms", sa.Float(), nullable=False),
            sa.Column("provider_latency_ms", sa.Float()),
            sa.Column("request_metadata", sa.JSON(), nullable=False),
            sa.Column("input_tokens", sa.Integer()),
            sa.Column("output_tokens", sa.Integer()),
            sa.Column("reasoning_tokens", sa.Integer()),
            sa.Column("cached_input_tokens", sa.Integer()),
            sa.Column("reported_total_tokens", sa.Integer()),
            sa.Column("usage_is_reported", sa.Boolean(), nullable=False),
            sa.Column("raw_usage", sa.JSON(), nullable=False),
            sa.Column("estimated_component_tokens", sa.Integer()),
            sa.Column("component_breakdown", sa.JSON()),
            sa.Column("attribution_gap_tokens", sa.Integer()),
            sa.Column("estimated_cost", sa.Numeric(18, 9)),
            sa.Column("cost_currency", sa.String(8)),
            sa.Column("pricing_effective_date", sa.Date()),
            sa.Column("error_type", sa.String(200)),
        )
        op.create_index("ix_llm_calls_run_id", "llm_calls", ["run_id"])
        op.create_index("ix_llm_calls_status", "llm_calls", ["status"])
        op.create_index("ix_llm_calls_run_requested", "llm_calls", ["run_id", "requested_at"])

    if "agent_skills" not in tables:
        op.create_table(
            "agent_skills",
            sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("skill_name", sa.String(80), primary_key=True),
            sa.Column("skill_version", sa.String(40), nullable=False),
            sa.Column("attached_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "agent_tools" not in tables:
        op.create_table(
            "agent_tools",
            sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("tool_name", sa.String(64), primary_key=True),
            sa.Column("attached_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    for table_name in (
        "agent_tools",
        "agent_skills",
        "llm_calls",
        "execution_events",
        "execution_runs",
        "tasks",
        "agents",
    ):
        raise RuntimeError(
            "This schema-adoption migration is forward-only; restore a database backup to roll back."
        )