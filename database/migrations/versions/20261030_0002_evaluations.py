"""Persist automatic and human task evaluations."""

from alembic import op
import sqlalchemy as sa

revision = "20261030_0002"
down_revision = "20261030_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "evaluations" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("execution_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evaluator_type", sa.String(16), nullable=False),
        sa.Column("evaluator_name", sa.String(120), nullable=False),
        sa.Column("task_success", sa.Boolean(), nullable=False),
        sa.Column("tool_use_success", sa.Boolean()),
        sa.Column("response_relevance", sa.Integer()),
        sa.Column("response_completeness", sa.Integer()),
        sa.Column("output_quality", sa.Integer()),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluations_run_id", "evaluations", ["run_id"])
    op.create_index("ix_evaluations_evaluator_type", "evaluations", ["evaluator_type"])
    op.create_index("ix_evaluations_run_evaluated", "evaluations", ["run_id", "evaluated_at"])


def downgrade() -> None:
    raise RuntimeError("Evaluation history migration is forward-only; restore a database backup to roll back.")