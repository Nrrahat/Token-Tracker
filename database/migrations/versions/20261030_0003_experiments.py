"""Persist reproducible experiments and paired execution results."""

from alembic import op
import sqlalchemy as sa

revision = "20261030_0003"
down_revision = "20261030_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "experiments" not in tables:
        op.create_table(
            "experiments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(160), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("baseline_agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("optimized_agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("baseline_snapshot", sa.JSON(), nullable=False),
            sa.Column("optimized_snapshot", sa.JSON(), nullable=False),
            sa.Column("dataset_snapshot", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True)),
            sa.Column("finished_at", sa.DateTime(timezone=True)),
            sa.Column("error_type", sa.String(200)),
        )
        op.create_index("ix_experiments_name", "experiments", ["name"])
        op.create_index("ix_experiments_status", "experiments", ["status"])

    if "experiment_results" not in tables:
        op.create_table(
            "experiment_results",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("experiment_id", sa.String(36), sa.ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False),
            sa.Column("case_id", sa.String(80), nullable=False),
            sa.Column("arm", sa.String(16), nullable=False),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("execution_runs.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_experiment_results_experiment_id", "experiment_results", ["experiment_id"])
        op.create_index(
            "ix_experiment_results_experiment_case_arm",
            "experiment_results",
            ["experiment_id", "case_id", "arm"],
            unique=True,
        )
        op.create_index("ix_experiment_results_run_id", "experiment_results", ["run_id"], unique=True)


def downgrade() -> None:
    raise RuntimeError("Experiment history migration is forward-only; restore a database backup to roll back.")