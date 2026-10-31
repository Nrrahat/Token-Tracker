import pytest
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_initial_migration_upgrades_phase_one_database(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE agents ("
                "id VARCHAR(36) PRIMARY KEY, "
                "name VARCHAR(120) NOT NULL, "
                "description TEXT NOT NULL DEFAULT '', "
                "provider VARCHAR(40) NOT NULL, "
                "model VARCHAR(200) NOT NULL, "
                "system_instructions TEXT NOT NULL DEFAULT '', "
                "developer_instructions TEXT NOT NULL DEFAULT '', "
                "temperature FLOAT NOT NULL DEFAULT 0.2, "
                "max_iterations INTEGER NOT NULL DEFAULT 10, "
                "max_tool_calls INTEGER NOT NULL DEFAULT 20, "
                "is_active BOOLEAN NOT NULL DEFAULT 1, "
                "created_at DATETIME NOT NULL, "
                "updated_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO agents (id, name, description, provider, model, system_instructions, "
                "developer_instructions, temperature, max_iterations, max_tool_calls, is_active, "
                "created_at, updated_at) VALUES ("
                "'legacy-agent', 'Legacy agent', '', 'openai', 'test-model', '', '', "
                "0.2, 10, 20, 1, '2026-01-01', '2026-01-01')"
            )
        )
    engine.dispose()

    project_root = Path(__file__).parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    migrated_engine = create_engine(database_url)
    inspector = inspect(migrated_engine)
    assert "request_timeout_seconds" in {
        column["name"] for column in inspector.get_columns("agents")
    }
    assert {
        "llm_calls",
        "agent_skills",
        "agent_tools",
        "evaluations",
        "experiments",
        "experiment_results",
        "alembic_version",
    }.issubset(
        set(inspector.get_table_names())
    )
    with migrated_engine.connect() as connection:
        migrated_agent = connection.execute(
            text("SELECT id, request_timeout_seconds FROM agents WHERE id = 'legacy-agent'")
        ).one()
    assert migrated_agent == ("legacy-agent", 60.0)
    with pytest.raises(RuntimeError, match="forward-only"):
        command.downgrade(config, "base")
    with migrated_engine.connect() as connection:
        preserved_agent = connection.execute(
            text("SELECT id FROM agents WHERE id = 'legacy-agent'")
        ).scalar_one()
    assert preserved_agent == "legacy-agent"
    migrated_engine.dispose()


def test_initial_migration_creates_empty_database(tmp_path: Path) -> None:
    database_path = tmp_path / "new.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    project_root = Path(__file__).parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    table_names = set(inspect(engine).get_table_names())
    assert {
        "agents",
        "tasks",
        "execution_runs",
        "execution_events",
        "llm_calls",
        "agent_skills",
        "agent_tools",
        "evaluations",
        "experiments",
        "experiment_results",
    }.issubset(table_names)
    engine.dispose()