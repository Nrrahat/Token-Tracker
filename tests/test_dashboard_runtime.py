from pathlib import Path

import pytest
from sqlalchemy import inspect

from config.settings import Settings
from dashboard.runtime import build_services
from dashboard.document_storage import DocumentUploadError, store_document


def test_dashboard_services_apply_migrations_and_register_safe_capabilities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "dashboard.db"
    monkeypatch.setattr(
        "dashboard.runtime.get_settings",
        lambda: Settings(database_url=f"sqlite:///{database_path.as_posix()}"),
    )

    services = build_services()

    table_names = set(inspect(services.engine).get_table_names())
    assert {"agents", "execution_runs", "llm_calls", "agent_skills", "agent_tools", "evaluations"}.issubset(
        table_names
    )
    assert {tool.name for tool in services.tool_registry.list_tools()} == {
        "calculator",
        "read_document",
    }
    assert [skill.metadata.name for skill in services.skill_registry.discover()] == [
        "document_analysis"
    ]
    assert services.custom_tool_store.registry_path == (
        Path(__file__).parents[1] / "custom_tools" / "tools.json"
    )
    assert services.agent_manager.list_agents() == []
    assert services.evaluation_manager.summary().automatic_evaluations == 0
    assert services.experiment_manager.list_experiments() == []

    services.tool_manager.shutdown()
    services.engine.dispose()


def test_document_upload_is_confined_bounded_and_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "uploads"
    saved = store_document(root, "notes.md", b"Safe content")

    assert saved.parent == root.resolve()
    assert saved.read_bytes() == b"Safe content"
    with pytest.raises(DocumentUploadError, match="plain file name"):
        store_document(root, "../outside.md", b"no")
    with pytest.raises(DocumentUploadError, match="File type"):
        store_document(root, "script.py", b"print(1)")
    with pytest.raises(DocumentUploadError, match="already exists"):
        store_document(root, "notes.md", b"replace")
    with pytest.raises(DocumentUploadError, match="1 MB"):
        store_document(root, "large.txt", b"x" * 1_000_001)