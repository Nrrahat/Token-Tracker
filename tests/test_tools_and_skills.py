from threading import Event
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from skills.skill_loader import SkillLoader
from skills.skill_registry import SkillNotFoundError, SkillRegistry
from tools.available.calculator import CalculatorTool
from tools.available.safe_document_reader import SafeDocumentReaderTool
from tools.base_tool import BaseTool, ToolStatus
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolNotFoundError, ToolRegistry


class CharacterTokenizer:
    def count_text(self, text: str) -> int:
        return len(text)


def test_calculator_accepts_arithmetic_and_rejects_code_execution() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    manager = ToolManager(registry, tokenizer=CharacterTokenizer())

    result = manager.execute("calculator", {"expression": "2 * (3 + 4)"})
    malicious = manager.execute("calculator", {"expression": "__import__('os').getcwd()"})

    assert result.status is ToolStatus.SUCCEEDED
    assert result.output == {"result": "14"}
    assert result.estimated_output_tokens == len('{"result":"14"}')
    assert malicious.status is ToolStatus.FAILED
    assert malicious.error_message == "Tool execution failed."
    invalid_schema = manager.execute(
        "calculator",
        {"expression": "1 + 1", "shell": "not accepted"},
    )
    assert invalid_schema.error_type == "ValidationError"
    assert manager.metrics()["calculator"].failure_count == 2
    manager.shutdown()


def test_calculator_bounds_expression_complexity_and_exponent() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    manager = ToolManager(registry)

    large_exponent = manager.execute("calculator", {"expression": "2 ** 101"})
    huge_result = manager.execute("calculator", {"expression": "10 ** 100"})

    assert large_exponent.status is ToolStatus.FAILED
    assert huge_result.status is ToolStatus.SUCCEEDED
    manager.shutdown()


def test_tool_registry_is_an_allowlist_and_metrics_count_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    manager = ToolManager(registry)

    manager.execute("calculator", {"expression": "1 + 1"})
    result = manager.execute("calculator", {"expression": "1 + 1"})
    with pytest.raises(ToolNotFoundError):
        manager.execute("python", {"code": "print(1)"})

    metrics = manager.metrics()["calculator"]
    assert result.status is ToolStatus.SUCCEEDED
    assert metrics.call_count == 2
    assert metrics.duplicate_call_count == 1
    manager.shutdown()


def test_tool_result_component_marks_content_for_later_prompt_use() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    manager = ToolManager(registry)
    result = manager.execute("calculator", {"expression": "1 + 1"})

    component = result.as_tool_result_message()

    assert component.kind.value == "tool_result"
    assert component.text == '{"result":"2"}'
    manager.shutdown()


def test_safe_document_reader_confines_paths_and_limits_file_types(tmp_path) -> None:
    allowed_root = tmp_path / "documents"
    allowed_root.mkdir()
    (allowed_root / "notes.md").write_text("Useful notes", encoding="utf-8")
    (allowed_root / "large.txt").write_text("x" * 101, encoding="utf-8")
    (tmp_path / "outside.txt").write_text("Do not read", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SafeDocumentReaderTool(allowed_root, max_bytes=100))
    manager = ToolManager(registry)

    valid = manager.execute("read_document", {"path": "notes.md"})
    traversal = manager.execute("read_document", {"path": "../outside.txt"})
    unsupported = manager.execute("read_document", {"path": "notes.exe"})
    oversized = manager.execute("read_document", {"path": "large.txt"})

    assert valid.status is ToolStatus.SUCCEEDED
    assert valid.output["content"] == "Useful notes"
    assert traversal.status is ToolStatus.FAILED
    assert unsupported.status is ToolStatus.FAILED
    assert oversized.status is ToolStatus.FAILED
    manager.shutdown()


class WaitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WaitingTool(BaseTool):
    name = "waiter"
    description = "Test a bounded tool timeout."
    input_model = WaitInput
    timeout_seconds = 0.01

    def __init__(self, release: Event) -> None:
        self._release = release

    def execute(self, arguments: BaseModel) -> str:
        self._release.wait(timeout=1)
        return "released"


def test_tool_manager_reports_timeout_and_cancels_pending_work() -> None:
    release = Event()
    registry = ToolRegistry()
    registry.register(WaitingTool(release))
    manager = ToolManager(registry, max_workers=1)

    result = manager.execute("waiter", {})
    release.set()

    assert result.status is ToolStatus.TIMED_OUT
    assert result.error_type == "TimeoutError"
    assert manager.metrics()["waiter"].timeout_count == 1
    manager.shutdown()


def test_skill_registry_loads_versioned_skill_and_token_estimate(tmp_path) -> None:
    skill_dir = tmp_path / "sample_skill"
    skill_dir.mkdir()
    (skill_dir / "metadata.json").write_text(
        '{"name":"sample_skill","description":"Sample","version":"2.1.0",'
        '"tags":["test"],"enabled":true}',
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Use evidence.", encoding="utf-8")
    registry = SkillRegistry(tmp_path, SkillLoader(CharacterTokenizer()))

    loaded = registry.load("sample_skill")

    assert loaded.metadata.version == "2.1.0"
    assert loaded.instructions == "Use evidence."
    assert loaded.estimated_instruction_tokens == len("Use evidence.")
    assert registry.load_many(["sample_skill", "sample_skill"]) == [loaded]


def test_skill_registry_discovers_bundled_document_skill_and_rejects_disabled(tmp_path) -> None:
    bundled_skills = Path(__file__).parents[1] / "skills" / "available"
    registry = SkillRegistry(bundled_skills, SkillLoader(CharacterTokenizer()))

    skills = registry.discover()

    assert [skill.metadata.name for skill in skills] == ["document_analysis"]
    assert skills[0].metadata.version == "1.0.0"
    assert skills[0].estimated_instruction_tokens is not None
    with pytest.raises(SkillNotFoundError):
        registry.load("..")


def test_skill_registry_honors_disabled_metadata(tmp_path) -> None:
    skill_dir = tmp_path / "disabled_skill"
    skill_dir.mkdir()
    (skill_dir / "metadata.json").write_text(
        '{"name":"disabled_skill","description":"Disabled","version":"1.0.0",'
        '"enabled":false}',
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Disabled instructions.", encoding="utf-8")
    registry = SkillRegistry(tmp_path)

    assert registry.discover() == []
    assert len(registry.discover(include_disabled=True)) == 1
    with pytest.raises(SkillNotFoundError):
        registry.load("disabled_skill")


def test_skill_registry_can_enable_and_disable_a_skill(tmp_path) -> None:
    skill_dir = tmp_path / "toggle_skill"
    skill_dir.mkdir()
    (skill_dir / "metadata.json").write_text(
        '{"name":"toggle_skill","description":"Toggle","version":"1.0.0",'
        '"enabled":true}',
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Toggle instructions.", encoding="utf-8")
    registry = SkillRegistry(tmp_path)

    disabled = registry.set_enabled("toggle_skill", False)
    assert disabled.metadata.enabled is False
    assert registry.discover() == []
    enabled = registry.set_enabled("toggle_skill", True)
    assert enabled.metadata.enabled is True
    assert [skill.metadata.name for skill in registry.discover()] == ["toggle_skill"]