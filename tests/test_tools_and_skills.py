from threading import Event
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from skills.skill_loader import SkillLoader
from skills.skill_registry import SkillNotFoundError, SkillRegistry
from tools.available.calculator import CalculatorTool
from tools.available.safe_document_reader import SafeDocumentReaderTool
from tools.base_tool import BaseTool, ToolStatus
from tools.custom_tool_store import (
    CustomToolAlreadyExistsError,
    CustomToolStorageError,
    CustomToolStore,
)
from tools.custom_tool_testing import test_custom_tool as run_custom_tool_test
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolNotFoundError, ToolRegistry


class CharacterTokenizer:
    def count_text(self, text: str) -> int:
        return len(text)


def test_custom_tool_store_persists_metadata_and_source_across_instances(tmp_path: Path) -> None:
    source = "def run(arguments):\n    return {'value': 42}\n"
    first_store = CustomToolStore(tmp_path / "custom_tools")

    metadata = first_store.create(
        "answer_tool",
        "Returns a fixed answer.",
        "run",
        source,
    )
    reopened_store = CustomToolStore(tmp_path / "custom_tools")

    assert metadata.name == "answer_tool"
    assert (tmp_path / "custom_tools" / "answer_tool.py").read_text(encoding="utf-8") == source
    assert reopened_store.list_tools() == [metadata]
    assert reopened_store.get("answer_tool").source == source
    registry = reopened_store.registry_path.read_text(encoding="utf-8")
    assert '"version": 1' in registry
    assert '"function_name": "run"' in registry


def test_custom_tool_store_rejects_duplicate_names_and_invalid_function_names(tmp_path: Path) -> None:
    store = CustomToolStore(tmp_path / "custom_tools")
    store.create("answer_tool", "Description", "run", "def run(arguments): return 42")

    with pytest.raises(CustomToolAlreadyExistsError):
        store.create("answer_tool", "Duplicate", "run", "def run(arguments): return 0")
    with pytest.raises(ValueError, match="valid, non-keyword Python identifier"):
        store.create("another_tool", "Description", "class", "pass")
    with pytest.raises(ValueError, match="Invalid custom tool metadata"):
        store.create("../outside", "Description", "run", "pass")


def test_custom_tool_store_reports_corrupt_registry(tmp_path: Path) -> None:
    root = tmp_path / "custom_tools"
    root.mkdir()
    (root / "tools.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(CustomToolStorageError, match="Could not read"):
        CustomToolStore(root).list_tools()


def test_custom_tool_store_updates_and_deletes_saved_tools(tmp_path: Path) -> None:
    store = CustomToolStore(tmp_path / "custom_tools")
    store.create("answer_tool", "Original", "run", "def run(arguments): return 1")

    updated = store.update(
        "answer_tool",
        "Updated description",
        "calculate",
        "def calculate(arguments): return {'value': 2}",
    )

    assert updated.description == "Updated description"
    assert store.get("answer_tool").source == "def calculate(arguments): return {'value': 2}"
    assert store.get("answer_tool").metadata.function_name == "calculate"
    store.delete("answer_tool")
    assert store.list_tools() == []
    assert not (tmp_path / "custom_tools" / "answer_tool.py").exists()
    with pytest.raises(LookupError):
        store.get("answer_tool")
    with pytest.raises(LookupError):
        store.delete("answer_tool")


def test_custom_tool_test_runs_safe_code_in_a_separate_process() -> None:
    result = run_custom_tool_test(
        "def run(arguments: dict[str, object]) -> dict[str, object]:\n"
        "    print('testing')\n"
        "    return {'answer': arguments['number'] * 2}\n",
        "run",
        {"number": 21},
    )

    assert result.success
    assert result.output == {"answer": 42}
    assert result.stdout == "testing\n"


def test_custom_tool_test_enforces_a_process_timeout() -> None:
    source = (
        "def run(arguments):\n"
        "    total = 0\n"
        "    for left in range(10000):\n"
        "        for right in range(10000):\n"
        "            total += left + right\n"
        "    return total\n"
    )

    result = run_custom_tool_test(source, "run", {})

    assert not result.success
    assert result.error == "Custom tool test exceeded the 2-second limit."


@pytest.mark.parametrize(
    ("source", "expected_error"),
    [
        ("def run(arguments):\n    import os\n", "Import is not allowed"),
        ("def run(arguments):\n    return arguments.__class__\n", "Attribute"),
        ("def run(arguments):\n    while True:\n        pass\n", "While is not allowed"),
        ("def wrong(arguments):\n    return 1\n", "Define 'run(arguments)'"),
        ("def run(arguments):\n    return [0] * 100001\n", "exceeded the supported size"),
        ("def run(arguments):\n    return int('invalid')\n", "ValueError"),
    ],
)
def test_custom_tool_test_rejects_unsafe_or_invalid_code(
    source: str,
    expected_error: str,
) -> None:
    result = run_custom_tool_test(source, "run", {})

    assert not result.success
    assert result.error is not None
    assert expected_error.lower() in result.error.lower()


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