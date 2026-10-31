"""Read small text documents from an explicitly configured directory."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tools.base_tool import BaseTool


class DocumentReaderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)


class SafeDocumentReaderTool(BaseTool):
    name = "read_document"
    description = "Read a UTF-8 text document inside the configured document directory."
    input_model = DocumentReaderInput
    timeout_seconds = 3.0
    allowed_extensions = {".txt", ".md", ".csv", ".json"}

    def __init__(self, root: str | Path, max_bytes: int = 1_000_000) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._root = Path(root).expanduser().resolve()
        self._max_bytes = max_bytes

    def execute(self, arguments: BaseModel) -> dict[str, Any]:
        parsed = DocumentReaderInput.model_validate(arguments)
        supplied_path = Path(parsed.path).expanduser()
        candidate = supplied_path if supplied_path.is_absolute() else self._root / supplied_path
        resolved = candidate.resolve(strict=True)
        try:
            relative_path = resolved.relative_to(self._root)
        except ValueError as error:
            raise PermissionError("Path is outside the configured document directory") from error

        if not resolved.is_file():
            raise ValueError("Path is not a regular file")
        if resolved.suffix.lower() not in self.allowed_extensions:
            raise ValueError("File type is not allowed")
        size = resolved.stat().st_size
        if size > self._max_bytes:
            raise ValueError("Document exceeds the configured size limit")
        content = resolved.read_text(encoding="utf-8", errors="strict")
        return {"path": relative_path.as_posix(), "content": content, "size_bytes": size}