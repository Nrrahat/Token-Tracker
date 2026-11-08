"""Persistent storage for custom tool source and metadata."""

import json
import keyword
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_MAX_SOURCE_CHARS = 32_000


class CustomToolStorageError(RuntimeError):
    """Raised when custom tool files or registry metadata cannot be read or written."""


class CustomToolAlreadyExistsError(ValueError):
    """Raised when creating a custom tool with an existing name."""


class CustomToolMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(max_length=2000)
    function_name: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True)
class StoredCustomTool:
    metadata: CustomToolMetadata
    source: str


class CustomToolStore:
    """Store custom tool source files and their persistent JSON registry."""

    _thread_lock = RLock()
    _registry_version = 1

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_path = root / "tools.json"

    def create(
        self,
        name: str,
        description: str,
        function_name: str,
        source: str,
    ) -> CustomToolMetadata:
        """Persist a new tool without importing or executing its Python source."""
        try:
            metadata = CustomToolMetadata(
                name=name,
                description=description,
                function_name=function_name,
            )
        except ValidationError as error:
            raise ValueError(f"Invalid custom tool metadata: {error}") from error
        self._validate_source(metadata.function_name, source)

        with self._thread_lock:
            tools = self._read_registry()
            source_path = self.root / f"{metadata.name}.py"
            if metadata.name in tools or source_path.exists():
                raise CustomToolAlreadyExistsError(
                    f"A custom tool named '{metadata.name}' already exists."
                )

            self.root.mkdir(parents=True, exist_ok=True)
            source_temp: Path | None = None
            registry_temp: Path | None = None
            source_published = False
            try:
                source_temp = self._write_temp(source)
                registry_temp = self._write_temp(
                    self._serialize_registry([*tools.values(), metadata])
                )
                os.replace(source_temp, source_path)
                source_temp = None
                source_published = True
                os.replace(registry_temp, self.registry_path)
                registry_temp = None
            except OSError as error:
                if source_published:
                    source_path.unlink(missing_ok=True)
                raise CustomToolStorageError(
                    f"Could not persist custom tool '{metadata.name}'."
                ) from error
            finally:
                if source_temp is not None:
                    source_temp.unlink(missing_ok=True)
                if registry_temp is not None:
                    registry_temp.unlink(missing_ok=True)
        return metadata

    def update(
        self,
        name: str,
        description: str,
        function_name: str,
        source: str,
    ) -> CustomToolMetadata:
        """Update a registered tool's description, function name, and source."""
        try:
            metadata = CustomToolMetadata(
                name=name,
                description=description,
                function_name=function_name,
            )
        except ValidationError as error:
            raise ValueError(f"Invalid custom tool metadata: {error}") from error
        self._validate_source(metadata.function_name, source)

        with self._thread_lock:
            tools = self._read_registry()
            if name not in tools:
                raise LookupError(f"Custom tool is not registered: {name}")
            source_path = self.root / f"{name}.py"
            try:
                if source_path.is_symlink():
                    raise CustomToolStorageError(
                        f"Custom tool source must not be a symbolic link: {name}"
                    )
                old_source = source_path.read_text(encoding="utf-8")
            except CustomToolStorageError:
                raise
            except OSError as error:
                raise CustomToolStorageError(
                    f"Could not read source for custom tool '{name}'."
                ) from error
            except UnicodeDecodeError as error:
                raise CustomToolStorageError(
                    f"Source for custom tool '{name}' is not valid UTF-8."
                ) from error

            source_temp: Path | None = None
            registry_temp: Path | None = None
            try:
                source_temp = self._write_temp(source)
                updated_tools = [metadata if tool.name == name else tool for tool in tools.values()]
                registry_temp = self._write_temp(self._serialize_registry(updated_tools))
                os.replace(source_temp, source_path)
                source_temp = None
                os.replace(registry_temp, self.registry_path)
                registry_temp = None
            except OSError as error:
                if source_temp is None:
                    try:
                        rollback_temp = self._write_temp(old_source)
                        os.replace(rollback_temp, source_path)
                    except OSError as rollback_error:
                        raise CustomToolStorageError(
                            f"Could not update custom tool '{name}' or restore its previous source."
                        ) from rollback_error
                raise CustomToolStorageError(
                    f"Could not update custom tool '{name}'."
                ) from error
            finally:
                if source_temp is not None:
                    source_temp.unlink(missing_ok=True)
                if registry_temp is not None:
                    registry_temp.unlink(missing_ok=True)
        return metadata

    def delete(self, name: str) -> None:
        """Remove a registered custom tool and its source file."""
        with self._thread_lock:
            tools = self._read_registry()
            metadata = tools.get(name)
            if metadata is None:
                raise LookupError(f"Custom tool is not registered: {name}")
            source_path = self.root / f"{metadata.name}.py"
            try:
                if source_path.is_symlink():
                    raise CustomToolStorageError(
                        f"Custom tool source must not be a symbolic link: {name}"
                    )
                old_source = source_path.read_text(encoding="utf-8")
            except CustomToolStorageError:
                raise
            except OSError as error:
                raise CustomToolStorageError(
                    f"Could not read source for custom tool '{name}'."
                ) from error
            except UnicodeDecodeError as error:
                raise CustomToolStorageError(
                    f"Source for custom tool '{name}' is not valid UTF-8."
                ) from error

            source_temp: Path | None = None
            registry_temp: Path | None = None
            source_removed = False
            try:
                source_temp = self._write_temp(old_source)
                registry_temp = self._write_temp(
                    self._serialize_registry(
                        [tool for tool_name, tool in tools.items() if tool_name != name]
                    )
                )
                source_path.unlink()
                source_removed = True
                os.replace(registry_temp, self.registry_path)
                registry_temp = None
            except OSError as error:
                if source_removed and source_temp is not None:
                    try:
                        os.replace(source_temp, source_path)
                        source_temp = None
                    except OSError as rollback_error:
                        raise CustomToolStorageError(
                            f"Could not delete custom tool '{name}' or restore its source."
                        ) from rollback_error
                raise CustomToolStorageError(
                    f"Could not delete custom tool '{name}'."
                ) from error
            finally:
                if source_temp is not None:
                    source_temp.unlink(missing_ok=True)
                if registry_temp is not None:
                    registry_temp.unlink(missing_ok=True)

    def list_tools(self) -> list[CustomToolMetadata]:
        """Return persisted custom tool metadata ordered by name."""
        with self._thread_lock:
            return sorted(self._read_registry().values(), key=lambda tool: tool.name)

    def get(self, name: str) -> StoredCustomTool:
        """Read a registered tool and its source without executing it."""
        with self._thread_lock:
            metadata = self._read_registry().get(name)
            if metadata is None:
                raise LookupError(f"Custom tool is not registered: {name}")
            try:
                source_path = self.root / f"{metadata.name}.py"
                if source_path.is_symlink():
                    raise CustomToolStorageError(
                        f"Custom tool source must not be a symbolic link: {metadata.name}"
                    )
                source = source_path.read_text(encoding="utf-8")
            except CustomToolStorageError:
                raise
            except OSError as error:
                raise CustomToolStorageError(
                    f"Could not read source for custom tool '{metadata.name}'."
                ) from error
            except UnicodeDecodeError as error:
                raise CustomToolStorageError(
                    f"Source for custom tool '{metadata.name}' is not valid UTF-8."
                ) from error
        return StoredCustomTool(metadata=metadata, source=source)

    def _validate_source(self, function_name: str, source: str) -> None:
        if (
            not function_name.isidentifier()
            or keyword.iskeyword(function_name)
            or function_name.startswith("_")
        ):
            raise ValueError(
                "Function name must be a valid, non-keyword Python identifier "
                "that does not start with '_'."
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Custom tool source must not be empty.")
        if len(source) > _MAX_SOURCE_CHARS:
            raise ValueError(f"Custom tool source exceeds {_MAX_SOURCE_CHARS:,} characters.")

    def _read_registry(self) -> dict[str, CustomToolMetadata]:
        if not self.registry_path.exists():
            return {}
        try:
            registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if (
                not isinstance(registry, dict)
                or registry.get("version") != self._registry_version
                or not isinstance(registry.get("tools"), list)
            ):
                raise CustomToolStorageError("Custom tool registry has an unsupported format.")

            tools: dict[str, CustomToolMetadata] = {}
            for entry in registry["tools"]:
                metadata = CustomToolMetadata.model_validate(entry)
                if (
                    not metadata.function_name.isidentifier()
                    or keyword.iskeyword(metadata.function_name)
                    or metadata.function_name.startswith("_")
                    or metadata.name in tools
                ):
                    raise CustomToolStorageError("Custom tool registry contains invalid entries.")
                tools[metadata.name] = metadata
            return tools
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise CustomToolStorageError("Could not read the custom tool registry.") from error

    def _serialize_registry(self, tools: list[CustomToolMetadata]) -> str:
        return json.dumps(
            {
                "version": self._registry_version,
                "tools": [tool.model_dump() for tool in sorted(tools, key=lambda item: item.name)],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _write_temp(self, content: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.root,
            prefix=".custom-tool-",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            return Path(file.name)
