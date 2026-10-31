"""Load a skill's validated metadata and instructions from disk."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    version: str = Field(min_length=1, max_length=40)
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True


class SkillLoadError(ValueError):
    """Raised when a skill directory is incomplete or invalid."""


class TextTokenizer(Protocol):
    def count_text(self, text: str) -> int: ...


@dataclass(frozen=True)
class LoadedSkill:
    metadata: SkillMetadata
    instructions: str
    directory: Path
    estimated_instruction_tokens: int | None


class SkillLoader:
    def __init__(self, tokenizer: TextTokenizer | None = None) -> None:
        self._tokenizer = tokenizer

    def load(self, directory: str | Path) -> LoadedSkill:
        skill_directory = Path(directory).resolve(strict=True)
        metadata_path = _contained_file(skill_directory, "metadata.json")
        instructions_path = _contained_file(skill_directory, "SKILL.md")
        try:
            metadata_data: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = SkillMetadata.model_validate(metadata_data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillLoadError(f"Invalid skill metadata in {skill_directory.name}") from error
        if metadata.name != skill_directory.name:
            raise SkillLoadError("Skill metadata name must match its directory name")
        try:
            instructions = instructions_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise SkillLoadError(f"Could not read instructions for {metadata.name}") from error
        if not instructions:
            raise SkillLoadError(f"Skill instructions are empty: {metadata.name}")
        token_estimate = self._tokenizer.count_text(instructions) if self._tokenizer else None
        return LoadedSkill(
            metadata=metadata,
            instructions=instructions,
            directory=skill_directory,
            estimated_instruction_tokens=token_estimate,
        )


def _contained_file(directory: Path, filename: str) -> Path:
    candidate = (directory / filename).resolve(strict=True)
    try:
        candidate.relative_to(directory)
    except ValueError as error:
        raise SkillLoadError(f"Skill file escapes its directory: {filename}") from error
    if not candidate.is_file():
        raise SkillLoadError(f"Required skill file is not a regular file: {filename}")
    return candidate