"""Discover and load enabled skills from a configured directory."""

import json
import os
from pathlib import Path
from typing import Any

from skills.skill_loader import LoadedSkill, SkillLoader


class SkillNotFoundError(LookupError):
    """Raised when a skill is absent or disabled."""


class SkillRegistry:
    def __init__(self, root: str | Path, loader: SkillLoader | None = None) -> None:
        self._root = Path(root).expanduser().resolve()
        self._loader = loader or SkillLoader()

    def discover(self, include_disabled: bool = False) -> list[LoadedSkill]:
        if not self._root.is_dir():
            return []
        skills: list[LoadedSkill] = []
        for directory in sorted(self._root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir():
                continue
            try:
                directory.resolve(strict=True).relative_to(self._root)
            except ValueError:
                continue
            skill = self._loader.load(directory)
            if include_disabled or skill.metadata.enabled:
                skills.append(skill)
        return skills

    def load(self, name: str) -> LoadedSkill:
        directory = (self._root / name).resolve()
        try:
            directory.relative_to(self._root)
        except ValueError as error:
            raise SkillNotFoundError(f"Skill is not available: {name}") from error
        try:
            skill = self._loader.load(directory)
        except (OSError, SkillNotFoundError) as error:
            raise SkillNotFoundError(f"Skill is not available: {name}") from error
        if skill.metadata.name != name or not skill.metadata.enabled:
            raise SkillNotFoundError(f"Skill is not available: {name}")
        return skill

    def load_many(self, names: list[str]) -> list[LoadedSkill]:
        return [self.load(name) for name in dict.fromkeys(names)]

    def set_enabled(self, name: str, enabled: bool) -> LoadedSkill:
        directory = (self._root / name).resolve(strict=True)
        try:
            directory.relative_to(self._root)
        except ValueError as error:
            raise SkillNotFoundError(f"Skill is not available: {name}") from error

        skill = self._loader.load(directory)
        if skill.metadata.name != name:
            raise SkillNotFoundError(f"Skill is not available: {name}")
        metadata_path = directory / "metadata.json"
        metadata_data: dict[str, Any] = skill.metadata.model_dump(mode="json")
        metadata_data["enabled"] = enabled
        temporary_path = metadata_path.with_suffix(".json.tmp")
        try:
            temporary_path.write_text(
                json.dumps(metadata_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, metadata_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return self._loader.load(directory)