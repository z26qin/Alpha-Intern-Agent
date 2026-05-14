"""Project-wide configuration for AlphaInternAgent.

Keep this intentionally small. Settings live in environment variables
or sensible defaults; we do not read secrets from disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_data_dir() -> Path:
    return Path(os.environ.get("ALPHA_INTERN_DATA_DIR", "./.alpha_intern"))


class AlphaInternSettings(BaseModel):
    """Runtime settings for AlphaInternAgent."""

    data_dir: Path = Field(default_factory=_default_data_dir)
    memory_file: str = "memory.jsonl"
    skills_file: str = "skills.json"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / self.memory_file

    @property
    def skills_path(self) -> Path:
        return self.data_dir / self.skills_file

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> AlphaInternSettings:
    """Return a fresh settings object based on current environment."""
    return AlphaInternSettings()
