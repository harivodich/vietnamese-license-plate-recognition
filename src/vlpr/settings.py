"""Application-wide settings shared by training and inference."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectSettings(BaseModel):
    """Stable project identity and reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)


class TrackingSettings(BaseModel):
    """Experiment-tracking policy."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["wandb"]
    project: str = Field(min_length=1)
    mode: Literal["auto", "online", "offline", "disabled"] = "auto"


class RuntimeSettings(BaseModel):
    """Root configuration used across project gates."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectSettings
    tracking: TrackingSettings


def load_runtime_settings(path: Path) -> RuntimeSettings:
    """Load typed runtime settings from YAML."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return RuntimeSettings.model_validate(raw)
