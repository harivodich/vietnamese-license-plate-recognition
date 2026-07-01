"""Typed project configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSettings(BaseModel):
    """Source dataset settings."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    version: int = Field(ge=1)
    country: str = Field(min_length=2, max_length=2)
    expected_license: str = Field(min_length=1)
    raw_dir: Path
    manifest_path: Path

    @property
    def versioned_handle(self) -> str:
        """Return an immutable Kaggle dataset handle."""
        return f"{self.handle}/versions/{self.version}"


class ValidationSettings(BaseModel):
    """Dataset quality thresholds."""

    model_config = ConfigDict(extra="forbid")

    image_extensions: tuple[str, ...]
    near_duplicate_hamming_distance: int = Field(ge=0, le=64)
    manual_review_sample_size: int = Field(ge=100)


class SplitSettings(BaseModel):
    """Group-aware split ratios and reproducibility seed."""

    model_config = ConfigDict(extra="forbid")

    train: float = Field(gt=0.0, lt=1.0)
    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitSettings":
        """Reject ratios that do not form a complete partition."""
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class ProjectConfig(BaseModel):
    """Root dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSettings
    validation: ValidationSettings
    split: SplitSettings


def load_config(path: Path) -> ProjectConfig:
    """Load and validate YAML before any data mutation."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return ProjectConfig.model_validate(raw)


def project_root(config_path: Path) -> Path:
    """Resolve the repository root from a config stored under ``configs/``."""
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError(f"config must be located in a configs directory: {resolved}")
    return resolved.parent.parent


def resolve_project_path(root: Path, configured_path: Path) -> Path:
    """Resolve a configured path and prevent accidental writes outside the repository."""
    candidate = (root / configured_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes project root: {configured_path}")
    return candidate
