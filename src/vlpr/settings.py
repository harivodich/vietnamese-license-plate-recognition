"""Cấu hình dùng chung cho quá trình huấn luyện và suy luận."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectSettings(BaseModel):
    """Chứa định danh ổn định của dự án và seed dùng để tái lập thí nghiệm."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)


class TrackingSettings(BaseModel):
    """Quy định nhà cung cấp và chế độ theo dõi thí nghiệm."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["wandb"]
    project: str = Field(min_length=1)
    mode: Literal["auto", "online", "offline", "disabled"] = "auto"


class RuntimeSettings(BaseModel):
    """Gom các nhóm cấu hình runtime được dùng xuyên suốt các gate."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectSettings
    tracking: TrackingSettings


def load_runtime_settings(path: Path) -> RuntimeSettings:
    """Đọc YAML runtime và chuyển thành object đã được Pydantic kiểm tra kiểu."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return RuntimeSettings.model_validate(raw)
