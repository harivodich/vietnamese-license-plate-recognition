"""Định nghĩa và kiểm tra kiểu cho cấu hình dataset của dự án."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSettings(BaseModel):
    """Mô tả một nguồn dataset cố định gồm định danh, phiên bản và đường dẫn lưu trữ."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    version: int = Field(ge=1)
    country: str = Field(min_length=2, max_length=2)
    task: str = Field(pattern=r"^(detection|ocr)$")
    expected_license: str = Field(min_length=1)
    raw_dir: Path
    manifest_path: Path

    @property
    def versioned_handle(self) -> str:
        """Tạo Kaggle handle có version để luôn tải đúng một phiên bản dữ liệu."""
        return f"{self.handle}/versions/{self.version}"


class ValidationSettings(BaseModel):
    """Chứa các ngưỡng dùng khi kiểm định chất lượng và duplicate của dataset."""

    model_config = ConfigDict(extra="forbid")

    image_extensions: tuple[str, ...]
    near_duplicate_hamming_distance: int = Field(ge=0, le=63)
    manual_review_sample_size: int = Field(ge=100)
    report_path: Path = Path("data/interim/dataset_audit.json")
    review_dir: Path = Path("data/interim/manual_review")


class SplitSettings(BaseModel):
    """Chứa tỷ lệ chia tập và seed để phép chia dữ liệu có thể tái lập."""

    model_config = ConfigDict(extra="forbid")

    train: float = Field(gt=0.0, lt=1.0)
    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitSettings":
        """Từ chối cấu hình nếu tổng tỷ lệ train, validation và test không bằng 1."""
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class ProjectConfig(BaseModel):
    """Gom registry dataset, quy tắc validation và cấu hình split của dự án."""

    model_config = ConfigDict(extra="forbid")

    datasets: dict[str, DatasetSettings]
    validation: ValidationSettings
    split: SplitSettings

    @model_validator(mode="after")
    def validate_dataset_registry(self) -> "ProjectConfig":
        """Bảo đảm cấu hình luôn có đủ hai nguồn detection và OCR bắt buộc."""
        required = {"detection", "ocr"}
        missing = required.difference(self.datasets)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing required dataset entries: {names}")
        return self

    def dataset(self, name: str) -> DatasetSettings:
        """Lấy cấu hình dataset theo tên logic và báo rõ các tên hợp lệ nếu không tồn tại."""
        try:
            return self.datasets[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset '{name}', expected one of: {known}") from exc


def load_config(path: Path) -> ProjectConfig:
    """Đọc YAML và kiểm tra toàn bộ schema trước khi thao tác lên dữ liệu."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return ProjectConfig.model_validate(raw)


def project_root(config_path: Path) -> Path:
    """Suy ra thư mục gốc repository từ file cấu hình nằm trong thư mục ``configs``."""
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError(f"config must be located in a configs directory: {resolved}")
    return resolved.parent.parent


def resolve_project_path(root: Path, configured_path: Path) -> Path:
    """Đổi đường dẫn cấu hình thành tuyệt đối và chặn đường dẫn thoát khỏi repository."""
    candidate = (root / configured_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes project root: {configured_path}")
    return candidate
