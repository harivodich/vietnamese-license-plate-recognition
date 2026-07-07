"""Schema cấu hình dùng chung cho chuẩn bị dữ liệu và train CRNN+CTC."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OcrTrainingDataSettings(BaseModel):
    """Đường dẫn dữ liệu và quy tắc biến compact crop thành hai text lines."""

    model_config = ConfigDict(extra="forbid")

    manifest: Path
    dataset_root: Path
    output_dir: Path
    compact_aspect_ratio: float = Field(gt=0.0)
    split_search_start: float = Field(gt=0.0, lt=1.0)
    split_search_end: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_search_range(self) -> "OcrTrainingDataSettings":
        """Giữ vùng tìm khe giữa hai dòng theo đúng thứ tự và quanh trung tâm."""
        if self.split_search_start >= self.split_search_end:
            raise ValueError("split_search_start phải nhỏ hơn split_search_end")
        return self


class OcrCrnnModelSettings(BaseModel):
    """Kích thước tensor và capacity của CRNN."""

    model_config = ConfigDict(extra="forbid")

    image_height: Literal[32]
    image_width: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    lstm_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    blank_bias: float = Field(le=0.0)


class OcrAugmentationSettings(BaseModel):
    """Biến đổi nhẹ cho ảnh dòng, không làm sai cấu trúc ký tự biển số."""

    model_config = ConfigDict(extra="forbid")

    rotation_degrees: float = Field(ge=0.0, le=5.0)
    brightness: float = Field(ge=0.0, lt=1.0)
    contrast: float = Field(ge=0.0, lt=1.0)
    blur_probability: float = Field(ge=0.0, le=1.0)


class OcrOptimizationSettings(BaseModel):
    """Hyperparameter tối ưu và chính sách checkpoint cho một training run."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    min_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    workers: int = Field(ge=0)
    device: str = Field(min_length=1)
    seed: int = Field(ge=0)
    deterministic: bool
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    patience: int = Field(ge=0)
    save_period: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_epoch_policy(self) -> "OcrOptimizationSettings":
        """Không cho early stopping chạy trước tổng số epoch hợp lệ."""
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs không được lớn hơn epochs")
        return self


class OcrOutputSettings(BaseModel):
    """Vị trí artifact được Git ignore và tên ổn định của experiment."""

    model_config = ConfigDict(extra="forbid")

    project: Path
    name: str = Field(min_length=1)


class OcrTrainingExperimentConfig(BaseModel):
    """Toàn bộ contract để tái tạo dataset và training CRNN."""

    model_config = ConfigDict(extra="forbid")

    data: OcrTrainingDataSettings
    model: OcrCrnnModelSettings
    augmentation: OcrAugmentationSettings
    train: OcrOptimizationSettings
    output: OcrOutputSettings


def load_ocr_training_config(path: Path) -> OcrTrainingExperimentConfig:
    """Đọc strict YAML để key sai không âm thầm đổi thí nghiệm."""
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"OCR training config root phải là mapping: {path}")
    return OcrTrainingExperimentConfig.model_validate(raw)
