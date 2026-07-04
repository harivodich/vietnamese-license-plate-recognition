"""Các schema kiểm tra kiểu cho annotation và record trong manifest dữ liệu."""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_YOLO_BOUNDARY_TOLERANCE = 1e-6


class YoloBox(BaseModel):
    """Biểu diễn bounding box YOLO bằng tâm và kích thước đã chuẩn hóa."""

    model_config = ConfigDict(extra="forbid")

    center_x: float = Field(ge=0.0, le=1.0)
    center_y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_inside_image(self) -> "YoloBox":
        """Từ chối bbox có cạnh vượt ra ngoài vùng ảnh chuẩn hóa từ 0 đến 1."""
        half_width = self.width / 2
        half_height = self.height / 2
        if (
            self.center_x - half_width < -_YOLO_BOUNDARY_TOLERANCE
            or self.center_x + half_width > 1 + _YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError("bounding box vượt khỏi ảnh theo chiều ngang")
        if (
            self.center_y - half_height < -_YOLO_BOUNDARY_TOLERANCE
            or self.center_y + half_height > 1 + _YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError("bounding box vượt khỏi ảnh theo chiều dọc")
        return self


class DetectionAnnotation(BaseModel):
    """Gắn một bounding box hợp lệ với class biển số duy nhất của dự án."""

    model_config = ConfigDict(extra="forbid")

    class_id: Literal[0] = 0
    class_name: Literal["license_plate"] = "license_plate"
    bbox: YoloBox


class OcrAnnotation(BaseModel):
    """Lưu nguyên văn nội dung biển số từ nguồn OCR, chưa chuẩn hóa."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str

    @field_validator("raw_text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Từ chối nhãn OCR chỉ chứa khoảng trắng nhưng không sửa raw text."""
        if not value.strip():
            raise ValueError("raw_text không được rỗng")
        return value


class _ManifestRecordBase(BaseModel):
    """Định nghĩa metadata chung của mọi ảnh trong manifest."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    group_id: str | None = None
    split: Literal["train", "validation", "test"] | None = None
    validation_status: Literal["valid", "warning", "invalid"] = "valid"

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        """Chỉ chấp nhận đường dẫn POSIX tương đối nằm bên trong dataset."""
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path phải là đường dẫn POSIX tương đối")
        return value


class DetectionManifestRecord(_ManifestRecordBase):
    """Mô tả một ảnh detection cùng toàn bộ bounding box của ảnh."""

    task: Literal["detection"]
    annotations: tuple[DetectionAnnotation, ...]


class OcrManifestRecord(_ManifestRecordBase):
    """Mô tả một ảnh crop OCR cùng nội dung biển số nguyên bản."""

    task: Literal["ocr"]
    annotation: OcrAnnotation


ManifestRecord = Annotated[
    DetectionManifestRecord | OcrManifestRecord,
    Field(discriminator="task"),
]
