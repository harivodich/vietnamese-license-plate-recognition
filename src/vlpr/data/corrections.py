"""Đọc và áp dụng các sửa lỗi nhãn đã được ghi nhận mà không đổi dữ liệu raw."""

import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vlpr.data.ocr_parser import OcrLabel


class OcrCorrection(BaseModel):
    """Mô tả một thay đổi OCR có nhãn gốc làm điều kiện an toàn."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    corrected_text: str | None = None
    exclude: bool = False
    reason: str = Field(min_length=1)
    review_method: Literal["visual_review"]

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        """Chỉ nhận đường dẫn POSIX tương đối giống đường dẫn trong OCR label."""
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path phải là đường dẫn POSIX tương đối")
        return value

    @model_validator(mode="after")
    def validate_action(self) -> "OcrCorrection":
        """Yêu cầu đúng một hành động: sửa text hoặc loại mẫu."""
        if self.exclude == (self.corrected_text is not None):
            raise ValueError("correction phải sửa text hoặc exclude, không được đồng thời")
        return self


class CorrectionError(ValueError):
    """Báo correction hỏng, trùng, không còn khớp hoặc không được sử dụng."""


def read_ocr_corrections(path: Path) -> dict[str, OcrCorrection]:
    """Đọc JSONL và lập chỉ mục correction duy nhất theo image_path."""
    corrections: dict[str, OcrCorrection] = {}
    file_path = path
    for line_number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise CorrectionError(f"{file_path}:{line_number}: dòng correction rỗng")
        try:
            correction = OcrCorrection.model_validate_json(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise CorrectionError(f"{file_path}:{line_number}: correction không hợp lệ") from exc
        if correction.image_path in corrections:
            raise CorrectionError(
                f"{file_path}:{line_number}: image_path correction bị trùng: "
                f"{correction.image_path}"
            )
        corrections[correction.image_path] = correction
    return corrections


def apply_ocr_corrections(
    labels: tuple[OcrLabel, ...],
    corrections: dict[str, OcrCorrection],
) -> tuple[OcrLabel, ...]:
    """Áp dụng correction và từ chối mục không khớp đúng nhãn nguồn."""
    corrected_labels: list[OcrLabel] = []
    unused_paths = set(corrections)
    for label in labels:
        image_path = label.image_path.as_posix()
        correction = corrections.get(image_path)
        if correction is None:
            corrected_labels.append(label)
            continue
        if label.text != correction.original_text:
            raise CorrectionError(
                f"correction {image_path} dự kiến nhãn {correction.original_text!r}, "
                f"nhưng nguồn hiện tại là {label.text!r}"
            )
        if not correction.exclude:
            if correction.corrected_text is None:
                raise CorrectionError(f"correction {image_path} thiếu corrected_text")
            corrected_labels.append(
                OcrLabel(
                    image_path=label.image_path,
                    text=correction.corrected_text,
                )
            )
        unused_paths.remove(image_path)

    if unused_paths:
        paths = ", ".join(sorted(unused_paths))
        raise CorrectionError(f"correction không khớp ảnh nào trong nguồn: {paths}")
    return tuple(corrected_labels)
