"""Các schema kiểm tra kiểu cho từng record trong manifest dữ liệu."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        if self.center_x - half_width < 0 or self.center_x + half_width > 1:
            raise ValueError("bounding box vượt khỏi ảnh theo chiều ngang")
        if self.center_y - half_height < 0 or self.center_y + half_height > 1:
            raise ValueError("bounding box vượt khỏi ảnh theo chiều dọc")
        return self


class DetectionAnnotation(BaseModel):
    """Gắn một bounding box hợp lệ với class biển số duy nhất của dự án."""

    model_config = ConfigDict(extra="forbid")

    class_id: Literal[0] = 0
    class_name: Literal["license_plate"] = "license_plate"
    bbox: YoloBox
