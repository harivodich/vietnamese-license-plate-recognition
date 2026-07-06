"""Kiểm thử phép tính EDA detection trước khi vẽ biểu đồ."""

from typing import Literal

import pytest

from vlpr.data.detection_eda import collect_detection_eda
from vlpr.data.manifest_schema import (
    DetectionAnnotation,
    DetectionManifestRecord,
    YoloBox,
)


def _record(
    name: str,
    split: Literal["train", "validation", "test"],
    bbox: YoloBox,
) -> DetectionManifestRecord:
    """Tạo record có bbox và project split do test kiểm soát."""
    return DetectionManifestRecord(
        sample_id=f"detection:{name}",
        dataset_name="detection",
        task="detection",
        image_path=f"images/{name}.jpg",
        source_split="source",
        split=split,
        width=640,
        height=640,
        sha256=f"{len(name):064x}",
        perceptual_hash="0" * 16,
        annotations=(DetectionAnnotation(bbox=bbox),),
    )


def test_collect_detection_eda_counts_classes_splits_and_sizes() -> None:
    """Xác nhận EDA dùng project split và phân nhóm diện tích sau resize."""
    records = (
        _record(
            "small",
            "train",
            YoloBox(center_x=0.5, center_y=0.5, width=0.02, height=0.02),
        ),
        _record(
            "medium",
            "validation",
            YoloBox(center_x=0.4, center_y=0.6, width=0.1, height=0.1),
        ),
        _record(
            "large",
            "test",
            YoloBox(center_x=0.3, center_y=0.7, width=0.2, height=0.2),
        ),
    )

    data = collect_detection_eda(records)

    assert data.class_counts == {"license_plate": 3}
    assert data.image_counts == {"train": 1, "validation": 1, "test": 1}
    assert data.size_counts == {"small": 1, "medium": 1, "large": 1}
    assert data.aspect_ratios == (1.0, 1.0, 1.0)
    assert data.sizes_by_split["train"] == {"small": 1}
    assert data.areas_by_split["test"] == pytest.approx((0.04,))
