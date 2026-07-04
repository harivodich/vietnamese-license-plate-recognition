"""Kiểm thử chọn mẫu và render visualization manual review."""

from pathlib import Path

from PIL import Image

from vlpr.data.manifest_schema import (
    DetectionAnnotation,
    DetectionManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
    YoloBox,
)
from vlpr.data.review import (
    render_detection_review,
    render_ocr_review,
    select_review_records,
)


def _detection_record(name: str) -> DetectionManifestRecord:
    """Tạo detection record dùng chung cho selection và render tests."""
    return DetectionManifestRecord(
        sample_id=f"detection:{name}",
        dataset_name="detection",
        task="detection",
        image_path=f"images/train/{name}.jpg",
        source_split="train",
        width=100,
        height=50,
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        annotations=(
            DetectionAnnotation(
                bbox=YoloBox(
                    center_x=0.5,
                    center_y=0.5,
                    width=0.4,
                    height=0.4,
                )
            ),
        ),
    )


def _ocr_record() -> OcrManifestRecord:
    """Tạo OCR record có Unicode để kiểm tra label visualization."""
    return OcrManifestRecord(
        sample_id="ocr:a",
        dataset_name="ocr",
        task="ocr",
        image_path="imgs/train/a.jpg",
        source_split="train",
        width=40,
        height=20,
        sha256="c" * 64,
        perceptual_hash="d" * 16,
        annotation=OcrAnnotation(raw_text="60MĐ1 01835"),
    )


def test_select_review_records_prioritizes_findings_then_fills_deterministically() -> None:
    """Xác nhận priority luôn đứng trước random sample và cùng seed cho cùng output."""
    records = tuple(_detection_record(name) for name in ("a", "b", "c", "d"))
    priority_paths = ("images/train/c.jpg",)
    reasons_by_path = {"images/train/c.jpg": ("annotation_conflict",)}

    first = select_review_records(
        records,
        sample_size=3,
        priority_paths=priority_paths,
        reasons_by_path=reasons_by_path,
        seed=42,
    )
    second = select_review_records(
        records,
        sample_size=3,
        priority_paths=priority_paths,
        reasons_by_path=reasons_by_path,
        seed=42,
    )

    assert first == second
    assert first[0].record.image_path == "images/train/c.jpg"
    assert first[0].reasons == ("annotation_conflict",)


def test_render_detection_review_draws_bbox(tmp_path: Path) -> None:
    """Xác nhận visualization giữ kích thước và vẽ viền đỏ tại bbox."""
    source = tmp_path / "source.png"
    output = tmp_path / "review.jpg"
    Image.new("RGB", (100, 50), color="white").save(source)

    render_detection_review(source, _detection_record("a"), output)

    with Image.open(output) as rendered:
        assert rendered.size == (100, 50)
        red, green, blue = rendered.convert("RGB").getpixel((30, 15))
        assert red > green + blue


def test_render_ocr_review_adds_unicode_label_strip(tmp_path: Path) -> None:
    """Xác nhận OCR crop được phóng và có vùng label bên dưới."""
    source = tmp_path / "source.png"
    output = tmp_path / "review.jpg"
    Image.new("RGB", (40, 20), color="gray").save(source)

    render_ocr_review(source, _ocr_record(), output)

    with Image.open(output) as rendered:
        assert rendered.width >= 180
        assert rendered.height > 20
