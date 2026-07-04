"""Tạo manifest record từ ảnh và annotation nguồn đã được kiểm tra."""

from pathlib import Path

from vlpr.data.detection_parser import parse_yolo_file
from vlpr.data.duplicates import fingerprint_image
from vlpr.data.image_validation import decode_image
from vlpr.data.manifest_schema import (
    DetectionManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
)
from vlpr.data.ocr_parser import OcrLabel


def build_detection_record(
    *,
    dataset_root: Path,
    image_path: Path,
    label_path: Path,
    dataset_name: str,
    source_split: str,
) -> DetectionManifestRecord:
    """Tạo một record detection từ ảnh và file nhãn YOLO tương ứng."""
    relative_path = _relative_image_path(dataset_root, image_path)
    metadata = decode_image(image_path)
    fingerprint = fingerprint_image(image_path)
    return DetectionManifestRecord(
        sample_id=_sample_id(dataset_name, relative_path),
        dataset_name=dataset_name,
        task="detection",
        image_path=relative_path,
        source_split=source_split,
        width=metadata.width,
        height=metadata.height,
        sha256=fingerprint.sha256,
        perceptual_hash=fingerprint.perceptual_hash,
        annotations=parse_yolo_file(label_path),
    )


def build_ocr_record(
    *,
    dataset_root: Path,
    label: OcrLabel,
    dataset_name: str,
    source_split: str,
) -> OcrManifestRecord:
    """Tạo một record OCR từ entry nhãn và ảnh crop tương ứng."""
    image_path = dataset_root / label.image_path
    relative_path = _relative_image_path(dataset_root, image_path)
    metadata = decode_image(image_path)
    fingerprint = fingerprint_image(image_path)
    return OcrManifestRecord(
        sample_id=_sample_id(dataset_name, relative_path),
        dataset_name=dataset_name,
        task="ocr",
        image_path=relative_path,
        source_split=source_split,
        width=metadata.width,
        height=metadata.height,
        sha256=fingerprint.sha256,
        perceptual_hash=fingerprint.perceptual_hash,
        annotation=OcrAnnotation(raw_text=label.text),
    )


def _relative_image_path(dataset_root: Path, image_path: Path) -> str:
    """Đổi path tuyệt đối thành POSIX path tương đối và chặn thoát dataset."""
    try:
        return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"ảnh nằm ngoài dataset root: {image_path}") from exc


def _sample_id(dataset_name: str, image_path: str) -> str:
    """Tạo định danh ổn định từ nguồn logic và đường dẫn tương đối."""
    return f"{dataset_name}:{image_path}"
