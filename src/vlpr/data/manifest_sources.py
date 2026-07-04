"""Duyệt cấu trúc hai nguồn Kaggle và tạo manifest record theo thứ tự ổn định."""

from collections.abc import Iterator
from pathlib import Path

from vlpr.data.manifest import build_detection_record, build_ocr_record
from vlpr.data.manifest_schema import DetectionManifestRecord, OcrManifestRecord
from vlpr.data.ocr_parser import OcrLabel, parse_ocr_file

_DETECTION_DIRECTORY = "License Plate Detection Dataset"
_OCR_DIRECTORY = "lp_ocr_dataset_vi"


class DatasetStructureError(ValueError):
    """Báo cấu trúc ảnh và annotation nguồn không khớp."""


def iter_detection_records(
    raw_root: Path,
    *,
    dataset_name: str,
    image_extensions: tuple[str, ...],
) -> Iterator[DetectionManifestRecord]:
    """Duyệt các split detection và tạo record sau khi xác nhận pairing."""
    dataset_root = raw_root / _DETECTION_DIRECTORY
    normalized_extensions = {extension.lower() for extension in image_extensions}
    for source_split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / source_split
        label_dir = dataset_root / "labels" / source_split
        images = _index_by_stem(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in normalized_extensions
            ),
            source=f"detection images/{source_split}",
        )
        labels = _index_by_stem(
            (path for path in label_dir.glob("*.txt") if path.is_file()),
            source=f"detection labels/{source_split}",
        )
        _validate_matching_stems(images, labels, source_split=source_split)

        for stem in sorted(images):
            yield build_detection_record(
                dataset_root=dataset_root,
                image_path=images[stem],
                label_path=labels[stem],
                dataset_name=dataset_name,
                source_split=source_split,
            )


def iter_ocr_records(
    raw_root: Path,
    *,
    dataset_name: str,
    image_extensions: tuple[str, ...],
) -> Iterator[OcrManifestRecord]:
    """Đọc hai label file OCR, kiểm tra pairing rồi tạo record theo thứ tự nguồn."""
    dataset_root = raw_root / _OCR_DIRECTORY
    labels_by_split: list[tuple[str, tuple[OcrLabel, ...]]] = []
    for source_split in ("train", "val"):
        labels_by_split.append(
            (
                source_split,
                parse_ocr_file(dataset_root / "labels" / f"{source_split}.txt"),
            )
        )

    labels = [label for _, split_labels in labels_by_split for label in split_labels]
    referenced_paths = [label.image_path.as_posix() for label in labels]
    if len(referenced_paths) != len(set(referenced_paths)):
        raise DatasetStructureError("OCR label files tham chiếu trùng đường dẫn ảnh")

    normalized_extensions = {extension.lower() for extension in image_extensions}
    actual_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "imgs").rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    }
    _validate_matching_paths(set(referenced_paths), actual_paths)

    for source_split, split_labels in labels_by_split:
        for label in split_labels:
            yield build_ocr_record(
                dataset_root=dataset_root,
                label=label,
                dataset_name=dataset_name,
                source_split=source_split,
            )


def _index_by_stem(paths: Iterator[Path], *, source: str) -> dict[str, Path]:
    """Lập chỉ mục stem và từ chối hai file cùng stem gây pairing mơ hồ."""
    index: dict[str, Path] = {}
    for path in paths:
        if path.stem in index:
            raise DatasetStructureError(f"{source} có stem trùng: {path.stem}")
        index[path.stem] = path
    return index


def _validate_matching_stems(
    images: dict[str, Path],
    labels: dict[str, Path],
    *,
    source_split: str,
) -> None:
    """Báo stem thiếu ở một trong hai phía detection."""
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise DatasetStructureError(
            f"detection/{source_split} pairing lỗi: "
            f"thiếu label={missing_labels[:5]}, thiếu ảnh={missing_images[:5]}"
        )


def _validate_matching_paths(referenced: set[str], actual: set[str]) -> None:
    """Báo ảnh OCR thiếu hoặc không được label file tham chiếu."""
    missing_images = sorted(referenced - actual)
    unreferenced_images = sorted(actual - referenced)
    if missing_images or unreferenced_images:
        raise DatasetStructureError(
            "OCR pairing lỗi: "
            f"thiếu ảnh={missing_images[:5]}, ảnh không nhãn={unreferenced_images[:5]}"
        )
