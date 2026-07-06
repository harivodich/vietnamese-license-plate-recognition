"""Xuất project split detection thành cấu hình và image list cho Ultralytics."""

import logging
import os
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import DetectionManifestRecord
from vlpr.data.source_status import build_parser
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
_DETECTION_DIRECTORY = "License Plate Detection Dataset"
_SPLIT_FILE_NAMES = {
    "train": "train.txt",
    "validation": "validation.txt",
    "test": "test.txt",
}


def export_yolo_detection_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> Path:
    """Materialize ảnh-label processed rồi ghi image lists và dataset YAML."""
    records = read_manifest(manifest_path)
    typed_records: list[DetectionManifestRecord] = []
    for record in records:
        if not isinstance(record, DetectionManifestRecord):
            raise ValueError("detection manifest chứa record không thuộc task detection")
        split = record.split
        if split is None:
            raise ValueError(f"record chưa được gán split: {record.sample_id}")
        typed_records.append(record)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    paths_by_split: dict[str, list[str]] = {split: [] for split in _SPLIT_FILE_NAMES}
    try:
        for record in typed_records:
            source_image = dataset_root / record.image_path
            source_label = _label_path(source_image)
            if not source_image.is_file():
                raise FileNotFoundError(f"thiếu ảnh detection: {source_image}")
            if not source_label.is_file():
                raise FileNotFoundError(f"thiếu label detection: {source_label}")
            split = record.split
            if split is None:
                raise ValueError(f"record chưa được gán split: {record.sample_id}")

            image_name = f"{record.sha256}{source_image.suffix.lower()}"
            staged_image = staging_dir / "images" / split / image_name
            staged_label = staging_dir / "labels" / split / f"{record.sha256}.txt"
            staged_image.parent.mkdir(parents=True, exist_ok=True)
            staged_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, staged_image)
            staged_label.write_text(
                _serialize_annotations(record),
                encoding="utf-8",
                newline="\n",
            )
            final_image = output_dir / "images" / split / image_name
            paths_by_split[split].append(final_image.resolve().as_posix())

        for split_name, file_name in _SPLIT_FILE_NAMES.items():
            _atomic_write_text(
                staging_dir / file_name,
                "".join(f"{path}\n" for path in paths_by_split[split_name]),
            )

        yaml_content = yaml.safe_dump(
            {
                "path": output_dir.resolve().as_posix(),
                "train": _SPLIT_FILE_NAMES["train"],
                "val": _SPLIT_FILE_NAMES["validation"],
                "test": _SPLIT_FILE_NAMES["test"],
                "names": {0: "license_plate"},
            },
            allow_unicode=True,
            sort_keys=False,
        )
        _atomic_write_text(staging_dir / "dataset.yaml", yaml_content)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return output_dir / "dataset.yaml"


def export_yolo_from_config(config_path: Path) -> Path:
    """Suy ra mọi đường dẫn từ config và xuất detection dataset."""
    config = load_config(config_path)
    root = project_root(config_path)
    detection = config.dataset("detection")
    dataset_root = resolve_project_path(root, detection.raw_dir) / _DETECTION_DIRECTORY
    manifest_path = resolve_project_path(root, config.split.output_dir) / "detection_manifest.jsonl"
    output_dir = resolve_project_path(root, config.split.detection_yolo_dir)
    return export_yolo_detection_dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )


def _label_path(image_path: Path) -> Path:
    """Đổi đúng segment images thành labels và phần mở rộng thành txt."""
    parts = list(image_path.parts)
    image_indexes = [index for index, part in enumerate(parts) if part == "images"]
    if len(image_indexes) != 1:
        raise ValueError(f"đường dẫn ảnh cần đúng một segment 'images': {image_path}")
    parts[image_indexes[0]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _serialize_annotations(record: DetectionManifestRecord) -> str:
    """Đổi typed annotations thành nội dung YOLO label ổn định."""
    lines = []
    for annotation in record.annotations:
        bbox = annotation.bbox
        lines.append(
            f"{annotation.class_id} {bbox.center_x:.8f} {bbox.center_y:.8f} "
            f"{bbox.width:.8f} {bbox.height:.8f}\n"
        )
    return "".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish text bằng file tạm cùng thư mục để tránh artifact viết dở."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Xuất YOLO dataset và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = build_parser(__doc__ or "Export YOLO detection dataset").parse_args(argv)
    try:
        dataset_yaml = export_yolo_from_config(args.config)
        counts = Counter(
            record.split
            for record in read_manifest(
                resolve_project_path(
                    project_root(args.config),
                    load_config(args.config).split.output_dir,
                )
                / "detection_manifest.jsonl"
            )
        )
    except (KeyError, OSError, ValueError) as exc:
        LOGGER.error("YOLO dataset export failed: %s", exc)
        return 1

    LOGGER.info(
        "YOLO dataset exported path=%s train=%d validation=%d test=%d",
        dataset_yaml,
        counts["train"],
        counts["validation"],
        counts["test"],
    )
    return 0
