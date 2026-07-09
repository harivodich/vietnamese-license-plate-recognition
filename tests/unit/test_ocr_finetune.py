"""Tests for PaddleOCR fine-tune data export."""

from pathlib import Path

from PIL import Image

from vlpr.data.manifest_io import write_manifest
from vlpr.data.manifest_schema import OcrAnnotation, OcrManifestRecord
from vlpr.data.ocr_finetune import export_ocr_finetune_data


def _write_image(path: Path, size: tuple[int, int]) -> None:
    """Create a plain OCR crop image for exporter tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _record(
    image_path: str,
    text: str,
    split: str,
    size: tuple[int, int],
    sha: str,
) -> OcrManifestRecord:
    """Create a minimal OCR manifest record for exporter tests."""
    return OcrManifestRecord(
        sample_id=f"ocr:{image_path}",
        dataset_name="ocr",
        image_path=image_path,
        source_split="train",
        width=size[0],
        height=size[1],
        sha256=sha * 64,
        perceptual_hash=sha * 16,
        group_id="sha256:" + sha * 64,
        split=split,
        validation_status="valid",
        task="ocr",
        annotation=OcrAnnotation(raw_text=text),
    )


def test_export_ocr_finetune_data_splits_compact_and_writes_charset(tmp_path: Path) -> None:
    """Exporter writes PaddleOCR list files and splits compact crops into two lines."""
    dataset_root = tmp_path / "raw"
    _write_image(dataset_root / "train_compact.jpg", (60, 50))
    _write_image(dataset_root / "val_wide.jpg", (120, 30))
    _write_image(dataset_root / "test_compact.jpg", (60, 50))
    manifest = tmp_path / "ocr_manifest.jsonl"
    write_manifest(
        manifest,
        [
            _record("train_compact.jpg", "30A 12345", "train", (60, 50), "a"),
            _record("val_wide.jpg", "51B 67890", "validation", (120, 30), "b"),
            _record("test_compact.jpg", "60M\u01101 01835", "test", (60, 50), "c"),
        ],
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "data:",
                f"  manifest: {manifest.as_posix()}",
                f"  dataset_root: {dataset_root.as_posix()}",
                f"  output_dir: {(tmp_path / 'out').as_posix()}",
                "  compact_aspect_ratio: 1.5",
                "  split_search_start: 0.35",
                "  split_search_end: 0.65",
                "  extra_characters: ABCDEFGHIJKLMNOPQRSTUVWXYZ\u01100123456789",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = export_ocr_finetune_data(config)

    output_dir = tmp_path / "out"
    assert summary == {
        "characters": 37,
        "compact_aspect_ratio": 1.5,
        "test_samples": 2,
        "train_samples": 2,
        "validation_samples": 1,
    }
    assert len((output_dir / "train_list.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output_dir / "val_list.txt").read_text(encoding="utf-8").splitlines()) == 1
    assert len((output_dir / "test_list.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert "\u0110" in (output_dir / "dict.txt").read_text(encoding="utf-8")
