"""Tạo biểu đồ EDA cho crop và nhãn của OCR dataset."""

import argparse
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import matplotlib
import numpy as np

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import OcrManifestRecord
from vlpr.utils.logging import configure_logging

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

LOGGER = logging.getLogger(__name__)
_LabelT = TypeVar("_LabelT")


@dataclass(frozen=True, slots=True)
class OcrEdaData:
    """Các vector hình học, text và bộ đếm dùng cho OCR EDA."""

    image_counts: Counter[str]
    character_counts: Counter[str]
    length_counts: Counter[int]
    normalized_length_counts: Counter[int]
    pattern_counts: Counter[str]
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    aspect_ratios: tuple[float, ...]
    areas: tuple[int, ...]
    lengths_by_split: dict[str, tuple[int, ...]]
    aspects_by_split: dict[str, tuple[float, ...]]


def collect_ocr_eda(records: tuple[OcrManifestRecord, ...]) -> OcrEdaData:
    """Tính phân phối crop, ký tự, độ dài và format theo project split."""
    image_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    normalized_length_counts: Counter[int] = Counter()
    pattern_counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    areas: list[int] = []
    lengths_by_split: dict[str, list[int]] = defaultdict(list)
    aspects_by_split: dict[str, list[float]] = defaultdict(list)

    for record in records:
        if record.split is None:
            raise ValueError(f"record chưa có project split: {record.sample_id}")
        text = record.annotation.raw_text
        normalized = text.replace(" ", "")
        aspect_ratio = record.width / record.height
        image_counts[record.split] += 1
        character_counts.update(text)
        length_counts[len(text)] += 1
        normalized_length_counts[len(normalized)] += 1
        pattern_counts[_text_pattern(text)] += 1
        widths.append(record.width)
        heights.append(record.height)
        aspect_ratios.append(aspect_ratio)
        areas.append(record.width * record.height)
        lengths_by_split[record.split].append(len(normalized))
        aspects_by_split[record.split].append(aspect_ratio)

    if not records:
        raise ValueError("OCR manifest không được rỗng")
    return OcrEdaData(
        image_counts=image_counts,
        character_counts=character_counts,
        length_counts=length_counts,
        normalized_length_counts=normalized_length_counts,
        pattern_counts=pattern_counts,
        widths=tuple(widths),
        heights=tuple(heights),
        aspect_ratios=tuple(aspect_ratios),
        areas=tuple(areas),
        lengths_by_split={split: tuple(values) for split, values in lengths_by_split.items()},
        aspects_by_split={split: tuple(values) for split, values in aspects_by_split.items()},
    )


def render_ocr_eda(data: OcrEdaData, output_dir: Path) -> tuple[Path, ...]:
    """Vẽ overview, crop geometry và so sánh project split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "overview.png",
        output_dir / "crop_geometry.png",
        output_dir / "split_comparison.png",
    )
    _render_overview(data, outputs[0])
    _render_crop_geometry(data, outputs[1])
    _render_split_comparison(data, outputs[2])
    return outputs


def generate_ocr_eda(config_path: Path) -> tuple[Path, ...]:
    """Đọc processed OCR manifest và tạo toàn bộ biểu đồ."""
    config = load_config(config_path)
    root = project_root(config_path)
    manifest_path = resolve_project_path(root, config.split.output_dir) / "ocr_manifest.jsonl"
    records = read_manifest(manifest_path)
    ocr_records = tuple(record for record in records if isinstance(record, OcrManifestRecord))
    if len(ocr_records) != len(records):
        raise ValueError("processed OCR manifest chứa record sai task")
    output_dir = resolve_project_path(root, config.validation.ocr_eda_dir)
    return render_ocr_eda(collect_ocr_eda(ocr_records), output_dir)


def _render_overview(data: OcrEdaData, path: Path) -> None:
    """Vẽ độ dài, tần suất ký tự, format và số mẫu theo split."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    length_counts = dict(sorted(data.normalized_length_counts.items()))
    _bar(axes[0, 0], length_counts, "Plate length without spaces")
    display_characters = {
        "SPACE" if character == " " else character: count
        for character, count in data.character_counts.most_common()
    }
    _bar(axes[0, 1], display_characters, "Character frequency")
    axes[0, 1].tick_params(axis="x", rotation=45)
    top_patterns = dict(data.pattern_counts.most_common(12))
    _bar(axes[1, 0], top_patterns, "Top OCR label patterns")
    axes[1, 0].tick_params(axis="x", rotation=40)
    image_counts = {split: data.image_counts[split] for split in ("train", "validation", "test")}
    _bar(axes[1, 1], image_counts, "Images by project split")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_crop_geometry(data: OcrEdaData, path: Path) -> None:
    """Vẽ mật độ width-height, aspect ratio và diện tích crop."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].hexbin(data.widths, data.heights, gridsize=45, bins="log", cmap="viridis")
    axes[0].set_title("OCR crop width-height density")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[1].hist(data.aspect_ratios, bins=60, color="#3366cc", alpha=0.85)
    axes[1].set_title("OCR crop aspect ratio")
    axes[1].set_xlabel("Width / height")
    area_bins = np.geomspace(min(data.areas), max(data.areas), 60)
    axes[2].hist(data.areas, bins=area_bins, color="#3366cc", alpha=0.85)
    axes[2].set_xscale("log")
    axes[2].set_title("OCR crop area")
    axes[2].set_xlabel("Pixels (log scale)")
    for axis in axes[1:]:
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_split_comparison(data: OcrEdaData, path: Path) -> None:
    """So sánh text length và crop aspect ratio giữa ba split."""
    splits = ("train", "validation", "test")
    colors = ("#3366cc", "#dc3912", "#109618")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    length_bins = np.arange(
        min(data.normalized_length_counts) - 0.5,
        max(data.normalized_length_counts) + 1.5,
    )
    aspect_bins = np.linspace(min(data.aspect_ratios), max(data.aspect_ratios), 60)
    for split, color in zip(splits, colors, strict=True):
        axes[0].hist(
            data.lengths_by_split[split],
            bins=length_bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=split,
            color=color,
        )
        axes[1].hist(
            data.aspects_by_split[split],
            bins=aspect_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=split,
            color=color,
        )
    axes[0].set_title("Plate length by split")
    axes[0].set_xlabel("Characters without spaces")
    axes[1].set_title("OCR crop aspect ratio by split")
    axes[1].set_xlabel("Width / height")
    for axis in axes:
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _text_pattern(text: str) -> str:
    """Đổi chữ thành L và số thành D nhưng giữ dấu cách."""
    return "".join(
        "D" if character.isdigit() else "L" if character.isalpha() else character
        for character in text
    )


def _bar(
    axis: Axes,
    counts: Mapping[_LabelT, int],
    title: str,
) -> None:
    """Vẽ bar chart và hiện số đếm trên cột."""
    labels = [str(label) for label in counts]
    values = list(counts.values())
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color="#3366cc")
    axis.set_xticks(positions, labels)
    axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.2)


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI nhận dataset config dùng chung."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Tạo OCR EDA figures và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        outputs = generate_ocr_eda(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("OCR EDA failed: %s", exc)
        return 1
    for output in outputs:
        LOGGER.info("OCR EDA figure written path=%s", output)
    return 0
