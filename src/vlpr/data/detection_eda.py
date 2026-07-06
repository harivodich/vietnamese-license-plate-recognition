"""Tạo biểu đồ EDA detection từ manifest đã đóng băng project split."""

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import DetectionManifestRecord
from vlpr.utils.logging import configure_logging

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionEdaData:
    """Các vector bbox và bộ đếm cần thiết để vẽ mà không đọc manifest lại."""

    class_counts: Counter[str]
    image_counts: Counter[str]
    box_counts: Counter[str]
    size_counts: Counter[str]
    widths: tuple[float, ...]
    heights: tuple[float, ...]
    areas: tuple[float, ...]
    aspect_ratios: tuple[float, ...]
    center_x: tuple[float, ...]
    center_y: tuple[float, ...]
    areas_by_split: dict[str, tuple[float, ...]]
    aspect_ratios_by_split: dict[str, tuple[float, ...]]
    sizes_by_split: dict[str, Counter[str]]


def collect_detection_eda(
    records: tuple[DetectionManifestRecord, ...],
    *,
    image_size: int = 640,
) -> DetectionEdaData:
    """Tính vector EDA và box size trong không gian letterbox của model."""
    class_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    center_x: list[float] = []
    center_y: list[float] = []
    areas_by_split: dict[str, list[float]] = defaultdict(list)
    aspect_ratios_by_split: dict[str, list[float]] = defaultdict(list)
    sizes_by_split: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if record.split is None:
            raise ValueError(f"record chưa có project split: {record.sample_id}")
        image_counts[record.split] += 1
        scale = min(image_size / record.width, image_size / record.height)
        for annotation in record.annotations:
            bbox = annotation.bbox
            class_counts[annotation.class_name] += 1
            box_counts[record.split] += 1
            widths.append(bbox.width)
            heights.append(bbox.height)
            areas.append(bbox.width * bbox.height)
            aspect_ratios.append(bbox.width / bbox.height)
            areas_by_split[record.split].append(bbox.width * bbox.height)
            aspect_ratios_by_split[record.split].append(bbox.width / bbox.height)
            center_x.append(bbox.center_x)
            center_y.append(bbox.center_y)
            resized_area = bbox.width * record.width * scale * bbox.height * record.height * scale
            if resized_area < 32**2:
                size_name = "small"
            elif resized_area < 96**2:
                size_name = "medium"
            else:
                size_name = "large"
            size_counts[size_name] += 1
            sizes_by_split[record.split][size_name] += 1

    if not widths:
        raise ValueError("detection manifest không có bounding box")
    return DetectionEdaData(
        class_counts=class_counts,
        image_counts=image_counts,
        box_counts=box_counts,
        size_counts=size_counts,
        widths=tuple(widths),
        heights=tuple(heights),
        areas=tuple(areas),
        aspect_ratios=tuple(aspect_ratios),
        center_x=tuple(center_x),
        center_y=tuple(center_y),
        areas_by_split={split: tuple(values) for split, values in areas_by_split.items()},
        aspect_ratios_by_split={
            split: tuple(values) for split, values in aspect_ratios_by_split.items()
        },
        sizes_by_split=dict(sizes_by_split),
    )


def render_detection_eda(data: DetectionEdaData, output_dir: Path) -> tuple[Path, ...]:
    """Vẽ overview, phân phối bbox và spatial plots dưới dạng PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "overview.png",
        output_dir / "bbox_distributions.png",
        output_dir / "bbox_spatial.png",
        output_dir / "split_comparison.png",
    )
    _render_overview(data, outputs[0])
    _render_distributions(data, outputs[1])
    _render_spatial(data, outputs[2])
    _render_split_comparison(data, outputs[3])
    return outputs


def generate_detection_eda(config_path: Path) -> tuple[Path, ...]:
    """Đọc processed detection manifest và tạo toàn bộ biểu đồ."""
    config = load_config(config_path)
    root = project_root(config_path)
    manifest_path = resolve_project_path(root, config.split.output_dir) / "detection_manifest.jsonl"
    records = read_manifest(manifest_path)
    detection_records = tuple(
        record for record in records if isinstance(record, DetectionManifestRecord)
    )
    if len(detection_records) != len(records):
        raise ValueError("processed detection manifest chứa record sai task")
    output_dir = resolve_project_path(root, config.validation.detection_eda_dir)
    return render_detection_eda(collect_detection_eda(detection_records), output_dir)


def _render_overview(data: DetectionEdaData, path: Path) -> None:
    """Vẽ class, số ảnh, số box theo split và nhóm box size."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    _bar(axes[0, 0], data.class_counts, "Class distribution", "Bounding boxes")
    _bar(axes[0, 1], data.image_counts, "Images by project split", "Images")
    _bar(axes[1, 0], data.box_counts, "Boxes by project split", "Bounding boxes")
    _bar(
        axes[1, 1],
        data.size_counts,
        "Box size after 640px letterbox",
        "Bounding boxes",
        order=("small", "medium", "large"),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_distributions(data: DetectionEdaData, path: Path) -> None:
    """Vẽ histogram width, height, area và aspect ratio của bbox."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = (
        (axes[0, 0], data.widths, "Normalized bbox width", False),
        (axes[0, 1], data.heights, "Normalized bbox height", False),
        (axes[1, 0], data.areas, "Normalized bbox area", True),
        (axes[1, 1], data.aspect_ratios, "BBox aspect ratio (width / height)", False),
    )
    for axis, values, title, log_x in plots:
        if log_x:
            bins = np.geomspace(min(values), max(values), 50)
            axis.set_xscale("log")
            axis.hist(values, bins=bins, color="#3366cc", alpha=0.85)
        else:
            axis.hist(values, bins=50, color="#3366cc", alpha=0.85)
        axis.set_title(title)
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_spatial(data: DetectionEdaData, path: Path) -> None:
    """Vẽ scatter kích thước và heatmap vị trí tâm bbox."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(data.widths, data.heights, gridsize=45, bins="log", cmap="viridis")
    axes[0].set_title("BBox width-height density")
    axes[0].set_xlabel("Normalized width")
    axes[0].set_ylabel("Normalized height")
    heatmap = axes[1].hist2d(
        data.center_x,
        data.center_y,
        bins=30,
        range=((0, 1), (0, 1)),
        cmap="magma",
    )
    figure.colorbar(heatmap[3], ax=axes[1], label="Bounding boxes")
    axes[1].invert_yaxis()
    axes[1].set_title("BBox center heatmap")
    axes[1].set_xlabel("Normalized center x")
    axes[1].set_ylabel("Normalized center y")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_split_comparison(data: DetectionEdaData, path: Path) -> None:
    """So sánh area, aspect ratio và tỷ lệ box size giữa ba project split."""
    splits = ("train", "validation", "test")
    colors = ("#3366cc", "#dc3912", "#109618")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    area_bins = np.geomspace(min(data.areas), max(data.areas), 45)
    aspect_bins = np.linspace(min(data.aspect_ratios), max(data.aspect_ratios), 45)
    for split, color in zip(splits, colors, strict=True):
        axes[0].hist(
            data.areas_by_split[split],
            bins=area_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=split,
            color=color,
        )
        axes[1].hist(
            data.aspect_ratios_by_split[split],
            bins=aspect_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=split,
            color=color,
        )
    axes[0].set_xscale("log")
    axes[0].set_title("Normalized bbox area by split")
    axes[0].set_xlabel("Normalized area")
    axes[1].set_title("BBox aspect ratio by split")
    axes[1].set_xlabel("Width / height")
    for axis in axes[:2]:
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)

    size_names = ("small", "medium", "large")
    x = np.arange(len(splits))
    bottom = np.zeros(len(splits))
    for size_name, color in zip(size_names, colors, strict=True):
        percentages = np.array(
            [
                100
                * data.sizes_by_split[split][size_name]
                / sum(data.sizes_by_split[split].values())
                for split in splits
            ]
        )
        axes[2].bar(x, percentages, bottom=bottom, label=size_name, color=color)
        bottom += percentages
    axes[2].set_xticks(x, splits)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Box size proportion by split")
    axes[2].set_ylabel("Bounding boxes (%)")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _bar(
    axis: Axes,
    counts: Counter[str],
    title: str,
    ylabel: str,
    *,
    order: tuple[str, ...] | None = None,
) -> None:
    """Vẽ bar chart có số tuyệt đối trên từng cột."""
    labels = order or tuple(sorted(counts))
    values = [counts[label] for label in labels]
    bars = axis.bar(labels, values, color="#3366cc")
    axis.bar_label(bars, padding=3)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
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
    """Tạo EDA figures và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        outputs = generate_detection_eda(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("Detection EDA failed: %s", exc)
        return 1
    for output in outputs:
        LOGGER.info("Detection EDA figure written path=%s", output)
    return 0
