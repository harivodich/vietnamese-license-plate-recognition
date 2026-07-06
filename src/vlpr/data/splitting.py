"""Chia manifest theo nhóm exact duplicate bằng thuật toán tái lập."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest, write_manifest
from vlpr.data.manifest_schema import ManifestRecord
from vlpr.data.source_status import build_parser, find_unready_sources
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
SplitName = Literal["train", "validation", "test"]
_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


def assign_splits(
    records: Iterable[ManifestRecord],
    *,
    ratios: dict[SplitName, float],
    seed: int,
) -> tuple[ManifestRecord, ...]:
    """Gán split theo SHA group và cân bằng gần tỷ lệ yêu cầu."""
    materialized = tuple(records)
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in materialized:
        groups[record.sha256].append(record)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
        ),
    )
    targets = {name: len(materialized) * ratios[name] for name in _SPLITS}
    counts: dict[SplitName, int] = {name: 0 for name in _SPLITS}
    assignment: dict[str, SplitName] = {}
    for sha256, group in ordered_groups:
        chosen = max(
            _SPLITS,
            key=lambda name: (targets[name] - counts[name], -_SPLITS.index(name)),
        )
        assignment[sha256] = chosen
        counts[chosen] += len(group)

    return tuple(
        record.model_copy(
            update={
                "group_id": f"sha256:{record.sha256}",
                "split": assignment[record.sha256],
            }
        )
        for record in materialized
    )


def deduplicate_exact_records(
    records: Iterable[ManifestRecord],
) -> tuple[ManifestRecord, ...]:
    """Giữ record có sample_id nhỏ nhất trong mỗi nhóm ảnh cùng SHA-256."""
    materialized = tuple(records)
    canonical_id_by_sha: dict[str, str] = {}
    for record in materialized:
        current = canonical_id_by_sha.get(record.sha256)
        if current is None or record.sample_id < current:
            canonical_id_by_sha[record.sha256] = record.sample_id
    return tuple(
        record for record in materialized if record.sample_id == canonical_id_by_sha[record.sha256]
    )


def split_manifests(config_path: Path) -> dict[str, Path]:
    """Đọc manifest interim, gán split và ghi manifest processed theo task."""
    config = load_config(config_path)
    root = project_root(config_path)
    output_dir = resolve_project_path(root, config.split.output_dir)
    ratios: dict[SplitName, float] = {
        "train": config.split.train,
        "validation": config.split.validation,
        "test": config.split.test,
    }
    outputs: dict[str, Path] = {}
    for dataset_name in ("detection", "ocr"):
        source = resolve_project_path(
            root,
            config.dataset(dataset_name).manifest_path,
        )
        destination = output_dir / f"{dataset_name}_manifest.jsonl"
        unique_records = deduplicate_exact_records(read_manifest(source))
        write_manifest(
            destination,
            assign_splits(unique_records, ratios=ratios, seed=config.split.seed),
        )
        outputs[dataset_name] = destination
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Kiểm tra nguồn, chia hai manifest và trả exit code cho CLI."""
    configure_logging()
    args = build_parser(__doc__ or "Split dataset manifests").parse_args(argv)
    try:
        unready = find_unready_sources(args.config)
        if unready:
            raise RuntimeError(f"raw sources chưa sẵn sàng: {', '.join(unready)}")
        outputs = split_manifests(args.config)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset split failed: %s", exc)
        return 1

    for dataset_name, output_path in outputs.items():
        LOGGER.info("Dataset split completed dataset=%s path=%s", dataset_name, output_path)
    return 0
