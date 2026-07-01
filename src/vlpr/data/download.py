"""Reproducible Kaggle dataset download."""

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def download_dataset(
    config_path: Path,
    dataset_name: str = "detection",
    force: bool = False,
) -> Path:
    """Download one pinned Kaggle dataset version and record retrieval metadata."""
    config = load_config(config_path)
    dataset = config.dataset(dataset_name)
    root = project_root(config_path)
    output_dir = resolve_project_path(root, dataset.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Downloading dataset name=%s handle=%s output=%s",
        dataset_name,
        dataset.versioned_handle,
        output_dir,
    )
    resolved_download = Path(
        kagglehub.dataset_download(
            dataset.versioned_handle,
            output_dir=str(output_dir),
            force_download=force,
        )
    ).resolve()

    metadata = {
        "dataset_name": dataset_name,
        "dataset_task": dataset.task,
        "dataset_handle": dataset.handle,
        "dataset_version": dataset.version,
        "dataset_url": f"https://www.kaggle.com/datasets/{dataset.handle}",
        "expected_license_from_data_card": dataset.expected_license,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_download_path": str(resolved_download),
    }
    metadata_path = output_dir / "download_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Dataset download completed metadata=%s", metadata_path)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset YAML configuration.",
    )
    parser.add_argument(
        "--dataset",
        default="detection",
        help="Logical dataset name declared under the datasets config mapping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite KaggleHub-managed output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset download CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        download_dataset(
            config_path=args.config,
            dataset_name=args.dataset,
            force=args.force,
        )
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
