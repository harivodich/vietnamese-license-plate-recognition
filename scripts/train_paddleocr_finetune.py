"""Prepare config then launch PaddleOCR fine-tuning."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from vlpr.training.paddleocr_finetune import run_paddleocr_finetune
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the user-facing PaddleOCR train command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-paddleocr-finetune.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from PaddleOCR checkpoint prefix or .pdparams path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run PaddleOCR fine-tuning through the project preflight wrapper."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return run_paddleocr_finetune(args.config, resume_from=args.resume)
    except (OSError, ValueError) as exc:
        LOGGER.error("PaddleOCR fine-tune failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
