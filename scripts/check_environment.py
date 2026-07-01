"""Inspect the local ML runtime without installing or changing packages."""

import argparse
from pathlib import Path

from vlpr.environment import inspect_environment, write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    """Build the environment-check command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown report path.",
    )
    return parser


def main() -> int:
    """Print an environment snapshot and optionally save Markdown."""
    args = build_parser().parse_args()
    report = inspect_environment()
    print(report.to_json())
    if args.output is not None:
        write_markdown_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
