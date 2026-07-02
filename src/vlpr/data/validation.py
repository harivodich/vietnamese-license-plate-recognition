"""Gate 1 raw-source check for the future full dataset validation command."""

from collections.abc import Sequence

from vlpr.data.source_status import run_source_check


def main(argv: Sequence[str] | None = None) -> int:
    """Verify receipts before Gate 2 adds image and annotation validation."""
    return run_source_check(argv, __doc__ or "Check raw sources")
