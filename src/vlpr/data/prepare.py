"""Gate 1 precondition check for the future manifest preparation command."""

from collections.abc import Sequence

from vlpr.data.source_status import run_source_check


def main(argv: Sequence[str] | None = None) -> int:
    """Verify raw sources before Gate 2 adds manifest generation."""
    return run_source_check(argv, __doc__ or "Check raw sources")
