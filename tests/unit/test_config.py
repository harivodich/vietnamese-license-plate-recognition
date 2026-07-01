"""Tests for strict project configuration."""

from vlpr.config import SplitSettings


def test_split_ratios_accept_complete_partition() -> None:
    split = SplitSettings(train=0.75, validation=0.125, test=0.125, seed=42)

    assert split.train + split.validation + split.test == 1.0
