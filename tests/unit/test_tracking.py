"""Tests for safe W&B fallback behavior."""

import pytest

from vlpr.training.tracking import resolve_wandb_mode


def test_auto_mode_is_online_when_api_key_exists() -> None:
    assert resolve_wandb_mode("auto", {"WANDB_API_KEY": "secret"}) == "online"


def test_auto_mode_falls_back_offline_without_api_key() -> None:
    assert resolve_wandb_mode("auto", {}) == "offline"


def test_environment_mode_overrides_configuration() -> None:
    assert resolve_wandb_mode("online", {"WANDB_MODE": "disabled"}) == "disabled"


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        resolve_wandb_mode("sometimes", {})
