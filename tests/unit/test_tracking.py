"""Kiểm thử chính sách chọn mode W&B và fallback an toàn."""

import pytest

from vlpr.training.tracking import resolve_wandb_mode


def test_auto_mode_is_online_when_api_key_exists() -> None:
    """Xác nhận mode auto chọn online khi có W&B API key."""
    assert resolve_wandb_mode("auto", {"WANDB_API_KEY": "secret"}) == "online"


def test_auto_mode_falls_back_offline_without_api_key() -> None:
    """Xác nhận mode auto chuyển offline khi không có W&B API key."""
    assert resolve_wandb_mode("auto", {}) == "offline"


def test_environment_mode_overrides_configuration() -> None:
    """Xác nhận biến môi trường được ưu tiên hơn giá trị trong YAML."""
    assert resolve_wandb_mode("online", {"WANDB_MODE": "disabled"}) == "disabled"


def test_invalid_mode_is_rejected() -> None:
    """Xác nhận mode ngoài danh sách cho phép bị từ chối."""
    with pytest.raises(ValueError, match="unsupported"):
        resolve_wandb_mode("sometimes", {})
