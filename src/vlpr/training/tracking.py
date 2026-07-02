"""Xác định chế độ Weights & Biases mà không tạo phụ thuộc runtime cứng."""

import os
from collections.abc import Mapping
from typing import Literal, cast

TrackingMode = Literal["online", "offline", "disabled"]
_VALID_EXPLICIT_MODES: frozenset[str] = frozenset({"online", "offline", "disabled"})


def resolve_wandb_mode(
    configured_mode: str,
    environ: Mapping[str, str] | None = None,
) -> TrackingMode:
    """Chọn mode W&B và tự chuyển sang offline khi mode auto không có API key."""
    environment = os.environ if environ is None else environ
    environment_mode = environment.get("WANDB_MODE", "").strip().lower()
    requested = environment_mode or configured_mode.strip().lower()

    if requested in _VALID_EXPLICIT_MODES:
        return cast(TrackingMode, requested)
    if requested != "auto":
        raise ValueError(f"unsupported W&B mode: {requested}")
    return "online" if environment.get("WANDB_API_KEY", "").strip() else "offline"
