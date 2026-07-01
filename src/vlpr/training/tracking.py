"""Weights & Biases mode resolution without a hard runtime dependency."""

import os
from collections.abc import Mapping
from typing import Literal, cast

TrackingMode = Literal["online", "offline", "disabled"]
_VALID_EXPLICIT_MODES: frozenset[str] = frozenset({"online", "offline", "disabled"})


def resolve_wandb_mode(
    configured_mode: str,
    environ: Mapping[str, str] | None = None,
) -> TrackingMode:
    """Resolve W&B mode, falling back offline when credentials are unavailable."""
    environment = os.environ if environ is None else environ
    environment_mode = environment.get("WANDB_MODE", "").strip().lower()
    requested = environment_mode or configured_mode.strip().lower()

    if requested in _VALID_EXPLICIT_MODES:
        return cast(TrackingMode, requested)
    if requested != "auto":
        raise ValueError(f"unsupported W&B mode: {requested}")
    return "online" if environment.get("WANDB_API_KEY", "").strip() else "offline"
