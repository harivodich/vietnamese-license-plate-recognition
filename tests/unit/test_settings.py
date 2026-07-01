"""Tests for typed project-wide settings."""

from pathlib import Path

from vlpr.settings import load_runtime_settings


def test_load_runtime_settings() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "project.yaml"

    settings = load_runtime_settings(config_path)

    assert settings.project.seed == 20260701
    assert settings.tracking.provider == "wandb"
    assert settings.tracking.mode == "auto"
