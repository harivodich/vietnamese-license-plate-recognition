"""Conservative plate-text validation without rewriting OCR output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vlpr.evaluation.ocr import normalize_plate_text

PlateStatus = Literal["valid", "low_confidence", "invalid_format", "empty"]


@dataclass(frozen=True)
class PlateAssessment:
    """A normalized OCR result and an explicit safety status."""

    raw_text: str
    normalized_text: str
    format_valid: bool
    status: PlateStatus


def has_plausible_plate_format(text: str) -> bool:
    """Accept broad Vietnamese plate-like text, without pretending to identify every plate class."""
    return (
        6 <= len(text) <= 10
        and text.isalnum()
        and any(character.isalpha() for character in text)
        and any(character.isdigit() for character in text)
    )


def assess_plate_text(
    raw_text: str,
    *,
    detection_confidence: float,
    ocr_confidence: float,
    min_detection_confidence: float = 0.5,
    min_ocr_confidence: float = 0.8,
) -> PlateAssessment:
    """Classify a prediction while preserving the model's original text for auditability."""
    normalized_text = normalize_plate_text(raw_text)
    if not normalized_text:
        status: PlateStatus = "empty"
    elif not has_plausible_plate_format(normalized_text):
        status = "invalid_format"
    elif detection_confidence < min_detection_confidence or ocr_confidence < min_ocr_confidence:
        status = "low_confidence"
    else:
        status = "valid"
    return PlateAssessment(
        raw_text=raw_text,
        normalized_text=normalized_text,
        format_valid=has_plausible_plate_format(normalized_text),
        status=status,
    )
