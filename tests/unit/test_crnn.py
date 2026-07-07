"""Kiểm thử charset, CRNN tensor shape và image preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vlpr.models.crnn import CrnnCtc, OcrCharset
from vlpr.training.ocr import (
    OcrSelectionScore,
    _is_better_ocr_checkpoint,
    preprocess_ocr_image,
)


def test_charset_encode_and_ctc_decode() -> None:
    """CTC decode phải bỏ blank và collapse lặp nhưng giữ lặp qua blank."""
    charset = OcrCharset(("A", "B", "Đ"))

    assert charset.encode("AĐ") == (1, 3)
    assert charset.decode([0, 1, 1, 0, 1, 2, 0]) == "AAB"


def test_charset_rejects_unknown_character() -> None:
    """Label có ký tự ngoài dictionary phải fail trước training."""
    charset = OcrCharset(("A", "B"))

    with pytest.raises(ValueError, match="ngoài charset"):
        charset.encode("AC")


def test_crnn_returns_time_batch_class_log_probabilities() -> None:
    """Forward phải hạ feature height về một và giữ batch/classes."""
    model = CrnnCtc(
        num_classes=36,
        hidden_size=32,
        lstm_layers=1,
        dropout=0.0,
        blank_index=0,
        blank_bias=-2.0,
    )

    output = model(torch.zeros(2, 1, 32, 160))

    assert output.shape == (40, 2, 36)
    assert torch.allclose(output.exp().sum(dim=2), torch.ones(40, 2), atol=1e-5)


def test_crnn_applies_blank_bias() -> None:
    """Blank bias thấp giúp CTC tránh collapse về blank ở đầu training."""
    model = CrnnCtc(
        num_classes=4,
        hidden_size=8,
        lstm_layers=1,
        dropout=0.0,
        blank_index=0,
        blank_bias=-2.0,
    )

    assert float(model.classifier.bias[0]) == pytest.approx(-2.0)


def test_crnn_rejects_invalid_blank_index() -> None:
    """Fail sớm nếu cấu hình CTC blank nằm ngoài output classes."""
    with pytest.raises(ValueError, match="blank_index"):
        CrnnCtc(
            num_classes=4,
            hidden_size=8,
            lstm_layers=1,
            dropout=0.0,
            blank_index=4,
            blank_bias=-2.0,
        )


def test_preprocess_preserves_aspect_ratio_and_output_shape(tmp_path: Path) -> None:
    """Preprocess phải tạo tensor cố định, pad trắng và không bóp méo ảnh."""
    image_path = tmp_path / "line.png"
    Image.fromarray(np.full((20, 80), 128, dtype=np.uint8), mode="L").save(image_path)

    tensor = preprocess_ocr_image(
        image_path,
        image_height=32,
        image_width=160,
        augmentation=None,
    )

    assert tensor.shape == (1, 32, 160)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_ocr_checkpoint_selection_uses_cer_when_exact_match_ties() -> None:
    """Khi exact còn bằng 0, CER tốt hơn vẫn phải được xem là checkpoint cải thiện."""
    best = OcrSelectionScore(exact_match=0.0, cer=1.0, loss=3.2)
    candidate = OcrSelectionScore(exact_match=0.0, cer=0.84, loss=3.0)

    assert _is_better_ocr_checkpoint(candidate, best)


def test_ocr_checkpoint_selection_prioritizes_exact_match_over_cer() -> None:
    """Exact match cao hơn quan trọng hơn CER vì production cần đúng toàn bộ biển/dòng."""
    best = OcrSelectionScore(exact_match=0.1, cer=0.2, loss=1.0)
    candidate = OcrSelectionScore(exact_match=0.09, cer=0.1, loss=0.8)

    assert not _is_better_ocr_checkpoint(candidate, best)
