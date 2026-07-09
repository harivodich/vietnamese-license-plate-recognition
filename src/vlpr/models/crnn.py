"""CRNN biến ảnh text line thành chuỗi ký tự bằng CTC."""

from pathlib import Path

import torch
from torch import Tensor, nn


class OcrCharset:
    """Ánh xạ hai chiều giữa ký tự và CTC class index."""

    def __init__(self, characters: tuple[str, ...]) -> None:
        """Dành index 0 cho CTC blank và từ chối charset mơ hồ."""
        if not characters:
            raise ValueError("OCR charset không được rỗng")
        if len(set(characters)) != len(characters):
            raise ValueError("OCR charset chứa ký tự trùng")
        if any(len(character) != 1 or character.isspace() for character in characters):
            raise ValueError("mỗi dòng charset phải chứa đúng một ký tự không phải whitespace")
        self.characters = characters
        self._index_by_character = {
            character: index for index, character in enumerate(characters, start=1)
        }

    @classmethod
    def from_file(cls, path: Path) -> "OcrCharset":
        """Đọc UTF-8 character dictionary, mỗi dòng đúng một ký tự."""
        return cls(tuple(path.read_text(encoding="utf-8").splitlines()))

    @property
    def blank_index(self) -> int:
        """CTC blank luôn dùng class 0."""
        return 0

    @property
    def num_classes(self) -> int:
        """Tổng số output classes gồm charset và blank."""
        return len(self.characters) + 1

    def encode(self, text: str) -> tuple[int, ...]:
        """Chuyển label thành class indices và báo ký tự ngoài charset."""
        try:
            return tuple(self._index_by_character[character] for character in text)
        except KeyError as exc:
            raise ValueError(f"label chứa ký tự ngoài charset: {exc.args[0]!r}") from exc

    def decode(self, indices: list[int]) -> str:
        """Greedy CTC decode bằng cách bỏ blank và collapse ký tự lặp liên tiếp."""
        output: list[str] = []
        previous = self.blank_index
        for index in indices:
            if index != self.blank_index and index != previous:
                if index < 0 or index > len(self.characters):
                    raise ValueError(f"CTC class index ngoài charset: {index}")
                output.append(self.characters[index - 1])
            previous = index
        return "".join(output)


class CrnnCtc(nn.Module):
    """CNN encoder, BiLSTM sequence model và CTC classification head."""

    def __init__(
        self,
        *,
        num_classes: int,
        hidden_size: int,
        lstm_layers: int,
        dropout: float,
        blank_index: int,
        blank_bias: float,
        input_height: int = 32,
    ) -> None:
        """Create CRNN for the fixed OCR image height from config."""
        super().__init__()
        vertical_feature_height = input_height // 16
        if vertical_feature_height < 1:
            raise ValueError("input_height is too small for CRNN pooling")
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Height-specific collapse is faster than adaptive pooling on GPU
            # and keeps the horizontal sequence unchanged.
            nn.Conv2d(512, 512, (vertical_feature_height, 1)),
            nn.ReLU(inplace=True),
        )
        recurrent_dropout = dropout if lstm_layers > 1 else 0.0
        self.sequence = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        if not 0 <= blank_index < num_classes:
            raise ValueError("blank_index phải nằm trong số class OCR")
        with torch.no_grad():
            self.classifier.bias[blank_index] = blank_bias

    def forward(self, images: Tensor) -> Tensor:
        """Trả log-probability tensor `[time, batch, classes]` cho CTCLoss."""
        features = self.features(images)
        if features.shape[2] != 1:
            raise ValueError(f"CRNN feature height phải bằng 1, nhận {features.shape[2]}")
        sequence = features.squeeze(2).permute(2, 0, 1)
        recurrent, _ = self.sequence(sequence)
        return torch.log_softmax(self.classifier(recurrent), dim=2)
