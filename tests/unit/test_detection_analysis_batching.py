"""Kiểm thử batching inference không làm mất mapping ảnh-label."""

from pathlib import Path
from typing import Any, cast

from vlpr.evaluation.detection import DetectionEvaluationConfig
from vlpr.evaluation.detection_analysis import _predict_in_batches


class _FakeModel:
    """Ghi lại các batch để test orchestration mà không nạp model thật."""

    def __init__(self) -> None:
        """Khởi tạo nhật ký lời gọi rỗng."""
        self.calls: list[list[str]] = []

    def predict(self, **kwargs: Any) -> list[str]:
        """Trả một kết quả giả cho từng đường dẫn theo đúng thứ tự đầu vào."""
        paths = cast(list[str], kwargs["source"])
        self.calls.append(paths)
        return [f"result:{path}" for path in paths]


def test_predict_in_batches_limits_memory_and_preserves_paths() -> None:
    """Batching phải giới hạn kích thước và không làm mất mapping ảnh-label."""
    config = DetectionEvaluationConfig(
        checkpoint=Path("best.pt"),
        data=Path("dataset.yaml"),
        project=Path("artifacts"),
        name="test",
        split="test",
        imgsz=640,
        batch=2,
        workers=0,
        device=None,
        conf=0.001,
        iou=0.7,
        max_det=100,
        plots=True,
        operating_conf=0.25,
        match_iou=0.5,
        failure_examples=5,
    )
    model = _FakeModel()
    paths = ["first.jpg", "second.jpg", "third.jpg"]

    results = list(_predict_in_batches(model, paths, config))

    assert model.calls == [["first.jpg", "second.jpg"], ["third.jpg"]]
    assert results == [
        (Path("first.jpg"), "result:first.jpg"),
        (Path("second.jpg"), "result:second.jpg"),
        (Path("third.jpg"), "result:third.jpg"),
    ]
