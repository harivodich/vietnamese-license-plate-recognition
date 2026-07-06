"""Kiểm thử IoU và matching của detection analysis."""

from vlpr.evaluation.detection_analysis import (
    GroundTruth,
    intersection_over_union,
    match_boxes,
)


def test_intersection_over_union_handles_overlap_and_disjoint_boxes() -> None:
    """IoU phải đúng cho cả bbox giao nhau và hoàn toàn tách rời."""
    assert intersection_over_union((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == 25 / 175
    assert intersection_over_union((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) == 0.0


def test_match_boxes_uses_each_ground_truth_once() -> None:
    """Hai prediction trùng nhau không được match cùng một ground truth hai lần."""
    targets = [GroundTruth(box=(0.0, 0.0, 10.0, 10.0), size_group="small")]
    predictions = [
        ((0.0, 0.0, 10.0, 10.0), 0.9),
        ((0.0, 0.0, 10.0, 10.0), 0.8),
    ]

    result = match_boxes(targets, predictions, iou_threshold=0.5)

    assert result.matched_ground_truth == frozenset({0})
    assert result.matched_predictions == frozenset({0})


def test_match_boxes_rejects_poor_localization() -> None:
    """Prediction confidence cao vẫn phải bị từ chối nếu IoU không đạt."""
    targets = [GroundTruth(box=(0.0, 0.0, 10.0, 10.0), size_group="large")]
    predictions = [((8.0, 8.0, 18.0, 18.0), 0.95)]

    result = match_boxes(targets, predictions, iou_threshold=0.5)

    assert not result.matched_ground_truth
    assert not result.matched_predictions
