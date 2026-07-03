"""Kiểm thử fingerprint ảnh và nhóm file trùng chính xác."""

from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.duplicates import (
    ImageFingerprint,
    NearDuplicatePair,
    find_exact_duplicate_groups,
    find_near_duplicate_pairs,
    fingerprint_image,
)


def test_fingerprint_image_contains_both_hash_types(tmp_path: Path) -> None:
    """Xác nhận một lần fingerprint tạo đủ hash byte và hash thị giác."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    fingerprint = fingerprint_image(image_path)

    assert fingerprint.path == image_path
    assert len(fingerprint.sha256) == 64
    assert len(fingerprint.perceptual_hash) == 16


def test_find_exact_duplicate_groups_ignores_unique_files() -> None:
    """Xác nhận chỉ SHA-256 lặp lại mới xuất hiện trong kết quả."""
    fingerprints = [
        ImageFingerprint(Path("unique.jpg"), "sha-a", "dhash-a"),
        ImageFingerprint(Path("copy-2.jpg"), "sha-b", "dhash-b"),
        ImageFingerprint(Path("copy-1.jpg"), "sha-b", "dhash-c"),
    ]

    groups = find_exact_duplicate_groups(fingerprints)

    assert groups == ((Path("copy-1.jpg"), Path("copy-2.jpg")),)


def test_find_exact_duplicate_groups_is_deterministic() -> None:
    """Xác nhận thứ tự input không làm thay đổi thứ tự nhóm và đường dẫn."""
    fingerprints = [
        ImageFingerprint(Path("z-2.jpg"), "sha-z", "dhash-1"),
        ImageFingerprint(Path("a-2.jpg"), "sha-a", "dhash-2"),
        ImageFingerprint(Path("z-1.jpg"), "sha-z", "dhash-1"),
        ImageFingerprint(Path("a-1.jpg"), "sha-a", "dhash-2"),
    ]

    groups = find_exact_duplicate_groups(reversed(fingerprints))

    assert groups == (
        (Path("a-1.jpg"), Path("a-2.jpg")),
        (Path("z-1.jpg"), Path("z-2.jpg")),
    )


def test_find_near_duplicate_pairs_filters_candidates_by_distance() -> None:
    """Xác nhận banding không biến candidate xa thành near-duplicate."""
    fingerprints = [
        ImageFingerprint(Path("base.jpg"), "sha-a", "0000000000000000"),
        ImageFingerprint(Path("near.jpg"), "sha-b", "0000000000000003"),
        ImageFingerprint(Path("far.jpg"), "sha-c", "ffffffffffffffff"),
    ]

    pairs = find_near_duplicate_pairs(fingerprints, max_distance=2)

    assert pairs == (
        NearDuplicatePair(
            first=Path("base.jpg"),
            second=Path("near.jpg"),
            distance=2,
        ),
    )


def test_find_near_duplicate_pairs_excludes_exact_duplicates() -> None:
    """Xác nhận cùng SHA-256 chỉ xuất hiện trong nhóm exact duplicate."""
    fingerprints = [
        ImageFingerprint(Path("copy-1.jpg"), "same-sha", "0000000000000000"),
        ImageFingerprint(Path("copy-2.jpg"), "same-sha", "0000000000000000"),
    ]

    assert find_near_duplicate_pairs(fingerprints, max_distance=6) == ()


def test_find_near_duplicate_pairs_is_deterministic() -> None:
    """Xác nhận thứ tự input không làm thay đổi hướng hoặc thứ tự cặp."""
    fingerprints = [
        ImageFingerprint(Path("z.jpg"), "sha-z", "0000000000000001"),
        ImageFingerprint(Path("a.jpg"), "sha-a", "0000000000000000"),
    ]

    pairs = find_near_duplicate_pairs(reversed(fingerprints), max_distance=1)

    assert pairs == (
        NearDuplicatePair(
            first=Path("a.jpg"),
            second=Path("z.jpg"),
            distance=1,
        ),
    )


def test_find_near_duplicate_pairs_rejects_invalid_input() -> None:
    """Xác nhận threshold và perceptual hash sai được báo trước khi audit."""
    valid = ImageFingerprint(Path("valid.jpg"), "sha", "0000000000000000")
    invalid = ImageFingerprint(Path("invalid.jpg"), "sha", "not-a-hash")

    with pytest.raises(ValueError, match="0 đến 63"):
        find_near_duplicate_pairs([valid], max_distance=64)
    with pytest.raises(ValueError, match="16 ký tự"):
        find_near_duplicate_pairs([invalid], max_distance=6)
