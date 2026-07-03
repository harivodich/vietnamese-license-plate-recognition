"""Kiểm thử fingerprint ảnh và nhóm file trùng chính xác."""

from pathlib import Path

from PIL import Image

from vlpr.data.duplicates import (
    ImageFingerprint,
    find_exact_duplicate_groups,
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
