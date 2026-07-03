"""Tạo fingerprint ảnh và gom các file trùng nội dung byte."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vlpr.data.hashing import difference_hash, sha256_file


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """Gắn đường dẫn ảnh với exact hash và perceptual hash."""

    path: Path
    sha256: str
    perceptual_hash: str


def fingerprint_image(path: Path) -> ImageFingerprint:
    """Tính một lần các dấu vân tay cần cho exact và near-duplicate audit."""
    return ImageFingerprint(
        path=path,
        sha256=sha256_file(path),
        perceptual_hash=difference_hash(path),
    )


def find_exact_duplicate_groups(
    fingerprints: Iterable[ImageFingerprint],
) -> tuple[tuple[Path, ...], ...]:
    """Gom các đường dẫn có cùng SHA-256 và chỉ trả nhóm có từ hai file."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for fingerprint in fingerprints:
        buckets[fingerprint.sha256].append(fingerprint.path)

    groups = [
        tuple(sorted(paths, key=lambda path: path.as_posix()))
        for paths in buckets.values()
        if len(paths) > 1
    ]
    return tuple(sorted(groups, key=lambda group: group[0].as_posix()))
