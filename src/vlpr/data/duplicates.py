"""Tạo fingerprint ảnh và gom các file trùng nội dung byte."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vlpr.data.hashing import difference_hash, sha256_file

_PERCEPTUAL_HASH_BITS = 64


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """Gắn đường dẫn ảnh với exact hash và perceptual hash."""

    path: Path
    sha256: str
    perceptual_hash: str


@dataclass(frozen=True, slots=True)
class NearDuplicatePair:
    """Mô tả hai ảnh khác byte nhưng có perceptual hash gần nhau."""

    first: Path
    second: Path
    distance: int


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


def find_near_duplicate_pairs(
    fingerprints: Iterable[ImageFingerprint],
    *,
    max_distance: int,
) -> tuple[NearDuplicatePair, ...]:
    """Tìm cặp gần trùng bằng banding rồi xác nhận bằng Hamming distance."""
    if not 0 <= max_distance < _PERCEPTUAL_HASH_BITS:
        raise ValueError("max_distance phải nằm trong khoảng từ 0 đến 63")

    ordered = sorted(fingerprints, key=lambda item: item.path.as_posix())
    hash_values = [_parse_perceptual_hash(item.perceptual_hash) for item in ordered]
    band_count = max_distance + 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()

    for current_index, hash_value in enumerate(hash_values):
        for band_index, band_value in enumerate(_split_into_bands(hash_value, band_count)):
            bucket = buckets[(band_index, band_value)]
            candidates.update((previous_index, current_index) for previous_index in bucket)
            bucket.append(current_index)

    pairs: list[NearDuplicatePair] = []
    for first_index, second_index in sorted(candidates):
        first = ordered[first_index]
        second = ordered[second_index]
        if first.sha256 == second.sha256:
            continue
        distance = (hash_values[first_index] ^ hash_values[second_index]).bit_count()
        if distance <= max_distance:
            pairs.append(
                NearDuplicatePair(
                    first=first.path,
                    second=second.path,
                    distance=distance,
                )
            )
    return tuple(pairs)


def _parse_perceptual_hash(value: str) -> int:
    """Chuyển dHash 16 ký tự thành số nguyên 64 bit có kiểm tra cấu trúc."""
    if len(value) != 16:
        raise ValueError("perceptual hash phải có đúng 16 ký tự hexadecimal")
    try:
        return int(value, 16)
    except ValueError as exc:
        raise ValueError("perceptual hash phải có đúng 16 ký tự hexadecimal") from exc


def _split_into_bands(hash_value: int, band_count: int) -> tuple[int, ...]:
    """Chia đủ 64 bit thành các đoạn gần bằng nhau để lập candidate index."""
    base_size, wider_band_count = divmod(_PERCEPTUAL_HASH_BITS, band_count)
    bands: list[int] = []
    shift = 0
    for band_index in range(band_count):
        band_size = base_size + (band_index < wider_band_count)
        bands.append((hash_value >> shift) & ((1 << band_size) - 1))
        shift += band_size
    return tuple(bands)
