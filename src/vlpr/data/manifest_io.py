"""Đọc và ghi manifest JSONL có kiểm tra schema và publish nguyên tử."""

import os
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from vlpr.data.manifest_schema import ManifestRecord

_MANIFEST_ADAPTER: TypeAdapter[ManifestRecord] = TypeAdapter(ManifestRecord)


class ManifestReadError(ValueError):
    """Báo vị trí một dòng JSONL không khớp schema manifest."""


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    """Ghi record vào file tạm rồi publish nguyên tử thành manifest đích."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(record.model_dump_json())
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def read_manifest(path: Path) -> tuple[ManifestRecord, ...]:
    """Đọc từng dòng JSONL và khôi phục đúng record detection hoặc OCR."""
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ManifestReadError(f"{path}:{line_number}: dòng manifest rỗng")
        try:
            records.append(_MANIFEST_ADAPTER.validate_json(line))
        except ValidationError as exc:
            raise ManifestReadError(f"{path}:{line_number}: record manifest không hợp lệ") from exc
    return tuple(records)
