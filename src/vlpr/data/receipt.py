"""Receipt có kiểm tra kiểu và fingerprint xác định cho dữ liệu raw."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vlpr.config import DatasetSettings

RECEIPT_NAME = "download_receipt.json"


class DownloadReceipt(BaseModel):
    """Bằng chứng một nguồn dữ liệu bất biến đã tải và fingerprint hoàn tất."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete"]
    dataset_name: str
    dataset_task: str
    dataset_handle: str
    dataset_version: int
    dataset_url: str
    expected_license_from_data_card: str
    layout_root: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)


def fingerprint_directory(directory: Path) -> tuple[str, int, int]:
    """Tính SHA-256 xác định, số file và tổng byte cho toàn bộ cây thư mục."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != RECEIPT_NAME
    )
    for path in files:
        relative_path = path.relative_to(directory).as_posix()
        file_digest, file_size = _fingerprint_file(path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += file_size
    return digest.hexdigest(), file_count, total_bytes


def read_receipt(target_dir: Path) -> DownloadReceipt | None:
    """Đọc receipt hợp lệ; trả ``None`` khi file thiếu, hỏng hoặc sai schema."""
    try:
        raw: Any = json.loads((target_dir / RECEIPT_NAME).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        supported = {
            key: value for key, value in raw.items() if key in DownloadReceipt.model_fields
        }
        return DownloadReceipt.model_validate(supported)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValidationError):
        return None


def receipt_matches(
    receipt: DownloadReceipt | None,
    dataset_name: str,
    dataset: DatasetSettings,
) -> bool:
    """Kiểm tra receipt có đúng tên, Kaggle handle và version trong cấu hình không."""
    return bool(
        receipt is not None
        and receipt.dataset_name == dataset_name
        and receipt.dataset_handle == dataset.handle
        and receipt.dataset_version == dataset.version
    )


def write_receipt(
    staging_dir: Path,
    dataset_name: str,
    dataset: DatasetSettings,
    resolved_download: Path,
) -> DownloadReceipt:
    """Fingerprint dữ liệu staging, tạo receipt có kiểu rồi ghi receipt xuống đĩa."""
    tree_sha256, file_count, total_bytes = fingerprint_directory(staging_dir)
    try:
        layout_root = resolved_download.relative_to(staging_dir).as_posix()
    except ValueError:
        layout_root = "."

    receipt = DownloadReceipt(
        status="complete",
        dataset_name=dataset_name,
        dataset_task=dataset.task,
        dataset_handle=dataset.handle,
        dataset_version=dataset.version,
        dataset_url=f"https://www.kaggle.com/datasets/{dataset.handle}",
        expected_license_from_data_card=dataset.expected_license,
        layout_root=layout_root,
        content_sha256=tree_sha256,
        file_count=file_count,
        total_bytes=total_bytes,
    )
    (staging_dir / RECEIPT_NAME).write_text(
        receipt.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _fingerprint_file(path: Path) -> tuple[str, int]:
    """Đọc file theo chunk 1 MiB để tính SHA-256 và kích thước mà không tốn nhiều RAM."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
