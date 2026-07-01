"""Read-only inspection of the local Python and accelerator environment."""

import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PACKAGES: tuple[str, ...] = (
    "torch",
    "ultralytics",
    "paddlepaddle",
    "paddleocr",
    "wandb",
)


@dataclass(frozen=True)
class GpuInfo:
    """Physical GPU information reported by the NVIDIA driver."""

    name: str
    memory_mib: int
    driver_version: str


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable snapshot of the current execution environment."""

    python_version: str
    operating_system: str
    logical_cpu_count: int | None
    package_versions: dict[str, str | None]
    physical_gpus: tuple[GpuInfo, ...]
    torch_cuda_build: str | None
    torch_cuda_available: bool

    def to_json(self) -> str:
        """Serialize the report for machines and CI logs."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def inspect_environment() -> EnvironmentReport:
    """Collect environment facts without modifying installed packages."""
    cuda_build, cuda_available = _inspect_torch()
    return EnvironmentReport(
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        logical_cpu_count=os.cpu_count(),
        package_versions={name: _package_version(name) for name in _PACKAGES},
        physical_gpus=_inspect_nvidia_gpus(),
        torch_cuda_build=cuda_build,
        torch_cuda_available=cuda_available,
    )


def write_markdown_report(report: EnvironmentReport, output_path: Path) -> None:
    """Write a human-readable, reproducible environment snapshot."""
    gpu_rows = (
        "\n".join(
            f"| {gpu.name} | {gpu.memory_mib} | {gpu.driver_version} |"
            for gpu in report.physical_gpus
        )
        or "| Không phát hiện | — | — |"
    )
    package_rows = "\n".join(
        f"| `{name}` | {version or 'chưa cài'} |"
        for name, version in report.package_versions.items()
    )
    runtime = "GPU" if report.torch_cuda_available else "CPU"
    content = f"""# Environment report

Report này được sinh bởi
`python scripts/check_environment.py --output reports/environment_report.md`.

## Runtime

- Python: `{report.python_version}`
- Hệ điều hành: `{report.operating_system}`
- Logical CPU: `{report.logical_cpu_count}`
- PyTorch runtime hiện dùng: **{runtime}**
- CUDA build của PyTorch: `{report.torch_cuda_build or "không có"}`

## GPU vật lý

| GPU | VRAM (MiB) | Driver |
|---|---:|---|
{gpu_rows}

## Python packages

| Package | Version |
|---|---|
{package_rows}

## Kết luận Gate 0

Máy local chỉ được xem là GPU runtime khi framework báo CUDA khả dụng. Việc có GPU vật lý
không đồng nghĩa PyTorch hiện tại sử dụng được GPU. Train nặng sẽ dùng cloud nếu dòng
`PyTorch runtime hiện dùng` là `CPU`; local vẫn dùng để phát triển, test và inference CPU.

PaddlePaddle/PaddleOCR chỉ được cài ở gate OCR. W&B dùng chế độ online khi có
`WANDB_API_KEY`; nếu không có key thì tự chuyển sang offline.
"""
    output_path.write_text(content, encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _inspect_torch() -> tuple[str | None, bool]:
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError:
        return None, False
    return torch.version.cuda, bool(torch.cuda.is_available())


def _inspect_nvidia_gpus() -> tuple[GpuInfo, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()

    gpus: list[GpuInfo] = []
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3:
            continue
        try:
            memory_mib = int(fields[1])
        except ValueError:
            continue
        gpus.append(GpuInfo(fields[0], memory_mib, fields[2]))
    return tuple(gpus)
