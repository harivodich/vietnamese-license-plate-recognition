# ADR 0002: Local CPU development with cloud GPU training

- Status: accepted

## Context

The local Conda environment `HariAI` uses Python 3.12.9. The machine has an NVIDIA Quadro M2200
with 4,096 MiB VRAM, but the installed PyTorch 2.6.0 package is a CPU build:

```text
torch.version.cuda = None
torch.cuda.is_available() = False
```

A pip dependency-resolution dry run found compatible Windows CPython 3.12 wheels for
PaddlePaddle 3.3.1, PaddleOCR 3.7.0, and W&B 0.28.0. This proves package availability, not GPU
support or model correctness.

## Decision

- Keep using the existing `HariAI` environment.
- Use local CPU execution for development, tests, data processing, and inference checks.
- Run expensive training on Kaggle or Colab when local CUDA is unavailable or too slow.
- Use the same versioned config, seed, dataset fingerprint, and Git commit locally and in cloud
  runs.
- Install model-specific dependency groups only in the gate that needs them.
- Treat a physical NVIDIA GPU and a CUDA-enabled framework runtime as separate facts.

## Consequences

Cloud and local benchmark results must be reported separately with their hardware details. A
future CUDA-enabled PyTorch installation may replace the CPU build only after a compatibility
smoke test; it must not be assumed from `nvidia-smi` output alone.

## Runtime update

The shared `HariAI` environment now contains PyTorch 2.6.0 with CUDA 12.4. A real CUDA matrix
multiplication completed successfully on the Quadro M2200, so local detection experiments may use
this GPU with conservative batch sizes. PaddlePaddle 3.3.1 is currently a CPU build. The hybrid
decision remains valid: cloud GPU is used when local VRAM or training time is insufficient, and
results from different hardware remain separate.
