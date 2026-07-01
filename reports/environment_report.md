# Environment report

Report này được sinh bởi
`python scripts/check_environment.py --output reports/environment_report.md`.

## Runtime

- Python: `3.12.9`
- Hệ điều hành: `Windows-10-10.0.19045-SP0`
- Logical CPU: `8`
- PyTorch runtime hiện dùng: **CPU**
- CUDA build của PyTorch: `không có`

## GPU vật lý

| GPU | VRAM (MiB) | Driver |
|---|---:|---|
| Quadro M2200 | 4096 | 573.22 |

## Python packages

| Package | Version |
|---|---|
| `torch` | 2.6.0 |
| `ultralytics` | chưa cài |
| `paddlepaddle` | chưa cài |
| `paddleocr` | chưa cài |
| `wandb` | 0.28.0 |

## Kết luận Gate 0

Máy local chỉ được xem là GPU runtime khi framework báo CUDA khả dụng. Việc có GPU vật lý
không đồng nghĩa PyTorch hiện tại sử dụng được GPU. Train nặng sẽ dùng cloud nếu dòng
`PyTorch runtime hiện dùng` là `CPU`; local vẫn dùng để phát triển, test và inference CPU.

PaddlePaddle/PaddleOCR chỉ được cài ở gate OCR. W&B dùng chế độ online khi có
`WANDB_API_KEY`; nếu không có key thì tự chuyển sang offline.
