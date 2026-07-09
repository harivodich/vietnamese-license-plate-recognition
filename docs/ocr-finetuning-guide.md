# OCR fine-tuning guide

## Current state

The project uses PaddleOCR pretrained recognition as the main OCR direction. The installed
`paddleocr` package is suitable for inference through `TextRecognition`, but it does not expose a
local training CLI. Fine-tuning therefore needs the official PaddleOCR training repository, while
this project owns the dataset export, evaluation protocol, reports, and post-processing.

## Export fine-tune data

```powershell
python scripts/export_ocr_finetune_data.py --config configs/ocr-finetune-paddleocr.yaml
```

The exporter writes PaddleOCR recognition-format data to:

```text
data/processed/ocr_finetune_paddleocr/
|-- images/
|   |-- train/
|   |-- validation/
|   `-- test/
|-- train_list.txt
|-- val_list.txt
|-- test_list.txt
|-- dict.txt
`-- summary.json
```

Each label file uses:

```text
relative/image/path.png<TAB>TEXT
```

Wide plate crops remain one sample. Compact crops are split into top and bottom line crops, because
PP-OCR recognition models read one text line at a time.

## Exported dataset summary

| Split | Line samples |
| --- | ---: |
| Train | 7595 |
| Validation | 1249 |
| Test | 1262 |

The exported charset has 37 characters: digits, `A-Z`, and `Đ`.


## Local training status

Prepared locally:

- official PaddleOCR repo cloned to `external/PaddleOCR`;
- PaddleOCR training requirements installed in `HariAI`;
- official `en_PP-OCRv5_mobile_rec_pretrained.pdparams` downloaded to `artifacts/ocr/paddleocr-v5-mobile-finetune/pretrained/`;
- generated train config written to `artifacts/ocr/paddleocr-v5-mobile-finetune/train_config.yml`;
- smoke train passed on a tiny subset and saved checkpoints under `artifacts/ocr/paddleocr-v5-mobile-finetune/smoke/`.

Current limitation:

```text
paddle.device.is_compiled_with_cuda() == False
```

So this environment can launch PaddleOCR training on CPU, but full fine-tuning will be slow. PyTorch
CUDA working for YOLO does not imply Paddle CUDA is installed. Do not install `paddlepaddle-gpu 2.6.x`
blindly over Paddle 3.x just because `pip index` shows it; that may downgrade the framework below
what PaddleOCR 3.x expects.

## PaddleOCR training command shape

Run this from a local clone of the official PaddleOCR repository, after installing its training
requirements in the same environment or a compatible Paddle environment.

```powershell
python scripts/prepare_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml

python scripts/train_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml
```

The generated config uses the official `en_PP-OCRv5_mobile_rec_pretrained.pdparams` training checkpoint. The final charset-specific heads are reinitialized because the project dictionary has 37 license-plate characters.

## Checkpoints and resume

PaddleOCR writes checkpoints under:

```text
artifacts/ocr/paddleocr-v5-mobile-finetune/checkpoints/
```

Important checkpoint prefixes:

- `latest`: saved every epoch; use this after a power cut or interrupted run.
- `best_accuracy`: best validation checkpoint selected by PaddleOCR's recognition metric.
- `iter_epoch_5`, `iter_epoch_10`, ...: periodic snapshots from `save_epoch_step: 5`.

Resume from the latest checkpoint with either the prefix or `.pdparams` file:

```powershell
python scripts/train_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml --resume artifacts/ocr/paddleocr-v5-mobile-finetune/checkpoints/latest
```

## Evaluation rule

After fine-tuning, export or load the trained recognizer and evaluate it with the same project
protocol:

```powershell
python scripts/evaluate_ocr.py --config configs/ocr-baseline-layout.yaml
```

The report must keep:

- full-plate exact match;
- CER;
- character accuracy;
- compact/wide subgroup metrics;
- CPU/GPU latency;
- failure examples.

Do not tune row-split thresholds or model choices repeatedly on the test split. Use validation for
selection, then run the fixed test protocol once for the final comparison.
