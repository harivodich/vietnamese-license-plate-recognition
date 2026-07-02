# Data directory

Data files are intentionally excluded from Git.

## Lifecycle

- `raw/`: immutable bytes downloaded from the original source.
- `interim/`: manifests, validation results, hashes, and normalized annotations.
- `processed/`: reproducible train/validation/test views generated from the manifest.
- `external/`: separately sourced images used only for final generalization testing.

Never manually edit `raw/`. Fix parsing or annotation issues through a reviewed correction
manifest so the transformation remains reproducible.

Each Kaggle source is stored by logical task and immutable source version:

```text
raw/kaggle/
├── detection/v1/
│   └── download_receipt.json
└── ocr/v1/
    └── download_receipt.json
```

Downloads are first written to a temporary sibling directory. The downloader publishes that
directory only after it has calculated the file count, byte count, and content fingerprint. A
matching completion receipt makes repeated commands idempotent.

## Credentials

Public Kaggle downloads normally work without credentials. If Kaggle requests authentication,
set `KAGGLE_API_TOKEN` or configure the credentials supported by `kagglehub`. Never commit tokens.
