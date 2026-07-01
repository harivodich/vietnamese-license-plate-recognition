# Data directory

Data files are intentionally excluded from Git.

## Lifecycle

- `raw/`: immutable bytes downloaded from the original source.
- `interim/`: manifests, validation results, hashes, and normalized annotations.
- `processed/`: reproducible train/validation/test views generated from the manifest.
- `external/`: separately sourced images used only for final generalization testing.

Never manually edit `raw/`. Fix parsing or annotation issues through a reviewed correction
manifest so the transformation remains reproducible.

## Credentials

Public Kaggle downloads normally work without credentials. If Kaggle requests authentication,
set `KAGGLE_API_TOKEN` or configure the credentials supported by `kagglehub`. Never commit tokens.
