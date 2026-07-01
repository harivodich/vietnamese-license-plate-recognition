# Dataset report

## Status

Downloaded and structurally inventoried. Image integrity, visual label quality, duplicate leakage,
source grouping, and license evidence still require validation.

## Source

- Kaggle handle: `miahuynh04/vietnamese-license-plate-detection`
- Dataset version: `1`
- Retrieved at: `2026-07-01T07:28:32Z`
- Data-card license claim: MIT
- Downloaded size after extraction: 366,309,079 bytes

The license value above is not yet treated as verified provenance. A copy of the license or other
primary evidence distributed with the dataset has not been found in the downloaded files.

## Structural inventory

| Split | Images | Label files | Bounding boxes | Multi-plate images |
|---|---:|---:|---:|---:|
| Train | 6,607 | 6,607 | 6,785 | 117 |
| Validation | 814 | 814 | 818 | 3 |
| Test | 838 | 838 | 849 | 9 |
| **Total** | **8,259** | **8,259** | **8,452** | **129** |

All images currently use the `.jpg` extension. Labels use YOLO detection format with one class:

```text
class_id center_x center_y width height
```

The initial structural scan found:

- zero missing image-label pairs;
- zero orphan label files;
- zero empty label files;
- no rows with a field count other than five;
- no normalized coordinate values outside `[0, 1]`;
- no OCR text labels.

## Current limitations

- The source-provided split has not been accepted as the project split.
- No video, vehicle, capture-session, or duplicate group identifiers are present.
- The absence of OCR text labels means this dataset supports detection only.
- Images have not yet been decoded to detect corruption or measure resolution.
- Bounding boxes have not yet been checked visually.
- Exact and near-duplicate leakage has not yet been measured.
- An external test set has not been selected.
