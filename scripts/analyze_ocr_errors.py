"""Analyze OCR failure examples and copy images for manual review."""

import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    metrics_path = (
        root / "artifacts" / "ocr" / "paddleocr-v5-mobile-finetune" / "eval" / "metrics.json"
    )
    dataset_root = root / "data" / "raw" / "kaggle" / "ocr" / "v1" / "lp_ocr_dataset_vi"
    output_dir = root / "artifacts" / "ocr" / "paddleocr-v5-mobile-finetune" / "eval" / "failures"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    failures = metrics.get("failure_examples", [])
    if not failures:
        print("No failure examples found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing failures directory
    for p in output_dir.iterdir():
        if p.is_file():
            p.unlink()

    print(f"Found {len(failures)} failure examples. Copying to {output_dir.relative_to(root)}...")

    for i, example in enumerate(failures, start=1):
        image_path = example["image_path"]
        source_path = dataset_root / image_path

        if not source_path.exists():
            print(f"WARNING: Image not found at {source_path}")
            continue

        gt = example["normalized_ground_truth"]
        pred = example["normalized_prediction"]
        ed = example["edit_distance"]

        # Sanitize filenames
        gt_safe = gt.replace("/", "_").replace("\\", "_")
        pred_safe = pred.replace("/", "_").replace("\\", "_")

        filename = f"{i:02d}_ed{ed}_GT_{gt_safe}_Pred_{pred_safe}{source_path.suffix}"
        dest_path = output_dir / filename

        shutil.copy2(source_path, dest_path)

    print("Done! Check the failures directory to visually inspect the worst errors.")


if __name__ == "__main__":
    main()
