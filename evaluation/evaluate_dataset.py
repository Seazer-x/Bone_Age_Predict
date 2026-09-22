from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from PIL import Image

from bone_age.bone_age import Bone_Age
from evaluation.metrics import bootstrap_mae_ci, regression_metrics


MODEL_NAMES = ["Radius", "Ulna", "MCPFirst", "MCP", "PIP", "PIPFirst", "MIP", "DIP", "DIPFirst"]
AGE_PATTERN = re.compile(r"骨龄约为\s*([0-9]+(?:\.[0-9]+)?)\s*岁")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bone_Age_Predict on a labeled manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("evaluation/results"), type=Path)
    parser.add_argument("--device", default="", help="YOLOv5 device string: '', 'cpu', or CUDA index such as '0'.")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"image", "sex", "age_years"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def parse_age(report: str) -> float:
    match = AGE_PATTERN.search(report)
    if not match:
        raise ValueError("could not parse predicted bone age from model report")
    return float(match.group(1))


def metric_block(rows: list[dict[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    truth = [float(row["age_years"]) for row in rows]
    pred = [float(row["predicted_age_years"]) for row in rows]
    metrics = regression_metrics(truth, pred)
    low, high = bootstrap_mae_ci(truth, pred, iterations=bootstrap, seed=seed)
    metrics["mae_95ci_years"] = [low, high]
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    manifest_dir = args.manifest.resolve().parent

    weights = [repo_root / "bone_age" / name / "best.pt" for name in MODEL_NAMES]
    detector = repo_root / "bone_age" / "bone_age.pt"
    model = Bone_Age(weights, MODEL_NAMES, device=args.device)

    results: list[dict[str, object]] = []

    for index, row in enumerate(load_manifest(args.manifest), start=1):
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = manifest_dir / image_path

        record: dict[str, object] = {
            "image": row["image"],
            "sex": row["sex"].strip().lower(),
            "age_years": float(row["age_years"]),
            "success": False,
            "predicted_age_years": "",
            "absolute_error_years": "",
            "error": "",
        }

        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            report, success = model.run(
                detector,
                record["sex"],
                image,
                conf_thres=args.confidence,
                iou_thres=args.iou,
            )
            if not success:
                raise RuntimeError(report)

            predicted = parse_age(report)
            record["predicted_age_years"] = predicted
            record["absolute_error_years"] = abs(predicted - float(record["age_years"]))
            record["success"] = True
        except Exception as exc:
            record["error"] = str(exc)

        results.append(record)
        print(f"[{index}] {record['image']}: {'ok' if record['success'] else record['error']}")

    results_path = args.output_dir / "predictions.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else [
            "image", "sex", "age_years", "success", "predicted_age_years", "absolute_error_years", "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    successful = [row for row in results if row["success"]]
    summary: dict[str, object] = {
        "manifest": str(args.manifest),
        "total_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "confidence": args.confidence,
        "iou": args.iou,
        "overall": None,
        "by_sex": {},
    }

    if successful:
        summary["overall"] = metric_block(successful, args.bootstrap, args.seed)
        for sex in ("boy", "girl"):
            group = [row for row in successful if row["sex"] == sex]
            if group:
                summary["by_sex"][sex] = metric_block(group, args.bootstrap, args.seed)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Predictions: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
