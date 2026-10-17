"""Evaluate the saved five-class classifier and current two-stage pipeline.

Run from the repository root with the project's Python environment:
    python scripts/diagnose_mosquito.py

The script reads held-out data and writes a JSON report under runs/diagnostics.
It does not modify datasets or checkpoints.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.uav_yolonet import UAV_YOLONet


DEFAULT_DATA = ROOT / "processed" / "five_class_grouped"
DEFAULT_WEIGHTS = ROOT / "models" / "best_uav_yolonet_grouped.pth"
DEFAULT_DETECTOR = ROOT / "yolo11s.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "grouped_inference.json"
FALLBACK_CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]


def metrics_from_confusion(cm, classes):
    result = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i].sum()),
            "false_positives": fp,
            "false_negatives": fn,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-pipeline", action="store_true", help="Only evaluate full-image classifier")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes", FALLBACK_CLASSES)
    image_size = int(checkpoint.get("image_size", 160))
    if "mosquito" not in classes:
        raise ValueError(f"Checkpoint lacks mosquito class: {classes}")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = UAV_YOLONet(num_classes=len(classes), in_channels=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_to_idx = {name: i for i, name in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    classifier_ms = []
    test_records = []
    counts = {}

    for true_class in classes:
        files = sorted(p for p in (args.data / "test" / true_class).glob("*") if p.is_file())
        counts[true_class] = len(files)
        for path in files:
            with Image.open(path) as im:
                tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
            start = time.perf_counter()
            with torch.inference_mode():
                probs = F.softmax(model(tensor), dim=1)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            classifier_ms.append((time.perf_counter() - start) * 1000)
            probs = probs.cpu().numpy()
            pred_idx = int(probs.argmax())
            cm[label_to_idx[true_class], pred_idx] += 1
            test_records.append({
                "filename": path.name,
                "true": true_class,
                "full_image_pred": classes[pred_idx],
                "full_image_confidence": float(probs[pred_idx]),
                "full_image_mosquito_probability": float(probs[label_to_idx["mosquito"]]),
            })

    total = int(cm.sum())
    val_records = []
    mosquito_idx = label_to_idx["mosquito"]
    for true_class in classes:
        for path in sorted(p for p in (args.data / "val" / true_class).glob("*") if p.is_file()):
            with Image.open(path) as im:
                tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
            val_records.append({
                "true": true_class,
                "pred_idx": int(probs.argmax()),
                "mosquito_probability": float(probs[mosquito_idx]),
            })
    candidate_thresholds = sorted({0.0, 1.0, *(r["mosquito_probability"] for r in val_records)})
    threshold_results = []
    for threshold in candidate_thresholds:
        tp = sum(r["true"] == "mosquito" and r["pred_idx"] == mosquito_idx and r["mosquito_probability"] >= threshold for r in val_records)
        fp = sum(r["true"] != "mosquito" and r["pred_idx"] == mosquito_idx and r["mosquito_probability"] >= threshold for r in val_records)
        fn = sum(r["true"] == "mosquito" and not (r["pred_idx"] == mosquito_idx and r["mosquito_probability"] >= threshold) for r in val_records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold_results.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    eligible = [r for r in threshold_results if r["precision"] >= 0.99 and (r["tp"] + r["fp"]) > 0]
    calibrated = max(eligible, key=lambda r: (r["recall"], r["threshold"])) if eligible else None
    report = {
        "checkpoint": str(args.weights),
        "checkpoint_architecture": checkpoint.get("architecture", "UAV_YOLONet"),
        "classes": classes,
        "class_to_idx": checkpoint.get("class_to_idx", label_to_idx),
        "input_size": image_size,
        "device": str(device),
        "dataset": str(args.data),
        "test_counts": counts,
        "full_image_classifier": {
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
            "metrics": metrics_from_confusion(cm, classes),
            "accuracy": float(np.trace(cm) / total) if total else 0.0,
            "mean_latency_ms": float(np.mean(classifier_ms)) if classifier_ms else None,
            "p95_latency_ms": float(np.percentile(classifier_ms, 95)) if classifier_ms else None,
        },
        "validation_mosquito_threshold_calibration": {
            "rule": "maximize recall among thresholds with at least 0.99 precision, requiring mosquito argmax",
            "selected": calibrated,
            "candidates": threshold_results,
        },
        "records": test_records,
    }

    if not args.skip_pipeline:
        from scripts.detect_and_classify import MultiTargetAirDefenseSystem

        system = MultiTargetAirDefenseSystem(
            str(args.detector), str(args.weights), conf_thresh=0.12, iou_thresh=0.45
        )
        pipeline_classes = list(classes) + ["no_detection"]
        pipeline_cm = np.zeros((len(pipeline_classes), len(pipeline_classes)), dtype=np.int64)
        pipeline_records = []
        pipeline_ms = []
        for true_class in pipeline_classes:
            for path in sorted(p for p in (args.data / "test" / true_class).glob("*") if p.is_file()):
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                start = time.perf_counter()
                _, detections = system.process_frame(frame)
                pipeline_ms.append((time.perf_counter() - start) * 1000)
                primary = max(detections, key=lambda d: d["confidence"]) if detections else None
                pred = primary["class"] if primary else "no_detection"
                if pred not in pipeline_classes:
                    pred = "no_detection"
                pipeline_cm[pipeline_classes.index(true_class), pipeline_classes.index(pred)] += 1
                pipeline_records.append({
                    "filename": path.name,
                    "true": true_class,
                    "pred": pred,
                    "confidence": float(primary["confidence"]) if primary else 0.0,
                    "detection_count": len(detections),
                })
        report["current_detector_crop_pipeline"] = {
            "matrix_class_order": pipeline_classes,
            "confusion_matrix_rows_true_columns_predicted": pipeline_cm.tolist(),
            "metrics": metrics_from_confusion(pipeline_cm, pipeline_classes),
            "no_detection": sum(r["pred"] == "no_detection" for r in pipeline_records),
            "evaluated": len(pipeline_records),
            "mean_latency_ms": float(np.mean(pipeline_ms)) if pipeline_ms else None,
            "p95_latency_ms": float(np.percentile(pipeline_ms, 95)) if pipeline_ms else None,
            "records": pipeline_records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()
