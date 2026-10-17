"""Evaluate an Ultralytics detector against YOLO-format test annotations.

The command also writes an audit-only report when required detector weights or
box labels are unavailable; it never substitutes classifier labels for boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WEIGHTS = ROOT / "runs/anti_uav/yolo_drone_s_v1/weights/best.pt"
DEFAULT_DATA = ROOT / "processed/yolo_drone/data.yaml"
DEFAULT_OUTPUT = ROOT / "runs/anti_uav_test_evaluation"


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def audit_classification_split(test_dir: Path | None):
    """Read-only SHA-256 and difference-hash overlap check for a class-folder split."""
    if not test_dir or not test_dir.is_dir() or test_dir.name.lower() != "test":
        return None
    root = test_dir.parent
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        return None

    def collect(split):
        out = []
        for path in image_files(root / split):
            try:
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as im:
                    pixels = list(im.convert("L").resize((9, 8)).tobytes())
                bits = 0
                for y in range(8):
                    for x in range(8):
                        bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
                out.append({"path": str(path), "sha256": sha, "dhash": bits})
            except Exception:
                continue
        return out

    train, val, test = collect("train"), collect("val"), collect("test")
    class_counts = {split: dict(Counter(Path(row["path"]).parent.name for row in rows))
                    for split, rows in (("train", train), ("val", val), ("test", test))}
    known = train + val
    known_hashes = defaultdict(list)
    for row in known:
        known_hashes[row["sha256"]].append(row["path"])
    exact = []
    near = []
    for row in test:
        for other in known_hashes.get(row["sha256"], []):
            exact.append({"test": row["path"], "train_or_val": other})
        for other in known:
            if row["sha256"] != other["sha256"] and (row["dhash"] ^ other["dhash"]).bit_count() <= 4:
                near.append({"test": row["path"], "train_or_val": other["path"],
                             "dhash_distance": (row["dhash"] ^ other["dhash"]).bit_count()})
    return {"root": str(root.resolve()), "hash_method": "SHA-256 exact; 64-bit difference hash near-duplicate candidate threshold <= 4",
            "counts": {"train": len(train), "val": len(val), "test": len(test)},
            "class_counts": class_counts,
            "exact_cross_split_duplicate_pairs": len(exact), "exact_examples": exact[:100],
            "near_duplicate_candidate_pairs": len(near), "near_duplicate_examples": near[:100],
            "near_duplicate_is_candidate_only": True}


def names_dict(value) -> dict[int, str]:
    if isinstance(value, list):
        return {i: str(name) for i, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(i): str(name) for i, name in value.items()}
    return {}


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def resolve_dataset_path(data: dict, yaml_path: Path, key: str) -> Path | None:
    value = data.get(key)
    if not value:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        base = Path(str(data.get("path", yaml_path.parent)))
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()
        p = base / p
    return p.resolve()


def locate_labels(images_dir: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            candidate = Path(*parts[:i], "labels", *parts[i + 1 :])
            if candidate.is_dir():
                return candidate
    candidate = images_dir.parent / "labels" / images_dir.name
    return candidate if candidate.is_dir() else None


def label_path_for(images_dir: Path, labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def read_yolo_label(path: Path, width: int, height: int, nclasses: int):
    boxes, errors = [], []
    if not path.exists():
        return boxes, ["missing label file"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return boxes, [f"cannot read label: {exc}"]
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            errors.append(f"line {lineno}: expected class and four box values")
            continue
        try:
            cls = int(values[0])
            cx, cy, w, h = map(float, values[1:])
        except ValueError:
            errors.append(f"line {lineno}: non-numeric value")
            continue
        if cls < 0 or cls >= nclasses:
            errors.append(f"line {lineno}: class id {cls} outside [0,{nclasses - 1}]")
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0 or min(cx, cy) < 0 or max(cx, cy) > 1 or w > 1 or h > 1:
            errors.append(f"line {lineno}: invalid normalized box")
            continue
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        if x1 < -1e-6 or y1 < -1e-6 or x2 > width + 1e-6 or y2 > height + 1e-6:
            errors.append(f"line {lineno}: box extends outside image")
            continue
        boxes.append({"class_id": cls, "xyxy": [x1, y1, x2, y2]})
    return boxes, errors


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def match_class(preds, gts, threshold):
    """Greedy score-ordered one-to-one, class-aware matches."""
    used = set()
    matches = []
    for pi in sorted(range(len(preds)), key=lambda i: preds[i]["confidence"], reverse=True):
        candidates = [(iou(preds[pi]["xyxy"], gt["xyxy"]), gi) for gi, gt in enumerate(gts)
                      if gi not in used and gt["class_id"] == preds[pi]["class_id"]]
        if candidates:
            ov, gi = max(candidates)
            if ov >= threshold:
                used.add(gi)
                matches.append((pi, gi, ov))
    return matches


def match_any_class(preds, gts, threshold):
    used_gt, used_pred, matches = set(), set(), []
    pairs = sorted(((iou(p["xyxy"], g["xyxy"]), pi, gi)
                    for pi, p in enumerate(preds) for gi, g in enumerate(gts)), reverse=True)
    for ov, pi, gi in pairs:
        if ov < threshold:
            break
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi); used_gt.add(gi); matches.append((pi, gi, ov))
    return matches, used_pred, used_gt


def ap101(tp_flags, confidences, n_gt):
    if n_gt == 0:
        return None
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    tp = fp = 0
    points = []
    for idx in order:
        tp += int(tp_flags[idx]); fp += int(not tp_flags[idx])
        points.append((tp / max(1, n_gt), tp / max(1, tp + fp)))
    return sum(max((p for r, p in points if r >= recall), default=0.0) for recall in [i / 100 for i in range(101)]) / 101


def metrics_for(all_preds, all_gts, class_names, conf, iou_threshold):
    n = len(class_names)
    cm = [[0 for _ in range(n + 1)] for __ in range(n + 1)]  # final index is background
    tp_by_class = Counter(); fp_by_class = Counter(); fn_by_class = Counter()
    matched_count = 0
    for preds, gts in zip(all_preds, all_gts):
        active = [p for p in preds if p["confidence"] >= conf]
        correct = match_class(active, gts, iou_threshold)
        used_p, used_g = {m[0] for m in correct}, {m[1] for m in correct}
        for pi, gi, _ in correct:
            cls = active[pi]["class_id"]; tp_by_class[cls] += 1; cm[gts[gi]["class_id"]][cls] += 1
        any_matches, any_p, any_g = match_any_class(active, gts, iou_threshold)
        for pi, gi, _ in any_matches:
            if pi not in used_p and gi not in used_g:
                cm[gts[gi]["class_id"]][active[pi]["class_id"]] += 1
                used_p.add(pi); used_g.add(gi)
        for pi, p in enumerate(active):
            if pi not in used_p:
                fp_by_class[p["class_id"]] += 1
                cm[n][p["class_id"]] += 1
        for gi, gt in enumerate(gts):
            if gi not in used_g:
                fn_by_class[gt["class_id"]] += 1
                cm[gt["class_id"]][n] += 1
        matched_count += len(correct)

    per_class = {}
    for cid, name in enumerate(class_names):
        tp, fp, fn = tp_by_class[cid], fp_by_class[cid], fn_by_class[cid]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0,
                           "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                           "ground_truth_objects": sum(g["class_id"] == cid for gs in all_gts for g in gs)}

    aps = {"0.50": {}, "0.50:0.95": {}}
    for cid, name in enumerate(class_names):
        n_gt = sum(g["class_id"] == cid for gs in all_gts for g in gs)
        if not n_gt:
            continue
        for thr in [0.5 + k * 0.05 for k in range(10)]:
            confidences, flags = [], []
            for preds, gts in zip(all_preds, all_gts):
                cp = [p for p in preds if p["class_id"] == cid]
                cg = [g for g in gts if g["class_id"] == cid]
                matches = match_class(cp, cg, thr)
                good = {m[0] for m in matches}
                confidences.extend(p["confidence"] for p in cp)
                flags.extend(i in good for i in range(len(cp)))
            val = ap101(flags, confidences, n_gt)
            if thr == 0.5:
                aps["0.50"][name] = val
            aps["0.50:0.95"][name] = val
        # overwrite with mean AP across IoU thresholds
        vals = []
        for thr in [0.5 + k * 0.05 for k in range(10)]:
            confidences, flags = [], []
            for preds, gts in zip(all_preds, all_gts):
                cp = [p for p in preds if p["class_id"] == cid]
                cg = [g for g in gts if g["class_id"] == cid]
                good = {m[0] for m in match_class(cp, cg, thr)}
                confidences.extend(p["confidence"] for p in cp)
                flags.extend(i in good for i in range(len(cp)))
            vals.append(ap101(flags, confidences, n_gt))
        aps["0.50:0.95"][name] = sum(vals) / len(vals)

    p_total = sum(tp_by_class.values()); predicted = sum(tp_by_class.values()) + sum(fp_by_class.values())
    gt_total = sum(len(g) for g in all_gts)
    recall = p_total / gt_total if gt_total else 0.0
    precision = p_total / predicted if predicted else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "mAP50": sum(aps["0.50"].values())/len(aps["0.50"]) if aps["0.50"] else None,
            "mAP50_95": sum(aps["0.50:0.95"].values())/len(aps["0.50:0.95"]) if aps["0.50:0.95"] else None,
            "per_class": per_class, "per_class_ap50": aps["0.50"], "per_class_map50_95": aps["0.50:0.95"],
            "confusion_matrix": cm, "confusion_matrix_labels": class_names + ["background"],
            "ground_truth_objects": gt_total, "predicted_objects": sum(sum(p["confidence"] >= conf for p in preds) for preds in all_preds),
            "raw_predictions_above_predict_floor": sum(len(p) for p in all_preds),
            "correct_detections": p_total, "false_positives": sum(fp_by_class.values()),
            "missed_objects": sum(fn_by_class.values()), "operating_confidence": conf,
            "matching_iou": iou_threshold}


def save_placeholder_png(path: Path, title: str, lines: list[str]):
    im = Image.new("RGB", (1100, 540), "white")
    d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", 24)
    except OSError: font = ImageFont.load_default()
    d.text((35, 28), title, fill="#17324d", font=font)
    for i, line in enumerate(lines):
        d.text((40, 115 + i*48), line, fill="#333333", font=font)
    im.save(path)


def save_confusion_png(path: Path, matrix, labels):
    n = len(labels)
    cell = 110
    left, top = 190, 120
    im = Image.new("RGB", (left + n * cell + 30, top + n * cell + 40), "white")
    d = ImageDraw.Draw(im)
    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        title_font = font = ImageFont.load_default()
    d.text((25, 25), "Detection confusion matrix (rows = ground truth, columns = prediction)", fill="#17324d", font=title_font)
    peak = max((v for row in matrix for v in row), default=0)
    for r in range(n):
        d.text((left - 165, top + r * cell + cell // 2 - 10), labels[r], fill="#222222", font=font)
        d.text((left + r * cell + cell // 2 - 18, top - 28), labels[r], fill="#222222", font=font)
        for c in range(n):
            value = matrix[r][c]
            strength = value / peak if peak else 0
            color = (int(245 - 100 * strength), int(249 - 65 * strength), int(252 - 20 * strength))
            x0, y0 = left + c * cell, top + r * cell
            d.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="#9aa7b2", width=1)
            label = str(value)
            box = d.textbbox((0, 0), label, font=title_font)
            d.text((x0 + (cell - (box[2] - box[0])) / 2, y0 + (cell - (box[3] - box[1])) / 2), label, fill="#152331", font=title_font)
    im.save(path)


def draw_boxes(image_path: Path, output_path: Path, preds, gts, names):
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for gt in gts:
        b = gt["xyxy"]
        d.rectangle(b, outline="#18a558", width=3)
        d.text((b[0], max(0, b[1]-18)), f"GT {names[gt['class_id']]}", fill="#087a3e")
    for p in preds:
        b = p["xyxy"]
        d.rectangle(b, outline="#d52b35", width=3)
        d.text((b[0], max(0, b[1]-18)), f"{names.get(p['class_id'], p['class_id'])} {p['confidence']:.2f}", fill="#b21e2b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--pipeline-run-dir", type=Path,
                        help="Ingest a completed scripts/detect_and_classify.py run for the same images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predict-conf", type=float, default=0.001,
                        help="Low prediction floor so AP includes the ranked confidence curve.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Operating confidence threshold for precision/recall/F1 and saved annotations.")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("annotated_predictions", "false_positives", "missed_detections"):
        (out / name).mkdir(exist_ok=True)

    dataset = load_yaml(args.data)
    yaml_names = names_dict(dataset.get("names"))
    img_dir = args.images_dir.resolve() if args.images_dir else resolve_dataset_path(dataset, args.data.resolve(), "test")
    labels_dir = locate_labels(img_dir, args.labels_dir.resolve() if args.labels_dir else None) if img_dir else None
    if img_dir is None or not img_dir.exists():
        # Show a discovered classification split as an unannotated candidate only.
        for candidate in (ROOT / "processed/five_class_grouped/test", ROOT / "processed/five_class_classification/test",
                          ROOT / "processed/classification/test", ROOT / "processed/balanced_classification/test"):
            if candidate.is_dir():
                img_dir = candidate
                break
    imgs = image_files(img_dir) if img_dir else []
    is_classification_layout = bool(img_dir and any((img_dir / cls).is_dir() for cls in ["bird", "drone", "helicopter", "other", "mosquito"]))
    split_integrity = audit_classification_split(img_dir) if is_classification_layout else None
    label_root = labels_dir

    # The YOLO training YAML has no test split; audit the existing image-folder test data
    # without interpreting its class directory as box annotations.
    test_gts, label_errors, per_image = [], [], []
    if label_root and not is_classification_layout:
        for p in imgs:
            try:
                with Image.open(p) as im: w, h = im.size
            except Exception as exc:
                w = h = 0; label_errors.append({"image": str(p), "error": f"unreadable image: {exc}"})
            gt, errs = read_yolo_label(label_path_for(img_dir, label_root, p), w, h, len(yaml_names))
            test_gts.append(gt)
            if errs:
                label_errors.append({"image": str(p), "errors": errs})

    model = None
    model_names = {}
    inference_error = None
    if args.weights.is_file():
        try:
            from ultralytics import YOLO
            model = YOLO(str(args.weights))
            model_names = names_dict(model.names)
        except Exception as exc:
            inference_error = f"Unable to load model: {type(exc).__name__}: {exc}"
    else:
        inference_error = f"Trained detector checkpoint not found: {args.weights}"

    class_names = [model_names[i] for i in sorted(model_names)] if model_names else [yaml_names[i] for i in sorted(yaml_names)]
    if not class_names:
        class_names = ["drone", "bird", "helicopter", "other"]
    class_map_matches = None if not yaml_names or not model_names else yaml_names == model_names
    all_preds = []
    inference_ms = []
    image_rows = []
    annotation_ok = bool(label_root and imgs and len(test_gts) == len(imgs) and not label_errors and class_map_matches)
    if model and imgs:
        for idx, p in enumerate(imgs):
            t0 = time.perf_counter()
            try:
                result = model.predict(source=str(p), conf=args.predict_conf, iou=args.nms_iou, device=args.device or None,
                                       verbose=False, save=False, max_det=300)[0]
                elapsed = (time.perf_counter() - t0) * 1000
                inference_ms.append(elapsed)
                preds = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.cpu().tolist()
                    confs = result.boxes.conf.cpu().tolist()
                    clss = result.boxes.cls.cpu().tolist()
                    preds = [{"class_id": int(c), "confidence": float(s), "xyxy": [float(v) for v in b]}
                             for b, s, c in zip(xyxy, confs, clss)]
                all_preds.append(preds)
                gts = test_gts[idx] if annotation_ok else []
                dest = out / "annotated_predictions" / f"{idx:06d}_{p.name}"
                shown_preds = [pred for pred in preds if pred["confidence"] >= args.conf]
                draw_boxes(p, dest, shown_preds, gts, model_names)
                if annotation_ok:
                    matches, matched_p, matched_g = match_any_class(preds, gts, args.match_iou)
                    correct_pairs = match_class(preds, gts, args.match_iou)
                    correct_p = {m[0] for m in correct_pairs}; correct_g = {m[1] for m in correct_pairs}
                    for j, pred in enumerate(preds):
                        pred["outcome"] = "correct" if j in correct_p else ("misclassified" if j in matched_p else "false_positive")
                        pred["matched_ground_truth_class"] = next((class_names[gts[gi]["class_id"]] for pi, gi, _ in matches if pi == j), "")
                    missed = [g for gi, g in enumerate(gts) if gi not in correct_g]
                    if any(x["outcome"] != "correct" for x in preds):
                        draw_boxes(p, out / "false_positives" / f"{idx:06d}_{p.name}", shown_preds, gts, model_names)
                    if missed:
                        draw_boxes(p, out / "missed_detections" / f"{idx:06d}_{p.name}", preds, gts, model_names)
                image_rows.append({"image": str(p), "status": "evaluated" if annotation_ok else "inference_only_no_valid_box_ground_truth",
                                   "ground_truth_class": p.parent.name if is_classification_layout else "",
                                   "ground_truth_boxes": json.dumps(gts), "predictions": json.dumps(preds),
                                   "correct_detections": sum(x.get("outcome") == "correct" for x in preds),
                                   "false_positives": sum(x.get("outcome") != "correct" for x in preds) if annotation_ok else "",
                                   "incorrect_classifications": sum(x.get("outcome") == "misclassified" for x in preds) if annotation_ok else "",
                                   "missed_detections": len(missed) if annotation_ok else ""})
            except Exception as exc:
                all_preds.append([])
                image_rows.append({"image": str(p), "status": f"inference_error: {type(exc).__name__}: {exc}",
                                   "ground_truth_class": p.parent.name if is_classification_layout else "",
                                   "ground_truth_boxes": "", "predictions": "", "correct_detections": 0,
                                   "false_positives": 0, "incorrect_classifications": 0, "missed_detections": ""})
    else:
        for p in imgs:
            label_class = p.parent.name if is_classification_layout else ""
            image_rows.append({"image": str(p), "status": "not_evaluated_missing_detector_or_dependency",
                               "ground_truth_class": label_class, "ground_truth_boxes": "[]", "predictions": "[]",
                               "correct_detections": "", "false_positives": "", "incorrect_classifications": "", "missed_detections": ""})

    if annotation_ok and model and len(all_preds) == len(imgs):
        metrics = metrics_for(all_preds, test_gts, class_names, args.conf, args.match_iou)
        metrics.update({"evaluation_status": "completed", "model_weights": str(args.weights.resolve()),
                        "test_images": len(imgs), "average_inference_ms": sum(inference_ms)/len(inference_ms) if inference_ms else None,
                        "class_names": class_names, "dataset_yaml_names": yaml_names, "label_errors": label_errors})
        cm = metrics["confusion_matrix"]
        cm_text = "\n".join(f"{name}: {row}" for name, row in zip(metrics["confusion_matrix_labels"], cm))
        save_confusion_png(out / "confusion_matrix.png", cm, metrics["confusion_matrix_labels"])
        summary_lines = [f"Images evaluated: {len(imgs)}", f"Precision: {metrics['precision']:.4f}", f"Recall: {metrics['recall']:.4f}",
                         f"F1: {metrics['f1']:.4f}", f"mAP@0.50: {metrics['mAP50']:.4f}", f"mAP@0.50:0.95: {metrics['mAP50_95']:.4f}"]
        save_placeholder_png(out / "results_summary.png", "Anti-UAV test evaluation", summary_lines)
    else:
        reason = []
        if inference_error: reason.append(inference_error)
        if not imgs: reason.append("No test images found from the configured dataset YAML or discovered test folders.")
        if not dataset.get("test") and not args.images_dir: reason.append("YOLO dataset YAML has no test split configured.")
        if is_classification_layout: reason.append("Discovered test directory is class-folder image classification data; it has no bounding-box ground truth.")
        if not label_root: reason.append("No YOLO label directory was found for the test image directory.")
        if label_errors: reason.append(f"{len(label_errors)} image/label integrity errors were found; see metrics.json.")
        if yaml_names and model_names and not class_map_matches: reason.append(f"Model class names {model_names} do not match dataset names {yaml_names}.")
        metrics = {"evaluation_status": "not_evaluated", "metrics_available": False,
                   "reason": reason, "model_weights": str(args.weights.resolve()), "model_checkpoint_exists": args.weights.is_file(),
                   "test_images_discovered": len(imgs), "test_images_inferred": len(all_preds),
                   "test_directory": str(img_dir.resolve()) if img_dir else None,
                   "labels_directory": str(label_root.resolve()) if label_root else None,
                   "dataset_yaml": str(args.data.resolve()), "dataset_yaml_exists": args.data.is_file(),
                   "dataset_yaml_names": yaml_names, "model_names": model_names,
                   "class_mapping_matches": class_map_matches, "classification_folder_layout": is_classification_layout,
                   "ground_truth_objects": None, "predicted_objects": sum(map(len, all_preds)) if all_preds else None,
                   "precision": None, "recall": None, "f1": None, "mAP50": None, "mAP50_95": None,
                   "label_errors": label_errors, "inference_error": inference_error,
                   "integrity_notes": ["No detection metrics are reported without valid box ground truth.",
                                       "Model predictions are never used as ground truth."]}
        metrics["classification_split_integrity"] = split_integrity
        save_placeholder_png(out / "confusion_matrix.png", "Detection confusion matrix unavailable", ["No valid detector + box-annotated test set was available."])
        save_placeholder_png(out / "results_summary.png", "Anti-UAV test evaluation", ["Metrics unavailable: detector weights and/or box labels are missing.",
                           f"Discovered test images: {len(imgs)}", "No classifier metrics are substituted for detection metrics."])

    pipeline_summary = None
    if args.pipeline_run_dir:
        pipeline_dir = args.pipeline_run_dir.resolve()
        pipeline_csv = pipeline_dir / "multi_target_predictions.csv"
        if not pipeline_csv.is_file():
            raise FileNotFoundError(f"Pipeline predictions CSV not found: {pipeline_csv}")
        grouped = defaultdict(list)
        with pipeline_csv.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                grouped[row["filename"]].append(row)
        row_by_name = {Path(row["image"]).name: row for row in image_rows}
        class_ids = {"bird": 0, "drone": 1, "helicopter": 2, "other": 3}
        class_histogram, confidences = Counter(), []
        pipeline_rows = 0
        overlap_pairs = 0
        for image_path in imgs:
            detections = grouped.get(image_path.name, [])
            predictions = []
            for detection in detections:
                cls_name = detection["target_class"]
                conf_value = float(detection["confidence"])
                box = [float(detection[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
                predictions.append({"class_id": class_ids.get(cls_name, -1), "class_name": cls_name,
                                    "confidence": conf_value, "xyxy": box,
                                    "outcome": "unverified_no_ground_truth"})
                class_histogram[cls_name] += 1
                confidences.append(conf_value)
            for pi, pred in enumerate(predictions):
                for other in predictions[pi + 1:]:
                    if iou(pred["xyxy"], other["xyxy"]) >= 0.50:
                        overlap_pairs += 1
            pipeline_rows += len(predictions)
            current = row_by_name.get(image_path.name)
            if current is not None:
                current.update({"status": "project_pipeline_inference_no_box_ground_truth",
                                "predictions": json.dumps(predictions), "predicted_count": len(predictions),
                                "inference_method": "COCO YOLO proposals + UAV-YOLONet four-class crop classifier"})
            annotated_src = pipeline_dir / image_path.name
            if annotated_src.is_file():
                shutil.copy2(annotated_src, out / "annotated_predictions" / image_path.name)
        pipeline_summary = {"run_directory": str(pipeline_dir),
                            "proposal_detector": str((ROOT / "yolo11s.pt").resolve()),
                            "proposal_detector_role": "generic COCO object proposal model; not trained Anti-UAV YOLO weights",
                            "classifier_checkpoint": str((ROOT / "models/best_uav_yolonet_4class_orig.pth").resolve()),
                            "classifier_classes": ["bird", "drone", "helicopter", "other"],
                            "images_inferred": sum(1 for p in imgs if p.name in row_by_name),
                            "detections": pipeline_rows, "detections_by_class": dict(class_histogram),
                            "confidence_range": [min(confidences), max(confidences)] if confidences else None,
                            "prediction_box_pairs_iou_ge_0_50": overlap_pairs,
                            "ground_truth_boxes_available": False,
                            "duplicate_or_correct_detection_status": "unverifiable without box annotations"}
        metrics["project_pipeline_inference"] = pipeline_summary
        # Add explicit columns without conflating project-pipeline output with YOLO test metrics.
        for row in image_rows:
            row.setdefault("predicted_count", "")
            row.setdefault("inference_method", "")
        if metrics.get("evaluation_status") != "completed":
            save_placeholder_png(out / "results_summary.png", "Anti-UAV test inference (metrics unavailable)",
                                 [f"Images passed through existing pipeline: {pipeline_summary['images_inferred']}",
                                  f"Candidate detections: {pipeline_rows}",
                                  f"Classes: {pipeline_summary['detections_by_class']}",
                                  "No bounding-box ground truth; precision / recall / mAP unavailable."])

    metrics["dataset_audit"] = {"image_count": len(imgs), "label_file_count": len(list(label_root.rglob("*.txt"))) if label_root and label_root.exists() else 0,
                                "label_error_count": len(label_errors),
                                "images_with_labels": sum(label_path_for(img_dir, label_root, p).is_file() for p in imgs) if label_root and img_dir else 0,
                                "train_or_val_overlap_check": "SHA-256 audit is recorded in evaluation_report.md; sequence-level provenance requires source metadata."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["image", "status", "ground_truth_class", "ground_truth_boxes", "predictions", "correct_detections", "false_positives", "incorrect_classifications", "missed_detections", "predicted_count", "inference_method"]
    with (out / "per_image_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(image_rows)

    report = ["# Anti-UAV test evaluation", "", f"- Status: **{metrics.get('evaluation_status')}**",
              f"- Checkpoint: `{args.weights.resolve()}` (exists: {args.weights.is_file()})",
              f"- Dataset YAML: `{args.data.resolve()}` (exists: {args.data.is_file()})",
              f"- Test images discovered: {len(imgs)}", f"- YOLO names in YAML: `{yaml_names}`",
              f"- Model class names: `{model_names}`", f"- Test labels directory: `{label_root}`",
              "", "## Evaluation result", ""]
    if metrics.get("evaluation_status") == "completed":
        report += [f"- Precision: {metrics['precision']:.4f}", f"- Recall: {metrics['recall']:.4f}", f"- F1: {metrics['f1']:.4f}",
                   f"- mAP@0.50: {metrics['mAP50']:.4f}", f"- mAP@0.50:0.95: {metrics['mAP50_95']:.4f}",
                   f"- Ground-truth objects: {metrics['ground_truth_objects']}", f"- Predicted objects: {metrics['predicted_objects']}",
                   f"- Average inference time: {metrics['average_inference_ms']:.2f} ms/image"]
    else:
        report += ["Detection precision, recall, F1, mAP, and confusion counts are **not calculable** from the currently available inputs.", "",
                   "Reasons:"] + [f"- {r}" for r in metrics.get("reason", [])]
    report += ["", "## Data integrity", "",
               ("The YOLO detector YAML defines only `drone` and has train/val entries, with no test split. The discovered `processed/five_class_grouped/test` folder is an image-classification split, not bounding-box ground truth." if is_classification_layout else f"The selected image folder `{img_dir}` contains {len(imgs)} image(s), but no corresponding box-label directory was found. It is a custom image collection, not a configured YOLO test split."),
               ("The repository's classification training scripts read their classification test split for final reporting after training. Their code selects checkpoints using validation accuracy; this inspection does not establish whether results later influenced choices. Sequence-level independence is unverified." if is_classification_layout else "The four-class classifier pipeline uses generic COCO-trained YOLO weights for proposals and a separate image classifier for crop classes; it is not a four-class Anti-UAV YOLO detector."),
               (f"Classification split SHA-256 cross-split exact duplicate pairs: {split_integrity['exact_cross_split_duplicate_pairs']}; difference-hash near-duplicate candidates (Hamming distance <=4): {split_integrity['near_duplicate_candidate_pairs']}. Candidates require visual review due repetitive backgrounds. Exact duplicate examples: `{json.dumps(split_integrity['exact_examples'], ensure_ascii=False)}`. This audit applies to the discovered classification split, not to a missing detector test split." if split_integrity else "No sibling train/val splits were available for a cross-split hash audit."),
               (f"Available classification test image counts by folder: `{split_integrity['class_counts']['test']}`." if split_integrity else ""),
               "No source data has been changed.",
               (f"The user-specified image folder was run through the existing proposal-plus-classifier pipeline. Run directory: `{pipeline_summary['run_directory']}`; images inferred: {pipeline_summary['images_inferred']}; candidate detections: {pipeline_summary['detections']}; predictions by class: `{pipeline_summary['detections_by_class']}`; predicted box pairs with IoU >= 0.50: {pipeline_summary['prediction_box_pairs_iou_ge_0_50']}. This overlap check cannot verify correctness or recall without box labels. The pipeline uses generic COCO proposals, not the missing Anti-UAV YOLO checkpoint." if pipeline_summary else ""),
               "", "## Output files", "", "See `metrics.json`, `per_image_results.csv`, `confusion_matrix.png`, `results_summary.png`, and the three image output folders. Placeholder charts explicitly indicate when metrics are unavailable.",
               "", "## Run with a labeled detector test split", "", "```powershell", "python scripts/evaluate_yolo_test.py --weights path\\to\\best.pt --data path\\to\\data.yaml --images-dir path\\to\\test\\images --labels-dir path\\to\\test\\labels", "```", ""]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": metrics.get("evaluation_status"), "images": len(imgs), "inferred": len(all_preds),
                      "metrics": {k: metrics.get(k) for k in ("precision", "recall", "f1", "mAP50", "mAP50_95")},
                      "output": str(out), "reason": metrics.get("reason")}, indent=2))


if __name__ == "__main__":
    main()
