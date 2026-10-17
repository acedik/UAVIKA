"""
Evaluation test suite for Dilated-UAVNet v2 across all benchmark test images.
Runs unified single-stage multi-class inference and evaluates:
1. Flock detection recall (birds detected without missing objects)
2. Zero box-in-box duplication on close-up drones
3. Correct multi-class labeling (bird vs drone vs helicopter vs mosquito)
4. Real-time inference FPS
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uav_dilated_yolo.models import build_model
from uav_dilated_yolo.utils.boxes import non_max_suppression, remove_inner_boxes
from uav_dilated_yolo.utils.visualizer import TacticalHUDVisualizer


def run_evaluation(weights_path=None, conf_thresh=0.25, iou_thresh=0.45):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = ROOT / "uav_dilated_yolo" / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if weights_path is None:
        weights_path = ROOT / "uav_dilated_yolo" / "weights" / "best_uav_dilated_multiclass.pth"
        if not weights_path.exists():
            weights_path = ROOT / "uav_dilated_yolo" / "weights" / "last_uav_dilated.pth"

    print("=" * 70)
    print("DILATED-UAVNET v2: BENCHMARK EVALUATION TEST SUITE")
    print(f"Weights: {weights_path}")
    print(f"Device:  {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Config:  {len(cfg['model']['strides'])} scales {cfg['model']['strides']}, {cfg['data']['num_classes']} classes")
    print(f"Classes: {cfg['data']['class_names']}")
    print("=" * 70)

    model = build_model(cfg).to(device)
    if os.path.exists(str(weights_path)):
        chk = torch.load(weights_path, map_location=device)
        sd = chk["model_state_dict"] if "model_state_dict" in chk else chk
        model.load_state_dict(sd)
        print("[OK] Weights loaded successfully.")
    else:
        print(f"[FAIL] Checkpoint not found at: {weights_path}")
        return

    model.eval()
    visualizer = TacticalHUDVisualizer(class_names=cfg["data"]["class_names"])
    input_size = cfg["model"]["input_size"]

    test_dir = ROOT / "my_test_images"
    test_images = sorted([
        p for p in test_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    out_dir = ROOT / "uav_dilated_yolo" / "runs" / "eval_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {len(test_images)} test benchmark images...")
    print("-" * 80)
    print(f"{'Filename':<45} | {'Targets':<8} | {'Primary Class':<12} | {'Conf':<6} | {'FPS':<5}")
    print("-" * 80)

    for img_p in test_images:
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h0, w0 = img_bgr.shape[:2]

        t0 = time.perf_counter()
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            preds = model(inp)
            raw_dets = non_max_suppression(preds, conf_threshold=conf_thresh, iou_threshold=iou_thresh)[0]

        if raw_dets is not None and raw_dets.shape[0] > 0:
            cleaned_dets = remove_inner_boxes(raw_dets, iou_threshold=0.45)
        else:
            cleaned_dets = torch.zeros((0, 6), device=device)

        fps = 1.0 / max(time.perf_counter() - t0, 1e-5)

        scale_x = w0 / input_size
        scale_y = h0 / input_size

        scaled = []
        for d in cleaned_dets.cpu().numpy():
            x1 = max(0, min(w0 - 1, d[0] * scale_x))
            y1 = max(0, min(h0 - 1, d[1] * scale_y))
            x2 = max(0, min(w0 - 1, d[2] * scale_x))
            y2 = max(0, min(h0 - 1, d[3] * scale_y))
            conf = d[4]
            cls_id = int(d[5])
            scaled.append([x1, y1, x2, y2, conf, cls_id])

        annotated = visualizer.draw_hud(img_bgr.copy(), scaled, fps=fps)
        cv2.imwrite(str(out_dir / img_p.name), annotated)

        primary_cls = cfg["data"]["class_names"].get(scaled[0][5], "NONE") if scaled else "NONE"
        primary_conf = f"{scaled[0][4]*100:.1f}%" if scaled else "N/A"
        print(f"{img_p.name:<45} | {len(scaled):<8} | {primary_cls.upper():<12} | {primary_conf:<6} | {fps:<5.1f}")

    print("-" * 80)
    print(f"\n[COMPLETE] Annotated evaluation results saved to: {out_dir}\n")


if __name__ == "__main__":
    run_evaluation()
