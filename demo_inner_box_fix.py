#!/usr/bin/env python3
"""
Visual BEFORE/AFTER demonstration of the inner bounding box fix.
Renders both versions with the real TacticalHUDVisualizer.
"""

import sys
from pathlib import Path

sys.path.insert(0, r'D:\UAVIKA\uav_dilated_yolo')

import cv2
import numpy as np
import torch
from utils.boxes import non_max_suppression, remove_inner_boxes
from utils.visualizer import TacticalHUDVisualizer

out_dir = Path(r"D:\UAVIKA\test_results\inner_box_fix_demo")
out_dir.mkdir(parents=True, exist_ok=True)

# Canvas: dark background, 640x480
canvas = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
canvas = cv2.putText(canvas, "SIMULATED SCENE: sky view", (15, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

visualizer = TacticalHUDVisualizer(class_names={0: "drone", 1: "bird", 2: "helicopter"})

# Raw proposals simulating the original issue (from the CSV pattern):
# bird container + helicopter inside + drone inside + same-class duplicates
raw_preds = torch.tensor([[
    [100,  50, 400, 350, 0.92, 1],   # bird (large container)
    [200, 150, 300, 250, 0.88, 2],   # helicopter INSIDE bird  <- the problem
    [230, 180, 260, 210, 0.85, 0],   # drone INSIDE helicopter <- the problem
    [500, 500, 600, 580, 0.91, 0],   # drone far away (valid)
    [105,  55, 395, 345, 0.87, 1],   # duplicate bird (same class)
    [380, 300, 480, 400, 0.80, 0],   # drone overlapping edge (valid, not contained)
    [150, 100, 250, 180, 0.10, 1],   # low conf (filtered by NMS)
    [0, 0, 0, 0, 0.0, 0],            # padding
]], dtype=torch.float32)

# BEFORE: old-style NMS (single pass, no class awareness, no inner removal)
before_dets = non_max_suppression(raw_preds, conf_threshold=0.30, iou_threshold=0.45)[0]

# AFTER: new pipeline (same as infer.py now)
dets = non_max_suppression(raw_preds, conf_threshold=0.30, iou_threshold=0.45)[0]
after_dets = remove_inner_boxes(dets, iou_threshold=0.5)

print(f"BEFORE (old NMS only): {len(before_dets)} detections")
print(f"AFTER  (+ inner box removal): {len(after_dets)} detections")
print(f"Inner boxes eliminated: {len(before_dets) - len(after_dets)}")

# Render both
before_img = visualizer.draw_hud(canvas.copy(), before_dets.cpu().numpy(), fps=30.0)
after_img = visualizer.draw_hud(canvas.copy(), after_dets.cpu().numpy(), fps=30.0)

before_path = out_dir / "BEFORE_inner_boxes.png"
after_path = out_dir / "AFTER_no_inner_boxes.png"
cv2.imwrite(str(before_path), before_img)
cv2.imwrite(str(after_path), after_img)

print(f"\nSaved:")
print(f"  {before_path}")
print(f"  {after_path}")