#!/usr/bin/env python3
"""
End-to-end test: Verify inner bounding boxes are eliminated from the full pipeline.

Tests:
1. Full inference pipeline (model -> NMS -> inner box removal)
2. Nested object scenarios (helicopter inside bird, bird inside bird)
3. Real test images from test_results folder
4. No regression on clean detections
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, r'D:\UAVIKA\uav_dilated_yolo')

import torch
import cv2
import numpy as np
import yaml

from models import build_model
from utils.boxes import non_max_suppression, remove_inner_boxes

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {name}" + (f" - {detail}" if detail else ""))
    results.append((name, condition))
    return condition

print("=" * 70)
print("END-TO-END TEST: Inner Bounding Box Elimination")
print("=" * 70)

# ------------------------------------------------------------------
# Setup: Load config and build the real model
# ------------------------------------------------------------------
config_path = r"D:\UAVIKA\uav_dilated_yolo\configs\config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = build_model(cfg)
model.eval()
print(f"\n{INFO} Model built: {cfg['model']['name']} (classes={cfg['data']['num_classes']})")

# ------------------------------------------------------------------
# Test 1: Full pipeline with synthetic nested objects
# ------------------------------------------------------------------
print("\n--- Test 1: Full pipeline - nested objects (different classes) ---")

# Build a synthetic prediction batch simulating raw model output.
# The head decodes to [B, N, 6] = (x1, y1, x2, y2, score, class_id)
# Simulate raw proposals: nested boxes at different classes
B, N = 1, 8
raw_preds = torch.zeros(B, N, 6)
# Large container box: bird (class 1) at 100,50 -> 400,350
raw_preds[0, 0] = torch.tensor([100, 50, 400, 350, 0.92, 1])
# Small box inside: helicopter (class 2) at 200,150 -> 300,250
raw_preds[0, 1] = torch.tensor([200, 150, 300, 250, 0.88, 2])
# Tiny box inside: drone (class 0) at 230,180 -> 260,210
raw_preds[0, 2] = torch.tensor([230, 180, 260, 210, 0.85, 0])
# Unrelated box far away: drone (class 0)
raw_preds[0, 3] = torch.tensor([500, 500, 600, 580, 0.91, 0])
# Low confidence below threshold (should be filtered by NMS)
raw_preds[0, 4] = torch.tensor([150, 100, 250, 180, 0.10, 1])
# Overlapping duplicate of container bird (same class, high IoU)
raw_preds[0, 5] = torch.tensor([105, 55, 395, 345, 0.87, 1])
# Non-contained partial overlap: drone (class 0) overlapping edge of bird
raw_preds[0, 6] = torch.tensor([380, 300, 480, 400, 0.80, 0])
# Zero padding row (empty)
raw_preds[0, 7] = torch.tensor([0, 0, 0, 0, 0.0, 0])

# Run the full post-processing chain used in infer.py
dets = non_max_suppression(raw_preds, conf_threshold=0.30, iou_threshold=0.45)[0]
print(f"  {INFO} After class-aware NMS: {len(dets)} detections")
dets_clean = remove_inner_boxes(dets, iou_threshold=0.5)
print(f"  {INFO} After inner box removal: {len(dets_clean)} detections")

# Verify no inner boxes remain in final output
def has_inner(boxes):
    n = len(boxes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = boxes[i][:4]
            b = boxes[j][:4]
            if (b[0] >= a[0] and b[1] >= a[1] and b[2] <= a[2] and b[3] <= a[3]):
                return True, (i, j)
    return False, None

inner_found, pair = has_inner(dets_clean)
check("No inner boxes in final output", not inner_found,
      f"contained pair: {pair}" if inner_found else "clean")

# Verify the container bird survived (highest area + high conf)
classes_kept = set(dets_clean[:, 5].tolist())
check("Container bird (class 1) survives", 1 in classes_kept)
check("Far-away drone survives", 0 in classes_kept, f"classes kept: {sorted(classes_kept)}")

# ------------------------------------------------------------------
# Test 2: Edge cases for remove_inner_boxes
# ------------------------------------------------------------------
print("\n--- Test 2: Edge cases ---")

# Empty input
empty = torch.zeros((0, 6))
empty_out = remove_inner_boxes(empty)
check("Empty input handled", len(empty_out) == 0)

# Single box
single = torch.tensor([[10, 10, 100, 100, 0.9, 0]])
single_out = remove_inner_boxes(single)
check("Single box passes through", len(single_out) == 1)

# Identical boxes (exact duplicates, different classes)
identical = torch.tensor([
    [10, 10, 100, 100, 0.9, 0],
    [10, 10, 100, 100, 0.8, 1],
])
identical_out = remove_inner_boxes(identical)
check("Identical nested boxes deduplicated", len(identical_out) == 1,
      f"kept: {len(identical_out)}")

# Boxes sharing an edge (touching but not contained)
touching = torch.tensor([
    [0, 0, 100, 100, 0.9, 0],
    [100, 0, 200, 100, 0.9, 0],
])
touching_out = remove_inner_boxes(touching)
check("Touching boxes NOT suppressed (not contained)", len(touching_out) == 2)

# Deeply nested chain: A contains B contains C
chain = torch.tensor([
    [0, 0, 300, 300, 0.9, 0],
    [50, 50, 250, 250, 0.85, 0],
    [100, 100, 200, 200, 0.80, 0],
])
chain_out = remove_inner_boxes(chain)
check("Nested chain collapses to outermost only", len(chain_out) == 1,
      f"kept: {len(chain_out)}")

# ------------------------------------------------------------------
# Test 3: Confidence-ordered output
# ------------------------------------------------------------------
print("\n--- Test 3: Output ordering ---")
mixed = torch.tensor([
    [500, 500, 600, 580, 0.70, 0],   # lower conf, far away
    [0, 0, 300, 300, 0.95, 0],       # higher conf
    [350, 50, 450, 150, 0.85, 0],    # middle conf
])
mixed_out = remove_inner_boxes(mixed)
confs = mixed_out[:, 4].tolist()
is_sorted_desc = all(confs[i] >= confs[i+1] for i in range(len(confs)-1))
check("Output sorted by confidence descending", is_sorted_desc, f"confs: {[round(c,2) for c in confs]}")
check("All valid boxes preserved", len(mixed_out) == 3)

# ------------------------------------------------------------------
# Test 4: Real test images through the full pipeline
# ------------------------------------------------------------------
print("\n--- Test 4: Real test images (full pipeline, no weights needed for NMS path) ---")
test_dir = Path(r"D:\UAVIKA\test_results\multi_detect_20260925_114047")
img_paths = sorted([p for p in test_dir.glob("*.png")])

print(f"  {INFO} Found {len(img_paths)} test images")

# Use random-weight model just to exercise the pipeline shape (weights are not in expected path)
all_images_clean = True
for p in img_paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"  {FAIL} Could not read: {p.name}")
        continue

    # Preprocess like infer.py
    h, w = img.shape[:2]
    resized = cv2.resize(img, (cfg["model"]["input_size"], cfg["model"]["input_size"]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        preds = model(tensor)

    dets = non_max_suppression(preds, conf_threshold=0.30, iou_threshold=0.45)[0]
    if dets.shape[0] > 0:
        dets = remove_inner_boxes(dets, iou_threshold=0.5)

    inner, pair = has_inner(dets)
    status = "clean" if not inner else f"INNER PAIR {pair}"
    if inner:
        all_images_clean = False
    print(f"  {INFO} {p.name}: {len(dets)} final detections ({status})")

check("All real images have no inner boxes", all_images_clean)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
passed = sum(1 for _, ok in results if ok)
total = len(results)
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} checks passed")
print("=" * 70)
if passed == total:
    print("ALL TESTS PASSED - Inner bounding boxes eliminated")
else:
    failed_names = [name for name, ok in results if not ok]
    print("FAILED checks:")
    for name in failed_names:
        print(f"  - {name}")

sys.exit(0 if passed == total else 1)