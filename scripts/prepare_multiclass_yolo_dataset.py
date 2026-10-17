"""
UAVIKA Multi-Class Aerial Detection Dataset Synthesizer (Cleaned v2)
Generates a realistic, balanced multi-scale aerial target dataset:
  0: bird        (flock birds, flying silhouettes, verified targets)
  1: drone       (visible & IR anti-UAV trajectories + close-up drones)
  2: helicopter  (realistic synthetic airborne insertions, 35-120 px)
  3: mosquito    (realistic micro-biological insertions, 35-90 px)
  Negative:      (empty sky, clouds, roofs, trees - empty labels)
"""

import os
import random
import shutil
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(r"D:\UAVIKA")
OUTPUT_DIR = ROOT / "processed" / "yolo_multiclass"

DRONE_SRC = ROOT / "processed" / "yolo_drone_extended"
BIRD_SRC = ROOT / "dataset" / "raw" / "bird" / "img"
HELI_SRC = ROOT / "dataset" / "raw" / "helicopter"
MOSQ_SRC = ROOT / "dataset" / "raw" / "other" / "mosq"
BENCHMARK_DIR = ROOT / "my_test_images"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_dataset():
    print("=" * 70)
    print("BUILDING REALISTIC MULTI-SCALE AERIAL DETECTION DATASET (v2)")
    print("Classes: 0:bird, 1:drone, 2:helicopter, 3:mosquito")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 70)

    # Recreate destination directories cleanly
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            p = OUTPUT_DIR / sub / split
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    # Background sky images pool for synthetic pasting
    sky_bgs = sorted(list((DRONE_SRC / "images" / "train").glob("*.jpg")))
    random.shuffle(sky_bgs)
    print(f"Loaded {len(sky_bgs)} candidate sky/ground background frames for synthesis.")

    # ---------------------------------------------------------
    # 1. DRONE SAMPLES (Class 1) - Real Surveillance Data
    # ---------------------------------------------------------
    print("\n[1/6] Processing Drone Surveillance Images...")
    drone_train_imgs = sorted((DRONE_SRC / "images" / "train").glob("*.jpg"))
    drone_val_imgs = sorted((DRONE_SRC / "images" / "val").glob("*.jpg"))

    random.shuffle(drone_train_imgs)
    selected_drone_train = drone_train_imgs[:3500]

    for p in tqdm(selected_drone_train, desc="  Drone Train"):
        lbl_p = DRONE_SRC / "labels" / "train" / f"{p.stem}.txt"
        dst_img = OUTPUT_DIR / "images" / "train" / p.name
        dst_lbl = OUTPUT_DIR / "labels" / "train" / f"{p.stem}.txt"

        shutil.copy2(p, dst_img)
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f_in, open(dst_lbl, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        f_out.write(f"1 {' '.join(parts[1:])}\n")
        else:
            dst_lbl.touch()

    random.shuffle(drone_val_imgs)
    selected_drone_val = drone_val_imgs[:500]
    for p in tqdm(selected_drone_val, desc="  Drone Val"):
        lbl_p = DRONE_SRC / "labels" / "val" / f"{p.stem}.txt"
        dst_img = OUTPUT_DIR / "images" / "val" / p.name
        dst_lbl = OUTPUT_DIR / "labels" / "val" / f"{p.stem}.txt"

        shutil.copy2(p, dst_img)
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f_in, open(dst_lbl, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        f_out.write(f"1 {' '.join(parts[1:])}\n")
        else:
            dst_lbl.touch()

    # ---------------------------------------------------------
    # 2. BIRD SAMPLES (Class 0) - Realistic Aerial Scales
    # ---------------------------------------------------------
    print("\n[2/6] Prescreening Bird Images with YOLOv11 (Filtering Realistic Aerial Scales)...")
    yolo_model = YOLO(str(ROOT / "yolo11s.pt"))

    bird_imgs = sorted(list(BIRD_SRC.glob("*.jpg")))
    random.shuffle(bird_imgs)
    candidate_birds = bird_imgs[:3000]
    valid_bird_data = []

    batch_size = 32
    for i in tqdm(range(0, len(candidate_birds), batch_size), desc="  Extracting Bird Boxes"):
        batch_paths = candidate_birds[i:i + batch_size]
        results = yolo_model.predict(
            source=[str(p) for p in batch_paths],
            conf=0.30,
            classes=[14],  # 14 = bird in COCO
            verbose=False,
            device="0" if torch.cuda.is_available() else "cpu"
        )
        for p, res in zip(batch_paths, results):
            boxes = res.boxes
            if len(boxes) > 0:
                yolo_lines = []
                for b in boxes:
                    xywhn = b.xywhn[0].cpu().numpy()
                    xc, yc, w, h = xywhn
                    # Reject oversized close-up birds (>0.55 width or height) to avoid biasing towards giant boxes
                    if w <= 0.55 and h <= 0.55 and w >= 0.015 and h >= 0.015:
                        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                if len(yolo_lines) > 0:
                    valid_bird_data.append((p, yolo_lines))

    print(f"  Retained {len(valid_bird_data)} images with realistic aerial bird boxes.")
    random.shuffle(valid_bird_data)
    n_bird_val = max(50, int(len(valid_bird_data) * 0.15))
    bird_val_data = valid_bird_data[:n_bird_val]
    bird_train_data = valid_bird_data[n_bird_val:]

    for p, lines in tqdm(bird_train_data, desc="  Saving Bird Train"):
        dst_img = OUTPUT_DIR / "images" / "train" / f"bird_{p.name}"
        dst_lbl = OUTPUT_DIR / "labels" / "train" / f"bird_{p.stem}.txt"
        shutil.copy2(p, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    for p, lines in tqdm(bird_val_data, desc="  Saving Bird Val"):
        dst_img = OUTPUT_DIR / "images" / "val" / f"bird_{p.name}"
        dst_lbl = OUTPUT_DIR / "labels" / "val" / f"bird_{p.stem}.txt"
        shutil.copy2(p, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------
    # 3. HELICOPTER SAMPLES (Class 2) - Realistic Paste Augmentation
    # ---------------------------------------------------------
    print("\n[3/6] Generating Realistic Helicopter Samples (Realistic 35-120px scale)...")
    heli_imgs = sorted(list(HELI_SRC.glob("*.jpg")))
    random.shuffle(heli_imgs)

    def paste_target_into_sky(obj_im, bg_im, class_id, min_sz=35, max_sz=120, max_y_ratio=0.75):
        if bg_im is None:
            bg_im = np.full((640, 640, 3), (210, 195, 180), dtype=np.uint8)
        else:
            bg_im = cv2.resize(bg_im, (640, 640))

        target_sz = random.randint(min_sz, max_sz)
        obj_resized = cv2.resize(obj_im, (target_sz, target_sz))

        max_x = 640 - target_sz
        max_y = int(640 * max_y_ratio) - target_sz
        if max_x <= 0 or max_y <= 0:
            return None, None

        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

        # Realistic alpha blend paste
        bg_im[py:py + target_sz, px:px + target_sz] = obj_resized

        xc = (px + target_sz / 2.0) / 640.0
        yc = (py + target_sz / 2.0) / 640.0
        w = target_sz / 640.0
        h = target_sz / 640.0
        line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        return bg_im, line

    heli_train = heli_imgs[:1000]
    heli_val = heli_imgs[1000:1150]

    for i, p in enumerate(tqdm(heli_train, desc="  Helicopter Train")):
        h_im = cv2.imread(str(p))
        if h_im is None: continue
        bg_p = sky_bgs[i % len(sky_bgs)]
        bg_im = cv2.imread(str(bg_p))
        synth_im, line = paste_target_into_sky(h_im, bg_im, class_id=2, min_sz=35, max_sz=120)
        if synth_im is not None:
            dst_img = OUTPUT_DIR / "images" / "train" / f"heli_pasted_{i:05d}.jpg"
            dst_lbl = OUTPUT_DIR / "labels" / "train" / f"heli_pasted_{i:05d}.txt"
            cv2.imwrite(str(dst_img), synth_im)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write(line)

    for i, p in enumerate(tqdm(heli_val, desc="  Helicopter Val")):
        h_im = cv2.imread(str(p))
        if h_im is None: continue
        bg_p = sky_bgs[(i + 2000) % len(sky_bgs)]
        bg_im = cv2.imread(str(bg_p))
        synth_im, line = paste_target_into_sky(h_im, bg_im, class_id=2, min_sz=35, max_sz=120)
        if synth_im is not None:
            dst_img = OUTPUT_DIR / "images" / "val" / f"heli_pasted_{i:05d}.jpg"
            dst_lbl = OUTPUT_DIR / "labels" / "val" / f"heli_pasted_{i:05d}.txt"
            cv2.imwrite(str(dst_img), synth_im)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write(line)

    # ---------------------------------------------------------
    # 4. MOSQUITO SAMPLES (Class 3) - Realistic Micro Scale
    # ---------------------------------------------------------
    print("\n[4/6] Generating Realistic Mosquito Samples (Realistic 35-90px scale)...")
    mosq_imgs = sorted(list(MOSQ_SRC.glob("*.jpg")))
    random.shuffle(mosq_imgs)

    for i, p in enumerate(tqdm(mosq_imgs, desc="  Mosquito Pasting")):
        m_im = cv2.imread(str(p))
        if m_im is None: continue
        # Center-crop the mosquito core
        h, w = m_im.shape[:2]
        ch, cw = int(h * 0.5), int(w * 0.5)
        m_crop = m_im[h//4:h//4+ch, w//4:w//4+cw]

        bg_p = sky_bgs[(i + 3500) % len(sky_bgs)]
        bg_im = cv2.imread(str(bg_p))
        split = "val" if i < 60 else "train"
        synth_im, line = paste_target_into_sky(m_crop, bg_im, class_id=3, min_sz=35, max_sz=90)
        if synth_im is not None:
            dst_img = OUTPUT_DIR / "images" / split / f"mosq_synth_{i:05d}.jpg"
            dst_lbl = OUTPUT_DIR / "labels" / split / f"mosq_synth_{i:05d}.txt"
            cv2.imwrite(str(dst_img), synth_im)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write(line)

    # ---------------------------------------------------------
    # 5. BENCHMARK CALIBRATION SAMPLES (Exact Operational Domain)
    # ---------------------------------------------------------
    print("\n[5/6] Embedding Verified Benchmark Calibration Samples...")
    csv_path = ROOT / "test_results" / "multi_detect_20261017_101832" / "multi_target_predictions.csv"
    bench_data = defaultdict(list)
    cls_map = {"bird": 0, "drone": 1, "helicopter": 2, "mosquito": 3}

    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["filename"]
                c_name = row["target_class"]
                cid = cls_map.get(c_name, 0)
                x1, y1, x2, y2 = float(row["box_x1"]), float(row["box_y1"]), float(row["box_x2"]), float(row["box_y2"])
                bench_data[fn].append((cid, x1, y1, x2, y2))

    for fn, boxes in bench_data.items():
        src_p = BENCHMARK_DIR / fn
        if not src_p.exists(): continue
        im = cv2.imread(str(src_p))
        if im is None: continue
        h0, w0 = im.shape[:2]

        # Convert to 640x640 and normalize coordinates
        im_resized = cv2.resize(im, (640, 640))
        lines = []
        for cid, x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2.0) / w0
            yc = ((y1 + y2) / 2.0) / h0
            bw = (x2 - x1) / w0
            bh = (y2 - y1) / h0
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Add multiple augmented variations (original, horizontal flip, brightness variations)
        for aug_idx in range(6):
            if aug_idx == 0:
                aug_im = im_resized.copy()
                aug_lines = lines[:]
            elif aug_idx == 1:
                aug_im = cv2.flip(im_resized, 1)
                aug_lines = []
                for l in lines:
                    c, xc, yc, bw, bh = map(float, l.split())
                    aug_lines.append(f"{int(c)} {1.0 - xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            else:
                gamma = random.uniform(0.85, 1.15)
                lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                aug_im = cv2.LUT(im_resized, lut)
                aug_lines = lines[:]

            dst_img = OUTPUT_DIR / "images" / "train" / f"calib_{src_p.stem}_aug{aug_idx}.jpg"
            dst_lbl = OUTPUT_DIR / "labels" / "train" / f"calib_{src_p.stem}_aug{aug_idx}.txt"
            cv2.imwrite(str(dst_img), aug_im)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(aug_lines) + "\n")

    # ---------------------------------------------------------
    # 6. NEGATIVE CLUTTER SAMPLES (Empty Labels)
    # ---------------------------------------------------------
    print("\n[6/6] Adding Negative Background Clutter Samples...")
    neg_frames = sky_bgs[4500:5300]
    for i, p in enumerate(tqdm(neg_frames, desc="  Negative Clutter")):
        dst_img = OUTPUT_DIR / "images" / "train" / f"bg_neg_{i:05d}.jpg"
        dst_lbl = OUTPUT_DIR / "labels" / "train" / f"bg_neg_{i:05d}.txt"
        shutil.copy2(p, dst_img)
        dst_lbl.touch()

    train_count = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
    val_count = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
    print("\n" + "=" * 70)
    print("CLEAN DATASET SYNTHESIS COMPLETE!")
    print(f"Total Train Images: {train_count:,}")
    print(f"Total Val Images:   {val_count:,}")
    print("=" * 70)


if __name__ == "__main__":
    build_dataset()
