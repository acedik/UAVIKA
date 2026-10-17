import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(r"D:\UAVIKA")
RAW_DIR = ROOT_DIR / "dataset" / "raw"
OUTPUT_DIR = ROOT_DIR / "processed" / "five_class_classification"

CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]
MAX_IMAGE_DIM = 640

def open_and_normalize(path_or_array):
    if isinstance(path_or_array, (str, Path)):
        im = Image.open(path_or_array)
    else:
        im = Image.fromarray(cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB))
    
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(w, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return im

def prepare_dataset():
    print("=" * 65)
    print("PREPARING 5-CLASS DATASET (SEPARATING MOSQUITO FROM OTHER)")
    print("=" * 65)
    print(f"Target Output Directory: {OUTPUT_DIR}")
    print(f"Classes: {CLASSES}")

    # Create destination folders
    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            dest = OUTPUT_DIR / split / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*.*"):
                f.unlink()

    # 1. MOSQUITO (from RAW other/mosq)
    print("\n--- 1. Processing MOSQUITO ---")
    mosq_dir = RAW_DIR / "other" / "mosq"
    mosq_images = sorted([p for p in mosq_dir.glob("*.jpg")])
    random.shuffle(mosq_images)
    print(f"Found {len(mosq_images)} mosquito images.")

    n_mosq = len(mosq_images)
    n_train = int(n_mosq * 0.70)
    n_val = int(n_mosq * 0.15)
    mosq_splits = {
        "train": mosq_images[:n_train],
        "val": mosq_images[n_train:n_train + n_val],
        "test": mosq_images[n_train + n_val:]
    }
    for split, paths in mosq_splits.items():
        print(f"  Saving {split} mosquito: {len(paths)} images")
        for idx, p in enumerate(paths):
            im = open_and_normalize(p)
            im.save(OUTPUT_DIR / split / "mosquito" / f"mosquito_{split}_{idx:05d}.jpg", quality=92)

    # 2. HELICOPTER (from RAW helicopter)
    print("\n--- 2. Processing HELICOPTER ---")
    heli_dir = RAW_DIR / "helicopter"
    heli_images = sorted([p for p in heli_dir.glob("*.jpg")])
    random.shuffle(heli_images)
    print(f"Found {len(heli_images)} helicopter images.")

    n_heli = len(heli_images)
    n_train = int(n_heli * 0.70)
    n_val = int(n_heli * 0.15)
    heli_splits = {
        "train": heli_images[:n_train],
        "val": heli_images[n_train:n_train + n_val],
        "test": heli_images[n_train + n_val:]
    }
    for split, paths in heli_splits.items():
        print(f"  Saving {split} helicopter: {len(paths)} images")
        for idx, p in enumerate(paths):
            im = open_and_normalize(p)
            im.save(OUTPUT_DIR / split / "helicopter" / f"helicopter_{split}_{idx:05d}.jpg", quality=92)

    # 3. BIRD (from RAW bird)
    print("\n--- 3. Processing BIRD ---")
    bird_dir = RAW_DIR / "bird"
    bird_images = sorted([p for p in bird_dir.rglob("*.jpg") if p.is_file()])
    random.shuffle(bird_images)
    # Take a representative sample of 1200 images (800 train, 200 val, 200 test)
    bird_sample = bird_images[:1200]
    bird_splits = {
        "train": bird_sample[:800],
        "val": bird_sample[800:1000],
        "test": bird_sample[1000:1200]
    }
    for split, paths in bird_splits.items():
        print(f"  Saving {split} bird: {len(paths)} images")
        for idx, p in enumerate(paths):
            im = open_and_normalize(p)
            im.save(OUTPUT_DIR / split / "bird" / f"bird_{split}_{idx:05d}.jpg", quality=92)

    # 4. DRONE (from drone_crops and raw drone)
    print("\n--- 4. Processing DRONE ---")
    drone_crops_dir = ROOT_DIR / "processed" / "drone_crops" / "infrared"
    drone_raw_dir = RAW_DIR / "drone" / "train" / "img"
    all_drone = []
    if drone_crops_dir.exists():
        all_drone.extend(list(drone_crops_dir.glob("*.jpg")))
    if drone_raw_dir.exists():
        all_drone.extend(list(drone_raw_dir.glob("*.jpg"))[:1000])
    random.shuffle(all_drone)
    drone_sample = all_drone[:1200]
    drone_splits = {
        "train": drone_sample[:800],
        "val": drone_sample[800:1000],
        "test": drone_sample[1000:1200]
    }
    for split, paths in drone_splits.items():
        print(f"  Saving {split} drone: {len(paths)} images")
        for idx, p in enumerate(paths):
            im = open_and_normalize(p)
            im.save(OUTPUT_DIR / split / "drone" / f"drone_{split}_{idx:05d}.jpg", quality=92)

    # 5. OTHER / BACKGROUND / CLUTTER
    print("\n--- 5. Processing OTHER (Negative Background & Environmental Clutter) ---")
    # Extract background crops from yolo_drone surveillance frames (avoiding drone bbox)
    yolo_img_dir = ROOT_DIR / "processed" / "yolo_drone" / "images" / "train"
    yolo_lbl_dir = ROOT_DIR / "processed" / "yolo_drone" / "labels" / "train"
    yolo_imgs = sorted(list(yolo_img_dir.glob("*.jpg")))
    random.shuffle(yolo_imgs)

    other_crops = []
    print("Extracting background / clutter patches for 'other' class...")
    for img_p in yolo_imgs:
        if len(other_crops) >= 800:
            break
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]
        
        # Read drone bbox
        lbl_p = yolo_lbl_dir / (img_p.stem + ".txt")
        bbox = None
        if lbl_p.exists():
            with open(lbl_p, "r") as fp:
                line = fp.readline()
                parts = line.strip().split()
                if len(parts) == 5:
                    bx = float(parts[1]) * w
                    by = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    bbox = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)

        # Candidate background patch: 160x160 to 300x300 crop
        crop_sz = min(w // 2, h // 2, 240)
        # Choose a corner away from the drone
        corners = [
            (0, 0, crop_sz, crop_sz),
            (w - crop_sz, 0, w, crop_sz),
            (0, h - crop_sz, crop_sz, h),
            (w - crop_sz, h - crop_sz, w, h)
        ]
        random.shuffle(corners)
        for (cx1, cy1, cx2, cy2) in corners:
            # Check overlap with drone bbox
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                overlap_x = max(0, min(cx2, bx2) - max(cx1, bx1))
                overlap_y = max(0, min(cy2, by2) - max(cy1, by1))
                if overlap_x * overlap_y > 0:
                    continue  # Overlaps with drone, skip
            patch = im[cy1:cy2, cx1:cx2]
            if patch.shape[0] > 50 and patch.shape[1] > 50:
                other_crops.append(patch)
                break

    print(f"Extracted {len(other_crops)} clean background/clutter patches.")
    random.shuffle(other_crops)
    n_other = len(other_crops)
    n_train = int(n_other * 0.70)
    n_val = int(n_other * 0.15)
    other_splits = {
        "train": other_crops[:n_train],
        "val": other_crops[n_train:n_train + n_val],
        "test": other_crops[n_train + n_val:]
    }
    for split, patches in other_splits.items():
        print(f"  Saving {split} other: {len(patches)} images")
        for idx, patch in enumerate(patches):
            im = open_and_normalize(patch)
            im.save(OUTPUT_DIR / split / "other" / f"other_{split}_{idx:05d}.jpg", quality=92)

    print("\n" + "=" * 65)
    print("5-CLASS DATASET PREPARATION COMPLETE!")
    print("=" * 65)

    # Print summary of final dataset
    for split in ["train", "val", "test"]:
        print(f"\nSplit: {split.upper()}")
        for cls_name in CLASSES:
            count = len(list((OUTPUT_DIR / split / cls_name).glob("*.jpg")))
            print(f"  {cls_name:<12}: {count} images")

if __name__ == "__main__":
    prepare_dataset()
