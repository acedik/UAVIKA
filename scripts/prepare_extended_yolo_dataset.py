import os
import random
from pathlib import Path
from collections import defaultdict
import yaml

ROOT = Path(r"D:\UAVIKA")
SRC_DIR = ROOT / "processed" / "yolo_drone"
DST_DIR = ROOT / "processed" / "yolo_drone_extended"

TARGET_TRAIN = 6000
TARGET_VAL = 1000

random.seed(42)

def prepare():
    print("=" * 65)
    print("PREPARING DIVERSE EXTENDED ANTI-UAV YOLODRONE DATASET")
    print(f"Target Train: {TARGET_TRAIN} | Target Val: {TARGET_VAL}")
    print("=" * 65)

    for split in ["train", "val"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Group training images by video sequence
    train_img_dir = SRC_DIR / "images" / "train"
    train_lbl_dir = SRC_DIR / "labels" / "train"

    seq_map = defaultdict(list)
    for img_p in train_img_dir.glob("*.jpg"):
        lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            seq_id = "_".join(img_p.stem.split("_")[:-1])
            seq_map[seq_id].append((img_p, lbl_p))

    print(f"Found {len(seq_map)} unique video sequences in training set.")

    # Uniform sampling across sequences
    selected_train = []
    num_seqs = len(seq_map)
    base_per_seq = TARGET_TRAIN // num_seqs
    remainder = TARGET_TRAIN % num_seqs

    for i, (seq_id, items) in enumerate(sorted(seq_map.items())):
        k = base_per_seq + (1 if i < remainder else 0)
        items_sorted = sorted(items, key=lambda x: x[0].name)
        if len(items_sorted) <= k:
            selected = items_sorted
        else:
            # Pick evenly spaced frames across the trajectory
            step = len(items_sorted) / k
            indices = [int(j * step) for j in range(k)]
            selected = [items_sorted[idx] for idx in indices]
        selected_train.extend(selected)

    print(f"Sampled {len(selected_train)} diverse training images across all {num_seqs} sequences.")

    # 2. Group validation images by video sequence
    val_img_dir = SRC_DIR / "images" / "val"
    val_lbl_dir = SRC_DIR / "labels" / "val"

    val_seq_map = defaultdict(list)
    for img_p in val_img_dir.glob("*.jpg"):
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            seq_id = "_".join(img_p.stem.split("_")[:-1])
            val_seq_map[seq_id].append((img_p, lbl_p))

    selected_val = []
    v_num_seqs = len(val_seq_map)
    v_base = TARGET_VAL // v_num_seqs
    v_rem = TARGET_VAL % v_num_seqs

    for i, (seq_id, items) in enumerate(sorted(val_seq_map.items())):
        k = v_base + (1 if i < v_rem else 0)
        items_sorted = sorted(items, key=lambda x: x[0].name)
        if len(items_sorted) <= k:
            selected = items_sorted
        else:
            step = len(items_sorted) / k
            indices = [int(j * step) for j in range(k)]
            selected = [items_sorted[idx] for idx in indices]
        selected_val.extend(selected)

    print(f"Sampled {len(selected_val)} diverse validation images across all {v_num_seqs} validation sequences.")

    # 3. Create hardlinks
    def link_files(pairs, split):
        dst_img_dir = DST_DIR / "images" / split
        dst_lbl_dir = DST_DIR / "labels" / split
        for img_src, lbl_src in pairs:
            img_dst = dst_img_dir / img_src.name
            lbl_dst = dst_lbl_dir / lbl_src.name
            if not img_dst.exists():
                os.link(img_src, img_dst)
            if not lbl_dst.exists():
                os.link(lbl_src, lbl_dst)

    print("\nCreating fast NTFS hardlinks...")
    link_files(selected_train, "train")
    link_files(selected_val, "val")

    # 4. Create data.yaml
    yaml_content = {
        "path": "D:/UAVIKA/processed/yolo_drone_extended",
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "drone"
        }
    }
    with open(DST_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    print(f"Dataset prepared successfully at: {DST_DIR}")
    print(f"Train images: {len(list((DST_DIR / 'images' / 'train').glob('*.jpg')))}")
    print(f"Val images:   {len(list((DST_DIR / 'images' / 'val').glob('*.jpg')))}")

if __name__ == "__main__":
    prepare()
