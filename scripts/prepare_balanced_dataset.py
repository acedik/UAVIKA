import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


SOURCE_DIR = Path(r"D:\UAVIkA\processed\classification")
OUTPUT_DIR = Path(r"D:\UAVIkA\processed\classification_balanced")

CLASSES = ["bird", "drone", "helicopter", "other"]

# Balanced target counts per split
TARGET_COUNTS = {
    "train": 2500,  # 2,500 * 4 = 10,000 balanced training images
    "val": 300,     # 300 * 4 = 1,200 balanced validation images
    "test": 300     # 300 * 4 = 1,200 balanced test images
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)
np.random.seed(42)


def augment_image(image):
    """Generates a realistic spatial/photometric variation of an image crop."""
    h, w = image.shape[:2]
    aug = image.copy()

    # 1. Random Horizontal Flip (50% chance)
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)

    # 2. Random Slight Rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Random Brightness / Contrast Jitter
    alpha = random.uniform(0.85, 1.15)  # Contrast
    beta = random.uniform(-20, 20)      # Brightness
    aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)

    # 4. Occasional slight Gaussian blur (simulate motion/optical blur)
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    return aug


def balance_split(split_name, target_count):
    print(f"\n================ Balancing Split: {split_name.upper()} (Target: {target_count} per class) ================")

    for cls in CLASSES:
        src_cls_dir = SOURCE_DIR / split_name / cls
        dst_cls_dir = OUTPUT_DIR / split_name / cls
        dst_cls_dir.mkdir(parents=True, exist_ok=True)

        # Get all source images for this class
        images = [
            p for p in src_cls_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        total_src = len(images)
        print(f"[{cls.upper()}] Found {total_src} original images.")

        if total_src == 0:
            print(f"Warning: No images found for {cls} in {src_cls_dir}")
            continue

        random.shuffle(images)

        # Case 1: Majority Class -> Subsample down to target_count
        if total_src >= target_count:
            selected = images[:target_count]
            print(f"  -> Subsampling from {total_src} down to {target_count}...")
            for idx, img_path in enumerate(tqdm(selected, desc=f"Copying {cls}", leave=False)):
                dst_name = f"{cls}_{idx:06d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dst_cls_dir / dst_name)

        # Case 2: Minority Class -> Copy all originals + Augment until target_count
        else:
            print(f"  -> Copying {total_src} originals and synthesizing {target_count - total_src} augmented crops...")
            # 1. Copy all available original images
            loaded_imgs = []
            for idx, img_path in enumerate(images):
                dst_name = f"{cls}_{idx:06d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dst_cls_dir / dst_name)
                img = cv2.imread(str(img_path))
                if img is not None:
                    loaded_imgs.append(img)

            # 2. Synthesize augmented variations until reaching target_count
            needed = target_count - total_src
            pbar = tqdm(total=needed, desc=f"Augmenting {cls}", leave=False)
            aug_idx = total_src

            while aug_idx < target_count:
                base_img = random.choice(loaded_imgs)
                aug_img = augment_image(base_img)
                dst_name = f"{cls}_aug_{aug_idx:06d}.jpg"
                cv2.imwrite(str(dst_cls_dir / dst_name), aug_img)
                aug_idx += 1
                pbar.update(1)
            pbar.close()

        final_count = len(list(dst_cls_dir.glob("*.*")))
        print(f"  -> Resulting balanced count for [{cls}]: {final_count} images.")


def main():
    print("=" * 60)
    print("DATASET BALANCING PIPELINE: ANTI-UAV CLASSIFICATION")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Classes: {CLASSES}")

    for split, target in TARGET_COUNTS.items():
        balance_split(split, target)

    print("\n" + "=" * 60)
    print("DATASET BALANCING COMPLETE!")
    print(f"Balanced dataset saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
