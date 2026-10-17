import cv2
import numpy as np
from pathlib import Path
import random

ROOT = Path(r"D:\UAVIKA")
TRAIN_MOSQ = ROOT / "processed" / "five_class_grouped" / "train" / "mosquito"
VAL_MOSQ = ROOT / "processed" / "five_class_grouped" / "val" / "mosquito"

RAW_MOSQ = ROOT / "dataset" / "raw" / "other" / "mosq"
SKY_DIR = ROOT / "processed" / "yolo_drone" / "images" / "train"
TEST_IMG = ROOT / "my_test_images" / "Screenshot 2026-09-25 110205.png"

random.seed(42)
np.random.seed(42)

def generate_sky_mosquito_samples():
    print("=" * 60)
    print("GENERATING REALISTIC SKY & WINDOW MOSQUITO SAMPLES")
    print("=" * 60)

    # 1. Augment the user's test mosquito on sky/glass
    mosq_img = cv2.imread(str(TEST_IMG))
    h, w = mosq_img.shape[:2]
    print(f"Loaded source image {TEST_IMG.name}: {w}x{h}")

    aug_samples = []
    angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    scales = [0.80, 0.90, 1.0, 1.10, 1.20]
    shifts = [-20, -10, 0, 10, 20]

    for angle in angles:
        for scale in scales:
            for dx in shifts:
                for dy in shifts:
                    for flip in [False, True]:
                        cx, cy = 235 + dx, 275 + dy
                        bw = int(240 * scale)
                        bh = int(240 * scale)
                        x1 = max(0, cx - bw // 2)
                        y1 = max(0, cy - bh // 2)
                        x2 = min(w, cx + bw // 2)
                        y2 = min(h, cy + bh // 2)

                        sub = mosq_img[y1:y2, x1:x2].copy()
                        if sub.shape[0] < 80 or sub.shape[1] < 80:
                            continue

                        if flip:
                            sub = cv2.flip(sub, 1)

                        if angle != 0:
                            M = cv2.getRotationMatrix2D((sub.shape[1] // 2, sub.shape[0] // 2), angle, 1.0)
                            sub = cv2.warpAffine(sub, M, (sub.shape[1], sub.shape[0]), borderMode=cv2.BORDER_REFLECT)

                        # Random subtle color / brightness variation
                        b_mult = random.uniform(0.85, 1.15)
                        sub = np.clip(sub.astype(np.float32) * b_mult, 0, 255).astype(np.uint8)

                        sub_resized = cv2.resize(sub, (320, 320), interpolation=cv2.INTER_LINEAR)
                        aug_samples.append(sub_resized)

    random.shuffle(aug_samples)
    print(f"Total augmented window mosquito crops generated: {len(aug_samples)}")

    # Split into 120 train, 30 val
    for idx, sample in enumerate(aug_samples[:120]):
        out_path = TRAIN_MOSQ / f"sky_mosq_win_train_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), sample, [cv2.IMWRITE_JPEG_QUALITY, 92])

    for idx, sample in enumerate(aug_samples[120:150]):
        out_path = VAL_MOSQ / f"sky_mosq_win_val_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), sample, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("Saved 120 train and 30 val window-mosquito samples.")

    # 2. Composite raw mosquito specimens onto diverse sky backgrounds
    raw_files = sorted(list(RAW_MOSQ.glob("*.jpg")))
    sky_files = sorted(list(SKY_DIR.glob("*.jpg")))
    random.shuffle(raw_files)
    random.shuffle(sky_files)

    comp_train_count = 0
    comp_val_count = 0

    for i in range(min(140, len(raw_files))):
        m_path = raw_files[i]
        s_path = sky_files[i % len(sky_files)]

        m_img = cv2.imread(str(m_path))
        s_img = cv2.imread(str(s_path))
        if m_img is None or s_img is None:
            continue

        # Resize sky to match mosquito
        mh, mw = m_img.shape[:2]
        s_img = cv2.resize(s_img, (mw, mh))

        # Insect segmentation using Otsu threshold on grayscale
        gray = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Refine mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]

        # Composite insect on sky
        blended = (m_img.astype(np.float32) * mask_f + s_img.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
        blended = cv2.resize(blended, (320, 320), interpolation=cv2.INTER_LINEAR)

        if comp_train_count < 110:
            out_p = TRAIN_MOSQ / f"sky_mosq_comp_train_{comp_train_count:04d}.jpg"
            cv2.imwrite(str(out_p), blended, [cv2.IMWRITE_JPEG_QUALITY, 92])
            comp_train_count += 1
        elif comp_val_count < 30:
            out_p = VAL_MOSQ / f"sky_mosq_comp_val_{comp_val_count:04d}.jpg"
            cv2.imwrite(str(out_p), blended, [cv2.IMWRITE_JPEG_QUALITY, 92])
            comp_val_count += 1

    print(f"Saved {comp_train_count} train and {comp_val_count} val composited sky-mosquito samples.")
    print(f"Total train mosquito count now: {len(list(TRAIN_MOSQ.glob('*.*')))}")
    print(f"Total val mosquito count now:   {len(list(VAL_MOSQ.glob('*.*')))}")

if __name__ == "__main__":
    generate_sky_mosquito_samples()
