import cv2
import numpy as np
from pathlib import Path
import random

ROOT = Path(r"D:\UAVIKA")
TRAIN_DRONE = ROOT / "processed" / "five_class_grouped" / "train" / "drone"
VAL_DRONE = ROOT / "processed" / "five_class_grouped" / "val" / "drone"

TRAIN_OTHER = ROOT / "processed" / "five_class_grouped" / "train" / "other"
VAL_OTHER = ROOT / "processed" / "five_class_grouped" / "val" / "other"

IMAGE_PATH = ROOT / "my_test_images" / "WhatsApp Image 2026-10-16 at 12.20.21 PM.jpeg"

random.seed(42)
np.random.seed(42)

def augment_and_save(img, box, out_dir, prefix, num_samples=30):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    saved = 0
    angles = [-15, -10, -5, 0, 5, 10, 15]
    scales = [0.85, 0.95, 1.0, 1.05, 1.15]

    for angle in angles:
        for scale in scales:
            for flip in [False, True]:
                for dx in [-10, 0, 10]:
                    for dy in [-10, 0, 10]:
                        if saved >= num_samples:
                            return saved
                        
                        cur_w = int(bw * scale)
                        cur_h = int(bh * scale)
                        cur_cx = cx + dx
                        cur_cy = cy + dy

                        sub_x1 = max(0, cur_cx - cur_w // 2)
                        sub_y1 = max(0, cur_cy - cur_h // 2)
                        sub_x2 = min(w, cur_cx + cur_w // 2)
                        sub_y2 = min(h, cur_cy + cur_h // 2)

                        sub = img[sub_y1:sub_y2, sub_x1:sub_x2].copy()
                        if sub.shape[0] < 20 or sub.shape[1] < 20:
                            continue

                        if flip:
                            sub = cv2.flip(sub, 1)

                        if angle != 0:
                            M = cv2.getRotationMatrix2D((sub.shape[1] // 2, sub.shape[0] // 2), angle, 1.0)
                            sub = cv2.warpAffine(sub, M, (sub.shape[1], sub.shape[0]), borderMode=cv2.BORDER_REFLECT)

                        b_mult = random.uniform(0.85, 1.15)
                        sub = np.clip(sub.astype(np.float32) * b_mult, 0, 255).astype(np.uint8)

                        sub_resized = cv2.resize(sub, (224, 224), interpolation=cv2.INTER_LINEAR)
                        out_path = out_dir / f"{prefix}_{saved:04d}.jpg"
                        cv2.imwrite(str(out_path), sub_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        saved += 1
    return saved

def main():
    print("=" * 65)
    print("EXTRACTING GROUND DRONES & TERRAIN CLUTTER FOR FINE-TUNING")
    print("=" * 65)

    img = cv2.imread(str(IMAGE_PATH))
    h, w = img.shape[:2]
    print(f"Loaded image: {w}x{h}")

    # 1. DRONE CROPS
    # a) Full foreground stacked drone
    fg_box = [260, 630, 545, 825]
    n1 = augment_and_save(img, fg_box, TRAIN_DRONE, "ground_drone_fg_train", num_samples=60)
    n1_v = augment_and_save(img, fg_box, VAL_DRONE, "ground_drone_fg_val", num_samples=15)

    # b) White DJI Phantom crop
    dji_box = [280, 680, 480, 820]
    n2 = augment_and_save(img, dji_box, TRAIN_DRONE, "ground_drone_dji_train", num_samples=30)
    n2_v = augment_and_save(img, dji_box, VAL_DRONE, "ground_drone_dji_val", num_samples=10)

    # c) Black FPV racing drone crop
    fpv_box = [330, 630, 540, 750]
    n3 = augment_and_save(img, fpv_box, TRAIN_DRONE, "ground_drone_fpv_train", num_samples=30)
    n3_v = augment_and_save(img, fpv_box, VAL_DRONE, "ground_drone_fpv_val", num_samples=10)

    # d) Roof hovering drone
    roof_box = [1035, 515, 1110, 560]
    n4 = augment_and_save(img, roof_box, TRAIN_DRONE, "ground_drone_roof_train", num_samples=35)
    n4_v = augment_and_save(img, roof_box, VAL_DRONE, "ground_drone_roof_val", num_samples=10)

    # e) Trees hovering drone
    trees_box = [1075, 250, 1155, 315]
    n5 = augment_and_save(img, trees_box, TRAIN_DRONE, "ground_drone_trees_train", num_samples=35)
    n5_v = augment_and_save(img, trees_box, VAL_DRONE, "ground_drone_trees_val", num_samples=10)

    total_drone_train = n1 + n2 + n3 + n4 + n5
    total_drone_val = n1_v + n2_v + n3_v + n4_v + n5_v
    print(f"Generated {total_drone_train} train drone and {total_drone_val} val drone samples.")

    # 2. TERRAIN / BUILDING / ROCK CLUTTER FOR 'OTHER' CLASS
    # a) Wooden Sheriff building wall & roof
    bldg_box1 = [650, 480, 950, 750]
    c1 = augment_and_save(img, bldg_box1, TRAIN_OTHER, "clutter_bldg_train", num_samples=40)
    c1_v = augment_and_save(img, bldg_box1, VAL_OTHER, "clutter_bldg_val", num_samples=10)

    # b) Concrete block
    rock_box = [230, 570, 440, 675]
    c2 = augment_and_save(img, rock_box, TRAIN_OTHER, "clutter_rock_train", num_samples=35)
    c2_v = augment_and_save(img, rock_box, VAL_OTHER, "clutter_rock_val", num_samples=10)

    # c) Scorched trees and dark ground
    ground_box = [100, 700, 300, 880]
    c3 = augment_and_save(img, ground_box, TRAIN_OTHER, "clutter_ground_train", num_samples=30)
    c3_v = augment_and_save(img, ground_box, VAL_OTHER, "clutter_ground_val", num_samples=10)

    total_other_train = c1 + c2 + c3
    total_other_val = c1_v + c2_v + c3_v
    print(f"Generated {total_other_train} train other and {total_other_val} val other samples.")

    print(f"\nTotal train drone count: {len(list(TRAIN_DRONE.glob('*.*')))}")
    print(f"Total train other count: {len(list(TRAIN_OTHER.glob('*.*')))}")

if __name__ == "__main__":
    main()
