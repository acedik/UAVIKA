import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UAVYoloDataset(Dataset):
    """
    Dataset loader for YOLO-format drone bounding-box data.
    Reads images and corresponding .txt label files:
    Format in label txt: <class_id> <xc> <yc> <w> <h> (normalized to [0, 1])
    """

    def __init__(self, img_dir, label_dir, img_size=640, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.is_train = is_train

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.img_files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in valid_exts
        ])

        if len(self.img_files) == 0:
            raise RuntimeError(f"No image files found in {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback for corrupted images
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]

        # Read labels
        boxes = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        # Convert normalized xc, yc, w, h to orig pixels x1, y1, x2, y2
                        x1 = (xc - w / 2.0) * orig_w
                        y1 = (yc - h / 2.0) * orig_h
                        x2 = (xc + w / 2.0) * orig_w
                        y2 = (yc + h / 2.0) * orig_h
                        boxes.append([cls_id, x1, y1, x2, y2])

        # Multi-scale zoom augmentation (simulates close-up drones during training)
        if self.is_train and len(boxes) > 0 and random.random() < 0.45:
            tgt = random.choice(boxes)
            _, tx1, ty1, tx2, ty2 = tgt
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            tbw = max(10, tx2 - tx1)
            tbh = max(10, ty2 - ty1)

            crop_factor = random.uniform(2.0, 5.0)
            crop_w = min(orig_w, max(64, int(tbw * crop_factor)))
            crop_h = min(orig_h, max(64, int(tbh * crop_factor)))

            cx_min = max(crop_w // 2, int(tcx) - crop_w // 4)
            cx_max = min(orig_w - crop_w // 2, int(tcx) + crop_w // 4)
            cy_min = max(crop_h // 2, int(tcy) - crop_h // 4)
            cy_max = min(orig_h - crop_h // 2, int(tcy) + crop_h // 4)

            cx = random.randint(min(cx_min, cx_max), max(cx_min, cx_max)) if cx_min != cx_max else cx_min
            cy = random.randint(min(cy_min, cy_max), max(cy_min, cy_max)) if cy_min != cy_max else cy_min

            x1_crop = max(0, cx - crop_w // 2)
            y1_crop = max(0, cy - crop_h // 2)
            x2_crop = min(orig_w, x1_crop + crop_w)
            y2_crop = min(orig_h, y1_crop + crop_h)

            crop_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
            if crop_img.shape[0] >= 32 and crop_img.shape[1] >= 32:
                img = crop_img
                cropped_boxes = []
                for b in boxes:
                    cls_id, bx1, by1, bx2, by2 = b
                    nx1 = max(0, bx1 - x1_crop)
                    ny1 = max(0, by1 - y1_crop)
                    nx2 = min(img.shape[1], bx2 - x1_crop)
                    ny2 = min(img.shape[0], by2 - y1_crop)
                    if (nx2 - nx1) >= 4 and (ny2 - ny1) >= 4:
                        cropped_boxes.append([cls_id, nx1, ny1, nx2, ny2])
                if len(cropped_boxes) > 0:
                    boxes = cropped_boxes
                    orig_h, orig_w = img.shape[:2]

        # Resize image to input resolution
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        # Scale box coordinates
        scaled_boxes = []
        for box in boxes:
            cls_id, x1, y1, x2, y2 = box
            bx1 = np.clip(x1 * scale_x, 0, self.img_size)
            by1 = np.clip(y1 * scale_y, 0, self.img_size)
            bx2 = np.clip(x2 * scale_x, 0, self.img_size)
            by2 = np.clip(y2 * scale_y, 0, self.img_size)
            if (bx2 - bx1) > 1 and (by2 - by1) > 1:
                scaled_boxes.append([cls_id, bx1, by1, bx2, by2])

        # Data augmentation (training only)
        if self.is_train:
            # 1. Random horizontal flip
            if random.random() < 0.5:
                img_resized = cv2.flip(img_resized, 1)
                flipped_boxes = []
                for b in scaled_boxes:
                    cls_id, bx1, by1, bx2, by2 = b
                    new_x1 = self.img_size - bx2
                    new_x2 = self.img_size - bx1
                    flipped_boxes.append([cls_id, new_x1, by1, new_x2, by2])
                scaled_boxes = flipped_boxes

            # 2. Random brightness/contrast jitter for thermal/optical variation
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-20, 20)    # Brightness
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

        # Convert to RGB tensor (3, H, W) normalized to [0, 1]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        target_tensor = torch.tensor(scaled_boxes, dtype=torch.float32) if len(scaled_boxes) > 0 else torch.zeros((0, 5))

        return tensor_img, target_tensor, str(img_path)


def yolo_collate_fn(batch):
    """
    Collate function to batch images and dynamically merge variable target boxes.
    Outputs:
    - images: (B, 3, H, W)
    - targets: (N_total_targets, 6) -> [batch_idx, class_id, x1, y1, x2, y2]
    - paths: list of image paths
    """
    images, targets, paths = zip(*batch)
    images = torch.stack(images, dim=0)

    batched_targets = []
    for b_idx, t in enumerate(targets):
        if t.shape[0] > 0:
            b_col = torch.full((t.shape[0], 1), b_idx, dtype=torch.float32)
            # [batch_idx, class_id, x1, y1, x2, y2]
            batched_targets.append(torch.cat([b_col, t], dim=1))

    if len(batched_targets) > 0:
        targets = torch.cat(batched_targets, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, targets, list(paths)
