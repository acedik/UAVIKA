import random
from pathlib import Path

import cv2

# Dataset paths
DATASET_DIR = Path(r"D:\UAVIkA\processed\yolo_drone")
IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

OUTPUT_DIR = Path(r"D:\UAVIkA\processed\yolo_preview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Find images that have label files
images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and (LABEL_DIR / f"{p.stem}.txt").exists()
]

if not images:
    raise RuntimeError("No matching images and label files found.")

# Select up to 12 images
selected = random.sample(images, min(12, len(images)))

print("Selected images:", len(selected))

for image_path in selected:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:

        values = line.split()

        if len(values) != 5:
            print("Invalid label:", label_path, line)
            continue

        class_id, xc, yc, bw, bh = map(float, values)

        # Convert normalized YOLO coordinates to pixels
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        # Draw red bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            "DRONE",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    print("Saved:", output_path)

print("\nPreview complete.")
print("Check the images in:", OUTPUT_DIR)