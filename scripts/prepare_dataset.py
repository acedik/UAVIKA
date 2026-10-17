import random
import shutil
from pathlib import Path

# Dataset locations
RAW_DIR = Path(r"D:\UAVIkA\dataset\raw")
OUTPUT_DIR = Path(r"D:\UAVIkA\processed\classification")

CLASSES = ["bird", "drone", "helicopter", "other"]

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Create output directories
for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# Collect images
for class_name in CLASSES:

    class_dir = RAW_DIR / class_name

    images = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Avoid accidentally including previously extracted crops
    images = [
        path for path in images
        if OUTPUT_DIR not in path.parents
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    print(f"\nClass: {class_name}")
    print(f"Total images: {total}")

    for split_name, split_images in split_data.items():

        destination = OUTPUT_DIR / split_name / class_name

        for index, source in enumerate(split_images):

            # Unique names prevent collisions between subfolders
            filename = f"{class_name}_{index:06d}{source.suffix.lower()}"

            target = destination / filename

            shutil.copy2(source, target)

        print(f"{split_name}: {len(split_images)} images")

print("\nDataset preparation completed.")
print(f"Dataset saved at: {OUTPUT_DIR}")