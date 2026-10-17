import cv2
import json
import random
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_DIR = Path(
    r"D:\UAVIkA\dataset\raw\drone\train\Vid\train"
)

OUTPUT_DIR = Path(
    r"D:\UAVIkA\processed\yolo_drone"
)

# Extract every 5th frame to keep the dataset manageable.
FRAME_STRIDE = 5

# 20% of sequences for validation.
VAL_RATIO = 0.20

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# --------------------------------------------------
# FIND VIDEO SEQUENCES
# --------------------------------------------------

sequences = sorted(
    p for p in RAW_DIR.iterdir()
    if p.is_dir()
)

if not sequences:
    raise RuntimeError(
        f"No sequence folders found in {RAW_DIR}"
    )

random.shuffle(sequences)

val_count = max(1, round(len(sequences) * VAL_RATIO))

val_sequences = set(sequences[:val_count])

print("Total sequences:", len(sequences))
print("Validation sequences:", len(val_sequences))
print("Training sequences:", len(sequences) - len(val_sequences))

# --------------------------------------------------
# PROCESS VIDEOS AND ANNOTATIONS
# --------------------------------------------------

total_frames_saved = 0
total_positive = 0
total_negative = 0
total_skipped = 0

for sequence in tqdm(sequences, desc="Sequences"):

    split = "val" if sequence in val_sequences else "train"

    for modality in ["infrared", "visible"]:

        video_path = sequence / f"{modality}.mp4"
        json_path = sequence / f"{modality}.json"

        # Skip missing modality files.
        if not video_path.exists() or not json_path.exists():
            continue

        # Load annotations.
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            exist = annotations["exist"]
            boxes = annotations["gt_rect"]

        except Exception as e:
            print(f"\nCould not read {json_path}: {e}")
            continue

        if len(exist) != len(boxes):
            print(f"\nAnnotation count mismatch: {sequence.name}/{modality}")
            continue

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"\nCould not open video: {video_path}")
            continue

        frame_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Only save every Nth frame.
            if frame_index % FRAME_STRIDE == 0:

                height, width = frame.shape[:2]

                present = exist[frame_index]

                # Unique filename for each sequence/modality/frame.
                filename = (
                    f"{sequence.name}_{modality}_{frame_index:06d}"
                )

                image_path = (
                    OUTPUT_DIR / "images" / split / f"{filename}.jpg"
                )

                label_path = (
                    OUTPUT_DIR / "labels" / split / f"{filename}.txt"
                )

                label_line = None

                if present == 1:

                    box = boxes[frame_index]

                    if isinstance(box, list) and len(box) == 4:

                        x, y, w, h = map(float, box)

                        # Convert xywh to clipped image coordinates.
                        x1 = max(0.0, min(x, float(width)))
                        y1 = max(0.0, min(y, float(height)))
                        x2 = max(0.0, min(x + w, float(width)))
                        y2 = max(0.0, min(y + h, float(height)))

                        bw = x2 - x1
                        bh = y2 - y1

                        if bw > 0 and bh > 0:

                            # YOLO format:
                            # class_id x_center y_center width height
                            xc = ((x1 + x2) / 2.0) / width
                            yc = ((y1 + y2) / 2.0) / height
                            nw = bw / width
                            nh = bh / height

                            label_line = (
                                f"0 {xc:.6f} {yc:.6f} "
                                f"{nw:.6f} {nh:.6f}"
                            )

                # Save image.
                if cv2.imwrite(str(image_path), frame):

                    # Positive image: one drone label.
                    # Negative image: empty label file.
                    with open(label_path, "w", encoding="utf-8") as f:
                        if label_line is not None:
                            f.write(label_line + "\n")
                            total_positive += 1
                        else:
                            total_negative += 1

                    total_frames_saved += 1

                else:
                    total_skipped += 1

            frame_index += 1

        cap.release()

# --------------------------------------------------
# WRITE YOLO DATASET YAML
# --------------------------------------------------

yaml_text = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: drone
"""

yaml_path = OUTPUT_DIR / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_text)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n========================================")
print("YOLO DATASET PREPARATION COMPLETE")
print("========================================")
print("Output folder:", OUTPUT_DIR)
print("Dataset YAML:", yaml_path)
print("Frames saved:", total_frames_saved)
print("Positive frames:", total_positive)
print("Empty-label frames:", total_negative)
print("Images skipped:", total_skipped)
print("========================================")