import cv2
import json
from pathlib import Path

# Input sequence
BASE = Path(
    r"D:\UAVIkA\dataset\raw\drone\train\Vid\train\20190925_101846_1_1"
)

# Output directory
OUTPUT = Path(r"D:\UAVIkA\processed\drone_crops\infrared")
OUTPUT.mkdir(parents=True, exist_ok=True)

video_path = BASE / "infrared.mp4"
json_path = BASE / "infrared.json"

# Load annotations
with open(json_path, "r") as f:
    annotations = json.load(f)

exists = annotations["exist"]
boxes = annotations["gt_rect"]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

frame_index = 0
saved = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame_index >= len(exists) or frame_index >= len(boxes):
        break

    # Extract only frames where the drone exists
    if exists[frame_index]:
        x, y, width, height = map(int, boxes[frame_index])

        # Clip coordinates to image boundaries
        h, w = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]

            if crop.size:
                output_file = OUTPUT / f"frame_{frame_index:05d}.jpg"
                cv2.imwrite(str(output_file), crop)
                saved += 1

    frame_index += 1

cap.release()

print("Frames processed:", frame_index)
print("Drone crops saved:", saved)
print("Output directory:", OUTPUT)