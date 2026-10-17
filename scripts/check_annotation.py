import cv2
import json
from pathlib import Path

base = Path(
    r"D:\UAVIkA\dataset\raw\drone\train\Vid\train\20190925_101846_1_1"
)

video_path = base / "infrared.mp4"
json_path = base / "infrared.json"

# Load annotation data
with open(json_path, "r") as f:
    annotations = json.load(f)

# Read the first video frame
cap = cv2.VideoCapture(str(video_path))
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read the video frame.")

# Get first-frame annotation
exists = annotations["exist"][0]
box = annotations["gt_rect"][0]

print("Target exists:", exists)
print("Raw bounding box:", box)

if exists:
    x, y, width, height = map(int, box)

    # Draw the bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

# Save the annotated frame
output_path = Path(r"D:\UAVIkA\scripts\annotated_frame.jpg")
cv2.imwrite(str(output_path), frame)

print("Saved annotated frame to:", output_path)