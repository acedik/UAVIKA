import os
import csv
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = r"D:\UAVIkA\models\best_anti_uav_cnn.pth"

TEST_FOLDER = r"D:\UAVIkA\my_test_images"

RESULTS_ROOT = r"D:\UAVIkA\test_results"

CLASS_NAMES = ["bird", "drone", "helicopter", "other"]

IMAGE_SIZE = 160

CONFIDENCE_THRESHOLD = 0.70


# =========================================================
# CREATE A NEW RESULTS FOLDER
# =========================================================

run_name = datetime.now().strftime("test_%Y%m%d_%H%M%S")

RESULTS_FOLDER = os.path.join(RESULTS_ROOT, run_name)

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("Results folder:", RESULTS_FOLDER)


# =========================================================
# SELECT DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])

else:
    model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# FIND TEST IMAGES
# =========================================================

if not os.path.isdir(TEST_FOLDER):
    raise FileNotFoundError(f"Test folder not found: {TEST_FOLDER}")

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

image_paths = []

for root, _, files in os.walk(TEST_FOLDER):

    for filename in files:

        if filename.lower().endswith(extensions):

            image_paths.append(os.path.join(root, filename))

image_paths.sort()

if not image_paths:
    raise RuntimeError(f"No images found inside {TEST_FOLDER}")

print(f"Found {len(image_paths)} test images.")


# =========================================================
# CREATE CSV REPORT
# =========================================================

csv_path = os.path.join(RESULTS_FOLDER, "predictions.csv")

correctly_named = 0
uncertain_count = 0
failed_count = 0


# =========================================================
# PREDICT AND ANNOTATE
# =========================================================

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Image",
        "Predicted_Class",
        "Confidence_Percent",
        "Status",
        "Saved_Image"
    ])

    for index, image_path in enumerate(image_paths, start=1):

        try:

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Preprocess
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, dim=1)

            class_name = CLASS_NAMES[predicted.item()]

            confidence_value = confidence.item() * 100

            # Apply confidence threshold
            if confidence_value < CONFIDENCE_THRESHOLD * 100:

                label = f"UNCERTAIN: {class_name.upper()} {confidence_value:.2f}%"

                status = "UNCERTAIN"

                uncertain_count += 1

            else:

                label = f"{class_name.upper()} {confidence_value:.2f}%"

                status = "PREDICTED"

                correctly_named += 1

            # -------------------------------------------------
            # PREPARE IMAGE FOR ANNOTATION
            # -------------------------------------------------

            annotated_image = cv2.imread(image_path)

            if annotated_image is None:
                raise ValueError("OpenCV could not read this image.")

            height, width = annotated_image.shape[:2]

            # Scale font according to image dimensions
            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = max(0.6, min(width, height) / 900)

            thickness = max(1, int(font_scale * 2))

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # Keep label inside the image
            x = 10
            y = min(height - 5, text_height + 20)

            # Draw background
            cv2.rectangle(
                annotated_image,
                (x, max(0, y - text_height - 10)),
                (
                    min(width - 1, x + text_width + 15),
                    min(height - 1, y + baseline + 5)
                ),
                (0, 0, 0),
                -1
            )

            # Green for predictions, yellow for uncertain results
            text_color = (0, 255, 0) if status == "PREDICTED" else (0, 255, 255)

            # Draw label
            cv2.putText(
                annotated_image,
                label,
                (x + 5, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # SAVE ANNOTATED IMAGE
            # -------------------------------------------------

            relative_path = os.path.relpath(image_path, TEST_FOLDER)

            relative_folder = os.path.dirname(relative_path)

            output_folder = os.path.join(RESULTS_FOLDER, relative_folder)

            os.makedirs(output_folder, exist_ok=True)

            original_name = os.path.basename(image_path)

            base_name, _ = os.path.splitext(original_name)

            output_path = os.path.join(
                output_folder,
                base_name + "_annotated.jpg"
            )

            # Save as JPEG
            success = cv2.imwrite(output_path, annotated_image)

            if not success:
                raise IOError("Could not save annotated image.")

            # -------------------------------------------------
            # WRITE CSV ENTRY
            # -------------------------------------------------

            writer.writerow([
                image_path,
                class_name,
                f"{confidence_value:.2f}",
                status,
                output_path
            ])

            print(
                f"[{index}/{len(image_paths)}] "
                f"{original_name} -> {label}"
            )

        except Exception as error:

            failed_count += 1

            print(f"Error processing {image_path}: {error}")

            writer.writerow([
                image_path,
                "ERROR",
                "",
                str(error),
                ""
            ])


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 55)

print("TESTING COMPLETED")

print("=" * 55)

print(f"Total images:          {len(image_paths)}")
print(f"Predictions >= 70%:    {correctly_named}")
print(f"Uncertain predictions: {uncertain_count}")
print(f"Failed images:         {failed_count}")

print(f"\nAnnotated images: {RESULTS_FOLDER}")

print(f"CSV report: {csv_path}")

print("=" * 55)