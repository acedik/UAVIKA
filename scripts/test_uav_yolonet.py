import os
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.uav_yolonet import UAV_YOLONet

# =========================================================
# CONFIGURATION
# =========================================================

CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]
IMAGE_SIZE = 160
CONFIDENCE_THRESHOLD = 0.50

# Threat Color Palette (BGR)
COLOR_DRONE    = (0, 0, 255)       # Red - Hostile Threat
COLOR_HELI     = (0, 140, 255)     # Orange - Elevated Traffic
COLOR_MOSQUITO = (0, 215, 255)     # Amber/Gold - Biological Pest / Micro-Target
COLOR_BIRD     = (0, 255, 0)       # Green - Biological Neutral
COLOR_OTHER    = (255, 200, 0)     # Cyan/Yellow - Unidentified


def parse_args():
    parser = argparse.ArgumentParser(description="Test Custom UAV-YOLONet on personal images")
    parser.add_argument(
        "--source",
        type=str,
        default=r"D:\UAVIkA\my_test_images",
        help="Path to a single image file or a folder of personal test images (default: D:\\UAVIkA\\my_test_images)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\UAVIKA\models\best_uav_yolonet_grouped.pth",
        help="Path to specimen-grouped five-class model weights"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\UAVIkA\test_results",
        help="Directory to save annotated images and predictions.csv"
    )
    return parser.parse_args()


def get_threat_profile(label_name):
    name = label_name.lower()
    if "drone" in name:
        return COLOR_DRONE, "DRONE [CRITICAL HOSTILE]", 3
    elif "helicopter" in name:
        return COLOR_HELI, "HELICOPTER [AIR TRAFFIC]", 2
    elif "mosquito" in name or "mosq" in name:
        return COLOR_MOSQUITO, "MOSQUITO [BIOLOGICAL TARGET]", 2
    elif "bird" in name:
        return COLOR_BIRD, "BIRD [BIOLOGICAL NEUTRAL]", 1
    else:
        return COLOR_OTHER, f"{name.upper()} [UNIDENTIFIED]", 1


def main():
    args = parse_args()
    model_path = Path(args.weights)
    source_path = Path(args.source)
    results_root = Path(args.output)

    run_name = datetime.now().strftime("yolonet_test_%Y%m%d_%H%M%S")
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("CUSTOM UAV-YOLONET: PERSONAL IMAGE INFERENCE")
    print("=" * 65)
    print(f"Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Checkpoint: {model_path}")
    print(f"Source:         {source_path}")
    print(f"Output Directory: {out_dir}")
    print("=" * 65)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load custom model
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", CLASSES)
    print(f"Supported Classes ({len(classes)}): {classes}")

    model = UAV_YOLONet(num_classes=len(classes), in_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        test_imgs = [source_path]
    elif source_path.is_dir():
        test_imgs = sorted([
            p for p in source_path.rglob("*")
            if p.suffix.lower() in valid_exts
        ])
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    if len(test_imgs) == 0:
        print(f"[WARNING] No images found with extensions {valid_exts} in {source_path}")
        return

    print(f"\nProcessing {len(test_imgs)} image(s)...")
    log_rows = []

    print("\n" + "-" * 75)
    print(f"{'Filename':<32} | {'Prediction':<12} | {'Confidence':<10} | {'Threat Level'}")
    print("-" * 75)

    for img_path in test_imgs:
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        h, w = orig.shape[:2]

        pil_im = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        inp = transform(pil_im).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        color, threat_text, threat_rank = get_threat_profile(pred_class)

        # Draw Tactical HUD Overlay
        annotated = orig.copy()
        
        # Telemetry Banner Top
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        # Status text
        tag = f"DETECTED: {threat_text} | CONFIDENCE: {confidence * 100:.1f}%"
        cv2.putText(annotated, tag, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        # Reticle in center
        cx, cy = w // 2, h // 2
        reticle_size = min(w, h) // 6
        cv2.rectangle(annotated, (cx - reticle_size, cy - reticle_size), 
                      (cx + reticle_size, cy + reticle_size), color, 2)
        cv2.drawMarker(annotated, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)

        # Bottom probabilities breakdown
        bot_bar = annotated.copy()
        cv2.rectangle(bot_bar, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(bot_bar, 0.8, annotated, 0.2, 0, annotated)

        breakdown = " | ".join([f"{c.capitalize()}: {probs[i]*100:.1f}%" for i, c in enumerate(classes)])
        cv2.putText(annotated, breakdown, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

        # Save output
        dest_path = out_dir / img_path.name
        cv2.imwrite(str(dest_path), annotated)

        print(f"{img_path.name:<32} | {pred_class.upper():<12} | {confidence*100:>8.1f}% | {threat_text}")

        row = {
            "filename": img_path.name,
            "prediction": pred_class,
            "confidence": f"{confidence:.4f}",
        }
        for c_idx, c_name in enumerate(classes):
            row[f"{c_name}_prob"] = f"{probs[c_idx]:.4f}"
        log_rows.append(row)

    # Save CSV
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "prediction", "confidence"] + [f"{c}_prob" for c in classes]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("-" * 75)
    print(f"\n[COMPLETE] Annotated images saved to: {out_dir}")
    print(f"[COMPLETE] Full predictions CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
