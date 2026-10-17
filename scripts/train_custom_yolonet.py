import os
import sys
import time
import json
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.uav_yolonet import UAV_YOLONet

# =====================================================================
# CONFIGURATION
# =====================================================================

DATA_DIR = Path(r"D:\UAVIKA\processed\balanced_classification")
OUTPUT_DIR = Path(r"D:\UAVIKA\models")
LOGS_DIR = Path(r"D:\UAVIKA\runs\uav_yolonet")

BATCH_SIZE = 32
EPOCHS = 60         # Extended from 10 -> 60: fixes 'other' class never firing in production
IMAGE_SIZE = 160
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Class weights: boost 'other' recall (index 3) - it never wins on real crops
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.4]

# MixUp: blends pairs of images -> calibrates softmax, prevents forced class collapse
MIXUP_ALPHA = 0.2

NUM_WORKERS = 0
SEED = 42

CLASSES = ["bird", "drone", "helicopter", "other"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_header(title):
    print("\n" + "=" * 65, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 65, flush=True)

# Pure Python / NumPy metrics to avoid Windows AppControl DLL blocking
def compute_classification_metrics(targets, preds, classes):
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    report_lines = []
    report_lines.append(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    report_lines.append("-" * 65)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        sup = cm[i, :].sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(sup)

        report_lines.append(f"{name:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {sup:<8}")

    total_sup = sum(supports)
    acc = np.trace(cm) / total_sup if total_sup > 0 else 0.0
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    report_lines.append("-" * 65)
    report_lines.append(f"{'Accuracy':<15} | {'':<10} | {'':<10} | {acc:<10.4f} | {total_sup:<8}")
    report_lines.append(f"{'Macro Avg':<15} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f} | {total_sup:<8}")

    return cm, "\n".join(report_lines), float(acc)

# =====================================================================
# DATA PIPELINE
# =====================================================================

def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # occlusion robustness
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    print(f"Class Mapping: {train_dataset.class_to_idx}", flush=True)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# =====================================================================
# MAIN TRAINING ENGINE
# =====================================================================

def train_model():
    print_header("CUSTOM YOLO-INSPIRED ANTI-UAV TRAINING (FROM SCRATCH)")
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Target Epochs:   {EPOCHS}", flush=True)
    print(f"Batch Size:      {BATCH_SIZE}", flush=True)
    print(f"Resolution:      {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)

    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders()

    # Instantiate custom UAV-YOLONet architecture
    model = UAV_YOLONet(num_classes=len(CLASSES), in_channels=3)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (100% custom from-scratch architecture)", flush=True)

    # Class-weighted label-smoothing CrossEntropy: boosts 'other' recall
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "best_uav_yolonet.pth"
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---------- 1. Training Phase ----------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # MixUp: blend random image pairs to calibrate softmax confidence
            use_mixup = MIXUP_ALPHA > 0 and random.random() < 0.5
            if use_mixup:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                index = torch.randperm(images.size(0), device=DEVICE)
                images = lam * images + (1 - lam) * images[index]

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                if use_mixup:
                    loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(train_correct / train_total) * 100:.1f}%"
            })

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        # Step LR scheduler
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # ---------- 2. Validation Phase ----------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start

        # Record metrics
        status = " "
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_epoch_acc,
                "classes": CLASSES,
                "class_to_idx": class_to_idx
            }, best_model_path)
            status = " [BEST MODEL SAVED]"

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc * 100:.2f}% | "
            f"Val Loss: {val_epoch_loss:.4f} - Acc: {val_epoch_acc * 100:.2f}% | "
            f"LR: {curr_lr:.6f}{status}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "train_acc": train_epoch_acc,
            "val_loss": val_epoch_loss,
            "val_acc": val_epoch_acc,
            "lr": curr_lr
        })

    total_training_time = time.time() - start_time
    print_header(f"TRAINING COMPLETE IN {total_training_time:.1f} SECONDS (~{total_training_time/60:.1f} MIN)")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%", flush=True)
    print(f"Best Model Weights Saved: {best_model_path}", flush=True)

    # ---------- 3. Final Test Evaluation ----------
    print_header("FINAL EVALUATION ON UNSEEN TEST SET")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    cm, report_text, test_acc = compute_classification_metrics(all_targets, all_preds, CLASSES)

    print("\nClassification Report:", flush=True)
    print(report_text, flush=True)
    print("\nConfusion Matrix:", flush=True)
    print(f"{'':>12}" + "".join([f"{c:>12}" for c in CLASSES]), flush=True)
    for idx, row in enumerate(cm):
        print(f"{CLASSES[idx]:>12}" + "".join([f"{val:>12}" for val in row]), flush=True)

    # Save summary report to JSON
    results = {
        "architecture": "UAV_YOLONet",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "train_time_sec": total_training_time,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "history": history,
        "confusion_matrix": cm.tolist()
    }
    with open(LOGS_DIR / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining logs and metrics saved to: {LOGS_DIR / 'training_summary.json'}", flush=True)

if __name__ == "__main__":
    train_model()
