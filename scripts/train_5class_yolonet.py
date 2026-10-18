import os
import sys
import time
import json
import random
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"D:\UAVIKA\processed\five_class_grouped")
PRETRAINED_WEIGHTS = Path(r"D:\UAVIKA\models\best_uav_yolonet_4class_orig.pth")
OUTPUT_DIR = Path(r"D:\UAVIKA\models")
LOGS_DIR = Path(r"D:\UAVIKA\runs\uav_yolonet")

BATCH_SIZE = 32
EPOCHS = 18
IMAGE_SIZE = 160
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_header(title):
    print("\n" + "=" * 70, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 70, flush=True)

def compute_classification_metrics(targets, preds, classes):
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    report_lines = []
    report_lines.append(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    report_lines.append("-" * 70)

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

    report_lines.append("-" * 70)
    report_lines.append(f"{'Accuracy':<15} | {'':<10} | {'':<10} | {acc:<10.4f} | {total_sup:<8}")
    report_lines.append(f"{'Macro Avg':<15} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f} | {total_sup:<8}")

    return cm, "\n".join(report_lines), float(acc), {
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "supports": supports,
        "accuracy": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1
    }


def calibrate_mosquito_threshold(model, val_loader, classes, precision_target=0.99):
    """Choose the highest-recall full-frame gate meeting validation precision."""
    mosquito_idx = classes.index("mosquito")
    records = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                probs = torch.softmax(model(images), dim=1)
            pred = probs.argmax(dim=1).cpu().numpy()
            mosquito_prob = probs[:, mosquito_idx].cpu().numpy()
            actual = labels.numpy()
            records.extend(zip(actual.tolist(), pred.tolist(), mosquito_prob.tolist()))

    candidates = sorted({0.0, 1.0, *(float(row[2]) for row in records)})
    eligible = []
    for threshold in candidates:
        tp = sum(y == mosquito_idx and p == mosquito_idx and score >= threshold for y, p, score in records)
        fp = sum(y != mosquito_idx and p == mosquito_idx and score >= threshold for y, p, score in records)
        fn = sum(y == mosquito_idx and not (p == mosquito_idx and score >= threshold) for y, p, score in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision >= precision_target and tp + fp:
            eligible.append((recall, threshold, precision, tp, fp, fn))

    if not eligible:
        return 1.0, {"precision_target": precision_target, "tp": 0, "fp": 0, "fn": int(sum(y == mosquito_idx for y, _, _ in records))}
    recall, threshold, precision, tp, fp, fn = max(eligible, key=lambda row: (row[0], row[1]))
    return threshold, {
        "precision_target": precision_target,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

def get_data_loaders(data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.15)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    print(f"Class Mapping: {train_dataset.class_to_idx}", flush=True)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx, train_dataset.targets

def load_transfer_weights(model, pretrained_path):
    """Transfers all compatible layers from 4-class UAV-YOLONet checkpoint."""
    if not pretrained_path.exists():
        print(f"Pretrained weights not found at {pretrained_path}, initializing from scratch.")
        return

    print(f"Loading pretrained backbone and feature layers from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=DEVICE)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    transferred = 0
    skipped = 0
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            transferred += 1
        else:
            skipped += 1

    model.load_state_dict(model_dict)
    baseline_val_acc = float(checkpoint.get("val_acc", 0.0))
    print(f"Weights transferred successfully! Loaded {transferred} layers, adapted {skipped} layers (final classifier head). Baseline Val Acc: {baseline_val_acc * 100:.2f}%")
    return baseline_val_acc

def train_model(args):
    data_dir = args.data_dir
    output_checkpoint = args.output_checkpoint
    summary_path = args.summary
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    print_header("5-CLASS FINE-TUNING & CALIBRATION: UAV-YOLONet")
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Target Epochs:   {epochs}", flush=True)
    print(f"Batch Size:      {batch_size}", flush=True)
    print(f"Resolution:      {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"Dataset:         {data_dir}", flush=True)
    print(f"Output:          {output_checkpoint}", flush=True)

    train_loader, val_loader, test_loader, class_to_idx, train_targets = get_data_loaders(data_dir, batch_size)

    # Calculate inverse class frequency weights for balanced training
    class_counts = np.bincount(train_targets, minlength=len(CLASSES))
    total_samples = len(train_targets)
    weights = total_samples / (len(CLASSES) * class_counts.astype(float))
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print(f"Training Class Counts: {dict(zip(CLASSES, class_counts))}")
    print(f"Computed Class Weights: {dict(zip(CLASSES, np.round(weights, 2)))}")

    # Instantiate 5-class UAV_YOLONet
    model = UAV_YOLONet(num_classes=len(CLASSES), in_channels=3).to(DEVICE)
    baseline_val_acc = load_transfer_weights(model, args.pretrained) or 0.0

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    best_val_acc = 0.0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_path = output_checkpoint
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
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
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start

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
            f"Epoch {epoch:02d}/{epochs:02d} ({epoch_time:.1f}s) | "
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
    print_header(f"FINE-TUNING COMPLETE IN {total_training_time:.1f}s (~{total_training_time/60:.1f} min)")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%", flush=True)

    # Final Test Evaluation on Unseen Test Set
    print_header("FINAL EVALUATION ON HELD-OUT UNSEEN TEST SET")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mosquito_threshold, mosquito_threshold_metrics = calibrate_mosquito_threshold(model, val_loader, CLASSES)
    checkpoint["mosquito_frame_threshold"] = float(mosquito_threshold)
    checkpoint["mosquito_threshold_validation"] = mosquito_threshold_metrics
    torch.save(checkpoint, best_model_path)
    print(
        f"Validation-calibrated full-frame mosquito threshold: {mosquito_threshold:.6f} "
        f"(P={mosquito_threshold_metrics.get('precision', 0):.3f}, "
        f"R={mosquito_threshold_metrics.get('recall', 0):.3f})",
        flush=True,
    )

    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / images.size(0))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    cm, report_text, test_acc, detailed_metrics = compute_classification_metrics(all_targets, all_preds, CLASSES)

    print("\nClassification Report:", flush=True)
    print(report_text, flush=True)
    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):", flush=True)
    print(f"{'':>12}" + "".join([f"{c:>12}" for c in CLASSES]), flush=True)
    for idx, row in enumerate(cm):
        print(f"{CLASSES[idx]:>12}" + "".join([f"{val:>12}" for val in row]), flush=True)

    mean_latency_ms = np.mean(latencies) * 1000
    print(f"\nMean Inference Latency: {mean_latency_ms:.2f} ms per candidate crop", flush=True)

    # Save summary report to JSON
    results = {
        "architecture": "UAV_YOLONet_5Class",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": IMAGE_SIZE,
        "classes": CLASSES,
        "train_time_sec": total_training_time,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "mean_latency_ms": float(mean_latency_ms),
        "mosquito_frame_threshold": float(mosquito_threshold),
        "mosquito_threshold_validation": mosquito_threshold_metrics,
        "detailed_metrics": detailed_metrics,
        "confusion_matrix": cm.tolist()
    }
    def to_py(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_py(v) for v in obj]
        return obj

    with open(summary_path, "w") as f:
        json.dump(to_py(results), f, indent=2)

    print(f"\nTraining logs and metrics saved to: {summary_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune UAV-YOLONet using the existing 5-class framework")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_WEIGHTS)
    parser.add_argument("--output-checkpoint", type=Path, default=OUTPUT_DIR / "best_uav_yolonet_grouped.pth")
    parser.add_argument("--summary", type=Path, default=ROOT / "runs" / "diagnostics" / "grouped_5class_training.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_model(parser.parse_args())
