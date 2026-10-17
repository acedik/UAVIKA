import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = Path(r"D:\UAVIkA\processed\classification")
OUTPUT_DIR = Path(r"D:\UAVIkA\models")

BATCH_SIZE = 16
EPOCHS = 20
IMAGE_SIZE = 160
LEARNING_RATE = 0.001

NUM_WORKERS = 0
SEED = 42

CLASSES = ["bird", "drone", "helicopter", "other"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("ANTI-UAV CNN TRAINING")
print("=" * 50)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# IMAGE TRANSFORMS
# ==============================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# LOAD DATASETS
# ==============================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

print("\nClass mapping:", train_dataset.class_to_idx)

assert train_dataset.classes == CLASSES, (
    f"Unexpected classes: {train_dataset.classes}"
)

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


# ==============================
# CLASS WEIGHTS
# ==============================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=len(CLASSES)
)

class_weights = len(train_dataset) / (
    len(CLASSES) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nTraining class counts:", class_counts)
print("Class weights:", class_weights.cpu().numpy())


# ==============================
# MODEL
# ==============================

# Use a pretrained ResNet18 as the CNN backbone.
# This reduces the amount of training needed from scratch.

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==============================
# TRAINING AND VALIDATION
# ==============================

best_val_accuracy = 0.0
best_model_path = OUTPUT_DIR / "best_anti_uav_cnn.pth"

for epoch in range(EPOCHS):

    # -------- Training --------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # -------- Validation --------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_accuracy)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the best validation checkpoint.

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "image_size": IMAGE_SIZE,
                "architecture": "resnet18",
                "best_val_accuracy": best_val_accuracy
            },
            best_model_path
        )

        print("Saved new best model:", best_model_path)


## ==============================
# TEST EVALUATION
# ==============================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Testing"):

        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


# Calculate confusion matrix without scikit-learn

num_classes = len(CLASSES)

cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for actual, predicted in zip(all_labels, all_predictions):
    cm[actual][predicted] += 1

print("\nConfusion Matrix")
print("Rows = actual class, Columns = predicted class")
print("Class order:", CLASSES)

for row in cm:
    print(row)


# Calculate per-class precision, recall, and F1-score

print("\nClassification Report")

for class_index, class_name in enumerate(CLASSES):

    true_positive = cm[class_index][class_index]

    actual_count = sum(cm[class_index])
    predicted_count = sum(
        cm[row][class_index] for row in range(num_classes)
    )

    precision = (
        true_positive / predicted_count
        if predicted_count > 0 else 0.0
    )

    recall = (
        true_positive / actual_count
        if actual_count > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    print(
        f"{class_name:12s} "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

test_accuracy = (
    sum(actual == predicted for actual, predicted in zip(
        all_labels, all_predictions
    )) / len(all_labels)
    if all_labels else 0.0
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

print("\nTraining completed.")
print("Model saved at:", best_model_path)