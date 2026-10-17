import os
import argparse
import time
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils import UAVYoloDataset, yolo_collate_fn, ComputeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dilated-UAVNet from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu")
    parser.add_argument("--val-limit", type=int, default=250, help="Max val batches per epoch (default 250 = 1000 images, set 0 for full validation)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def train():
    args = parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["learning_rate"]
    device_str = args.device or cfg["train"]["device"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DILATED-UAVNET: FROM-SCRATCH TRAINING")
    print("=" * 60)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"Input Resolution: {cfg['model']['input_size']}x{cfg['model']['input_size']}")
    print(f"Dataset: {cfg['data']['dataset_dir']}")
    print("=" * 60)

    # Output directory
    save_dir = Path(__file__).parent / cfg["train"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_weight_path = save_dir / cfg["train"]["checkpoint_name"]
    last_weight_path = save_dir / "last_uav_dilated.pth"

    # 1. Build Model from Scratch
    print("\n[1/4] Initializing model architecture (100% from scratch)...")
    model = build_model(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f}M (Optimized for RTX 2050)")

    # 2. Setup Datasets & DataLoaders
    print("\n[2/4] Initializing datasets and data loaders...")
    base_data = Path(cfg["data"]["dataset_dir"])
    train_dataset = UAVYoloDataset(
        img_dir=base_data / cfg["data"]["train_images"],
        label_dir=base_data / cfg["data"]["train_labels"],
        img_size=cfg["model"]["input_size"],
        is_train=True
    )
    val_dataset = UAVYoloDataset(
        img_dir=base_data / cfg["data"]["val_images"],
        label_dir=base_data / cfg["data"]["val_labels"],
        img_size=cfg["model"]["input_size"],
        is_train=False
    )

    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=yolo_collate_fn,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=yolo_collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # 3. Setup Optimizer, Scheduler & Mixed Precision
    print("\n[3/4] Initializing optimizer, scheduler, and loss...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg["train"].get("min_lr", 1e-5)
    )

    compute_loss = ComputeLoss(
        model=model,
        num_classes=cfg["data"]["num_classes"]
    )

    use_amp = cfg["train"]["amp"] and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("Mixed Precision (FP16 AMP) enabled for high-speed training.")

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk["epoch"] + 1
        best_val_loss = chk.get("best_val_loss", float("inf"))

    # 4. Training Loop
    print("\n[4/4] Starting training loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0
        total_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]")
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                predictions = model(images)
                loss, loss_items = compute_loss(predictions, targets)

            scaler.scale(loss).backward()
            # Gradient clipping to prevent gradient explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_box_loss += loss_items["loss_box"]
            epoch_obj_loss += loss_items["loss_obj"]
            epoch_cls_loss += loss_items.get("loss_cls", 0.0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "box": f"{loss_items['loss_box']:.3f}",
                "obj": f"{loss_items['loss_obj']:.3f}",
                "cls": f"{loss_items.get('loss_cls', 0.0):.3f}",
                "pos": loss_items["positives"]
            })

        avg_train_loss = epoch_loss / total_batches
        avg_train_box = epoch_box_loss / total_batches
        avg_train_obj = epoch_obj_loss / total_batches
        avg_train_cls = epoch_cls_loss / total_batches

        # Learning Rate Schedule: Linear Warmup (epochs 1-3) then Cosine Annealing
        warmup_epochs = cfg["train"].get("warmup_epochs", 1)
        if epoch < warmup_epochs:
            warmup_lr = lr * float(epoch + 1) / float(max(1, warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_limit = args.val_limit

        with torch.no_grad():
            for b_idx, (images, targets, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Val]", leave=False)):
                if val_limit > 0 and b_idx >= val_limit:
                    break
                images = images.to(device)
                targets = targets.to(device)

                # return_raw=True: loss needs the raw logits dict format even
                # though the model is in eval mode (BN uses running stats)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    predictions = model(images, return_raw=True)
                    loss, _ = compute_loss(predictions, targets)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / max(1, val_count)

        print(f"\n--- Epoch {epoch+1:02d}/{epochs:02d} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} (Box: {avg_train_box:.4f}, Obj: {avg_train_obj:.4f}, Cls: {avg_train_cls:.4f})")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save last checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "config": cfg
        }
        torch.save(checkpoint, last_weight_path)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_weight_path)
            print(f"[BEST] Saved new best checkpoint to: {best_weight_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best model saved at: {best_weight_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
