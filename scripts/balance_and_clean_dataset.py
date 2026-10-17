import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

RAW_DIR = Path(r"D:\UAVIKA\dataset\raw")
OUTPUT_DIR = Path(r"D:\UAVIKA\processed\balanced_classification")

CLASSES = ["bird", "drone", "helicopter", "other"]
TARGET_TRAIN_PER_CLASS = 3000
TARGET_VAL_PER_CLASS = 300
TARGET_TEST_PER_CLASS = 300
MAX_IMAGE_DIM = 640  # Standard vision/YOLO resolution for optimal speed & quality

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def is_valid_image(path):
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def open_and_normalize(path):
    """Opens image, converts to RGB, and caps dimensions to MAX_IMAGE_DIM."""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(w, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return im

def augment_image(im):
    """Applies realistic optical and geometric augmentations."""
    # 1. Random horizontal flip
    if random.random() > 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 2. Random vertical flip
    if random.random() > 0.6:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        
    # 3. Random rotation (-25 to +25 deg)
    angle = random.uniform(-25, 25)
    im = im.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    # 4. Color / Brightness / Contrast jitter
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        im = ImageEnhance.Brightness(im).enhance(factor)
        
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        im = ImageEnhance.Contrast(im).enhance(factor)

    # 5. Mild Sharpness / Blur
    if random.random() > 0.75:
        im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
    elif random.random() > 0.75:
        im = ImageEnhance.Sharpness(im).enhance(random.uniform(1.2, 1.6))
        
    return im

def main():
    print("=" * 60)
    print("OPTIMIZED DATASET CLEANING & BALANCING ENGINE")
    print("=" * 60)
    print(f"Target Train per class: {TARGET_TRAIN_PER_CLASS}")
    print(f"Target Val per class:   {TARGET_VAL_PER_CLASS}")
    print(f"Target Test per class:  {TARGET_TEST_PER_CLASS}")
    print(f"Max Dimension Capped:   {MAX_IMAGE_DIM}px")
    print(f"Output directory:       {OUTPUT_DIR}")

    # Prepare output folders
    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            dest = OUTPUT_DIR / split / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in dest.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    summary = {}

    for cls_name in CLASSES:
        print(f"\n[PROCESSING CLASS: {cls_name.upper()}]")
        cls_raw_dir = RAW_DIR / cls_name
        
        raw_candidates = [
            p for p in cls_raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        
        random.shuffle(raw_candidates)
        print(f"Found {len(raw_candidates)} candidate raw images.")
        
        # Verify integrity
        valid_images = []
        for p in tqdm(raw_candidates, desc=f"Verifying {cls_name}"):
            if is_valid_image(p):
                valid_images.append(p)
                
        total_valid = len(valid_images)
        val_count = min(TARGET_VAL_PER_CLASS, max(50, int(total_valid * 0.15)))
        test_count = min(TARGET_TEST_PER_CLASS, max(50, int(total_valid * 0.15)))
        
        val_pool = valid_images[:val_count]
        test_pool = valid_images[val_count:val_count + test_count]
        train_pool = valid_images[val_count + test_count:]
        
        print(f"Splits -> Train pool: {len(train_pool)}, Val: {len(val_pool)}, Test: {len(test_pool)}")
        
        # 1. Copy Validation Images (normalized to <= 640px)
        for idx, src in enumerate(tqdm(val_pool, desc=f"Saving {cls_name} val")):
            dest = OUTPUT_DIR / "val" / cls_name / f"{cls_name}_val_{idx:05d}.jpg"
            im = open_and_normalize(src)
            im.save(dest, quality=92)
            
        # 2. Copy Test Images (normalized to <= 640px)
        for idx, src in enumerate(tqdm(test_pool, desc=f"Saving {cls_name} test")):
            dest = OUTPUT_DIR / "test" / cls_name / f"{cls_name}_test_{idx:05d}.jpg"
            im = open_and_normalize(src)
            im.save(dest, quality=92)
            
        # 3. Process Train Images & Balance to Target
        train_dest_dir = OUTPUT_DIR / "train" / cls_name
        saved_train_count = 0
        
        if len(train_pool) >= TARGET_TRAIN_PER_CLASS:
            selected_train = train_pool[:TARGET_TRAIN_PER_CLASS]
            for idx, src in enumerate(tqdm(selected_train, desc=f"Saving {cls_name} train")):
                dest = train_dest_dir / f"{cls_name}_train_{idx:05d}.jpg"
                im = open_and_normalize(src)
                im.save(dest, quality=92)
                saved_train_count += 1
        else:
            print(f"Augmenting {cls_name} from {len(train_pool)} to {TARGET_TRAIN_PER_CLASS}...")
            # Pre-load normalized images into memory for high-speed augmentation
            print("Pre-loading base images for rapid augmentation...")
            cached_images = []
            for src in tqdm(train_pool, desc="Preloading"):
                try:
                    cached_images.append(open_and_normalize(src))
                except Exception:
                    pass
            
            # Save original normalized images first
            for idx, im in enumerate(cached_images):
                dest = train_dest_dir / f"{cls_name}_train_orig_{idx:05d}.jpg"
                im.save(dest, quality=92)
                saved_train_count += 1
                
            # Rapidly generate augmented variations from cached images
            needed_augment = TARGET_TRAIN_PER_CLASS - saved_train_count
            aug_idx = 0
            with tqdm(total=needed_augment, desc=f"Generating {cls_name} augs") as pbar:
                while saved_train_count < TARGET_TRAIN_PER_CLASS:
                    base_im = random.choice(cached_images)
                    aug_im = augment_image(base_im)
                    aug_dest = train_dest_dir / f"{cls_name}_train_aug_{aug_idx:05d}.jpg"
                    aug_im.save(aug_dest, quality=92)
                    saved_train_count += 1
                    aug_idx += 1
                    pbar.update(1)

        summary[cls_name] = {
            "train": saved_train_count,
            "val": len(val_pool),
            "test": len(test_pool)
        }

    print("\n" + "=" * 60)
    print("FINAL BALANCED DATASET REPORT")
    print("=" * 60)
    print(f"{'Class':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 50)
    tot_tr = tot_vl = tot_ts = 0
    for cls_name, s in summary.items():
        print(f"{cls_name:<15} | {s['train']:<8} | {s['val']:<8} | {s['test']:<8}")
        tot_tr += s['train']
        tot_vl += s['val']
        tot_ts += s['test']
    print("-" * 50)
    print(f"{'TOTAL':<15} | {tot_tr:<8} | {tot_vl:<8} | {tot_ts:<8}")
    print("=" * 60)

if __name__ == "__main__":
    main()
