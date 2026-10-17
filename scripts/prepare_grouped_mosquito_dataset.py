"""Build a copy of the five-class dataset with mosquito specimens split by ID.

The original prepared dataset is left untouched. Mosquito filenames are of the
form aegypti<number><letter>; all views with the same number stay in one split.
"""

import json
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "processed" / "five_class_classification"
MOSQUITO_SOURCE = ROOT / "dataset" / "raw" / "other" / "mosq"
OUTPUT_DATA = ROOT / "processed" / "five_class_grouped"
SEED = 42
CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]


def normalize_copy(src: Path, dst: Path, max_dim: int = 640):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
        im.save(dst, quality=92)


def main():
    if OUTPUT_DATA.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT_DATA}")
    groups = defaultdict(list)
    seen_hashes = set()
    duplicate_files = []
    for path in sorted(MOSQUITO_SOURCE.glob("*.jpg")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            duplicate_files.append(path.name)
            continue
        seen_hashes.add(digest)
        normalized_stem = re.sub(r"\s+-\s+copy$", "", path.stem, flags=re.IGNORECASE)
        match = re.fullmatch(r"([a-z]+\d+)[a-z]+", normalized_stem, re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot infer mosquito specimen ID from filename: {path.name}")
        groups[match.group(1).lower()].append(path)
    if not groups:
        raise RuntimeError(f"No mosquito source images found under {MOSQUITO_SOURCE}")

    shuffled_groups = sorted(groups.items())
    random.Random(SEED).shuffle(shuffled_groups)
    total = sum(len(paths) for _, paths in shuffled_groups)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    assigned = {"train": [], "val": [], "test": []}
    counts = {split: 0 for split in assigned}
    active = "train"
    for group_id, paths in shuffled_groups:
        assigned[active].append((group_id, paths))
        counts[active] += len(paths)
        if active == "train" and counts[active] >= targets[active]:
            active = "val"
        elif active == "val" and counts[active] >= targets[active]:
            active = "test"

    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (OUTPUT_DATA / split / class_name).mkdir(parents=True, exist_ok=True)
            if class_name == "mosquito":
                for _, paths in assigned[split]:
                    for src in paths:
                        normalize_copy(src, OUTPUT_DATA / split / class_name / f"{src.stem}.jpg")
            else:
                source_dir = SOURCE_DATA / split / class_name
                for src in source_dir.iterdir():
                    if src.is_file():
                        shutil.copy2(src, OUTPUT_DATA / split / class_name / src.name)

    group_map = {split: sorted(group_id for group_id, _ in items) for split, items in assigned.items()}
    manifest = {
        "seed": SEED,
        "grouping_rule": "lowercase filename stem without final letter, e.g. aegypti100a -> aegypti100",
        "exact_duplicate_files_excluded": duplicate_files,
        "counts": counts,
        "group_counts": {split: len(ids) for split, ids in group_map.items()},
        "groups": group_map,
        "class_counts": {
            split: {
                class_name: len(list((OUTPUT_DATA / split / class_name).glob("*")))
                for class_name in CLASSES
            }
            for split in ("train", "val", "test")
        },
    }
    (OUTPUT_DATA / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "group_counts", "class_counts")}, indent=2))
    print(f"Dataset created at {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
