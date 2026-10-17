"""Hash-based duplicate report and safe quarantine helper for this project.

Default mode is a dry run. Only byte-identical files outside protected data,
model, configuration, cache, and generated-output locations are eligible for
quarantine. No operation permanently deletes files.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "duplicate_candidates.json"
SKIP_DIR_NAMES = {
    ".git", ".uv-cache", "__pycache__", ".duplicate_quarantine",
    ".venv", "venv", "env", "venvs", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
PROTECTED_PARTS = {
    ".git", ".uv-cache", "__pycache__", ".venv", "venv", "env", "venvs",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dataset", "processed", "models", "weights", "runs", "test_results",
}
PROTECTED_SUFFIXES = {".pt", ".pth", ".onnx", ".h5", ".keras", ".pkl"}
PROTECTED_NAMES = {"data.yaml", "config.yaml", "config.yml", "args.yaml", "agents.md"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(root):
    """Recompute a directory content signature before any quarantine move."""
    digests = {}
    for base, dirnames, filenames in os.walk(root):
        dirnames.sort()
        # Store the inventory now; digest bottom-up after walking.
    folders = [Path(base) for base, _, _ in os.walk(root)]
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        entries = []
        try:
            for child in folder.iterdir():
                if child.is_symlink():
                    continue
                if child.is_file():
                    size = child.stat().st_size
                    entries.append(("file", child.name, size, sha256(child)))
                elif child.is_dir() and child in folders:
                    entries.append(("dir", child.name, 0, digests.get(child, "")))
        except OSError:
            return ""
        payload = json.dumps(sorted(entries), separators=(",", ":")).encode("utf-8")
        digests[folder] = hashlib.sha256(payload).hexdigest()
    return digests.get(root, "")


def inside_root(path):
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def protected(path):
    rel = path.resolve().relative_to(ROOT.resolve())
    return (
        any(part.lower() in PROTECTED_PARTS for part in rel.parts)
        or path.suffix.lower() in PROTECTED_SUFFIXES
        or path.name.lower() in PROTECTED_NAMES
    )


def scan_project():
    hashes = defaultdict(list)
    sizes = {}
    directories = []
    all_files = []
    zero_byte_paths = []
    for base, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
        folder = Path(base)
        directories.append(folder)
        for name in filenames:
            path = folder / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
                digest = sha256(path)
            except (OSError, PermissionError) as exc:
                print(f"[SKIP] {path}: {exc}")
                continue
            rel = path.relative_to(ROOT).as_posix()
            if size == 0:
                # Empty YOLO annotation files are valid negative-image labels;
                # identical empty content does not make their paths duplicates.
                zero_byte_paths.append(rel)
            else:
                hashes[(size, digest)].append(rel)
            sizes[rel] = size
            all_files.append((folder, name, size, digest))

    duplicate_files = []
    for (size, digest), paths in sorted(hashes.items()):
        if len(paths) < 2:
            continue
        ordered = sorted(paths, key=lambda p: (protected(ROOT / Path(p)), p.lower()))
        keep = ordered[0]
        duplicate_paths = [p for p in ordered[1:] if p != keep]
        duplicate_files.append({
            "size_bytes": size,
            "sha256": digest,
            "canonical_keep": keep,
            "duplicate_paths": duplicate_paths,
            "proposed_action": "quarantine exact duplicate copies only; retain canonical path",
            "automatic_quarantine_eligible": [p for p in duplicate_paths if not protected(ROOT / Path(p))],
            "protected_paths": [p for p in ordered if protected(ROOT / Path(p))],
        })

    # Directory signatures include each immediate child's name and content
    # signature, so folders with identical relative layouts and file bytes match.
    children = defaultdict(list)
    folder_metrics = {folder: [0, 0] for folder in directories}  # recursive file count, bytes
    for folder, name, size, digest in all_files:
        children[folder].append(("file", name, size, digest))
        folder_metrics[folder][0] += 1
        folder_metrics[folder][1] += size
    directory_set = set(directories)
    for folder in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        try:
            child_dirs = (child for child in folder.iterdir() if child.is_dir() and child in directory_set)
            for child in child_dirs:
                children[folder].append(("dir", child.name, 0, folder_digests.get(child, "")))
                folder_metrics[folder][0] += folder_metrics.get(child, [0, 0])[0]
                folder_metrics[folder][1] += folder_metrics.get(child, [0, 0])[1]
        except OSError:
            pass
        payload = json.dumps(sorted(children[folder]), separators=(",", ":")).encode("utf-8")
        folder_digests[folder] = hashlib.sha256(payload).hexdigest()

    by_digest = defaultdict(list)
    for folder in directories:
        if folder == ROOT:
            continue
        by_digest[folder_digests.get(folder, "")].append(folder)
    duplicate_dirs = []
    for digest, folders in by_digest.items():
        if len(folders) < 2:
            continue
        # Empty directories share a signature but are not useful duplicate candidates.
        if not any(children[f] for f in folders):
            continue
        rels = sorted(p.relative_to(ROOT).as_posix() for p in folders)
        group = {
            "content_sha256": digest,
            "paths": rels,
            "file_count": folder_metrics.get(folders[0], [0, 0])[0],
            "total_size_bytes": folder_metrics.get(folders[0], [0, 0])[1],
            "proposed_action": "review contents and references manually; folders are never auto-quarantined",
            "automatic_quarantine_eligible": False,
        }
        if all(p.startswith("processed/five_class_classification/") or p.startswith("processed/five_class_grouped/") for p in rels):
            group["canonical_keep_for_training"] = next((p for p in rels if p.startswith("processed/five_class_grouped/")), None)
            group["related_source"] = next((p for p in rels if p.startswith("processed/five_class_classification/")), None)
            group["proposed_action"] = (
                "Retain both: five_class_classification is the source used by "
                "scripts/prepare_grouped_mosquito_dataset.py; five_class_grouped is the active five-class training data. "
                "No removal recommended."
            )
        elif all(p.startswith("test_results/yolonet_test_") for p in rels):
            group["canonical_keep"] = rels[0]
            group["review_candidates"] = rels[1:]
            group["folder_quarantine_eligible"] = rels[1:]
            group["proposed_action"] = (
                "These generated result folders have identical contents. Keep the earliest path shown; the other "
                "folders can be quarantined after confirming no local notes or workflow refer to them. "
                "The script does not move folders."
            )
        duplicate_dirs.append(group)

    return {
        "project_root": str(ROOT),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scan_notes": [
            "SHA-256 and size were checked for every regular file under the project root.",
            "Git metadata, virtual environments, common caches, and the quarantine directory are excluded from scanning.",
            "Dataset splits, model files, configuration, runs, and test outputs are reported but protected from automatic quarantine.",
            "Directory matches require identical child names, sizes, and file hashes; no folder is eligible for automatic quarantine.",
        ],
        "scanned_file_count": len(all_files),
        "zero_byte_files_excluded_as_duplicate_candidates": len(zero_byte_paths),
        "duplicate_file_groups": duplicate_files,
        "duplicate_directory_groups": duplicate_dirs,
    }


folder_digests = {}


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def quarantine(report, dry_run):
    targets = []
    for group in report["duplicate_file_groups"]:
        for rel in group["automatic_quarantine_eligible"]:
            targets.append((rel, group["canonical_keep"], group["sha256"], group["size_bytes"]))
    folder_targets = []
    for group in report["duplicate_directory_groups"]:
        for rel in group.get("folder_quarantine_eligible", []):
            folder_targets.append((rel, group["canonical_keep"], group["content_sha256"], group["total_size_bytes"]))
    for group in report["duplicate_directory_groups"]:
        print(
            f"DIRECTORY REVIEW ({group['file_count']} files, {group['total_size_bytes']} bytes, "
            f"content SHA256 {group['content_sha256']}):"
        )
        for rel in group["paths"]:
            print(f"  {rel}")
        print(f"  Action: {group['proposed_action']}")
    for rel, keep, digest, size in folder_targets:
        print(f"  QUARANTINE FOLDER {rel} ({size} bytes, content SHA256 {digest}) [keep {keep}]")
    if not targets and not folder_targets:
        print("No unprotected exact duplicate files are eligible for quarantine.")
        return
    print(f"Exact duplicate file copies: {len(targets)}; reviewed duplicate folders: {len(folder_targets)}")
    for rel, keep, digest, size in targets:
        print(f"  QUARANTINE {rel} ({size} bytes, SHA256 {digest}) [keep {keep}]")
    if dry_run:
        print("Dry run only. Re-run with --delete to review and move listed files/folders to project quarantine.")
        return

    response = input(f"Type QUARANTINE to move these {len(targets)} files and {len(folder_targets)} folders (no permanent deletion): ")
    if response != "QUARANTINE":
        print("Cancelled; no files moved.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_root = ROOT / ".duplicate_quarantine" / stamp
    for rel, keep, expected_digest, expected_size in folder_targets:
        source_path = ROOT / Path(rel)
        source = source_path.resolve()
        canonical = (ROOT / Path(keep)).resolve()
        if not inside_root(source) or not inside_root(canonical) or source == canonical:
            raise RuntimeError(f"Refusing unsafe folder path: {source}")
        if source_path.is_symlink() or not source.is_dir():
            print(f"[SKIP] Folder changed or missing: {rel}")
            continue
        if directory_sha256(source) != expected_digest:
            print(f"[SKIP] Folder content changed since report: {rel}")
            continue
        destination = quarantine_root / Path(rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), str(destination))
        print(f"[MOVED FOLDER] {rel} -> {destination.relative_to(ROOT)}")
    for rel, keep, expected_hash, expected_size in targets:
        source_path = ROOT / Path(rel)
        source = source_path.resolve()
        canonical = (ROOT / Path(keep)).resolve()
        if source_path.is_symlink() or not inside_root(source) or not inside_root(canonical):
            raise RuntimeError(f"Refusing path outside project root: {source}")
        if source == canonical or not source.is_file():
            print(f"[SKIP] Source changed or missing: {rel}")
            continue
        if source.stat().st_size != expected_size or sha256(source) != expected_hash:
            print(f"[SKIP] Content changed since report: {rel}")
            continue
        destination = quarantine_root / Path(rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), str(destination))
        print(f"[MOVED] {rel} -> {destination.relative_to(ROOT)}")
    print(f"Quarantined files are under: {quarantine_root}")


def main():
    parser = argparse.ArgumentParser(description="Report exact duplicates; quarantine is opt-in and dry-run is default.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report output path")
    parser.add_argument("--delete", action="store_true", help="Prompt, then move eligible exact duplicates into project quarantine")
    args = parser.parse_args()
    if not inside_root(args.report):
        raise ValueError("Report output must stay inside the project root.")
    report = scan_project()
    write_report(report, args.report)
    print(f"Report: {args.report}")
    print(f"Scanned files: {report['scanned_file_count']}")
    print(f"Duplicate file groups: {len(report['duplicate_file_groups'])}")
    print(f"Duplicate directory groups: {len(report['duplicate_directory_groups'])}")
    quarantine(report, dry_run=not args.delete)


if __name__ == "__main__":
    main()
