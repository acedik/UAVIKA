from pathlib import Path
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

# Search recursively inside this folder for bird videos
VIDEO_DIR = Path(r"D:\UAVIkA\dataset\raw\bird")

# Extracted frames will be saved here
OUTPUT_DIR = Path(r"D:\UAVIkA\processed\bird_frames")

# Extract one frame every 2 seconds
FRAME_INTERVAL_SECONDS = 2

# Supported video formats
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".m4v",
}


# ============================================================
# FIND VIDEOS
# ============================================================

def find_videos(folder):
    """Recursively find supported video files."""
    return [
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    ]


# ============================================================
# EXTRACT FRAMES
# ============================================================

def extract_frames(video_path, output_folder):
    """Extract one frame at the configured time interval."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[SKIPPED] Could not open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"[SKIPPED] Invalid FPS: {video_path}")
        cap.release()
        return 0

    duration = total_frames / fps if total_frames > 0 else 0

    # Calculate how many frames to skip between saved frames
    frame_interval = max(1, int(fps * FRAME_INTERVAL_SECONDS))

    output_folder.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    saved_count = 0

    print(f"\nProcessing: {video_path.name}")
    print(f"FPS: {fps:.2f} | Duration: {duration:.1f} seconds")

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            # Include the video name to avoid filename collisions
            output_name = (
                f"{video_path.stem}_frame_{frame_number:07d}.jpg"
            )

            output_path = output_folder / output_name

            saved = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if saved:
                saved_count += 1

        frame_number += 1

    cap.release()

    print(f"Frames read: {frame_number}")
    print(f"Frames saved: {saved_count}")

    return saved_count


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BIRD VIDEO FRAME EXTRACTION")
    print("=" * 60)

    if not VIDEO_DIR.exists():
        print(f"[ERROR] Dataset folder does not exist:\n{VIDEO_DIR}")
        return

    videos = find_videos(VIDEO_DIR)

    if not videos:
        print("[ERROR] No supported video files found.")
        print(f"Folder searched recursively:\n{VIDEO_DIR}")
        print("\nSupported extensions:")
        print(", ".join(sorted(VIDEO_EXTENSIONS)))
        print("\nIf your videos are inside a ZIP file, extract them first.")
        return

    print(f"Videos found: {len(videos)}")
    print(f"Output folder: {OUTPUT_DIR}")

    total_saved = 0

    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}]")

        # Put each video's extracted frames in its own folder
        video_output_folder = OUTPUT_DIR / video_path.stem

        total_saved += extract_frames(
            video_path,
            video_output_folder,
        )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Videos processed: {len(videos)}")
    print(f"Total frames saved: {total_saved}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()