"""Live four-class Anti-UAV detection using the project's existing models."""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detect_and_classify import MultiTargetAirDefenseSystem


DETECTOR_PATH = ROOT / "yolo11s.pt"
CLASSIFIER_PATH = ROOT / "models" / "best_uav_yolonet_grouped_retrain5ep.pth"
EXPECTED_CLASSES = {"drone", "bird", "helicopter", "other", "mosquito"}


def parse_args():
    parser = argparse.ArgumentParser(description="Live Multi-Class Anti-UAV webcam detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--detector", type=str, default=str(DETECTOR_PATH), help="Locator weights ('yolo11s.pt', 'dilated', or 'uav_dilated_yolo/weights/best_uav_dilated.pth')")
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER_PATH, help="Calibrated UAV-YOLONet checkpoint")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO proposal confidence threshold (default: 0.10)")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Minimum confirmed detection confidence threshold (default: 0.30)")
    parser.add_argument("--iou", type=float, default=0.45, help="Class-agnostic NMS IoU threshold")
    parser.add_argument("--drone-threshold", type=float, default=0.55, help="Minimum confidence threshold for drone threat (default: 0.55)")
    parser.add_argument("--multi-zoom", action="store_true", default=True, help="Enable 3-stage zoom inspection")
    parser.add_argument("--no-multi-zoom", dest="multi_zoom", action="store_false", help="Disable 3-stage zoom inspection")
    parser.add_argument("--zoom-threshold", type=float, default=0.70, help="Confidence threshold across zoom stages to confirm a drone lock (default: 0.70)")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum candidate detections per frame (default: 100)")
    parser.add_argument(
        "--enable-saliency",
        action="store_true",
        default=False,
        help="Enable edge/contour saliency proposals (disabled by default to prevent indoor false alarms)"
    )
    parser.add_argument(
        "--show-other",
        action="store_true",
        default=False,
        help="Display bounding boxes for neutral background clutter ('other')"
    )
    parser.add_argument(
        "--no-tta",
        dest="enable_tta",
        action="store_false",
        default=True,
        help="Disable multi-angle Test-Time Augmentation (TTA)"
    )
    return parser.parse_args()


def open_camera(index):
    """Open one camera handle, preferring DirectShow on Windows."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def main():
    args = parse_args()
    det_str = str(args.detector).lower()
    if det_str != "dilated" and not Path(args.detector).is_file():
        raise FileNotFoundError(f"Detector weights not found: {args.detector}")
    if not args.classifier.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier}")

    system = MultiTargetAirDefenseSystem(
        detector_path=str(args.detector),
        classifier_path=str(args.classifier),
        conf_thresh=args.conf,
        min_conf=args.min_conf,
        iou_thresh=args.iou,
        drone_threshold=args.drone_threshold,
        multi_zoom=args.multi_zoom,
        zoom_threshold=args.zoom_threshold,
        max_det=args.max_det,
        enable_saliency=args.enable_saliency,
        show_other=args.show_other,
        enable_tta=args.enable_tta,
    )
    classes = set(system.classes)
    if not {"drone", "bird", "helicopter"}.issubset(classes):
        raise ValueError(f"Expected model classes to include ['drone', 'bird', 'helicopter'], got {system.classes}")

    cap = open_camera(args.camera)
    if cap is None:
        print(f"[ERROR] Could not open webcam index {args.camera}.")
        if args.camera != 0:
            print("[INFO] Checking for default camera at index 0...")
            cap = open_camera(0)
            if cap is not None:
                args.camera = 0
                print("[SUCCESS] Successfully connected to default camera index 0.")
        if cap is None:
            print("[ERROR] No working webcam found. Please verify camera permissions and connection.")
            return 1

    window = "Anti-UAV Live Detection"
    print(f"[ACTIVE] Camera {args.camera}; classes: {', '.join(system.classes)}")
    print("Press Q in the video window to exit.")
    failed_reads = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 10:
                    print("[ERROR] Could not read 10 consecutive webcam frames; stopping.")
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            failed_reads = 0

            # One frame, one process_frame call, one set of boxes drawn by our HUD.
            annotated, _ = system.process_frame(frame)
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("[CLOSED] Webcam released and windows closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
