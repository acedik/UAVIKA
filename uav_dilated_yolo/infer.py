import os
import argparse
import time
from pathlib import Path
import glob
import cv2
import numpy as np
import yaml
import torch

from models import build_model
from utils import non_max_suppression, remove_inner_boxes, TacticalHUDVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Dilated-UAVNet Inference & Tactical HUD")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).parent / "weights" / "best_uav_dilated.pth"),
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"D:\UAVIkA\my_test_images",
        help="Path to image, video, directory, or camera index (0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "runs" / "detections"),
        help="Directory to save annotated detections"
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--view", action="store_true", help="Display live OpenCV window")
    return parser.parse_args()


class UAVInferEngine:
    def __init__(self, weights_path, config_path, conf_thresh=0.30, iou_thresh=0.45, device="cuda:0"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.input_size = self.cfg["model"]["input_size"]
        self.model = build_model(self.cfg).to(self.device)

        if os.path.exists(weights_path):
            print(f"[SYSTEM] Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        else:
            print(f"[WARNING] Weights file '{weights_path}' not found! Running with initialized weights.")

        self.model.eval()
        self.visualizer = TacticalHUDVisualizer(
            class_names=self.cfg["data"].get("class_names", {0: "drone"})
        )

    def preprocess(self, img_bgr):
        """Preprocesses arbitrary image frame to model input tensor."""
        orig_h, orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, orig_w, orig_h

    def detect_frame(self, frame_bgr):
        t0 = time.perf_counter()
        tensor, orig_w, orig_h = self.preprocess(frame_bgr)

        with torch.no_grad():
            preds = self.model(tensor)  # (1, N_proposals, 6)
            detections = non_max_suppression(
                preds,
                conf_threshold=self.conf_thresh,
                iou_threshold=self.iou_thresh
            )[0]

        # Remove inner/contained bounding boxes (e.g., helicopter inside bird)
        if detections.shape[0] > 0:
            detections = remove_inner_boxes(detections, iou_threshold=0.5)

        scaled_detections = []
        if detections.shape[0] > 0:
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size

            for det in detections.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = det
                sx1 = max(0, min(orig_w - 1, x1 * scale_x))
                sy1 = max(0, min(orig_h - 1, y1 * scale_y))
                sx2 = max(0, min(orig_w - 1, x2 * scale_x))
                sy2 = max(0, min(orig_h - 1, y2 * scale_y))
                scaled_detections.append([sx1, sy1, sx2, sy2, conf, cls_id])

        fps = 1.0 / max(time.perf_counter() - t0, 1e-5)
        annotated = self.visualizer.draw_hud(frame_bgr.copy(), scaled_detections, fps=fps)
        return annotated, scaled_detections


def run():
    args = parse_args()
    engine = UAVInferEngine(
        weights_path=args.weights,
        config_path=args.config,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Camera stream
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        print(f"[ACTIVE] Live video stream (source {args.source}). Press 'Q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = engine.detect_frame(frame)
            cv2.imshow("Dilated-UAVNet Tactical HUD", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # 2. Video file
    elif Path(args.source).is_file() and Path(args.source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_file = str(out_dir / f"annotated_{Path(args.source).name}")
        writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        print(f"[VIDEO] Processing: {args.source} -> {save_file}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = engine.detect_frame(frame)
            writer.write(annotated)
            if args.view:
                cv2.imshow("Dilated-UAVNet Tactical HUD", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[COMPLETE] Annotated video saved to: {save_file}")

    # 3. Directory of images or single image
    else:
        src = Path(args.source)
        if src.is_file():
            img_paths = [src]
        elif src.is_dir():
            img_paths = sorted([
                p for p in src.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ])
        else:
            raise FileNotFoundError(f"Source not found: {args.source}")

        print(f"[BATCH] Processing {len(img_paths)} images...")
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            annotated, dets = engine.detect_frame(img)
            save_path = out_dir / p.name
            cv2.imwrite(str(save_path), annotated)
            print(f"[{len(dets)} Detections] Saved: {save_path}")

            if args.view:
                cv2.imshow("Dilated-UAVNet Tactical HUD", annotated)
                # Auto-advance after 1.5s; press 'q' to quit, any other key to skip ahead
                key = cv2.waitKey(1500) & 0xFF
                if key == ord('q'):
                    break
        if args.view:
            cv2.destroyAllWindows()

        print(f"\n[COMPLETE] All detections saved to: {out_dir}")


if __name__ == "__main__":
    run()
