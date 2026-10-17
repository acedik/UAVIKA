import os
import glob
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

class MultiClassAntiUAVSystem:
    def __init__(self, weights_path: str, conf_thresh: float = 0.35, device: str = "0"):
        """
        Tactical Air Defense Surveillance Engine with Threat Color Coding:
        - Drone: RED (Critical Threat)
        - Helicopter: ORANGE (Caution / Elevated)
        - Birds / Flying Objects / Others: GREEN (Neutral / Biological)
        """
        print(f"[SYSTEM INITIALIZING] Loading weights from: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        self.device = device
        
        # Color definitions (OpenCV uses BGR format)
        self.COLOR_DRONE = (0, 0, 255)       # Red
        self.COLOR_HELI  = (0, 140, 255)     # Tactical Orange
        self.COLOR_OTHER = (0, 255, 0)       # Green (Birds/Insects/Others)
        self.COLOR_BG    = (15, 15, 15)      # Dark HUD Background
        self.COLOR_TEXT  = (255, 255, 255)   # Crisp White
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def classify_threat(self, class_name: str):
        """
        Categorizes detected labels into tactical threat levels and assigned colors.
        """
        name_lower = class_name.lower().strip()

        # 1. High Priority: Drone / UAV
        if any(keyword in name_lower for keyword in ["drone", "uav", "quadcopter", "multirotor"]):
            return self.COLOR_DRONE, "DRONE [CRITICAL THREAT]", 3

        # 2. Medium Priority: Helicopter / Rotorcraft
        elif any(keyword in name_lower for keyword in ["helicopter", "helecpoerr", "heli", "chopper"]):
            return self.COLOR_HELI, "HELICOPTER [AIR TRAFFIC]", 2

        # 3. Low Priority: Birds, Flies, Insects, or Other Unknown Entities
        else:
            return self.COLOR_OTHER, f"{class_name.upper()} [NEUTRAL]", 1

    def _draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=16, thickness=2):
        """Renders tactical corner targeting reticles."""
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_tactical_hud(self, frame, detections, fps=0.0):
        h, w = frame.shape[:2]
        highest_threat_rank = 0
        status_color = self.COLOR_OTHER

        # 1. Target Highlights & Bounding Elements
        for idx, (x1, y1, x2, y2, conf, cls_name, color, label_str, threat_rank) in enumerate(detections, start=1):
            if threat_rank > highest_threat_rank:
                highest_threat_rank = threat_rank
                status_color = color

            # Glowing Bounding Box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=4)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

            # Tactical Reticle & Center Target Dot
            self._draw_corner_brackets(frame, x1, y1, x2, y2, color=color, length=16, thickness=2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, color, -1)

            # Target Lock Tag (Positioned below target with boundary protection)
            tag_text = f"#{idx} {label_str}: {conf * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(tag_text, self.font, 0.42, 1)

            tag_y1 = y2 + 4
            tag_y2 = y2 + th + baseline + 6
            if tag_y2 > h - 42:  # Flip inside box if near bottom edge
                tag_y1 = y1 - th - baseline - 6
                tag_y2 = y1 - 4

            # Dark tag background matching target color outline
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, 1)
            cv2.putText(frame, tag_text, (x1 + 3, tag_y2 - baseline - 2),
                        self.font, 0.42, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # 2. Top System Telemetry Bar
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 35), self.COLOR_BG, -1)
        cv2.addWeighted(top_bar, 0.78, frame, 0.22, 0, frame)

        if highest_threat_rank == 3:
            sys_status = "THREAT ENGAGED - HOSTILE DRONE DETECTED"
        elif highest_threat_rank == 2:
            sys_status = "AIRSPACE CONFLICT - HELICOPTER IDENTIFIED"
        elif highest_threat_rank == 1:
            sys_status = "OBJECT MONITORED - BIOLOGICAL / LOW RISK"
        else:
            sys_status = "AIRSPACE SECURE - SCANNING ACTIVE"

        telemetry = f"SYS: ONLINE | STATUS: {sys_status} | FPS: {fps:.1f} | TARGETS: {len(detections)}"
        cv2.putText(frame, telemetry, (15, 23), self.font, 0.45, status_color, 1, cv2.LINE_AA)

        # 3. Bottom Target Telemetry Banner
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 35), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(bot_bar, 0.78, frame, 0.22, 0, frame)

        if detections:
            primary = detections[0]  # First detected target coords
            bot_text = f"[ENGAGED TARGETS: {len(detections)}] PRIORITY: {primary[7]} at ({primary[0]}, {primary[1]}) - ({primary[2]}, {primary[3]})"
            cv2.putText(frame, bot_text, (15, h - 12), self.font, 0.42, primary[6], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[ALL CLEAR] NO AIRBORNE TARGETS IN DESIGNATED SECTOR", (15, h - 12),
                        self.font, 0.42, self.COLOR_OTHER, 1, cv2.LINE_AA)

        return frame

    def process_frame(self, frame):
        """Runs model inference and resolves color profiles per class."""
        t_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=0.45,
            agnostic_nms=True,
            max_det=30,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            raw_name = self.model.names.get(cls_id, "unknown")

            # Determine color & threat ranking
            color, label_str, threat_rank = self.classify_threat(raw_name)
            detections.append((*coords, conf, raw_name, color, label_str, threat_rank))

        # Sort detections so highest threat (Drone=3) draws on top and reports first
        detections.sort(key=lambda d: d[8], reverse=True)

        fps = 1.0 / max(time.perf_counter() - t_start, 1e-5)
        annotated = self.draw_tactical_hud(frame, detections, fps=fps)
        return annotated, detections

    def run_live(self, source=0):
        """Live feed inference (Webcam, RTSP stream, or Video file)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video feed: {source}")
            return

        print("\n[ACTIVE] Press 'Q' to quit stream.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = self.process_frame(frame)
            cv2.imshow("Dedicated Anti-UAV System", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_batch_images(self, input_dir: str, output_dir: str):
        """Batch evaluates folder of test images."""
        os.makedirs(output_dir, exist_ok=True)
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"[BATCH] Processing {len(img_paths)} test frames...")
        for path in img_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            annotated, _ = self.process_frame(frame)
            save_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(save_path, annotated)
            print(f"Saved: {save_path}")


# ================= Execution =================
if __name__ == "__main__":
    # Never silently substitute COCO weights for the project's trained model.
    MODEL_WEIGHTS = Path(__file__).resolve().parents[1] / "runs" / "anti_uav" / "yolo_drone_s_v1" / "weights" / "best.pt"
    if not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Trained four-class YOLO weights are missing: {MODEL_WEIGHTS}. "
            "Use `python scripts/live.py` for the available four-class classifier pipeline."
        )

    app = MultiClassAntiUAVSystem(weights_path=str(MODEL_WEIGHTS), conf_thresh=0.20, device="0")

    # Run on live webcam / drone camera feed
    app.run_live(source=0)

    # Or batch test your images folder:
    # app.run_batch_images(
    #     input_dir=r"D:\UAVIkA\my_test_images",
    #     output_dir=r"D:\UAVIkA\runs\anti_uav\yolo_drone_s_v1\custom_detections"
    # )
