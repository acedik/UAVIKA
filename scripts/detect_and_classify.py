import os
import sys
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
from ultralytics import YOLO

from models.uav_yolonet import UAV_YOLONet

# =========================================================
# CONFIGURATION
# =========================================================

CLASSES = ["bird", "drone", "helicopter", "mosquito", "other"]
IMAGE_SIZE = 160
MOSQUITO_FRAME_THRESHOLD = 0.85
MOSQUITO_CAND_THRESHOLD = 0.70

# Threat Color Palette (BGR)
COLOR_DRONE    = (0, 0, 255)       # Red - Hostile Drone Threat
COLOR_HELI     = (0, 140, 255)     # Orange - Air Traffic (Helicopter)
COLOR_MOSQUITO = (0, 215, 255)     # Amber/Gold - Biological Pest / Micro-Target
COLOR_BIRD     = (0, 255, 0)       # Green - Biological Neutral (Bird)
COLOR_OTHER    = (255, 200, 0)     # Cyan/Yellow - Other / Unknown

# Comprehensive exclusion list of non-aerial / terrestrial COCO classes.
# Rejects humans, indoor household items, furniture, and street fixtures from
# generating candidate aerial proposals in live webcam and room feeds.
NON_AERIAL_CLASSES = {
    # Humans, personal items, apparel
    "person", "backpack", "umbrella", "handbag", "tie", "suitcase",
    # Furniture, indoor fixtures, electronics
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "bench",
    # Tableware & food
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake",
    # Terrestrial animals
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Terrestrial infrastructure & heavy ground transport
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bus", "truck", "train"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Target Detection & Four-Class Air Defense Recognition")
    parser.add_argument(
        "--source",
        type=str,
        default=r"D:\UAVIkA\my_test_images",
        help="Path to image, video file, directory, or camera index (0)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=r"D:\UAVIkA\yolo11s.pt",
        help="Path to object locator weights (e.g. 'yolo11s.pt', 'dilated', or 'uav_dilated_yolo/weights/best_uav_dilated.pth')"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=r"D:\UAVIKA\models\best_uav_yolonet_grouped_retrain5ep.pth",
        help="Path to the calibrated UAV-YOLONet checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\UAVIkA\test_results",
        help="Directory to save annotated images and predictions.csv"
    )
    parser.add_argument("--conf", type=float, default=0.02, help="Candidate proposal confidence threshold (default: 0.02)")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold required to confirm and render any target detection (default: 0.25)"
    )
    parser.add_argument("--iou", type=float, default=0.45, help="Class-agnostic NMS IoU threshold")
    parser.add_argument(
        "--mosquito-frame-threshold", type=float, default=MOSQUITO_FRAME_THRESHOLD,
        help="Full-frame probability threshold to confirm mosquito presence (default: 0.85)"
    )
    parser.add_argument(
        "--mosquito-cand-threshold", type=float, default=MOSQUITO_CAND_THRESHOLD,
        help="Candidate crop probability threshold to confirm localized mosquito target (default: 0.70)"
    )
    parser.add_argument(
        "--drone-threshold",
        type=float,
        default=0.50,
        help="Minimum confidence threshold required to confirm a hostile drone detection (default: 0.50)"
    )
    parser.add_argument(
        "--heli-threshold",
        type=float,
        default=0.70,
        help="Minimum confidence threshold required to confirm a helicopter detection (default: 0.70)"
    )
    parser.add_argument(
        "--multi-zoom",
        action="store_true",
        default=True,
        help="Enable 3-stage progressive zoom verification"
    )
    parser.add_argument(
        "--no-multi-zoom",
        dest="multi_zoom",
        action="store_false",
        help="Disable 3-stage progressive zoom verification"
    )
    parser.add_argument(
        "--zoom-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold across zoom stages to confirm a drone lock (default: 0.70)"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum number of target detections per frame (default: 100)"
    )
    parser.add_argument("--view", action="store_true", help="Display live OpenCV window")
    parser.add_argument(
        "--enable-saliency",
        action="store_true",
        default=True,
        help="Enable edge/contour saliency proposals as fallback for close-up/challenging targets (default: True)"
    )
    parser.add_argument(
        "--no-saliency",
        dest="enable_saliency",
        action="store_false",
        help="Disable edge/contour saliency proposals fallback"
    )
    parser.add_argument(
        "--show-other",
        action="store_true",
        default=False,
        help="Display bounding boxes for neutral background clutter ('other')"
    )
    return parser.parse_args()


def get_threat_profile(label_name):
    name = label_name.lower()
    if "drone" in name:
        return COLOR_DRONE, "DRONE", 3
    elif "helicopter" in name:
        return COLOR_HELI, "HELICOPTER", 2
    elif "mosquito" in name or "mosq" in name:
        return COLOR_MOSQUITO, "MOSQUITO", 1
    elif "bird" in name:
        return COLOR_BIRD, "BIRD", 1
    else:
        return COLOR_OTHER, name.upper(), 1


class MultiTargetAirDefenseSystem:
    """
    Two-Stage Multi-Target Detection & Recognition Engine:
    - Stage 1: Scans the full frame to locate candidate bounding boxes anywhere in the image.
    - Stage 2: Crops each candidate and classifies it using the custom UAV-YOLONet.
    - Stage 3: Renders bounding boxes around each object and displays class + confidence above the box.
    """

    def __init__(self, detector_path, classifier_path, conf_thresh=0.02, min_conf=0.25, iou_thresh=0.45,
                 mosquito_frame_threshold=None, mosquito_cand_threshold=0.70, drone_threshold=0.50,
                 heli_threshold=0.70, multi_zoom=True, zoom_threshold=0.70, max_det=100,
                 enable_saliency=True, show_other=False):
        self.conf_thresh = conf_thresh
        self.min_conf = min_conf
        self.iou_thresh = iou_thresh
        self.drone_threshold = drone_threshold
        self.heli_threshold = heli_threshold
        self.multi_zoom = multi_zoom
        self.zoom_threshold = zoom_threshold
        self.max_det = max_det
        self.enable_saliency = enable_saliency
        self.show_other = show_other
        self.mosquito_cand_threshold = mosquito_cand_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SYSTEM INITIALIZING] Device: {self.device}")
        detector_str = str(detector_path).lower()
        if detector_str == "dilated" or detector_str.endswith(".pth") or "uav_dilated" in detector_str:
            self.detector_type = "dilated"
            weights_file = (
                Path(detector_path) if detector_str.endswith(".pth")
                else Path(__file__).resolve().parent.parent / "uav_dilated_yolo" / "weights" / "best_uav_dilated.pth"
            )
            cfg_file = Path(__file__).resolve().parent.parent / "uav_dilated_yolo" / "configs" / "config.yaml"
            print(f"[SYSTEM INITIALIZING] Loading Dilated-UAVNet CNN Locator: {weights_file}")
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.dilated_cfg = yaml.safe_load(f)
            from uav_dilated_yolo.models import build_model
            self.detector = build_model(self.dilated_cfg).to(self.device)
            if weights_file.exists():
                chk = torch.load(weights_file, map_location=self.device)
                sd = chk["model_state_dict"] if "model_state_dict" in chk else chk
                self.detector.load_state_dict(sd)
            self.detector.eval()
            self.dilated_input_size = self.dilated_cfg["model"]["input_size"]
        else:
            self.detector_type = "yolo"
            print(f"[SYSTEM INITIALIZING] Loading Object Locator: {detector_path}")
            self.detector = YOLO(detector_path)

        print(f"[SYSTEM INITIALIZING] Loading Custom Classifier: {classifier_path}")
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.classes = checkpoint.get("classes", CLASSES)
        saved_threshold = checkpoint.get("mosquito_frame_threshold", None)
        if mosquito_frame_threshold is not None:
            self.mosquito_frame_threshold = float(mosquito_frame_threshold)
        elif saved_threshold is not None and saved_threshold <= 0.90:
            self.mosquito_frame_threshold = float(saved_threshold)
        else:
            self.mosquito_frame_threshold = float(MOSQUITO_FRAME_THRESHOLD)
        print(f"[SYSTEM INITIALIZING] Supported Classes ({len(self.classes)}): {self.classes}")
        print(f"[SYSTEM INITIALIZING] Calibrated Gates: Mosquito-Frame={self.mosquito_frame_threshold:.2f}, Drone={self.drone_threshold:.2f}, MinConf={self.min_conf:.2f}")

        self.classifier = UAV_YOLONet(num_classes=len(self.classes), in_channels=3).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _classify_bgr(self, image_bgr):
        """Apply the same RGB, resize, and normalization path used in training."""
        pil_im = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inp = self.transform(pil_im).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.classifier(inp)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def _find_saliency_proposals(self, frame_bgr):
        """Fallback for images with sky targets that have low general COCO detector activations."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter tiny noise specks and borders
            if bw > 15 and bh > 15 and (bw * bh) < (h * w * 0.90):
                boxes.append([x, y, x + bw, y + bh])

        if len(boxes) == 0:
            return []

        # Merge overlapping / nearby bounding boxes
        boxes = np.array(boxes)
        # Bounding box of the largest salient cluster
        min_x = max(0, int(np.min(boxes[:, 0])) - 10)
        min_y = max(0, int(np.min(boxes[:, 1])) - 10)
        max_x = min(w, int(np.max(boxes[:, 2])) + 10)
        max_y = min(h, int(np.max(boxes[:, 3])) + 10)

        if (max_x - min_x) > 20 and (max_y - min_y) > 20:
            return [[min_x, min_y, max_x, max_y]]
        return []

    def _get_multi_zoom_crops(self, frame_bgr, box):
        """
        Extracts 3 progressive optical aperture zoom crops for a candidate bounding box:
        - Stage 1: Base Target (1.0x) - standard tight bounding box
        - Stage 2: Optical Near-Field (+25% context) - expanded window capturing rotor blades/wake or wing sweep
        - Stage 3: Far-Field Aperture (+50% context) - wider context verifying flight silhouette against environment
        """
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Stage 1: 1.0x base crop
        crop_base = frame_bgr[y1:y2, x1:x2]

        # Stage 2: +25% contextual aperture (captures complete propeller disk & motor arms / wing spread)
        pad2_w = int(bw * 0.25)
        pad2_h = int(bh * 0.25)
        z2_x1, z2_y1 = max(0, x1 - pad2_w), max(0, y1 - pad2_h)
        z2_x2, z2_y2 = min(W, x2 + pad2_w), min(H, y2 + pad2_h)
        crop_zoom2 = frame_bgr[z2_y1:z2_y2, z2_x1:z2_x2]

        # Stage 3: +50% contextual aperture (captures wider airspace/background separation)
        pad3_w = int(bw * 0.50)
        pad3_h = int(bh * 0.50)
        z3_x1, z3_y1 = max(0, x1 - pad3_w), max(0, y1 - pad3_h)
        z3_x2, z3_y2 = min(W, x2 + pad3_w), min(H, y2 + pad3_h)
        crop_zoom3 = frame_bgr[z3_y1:z3_y2, z3_x1:z3_x2]

        return [crop_base, crop_zoom2, crop_zoom3]

    @staticmethod
    def remove_inner_boxes(boxes, containment_ratio=0.50, iom_thresh=0.35, iou_thresh=0.30):
        """
        Suppresses candidate bounding boxes that are nested inside or heavily overlapping
        with another bounding box (eliminates duplicate box-in-box and sub-component clutter).
        Prioritizes high-confidence detector proposals over raw bounding box area.
        Supports both [x1, y1, x2, y2] lists and {"box": [x1, y1, x2, y2]} dicts.
        """
        if len(boxes) <= 1:
            return boxes

        def get_coords(item):
            return item["box"] if isinstance(item, dict) else item[:4]

        def get_conf(item):
            return item.get("detector_conf", 0.5) if isinstance(item, dict) else 0.5

        has_high_conf = any(get_conf(b) >= 0.40 for b in boxes)
        if has_high_conf:
            boxes_sorted = sorted(
                boxes,
                key=lambda b: (get_conf(b), (get_coords(b)[2] - get_coords(b)[0]) * (get_coords(b)[3] - get_coords(b)[1])),
                reverse=True
            )
        else:
            boxes_sorted = sorted(
                boxes,
                key=lambda b: (get_coords(b)[2] - get_coords(b)[0]) * (get_coords(b)[3] - get_coords(b)[1]),
                reverse=True
            )

        kept = []
        for c in boxes_sorted:
            x1_c, y1_c, x2_c, y2_c = get_coords(c)
            area_c = max(1, (x2_c - x1_c) * (y2_c - y1_c))
            is_dup = False
            for k in kept:
                x1_k, y1_k, x2_k, y2_k = get_coords(k)
                area_k = max(1, (x2_k - x1_k) * (y2_k - y1_k))
                ix1 = max(x1_c, x1_k)
                iy1 = max(y1_c, y1_k)
                ix2 = min(x2_c, x2_k)
                iy2 = min(y2_c, y2_k)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    containment = inter / area_c
                    iom = inter / min(area_c, area_k)
                    iou = inter / (area_c + area_k - inter)
                    if containment >= containment_ratio or iom >= iom_thresh or iou >= iou_thresh:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(c)
        return kept

    @staticmethod
    def remove_inner_detections(detections, containment_ratio=0.50, iom_thresh=0.35, iou_thresh=0.30):
        """
        Filters detections so that if a smaller object is nested within or heavily overlapping
        another detection (e.g. sub-component boxes, box-in-box), the duplicate is suppressed.
        """
        if len(detections) <= 1:
            return detections
        # Sort by bounding box area descending
        detections_sorted = sorted(
            detections,
            key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
            reverse=True
        )
        kept = []
        for c in detections_sorted:
            x1_c, y1_c, x2_c, y2_c = c["box"]
            area_c = max(1, (x2_c - x1_c) * (y2_c - y1_c))
            is_dup = False
            for k in kept:
                x1_k, y1_k, x2_k, y2_k = k["box"]
                area_k = max(1, (x2_k - x1_k) * (y2_k - y1_k))
                ix1 = max(x1_c, x1_k)
                iy1 = max(y1_c, y1_k)
                ix2 = min(x2_c, x2_k)
                iy2 = min(y2_c, y2_k)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    containment = inter / area_c
                    iom = inter / min(area_c, area_k)
                    iou = inter / (area_c + area_k - inter)
                    if containment >= containment_ratio or iom >= iom_thresh or iou >= iou_thresh:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(c)
        return kept

    @staticmethod
    def cluster_candidate_boxes(candidate_boxes, dist_thresh=20):
        """
        Groups adjacent/overlapping sub-component candidate boxes into unified
        bounding envelopes, allowing fragmented parts of large close-up objects
        (e.g., drone body + rotors + landing skids) to be evaluated as a whole.
        """
        if len(candidate_boxes) <= 1:
            return candidate_boxes

        n = len(candidate_boxes)
        adj = {i: [] for i in range(n)}
        for i in range(n):
            b1 = candidate_boxes[i]["box"]
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            for j in range(i + 1, n):
                b2 = candidate_boxes[j]["box"]
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])

                # Must have PHYSICAL overlap (no bridging across empty sky between separate flock targets)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    iom = inter / min(a1, a2)
                    is_subpart = (min(a1, a2) <= 0.35 * max(a1, a2) and min(a1, a2) <= 20000) or (max(a1, a2) <= 25000)
                    if is_subpart and (iom >= 0.15 or inter / (a1 + a2 - inter) >= 0.10):
                        adj[i].append(j)
                        adj[j].append(i)

        visited = set()
        envelopes = []
        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(comp) > 1:
                    min_x = min(candidate_boxes[k]["box"][0] for k in comp)
                    min_y = min(candidate_boxes[k]["box"][1] for k in comp)
                    max_x = max(candidate_boxes[k]["box"][2] for k in comp)
                    max_y = max(candidate_boxes[k]["box"][3] for k in comp)
                    max_conf = max(candidate_boxes[k].get("detector_conf", 0.5) for k in comp)
                    env_w = max_x - min_x
                    env_h = max_y - min_y
                    env_aspect = max(env_w, env_h) / max(1, min(env_w, env_h))
                    if env_aspect <= 2.2:
                        envelopes.append({
                            "box": [min_x, min_y, max_x, max_y],
                            "detector_cls": "drone",
                            "detector_conf": max_conf
                        })
        return envelopes + candidate_boxes

    def _draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=14, thickness=2):
        """Renders tactical corner targeting brackets around the bounding box."""
        # Clamp length to box size
        l = min(length, (x2 - x1) // 3, (y2 - y1) // 3)
        if l < 2:
            return
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        detections = []

        # COCO YOLO has no mosquito class, so its proposals can crop away the
        # insect or classify an unrelated patch. A high-confidence full-frame
        # mosquito result is therefore handled before COCO proposal generation.
        mosquito_idx = self.classes.index("mosquito") if "mosquito" in self.classes else None
        if mosquito_idx is not None:
            frame_probs = self._classify_bgr(frame_bgr)
            frame_pred = int(frame_probs.argmax())
            frame_mosquito_conf = float(frame_probs[mosquito_idx])
            if frame_pred == mosquito_idx and frame_mosquito_conf >= self.mosquito_frame_threshold:
                color, display_name, threat_rank = get_threat_profile("mosquito")
                detections.append({
                    "box": [0, 0, w - 1, h - 1],
                    "class": "mosquito",
                    "display_name": display_name,
                    "confidence": frame_mosquito_conf,
                    "color": color,
                    "threat_rank": threat_rank,
                    "probs": frame_probs,
                    "frame_classification": True,
                })

        if not detections:
            # Step 1: Detect candidate bounding boxes across the full frame.
            candidate_boxes = []

            if self.detector_type == "dilated":
                from uav_dilated_yolo.utils.boxes import non_max_suppression as dilated_nms
                img_resized = cv2.resize(frame_bgr, (self.dilated_input_size, self.dilated_input_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                inp_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    preds = self.detector(inp_t)
                    dets = dilated_nms(preds, conf_threshold=self.conf_thresh, iou_threshold=self.iou_thresh)[0]
                if dets.shape[0] > 0:
                    scale_x = w / self.dilated_input_size
                    scale_y = h / self.dilated_input_size
                    for d in dets:
                        x1 = max(0, int(d[0].item() * scale_x))
                        y1 = max(0, int(d[1].item() * scale_y))
                        x2 = min(w, int(d[2].item() * scale_x))
                        y2 = min(h, int(d[3].item() * scale_y))
                        bw = x2 - x1
                        bh = y2 - y1
                        aspect = max(bw, bh) / max(1, min(bw, bh))
                        if bw >= 6 and bh >= 6 and aspect <= 2.6:
                            if bw * bh > 100000 and float(d[4].item()) < 0.15:
                                continue
                            candidate_boxes.append({
                                "box": [x1, y1, x2, y2],
                                "detector_cls": "drone",
                                "detector_conf": float(d[4].item())
                            })
            else:
                results = self.detector.predict(
                    source=frame_bgr,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    agnostic_nms=True,
                    max_det=self.max_det,
                    device="0" if self.device.type == "cuda" else "cpu",
                    verbose=False
                )[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bw = x2 - x1
                    bh = y2 - y1

                    det_cls_id = int(box.cls[0])
                    det_cls_name = self.detector.names[det_cls_id] if hasattr(self.detector, 'names') else "unknown"

                    # Reject terrestrial fixtures, humans, indoor objects that cannot be aerial targets
                    if det_cls_name in NON_AERIAL_CLASSES:
                        continue

                    if bw >= 6 and bh >= 6:
                        det_conf = float(box.conf[0])
                        candidate_boxes.append({
                            "box": [x1, y1, x2, y2],
                            "detector_cls": det_cls_name,
                            "detector_conf": det_conf
                        })

            # If using dilated detector with fragmented sub-components, group into candidate envelopes
            if self.detector_type == "dilated" and len(candidate_boxes) > 1:
                candidate_boxes = self.cluster_candidate_boxes(candidate_boxes, dist_thresh=18)

            # Remove nested candidate boxes (camera gimbal, rotor arms inside main body)
            candidate_boxes = self.remove_inner_boxes(candidate_boxes, containment_ratio=0.50, iom_thresh=0.35, iou_thresh=0.30)

            # If detector found no candidate boxes, check salient edge content ONLY if explicitly enabled
            if len(candidate_boxes) == 0 and self.enable_saliency:
                salient = self._find_saliency_proposals(frame_bgr)
                for sb in salient:
                    candidate_boxes.append({
                        "box": sb,
                        "detector_cls": "unknown",
                        "detector_conf": 0.50
                    })

            # Step 2: Crop each candidate and classify with Dual-Stage Consensus & Threat Gating.
            drone_idx = self.classes.index("drone") if "drone" in self.classes else -1
            bird_idx = self.classes.index("bird") if "bird" in self.classes else -1
            heli_idx = self.classes.index("helicopter") if "helicopter" in self.classes else -1
            mosq_idx = self.classes.index("mosquito") if "mosquito" in self.classes else -1
            other_idx = self.classes.index("other") if "other" in self.classes else -1

            for cand in candidate_boxes:
                x1, y1, x2, y2 = cand["box"]
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                if self.multi_zoom:
                    zoom_crops = self._get_multi_zoom_crops(frame_bgr, cand["box"])
                    zoom_probs = []
                    for z_crop in zoom_crops:
                        if z_crop.size == 0 or z_crop.shape[0] < 5 or z_crop.shape[1] < 5:
                            continue
                        p = self._classify_bgr(z_crop)
                        zoom_probs.append(p)
                    if len(zoom_probs) > 0:
                        # Multi-scale contextual weighting: 1.0x (35%), 1.25x (40%), 1.50x (25%)
                        weights = [0.35, 0.40, 0.25][:len(zoom_probs)]
                        weights = np.array(weights) / sum(weights)
                        probs = np.sum([w * p for w, p in zip(weights, zoom_probs)], axis=0)

                        z1_drone = float(zoom_probs[0][drone_idx]) if drone_idx >= 0 else 0.0
                        z2_drone = float(zoom_probs[1][drone_idx]) if (len(zoom_probs) > 1 and drone_idx >= 0) else z1_drone
                        z3_drone = float(zoom_probs[2][drone_idx]) if (len(zoom_probs) > 2 and drone_idx >= 0) else z2_drone
                    else:
                        probs = self._classify_bgr(crop)
                        z1_drone = z2_drone = z3_drone = float(probs[drone_idx]) if drone_idx >= 0 else 0.0
                else:
                    probs = self._classify_bgr(crop)
                    z1_drone = z2_drone = z3_drone = float(probs[drone_idx]) if drone_idx >= 0 else 0.0

                pred_idx = int(probs.argmax())
                raw_class = self.classes[pred_idx]
                raw_conf = float(probs[pred_idx])

                drone_prob = float(probs[drone_idx]) if drone_idx >= 0 else 0.0
                bird_prob = float(probs[bird_idx]) if bird_idx >= 0 else 0.0
                heli_prob = float(probs[heli_idx]) if heli_idx >= 0 else 0.0
                mosq_prob = float(probs[mosq_idx]) if mosq_idx >= 0 else 0.0
                other_prob = float(probs[other_idx]) if other_idx >= 0 else 0.0

                det_cls = cand.get("detector_cls", "unknown")
                det_conf = cand.get("detector_conf", 0.0)

                peak_drone = max([z1_drone, z2_drone, z3_drone])
                drone_votes = sum(1 for zp in (zoom_probs if self.multi_zoom and len(zoom_probs) > 0 else [probs]) if zp[drone_idx] >= 0.50) if drone_idx >= 0 else 0

                # 3-Stage Calibrated Progressive Consensus Decision Logic:
                if det_cls == "bird":
                    # If YOLO detected a bird, require high drone confidence (>=0.65) to overturn the proposal
                    if drone_prob >= 0.65 and drone_prob > bird_prob:
                        pred_class = "drone"
                        confidence = drone_prob
                    elif heli_prob >= self.heli_threshold and (heli_prob >= 1.5 * bird_prob or heli_prob >= 0.80):
                        pred_class = "helicopter"
                        confidence = heli_prob
                    elif bird_prob >= 0.25:
                        pred_class = "bird"
                        confidence = max(bird_prob, det_conf)
                    else:
                        pred_class = "other"
                        confidence = other_prob
                elif drone_prob >= self.drone_threshold and drone_prob > bird_prob and z3_drone >= 0.10:
                    pred_class = "drone"
                    confidence = max(drone_prob, peak_drone)
                elif drone_votes >= 2 and peak_drone >= self.zoom_threshold and drone_prob >= self.drone_threshold and drone_prob > bird_prob and z3_drone >= 0.10:
                    pred_class = "drone"
                    confidence = peak_drone
                elif mosq_prob >= self.mosquito_cand_threshold and mosq_prob > bird_prob and mosq_prob > drone_prob:
                    pred_class = "mosquito"
                    confidence = mosq_prob
                elif raw_class == "drone":
                    if drone_prob >= self.drone_threshold and drone_prob > bird_prob and z3_drone >= 0.10:
                        pred_class = "drone"
                        confidence = max(drone_prob, peak_drone)
                    elif peak_drone >= self.zoom_threshold and drone_prob >= self.drone_threshold and drone_prob > bird_prob and z3_drone >= 0.10:
                        pred_class = "drone"
                        confidence = peak_drone
                    else:
                        if bird_prob >= 0.25:
                            pred_class = "bird"
                            confidence = bird_prob
                        else:
                            pred_class = "other"
                            confidence = other_prob
                elif raw_class == "helicopter":
                    if heli_prob >= self.heli_threshold and (heli_prob >= 1.5 * bird_prob or heli_prob >= 0.80):
                        pred_class = "helicopter"
                        confidence = heli_prob
                    elif bird_prob >= 0.25:
                        pred_class = "bird"
                        confidence = bird_prob
                    else:
                        pred_class = "other"
                        confidence = other_prob
                else:
                    pred_class = raw_class
                    confidence = raw_conf

                # Reject detections failing the minimum confirmed confidence threshold
                if confidence < self.min_conf:
                    continue

                # Suppress neutral background clutter ('other') unless explicitly requested
                if pred_class == "other" and not self.show_other:
                    continue

                color, display_name, threat_rank = get_threat_profile(pred_class)
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": pred_class,
                    "display_name": display_name,
                    "confidence": confidence,
                    "color": color,
                    "threat_rank": threat_rank,
                    "probs": probs,
                    "zoom_scores": [z1_drone, z2_drone, z3_drone, drone_prob],
                })

        # Fallback: if no valid targets were confirmed from detector candidate proposals,
        # evaluate salient contours to capture close-up drones or unconventional silhouettes
        if len(detections) == 0 and self.enable_saliency:
            salient = self._find_saliency_proposals(frame_bgr)
            for sb in salient:
                sx1, sy1, sx2, sy2 = sb
                scrop = frame_bgr[sy1:sy2, sx1:sx2]
                if scrop.size == 0 or scrop.shape[0] < 5 or scrop.shape[1] < 5:
                    continue
                if self.multi_zoom:
                    s_zoom_crops = self._get_multi_zoom_crops(frame_bgr, sb)
                    s_zoom_probs = []
                    for sz_crop in s_zoom_crops:
                        if sz_crop.size == 0 or sz_crop.shape[0] < 5 or sz_crop.shape[1] < 5:
                            continue
                        s_zoom_probs.append(self._classify_bgr(sz_crop))
                    if len(s_zoom_probs) > 0:
                        s_weights = [0.35, 0.40, 0.25][:len(s_zoom_probs)]
                        s_weights = np.array(s_weights) / sum(s_weights)
                        s_probs = np.sum([w * p for w, p in zip(s_weights, s_zoom_probs)], axis=0)
                        sz1_drone = float(s_zoom_probs[0][drone_idx]) if drone_idx >= 0 else 0.0
                        sz2_drone = float(s_zoom_probs[1][drone_idx]) if (len(s_zoom_probs) > 1 and drone_idx >= 0) else sz1_drone
                        sz3_drone = float(s_zoom_probs[2][drone_idx]) if (len(s_zoom_probs) > 2 and drone_idx >= 0) else sz2_drone
                    else:
                        s_probs = self._classify_bgr(scrop)
                        sz1_drone = sz2_drone = sz3_drone = float(s_probs[drone_idx]) if drone_idx >= 0 else 0.0
                else:
                    s_probs = self._classify_bgr(scrop)
                    sz1_drone = sz2_drone = sz3_drone = float(s_probs[drone_idx]) if drone_idx >= 0 else 0.0

                s_pred_idx = int(s_probs.argmax())
                s_raw_class = self.classes[s_pred_idx]
                s_drone_prob = float(s_probs[drone_idx]) if drone_idx >= 0 else 0.0
                s_bird_prob = float(s_probs[bird_idx]) if bird_idx >= 0 else 0.0
                s_heli_prob = float(s_probs[heli_idx]) if heli_idx >= 0 else 0.0
                s_mosq_prob = float(s_probs[mosq_idx]) if mosq_idx >= 0 else 0.0
                s_peak_drone = max([sz1_drone, sz2_drone, sz3_drone])

                if s_drone_prob >= self.drone_threshold and s_drone_prob > s_bird_prob:
                    s_pred_class = "drone"
                    s_confidence = max(s_drone_prob, s_peak_drone)
                elif s_peak_drone >= 0.65 and s_drone_prob > s_bird_prob:
                    s_pred_class = "drone"
                    s_confidence = s_peak_drone
                elif s_raw_class == "drone" and s_drone_prob >= self.drone_threshold:
                    s_pred_class = "drone"
                    s_confidence = max(s_drone_prob, s_peak_drone)
                elif s_raw_class == "helicopter" and s_heli_prob >= 0.50:
                    s_pred_class = "helicopter"
                    s_confidence = s_heli_prob
                elif s_mosq_prob >= self.mosquito_cand_threshold:
                    s_pred_class = "mosquito"
                    s_confidence = s_mosq_prob
                elif s_raw_class == "bird" and s_bird_prob >= 0.35:
                    s_pred_class = "bird"
                    s_confidence = s_bird_prob
                else:
                    s_pred_class = "other"
                    s_confidence = float(s_probs[s_pred_idx])

                if s_confidence >= self.min_conf and (s_pred_class != "other" or self.show_other):
                    s_color, s_display_name, s_threat_rank = get_threat_profile(s_pred_class)
                    detections.append({
                        "box": [sx1, sy1, sx2, sy2],
                        "class": s_pred_class,
                        "display_name": s_display_name,
                        "confidence": s_confidence,
                        "color": s_color,
                        "threat_rank": s_threat_rank,
                        "probs": s_probs,
                        "zoom_scores": [sz1_drone, sz2_drone, sz3_drone, s_drone_prob],
                    })

        # Remove inner detections where a sub-object is nested inside another detection
        detections = self.remove_inner_detections(detections, containment_ratio=0.50, iom_thresh=0.35, iou_thresh=0.30)

        # Hostile Target Co-occurrence Masking:
        # If an aerial vehicle (drone/helicopter) is locked, any small detection classified as a bird
        # that lies within the vehicle's structural perimeter is a sub-component (landing skid, gimbal,
        # rotor arm) rather than a biological bird flying inside the vehicle body.
        drone_dets = [d for d in detections if d["class"] in ("drone", "helicopter")]
        if len(drone_dets) > 0:
            cleaned_dets = []
            for d in detections:
                if d["class"] in ("drone", "helicopter", "mosquito"):
                    cleaned_dets.append(d)
                else:
                    # Check if this bird/other detection lies inside/adjacent to any drone envelope
                    cx = (d["box"][0] + d["box"][2]) / 2.0
                    cy = (d["box"][1] + d["box"][3]) / 2.0
                    is_sub_part = False
                    for dd in drone_dets:
                        dx1, dy1, dx2, dy2 = dd["box"]
                        # Spatial bounding envelope around the drone
                        pad_x = max(20, (dx2 - dx1) * 0.25)
                        pad_y = max(20, (dy2 - dy1) * 0.25)
                        if (dx1 - pad_x) <= cx <= (dx2 + pad_x) and (dy1 - pad_y) <= cy <= (dy2 + pad_y):
                            is_sub_part = True
                            break
                    if not is_sub_part:
                        cleaned_dets.append(d)
            detections = cleaned_dets

        # Sort detections: priority to highest threat (Drone=3, Helicopter=2, etc.), then highest confidence
        detections.sort(key=lambda d: (d["threat_rank"], d["confidence"]), reverse=True)

        # Step 3: Render HUD with dynamic bounding boxes and above-box labels
        annotated = frame_bgr.copy()

        highest_threat_rank = 0
        status_color = COLOR_BIRD

        for idx, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            conf = det["confidence"]
            cls_name = det["display_name"]
            rank = det["threat_rank"]

            if rank > highest_threat_rank:
                highest_threat_rank = rank
                status_color = color

            # 1. Glowing bounding box around the object
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=3)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            # 2. Tactical corner brackets
            self._draw_corner_brackets(annotated, x1, y1, x2, y2, color, length=16, thickness=2)

            # 3. Label tag placed strictly ABOVE the bounding box
            label_text = f"#{idx} {cls_name} {conf * 100:.1f}%"
            font_scale = max(0.40, min(0.65, (x2 - x1) / 320.0))
            thickness = 1 if font_scale < 0.55 else 2

            (tw, th), baseline = cv2.getTextSize(label_text, self.font, font_scale, thickness)

            # Position above top-left corner of the box
            tag_y2 = y1 - 4
            tag_y1 = y1 - th - baseline - 8

            # Flip inside box if too close to the top edge of the image
            if tag_y1 < 35:
                tag_y1 = y1 + 4
                tag_y2 = y1 + th + baseline + 8

            # Dark background pill with threat-colored border
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), color, 1)

            # Crisp white text with prediction and accuracy
            cv2.putText(
                annotated,
                label_text,
                (x1 + 4, tag_y2 - baseline - 2),
                self.font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # 4. Top System Telemetry Bar
        top_bar = annotated.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 35), (15, 15, 15), -1)
        cv2.addWeighted(top_bar, 0.80, annotated, 0.20, 0, annotated)

        if highest_threat_rank == 3:
            sys_status = "HOSTILE DRONE DETECTED - AIRSPACE ENGAGED"
        elif highest_threat_rank == 2:
            sys_status = "AIRSPACE CONFLICT - HELICOPTER IDENTIFIED"
        elif highest_threat_rank == 1:
            sys_status = "BIOLOGICAL / NEUTRAL OBJECT MONITORED"
        else:
            sys_status = "AIRSPACE SECURE - SCANNING SECTOR"

        fps = 1.0 / max(time.perf_counter() - t0, 1e-5)
        telemetry = f"SYS: MULTI-TARGET AIR DEFENSE | {sys_status} | TARGETS: {len(detections)} | FPS: {fps:.1f}"
        cv2.putText(annotated, telemetry, (15, 23), self.font, 0.44, status_color, 1, cv2.LINE_AA)

        # 5. Bottom Telemetry Banner
        bot_bar = annotated.copy()
        cv2.rectangle(bot_bar, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(bot_bar, 0.80, annotated, 0.20, 0, annotated)

        if len(detections) > 0:
            pri = detections[0]
            bot_text = f"[LOCK #1] {pri['display_name']} ({pri['confidence']*100:.1f}%) at [{pri['box'][0]}, {pri['box'][1]}, {pri['box'][2]}, {pri['box'][3]}]"
            cv2.putText(annotated, bot_text, (15, h - 12), self.font, 0.42, pri["color"], 1, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "[ALL CLEAR] NO AIRBORNE TARGETS IN MONITORED SECTOR", (15, h - 12),
                        self.font, 0.42, COLOR_BIRD, 1, cv2.LINE_AA)

        return annotated, detections


def run():
    args = parse_args()
    system = MultiTargetAirDefenseSystem(
        detector_path=args.detector,
        classifier_path=args.classifier,
        conf_thresh=args.conf,
        min_conf=args.min_conf,
        iou_thresh=args.iou,
        mosquito_frame_threshold=args.mosquito_frame_threshold,
        mosquito_cand_threshold=args.mosquito_cand_threshold,
        drone_threshold=args.drone_threshold,
        heli_threshold=args.heli_threshold,
        multi_zoom=args.multi_zoom,
        zoom_threshold=args.zoom_threshold,
        max_det=args.max_det,
        enable_saliency=args.enable_saliency,
        show_other=args.show_other,
    )

    out_dir = Path(args.output) / datetime.now().strftime("multi_detect_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)

    # 1. Camera index
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        print(f"[ACTIVE] Live stream on camera {args.source}. Press 'Q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = system.process_frame(frame)
            cv2.imshow("Multi-Target Aerial Defense HUD", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # 2. Video file
    elif src.is_file() and src.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_file = str(out_dir / f"annotated_{src.name}")
        writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        print(f"[VIDEO] Processing: {src.name} -> {save_file}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = system.process_frame(frame)
            writer.write(annotated)
            if args.view:
                cv2.imshow("Multi-Target Aerial Defense HUD", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[COMPLETE] Annotated video saved to: {save_file}")

    # 3. Directory of images or single image
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if src.is_file():
            img_paths = [src]
        elif src.is_dir():
            img_paths = sorted([
                p for p in src.rglob("*")
                if p.suffix.lower() in valid_exts
            ])
        else:
            raise FileNotFoundError(f"Source not found: {args.source}")

        print(f"\nProcessing {len(img_paths)} images with Multi-Target Detection & Classification...")
        print("-" * 80)
        print(f"{'Filename':<32} | {'Targets':<8} | {'Primary Object':<15} | {'Confidence'}")
        print("-" * 80)

        csv_rows = []

        for p in img_paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue

            annotated, dets = system.process_frame(frame)
            save_path = out_dir / p.name
            cv2.imwrite(str(save_path), annotated)

            primary_desc = "None"
            primary_conf = 0.0
            if len(dets) > 0:
                primary_desc = dets[0]["display_name"]
                primary_conf = dets[0]["confidence"]

            print(f"{p.name:<32} | {len(dets):<8} | {primary_desc:<15} | {primary_conf*100:>8.1f}%")

            for d in dets:
                row = {
                    "filename": p.name,
                    "target_class": d["class"],
                    "confidence": f"{d['confidence']:.4f}",
                    "box_x1": d["box"][0],
                    "box_y1": d["box"][1],
                    "box_x2": d["box"][2],
                    "box_y2": d["box"][3],
                }
                if "zoom_scores" in d:
                    row["zoom1_drone"] = f"{d['zoom_scores'][0]:.4f}"
                    row["zoom2_drone"] = f"{d['zoom_scores'][1]:.4f}"
                    row["zoom3_drone"] = f"{d['zoom_scores'][2]:.4f}"
                    row["avg_drone"] = f"{d['zoom_scores'][3]:.4f}"
                for c_idx, c_name in enumerate(system.classes):
                    row[f"{c_name}_prob"] = f"{d['probs'][c_idx]:.4f}"
                csv_rows.append(row)

        # Save CSV report
        csv_path = out_dir / "multi_target_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "filename", "target_class", "confidence",
                "box_x1", "box_y1", "box_x2", "box_y2",
                "zoom1_drone", "zoom2_drone", "zoom3_drone", "avg_drone"
            ] + [f"{c}_prob" for c in system.classes]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print("-" * 80)
        print(f"\n[COMPLETE] Annotated multi-object images saved to: {out_dir}")
        print(f"[COMPLETE] Multi-object predictions CSV saved to: {csv_path}")


if __name__ == "__main__":
    run()
