import cv2
import numpy as np


class TacticalHUDVisualizer:
    """
    Renders high-contrast Tactical Air Defense Head-Up Display (HUD) overlays
    with glowing target reticles, threat level indicators, and telemetry.
    """

    def __init__(self, class_names=None):
        self.class_names = class_names or {0: "drone"}
        self.COLOR_DRONE     = (0, 0, 255)       # Red (Critical Threat)
        self.COLOR_HELI      = (0, 140, 255)     # Orange (Air Traffic Caution)
        self.COLOR_MOSQUITO  = (0, 215, 255)     # Amber/Gold (Micro Biological Target)
        self.COLOR_BIRD      = (0, 255, 0)       # Green (Neutral Biological)
        self.COLOR_BG        = (15, 15, 15)      # HUD Banner Dark Background
        self.COLOR_TEXT      = (255, 255, 255)   # White
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=16, thickness=2):
        """Renders tactical corner targeting brackets."""
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

    def draw_hud(self, frame, detections, fps=0.0):
        """
        frame: BGR image (H, W, 3)
        detections: list or array of [x1, y1, x2, y2, conf, class_id]
        fps: frames per second
        """
        h, w = frame.shape[:2]
        drone_count = 0
        heli_count = 0
        bio_count = 0

        # Sort detections by threat level (drone=1 highest, then heli=2, etc.)
        def threat_key(d):
            cid = int(d[5])
            if cid == 1: return (3, d[4])  # Drone
            if cid == 2: return (2, d[4])  # Helicopter
            if cid == 3: return (1, d[4])  # Mosquito
            return (0, d[4])               # Bird

        detections = sorted(detections, key=threat_key, reverse=True)

        # 1. Draw Target Reticles
        for idx, det in enumerate(detections, start=1):
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cls_name = self.class_names.get(int(cls_id), "target").lower()

            if "drone" in cls_name or int(cls_id) == 1:
                color = self.COLOR_DRONE
                drone_count += 1
            elif "helicopter" in cls_name or int(cls_id) == 2:
                color = self.COLOR_HELI
                heli_count += 1
            elif "mosquito" in cls_name or int(cls_id) == 3:
                color = self.COLOR_MOSQUITO
                bio_count += 1
            else:
                color = self.COLOR_BIRD
                bio_count += 1

            # Glowing bounding rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=3)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

            # Tactical brackets and center dot
            self._draw_corner_brackets(frame, x1, y1, x2, y2, color, length=14, thickness=2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 2, color, -1)

            # Target Lock Tag
            tag_text = f"#{idx} {cls_name.upper()} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(tag_text, self.font, 0.40, 1)

            tag_y1 = y1 - th - baseline - 4
            tag_y2 = y1 - 2
            if tag_y1 < 36:
                tag_y1 = y2 + 2
                tag_y2 = y2 + th + baseline + 4

            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, 1)
            cv2.putText(frame, tag_text, (x1 + 3, tag_y2 - baseline - 1),
                        self.font, 0.40, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # 2. Top System Telemetry Bar
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 35), self.COLOR_BG, -1)
        cv2.addWeighted(top_bar, 0.80, frame, 0.20, 0, frame)

        if drone_count > 0:
            status_text = f"THREAT ENGAGED - {drone_count} HOSTILE DRONE{'S' if drone_count>1 else ''} DETECTED"
            status_color = self.COLOR_DRONE
        elif heli_count > 0:
            status_text = "AIR TRAFFIC - HELICOPTER MONITORED"
            status_color = self.COLOR_HELI
        elif bio_count > 0:
            status_text = "BIOLOGICAL / NEUTRAL OBJECTS TRACKED"
            status_color = self.COLOR_BIRD
        else:
            status_text = "AIRSPACE SECURE - SCANNING SECTOR"
            status_color = self.COLOR_BIRD

        telemetry = f"SYS: DILATED-UAVNET v2 | {status_text} | TARGETS: {len(detections)} | FPS: {fps:.1f}"
        cv2.putText(frame, telemetry, (15, 23), self.font, 0.44, status_color, 1, cv2.LINE_AA)

        # 3. Bottom Target Telemetry Banner
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 35), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(bot_bar, 0.80, frame, 0.20, 0, frame)

        if len(detections) > 0:
            p = detections[0]
            bot_text = f"[LOCK] PRIMARY TARGET: ({int(p[0])}, {int(p[1])}) -> ({int(p[2])}, {int(p[3])}) | CONF: {p[4]*100:.1f}%"
            cv2.putText(frame, bot_text, (15, h - 12), self.font, 0.42, self.COLOR_DRONE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[ALL CLEAR] NO AIRBORNE THREATS IN MONITORED AIRSPACE", (15, h - 12),
                        self.font, 0.42, self.COLOR_OTHER, 1, cv2.LINE_AA)

        return frame
