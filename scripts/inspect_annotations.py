import cv2
import numpy as np


def draw_glowing_box(
    frame,
    x1, y1, x2, y2,
    confidence,
    label="DRONE DETECTED"
):
    # Ensure coordinates are integers
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # OpenCV uses BGR, so red is (0, 0, 255)
    red = (0, 0, 255)

    # Create a separate overlay for the glow
    overlay = frame.copy()

    # Draw thick red lines on the overlay
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        red,
        thickness=12
    )

    # Blur the lines to create a glow
    glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=10, sigmaY=10)

    # Blend the glow with the original frame
    frame = cv2.addWeighted(glow, 0.55, frame, 0.75, 0)

    # Draw a crisp red bounding box over the glow
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        red,
        thickness=2
    )

    # Add a second inner border for a brighter neon appearance
    cv2.rectangle(
        frame,
        (x1 + 3, y1 + 3),
        (x2 - 3, y2 - 3),
        (0, 0, 180),
        thickness=1
    )

    # Prepare detection label
    text = f"{label} | {confidence:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, text_thickness
    )

    # Place label above the box where possible
    label_y = max(y1 - 12, text_height + 12)

    # Draw a red label background
    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 10),
        (x1 + text_width + 12, label_y + baseline),
        red,
        thickness=-1
    )

    # Draw white text
    cv2.putText(
        frame,
        text,
        (x1 + 6, label_y - 4),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA
    )

    return frame