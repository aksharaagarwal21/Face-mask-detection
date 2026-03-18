# pyre-unsafe
"""
utils.py — Helper utilities: drawing, FPS, frame processing, overlays
"""

import cv2
import numpy as np
import time
from collections import deque
from config import CLASS_COLORS, INPUT_SIZE


# ─── FPS Calculator ───────────────────────────────────────────────────────────
class FPSCounter:
    """Rolling-window FPS calculator."""
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.time())

    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Drawing Utilities ────────────────────────────────────────────────────────
def draw_detection_box(frame, startX, startY, endX, endY, label, confidence, color):
    """Draw a styled bounding box with label and confidence."""
    # Main box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Build label text
    text = f"{label}: {confidence*100:.1f}%"

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_y = max(startY - 10, text_h + 10)

    # Fill label background
    cv2.rectangle(frame,
                  (startX, label_y - text_h - baseline - 4),
                  (startX + text_w + 4, label_y + baseline - 4),
                  color, -1)

    # Draw white text
    cv2.putText(frame, text,
                (startX + 2, label_y - 4),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def draw_stats_overlay(frame, stats: dict, fps: float):
    """Draw a top-left statistics panel on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    panel_h = 130
    panel_w = 320
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    dy = 24

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    y += dy
    cv2.putText(frame, f"Total Faces: {stats.get('total', 0)}", (10, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += dy
    cv2.putText(frame, f"With Mask:   {stats.get('with_mask', 0)}", (10, y), font, 0.55, (0, 200, 0), 1, cv2.LINE_AA)
    y += dy
    cv2.putText(frame, f"No Mask:     {stats.get('without_mask', 0)}", (10, y), font, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    y += dy
    cv2.putText(frame, f"Incorrect:   {stats.get('mask_weared_incorrect', 0)}", (10, y), font, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

    compliance = stats.get('compliance_pct', 0)
    color = (0, 200, 0) if compliance >= 80 else (0, 165, 255) if compliance >= 50 else (0, 0, 220)
    cv2.putText(frame, f"Compliance: {compliance:.0f}%", (165, 22), font, 0.55, color, 1, cv2.LINE_AA)

    return frame


def draw_alert_banner(frame, message="⚠ NO MASK DETECTED!", color=(0, 0, 220)):
    """Draw flashing alert banner at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.8, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, message, (text_x, h - 16), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_tracker_id(frame, cx, cy, track_id):
    """Draw tracker ID near face centroid."""
    cv2.putText(frame, f"ID:{track_id}", (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)


# ─── Frame Processing ─────────────────────────────────────────────────────────
def preprocess_face(face_roi, target_size=INPUT_SIZE):
    """Resize and normalize a face ROI for the mask classifier."""
    face = cv2.resize(face_roi, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


def resize_frame(frame, width=1280):
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def encode_frame_to_jpeg(frame, quality=80):
    """Encode frame to JPEG bytes for MJPEG streaming."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes()


def get_frame_dimensions(cap):
    """Get capture width and height."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h


def compute_compliance_pct(stats):
    """Compute compliance percentage from stats dict."""
    total = stats.get('total', 0)
    if total == 0:
        return 0.0
    return (stats.get('with_mask', 0) / total) * 100
