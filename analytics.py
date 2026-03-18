# pyre-unsafe
"""
analytics.py — Compliance statistics, heatmap generation, and session reporting
"""

import os
import json
import logging
import numpy as np
import cv2
from datetime import datetime
from config import (
    HEATMAP_RESOLUTION, HEATMAP_DECAY, HEATMAP_FRAMES,
    LOGS_DIR, SCREENSHOTS_DIR, CLASSES
)

logger = logging.getLogger("Analytics")


class SessionAnalytics:
    """
    Tracks per-session detection statistics and generates analytics.

    Tracks:
        - Total detections per class
        - Per-face-ID violation history
        - Time-series compliance snapshots (every N frames)
        - Heatmap of face positions (separate for violations)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.frame_count = 0

        # Counts
        self.counts: dict[str, int] = {}   # {label: count}
        self.total_faces = 0

        # Time-series data
        self.timeline = []               # [{frame, ts, compliance_pct, counts}]

        # Heatmap
        h, w = HEATMAP_RESOLUTION
        self.heatmap_all = np.zeros((h, w), dtype=np.float32)
        self.heatmap_violations = np.zeros((h, w), dtype=np.float32)

        # Screenshots
        self.screenshot_count = 0

        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    def update(self, detections):
        """
        Update analytics with detections from the current frame.

        Args:
            detections: list of dicts with keys:
                        {label, confidence, bbox: (x1,y1,x2,y2), centroid: (cx,cy)}
        """
        self.frame_count += 1

        frame_h, frame_w = HEATMAP_RESOLUTION

        for det in detections:
            label = det.get("label", "unknown")
            cx, cy = det.get("centroid", (0, 0))
            self.counts[label] = self.counts.get(label, 0) + 1
            self.total_faces += 1

            # Update heatmap
            if 0 <= cx < frame_w and 0 <= cy < frame_h:
                cv2.circle(self.heatmap_all, (cx, cy), 20, 1.0, -1)
                if label in ("without_mask", "mask_weared_incorrect"):
                    cv2.circle(self.heatmap_violations, (cx, cy), 20, 1.0, -1)

        # Decay heatmaps
        self.heatmap_all *= HEATMAP_DECAY
        self.heatmap_violations *= HEATMAP_DECAY

        # Record time-series snapshot every HEATMAP_FRAMES frames
        if self.frame_count % HEATMAP_FRAMES == 0:
            self.timeline.append({
                "frame": self.frame_count,
                "ts": datetime.now().isoformat(),
                "compliance_pct": self.compliance_pct,
                "with_mask": self.counts.get("with_mask", 0),
                "without_mask": self.counts.get("without_mask", 0),
                "mask_weared_incorrect": self.counts.get("mask_weared_incorrect", 0),
            })

    @property
    def compliance_pct(self):
        total = self.total_faces
        if total == 0:
            return 0.0
        return (float(self.counts.get("with_mask", 0)) / total) * 100

    def get_stats(self):
        """Return current stats as a JSON-serializable dict."""
        return {
            "total": self.total_faces,
            "with_mask": self.counts.get("with_mask", 0),
            "without_mask": self.counts.get("without_mask", 0),
            "mask_weared_incorrect": self.counts.get("mask_weared_incorrect", 0),
            "compliance_pct": round(self.compliance_pct, 1),
            "frames_processed": self.frame_count,
            "session_id": self.session_id,
        }

    def save_screenshot(self, frame, reason="manual"):
        """Save a screenshot frame to the screenshots directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screenshot_{reason}_{ts}.jpg"
        path = os.path.join(SCREENSHOTS_DIR, filename)
        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self.screenshot_count += 1
        logger.info(f"Screenshot saved: {path}")
        return path

    def generate_heatmap_image(self, base_frame=None):
        """
        Generate a colored heatmap visualization.

        Args:
            base_frame: Optional BGR frame to overlay heatmap on

        Returns:
            Colored heatmap as BGR numpy array
        """
        h, w = HEATMAP_RESOLUTION

        # Normalize and colorize
        norm_all = cv2.normalize(self.heatmap_all, None, 0, 255,
                                  cv2.NORM_MINMAX, cv2.CV_8U)
        colored = cv2.applyColorMap(norm_all, cv2.COLORMAP_JET)

        if base_frame is not None:
            base = cv2.resize(base_frame, (w, h))
            result = cv2.addWeighted(base, 0.5, colored, 0.5, 0)
        else:
            # Dark background
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            result = cv2.addWeighted(bg, 0.3, colored, 0.7, 0)

        # Overlay violation heatmap in red
        norm_viol = cv2.normalize(self.heatmap_violations, None, 0, 255,
                                   cv2.NORM_MINMAX, cv2.CV_8U)
        red_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        red_overlay[:, :, 2] = norm_viol   # Red channel
        result = cv2.addWeighted(result, 0.7, red_overlay, 0.3, 0)

        # Title text
        cv2.putText(result, "Compliance Heatmap",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOTS_DIR, f"heatmap_{ts}.jpg")
        cv2.imwrite(path, result)
        logger.info(f"Heatmap saved: {path}")
        return result, path

    def export_json_report(self):
        """Export full session report as JSON."""
        report = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "stats": self.get_stats(),
            "timeline": self.timeline,
            "screenshot_count": self.screenshot_count,
        }
        path = os.path.join(LOGS_DIR, f"report_{self.session_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Session report saved: {path}")
        return path, report

    def reset(self):
        """Reset all analytics for a new session."""
        self.__init__(self.session_id)
