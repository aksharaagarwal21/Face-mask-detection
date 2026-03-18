# pyre-unsafe
"""
logger.py — Structured CSV and JSON logger for the face mask detection system
"""

import os
import csv
import json
import logging
import threading
from datetime import datetime
from config import VIOLATION_LOG_PATH, SESSION_LOG_PATH, ANALYTICS_LOG_PATH, LOGS_DIR

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "system.log"), mode='a', encoding='utf-8')
    ]
)

logger = logging.getLogger("FaceMaskDetector")


# ─── Violation CSV Logger ─────────────────────────────────────────────────────
class ViolationLogger:
    """Thread-safe CSV logger that records each detected face event."""

    FIELDNAMES = ["timestamp", "session_id", "face_id", "label",
                  "confidence", "frame_number", "bbox_x", "bbox_y",
                  "bbox_w", "bbox_h"]

    def __init__(self, csv_path=VIOLATION_LOG_PATH):
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self, session_id, face_id, label, confidence,
            frame_number, bbox=None):
        """Log a single face detection event."""
        bbox = bbox or (0, 0, 0, 0)
        row = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "face_id": face_id,
            "label": label,
            "confidence": round(float(confidence), 4),
            "frame_number": frame_number,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
        }
        with self._lock:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)

    def read_recent(self, n=50):
        """Return the last n violation rows as list of dicts."""
        try:
            with self._lock:
                with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
            return rows[-n:]
        except Exception:
            return []

    def clear(self):
        """Clear all log entries (keep header)."""
        with self._lock:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()


# ─── Session Summary Logger ───────────────────────────────────────────────────
class SessionLogger:
    """Saves a session summary JSON after each detection run."""

    def __init__(self, path=SESSION_LOG_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, session_id, stats, duration_sec, source="webcam"):
        summary = {
            "session_id": session_id,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration_sec, 2),
            "stats": stats,
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Session {session_id} summary saved → {self.path}")


# ─── Analytics Time-Series Logger ─────────────────────────────────────────────
class AnalyticsLogger:
    """Appends time-stamped compliance snapshots to a JSONL file."""

    def __init__(self, path=ANALYTICS_LOG_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, stats: dict):
        entry = {
            "ts": datetime.now().isoformat(),
            **stats
        }
        with self._lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")

    def read_all(self):
        entries = []
        try:
            with self._lock:
                with open(self.path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
        except FileNotFoundError:
            pass
        return entries

    def clear(self):
        with self._lock:
            open(self.path, 'w').close()
