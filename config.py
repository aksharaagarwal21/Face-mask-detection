# pyre-unsafe
"""
config.py — Centralized configuration for Face Mask Detection System
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# ─── Model Paths ──────────────────────────────────────────────────────────────
MASK_MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")      # backbone, input size, val metrics
METRICS_PATH = os.path.join(MODEL_DIR, "test_metrics.json")       # written by evaluate.py

# Face Detector: YuNet (default) or the older Caffe SSD
FACE_DETECTOR_BACKEND = "yunet"    # yunet | ssd
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
FACE_PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_WEIGHTS_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ─── Detection Parameters ─────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.5    # Min confidence to consider a face detection valid
FACE_NMS_THRESHOLD = 0.3           # IoU above which overlapping face boxes are merged
MIN_DETECT_FACE_SIZE = 10          # Ignore detections smaller than this (px)
DETECT_UPSCALE_TO = 800            # Enlarge smaller frames to this longer side before YuNet (0 = off)
DETECT_MAX_UPSCALE = 4.0           # ...but never by more than this factor
MASK_CONFIDENCE_THRESHOLD = 0.6    # Min confidence for mask prediction
MASK_TTA = True                    # Average each face with its mirror image (≈2x classifier cost)
INPUT_SIZE = (160, 160)            # Model input size (median training face is ~20px, so 224 buys nothing)
BACKBONE = "efficientnetv2b0"      # efficientnetv2b0 | efficientnetv2b1 | mobilenetv2 (see model.py)

# ─── Class Labels ─────────────────────────────────────────────────────────────
CLASSES = ["mask_weared_incorrect", "with_mask", "without_mask"]
CLASS_COLORS = {
    "with_mask": (0, 200, 0),          # Green (BGR)
    "without_mask": (0, 0, 220),       # Red (BGR)
    "mask_weared_incorrect": (0, 165, 255),  # Orange (BGR)
}
CLASS_EMOJI = {
    "with_mask": "✅",
    "without_mask": "❌",
    "mask_weared_incorrect": "⚠️",
}

# ─── Dataset Preparation ─────────────────────────────────────────────────────
CROP_MARGIN = 0.15                 # Context added on each side of a face box (fraction of face size)
MIN_FACE_SIZE = 10                 # Skip annotated faces smaller than this (px, shorter side)
SPLIT_SEED = 42

# ─── Training Parameters ─────────────────────────────────────────────────────
# Split is done per source photo, so faces from one photo never leak across splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 32
HEAD_EPOCHS = 3                    # Phase 1: head only, backbone frozen
INITIAL_LR = 1e-3                  # Phase 1 learning rate
EPOCHS = 30                        # Phase 2: full fine-tuning
FINE_TUNE_LR = 3e-4                # Phase 2 peak LR (cosine decay after warmup)
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 1e-4                # AdamW
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 10           # epochs without val macro-F1 improvement

# ─── Camera / Video ──────────────────────────────────────────────────────────
CAMERA_INDEX = 0                   # Default webcam (0 = primary)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# ─── Alert System ─────────────────────────────────────────────────────────────
ALERT_ENABLED = True
ALERT_COOLDOWN_FRAMES = 90         # ~3 seconds at 30 FPS between alerts
ALERT_TRIGGER_FRAMES = 3           # Consecutive frames before alert fires
ALERT_BEEP_FREQ = 1000             # Hz
ALERT_BEEP_DURATION = 300          # ms

# ─── Tracker ─────────────────────────────────────────────────────────────────
TRACKER_MAX_DISAPPEARED = 30       # Frames before removing a tracked face
TRACKER_MAX_DISTANCE = 100         # Max centroid distance to match faces
TRACK_SMOOTHING = 0.5              # Weight of the newest frame in each face's probability average (1 = off)

# ─── Analytics ───────────────────────────────────────────────────────────────
VIOLATION_LOG_PATH = os.path.join(LOGS_DIR, "violations.csv")
ANALYTICS_LOG_PATH = os.path.join(LOGS_DIR, "analytics.json")
SESSION_LOG_PATH = os.path.join(LOGS_DIR, "session_summary.json")

# ─── Web Dashboard ───────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
STREAM_QUALITY = 80                # JPEG quality for MJPEG stream (1-100)

# ─── Augmentation ────────────────────────────────────────────────────────────
AUGMENTATION = {
    "rotation": 0.05,              # fraction of a full turn (~18°)
    "zoom": 0.15,                  # mimics detector boxes being a bit loose or tight
    "shift": 0.08,
    "brightness": 0.25,
    "contrast": 0.25,
    "low_res_prob": 0.35,          # downscale to 12-48px and back: most real faces are tiny
    "low_res_range": (12, 48),
    "grayscale_prob": 0.05,        # IR / low-light cameras
}
BALANCE_POWER = 0.5                # class sampling ∝ count**power (0 = uniform, 1 = natural)

# ─── Heatmap ─────────────────────────────────────────────────────────────────
HEATMAP_RESOLUTION = (720, 1280)
HEATMAP_DECAY = 0.98              # How fast old positions fade
HEATMAP_FRAMES = 300              # Generate heatmap every N frames
