# 🎭 Real-Time Face Mask Detection System

> AI-powered face mask compliance monitoring using optimized deep learning and computer vision.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **3-Class Detection** | `with_mask` / `without_mask` / `mask_weared_incorrect` |
| **Face Tracking** | Centroid-based tracker — unique ID per person |
| **Alert System** | Audio + visual alerts for violations |
| **Web Dashboard** | Live MJPEG stream, Chart.js analytics, violation logs |
| **Analytics** | Heatmaps, CSV logs, session JSON reports |
| **Multiple Modes** | Webcam, video file, single/batch images |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd c:\clientproject1\face_mask_detection
pip install -r requirements.txt
```

### 2. Download Face Detector Models

The models download automatically on first run. If manual download is needed:
- `deploy.prototxt` → `models/`
- `res10_300x300_ssd_iter_140000.caffemodel` → `models/`

### 3. Prepare Dataset & Train Model

```bash
# Option A: Generate sample dataset (for testing only)
python dataset_utils.py --generate-sample --n 100

# Option B: Download real dataset from Kaggle
# https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
# Place in dataset/with_mask/, dataset/without_mask/, dataset/mask_weared_incorrect/

# Train model
python train.py --epochs 20 --fine-tune

# Evaluate model
python evaluate.py
```

---

## 🖥️ Usage

### Web Dashboard (Recommended)
```bash
python app.py
# Open: http://127.0.0.1:5000
```

### Real-Time Webcam Detection
```bash
python detect_realtime.py

# Options:
python detect_realtime.py --camera 0 --no-alert --auto-save
```

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `A` | Toggle audio alerts |
| `H` | Save heatmap |
| `R` | Reset session stats |

### Video File Detection
```bash
python detect_video.py --video path/to/video.mp4
python detect_video.py --video input.mp4 --output annotated.mp4 --skip-frames 2
```

### Image Detection
```bash
# Single image
python detect_image.py --image path/to/photo.jpg --show

# Batch directory
python detect_image.py --dir path/to/images/ --output-dir results/
```

---

## 📁 Project Structure

```
face_mask_detection/
├── models/              # Model weights + face detector files (auto-downloaded)
│   └── plots/           # Training curves, confusion matrix, ROC curves
├── dataset/             # Training images
│   ├── with_mask/
│   ├── without_mask/
│   └── mask_weared_incorrect/
├── logs/                # Session logs, violation CSVs, analytics
├── screenshots/         # Auto-saved violation frames + heatmaps
├── static/              # Dashboard JS + CSS
├── templates/           # Flask HTML template
├── config.py            # ⚙️ All configuration settings
├── model.py             # MobileNetV2 model architecture
├── face_detector.py     # OpenCV DNN SSD face detector
├── mask_detector.py     # Mask classifier inference
├── tracker.py           # Centroid-based face tracker
├── alert_system.py      # Audio/visual alert system
├── analytics.py         # Statistics + heatmap generation
├── logger.py            # CSV + JSON structured loggers
├── utils.py             # Drawing, FPS, preprocessing utilities
├── train.py             # Two-phase training pipeline
├── evaluate.py          # Classification report, ROC curves, confusion matrix
├── detect_realtime.py   # Live webcam detection
├── detect_video.py      # Video file detection
├── detect_image.py      # Image/batch detection
├── dataset_utils.py     # Dataset prep + validation
└── app.py               # Flask web dashboard
```

---

## ⚙️ Configuration

Edit `config.py` to customize:
- `FACE_CONFIDENCE_THRESHOLD` — Face detection sensitivity (default: 0.5)
- `MASK_CONFIDENCE_THRESHOLD` — Mask prediction threshold (default: 0.6)
- `CAMERA_INDEX` — Webcam index (default: 0)
- `ALERT_ENABLED` — Enable/disable audio alerts
- `FLASK_PORT` — Web dashboard port (default: 5000)

---

## 🌐 Web Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/video_feed` | GET | MJPEG live stream |
| `/api/stats` | GET | Current detection statistics |
| `/api/timeline` | GET | Compliance time-series data |
| `/api/logs` | GET | Recent violation log entries |
| `/api/stream/start` | POST | Start camera detection |
| `/api/stream/stop` | POST | Stop camera detection |
| `/api/screenshot` | POST | Save current frame |
| `/api/heatmap` | POST | Generate & save heatmap |
| `/api/reset` | POST | Reset session stats + logs |
| `/api/logs/download` | GET | Download violations.csv |

---

## 🤖 Model Architecture

```
MobileNetV2 (ImageNet weights, frozen base)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.5)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(3, Softmax)  →  [with_mask, without_mask, mask_weared_incorrect]
```

**Training Strategy:**
- Phase 1: Train classification head only
- Phase 2: Fine-tune top 30 MobileNetV2 layers (lower LR)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python train.py --auto-sample` first |
| `Cannot open camera` | Check `CAMERA_INDEX` in `config.py` |
| No audio alerts (Linux/Mac) | Alerts use `winsound` on Windows; fallback to terminal bell |
| Low FPS | Reduce `FRAME_WIDTH/HEIGHT` in `config.py` or use `--skip-frames` |
| Face model download fails | Manually download from OpenCV GitHub (URLs in `face_detector.py`) |

---

## 📦 Requirements

- Python 3.8+
- TensorFlow ≥ 2.10
- OpenCV ≥ 4.7
- Flask ≥ 2.3
- Webcam (for real-time modes)
- ~500MB disk space (model weights)
