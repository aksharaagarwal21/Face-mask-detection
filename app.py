# pyre-unsafe
"""
app.py — Flask web dashboard for Face Mask Detection
Features: MJPEG live stream, REST API, real-time stats, violation logs
"""

import os
import re
import cv2
import uuid
import time
import threading
import logging
from datetime import datetime
from flask import (
    Flask, render_template, Response, jsonify, request, send_file
)
from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, STREAM_QUALITY,
    CAMERA_INDEX, CLASS_COLORS, MASK_CONFIDENCE_THRESHOLD,
    MODEL_INFO_PATH, METRICS_PATH, MODEL_DIR, MAX_UPLOAD_MB,
    FIELD_TRACK_MAX_DISTANCE, FIELD_TRACK_MAX_DISAPPEARED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FlaskApp")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024   # larger uploads get HTTP 413

# Under gunicorn (e.g. the Docker image) the __main__ block doesn't run, so
# FMD_ACCESS_KEY is how to require an access key there.
if os.environ.get("FMD_ACCESS_KEY"):
    from secure_access import require_access_key as _require_access_key
    _require_access_key(app, os.environ["FMD_ACCESS_KEY"])

# Photo analysis (/api/predict) shares one pipeline; the lock serialises
# requests because a TFLite interpreter must not run on two threads at once.
_pipeline = None
_pipeline_lock = threading.Lock()

# Clients that stream frames (/field) pass ?session=<id>; each session has its
# own tracker so labels are smoothed per face across that camera's frames.
_track_sessions = {}    # session id -> (CentroidTracker, last used time)
TRACK_SESSION_TTL = 300
MAX_TRACK_SESSIONS = 50

# ─── Global State ─────────────────────────────────────────────────────────────
_frame_lock = threading.Lock()
_latest_frame = None
_session_stats = {
    "total": 0, "with_mask": 0, "without_mask": 0,
    "mask_weared_incorrect": 0, "compliance_pct": 0.0,
    "fps": 0.0, "frames_processed": 0, "session_id": "",
    "stream_active": False
}
_timeline = []          # List of {ts, compliance_pct}
_camera_thread = None
_stop_event = threading.Event()


# ─── Detection Thread ──────────────────────────────────────────────────────────
def _detection_loop():
    """Background thread: captures webcam + runs detection + updates global state."""
    from face_detector import FaceDetector
    from mask_detector import MaskDetector
    from tracker import CentroidTracker
    from analytics import SessionAnalytics
    from logger import ViolationLogger
    from utils import (
        FPSCounter, draw_detection_box, draw_stats_overlay,
        draw_alert_banner, encode_frame_to_jpeg, compute_compliance_pct
    )
    from alert_system import AlertSystem

    global _latest_frame, _session_stats, _timeline

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_web"
    _session_stats["session_id"] = session_id

    try:
        face_detector = FaceDetector()
        mask_detector = MaskDetector()
        tracker = CentroidTracker()
        alert_system = AlertSystem()
        analytics = SessionAnalytics(session_id)
        violation_logger = ViolationLogger()
        fps_counter = FPSCounter(window=30)
    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
        _session_stats["stream_active"] = False
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {CAMERA_INDEX}")
        _session_stats["stream_active"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    _session_stats["stream_active"] = True
    frame_count = 0
    timeline_interval = 30   # Record timeline snapshot every 30 frames
    last_logged = {}         # track id -> frame of its last violation log entry

    logger.info("🟢 Detection thread started")

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        fps_counter.tick()

        # ── Detect + classify, then smooth each face's scores over time
        locs, rois = face_detector.detect_faces_rois(frame)
        tracker.update([loc[:4] for loc in locs], probs=mask_detector.predict_probs(rois))

        frame_stats = {"total": 0, "with_mask": 0,
                       "without_mask": 0, "mask_weared_incorrect": 0}
        analytics_dets = []
        has_violation = False

        for i, (startX, startY, endX, endY, _) in enumerate(locs):
            track_id = tracker.detection_ids[i]
            label, conf = tracker.smoothed(track_id)
            color = CLASS_COLORS.get(label, (255, 255, 255))
            draw_detection_box(frame, startX, startY, endX, endY, label, conf, color)
            frame_stats["total"] += 1
            frame_stats[label] = frame_stats.get(label, 0) + 1

            if mask_detector.is_violation(label, conf):
                has_violation = True
                # at most one log entry per face every 60 frames
                if frame_count - last_logged.get(track_id, -60) >= 60:
                    last_logged[track_id] = frame_count
                    violation_logger.log(
                        session_id=session_id, face_id=track_id,
                        label=label, confidence=conf,
                        frame_number=frame_count,
                        bbox=(startX, startY, endX - startX, endY - startY)
                    )
            analytics_dets.append({
                "label": label, "confidence": conf,
                "bbox": (startX, startY, endX - startX, endY - startY),
                "centroid": ((startX + endX) // 2, (startY + endY) // 2)
            })

        analytics.update(analytics_dets)
        frame_stats["compliance_pct"] = round(compute_compliance_pct(frame_stats), 1)
        frame_stats["fps"] = round(fps_counter.fps(), 1)
        frame_stats["frames_processed"] = frame_count
        frame_stats["stream_active"] = True
        frame_stats["session_id"] = session_id

        # Update global stats
        _session_stats.update(frame_stats)

        # Alert
        if alert_system.update(has_violation):
            violators = frame_stats.get("without_mask", 0) + frame_stats.get("mask_weared_incorrect", 0)
            draw_alert_banner(frame, f"⚠  MASK VIOLATION — {violators} person(s) non-compliant")

        # Stats overlay
        draw_stats_overlay(frame, frame_stats, fps_counter.fps())

        # Timeline snapshot
        if frame_count % timeline_interval == 0:
            _timeline.append({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "compliance_pct": frame_stats["compliance_pct"],
                "with_mask": frame_stats["with_mask"],
                "without_mask": frame_stats["without_mask"],
                "mask_weared_incorrect": frame_stats["mask_weared_incorrect"],
            })
            _timeline = _timeline[-60:]   # Keep last 60 snapshots

        # Encode frame for MJPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)

        with _frame_lock:
            _latest_frame = buffer.tobytes()

    cap.release()
    _session_stats["stream_active"] = False
    logger.info("🔴 Detection thread stopped")


def _get_frame_generator():
    """MJPEG frame generator for streaming."""
    while True:
        with _frame_lock:
            frame_bytes = _latest_frame

        if frame_bytes is None:
            # Send placeholder while waiting
            blank = cv2.imencode('.jpg', _make_placeholder())[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank + b'\r\n')
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)    # ~30 FPS limit


def _make_placeholder():
    """Generate a dark placeholder frame when stream is not active."""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype='uint8')
    frame[:] = (20, 20, 30)
    cv2.putText(frame, "Stream not active", (160, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 100), 2)
    cv2.putText(frame, "Click 'Start Stream' to begin",
                (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 80), 1)
    return frame


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/field")
def field():
    """Phone / tablet field mode: back camera, zoom, flagged faces."""
    return render_template("field.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _get_frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(_session_stats)


@app.route("/api/timeline")
def api_timeline():
    return jsonify(_timeline)


@app.route("/api/logs")
def api_logs():
    from logger import ViolationLogger
    vl = ViolationLogger()
    n = request.args.get("n", 50, type=int)
    rows = vl.read_recent(n)
    return jsonify(rows)


@app.route("/api/stream/start", methods=["POST"])
def api_stream_start():
    global _camera_thread
    _stop_event.clear()
    if _camera_thread is None or not _camera_thread.is_alive():
        _camera_thread = threading.Thread(target=_detection_loop, daemon=True)
        _camera_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


@app.route("/api/stream/stop", methods=["POST"])
def api_stream_stop():
    _stop_event.set()
    return jsonify({"status": "stopped"})


@app.route("/api/screenshot", methods=["POST"])
def api_screenshot():
    global _latest_frame
    with _frame_lock:
        frame_bytes = _latest_frame
    if frame_bytes is None:
        return jsonify({"error": "No frame available"}), 400

    import numpy as np
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    from analytics import SessionAnalytics
    analytics = SessionAnalytics(_session_stats.get("session_id", "web"))
    path = analytics.save_screenshot(frame, reason="web_manual")
    return jsonify({"status": "saved", "path": path})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _session_stats, _timeline
    _session_stats.update({
        "total": 0, "with_mask": 0, "without_mask": 0,
        "mask_weared_incorrect": 0, "compliance_pct": 0.0, "frames_processed": 0
    })
    _timeline = []
    from logger import ViolationLogger, AnalyticsLogger
    ViolationLogger().clear()
    AnalyticsLogger().clear()
    return jsonify({"status": "reset"})


@app.route("/api/logs/download")
def api_logs_download():
    from config import VIOLATION_LOG_PATH
    if os.path.exists(VIOLATION_LOG_PATH):
        return send_file(
            VIOLATION_LOG_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="violations.csv"
        )
    return jsonify({"error": "Log file not found"}), 404


def _get_pipeline():
    """Create the photo-analysis pipeline on first use (call with _pipeline_lock held)."""
    global _pipeline
    if _pipeline is None:
        from pipeline import MaskPipeline
        _pipeline = MaskPipeline()
    return _pipeline


def _session_tracker(session_id, reset=False):
    """Tracker for one streaming client (call with _pipeline_lock held)."""
    from tracker import CentroidTracker
    now = time.time()
    for sid in [k for k, (_, ts) in _track_sessions.items() if now - ts > TRACK_SESSION_TTL]:
        del _track_sessions[sid]
    entry = _track_sessions.get(session_id)
    if entry is None or reset:
        if entry is None and len(_track_sessions) >= MAX_TRACK_SESSIONS:
            del _track_sessions[min(_track_sessions, key=lambda k: _track_sessions[k][1])]
        tracker = CentroidTracker(max_distance=FIELD_TRACK_MAX_DISTANCE,
                                  max_disappeared=FIELD_TRACK_MAX_DISAPPEARED)
    else:
        tracker = entry[0]
    _track_sessions[session_id] = (tracker, now)
    return tracker


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Analyse one photo.

    Send the image as multipart form field "image" or as the raw request
    body. Returns the MaskPipeline.analyze() JSON, or the annotated image as
    JPEG with ?annotate=1. ?range=far enables long-range detection for
    distant faces (slower). Streaming clients add ?session=<id> to get a
    stable track_id per face and labels smoothed across their frames;
    &reset=1 starts the tracking over (e.g. after zooming).

        curl -F image=@photo.jpg http://127.0.0.1:5000/api/predict
    """
    from pipeline import decode_image
    upload = request.files.get("image")
    data = upload.read() if upload else request.get_data()
    frame = decode_image(data)
    if frame is None:
        return jsonify({"error": "Send an image (JPEG/PNG) as form field 'image' or as the body"}), 400

    far = request.args.get("range", "near").lower() == "far"
    session_id = request.args.get("session", "")
    if session_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        return jsonify({"error": "session must be 1-64 letters, digits, '-' or '_'"}), 400

    with _pipeline_lock:
        pipeline = _get_pipeline()          # first request loads the models
        t0 = time.perf_counter()
        result = pipeline.analyze(frame, far=far)
        if session_id:
            reset = request.args.get("reset", "").lower() in ("1", "true", "yes")
            pipeline.apply_tracking(result, _session_tracker(session_id, reset))
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["range"] = "far" if far else "near"

    if request.args.get("annotate", "").lower() in ("1", "true", "yes"):
        ok, buf = cv2.imencode(".jpg", pipeline.annotate(frame, result),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")
    return jsonify(result)


@app.route("/api/model")
def api_model():
    """Model card numbers: backbone, calibration and held-out test results."""
    import json

    def load(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    info = load(MODEL_INFO_PATH) or {}
    test = load(METRICS_PATH) or {}
    pipeline = load(os.path.join(MODEL_DIR, "pipeline_metrics.json")) or []
    return jsonify({
        "backbone": info.get("backbone"),
        "input_size": info.get("input_size"),
        "classes": info.get("classes"),
        "temperature": info.get("temperature"),
        "test": {k: test.get(k) for k in
                 ("n_samples", "accuracy", "macro_f1", "balanced_accuracy", "macro_auc", "ece")},
        "end_to_end": [{k: p.get(k) for k in
                        ("backend", "detection_recall", "accuracy_on_detected", "end_to_end_accuracy")}
                       for p in pipeline],
    })


@app.route("/api/connect")
def api_connect():
    """Link (with access key) for phones; only shown on this computer itself."""
    from secure_access import is_local_request
    if not is_local_request():
        return jsonify({"error": "only available on the computer running the server"}), 403
    return jsonify({"phone_urls": app.config.get("PHONE_URLS", [])})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Upload larger than {MAX_UPLOAD_MB} MB"}), 413


@app.route("/api/heatmap", methods=["POST"])
def api_heatmap():
    global _latest_frame
    with _frame_lock:
        frame_bytes = _latest_frame
    if frame_bytes is None:
        return jsonify({"error": "No frame available"}), 400

    import numpy as np
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    from analytics import SessionAnalytics
    analytics = SessionAnalytics(_session_stats.get("session_id", "web"))
    _, path = analytics.generate_heatmap_image(frame)
    return jsonify({"status": "saved", "path": path})


if __name__ == "__main__":
    import argparse
    from secure_access import lan_ip, new_access_key, ensure_certificate, require_access_key

    parser = argparse.ArgumentParser(description="Face Mask Detection Web Dashboard")
    parser.add_argument("--host", default=FLASK_HOST)
    parser.add_argument("--port", type=int, default=FLASK_PORT)
    parser.add_argument("--debug", action="store_true", default=FLASK_DEBUG)
    parser.add_argument("--no-autostart", action="store_true",
                        help="Don't auto-start camera on launch")
    parser.add_argument("--https", action="store_true",
                        help="Serve over HTTPS with a self-signed certificate "
                             "(phones only allow the camera on HTTPS pages)")
    parser.add_argument("--access-key", default=os.environ.get("FMD_ACCESS_KEY", "auto"),
                        help="Key other devices must present: 'auto' generates one when the server "
                             "is reachable from the network, 'none' disables it "
                             "(default: $FMD_ACCESS_KEY, else auto)")
    parser.add_argument("--field", action="store_true",
                        help="Phone mode: reachable on the local network over HTTPS, access key on, "
                             "this computer's webcam off; prints the link for phones")
    args = parser.parse_args()

    if args.field:
        args.host, args.https, args.no_autostart = "0.0.0.0", True, True

    networked = args.host not in ("127.0.0.1", "localhost", "::1")
    key = {"auto": new_access_key() if networked else None, "none": None}.get(args.access_key,
                                                                             args.access_key)
    require_access_key(app, key)

    ssl_context = None
    scheme = "https" if args.https else "http"
    ip = lan_ip()
    if args.https:
        ssl_context = ensure_certificate(os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs"),
                                         [ip] if networked else [])

    phone_urls = []
    if networked:
        phone_urls = [f"{scheme}://{ip}:{args.port}/field" + (f"?key={key}" if key else "")]
    app.config["PHONE_URLS"] = phone_urls

    # Auto-start detection thread
    if not args.no_autostart:
        _stop_event.clear()
        _camera_thread = threading.Thread(target=_detection_loop, daemon=True)
        _camera_thread.start()
        logger.info("🟢 Camera detection thread auto-started")

    local = f"{scheme}://127.0.0.1:{args.port}"
    print(f"\n{'='*64}")
    print("  🎭 FACE MASK DETECTION — WEB DASHBOARD")
    print(f"{'='*64}")
    print(f"  Dashboard (this computer): {local}")
    print(f"  Field mode (this computer): {local}/field")
    for url in phone_urls:
        print(f"  Phone, same Wi-Fi:          {url}")
        print("                              (or scan the QR code on the dashboard)")
    if key:
        print(f"  Access key: {key}")
    if args.https:
        print("  Self-signed certificate: the browser warns once; choose Advanced -> Proceed.")
    if networked and not key:
        print("  ⚠ No access key: anyone on this network can open the camera stream and API.")
    print(f"{'='*64}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, ssl_context=ssl_context)
