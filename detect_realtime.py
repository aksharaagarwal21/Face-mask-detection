# pyre-unsafe
"""
detect_realtime.py — Real-time webcam face mask detection
Features: live overlay stats, face tracking, alerts, screenshot, FPS counter
"""

import cv2
import os
import sys
import time
import uuid
import argparse
import logging
from datetime import datetime

from face_detector import FaceDetector
from mask_detector import MaskDetector
from tracker import CentroidTracker
from alert_system import AlertSystem
from analytics import SessionAnalytics
from logger import ViolationLogger, SessionLogger, AnalyticsLogger
from utils import (
    FPSCounter, draw_detection_box, draw_stats_overlay,
    draw_alert_banner, draw_tracker_id, compute_compliance_pct
)
from config import (
    CAMERA_INDEX, CLASS_COLORS, FRAME_WIDTH, FRAME_HEIGHT,
    MASK_CONFIDENCE_THRESHOLD, SCREENSHOTS_DIR
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealtimeDetect")


def run(args):
    """Main real-time detection loop."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    logger.info(f"Starting session: {session_id}")

    # ── Initialize components
    face_detector = FaceDetector(confidence_threshold=args.face_conf)
    mask_detector = MaskDetector()
    tracker = CentroidTracker()
    alert_system = AlertSystem(enabled=not args.no_alert)
    analytics = SessionAnalytics(session_id)
    violation_logger = ViolationLogger()
    session_logger = SessionLogger()
    analytics_logger = AnalyticsLogger()
    fps_counter = FPSCounter(window=30)

    # ── Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize buffer lag

    # ── Window setup
    window_name = "Face Mask Detection | Q=Quit  S=Screenshot  A=Alert  H=Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_count = 0
    start_time = time.time()
    alert_visible = False
    show_help = True
    help_timer = 180   # frames to show help bar

    logger.info("🟢 Detection started. Press Q to quit.")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Q       - Quit")
    print("  S       - Save screenshot")
    print("  A       - Toggle alert system")
    print("  H       - Save heatmap")
    print("  R       - Reset session stats")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        fps_counter.tick()
        fps = fps_counter.fps()

        # ── Detect faces
        locs, rois = face_detector.detect_faces_rois(frame)

        # ── Classify masks (batch)
        predictions = mask_detector.predict_batch(rois) if rois else []

        # ── Build detection list for tracker + analytics
        rects = [(s, t, e, b) for (s, t, e, b, _) in locs]
        labels = [pred[0] for pred in predictions]
        confs = [pred[1] for pred in predictions]

        # ── Update tracker
        tracker.update(rects, labels)
        tracked = tracker.get_all()

        # ── Frame stats
        frame_stats = {"total": 0, "with_mask": 0,
                       "without_mask": 0, "mask_weared_incorrect": 0}

        # ── Update analytics per-frame detections
        analytics_dets = []
        has_violation_this_frame = False

        for i, (startX, startY, endX, endY, face_conf) in enumerate(locs):
            if i >= len(predictions):
                break
            label, conf = predictions[i]
            color = CLASS_COLORS.get(label, (255, 255, 255))

            # Draw bounding box
            draw_detection_box(frame, startX, startY, endX, endY, label, conf, color)

            # Stats
            frame_stats["total"] += 1
            frame_stats[label] = frame_stats.get(label, 0) + 1

            # Violation check
            if mask_detector.is_violation(label, conf):
                has_violation_this_frame = True
                # Log violation every 30 frames per face
                if frame_count % 30 == 0:
                    cx = (startX + endX) // 2
                    cy = (startY + endY) // 2
                    violation_logger.log(
                        session_id=session_id,
                        face_id=i,
                        label=label,
                        confidence=conf,
                        frame_number=frame_count,
                        bbox=(startX, startY, endX - startX, endY - startY)
                    )

                    # Auto-save screenshot for severe violations
                    if args.auto_save and conf >= 0.85:
                        analytics.save_screenshot(frame, reason="auto_violation")

            analytics_dets.append({
                "label": label,
                "confidence": conf,
                "bbox": (startX, startY, endX - startX, endY - startY),
                "centroid": ((startX + endX) // 2, (startY + endY) // 2)
            })

        # ── Draw tracker IDs
        for obj_id, centroid, bbox, lbl, vcount in tracked:
            cx, cy = centroid
            draw_tracker_id(frame, cx, cy, obj_id)

        # ── Update analytics
        analytics.update(analytics_dets)
        frame_stats["compliance_pct"] = compute_compliance_pct(frame_stats)

        # ── Analytics time-series log every 5 seconds
        analytics_interval = max(args.fps * 5, 1)
        if frame_count % analytics_interval == 0:
            analytics_logger.record(frame_stats)

        # ── Alert system
        alert_visible = alert_system.update(has_violation_this_frame)
        if alert_visible:
            violator_count = (frame_stats.get("without_mask", 0) +
                              frame_stats.get("mask_weared_incorrect", 0))
            draw_alert_banner(
                frame,
                f"⚠  MASK VIOLATION DETECTED  |  {violator_count} person(s) not compliant"
            )

        # ── Draw stats overlay
        draw_stats_overlay(frame, frame_stats, fps)

        # ── Help bar
        if show_help and help_timer > 0:
            help_timer -= 1
            h, w = frame.shape[:2]
            help_text = "Q=Quit  S=Screenshot  A=Alert  H=Heatmap  R=Reset"
            cv2.putText(frame, help_text, (w // 2 - 220, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Show frame
        cv2.imshow(window_name, frame)

        # ── Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            path = analytics.save_screenshot(frame, reason="manual")
            logger.info(f"📸 Screenshot saved: {path}")
        elif key == ord('a'):
            alert_system.set_enabled(not alert_system.enabled)
            logger.info(f"Alert system: {'ON' if alert_system.enabled else 'OFF'}")
        elif key == ord('h'):
            _, hmap_path = analytics.generate_heatmap_image(frame)
            logger.info(f"🗺 Heatmap saved: {hmap_path}")
        elif key == ord('r'):
            analytics.reset()
            tracker.reset()
            alert_system.reset()
            frame_stats = {"total": 0, "with_mask": 0,
                           "without_mask": 0, "mask_weared_incorrect": 0}
            logger.info("🔄 Session stats reset")

    # ── Cleanup
    duration = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()

    # ── Save session summary
    session_logger.save(session_id, frame_stats, duration, source="webcam")
    report_path, _ = analytics.export_json_report()

    logger.info(f"\n✅ Session ended | Duration: {duration:.1f}s | Frames: {frame_count}")
    logger.info(f"   Report: {report_path}")
    logger.info(f"   Violations log: {violation_logger.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time face mask detection")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera device index (default: 0)")
    parser.add_argument("--face-conf", type=float, default=0.5,
                        help="Face detection confidence threshold")
    parser.add_argument("--mask-conf", type=float, default=MASK_CONFIDENCE_THRESHOLD,
                        help="Mask classification confidence threshold")
    parser.add_argument("--no-alert", action="store_true",
                        help="Disable audio alerts")
    parser.add_argument("--auto-save", action="store_true",
                        help="Auto-save screenshots on high-confidence violations")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (used for analytics interval)")
    args = parser.parse_args()

    run(args)
