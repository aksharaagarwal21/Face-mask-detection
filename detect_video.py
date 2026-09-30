# pyre-unsafe
"""
detect_video.py — Video file face mask detection with annotated output

Faces are tracked across frames and each face's class probabilities are
smoothed over time (tracker.py), the same as in detect_realtime.py, so a
single blurry frame doesn't flip a label or fire a violation.
"""

import cv2
import os
import sys
import uuid
import time
import argparse
import logging
from tqdm import tqdm
from datetime import datetime

from face_detector import FaceDetector
from mask_detector import MaskDetector
from tracker import CentroidTracker
from analytics import SessionAnalytics
from logger import ViolationLogger, SessionLogger
from utils import (
    FPSCounter, draw_detection_box, draw_stats_overlay,
    draw_alert_banner, compute_compliance_pct
)
from config import (
    CLASS_COLORS, FACE_CONFIDENCE_THRESHOLD, FACE_DETECTOR_BACKEND, MASK_RUNTIME, TRACK_SMOOTHING
)

VIOLATION_LOG_EVERY = 60   # frames between log entries for the same tracked face

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VideoDetect")


def process_video(args):
    """Process a video file and write annotated output."""
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    logger.info(f"Processing video: {args.video}")

    # ── Setup
    face_detector = FaceDetector(confidence_threshold=args.face_conf, backend=args.backend)
    mask_detector = MaskDetector(runtime=args.runtime)
    tracker = CentroidTracker(smoothing=1.0 if args.no_smoothing else TRACK_SMOOTHING)
    analytics = SessionAnalytics(session_id)
    violation_logger = ViolationLogger()
    session_logger = SessionLogger()
    fps_counter = FPSCounter(window=30)

    # ── Open input video
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video info: {width}x{height} @ {video_fps:.1f}fps | {total_frames} frames")

    # ── Setup output writer
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(os.path.dirname(args.video), f"output_{base}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (width, height))
    logger.info(f"Output: {out_path}")

    frame_count = 0
    start_time = time.time()
    frame_stats = {"total": 0, "with_mask": 0,
                   "without_mask": 0, "mask_weared_incorrect": 0, "compliance_pct": 0.0}
    last_boxes = []        # (box, label, conf) from the latest detection, redrawn on skipped frames
    last_logged = {}       # track id -> frame of its last violation log entry
    has_violation = False
    every = args.skip_frames + 1

    # ── Process with progress bar
    with tqdm(total=total_frames, desc="Processing", unit="frame",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fps_counter.tick()

            # ── Detect + classify on every `every`-th frame (starting with the first).
            # Skipped frames are still written, with the latest boxes, so the output
            # keeps the input's length and speed.
            if (frame_count - 1) % every == 0:
                locs, rois = face_detector.detect_faces_rois(frame)
                probs = mask_detector.predict_probs(rois)
                tracker.update([loc[:4] for loc in locs], probs=probs)

                frame_stats = {"total": 0, "with_mask": 0,
                               "without_mask": 0, "mask_weared_incorrect": 0}
                analytics_dets = []
                last_boxes = []
                has_violation = False

                for i, (startX, startY, endX, endY, _) in enumerate(locs):
                    track_id = tracker.detection_ids[i]
                    label, conf = tracker.smoothed(track_id)
                    last_boxes.append(((startX, startY, endX, endY), label, conf))
                    frame_stats["total"] += 1
                    frame_stats[label] = frame_stats.get(label, 0) + 1

                    if mask_detector.is_violation(label, conf):
                        has_violation = True
                        # at most one log entry per face every VIOLATION_LOG_EVERY frames
                        if frame_count - last_logged.get(track_id, -VIOLATION_LOG_EVERY) >= VIOLATION_LOG_EVERY:
                            last_logged[track_id] = frame_count
                            violation_logger.log(
                                session_id=session_id,
                                face_id=track_id,
                                label=label,
                                confidence=conf,
                                frame_number=frame_count,
                                bbox=(startX, startY, endX - startX, endY - startY)
                            )

                    analytics_dets.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (startX, startY, endX - startX, endY - startY),
                        "centroid": ((startX + endX) // 2, (startY + endY) // 2)
                    })

                analytics.update(analytics_dets)
                frame_stats["compliance_pct"] = compute_compliance_pct(frame_stats)

            for (x1, y1, x2, y2), label, conf in last_boxes:
                draw_detection_box(frame, x1, y1, x2, y2, label, conf,
                                   CLASS_COLORS.get(label, (255, 255, 255)))
            draw_stats_overlay(frame, frame_stats, fps_counter.fps())

            if has_violation:
                draw_alert_banner(frame, "MASK VIOLATION DETECTED")

            # ── Frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                        (width - 210, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()

    duration = time.time() - start_time
    stats = analytics.get_stats()
    session_logger.save(session_id, stats, duration, source=args.video)
    report_path, _ = analytics.export_json_report()

    print("\n" + "="*60)
    print("✅ VIDEO PROCESSING COMPLETE")
    print("="*60)
    print(f"  Input:        {args.video}")
    print(f"  Output:       {out_path}")
    print(f"  Frames:       {frame_count}")
    print(f"  Duration:     {duration:.1f}s")
    print(f"  Total faces:  {stats['total']}")
    print(f"  Compliance:   {stats['compliance_pct']:.1f}%")
    print(f"  Report:       {report_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video file face mask detection")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Output video path (optional)")
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--backend", default=FACE_DETECTOR_BACKEND, choices=["yunet", "ssd"])
    parser.add_argument("--runtime", default=MASK_RUNTIME, choices=["auto", "keras", "tflite"])
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Label each frame on its own instead of smoothing per face")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Frames between detections (0 = detect on every frame); "
                             "skipped frames reuse the latest boxes")
    args = parser.parse_args()
    process_video(args)
