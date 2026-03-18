# pyre-unsafe
"""
detect_video.py — Video file face mask detection with annotated output
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
from analytics import SessionAnalytics
from logger import ViolationLogger, SessionLogger
from utils import (
    FPSCounter, draw_detection_box, draw_stats_overlay,
    draw_alert_banner, compute_compliance_pct
)
from config import CLASS_COLORS, MASK_CONFIDENCE_THRESHOLD

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
    face_detector = FaceDetector(confidence_threshold=args.face_conf)
    mask_detector = MaskDetector()
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
                   "without_mask": 0, "mask_weared_incorrect": 0}

    # ── Process with progress bar
    with tqdm(total=total_frames, desc="Processing", unit="frame",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fps_counter.tick()

            # Skip frames if requested
            if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                pbar.update(1)
                continue

            # ── Detect + classify
            locs, rois = face_detector.detect_faces_rois(frame)
            predictions = mask_detector.predict_batch(rois) if rois else []

            # Reset frame stats
            frame_stats = {"total": 0, "with_mask": 0,
                           "without_mask": 0, "mask_weared_incorrect": 0}
            analytics_dets = []
            has_violation = False

            for i, (startX, startY, endX, endY, _) in enumerate(locs):
                if i >= len(predictions):
                    break
                label, conf = predictions[i]
                color = CLASS_COLORS.get(label, (255, 255, 255))
                draw_detection_box(frame, startX, startY, endX, endY, label, conf, color)

                frame_stats["total"] += 1
                frame_stats[label] = frame_stats.get(label, 0) + 1

                if mask_detector.is_violation(label, conf):
                    has_violation = True
                    if frame_count % 60 == 0:
                        violation_logger.log(
                            session_id=session_id,
                            face_id=i,
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
            draw_stats_overlay(frame, frame_stats, fps_counter.fps())

            if has_violation:
                draw_alert_banner(frame, "⚠  MASK VIOLATION DETECTED")

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
    parser.add_argument("--face-conf", type=float, default=0.5)
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Number of frames to skip between detections (0=none)")
    args = parser.parse_args()
    process_video(args)
