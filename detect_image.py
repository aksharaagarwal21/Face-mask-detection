# pyre-unsafe
"""
detect_image.py — Single image or batch directory face mask detection
"""

import cv2
import os
import sys
import glob
import argparse
import logging
from datetime import datetime

from face_detector import FaceDetector
from mask_detector import MaskDetector
from utils import draw_detection_box, compute_compliance_pct
from config import CLASS_COLORS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ImageDetect")

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def process_image(image_path, face_detector, mask_detector, output_dir=None, show=False):
    """Detect and classify masks in a single image. Returns stats dict."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Cannot read image: {image_path}")
        return None

    locs, rois = face_detector.detect_faces_rois(frame)
    predictions = mask_detector.predict_batch(rois) if rois else []

    stats = {"total": 0, "with_mask": 0,
             "without_mask": 0, "mask_weared_incorrect": 0}

    for i, (startX, startY, endX, endY, _) in enumerate(locs):
        if i >= len(predictions):
            break
        label, conf = predictions[i]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        draw_detection_box(frame, startX, startY, endX, endY, label, conf, color)
        stats["total"] += 1
        stats[label] = stats.get(label, 0) + 1

    stats["compliance_pct"] = compute_compliance_pct(stats)

    # ── Save annotated output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(output_dir, f"{name}_detected{ext}")
        cv2.imwrite(out_path, frame)
    else:
        name, ext = os.path.splitext(image_path)
        out_path = f"{name}_detected{ext}"
        cv2.imwrite(out_path, frame)

    # ── Add summary text to bottom of image
    h, w = frame.shape[:2]
    summary = (f"Faces:{stats['total']} | "
               f"✅{stats['with_mask']} | "
               f"❌{stats['without_mask']} | "
               f"⚠{stats['mask_weared_incorrect']} | "
               f"Compliance:{stats['compliance_pct']:.0f}%")
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, summary, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, frame)

    if show:
        cv2.imshow("Detection Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info(
        f"  {os.path.basename(image_path)} → "
        f"faces={stats['total']} compliance={stats['compliance_pct']:.0f}% "
        f"→ {out_path}"
    )
    return stats


def process_batch(image_paths, face_detector, mask_detector, output_dir=None):
    """Process multiple images and print a summary table."""
    all_stats = []
    print(f"\n{'FILE':<35} {'FACES':>5} {'MASK':>6} {'NO MASK':>8} {'INCORRECT':>10} {'COMPLI%':>8}")
    print("-" * 80)

    for path in image_paths:
        stats = process_image(path, face_detector, mask_detector, output_dir)
        if stats is None:
            continue
        all_stats.append(stats)
        fname = os.path.basename(path)[:34]
        print(f"  {fname:<33} {stats['total']:>5} "
              f"{stats['with_mask']:>6} {stats['without_mask']:>8} "
              f"{stats['mask_weared_incorrect']:>10} "
              f"{stats['compliance_pct']:>7.1f}%")

    # Summary
    if all_stats:
        total = sum(s['total'] for s in all_stats)
        with_mask = sum(s['with_mask'] for s in all_stats)
        without_mask = sum(s['without_mask'] for s in all_stats)
        incorrect = sum(s['mask_weared_incorrect'] for s in all_stats)
        compliance = (with_mask / total * 100) if total > 0 else 0

        print("-" * 80)
        print(f"  {'TOTAL':<33} {total:>5} {with_mask:>6} {without_mask:>8} "
              f"{incorrect:>10} {compliance:>7.1f}%")
        print(f"\n  Overall compliance: {compliance:.1f}%  |  Images: {len(all_stats)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image/batch face mask detection")
    parser.add_argument("--image", default=None,
                        help="Path to a single image file")
    parser.add_argument("--dir", default=None,
                        help="Path to directory of images (batch mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save annotated output images")
    parser.add_argument("--show", action="store_true",
                        help="Show result window (single image only)")
    parser.add_argument("--face-conf", type=float, default=0.5)
    args = parser.parse_args()

    face_detector = FaceDetector(confidence_threshold=args.face_conf)
    mask_detector = MaskDetector()

    if args.image:
        process_image(args.image, face_detector, mask_detector,
                      args.output_dir, show=args.show)

    elif args.dir:
        paths = []
        for ext in SUPPORTED_EXTS:
            paths += glob.glob(os.path.join(args.dir, f"*{ext}"))
            paths += glob.glob(os.path.join(args.dir, f"*{ext.upper()}"))
        paths = sorted(set(paths))
        logger.info(f"Found {len(paths)} images in {args.dir}")
        process_batch(paths, face_detector, mask_detector, args.output_dir)

    else:
        print("Usage: python detect_image.py --image path/to/image.jpg")
        print("       python detect_image.py --dir path/to/images/ --output-dir results/")
        sys.exit(0)
