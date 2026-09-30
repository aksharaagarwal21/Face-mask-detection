# pyre-unsafe
"""
detect_image.py — Single image or batch directory face mask detection

Writes an annotated copy of every image and, optionally, machine-readable
results:
    --json report.json   one entry per image with every face (MaskPipeline format)
    --csv  faces.csv     one row per face: image, box, label, confidence, violation
"""

import cv2
import os
import sys
import csv
import json
import glob
import argparse
import logging

from pipeline import MaskPipeline
from config import FACE_CONFIDENCE_THRESHOLD, FACE_DETECTOR_BACKEND, MASK_RUNTIME

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ImageDetect")

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def output_path_for(image_path, output_dir=None):
    name, ext = os.path.splitext(os.path.basename(image_path))
    folder = output_dir or os.path.dirname(image_path)
    return os.path.join(folder, f"{name}_detected{ext}")


def draw_summary_bar(frame, result):
    """Counts and compliance along the bottom edge (ASCII: cv2 can't draw emoji)."""
    c = result["counts"]
    h, w = frame.shape[:2]
    text = (f"Faces: {c['total']} | mask: {c.get('with_mask', 0)} | "
            f"no mask: {c.get('without_mask', 0)} | incorrect: {c.get('mask_weared_incorrect', 0)} | "
            f"compliance: {result['compliance_pct']:.0f}%")
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    scale = min(0.5, 0.5 * (w - 16) / tw)          # shrink to fit narrow images
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_image(image_path, pipeline, output_dir=None, show=False):
    """Detect and classify masks in one image. Returns the analyze() result, or None."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Cannot read image: {image_path}")
        return None

    result = pipeline.analyze(frame)
    annotated = draw_summary_bar(pipeline.annotate(frame, result), result)

    out_path = output_path_for(image_path, output_dir)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, annotated)
    result["image"] = image_path
    result["annotated"] = out_path

    if show:
        cv2.imshow("Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    c = result["counts"]
    logger.info(f"  {os.path.basename(image_path)} → faces={c['total']} "
                f"compliance={result['compliance_pct']:.0f}% → {out_path}")
    return result


def process_batch(image_paths, pipeline, output_dir=None):
    """Process multiple images and print a summary table. Returns the list of results."""
    results = []
    print(f"\n{'FILE':<35} {'FACES':>5} {'MASK':>6} {'NO MASK':>8} {'INCORRECT':>10} {'COMPLI%':>8}")
    print("-" * 80)

    for path in image_paths:
        result = process_image(path, pipeline, output_dir)
        if result is None:
            continue
        results.append(result)
        c = result["counts"]
        fname = os.path.basename(path)[:34]
        print(f"  {fname:<33} {c['total']:>5} {c['with_mask']:>6} {c['without_mask']:>8} "
              f"{c['mask_weared_incorrect']:>10} {result['compliance_pct']:>7.1f}%")

    if results:
        total = sum(r["counts"]["total"] for r in results)
        sums = {k: sum(r["counts"][k] for r in results)
                for k in ("with_mask", "without_mask", "mask_weared_incorrect")}
        compliance = (sums["with_mask"] / total * 100) if total > 0 else 0
        print("-" * 80)
        print(f"  {'TOTAL':<33} {total:>5} {sums['with_mask']:>6} {sums['without_mask']:>8} "
              f"{sums['mask_weared_incorrect']:>10} {compliance:>7.1f}%")
        print(f"\n  Overall compliance: {compliance:.1f}%  |  Images: {len(results)}")
    return results


def write_json(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON report: {path}")


def write_csv(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "face", "x1", "y1", "x2", "y2", "face_confidence",
                    "label", "confidence", "violation"])
        for r in results:
            for i, face in enumerate(r["faces"]):
                w.writerow([r["image"], i, *face["box"], face["face_confidence"],
                            face["label"], face["confidence"], int(face["violation"])])
    logger.info(f"CSV report: {path}")


def find_images(folder):
    paths = []
    for ext in SUPPORTED_EXTS:
        paths += glob.glob(os.path.join(folder, f"*{ext}"))
        paths += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image/batch face mask detection")
    parser.add_argument("--image", default=None,
                        help="Path to a single image file")
    parser.add_argument("--dir", default=None,
                        help="Path to directory of images (batch mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save annotated output images (default: next to the input)")
    parser.add_argument("--json", default=None, help="Write all results to this JSON file")
    parser.add_argument("--csv", default=None, help="Write one row per face to this CSV file")
    parser.add_argument("--show", action="store_true",
                        help="Show result window (single image only)")
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--backend", default=FACE_DETECTOR_BACKEND, choices=["yunet", "ssd"])
    parser.add_argument("--runtime", default=MASK_RUNTIME, choices=["auto", "keras", "tflite"])
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Usage: python detect_image.py --image path/to/image.jpg")
        print("       python detect_image.py --dir path/to/images/ --output-dir results/ --json results/report.json")
        sys.exit(0)

    from face_detector import FaceDetector
    from mask_detector import MaskDetector
    pipeline = MaskPipeline(FaceDetector(confidence_threshold=args.face_conf, backend=args.backend),
                            MaskDetector(runtime=args.runtime))

    if args.image:
        result = process_image(args.image, pipeline, args.output_dir, show=args.show)
        results = [result] if result else []
    else:
        paths = find_images(args.dir)
        logger.info(f"Found {len(paths)} images in {args.dir}")
        results = process_batch(paths, pipeline, args.output_dir)

    if args.json:
        write_json(results, args.json)
    if args.csv:
        write_csv(results, args.csv)
