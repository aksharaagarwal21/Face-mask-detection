# pyre-unsafe
"""
evaluate_pipeline.py — End-to-end evaluation: face detector + mask classifier

evaluate.py scores the classifier on ground-truth face crops. This script
scores what actually runs live: the face detector on full photos, then the
classifier on whatever the detector found. It uses the test-split photos
listed in dataset/manifest.csv, so none of them were seen in training.

Reports, per class:
    - detection recall   (annotated faces the detector found, IoU ≥ --iou)
    - end-to-end accuracy (found AND classified correctly)

Usage:
    python evaluate_pipeline.py                       # yunet vs ssd, with classifier
    python evaluate_pipeline.py --backends yunet --no-classify
"""

import os
import csv
import json
import argparse
import logging
from collections import defaultdict

import cv2
import numpy as np

from config import (DATASET_DIR, CLASSES, MODEL_DIR, FACE_CONFIDENCE_THRESHOLD,
                    DETECT_UPSCALE_TO)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluatePipeline")

DEFAULT_SOURCE = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
    "andrewmvd", "face-mask-detection", "versions", "1", "images")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, det_boxes, thr):
    """Greedy one-to-one matching by IoU. Returns {gt_index: det_index}."""
    pairs = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gt_boxes)
                    for di, d in enumerate(det_boxes)), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for score, gi, di in pairs:
        if score < thr:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di); out[gi] = di
    return out


def load_test_faces(manifest_path, split="test"):
    """{source_image: [(label, (x1, y1, x2, y2)), ...]} for one split."""
    faces = defaultdict(list)
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                box = tuple(int(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                faces[row["source_image"]].append((row["label"], box))
    return faces


def find_image(source_dir, name):
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(source_dir, name + ext)
        if os.path.exists(path):
            return path
    return None


def run(backend, faces_by_image, source_dir, mask_detector, iou_thr, conf,
        upscale_to=DETECT_UPSCALE_TO):
    from face_detector import FaceDetector
    detector = FaceDetector(confidence_threshold=conf, backend=backend, upscale_to=upscale_to)

    total = defaultdict(int)
    found = defaultdict(int)
    correct = defaultdict(int)
    n_det = n_fp = 0

    for name, gts in faces_by_image.items():
        path = find_image(source_dir, name)
        if path is None:
            logger.warning(f"Missing source image {name}")
            continue
        frame = cv2.imread(path)
        locs, rois = detector.detect_faces_rois(frame)
        det_boxes = [loc[:4] for loc in locs]
        m = match([box for _, box in gts], det_boxes, iou_thr)
        n_det += len(det_boxes)
        n_fp += len(det_boxes) - len(m)

        preds = mask_detector.predict_batch(rois) if (mask_detector and rois) else []
        for gi, (label, _) in enumerate(gts):
            total[label] += 1
            if gi in m:
                found[label] += 1
                if preds and preds[m[gi]][0] == label:
                    correct[label] += 1

    summary = {"backend": backend, "iou": iou_thr, "confidence": conf,
               "upscale_to": upscale_to if backend == "yunet" else 0,
               "detections": n_det, "unmatched_detections": n_fp, "per_class": {}}
    for cls in CLASSES:
        t = total[cls]
        summary["per_class"][cls] = {
            "faces": t,
            "detection_recall": round(found[cls] / t, 4) if t else None,
            "end_to_end_accuracy": round(correct[cls] / t, 4) if (t and mask_detector) else None,
        }
    all_t = sum(total.values())
    summary["detection_recall"] = round(sum(found.values()) / all_t, 4) if all_t else None
    if mask_detector:
        all_found = sum(found.values())
        summary["accuracy_on_detected"] = round(sum(correct.values()) / all_found, 4) if all_found else None
        summary["end_to_end_accuracy"] = round(sum(correct.values()) / all_t, 4) if all_t else None
    return summary


def print_summary(s):
    print(f"\n── {s['backend'].upper()}  (IoU ≥ {s['iou']}, conf ≥ {s['confidence']}, "
          f"upscale to {s['upscale_to'] or 'off'}) " + "─" * 20)
    print(f"  {'class':<24}{'faces':>7}{'found':>9}{'found+correct':>15}")
    for cls, r in s["per_class"].items():
        rec = f"{r['detection_recall']*100:.1f}%" if r["detection_recall"] is not None else "-"
        e2e = f"{r['end_to_end_accuracy']*100:.1f}%" if r["end_to_end_accuracy"] is not None else "-"
        print(f"  {cls:<24}{r['faces']:>7}{rec:>9}{e2e:>15}")
    print(f"  overall detection recall:        {s['detection_recall']*100:.1f}%")
    if "accuracy_on_detected" in s:
        print(f"  classifier accuracy on detected: {s['accuracy_on_detected']*100:.1f}%")
        print(f"  end-to-end accuracy:             {s['end_to_end_accuracy']*100:.1f}%")
    print(f"  detections not matching a labelled face: {s['unmatched_detections']} of {s['detections']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end detector + classifier evaluation")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Folder with the original Kaggle images")
    parser.add_argument("--manifest", default=os.path.join(DATASET_DIR, "manifest.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--backends", nargs="+", default=["yunet", "ssd"])
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--conf", type=float, default=FACE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--upscale-to", type=int, default=DETECT_UPSCALE_TO,
                        help="Enlarge smaller photos to this longer side before detection (0 = off)")
    parser.add_argument("--no-classify", action="store_true", help="Only measure detection recall")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "pipeline_metrics.json"))
    args = parser.parse_args()

    faces_by_image = load_test_faces(args.manifest, args.split)
    logger.info(f"{sum(len(v) for v in faces_by_image.values())} labelled faces "
                f"in {len(faces_by_image)} {args.split} photos")

    mask_detector = None
    if not args.no_classify:
        from mask_detector import MaskDetector
        mask_detector = MaskDetector()

    results = [run(b, faces_by_image, args.source, mask_detector, args.iou, args.conf,
                   args.upscale_to)
               for b in args.backends]
    for r in results:
        print_summary(r)
    if args.out and not args.no_classify:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {args.out}")
