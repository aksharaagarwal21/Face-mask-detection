# pyre-unsafe
"""
pipeline.py — Face detection + mask classification on one image, as data

MaskPipeline.analyze() returns a JSON-serialisable dict, which is what the
REST API sends back and what batch tools write to disk:

    {
      "faces": [
        {"box": [x1, y1, x2, y2], "face_confidence": 0.93,
         "label": "with_mask", "confidence": 0.998,
         "scores": {"mask_weared_incorrect": ..., "with_mask": ..., "without_mask": ...},
         "violation": false},
        ...
      ],
      "counts": {"total": 3, "with_mask": 2, "without_mask": 1, "mask_weared_incorrect": 0},
      "compliance_pct": 66.7,
      "image_size": [width, height]
    }
"""

import cv2
import numpy as np

from config import CLASS_COLORS
from utils import draw_detection_box, compute_compliance_pct


class MaskPipeline:
    """FaceDetector + MaskDetector with a data-only result."""

    def __init__(self, face_detector=None, mask_detector=None):
        if face_detector is None:
            from face_detector import FaceDetector
            face_detector = FaceDetector()
        if mask_detector is None:
            from mask_detector import MaskDetector
            mask_detector = MaskDetector()
        self.face_detector = face_detector
        self.mask_detector = mask_detector

    def analyze(self, frame, far=False):
        """Detect and classify every face in a BGR image (far=True: long-range detection)."""
        locs, rois = self.face_detector.detect_faces_rois(frame, far=far)
        probs = self.mask_detector.predict_probs(rois)
        classes = self.mask_detector.classes

        faces = []
        counts = {"total": 0, **{cls: 0 for cls in classes}}
        for (x1, y1, x2, y2, face_conf), p in zip(locs, probs):
            label, conf = self.mask_detector.label_of(p)
            faces.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "face_confidence": round(float(face_conf), 4),
                "label": label,
                "confidence": round(conf, 4),
                "scores": {cls: round(float(s), 4) for cls, s in zip(classes, p)},
                "violation": bool(self.mask_detector.is_violation(label, conf)),
            })
            counts["total"] += 1
            counts[label] += 1

        h, w = frame.shape[:2]
        return {
            "faces": faces,
            "counts": counts,
            "compliance_pct": round(compute_compliance_pct(counts), 1),
            "image_size": [int(w), int(h)],
        }

    @staticmethod
    def annotate(frame, result):
        """Draw the boxes and labels of an analyze() result onto a copy of frame."""
        out = frame.copy()
        for f in result["faces"]:
            x1, y1, x2, y2 = f["box"]
            draw_detection_box(out, x1, y1, x2, y2, f["label"], f["confidence"],
                               CLASS_COLORS.get(f["label"], (255, 255, 255)))
        return out


def decode_image(data):
    """BGR image from encoded bytes (JPEG/PNG/...), or None if they aren't an image."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
