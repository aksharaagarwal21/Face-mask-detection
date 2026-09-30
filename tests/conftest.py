import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CLASSES  # noqa: E402


class FakeFaceDetector:
    """Returns fixed boxes, crops them like the real detector."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect_faces_rois(self, frame):
        from utils import crop_face
        return list(self.boxes), [crop_face(frame, b[:4]) for b in self.boxes]


class FakeMaskDetector:
    """Returns fixed probabilities, one row per face."""

    classes = list(CLASSES)
    confidence_threshold = 0.6

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype="float64")

    def predict_probs(self, rois):
        return self.probs[:len(rois)]

    def label_of(self, p):
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])

    def is_violation(self, label, conf):
        return label in ("without_mask", "mask_weared_incorrect") and conf >= self.confidence_threshold


def probs_for(label, conf=0.9):
    """Probability row with `conf` on `label` and the rest split evenly."""
    p = np.full(len(CLASSES), (1.0 - conf) / (len(CLASSES) - 1))
    p[CLASSES.index(label)] = conf
    return p


@pytest.fixture
def frame():
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)
