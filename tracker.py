# pyre-unsafe
"""
tracker.py — Centroid-based face tracker (unique ID per face across frames)

Besides keeping IDs stable, the tracker smooths each face's class
probabilities over time (exponential moving average). A single blurry frame
then can't flip someone from with_mask to without_mask and back, which is
where most of the visible flicker and false alerts in live video came from.
"""

import logging
import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from config import TRACKER_MAX_DISAPPEARED, TRACKER_MAX_DISTANCE, TRACK_SMOOTHING, CLASSES

logger = logging.getLogger("FaceTracker")

VIOLATION_CLASSES = ("without_mask", "mask_weared_incorrect")


class CentroidTracker:
    """
    Assigns unique IDs to faces across frames using centroid matching.

    Each face gets an ID that persists as long as the face remains
    in view. If a face disappears for more than maxDisappeared frames,
    its ID is deregistered.

    Usage:
        tracker.update(rects, probs=mask_detector.predict_probs(rois))
        for i, rect in enumerate(rects):
            obj_id = tracker.detection_ids[i]
            label, conf = tracker.smoothed(obj_id)
    """

    def __init__(self,
                 max_disappeared=TRACKER_MAX_DISAPPEARED,
                 max_distance=TRACKER_MAX_DISTANCE,
                 smoothing=TRACK_SMOOTHING,
                 classes=CLASSES):
        self.next_object_id = 0
        self.objects = OrderedDict()          # {id: centroid}
        self.disappeared = OrderedDict()      # {id: frames_missing}
        self.bboxes = OrderedDict()           # {id: (startX,startY,endX,endY)}
        self.labels = OrderedDict()           # {id: label}
        self.probs = OrderedDict()            # {id: smoothed class probabilities}
        self.violation_counts = OrderedDict() # {id: count}
        self.detection_ids = []               # object id for each rect of the last update()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.smoothing = smoothing            # weight of the newest frame (1.0 = no smoothing)
        self.classes = list(classes)

    def _centroid(self, startX, startY, endX, endY):
        return (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

    def register(self, centroid, bbox, label="unknown", probs=None):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.bboxes[obj_id] = bbox
        self.labels[obj_id] = "unknown"
        self.probs[obj_id] = None
        self.violation_counts[obj_id] = 0
        self.next_object_id += 1
        self._observe(obj_id, label, probs)
        return obj_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.labels[object_id]
        del self.probs[object_id]
        del self.violation_counts[object_id]

    def _observe(self, obj_id, label, probs):
        """Fold one frame's prediction into the object's smoothed state."""
        if probs is not None:
            p = np.asarray(probs, dtype="float64")
            prev = self.probs.get(obj_id)
            self.probs[obj_id] = p if prev is None else \
                (1.0 - self.smoothing) * prev + self.smoothing * p
            label = self.classes[int(np.argmax(self.probs[obj_id]))]
        self.labels[obj_id] = label
        if label in VIOLATION_CLASSES:
            self.violation_counts[obj_id] = self.violation_counts.get(obj_id, 0) + 1

    def smoothed(self, obj_id):
        """(label, confidence) for a tracked face after temporal smoothing."""
        p = self.probs.get(obj_id)
        if p is None:
            return self.labels.get(obj_id, "unknown"), 0.0
        idx = int(np.argmax(p))
        return self.classes[idx], float(p[idx])

    def update(self, rects, labels=None, probs=None):
        """
        Update tracker with new frame detections.

        Args:
            rects: list of (startX, startY, endX, endY) bounding boxes
            labels: optional list of labels matching each rect
            probs: optional per-rect class probabilities (enables smoothing)

        Returns:
            OrderedDict: {object_id: (centroid_x, centroid_y)}
        """
        if labels is None:
            labels = ["unknown"] * len(rects)
        if probs is None:
            probs = [None] * len(rects)
        self.detection_ids = [None] * len(rects)

        # No detections — mark all as disappeared
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Compute input centroids for this frame
        input_bboxes = [tuple(r[:4]) for r in rects]
        input_centroids = [self._centroid(*r) for r in input_bboxes]

        # No existing objects — register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.detection_ids[i] = self.register(centroid, input_bboxes[i], labels[i], probs[i])
            return self.objects

        # Match existing objects to new detections (optimal one-to-one assignment)
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
        rows, cols = linear_sum_assignment(D)

        used_rows = set()
        used_cols = set()
        for (row, col) in zip(rows, cols):
            if D[row, col] > self.max_distance:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = input_bboxes[col]
            self.disappeared[obj_id] = 0
            self._observe(obj_id, labels[col], probs[col])
            self.detection_ids[col] = obj_id
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in set(range(D.shape[0])) - used_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Register new unmatched detections
        for col in set(range(D.shape[1])) - used_cols:
            self.detection_ids[col] = self.register(
                input_centroids[col], input_bboxes[col], labels[col], probs[col])

        return self.objects

    def get_all(self):
        """Return list of (object_id, centroid, bbox, label, violation_count)."""
        result = []
        for obj_id, centroid in self.objects.items():
            result.append((
                obj_id,
                centroid,
                self.bboxes.get(obj_id, (0, 0, 0, 0)),
                self.labels.get(obj_id, "unknown"),
                self.violation_counts.get(obj_id, 0)
            ))
        return result

    def reset(self):
        """Reset all tracked objects."""
        self.objects.clear()
        self.disappeared.clear()
        self.bboxes.clear()
        self.labels.clear()
        self.probs.clear()
        self.violation_counts.clear()
        self.detection_ids = []
        self.next_object_id = 0
