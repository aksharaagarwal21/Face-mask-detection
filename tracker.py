# pyre-unsafe
"""
tracker.py — Centroid-based face tracker (unique ID per face across frames)
"""

import logging
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from config import TRACKER_MAX_DISAPPEARED, TRACKER_MAX_DISTANCE

logger = logging.getLogger("FaceTracker")


class CentroidTracker:
    """
    Assigns unique IDs to faces across frames using centroid matching.

    Each face gets an ID that persists as long as the face remains
    in view. If a face disappears for more than maxDisappeared frames,
    its ID is deregistered.
    """

    def __init__(self,
                 max_disappeared=TRACKER_MAX_DISAPPEARED,
                 max_distance=TRACKER_MAX_DISTANCE):
        self.next_object_id = 0
        self.objects = OrderedDict()          # {id: centroid}
        self.disappeared = OrderedDict()      # {id: frames_missing}
        self.bboxes = OrderedDict()           # {id: (startX,startY,endX,endY)}
        self.labels = OrderedDict()           # {id: label}
        self.violation_counts = OrderedDict() # {id: count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, startX, startY, endX, endY):
        return (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bboxes[self.next_object_id] = bbox
        self.labels[self.next_object_id] = "unknown"
        self.violation_counts[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.labels[object_id]
        del self.violation_counts[object_id]

    def update(self, rects, labels=None):
        """
        Update tracker with new frame detections.

        Args:
            rects: list of (startX, startY, endX, endY) bounding boxes
            labels: optional list of labels matching each rect

        Returns:
            OrderedDict: {object_id: (centroid_x, centroid_y)}
        """
        if labels is None:
            labels = ["unknown"] * len(rects)

        # No detections — mark all as disappeared
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Compute input centroids for this frame
        input_centroids = []
        input_bboxes = []
        for rect in rects:
            startX, startY, endX, endY = rect
            cx, cy = self._centroid(startX, startY, endX, endY)
            input_centroids.append((cx, cy))
            input_bboxes.append(rect)

        # No existing objects — register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
                self.labels[self.next_object_id - 1] = labels[i]
            return self.objects

        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))

        # Sort rows/cols by minimum distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = input_bboxes[col]
            self.labels[obj_id] = labels[col]
            self.disappeared[obj_id] = 0

            if labels[col] in ("without_mask", "mask_weared_incorrect"):
                self.violation_counts[obj_id] = self.violation_counts.get(obj_id, 0) + 1

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Register new unmatched detections
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col], input_bboxes[col])
            self.labels[self.next_object_id - 1] = labels[col]

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
        self.violation_counts.clear()
        self.next_object_id = 0
