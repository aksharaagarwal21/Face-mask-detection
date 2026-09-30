# pyre-unsafe
"""
mask_detector.py — Mask classification inference engine

Face ROIs should come from FaceDetector.detect_faces_rois(), which crops them
the same way the training set was built. With TTA on (default), each face is
scored together with its mirror image and the two probability vectors are
averaged, the same as evaluate.py does. The averaged probabilities are then
temperature-scaled (calibrate.py), so a confidence of 0.9 means the model is
right about 90% of the time and thresholds behave predictably.
"""

import os
import pickle
import logging
import numpy as np
from config import (
    MASK_MODEL_PATH, LABEL_ENCODER_PATH, CLASSES, MASK_CONFIDENCE_THRESHOLD, MASK_TTA
)
from calibrate import apply_temperature, load_temperature

VIOLATION_CLASSES = ("without_mask", "mask_weared_incorrect")

logger = logging.getLogger("MaskDetector")


class MaskDetector:
    """
    Classifies face ROIs as: with_mask | without_mask | mask_weared_incorrect

    Usage:
        md = MaskDetector()
        results = md.predict_batch(face_rois)
        # results: list of (label, confidence)
    """

    def __init__(self, model_path=MASK_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH,
                 tta=MASK_TTA, confidence_threshold=MASK_CONFIDENCE_THRESHOLD,
                 temperature=None):
        self.model = self._load_model(model_path)
        self.classes = self._load_classes(label_encoder_path)
        self.input_size = tuple(self.model.input_shape[1:3])   # (height, width)
        self.tta = tta
        self.confidence_threshold = confidence_threshold
        # None → the value calibrate.py stored in model_info.json (1.0 if never run)
        self.temperature = load_temperature() if temperature is None else float(temperature)
        logger.info(f"MaskDetector initialized | classes: {self.classes} | "
                    f"input {self.input_size} | TTA {'on' if tta else 'off'} | "
                    f"temperature {self.temperature:.3f}")

    def _load_model(self, path):
        import keras
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found at: {path}\n"
                "Please train the model first: python train.py"
            )
        model = keras.models.load_model(path, compile=False)
        logger.info(f"✅ Model loaded from {path}")
        return model

    def _load_classes(self, path):
        """Load label encoder or fall back to config CLASSES."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                le = pickle.load(f)
            return list(le.classes_)
        logger.warning("Label encoder not found — using default class order from config")
        return CLASSES

    def _preprocess(self, face_roi):
        """Resize a BGR face ROI to the model input (the model scales pixels itself)."""
        import cv2
        h, w = self.input_size
        # INTER_AREA when shrinking avoids aliasing on large webcam faces
        interp = cv2.INTER_AREA if face_roi.shape[0] > h else cv2.INTER_LINEAR
        face = cv2.resize(face_roi, (w, h), interpolation=interp)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face.astype("float32")

    def predict_probs(self, face_rois):
        """
        Class probabilities for a list of BGR face ROIs.

        Returns:
            np.ndarray of shape (len(face_rois), n_classes), rows in self.classes order
        """
        if not face_rois:
            return np.zeros((0, len(self.classes)), dtype="float32")
        batch = np.stack([self._preprocess(roi) for roi in face_rois], axis=0)
        if self.tta:
            batch = np.concatenate([batch, batch[:, :, ::-1, :]], axis=0)
        # predict_on_batch runs the compiled graph: ~5x faster than an eager call
        # and without model.predict()'s per-call dataset overhead
        probs = np.asarray(self.model.predict_on_batch(batch))
        if self.tta:
            n = len(face_rois)
            probs = (probs[:n] + probs[n:]) / 2.0
        return apply_temperature(probs, self.temperature)

    def label_of(self, probs):
        """(label, confidence) for one probability vector."""
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def predict_single(self, face_roi):
        """
        Predict mask status for a single face ROI.

        Returns:
            (label: str, confidence: float)
        """
        return self.label_of(self.predict_probs([face_roi])[0])

    def predict_batch(self, face_rois):
        """
        Predict mask status for a list of face ROIs (efficient batch inference).

        Args:
            face_rois: list of BGR numpy arrays

        Returns:
            list of (label, confidence) tuples
        """
        return [self.label_of(p) for p in self.predict_probs(face_rois)]

    def predict_with_all_scores(self, face_roi):
        """
        Return scores for ALL classes (useful for visualization / debugging).

        Returns:
            dict: {class_name: confidence_score}
        """
        preds = self.predict_probs([face_roi])[0]
        return {cls: float(preds[i]) for i, cls in enumerate(self.classes)}

    def is_violation(self, label, confidence):
        """Return True if this detection represents a mask policy violation."""
        return label in VIOLATION_CLASSES and confidence >= self.confidence_threshold


if __name__ == "__main__":
    import sys
    import cv2
    logging.basicConfig(level=logging.INFO)

    detector = MaskDetector()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        h, w = frame.shape[:2]
        face_roi = frame[h//4:3*h//4, w//4:3*w//4]
        label, conf = detector.predict_single(face_roi)
        print(f"Prediction: {label} ({conf*100:.1f}%)")
        all_scores = detector.predict_with_all_scores(face_roi)
        for cls, score in all_scores.items():
            print(f"  {cls}: {score*100:.1f}%")
        print("✅ MaskDetector working!")
    else:
        print("⚠ No webcam available for quick test")
