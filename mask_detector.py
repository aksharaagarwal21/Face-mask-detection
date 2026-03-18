# pyre-unsafe
"""
mask_detector.py — Mask classification inference engine using the trained MobileNetV2 model
"""

import os
import pickle
import logging
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import preprocess_face
from config import MASK_MODEL_PATH, LABEL_ENCODER_PATH, CLASSES, MASK_CONFIDENCE_THRESHOLD

logger = logging.getLogger("MaskDetector")


class MaskDetector:
    """
    Classifies face ROIs as: with_mask | without_mask | mask_weared_incorrect

    Usage:
        md = MaskDetector()
        results = md.predict_batch(face_rois)
        # results: list of (label, confidence)
    """

    def __init__(self, model_path=MASK_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        self.model = self._load_model(model_path)
        self.classes = self._load_classes(label_encoder_path)
        logger.info(f"MaskDetector initialized | classes: {self.classes}")

    def _load_model(self, path):
        from tensorflow.keras.models import load_model
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found at: {path}\n"
                "Please train the model first: python train.py"
            )
        model = load_model(path)
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
        """Preprocess a single face ROI for the model."""
        import cv2
        face = cv2.resize(face_roi, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32")
        face = preprocess_input(face)       # MobileNetV2-specific normalization
        return face

    def predict_single(self, face_roi):
        """
        Predict mask status for a single face ROI.

        Returns:
            (label: str, confidence: float)
        """
        face = self._preprocess(face_roi)
        face = np.expand_dims(face, axis=0)
        preds = self.model.predict(face, verbose=0)[0]
        idx = np.argmax(preds)
        label = self.classes[idx]
        confidence = float(preds[idx])
        return label, confidence

    def predict_batch(self, face_rois):
        """
        Predict mask status for a list of face ROIs (efficient batch inference).

        Args:
            face_rois: list of BGR numpy arrays

        Returns:
            list of (label, confidence) tuples
        """
        if not face_rois:
            return []

        batch = np.stack([self._preprocess(roi) for roi in face_rois], axis=0)
        preds = self.model.predict(batch, verbose=0)

        results = []
        for pred in preds:
            idx = np.argmax(pred)
            label = self.classes[idx]
            confidence = float(pred[idx])
            results.append((label, confidence))
        return results

    def predict_with_all_scores(self, face_roi):
        """
        Return scores for ALL classes (useful for visualization / debugging).

        Returns:
            dict: {class_name: confidence_score}
        """
        face = self._preprocess(face_roi)
        face = np.expand_dims(face, axis=0)
        preds = self.model.predict(face, verbose=0)[0]
        return {cls: float(preds[i]) for i, cls in enumerate(self.classes)}

    def is_violation(self, label, confidence):
        """Return True if this detection represents a mask policy violation."""
        return label in ("without_mask", "mask_weared_incorrect") and \
               confidence >= MASK_CONFIDENCE_THRESHOLD


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
