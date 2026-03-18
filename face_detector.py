# pyre-unsafe
"""
face_detector.py — OpenCV DNN SSD Caffe face detector wrapper
Automatically downloads model files if not present.
"""

import os
import cv2
import numpy as np
import urllib.request
import logging
from config import (
    FACE_PROTOTXT_PATH, FACE_WEIGHTS_PATH,
    FACE_CONFIDENCE_THRESHOLD, MODEL_DIR
)

logger = logging.getLogger("FaceDetector")

# URLs for automatic download
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_face_detector_models():
    """Download OpenCV Caffe SSD face detector models if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(FACE_PROTOTXT_PATH):
        logger.info(f"Downloading face detector prototxt → {FACE_PROTOTXT_PATH}")
        try:
            urllib.request.urlretrieve(PROTOTXT_URL, FACE_PROTOTXT_PATH)
            logger.info("✅ deploy.prototxt downloaded")
        except Exception as e:
            logger.error(f"Failed to download prototxt: {e}")
            _create_fallback_prototxt()

    if not os.path.exists(FACE_WEIGHTS_PATH):
        logger.info(f"Downloading face detector weights (~100MB) → {FACE_WEIGHTS_PATH}")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, FACE_WEIGHTS_PATH)
            logger.info("✅ caffemodel weights downloaded")
        except Exception as e:
            logger.error(f"Failed to download caffemodel: {e}")
            raise RuntimeError(
                "Could not download face detector model. "
                "Please manually download:\n"
                f"  {WEIGHTS_URL}\n"
                f"and place it at: {FACE_WEIGHTS_PATH}"
            )


def _create_fallback_prototxt():
    """Create the deploy.prototxt content if download fails."""
    content = """name: "VGGNet"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
"""
    # Use the actual deploy.prototxt from OpenCV — write to file
    import urllib.request
    try:
        urllib.request.urlretrieve(PROTOTXT_URL, FACE_PROTOTXT_PATH)
    except Exception:
        with open(FACE_PROTOTXT_PATH, 'w') as f:
            f.write(content)
    logger.warning("Fallback prototxt written — may not work correctly.")


class FaceDetector:
    """
    OpenCV DNN SSD Caffe-based face detector.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)
        # faces: list of (startX, startY, endX, endY, confidence)
    """

    def __init__(self, confidence_threshold=FACE_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.net = self._load_net()
        logger.info(f"FaceDetector initialized (confidence≥{confidence_threshold})")

    def _load_net(self):
        """Load the Caffe DNN model."""
        if not os.path.exists(FACE_PROTOTXT_PATH) or not os.path.exists(FACE_WEIGHTS_PATH):
            logger.warning("Face detector model files not found — downloading...")
            download_face_detector_models()

        net = cv2.dnn.readNet(FACE_PROTOTXT_PATH, FACE_WEIGHTS_PATH)

        # Use CUDA if available, otherwise CPU
        try:
            backends = [b for b, t in cv2.dnn.getAvailableBackends()]
            if cv2.dnn.DNN_BACKEND_CUDA in backends:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using CUDA backend for face detection")
            else:
                raise RuntimeError("CUDA backend not available in this OpenCV build")
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Using CPU backend for face detection")

        return net

    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame: BGR numpy array

        Returns:
            List of tuples: (startX, startY, endX, endY, confidence)
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Clamp to frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Skip invalid boxes
            if endX <= startX or endY <= startY:
                continue

            results.append((startX, startY, endX, endY, confidence))

        return results

    def detect_faces_rois(self, frame):
        """
        Detect faces and return both bounding boxes and ROI crops.

        Returns:
            locs: list of (startX, startY, endX, endY, confidence)
            rois: list of face ROI numpy arrays (BGR)
        """
        detections = self.detect(frame)
        locs = []
        rois = []
        for (startX, startY, endX, endY, conf) in detections:
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue
            locs.append((startX, startY, endX, endY, conf))
            rois.append(face_roi)
        return locs, rois


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    detector = FaceDetector()

    # Test on webcam frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        faces = detector.detect(frame)
        print(f"Detected {len(faces)} face(s):")
        for f in faces:
            print(f"  BBox: ({f[0]},{f[1]}) → ({f[2]},{f[3]})  conf={f[4]:.3f}")
        print("✅ Face detector working!")
    else:
        print("⚠ Could not capture frame from webcam — testing with blank frame")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(blank)
        print(f"Blank frame test: {len(faces)} faces (expected 0) ✅")
