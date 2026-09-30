# pyre-unsafe
"""
face_detector.py — Face detection with OpenCV DNN

Backends:
    yunet (default) — YuNet (OpenCV Zoo, 2023). Small (230 KB), fast on CPU, and
                      much better than the SSD at finding faces that are
                      partly covered by a mask.
    ssd             — ResNet-10 SSD (Caffe, 2017). Kept as a fallback.

Model files download automatically if missing.
"""

import os
import cv2
import numpy as np
import urllib.request
import logging
from config import (
    FACE_PROTOTXT_PATH, FACE_WEIGHTS_PATH, YUNET_MODEL_PATH, FACE_DETECTOR_BACKEND,
    FACE_CONFIDENCE_THRESHOLD, FACE_NMS_THRESHOLD, MIN_DETECT_FACE_SIZE, MODEL_DIR
)
from utils import crop_face

logger = logging.getLogger("FaceDetector")

# URLs for automatic download
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _download(url, path, what):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Downloading {what} → {path}")
    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"✅ {what} downloaded")
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Could not download {what}: {e}\n"
            f"Please download it manually from:\n  {url}\nand place it at: {path}"
        )


def download_face_detector_models(backend="ssd"):
    """Download face detector model files for `backend` if not present."""
    if backend == "yunet":
        if not os.path.exists(YUNET_MODEL_PATH):
            _download(YUNET_URL, YUNET_MODEL_PATH, "YuNet face detector (~230 KB)")
        return
    if not os.path.exists(FACE_PROTOTXT_PATH):
        _download(PROTOTXT_URL, FACE_PROTOTXT_PATH, "SSD deploy.prototxt")
    if not os.path.exists(FACE_WEIGHTS_PATH):
        _download(WEIGHTS_URL, FACE_WEIGHTS_PATH, "SSD caffemodel weights (~10 MB)")


class FaceDetector:
    """
    Face detector wrapper with a single interface for both backends.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)
        # faces: list of (startX, startY, endX, endY, confidence)
    """

    def __init__(self, confidence_threshold=FACE_CONFIDENCE_THRESHOLD,
                 backend=FACE_DETECTOR_BACKEND, nms_threshold=FACE_NMS_THRESHOLD,
                 min_face_size=MIN_DETECT_FACE_SIZE):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.backend = backend
        self._input_size = None

        if backend == "yunet":
            self.net = self._load_yunet()
        elif backend == "ssd":
            self.net = self._load_ssd()
        else:
            raise ValueError(f"Unknown face detector backend '{backend}' (use 'yunet' or 'ssd')")
        logger.info(f"FaceDetector initialized ({backend}, confidence≥{confidence_threshold})")

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("YuNet needs opencv-python >= 4.8. Upgrade OpenCV or use backend='ssd'.")
        download_face_detector_models("yunet")
        return cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )

    def _load_ssd(self):
        download_face_detector_models("ssd")
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

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame: BGR numpy array

        Returns:
            List of tuples: (startX, startY, endX, endY, confidence)
        """
        h, w = frame.shape[:2]
        raw = self._detect_yunet(frame) if self.backend == "yunet" else self._detect_ssd(frame)

        results = []
        for (x1, y1, x2, y2, conf) in raw:
            startX, startY = max(0, int(x1)), max(0, int(y1))
            endX, endY = min(w - 1, int(x2)), min(h - 1, int(y2))
            if endX <= startX or endY <= startY:
                continue
            if min(endX - startX, endY - startY) < self.min_face_size:
                continue
            results.append((startX, startY, endX, endY, float(conf)))
        return results

    def _detect_yunet(self, frame):
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self.net.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self.net.detect(frame)
        if faces is None:
            return []
        # Each row: x, y, w, h, 5 landmark (x, y) pairs, score
        return [(f[0], f[1], f[0] + f[2], f[1] + f[3], f[14]) for f in faces]

    def _detect_ssd(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes, scores = [], []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(confidence)

        # The SSD often fires several overlapping boxes on one face
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
                 boxes[i][1] + boxes[i][3], scores[i]) for i in np.array(keep).flatten()]

    def detect_faces_rois(self, frame):
        """
        Detect faces and return both bounding boxes and ROI crops.

        ROIs are square crops with the same context margin used to build the
        training set (utils.crop_face), not the tight detector box.

        Returns:
            locs: list of (startX, startY, endX, endY, confidence)
            rois: list of face ROI numpy arrays (BGR)
        """
        locs, rois = [], []
        for loc in self.detect(frame):
            face_roi = crop_face(frame, loc[:4])
            if face_roi.size == 0:
                continue
            locs.append(loc)
            rois.append(face_roi)
        return locs, rois


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Face detector quick test")
    parser.add_argument("--backend", default=FACE_DETECTOR_BACKEND, choices=["yunet", "ssd"])
    parser.add_argument("--image", default=None, help="Test on an image instead of the webcam")
    args = parser.parse_args()
    detector = FaceDetector(backend=args.backend)

    if args.image:
        frame = cv2.imread(args.image)
        ret = frame is not None
    else:
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
        print("⚠ Could not capture a frame — testing with blank frame")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(blank)
        print(f"Blank frame test: {len(faces)} faces (expected 0) ✅")
