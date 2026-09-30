import os

import cv2
import numpy as np
import pytest

from config import YUNET_MODEL_PATH
from evaluate_pipeline import iou, match
from face_detector import FaceDetector


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_is_one_to_one_and_respects_threshold():
    gts = [(0, 0, 10, 10), (20, 0, 30, 10)]
    dets = [(1, 0, 11, 10), (0, 0, 10, 10), (50, 50, 60, 60)]
    # gt 0 takes the exact box (IoU 1.0) over the shifted one; gt 1 has no match
    assert match(gts, dets, thr=0.3) == {0: 1}
    # with the exact box gone, the shifted one (IoU 0.82) is used instead
    assert match(gts, dets[:1], thr=0.3) == {0: 0}


def detector_without_net(backend="yunet", upscale_to=800, max_upscale=4.0):
    d = FaceDetector.__new__(FaceDetector)
    d.backend, d.upscale_to, d.max_upscale = backend, upscale_to, max_upscale
    return d


@pytest.mark.parametrize("h,w,expected", [
    (267, 400, 2.0),       # small photo: enlarged to 800 on the longer side
    (100, 100, 4.0),       # capped at max_upscale
    (720, 1280, 1.0),      # 720p webcam: untouched
    (600, 800, 1.0),
])
def test_upscale_factor(h, w, expected):
    assert detector_without_net()._upscale_factor(h, w) == pytest.approx(expected)


def test_no_upscaling_for_ssd_or_when_disabled():
    assert detector_without_net(backend="ssd")._upscale_factor(267, 400) == 1.0
    assert detector_without_net(upscale_to=0)._upscale_factor(267, 400) == 1.0


@pytest.mark.skipif(not (os.path.exists(YUNET_MODEL_PATH) and hasattr(cv2, "FaceDetectorYN")),
                    reason="YuNet model or OpenCV >= 4.8 not available")
def test_yunet_finds_no_faces_in_a_blank_frame():
    detector = FaceDetector(backend="yunet")
    assert detector.detect(np.zeros((267, 400, 3), np.uint8)) == []
