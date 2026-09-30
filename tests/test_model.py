"""Tests that need TensorFlow and the trained model; skipped when either is missing."""
import glob
import os

import cv2
import numpy as np
import pytest

from config import MASK_MODEL_PATH, TEST_DIR, CLASSES

pytest.importorskip("tensorflow")
pytestmark = pytest.mark.skipif(not os.path.exists(MASK_MODEL_PATH),
                                reason="trained model not present")


@pytest.fixture(scope="module")
def detector():
    from mask_detector import MaskDetector
    return MaskDetector(runtime="keras")


def test_model_io_shapes(detector):
    assert detector.input_size == (160, 160)
    assert detector.classes == CLASSES
    probs = detector.predict_probs([np.zeros((30, 30, 3), np.uint8)] * 3)
    assert probs.shape == (3, len(CLASSES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_empty_batch(detector):
    assert detector.predict_probs([]).shape == (0, len(CLASSES))


def test_tta_output_is_mirror_invariant(detector):
    face = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    a = detector.predict_probs([face])
    b = detector.predict_probs([face[:, ::-1].copy()])
    np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.skipif(not os.path.isdir(TEST_DIR), reason="prepared dataset not present")
@pytest.mark.parametrize("cls", CLASSES)
def test_classifies_test_faces(detector, cls):
    files = sorted(glob.glob(os.path.join(TEST_DIR, cls, "*.png")))[:10]
    if not files:
        pytest.skip(f"no test images for {cls}")
    labels = [label for label, _ in detector.predict_batch([cv2.imread(f) for f in files])]
    assert np.mean([l == cls for l in labels]) >= 0.7


def test_gradcam_heatmap_shape():
    from explain import GradCAM
    from model import load_trained_model
    cam = GradCAM(load_trained_model())
    heats, probs = cam.heatmaps([np.full((50, 50, 3), 128, np.uint8)])
    assert heats[0].shape == (160, 160)
    assert 0.0 <= heats[0].min() and heats[0].max() <= 1.0
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
