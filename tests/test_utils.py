import numpy as np

from utils import square_box, crop_face, cv_text, compute_compliance_pct, draw_detection_box


def test_square_box_is_square_and_centred():
    x1, y1, x2, y2 = square_box((10, 20, 30, 60), margin=0.0)
    assert x2 - x1 == y2 - y1 == 40
    assert (x1 + x2) / 2 == 20 and (y1 + y2) / 2 == 40


def test_square_box_margin_grows_each_side():
    x1, y1, x2, y2 = square_box((0, 0, 100, 100), margin=0.15)
    assert x2 - x1 == 130


def test_crop_face_is_square_even_at_the_border(frame):
    crop = crop_face(frame, (0, 0, 40, 60))          # sticks out top-left
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[2] == 3


def test_crop_face_inside_image_matches_slice(frame):
    crop = crop_face(frame, (100, 100, 140, 140), margin=0.0)
    np.testing.assert_array_equal(crop, frame[100:140, 100:140])


def test_cv_text_drops_glyphs_opencv_cannot_draw():
    assert cv_text("⚠  MASK VIOLATION — 2 person(s)") == "MASK VIOLATION - 2 person(s)"
    assert cv_text("with_mask: 99.0%") == "with_mask: 99.0%"


def test_compliance_pct():
    assert compute_compliance_pct({"total": 0}) == 0.0
    assert compute_compliance_pct({"total": 4, "with_mask": 3}) == 75.0


def test_label_stays_inside_frame():
    img = np.zeros((100, 200, 3), np.uint8)
    draw_detection_box(img, 180, 40, 199, 70, "without_mask", 0.99, (0, 0, 255))
    # the label is shifted left instead of being cut off at the right edge
    lit_cols = np.where(img[:40].any(axis=(0, 2)))[0]
    assert lit_cols.min() < 120 and lit_cols.max() <= 199
