import io
import json

import cv2
import numpy as np
import pytest

from conftest import FakeFaceDetector, FakeMaskDetector, probs_for
from pipeline import MaskPipeline, decode_image


@pytest.fixture
def pipeline():
    boxes = [(10, 10, 60, 60, 0.95), (100, 20, 150, 70, 0.9), (200, 30, 250, 80, 0.8)]
    probs = [probs_for("with_mask", 0.98), probs_for("without_mask", 0.95),
             probs_for("mask_weared_incorrect", 0.5)]
    return MaskPipeline(FakeFaceDetector(boxes), FakeMaskDetector(probs))


def test_analyze_counts_and_compliance(pipeline, frame):
    r = pipeline.analyze(frame)
    assert r["counts"] == {"total": 3, "with_mask": 1, "without_mask": 1, "mask_weared_incorrect": 1}
    assert r["compliance_pct"] == pytest.approx(33.3)
    assert r["image_size"] == [320, 240]
    json.dumps(r)                                   # must be JSON-serialisable


def test_violation_needs_a_confident_violation_class(pipeline, frame):
    faces = pipeline.analyze(frame)["faces"]
    assert [f["violation"] for f in faces] == [False, True, False]   # 0.5 < threshold 0.6


def test_annotate_leaves_the_input_untouched(pipeline, frame):
    before = frame.copy()
    out = pipeline.annotate(frame, pipeline.analyze(frame))
    np.testing.assert_array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_decode_image(frame):
    ok, buf = cv2.imencode(".png", frame)
    np.testing.assert_array_equal(decode_image(buf.tobytes()), frame)
    assert decode_image(b"not an image") is None
    assert decode_image(b"") is None


@pytest.fixture
def client(pipeline, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "_pipeline", pipeline)
    return appmod.app.test_client()


def test_api_predict_multipart_and_raw_body(client, frame):
    data = cv2.imencode(".jpg", frame)[1].tobytes()
    r = client.post("/api/predict", data={"image": (io.BytesIO(data), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["counts"]["total"] == 3
    r = client.post("/api/predict", data=data, content_type="image/jpeg")
    assert r.status_code == 200 and "inference_ms" in r.get_json()


def test_api_predict_annotated_jpeg(client, frame):
    r = client.post("/api/predict?annotate=1", data=cv2.imencode(".png", frame)[1].tobytes())
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    assert decode_image(r.data).shape == frame.shape


def test_api_predict_rejects_bad_input(client):
    assert client.post("/api/predict", data=b"hello").status_code == 400
    from config import MAX_UPLOAD_MB
    big = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1)
    assert client.post("/api/predict", data=big).status_code == 413


def test_api_model(client):
    r = client.get("/api/model")
    assert r.status_code == 200
    assert {"backbone", "test", "end_to_end", "temperature"} <= set(r.get_json())
