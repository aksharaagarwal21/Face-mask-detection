import json

import numpy as np
import pytest

from calibrate import (
    apply_temperature, fit_temperature, expected_calibration_error, nll, load_temperature
)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_temperature_one_is_identity():
    p = np.array([[0.2, 0.5, 0.3]])
    np.testing.assert_allclose(apply_temperature(p, 1.0), p)


def test_temperature_keeps_argmax_and_normalisation():
    p = softmax(np.random.RandomState(0).normal(size=(50, 3)))
    for t in (0.3, 2.5):
        q = apply_temperature(p, t)
        np.testing.assert_allclose(q.sum(axis=1), 1.0)
        np.testing.assert_array_equal(q.argmax(axis=1), p.argmax(axis=1))


def test_low_temperature_sharpens():
    p = np.array([[0.2, 0.5, 0.3]])
    assert apply_temperature(p, 0.5).max() > p.max()


def test_fit_recovers_a_known_temperature():
    rng = np.random.RandomState(1)
    logits = rng.normal(scale=3.0, size=(4000, 3))
    true_p = softmax(logits)
    labels = np.array([rng.choice(3, p=row) for row in true_p])
    # the model is overconfident by a factor 2: it outputs softmax(2 * logits)
    overconfident = softmax(2.0 * logits)
    t = fit_temperature(overconfident, labels)
    assert t == pytest.approx(2.0, rel=0.1)
    assert nll(apply_temperature(overconfident, t), labels) < nll(overconfident, labels)


def test_ece_is_zero_for_perfect_calibration():
    # confidence 0.75, right 3 times out of 4
    p = np.tile([[0.75, 0.25, 0.0]], (4, 1))
    labels = np.array([0, 0, 0, 1])
    assert expected_calibration_error(p, labels) == pytest.approx(0.0, abs=1e-9)


def test_load_temperature_defaults_to_one(tmp_path):
    assert load_temperature(str(tmp_path / "missing.json")) == 1.0
    path = tmp_path / "info.json"
    path.write_text(json.dumps({"temperature": 0.4}))
    assert load_temperature(str(path)) == 0.4
