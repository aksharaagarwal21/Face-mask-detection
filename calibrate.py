# pyre-unsafe
"""
calibrate.py — Temperature scaling for the mask classifier's confidences

Training uses label smoothing, which caps how confident the softmax gets:
a face the model is sure about still scores ~0.93. Thresholds such as
MASK_CONFIDENCE_THRESHOLD and the "severe violation" cut-off only mean
something if a confidence of 0.9 is right about 90% of the time.

This fits one temperature T on the validation split (the class order and
the argmax, so accuracy, don't change):

    p_calibrated = softmax(log(p) / T)

T is stored in models/model_info.json and MaskDetector applies it. The
script also reports expected calibration error (ECE) and negative
log-likelihood before and after, on val and on the held-out test split,
and draws a reliability diagram.

Usage:
    python calibrate.py
"""

import os
import json
import argparse
import logging
import numpy as np
from scipy.optimize import minimize_scalar

from config import MODEL_DIR, MODEL_INFO_PATH, MASK_MODEL_PATH, VAL_DIR, TEST_DIR, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Calibrate")

EPS = 1e-7


def apply_temperature(probs, temperature):
    """Rescale probability rows by a temperature (T < 1 sharpens, T > 1 softens)."""
    probs = np.asarray(probs, dtype="float64")
    if temperature is None or temperature == 1.0 or probs.size == 0:
        return probs
    logits = np.log(np.clip(probs, EPS, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, labels):
    """Mean negative log-likelihood of the true labels."""
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0))))


def expected_calibration_error(probs, labels, n_bins=15):
    """
    ECE: bin predictions by confidence, then average |accuracy - confidence|
    over bins, weighted by how many predictions fall in each.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def fit_temperature(probs, labels, bounds=(0.05, 10.0)):
    """Temperature that minimises NLL on (probs, labels)."""
    res = minimize_scalar(lambda t: nll(apply_temperature(probs, t), labels),
                          bounds=bounds, method="bounded")
    return float(res.x)


def load_temperature(info_path=MODEL_INFO_PATH):
    """Stored temperature, or 1.0 (no rescaling) if calibrate.py hasn't been run."""
    try:
        with open(info_path) as f:
            return float(json.load(f).get("temperature", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


def plot_reliability(results, path, n_bins=10):
    """Reliability diagram: accuracy vs. confidence, before and after scaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, (title, probs, labels) in zip(axes[0], results):
        conf = probs.max(axis=1)
        correct = probs.argmax(axis=1) == labels
        acc = [correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any()
               else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        counts = [int(((conf > lo) & (conf <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
        ax.bar(centers, acc, width=1.0 / n_bins, edgecolor="#1565C0", color="#90CAF9", label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for x, a, n in zip(centers, acc, counts):
            if n:
                ax.text(x, min(a, 1.0) + 0.02, str(n), ha="center", fontsize=7, color="#555")
        ax.set_title(f"{title}\nECE {expected_calibration_error(probs, labels):.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def main(args):
    from model import load_trained_model
    from data_pipeline import make_eval_dataset
    from evaluate import predict_probs

    model = load_trained_model(args.model)
    input_size = tuple(model.input_shape[1:3])

    def split_probs(split_dir):
        ds, labels, _ = make_eval_dataset(split_dir, BATCH_SIZE, input_size, one_hot=False)
        return predict_probs(model, ds, tta=not args.no_tta), labels

    val_p, val_y = split_probs(args.val_dir)
    test_p, test_y = split_probs(args.test_dir)

    t = fit_temperature(val_p, val_y)
    report = {"temperature": round(t, 4), "tta": not args.no_tta}
    for name, p, y in (("val", val_p, val_y), ("test", test_p, test_y)):
        cal = apply_temperature(p, t)
        report[name] = {
            "n": int(len(y)),
            "ece_before": round(expected_calibration_error(p, y), 4),
            "ece_after": round(expected_calibration_error(cal, y), 4),
            "nll_before": round(nll(p, y), 4),
            "nll_after": round(nll(cal, y), 4),
            "mean_confidence_before": round(float(p.max(axis=1).mean()), 4),
            "mean_confidence_after": round(float(cal.max(axis=1).mean()), 4),
            "accuracy": round(float((p.argmax(axis=1) == y).mean()), 4),
        }

    plot_path = plot_reliability(
        [("Test, before scaling", test_p, test_y),
         (f"Test, after scaling (T={t:.2f})", apply_temperature(test_p, t), test_y)],
        os.path.join(MODEL_DIR, "plots", "reliability.png"))

    if not args.dry_run:
        with open(args.info) as f:
            info = json.load(f)
        info["temperature"] = report["temperature"]
        info["calibration"] = {k: report[k] for k in ("val", "test")}
        with open(args.info, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Temperature saved to {args.info}")

    print("\n" + "=" * 60)
    print(f"TEMPERATURE SCALING   T = {t:.4f}  (fit on val, {len(val_y)} faces)")
    print("=" * 60)
    print(f"  {'split':<6}{'ECE before':>12}{'ECE after':>11}{'NLL before':>12}{'NLL after':>11}")
    for name in ("val", "test"):
        r = report[name]
        print(f"  {name:<6}{r['ece_before']:>12.4f}{r['ece_after']:>11.4f}"
              f"{r['nll_before']:>12.4f}{r['nll_after']:>11.4f}")
    print(f"  Reliability diagram: {plot_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit temperature scaling on the val split")
    parser.add_argument("--model", default=MASK_MODEL_PATH)
    parser.add_argument("--val-dir", default=VAL_DIR)
    parser.add_argument("--test-dir", default=TEST_DIR)
    parser.add_argument("--info", default=MODEL_INFO_PATH, help="model_info.json to update")
    parser.add_argument("--no-tta", action="store_true", help="Calibrate single-pass predictions")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't save T")
    main(parser.parse_args())
