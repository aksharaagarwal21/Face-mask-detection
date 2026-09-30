# pyre-unsafe
"""
evaluate.py — Model evaluation on the held-out test split

Outputs (models/plots/ and models/):
    - classification report (printed)
    - confusion_matrix.png   (row-normalised, raw counts annotated)
    - roc_curves.png         (one-vs-rest + macro AUC)
    - test_errors.jpg        (every misclassified test face, for error analysis)
    - test_metrics.json      (accuracy, macro-F1, balanced accuracy, per-class metrics)

By default predictions average the image and its mirror (flip TTA) and are
temperature-scaled with the value calibrate.py stored, which is what
MaskDetector does at runtime. Use --no-tta / --no-calibration to turn either off.
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
    f1_score, balanced_accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

from model import load_trained_model
from calibrate import apply_temperature, load_temperature, expected_calibration_error
from data_pipeline import make_eval_dataset
from config import (
    TEST_DIR, MODEL_DIR, CLASSES, MASK_MODEL_PATH, BATCH_SIZE, METRICS_PATH
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluate")


def predict_probs(model, dataset, tta=True):
    """Class probabilities for a (x, y) dataset; averages with the mirrored image if tta."""
    images = dataset.map(lambda x, y: x)
    probs = model.predict(images, verbose=0)
    if tta:
        flipped = images.map(tf.image.flip_left_right)
        probs = (probs + model.predict(flipped, verbose=0)) / 2.0
    return probs


def evaluate_model(model_path=MASK_MODEL_PATH, dataset_dir=TEST_DIR, output_dir=None,
                   tta=True, metrics_path=METRICS_PATH, calibrated=True):
    """Full evaluation pipeline for the trained model. Returns the metrics dict."""
    output_dir = output_dir or os.path.join(MODEL_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model: {model_path}")
    model = load_trained_model(model_path)
    input_size = tuple(model.input_shape[1:3])

    logger.info(f"Loading test split: {dataset_dir}")
    test_ds, y_true, paths = make_eval_dataset(dataset_dir, BATCH_SIZE, input_size,
                                               one_hot=False)
    if len(y_true) == 0:
        raise SystemExit(f"No test images found under {dataset_dir}. Run download_dataset.py first.")

    logger.info(f"Predicting {len(y_true)} faces (flip TTA: {'on' if tta else 'off'})...")
    temperature = load_temperature() if calibrated else 1.0
    y_prob = apply_temperature(predict_probs(model, test_ds, tta=tta), temperature)
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = len(CLASSES)

    # ── Classification report
    print("\n" + "="*65)
    print("📊 CLASSIFICATION REPORT (held-out test split)")
    print("="*65)
    print(classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                target_names=CLASSES, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    _plot_confusion_matrix(cm, CLASSES, output_dir)

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    auc_scores = _plot_roc_curves(y_true_bin, y_prob, CLASSES, n_classes, output_dir)
    _save_error_gallery(paths, y_true, y_pred, y_prob, output_dir)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    metrics = {
        "split": os.path.relpath(dataset_dir, os.path.dirname(MODEL_DIR)).replace("\\", "/"),
        "n_samples": int(len(y_true)),
        "tta": tta,
        "temperature": temperature,
        "accuracy": round(float(np.mean(y_pred == y_true)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_auc": round(float(auc_scores.get("macro", float("nan"))), 4),
        "ece": round(expected_calibration_error(y_prob, y_true), 4),
        "per_class": {
            cls: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                  "f1": round(float(f1[i]), 4), "support": int(support[i])}
            for i, cls in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }
    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")

    print(f"\nAccuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Calibration (ECE):  {metrics['ece']:.4f}  (temperature {temperature:.3f})")
    print(f"Plots saved to:     {output_dir}/")
    return metrics


def _plot_confusion_matrix(cm, class_names, output_dir):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        cm_norm,
        annot=cm,               # Show raw counts
        fmt="d",
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={"label": "Fraction of true class"}
    )
    plt.title("Confusion Matrix (test split)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def _plot_roc_curves(y_true_bin, y_prob, class_names, n_classes, output_dir):
    """Plot one-vs-rest ROC curves for each class + macro-average."""
    colors = ["#FF9800", "#4CAF50", "#F44336"]
    plt.figure(figsize=(9, 6))

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    try:
        auc_scores["macro"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        plt.plot([], [], ' ', label=f"Macro AUC = {auc_scores['macro']:.3f}")
    except ValueError:
        pass

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curves (One-vs-Rest, test split)", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved: {path}")
    return auc_scores


def _save_error_gallery(paths, y_true, y_pred, y_prob, output_dir, tile=96, cols=8):
    """Grid of misclassified test faces labelled 'true -> predicted (conf)'."""
    import cv2
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return None
    short = {"mask_weared_incorrect": "incorrect", "with_mask": "mask", "without_mask": "no mask"}
    rows = int(np.ceil(len(wrong) / cols))
    grid = np.full((rows * (tile + 28), cols * tile, 3), 30, dtype="uint8")
    for k, i in enumerate(wrong):
        img = cv2.imread(str(paths[i]))
        if img is None:
            continue
        r, c = divmod(k, cols)
        y0, x0 = r * (tile + 28), c * tile
        grid[y0:y0 + tile, x0:x0 + tile] = cv2.resize(img, (tile, tile))
        cv2.putText(grid, short[CLASSES[y_true[i]]], (x0 + 3, y0 + tile + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(grid, f"> {short[CLASSES[y_pred[i]]]} {y_prob[i].max():.2f}",
                    (x0 + 3, y0 + tile + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (140, 170, 255), 1, cv2.LINE_AA)
    path = os.path.join(output_dir, "test_errors.jpg")
    cv2.imwrite(path, grid)
    logger.info(f"Error gallery ({len(wrong)} faces) saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained mask detector model")
    parser.add_argument("--model", default=MASK_MODEL_PATH,
                        help="Path to the .keras model file")
    parser.add_argument("--dataset", default=TEST_DIR,
                        help="Held-out test directory (one sub-folder per class)")
    parser.add_argument("--output-dir", default=os.path.join(MODEL_DIR, "plots"),
                        help="Directory to save plots")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable flip test-time augmentation")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Score raw softmax outputs (ignore the stored temperature)")
    args = parser.parse_args()

    results = evaluate_model(args.model, args.dataset, args.output_dir, tta=not args.no_tta,
                             calibrated=not args.no_calibration)
    print(f"\n✅ Evaluation complete | Accuracy: {results['accuracy']*100:.2f}% "
          f"| Macro F1: {results['macro_f1']:.4f}")
