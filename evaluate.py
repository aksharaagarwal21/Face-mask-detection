# pyre-unsafe
"""
evaluate.py — Model evaluation: classification report, confusion matrix, ROC curves
"""

import os
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from config import (
    TEST_DIR, MODEL_DIR, CLASSES, MASK_MODEL_PATH, INPUT_SIZE, BATCH_SIZE
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluate")


def load_test_generator(dataset_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """Load the held-out test split (no augmentation)."""
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = gen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return test_gen


def evaluate_model(model_path=MASK_MODEL_PATH, dataset_dir=TEST_DIR, output_dir=None):
    """
    Full evaluation pipeline for the trained model.

    Outputs:
        - Classification report (printed)
        - Confusion matrix (saved as PNG)
        - ROC curves per class (saved as PNG)
        - Summary metrics JSON
    """
    output_dir = output_dir or os.path.join(MODEL_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model
    logger.info(f"Loading model: {model_path}")
    model = load_model(model_path)

    # ── Load test data
    logger.info("Loading dataset...")
    test_gen = load_test_generator(dataset_dir)
    class_indices = test_gen.class_indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    n_classes = len(class_names)

    # ── Predictions
    logger.info("Running predictions...")
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes[:len(y_pred)]

    # ── Classification Report
    print("\n" + "="*65)
    print("📊 CLASSIFICATION REPORT")
    print("="*65)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # ── Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── ROC Curves
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    _plot_roc_curves(y_true_bin, y_prob, class_names, n_classes, output_dir)

    # ── Per-class metrics
    accuracy = np.mean(y_pred == y_true)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"Plots saved to:   {output_dir}/")

    return {
        "accuracy": float(accuracy),
        "n_samples": len(y_true),
        "class_names": class_names,
    }


def _plot_confusion_matrix(cm, class_names, output_dir):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=cm,               # Show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={"label": "Normalized Fraction"}
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
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
    colors = ["#4CAF50", "#F44336", "#FF9800"]
    plt.figure(figsize=(9, 6))

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_true_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{cls} (AUC = {roc_auc:.3f})")

    # Macro-average
    try:
        macro_auc = roc_auc_score(y_true_bin, y_prob,
                                   multi_class="ovr", average="macro")
        plt.plot([], [], ' ', label=f"Macro AUC = {macro_auc:.3f}")
    except Exception:
        pass

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved: {path}")
    return auc_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained mask detector model")
    parser.add_argument("--model", default=MASK_MODEL_PATH,
                        help="Path to .h5 model file")
    parser.add_argument("--dataset", default=TEST_DIR,
                        help="Held-out test directory (one sub-folder per class)")
    parser.add_argument("--output-dir", default=os.path.join(MODEL_DIR, "plots"),
                        help="Directory to save plots")
    args = parser.parse_args()

    results = evaluate_model(args.model, args.dataset, args.output_dir)
    print(f"\n✅ Evaluation complete | Accuracy: {results['accuracy']*100:.2f}%")
