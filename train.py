# pyre-unsafe
"""
train.py — Training pipeline for the face mask classifier

Phase 1: train the head on a frozen backbone (a few epochs, higher LR)
Phase 2: fine-tune the whole backbone (BatchNorm frozen) with AdamW,
         cosine-decayed LR, label smoothing and class-rebalanced batches.
The checkpoint with the best validation macro-F1 is kept, so the rare
'mask_weared_incorrect' class counts as much as the common ones.
"""

import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
from sklearn.preprocessing import LabelEncoder

from model import build_model, unfreeze_top_layers
from data_pipeline import make_train_dataset, make_eval_dataset
from dataset_utils import validate_dataset, print_dataset_summary, create_sample_dataset
from config import (
    DATASET_DIR, MODEL_DIR, CLASSES, BATCH_SIZE, INITIAL_LR,
    HEAD_EPOCHS, EPOCHS, FINE_TUNE_LR, WARMUP_EPOCHS, WEIGHT_DECAY,
    LABEL_SMOOTHING, EARLY_STOP_PATIENCE, BACKBONE, INPUT_SIZE,
    MASK_MODEL_PATH, LABEL_ENCODER_PATH, MODEL_INFO_PATH, CROP_MARGIN, SPLIT_SEED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train")


def compile_model(model, learning_rate, weight_decay=WEIGHT_DECAY,
                  label_smoothing=LABEL_SMOOTHING):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate,
                                         weight_decay=weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.F1Score(average="macro", name="macro_f1"),
        ],
    )


def cosine_schedule(peak_lr, total_steps, warmup_steps):
    """Linear warmup to peak_lr, then cosine decay to ~0."""
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=peak_lr * 0.01,
        warmup_target=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.01,
    )


def get_callbacks(weights_path, log_path, append_log, patience=EARLY_STOP_PATIENCE):
    """Keep the best-val-macro-F1 weights; stop when it stops improving."""
    return [
        keras.callbacks.ModelCheckpoint(
            weights_path, monitor="val_macro_f1", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path, append=append_log),
    ]


def plot_history(history, phase_boundary=None, output_dir=MODEL_DIR):
    """Plot and save accuracy / macro-F1 / loss curves for the whole run."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training ({BACKBONE}, {INPUT_SIZE[0]}px)", fontsize=14, fontweight='bold')
    panels = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("loss", "Loss")]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(history[key], label=f"Train {title}", color="#4CAF50", linewidth=2)
        ax.plot(history[f"val_{key}"], label=f"Val {title}", color="#2196F3", linewidth=2)
        if phase_boundary:
            ax.axvline(phase_boundary - 0.5, color="#999", linestyle="--", linewidth=1,
                       label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Training plot saved: {plot_path}")
    return plot_path


def save_label_encoder(classes=CLASSES, path=LABEL_ENCODER_PATH):
    """Save a label encoder (class index map) for inference."""
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved: {path}  | classes: {le.classes_.tolist()}")
    return le


def merge_histories(*histories):
    merged = {}
    for h in histories:
        if h is None:
            continue
        for k, v in h.history.items():
            if k == "learning_rate":
                continue
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


def train(args):
    """Main training entry point."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir = os.path.join(args.dataset_dir, "val")

    # ── Dataset check
    valid, _ = validate_dataset(train_dir, min_per_class=5)
    if not valid:
        if args.auto_sample:
            logger.info("Generating sample dataset for testing...")
            create_sample_dataset(n_per_class=50)
        else:
            logger.error(
                "Dataset insufficient! Run: python download_dataset.py\n"
                "(or --auto-sample for a synthetic smoke-test dataset)."
            )
            sys.exit(1)
    print_dataset_summary(args.dataset_dir)

    # ── Data
    train_ds, steps, counts = make_train_dataset(train_dir, args.batch_size)
    val_ds, _, _ = make_eval_dataset(val_dir, args.batch_size)
    logger.info(f"Train counts {dict(zip(CLASSES, counts.tolist()))} | {steps} steps/epoch")

    # ── Model
    model = build_model(backbone=args.backbone, freeze_base=True)
    save_label_encoder()

    weights_path = os.path.join(MODEL_DIR, "checkpoints", "best.weights.h5")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    log_path = os.path.join(MODEL_DIR, "training_log.csv")

    # ── Phase 1: head only
    logger.info(f"\n{'='*60}\nPHASE 1: training the head ({args.head_epochs} epochs, backbone frozen)\n{'='*60}")
    compile_model(model, args.lr)
    h1 = model.fit(
        train_ds, steps_per_epoch=steps, epochs=args.head_epochs,
        validation_data=val_ds,
        callbacks=[keras.callbacks.CSVLogger(log_path, append=False)],
        verbose=2,
    )

    # ── Phase 2: fine-tune the whole backbone
    h2 = None
    if args.epochs > 0:
        logger.info(f"\n{'='*60}\nPHASE 2: fine-tuning {args.backbone} ({args.epochs} epochs)\n{'='*60}")
        unfreeze_top_layers(model)
        compile_model(model, cosine_schedule(args.fine_tune_lr, steps * args.epochs,
                                             steps * WARMUP_EPOCHS))
        h2 = model.fit(
            train_ds, steps_per_epoch=steps, epochs=args.epochs,
            validation_data=val_ds,
            callbacks=get_callbacks(weights_path, log_path, append_log=True),
            verbose=2,
        )
        if os.path.exists(weights_path):
            model.load_weights(weights_path)   # best val macro-F1 epoch

    history = merge_histories(h1, h2)
    plot_history(history, phase_boundary=args.head_epochs if h2 else None)

    # ── Save a slim inference model (no optimizer state) with the best weights
    final = build_model(backbone=args.backbone, freeze_base=True)
    final.set_weights(model.get_weights())
    final.save(MASK_MODEL_PATH)
    logger.info(f"✅ Model saved to: {MASK_MODEL_PATH}")

    # ── Final validation metrics with the restored best weights
    compile_model(final, 1e-4)
    val_loss, val_acc, val_f1 = final.evaluate(val_ds, verbose=0)
    info = {
        "backbone": args.backbone,
        "input_size": list(INPUT_SIZE),
        "classes": CLASSES,
        "crop_margin": CROP_MARGIN,
        "train_counts": dict(zip(CLASSES, counts.tolist())),
        "epochs_run": len(history.get("loss", [])),
        "val_accuracy": round(float(val_acc), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "seed": args.seed,
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"  Model saved:   {MASK_MODEL_PATH}")
    print(f"  Val accuracy:  {val_acc*100:.2f}%")
    print(f"  Val macro-F1:  {val_f1:.4f}")
    print(f"  Plots saved:   {os.path.join(MODEL_DIR, 'plots')}/")
    print("  Next: python evaluate.py   (scores the held-out test split)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face mask detection model")
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help="Prepared dataset root (with train/ val/ test/)")
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Fine-tuning epochs (0 = train the head only)")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, help="Head learning rate")
    parser.add_argument("--fine-tune-lr", type=float, default=FINE_TUNE_LR,
                        help="Peak learning rate while fine-tuning")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--auto-sample", action="store_true",
                        help="Auto-generate sample dataset if insufficient data")
    args = parser.parse_args()

    train(args)
