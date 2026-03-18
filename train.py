# pyre-unsafe
"""
train.py — Full training pipeline for face mask detection model
Trains MobileNetV2-based classifier with augmentation, callbacks, and fine-tuning.
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight as sklearn_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from model import build_model, unfreeze_top_layers
from dataset_utils import (
    validate_dataset, get_class_distribution,
    compute_class_weights, print_dataset_summary, create_sample_dataset
)
from config import (
    DATASET_DIR, MODEL_DIR, CLASSES, BATCH_SIZE, INITIAL_LR,
    EPOCHS, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
    MASK_MODEL_PATH, LABEL_ENCODER_PATH, INPUT_SIZE, AUGMENTATION_PARAMS
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train")


def build_generators(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE):
    """Build train/val/test ImageDataGenerators."""
    # Training augmentation
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        **AUGMENTATION_PARAMS
    )

    # Val/test — only normalization
    val_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_gen = train_aug.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_gen = val_aug.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen


def get_callbacks(model_path=MASK_MODEL_PATH, log_dir=MODEL_DIR):
    """Build training callbacks."""
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            os.path.join(log_dir, "training_log.csv"),
            append=False
        )
    ]
    return callbacks


def plot_history(H, phase="initial", output_dir=MODEL_DIR):
    """Plot and save training accuracy and loss curves."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training ({phase})", fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(H.history["accuracy"], label="Train Acc", color="#4CAF50", linewidth=2)
    axes[0].plot(H.history["val_accuracy"], label="Val Acc", color="#2196F3", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(H.history["loss"], label="Train Loss", color="#F44336", linewidth=2)
    axes[1].plot(H.history["val_loss"], label="Val Loss", color="#FF9800", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"training_curves_{phase}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Training plot saved: {plot_path}")
    return plot_path


def save_label_encoder(class_indices, path=LABEL_ENCODER_PATH):
    """Save a label encoder (class index map) for inference."""
    le = LabelEncoder()
    le.classes_ = np.array(sorted(class_indices, key=class_indices.get))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved: {path}  | classes: {le.classes_.tolist()}")
    return le


def train(args):
    """Main training entry point."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Dataset check
    print_dataset_summary(args.dataset_dir)
    valid, dist = validate_dataset(args.dataset_dir, min_per_class=5)

    if not valid:
        if args.auto_sample:
            logger.info("Generating sample dataset for testing...")
            create_sample_dataset(n_per_class=50)
        else:
            logger.error(
                "Dataset insufficient! Run: python dataset_utils.py --generate-sample\n"
                "Or use --auto-sample flag.\n"
                "For real training, download a dataset from Kaggle."
            )
            sys.exit(1)

    # ── Data generators
    logger.info("Building data generators...")
    train_gen, val_gen = build_generators(args.dataset_dir, args.batch_size)

    # ── Class weights
    class_weights = compute_class_weights(dist)

    # ── Build model
    logger.info("Building model...")
    model = build_model(num_classes=len(CLASSES), freeze_base=True)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # ── Save label encoder
    save_label_encoder(train_gen.class_indices)

    # ── Phase 1: Train with frozen base
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: Training classification head (base frozen)")
    logger.info(f"{'='*60}")

    callbacks = get_callbacks(MASK_MODEL_PATH)

    H1 = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    plot_history(H1, phase="phase1")

    # ── Phase 2: Fine-tune top layers
    if args.fine_tune:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2: Fine-tuning top MobileNetV2 layers")
        logger.info(f"{'='*60}")

        unfreeze_top_layers(model, num_layers=30)
        model.compile(
            optimizer=Adam(learning_rate=args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        H2 = model.fit(
            train_gen,
            epochs=args.fine_tune_epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=get_callbacks(MASK_MODEL_PATH),
            verbose=1
        )
        plot_history(H2, phase="phase2_finetune")

    # ── Save final model
    model.save(MASK_MODEL_PATH)
    logger.info(f"\n✅ Model saved to: {MASK_MODEL_PATH}")

    # ── Print final metrics
    logger.info("\n📊 Final validation metrics:")
    val_gen.reset()
    results = model.evaluate(val_gen, verbose=1)
    logger.info(f"Val Loss: {results[0]:.4f}  |  Val Accuracy: {results[1]*100:.2f}%")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"  Model saved:  {MASK_MODEL_PATH}")
    print(f"  Val Accuracy: {results[1]*100:.2f}%")
    print(f"  Plots saved:  {os.path.join(MODEL_DIR, 'plots')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face mask detection model")
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=INITIAL_LR)
    parser.add_argument("--fine-tune", action="store_true", default=True,
                        help="Enable fine-tuning phase 2")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--fine-tune-lr", type=float, default=FINE_TUNE_LR)
    parser.add_argument("--auto-sample", action="store_true",
                        help="Auto-generate sample dataset if insufficient data")
    args = parser.parse_args()

    train(args)
