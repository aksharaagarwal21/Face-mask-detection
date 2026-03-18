# pyre-unsafe
"""
model.py — MobileNetV2-based face mask classifier (3-class)
Classes: with_mask | without_mask | mask_weared_incorrect
"""

import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    AveragePooling2D, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from config import INPUT_SIZE, CLASSES, MODEL_DIR


def build_model(num_classes=3, freeze_base=True):
    """
    Build and return the MobileNetV2-based mask classifier.

    Architecture:
        MobileNetV2 (imagenet weights, top removed)
        → GlobalAveragePooling2D
        → Dense(256, relu) + BatchNorm + Dropout(0.5)
        → Dense(128, relu) + BatchNorm + Dropout(0.3)
        → Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes (default 3)
        freeze_base: Whether to freeze MobileNetV2 base layers

    Returns:
        Compiled Keras Model
    """
    # Base model
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    )

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def unfreeze_top_layers(model, num_layers=30):
    """
    Unfreeze the top N layers of the MobileNetV2 base for fine-tuning.
    """
    base = model.layers[0] if hasattr(model.layers[0], 'layers') else model
    # Find MobileNetV2 base layer
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # It's the MobileNetV2 sub-model
            for sub_layer in layer.layers[-num_layers:]:
                sub_layer.trainable = True
            break


def load_trained_model(path=None):
    """Load a saved mask detector model from disk."""
    from tensorflow.keras.models import load_model
    path = path or os.path.join(MODEL_DIR, "mask_detector.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return load_model(path)


def get_model_summary(model):
    """Print model summary."""
    model.summary()
    total_params = model.count_params()
    trainable_params = sum([
        np.prod(v.shape) for v in model.trainable_weights
    ])
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    print("Building model...")
    model = build_model()
    get_model_summary(model)
    print("\n✅ Model built successfully!")
