# pyre-unsafe
"""
model.py — Face mask classifier (3-class) on a pretrained ImageNet backbone
Classes: with_mask | without_mask | mask_weared_incorrect

The model takes raw RGB pixels in [0, 255]; the backbone-specific scaling is
part of the graph, so inference code never has to know which backbone it is.
"""

import os
import numpy as np
import keras
from keras import layers
from keras.models import Model
from config import INPUT_SIZE, CLASSES, MODEL_DIR, BACKBONE, MASK_MODEL_PATH

BACKBONES = {
    # name: (constructor, has built-in preprocessing, input rescale if not)
    "efficientnetv2b0": (keras.applications.EfficientNetV2B0, True, None),
    "efficientnetv2b1": (keras.applications.EfficientNetV2B1, True, None),
    "mobilenetv2": (keras.applications.MobileNetV2, False, (1 / 127.5, -1.0)),
}


def build_model(num_classes=len(CLASSES), backbone=BACKBONE, freeze_base=True,
                input_size=INPUT_SIZE):
    """
    Build the mask classifier.

    Architecture:
        backbone (ImageNet weights, top removed, called with training=False)
        → GlobalAveragePooling2D
        → Dropout(0.3) → Dense(256, swish) → Dropout(0.3)
        → Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes (default 3)
        backbone: One of BACKBONES
        freeze_base: Whether to freeze the backbone (phase 1: head only)
        input_size: (height, width) of the input crops

    Returns:
        Uncompiled Keras Model
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Options: {list(BACKBONES)}")
    ctor, builtin_preprocessing, rescale = BACKBONES[backbone]
    shape = (input_size[0], input_size[1], 3)

    inputs = keras.Input(shape=shape, name="rgb_0_255")
    x = inputs
    if rescale is not None:
        x = layers.Rescaling(scale=rescale[0], offset=rescale[1], name="rescale")(x)

    kwargs = {"include_preprocessing": True} if builtin_preprocessing else {}
    base_model = ctor(weights="imagenet", include_top=False, input_shape=shape, **kwargs)
    base_model.trainable = not freeze_base

    # training=False keeps BatchNorm statistics fixed, even while fine-tuning
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="swish", name="head_dense")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"mask_classifier_{backbone}")


def get_backbone(model):
    """Return the nested backbone sub-model."""
    return next(layer for layer in model.layers if isinstance(layer, Model))


def unfreeze_top_layers(model, num_layers=None):
    """
    Unfreeze the top `num_layers` layers of the backbone for fine-tuning
    (all of them when num_layers is None). BatchNorm layers stay frozen so
    small batches don't wreck their statistics.
    """
    base = get_backbone(model)
    base.trainable = True
    first = 0 if num_layers is None else max(0, len(base.layers) - num_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= first and not isinstance(layer, layers.BatchNormalization)


def load_trained_model(path=MASK_MODEL_PATH):
    """Load a saved mask detector model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return keras.models.load_model(path, compile=False)


def get_model_summary(model):
    """Print model summary."""
    model.summary()
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    print("Building model...")
    model = build_model()
    get_model_summary(model)
    print("\n✅ Model built successfully!")
