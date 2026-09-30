# pyre-unsafe
"""
data_pipeline.py — tf.data input pipeline + augmentation for the mask classifier

Images are decoded and resized once, cached as uint8, then augmented on the
fly. Rare classes are oversampled (sampling ∝ count**BALANCE_POWER) instead
of relying on large class weights, which were unstable with only ~100
'mask_weared_incorrect' examples.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from config import (
    CLASSES, INPUT_SIZE, BATCH_SIZE, SPLIT_SEED, AUGMENTATION, BALANCE_POWER
)

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_split(split_dir):
    """Return (paths, labels) for a split folder that has one sub-folder per class."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(cls_dir, name))
                labels.append(idx)
    return np.array(paths), np.array(labels, dtype="int32")


def _decode(path, input_size):
    """Read an image file → RGB uint8 at input_size."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, input_size, method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _per_image_augment(img, params, input_size):
    """Augmentations that need a per-image random size or branch."""
    img = tf.cast(img, tf.float32)

    def low_res():
        lo, hi = params["low_res_range"]
        side = tf.random.uniform([], lo, hi + 1, dtype=tf.int32)
        small = tf.image.resize(img, tf.stack([side, side]), method="area")
        return tf.image.resize(small, input_size, method="bilinear")

    img = tf.cond(tf.random.uniform([]) < params["low_res_prob"], low_res, lambda: img)
    img = tf.cond(
        tf.random.uniform([]) < params["grayscale_prob"],
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img)),
        lambda: img,
    )
    return img


def build_augmenter(params=AUGMENTATION, seed=SPLIT_SEED):
    """Batch-level geometric/photometric augmentation (per-sample random params)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(params["rotation"], fill_mode="reflect", seed=seed),
        layers.RandomZoom(params["zoom"], fill_mode="reflect", seed=seed),
        layers.RandomTranslation(params["shift"], params["shift"], fill_mode="reflect", seed=seed),
        layers.RandomBrightness(params["brightness"], value_range=(0, 255), seed=seed),
        layers.RandomContrast(params["contrast"], seed=seed),
    ], name="augment")


def make_train_dataset(split_dir, batch_size=BATCH_SIZE, input_size=INPUT_SIZE,
                       balance_power=BALANCE_POWER, params=AUGMENTATION, seed=SPLIT_SEED):
    """
    Infinite, class-rebalanced, augmented training stream.

    Returns:
        (dataset, steps_per_epoch, class_counts)
    """
    paths, labels = list_split(split_dir)
    counts = np.bincount(labels, minlength=len(CLASSES))
    if len(paths) == 0:
        raise ValueError(f"No training images found under {split_dir}")

    per_class, weights = [], []
    for c, n in enumerate(counts):
        if n == 0:
            continue
        ds = tf.data.Dataset.from_tensor_slices((paths[labels == c], labels[labels == c]))
        ds = ds.map(lambda p, y: (_decode(p, input_size), y), num_parallel_calls=AUTOTUNE)
        ds = ds.cache().shuffle(int(n), seed=seed, reshuffle_each_iteration=True).repeat()
        per_class.append(ds)
        weights.append(float(n) ** balance_power)
    weights = np.array(weights) / np.sum(weights)

    augmenter = build_augmenter(params, seed)
    ds = tf.data.Dataset.sample_from_datasets(per_class, weights=weights.tolist(), seed=seed)
    ds = ds.map(lambda x, y: (_per_image_augment(x, params, input_size), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0),
                              tf.one_hot(y, len(CLASSES))),
                num_parallel_calls=AUTOTUNE)
    steps_per_epoch = max(1, int(np.ceil(len(paths) / batch_size)))
    return ds.prefetch(AUTOTUNE), steps_per_epoch, counts


def make_eval_dataset(split_dir, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, one_hot=True):
    """
    Deterministic, un-augmented dataset for validation / testing.

    Returns:
        (dataset, labels, paths)  — labels/paths in dataset order
    """
    paths, labels = list_split(split_dir)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (tf.cast(_decode(p, input_size), tf.float32),
                              tf.one_hot(y, len(CLASSES)) if one_hot else y),
                num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).cache().prefetch(AUTOTUNE), labels, paths


def save_augmentation_preview(split_dir, out_path, n=32, input_size=INPUT_SIZE):
    """Write a grid of augmented training samples (sanity check for the pipeline)."""
    import cv2
    ds, _, _ = make_train_dataset(split_dir, batch_size=n, input_size=input_size)
    images, onehot = next(iter(ds))
    images = images.numpy().astype("uint8")
    cols = 8
    rows = int(np.ceil(n / cols))
    h, w = input_size
    grid = np.zeros((rows * h, cols * w, 3), dtype="uint8")
    for i, (img, y) in enumerate(zip(images, onehot.numpy())):
        r, c = divmod(i, cols)
        tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, CLASSES[int(np.argmax(y))][:12], (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path


if __name__ == "__main__":
    import argparse
    from config import TRAIN_DIR, MODEL_DIR

    parser = argparse.ArgumentParser(description="Preview the augmented training stream")
    parser.add_argument("--split-dir", default=TRAIN_DIR)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "plots", "augmentation_preview.jpg"))
    args = parser.parse_args()
    print(f"Saved preview: {save_augmentation_preview(args.split_dir, args.out)}")
