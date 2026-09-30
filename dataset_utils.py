# pyre-unsafe
"""
dataset_utils.py — Dataset download, preparation, splitting, and augmentation utilities
"""

import os
import shutil
import random
import zipfile
import logging
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from config import DATASET_DIR, TRAIN_DIR, CLASSES

SPLITS = ("train", "val", "test")

logger = logging.getLogger("DatasetUtils")

# Kaggle-compatible dataset (Face Mask Detection)
DATASET_SOURCES = {
    "sample_description": (
        "Prepare the real dataset (downloads from Kaggle, crops faces, splits by photo):\n"
        "  python download_dataset.py\n\n"
        "This creates dataset/{train,val,test}/{with_mask,without_mask,mask_weared_incorrect}/\n"
        "Source: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection\n\n"
        "Or, for a quick smoke test only: python dataset_utils.py --generate-sample"
    )
}


def create_sample_dataset(n_per_class=30):
    """
    Generate synthetic sample images for quick testing.
    Creates solid-color images with patterns — not for real training!
    """
    logger.info(f"Generating synthetic sample dataset ({n_per_class} per class)...")

    colors = {
        "with_mask": [(0, 180, 0), (0, 150, 50), (50, 200, 100)],
        "without_mask": [(200, 0, 0), (180, 50, 50), (220, 0, 30)],
        "mask_weared_incorrect": [(200, 100, 0), (180, 120, 50), (220, 90, 0)],
    }

    def split_of(i):
        return "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"

    for cls in CLASSES:
        for split in SPLITS:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)
        cls_colors = colors.get(cls, [(128, 128, 128)])

        for i in range(n_per_class):
            img = Image.new("RGB", (224, 224),
                            color=cls_colors[i % len(cls_colors)])

            # Add noise to make more varied
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 20, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            # Add slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            # Brightness variation
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

            path = os.path.join(DATASET_DIR, split_of(i), cls, f"{cls}_{i:04d}.jpg")
            img.save(path, "JPEG", quality=90)

        logger.info(f"  ✅ {cls}: {n_per_class} images")

    logger.info(f"Sample dataset created at: {DATASET_DIR}")
    return DATASET_DIR


def get_class_distribution(dataset_dir=TRAIN_DIR):
    """Return dict of {class_name: file_count}."""
    dist = {}
    for cls in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            dist[cls] = len(files)
        else:
            dist[cls] = 0
    return dist


def validate_dataset(dataset_dir=TRAIN_DIR, min_per_class=5):
    """Check dataset structure and minimum counts. Returns True if valid."""
    dist = get_class_distribution(dataset_dir)
    valid = True
    for cls, count in dist.items():
        if count < min_per_class:
            logger.warning(f"Class '{cls}' has only {count} images (min: {min_per_class})")
            valid = False
        else:
            logger.info(f"  {cls}: {count} images ✅")
    return valid, dist


def compute_class_weights(dist: dict):
    """Compute per-class weights to handle class imbalance."""
    total = sum(dist.values())
    n_classes = len(dist)
    weights = {}
    for i, (cls, count) in enumerate(dist.items()):
        if count > 0:
            weights[i] = total / (n_classes * count)
        else:
            weights[i] = 1.0
    logger.info(f"Class weights: {weights}")
    return weights


def print_dataset_summary(dataset_root=DATASET_DIR):
    """Print per-split class counts of the prepared dataset."""
    dists = {split: get_class_distribution(os.path.join(dataset_root, split)) for split in SPLITS}
    valid, _ = validate_dataset(os.path.join(dataset_root, "train"))
    print("\n" + "="*62)
    print("📊 DATASET SUMMARY")
    print("="*62)
    print(f"  {'class':<26}" + "".join(f"{s:>10}" for s in SPLITS))
    for cls in CLASSES:
        print(f"  {cls:<26}" + "".join(f"{dists[s][cls]:>10}" for s in SPLITS))
    print(f"  {'TOTAL':<26}" + "".join(f"{sum(dists[s].values()):>10}" for s in SPLITS))
    print("="*62)
    print(f"Status: {'✅ Valid' if valid else '❌ Insufficient data'}\n")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate synthetic sample dataset for testing")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of images per class for sample generation")
    parser.add_argument("--summary", action="store_true",
                        help="Print dataset summary")
    args = parser.parse_args()

    if args.generate_sample:
        create_sample_dataset(n_per_class=args.n)

    if args.summary or not args.generate_sample:
        print_dataset_summary()
        print(DATASET_SOURCES["sample_description"])
