# pyre-unsafe
import os
import sys
import csv
import shutil
import logging
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from config import (
    DATASET_DIR, CLASSES, CROP_MARGIN, MIN_FACE_SIZE, SPLIT_SEED,
    VAL_SPLIT, TEST_SPLIT
)
from utils import crop_face

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownloadDataset")

SPLITS = ("train", "val", "test")


def download_kaggle_dataset():
    """Download Face Mask Detection dataset from Kaggle using kagglehub."""
    logger.info("📥 Downloading Face Mask Detection dataset from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
        logger.info(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("\nManual download instructions:")
        logger.info("1. Go to: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
        logger.info("2. Click 'Download' and extract the zip")
        logger.info("3. Run: python download_dataset.py --source <extracted_folder_path>")
        sys.exit(1)


def find_dataset_dirs(source_path):
    """Find the images and annotations directories in the downloaded dataset."""
    source_path = str(source_path)
    
    # Check common structures
    possible_structures = [
        # Direct structure
        (os.path.join(source_path, "images"), os.path.join(source_path, "annotations")),
        # Nested structure
        (os.path.join(source_path, "face-mask-detection", "images"),
         os.path.join(source_path, "face-mask-detection", "annotations")),
    ]
    
    for img_dir, ann_dir in possible_structures:
        if os.path.isdir(img_dir) and os.path.isdir(ann_dir):
            return img_dir, ann_dir
    
    # Search recursively for images and annotations folders
    for root, dirs, files in os.walk(source_path):
        if "images" in dirs and "annotations" in dirs:
            return os.path.join(root, "images"), os.path.join(root, "annotations")
    
    # Last resort: look for XML files and image files in any subdirectory
    xml_dir = None
    img_dir = None
    for root, dirs, files in os.walk(source_path):
        xmls = [f for f in files if f.endswith('.xml')]
        imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if xmls and not xml_dir:
            xml_dir = root
        if imgs and not img_dir:
            img_dir = root
    
    if xml_dir and img_dir:
        return img_dir, xml_dir
    
    logger.error(f"Could not find images/annotations directories in: {source_path}")
    logger.info("Contents of source path:")
    for item in os.listdir(source_path):
        logger.info(f"  {item}")
    sys.exit(1)


def parse_annotation(xml_path):
    """
    Parse a PASCAL VOC XML annotation file.
    
    Returns:
        list of dicts: [{name, xmin, ymin, xmax, ymax}, ...]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            
            # Normalize class names
            if name in ("with_mask", "mask"):
                name = "with_mask"
            elif name in ("without_mask", "no_mask"):
                name = "without_mask"
            elif name in ("mask_weared_incorrect", "mask_worn_incorrect", "incorrect_mask"):
                name = "mask_weared_incorrect"
            
            objects.append({
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })
        return objects
    except Exception as e:
        logger.warning(f"Failed to parse {xml_path}: {e}")
        return []


def split_by_image(faces, seed=SPLIT_SEED):
    """
    Assign each source image to train/val/test.

    Faces from one photo share lighting, camera and often the same people, so
    splitting per face would leak near-duplicates into val/test. Splitting per
    image (stratified on face labels) keeps the test score honest.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_folds = 20
    n_test = round(TEST_SPLIT * n_folds)
    n_val = round(VAL_SPLIT * n_folds)
    labels = [f["label"] for f in faces]
    groups = [f["image"] for f in faces]

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split_of_image = {}
    for fold, (_, idx) in enumerate(sgkf.split(np.zeros(len(faces)), labels, groups)):
        split = "test" if fold < n_test else "val" if fold < n_test + n_val else "train"
        for i in idx:
            split_of_image[groups[i]] = split
    return split_of_image


def prepare_dataset(source_path, output_dir=DATASET_DIR, margin=CROP_MARGIN,
                    min_size=MIN_FACE_SIZE):
    """
    Process downloaded dataset: parse XMLs, crop faces, save to split/class folders.

    Output layout:
        dataset/{train,val,test}/{with_mask,without_mask,mask_weared_incorrect}/
        dataset/manifest.csv   (one row per crop: split, label, source image, box)

    Args:
        source_path: Path to downloaded Kaggle dataset
        output_dir: Where to save cropped face images
        margin: Context around each face, as a fraction of face size (see utils.crop_face)
        min_size: Skip faces whose shorter side is below this many pixels
    """
    img_dir, ann_dir = find_dataset_dirs(source_path)
    logger.info(f"📂 Images dir:      {img_dir}")
    logger.info(f"📂 Annotations dir: {ann_dir}")

    xml_files = sorted(f for f in os.listdir(ann_dir) if f.endswith('.xml'))
    logger.info(f"Found {len(xml_files)} annotation files")

    # ── Collect every usable face first so the split can see all labels
    faces = []
    skipped_small = 0
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        for j, obj in enumerate(parse_annotation(os.path.join(ann_dir, xml_file))):
            if obj["name"] not in CLASSES:
                logger.debug(f"Skipping unknown class: {obj['name']}")
                continue
            if min(obj["xmax"] - obj["xmin"], obj["ymax"] - obj["ymin"]) < min_size:
                skipped_small += 1
                continue
            faces.append({"image": base_name, "index": j, "label": obj["name"],
                          "box": (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])})

    split_of_image = split_by_image(faces)

    if os.path.isdir(output_dir):
        for split in SPLITS:
            shutil.rmtree(os.path.join(output_dir, split), ignore_errors=True)
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    manifest = []
    image_cache_name, image = None, None

    for face in faces:
        if face["image"] != image_cache_name:
            image_cache_name, image = face["image"], None
            for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
                candidate = os.path.join(img_dir, face["image"] + ext)
                if os.path.exists(candidate):
                    image = cv2.imread(candidate)
                    break
            if image is None:
                logger.warning(f"Missing or unreadable image for {face['image']}")
        if image is None:
            continue

        split = split_of_image[face["image"]]
        crop = crop_face(image, face["box"], margin)
        out_name = f"{face['image']}_face{face['index']}.png"
        cv2.imwrite(os.path.join(output_dir, split, face["label"], out_name), crop)
        counts[split][face["label"]] += 1
        manifest.append((split, face["label"], face["image"], *face["box"], out_name))

    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "source_image", "xmin", "ymin", "xmax", "ymax", "file"])
        writer.writerows(manifest)

    # ── Summary
    print("\n" + "=" * 62)
    print("📊 DATASET PREPARATION COMPLETE")
    print("=" * 62)
    print(f"  Source images:              {len(split_of_image)}")
    print(f"  Faces skipped (< {min_size}px):     {skipped_small}")
    print(f"  Face crops written:         {len(manifest)}")
    print("-" * 62)
    print(f"  {'class':<24}" + "".join(f"{s:>10}" for s in SPLITS))
    for cls in CLASSES:
        print(f"  {cls:<24}" + "".join(f"{counts[s][cls]:>10}" for s in SPLITS))
    print("=" * 62)
    print(f"\n  Dataset saved to: {output_dir}")
    print(f"\n  ✅ Ready to train! Run:")
    print(f"     python train.py\n")

    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download & prepare Kaggle Face Mask Dataset")
    parser.add_argument("--source", default=None,
                        help="Path to already-downloaded dataset (skip Kaggle download)")
    parser.add_argument("--output", default=DATASET_DIR,
                        help="Output directory for prepared dataset")
    parser.add_argument("--margin", type=float, default=CROP_MARGIN,
                        help="Context around each face as a fraction of face size")
    parser.add_argument("--min-size", type=int, default=MIN_FACE_SIZE,
                        help="Skip faces smaller than this (px, shorter side)")
    args = parser.parse_args()
    
    if args.source:
        source = args.source
    else:
        source = download_kaggle_dataset()
    
    prepare_dataset(source, args.output, margin=args.margin, min_size=args.min_size)
