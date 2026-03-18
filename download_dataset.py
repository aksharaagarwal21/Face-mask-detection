# pyre-unsafe
import os
import sys
import shutil
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownloadDataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


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


def prepare_dataset(source_path, output_dir=DATASET_DIR, padding=15, min_size=30):
    """
    Process downloaded dataset: parse XMLs, crop faces, save to class folders.
    
    Args:
        source_path: Path to downloaded Kaggle dataset
        output_dir: Where to save cropped face images
        padding: Pixels to add around each face crop for context
        min_size: Minimum face crop dimension (skip tiny faces)
    """
    img_dir, ann_dir = find_dataset_dirs(source_path)
    logger.info(f"📂 Images dir:      {img_dir}")
    logger.info(f"📂 Annotations dir: {ann_dir}")
    
    # Count XML files
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    logger.info(f"Found {len(xml_files)} annotation files")
    
    # Create output class directories
    for cls in CLASSES:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    counts = {cls: 0 for cls in CLASSES}
    skipped = 0
    processed_images = 0
    
    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(ann_dir, xml_file)
        objects = parse_annotation(xml_path)
        
        if not objects:
            continue
        
        # Find corresponding image
        base_name = os.path.splitext(xml_file)[0]
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            skipped += 1
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception as e:
            logger.warning(f"Cannot open image {img_path}: {e}")
            skipped += 1
            continue
        
        processed_images += 1
        
        for j, obj in enumerate(objects):
            label = obj["name"]
            if label not in CLASSES:
                logger.debug(f"Skipping unknown class: {label}")
                continue
            
            # Add padding around the face crop
            xmin = max(0, obj["xmin"] - padding)
            ymin = max(0, obj["ymin"] - padding)
            xmax = min(img_w, obj["xmax"] + padding)
            ymax = min(img_h, obj["ymax"] + padding)
            
            # Skip tiny crops
            if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
                continue
            
            # Crop and save
            face = img.crop((xmin, ymin, xmax, ymax))
            face = face.resize((224, 224), Image.LANCZOS)
            
            out_name = f"{base_name}_face{j}.jpg"
            out_path = os.path.join(output_dir, label, out_name)
            face.save(out_path, "JPEG", quality=95)
            counts[label] += 1
        
        # Progress log
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(xml_files)} annotations...")
    
    # ── Summary
    total = sum(counts.values())
    print("\n" + "=" * 55)
    print("📊 DATASET PREPARATION COMPLETE")
    print("=" * 55)
    print(f"  Images processed:      {processed_images}")
    print(f"  Annotations skipped:   {skipped}")
    print(f"  Total face crops:      {total}")
    print("-" * 55)
    for cls, count in counts.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:<28} {count:>5} ({pct:>5.1f}%) {bar}")
    print("=" * 55)
    print(f"\n  Dataset saved to: {output_dir}")
    print(f"\n  ✅ Ready to train! Run:")
    print(f"     python train.py --epochs 20 --fine-tune\n")
    
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download & prepare Kaggle Face Mask Dataset")
    parser.add_argument("--source", default=None,
                        help="Path to already-downloaded dataset (skip Kaggle download)")
    parser.add_argument("--output", default=DATASET_DIR,
                        help="Output directory for prepared dataset")
    parser.add_argument("--padding", type=int, default=15,
                        help="Padding pixels around face crops")
    args = parser.parse_args()
    
    if args.source:
        source = args.source
    else:
        source = download_kaggle_dataset()
    
    prepare_dataset(source, args.output, padding=args.padding)
