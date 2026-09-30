# pyre-unsafe
"""
explain.py — Grad-CAM heatmaps: where the classifier looks on each face

Grad-CAM weights the backbone's last feature map by the gradient of the
predicted class's logit, which shows the image regions that pushed the
decision. For a sound mask classifier the heat should sit on the nose,
mouth and chin; heat on the background or hair points at a shortcut.

Usage:
    python explain.py --image photo.jpg            # detect faces, explain each one
    python explain.py --image face.png --crop      # the image already is a face crop
    python explain.py --test-samples 6             # gallery of test faces per class
"""

import os
import argparse
import logging
import numpy as np
import cv2
import tensorflow as tf

from config import MODEL_DIR, MASK_MODEL_PATH, CLASSES, TEST_DIR
from model import load_trained_model, get_backbone
from calibrate import apply_temperature, load_temperature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Explain")

SHORT = {"mask_weared_incorrect": "incorrect", "with_mask": "mask", "without_mask": "no mask"}


class GradCAM:
    """Grad-CAM for the classifier built by model.build_model()."""

    def __init__(self, model):
        self.model = model
        self.input_size = tuple(model.input_shape[1:3])
        self.backbone = get_backbone(model)
        idx = model.layers.index(self.backbone)
        # Layers before the backbone (optional Rescaling) and the head after it
        self.pre = [l for l in model.layers[1:idx]]
        self.head = model.layers[idx + 1:-1]
        self.out = model.layers[-1]            # Dense(softmax): use its pre-softmax logits

    def _prepare(self, face_bgr):
        h, w = self.input_size
        interp = cv2.INTER_AREA if face_bgr.shape[0] > h else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(face_bgr, (w, h), interpolation=interp), cv2.COLOR_BGR2RGB)
        return rgb.astype("float32")

    def heatmaps(self, faces_bgr, class_idx=None):
        """
        Args:
            faces_bgr: list of BGR face crops
            class_idx: class to explain per face (default: the predicted one)

        Returns:
            (heatmaps in [0, 1] at input size, softmax probabilities)
        """
        x = tf.convert_to_tensor(np.stack([self._prepare(f) for f in faces_bgr]))
        for layer in self.pre:
            x = layer(x)
        with tf.GradientTape() as tape:
            feats = self.backbone(x, training=False)
            tape.watch(feats)
            y = feats
            for layer in self.head:
                y = layer(y, training=False)
            logits = tf.matmul(y, self.out.kernel) + self.out.bias
            probs = tf.nn.softmax(logits)
            idx = tf.argmax(logits, axis=1) if class_idx is None else \
                tf.constant(np.broadcast_to(class_idx, (len(faces_bgr),)), dtype=tf.int64)
            score = tf.gather(logits, idx, axis=1, batch_dims=1)
        grads = tape.gradient(score, feats)                    # (n, h, w, c)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1)).numpy()

        out = []
        for c in cam:
            c = cv2.resize(c, self.input_size[::-1], interpolation=cv2.INTER_CUBIC)
            c = np.clip(c, 0, None)
            out.append(c / c.max() if c.max() > 0 else c)
        return out, probs.numpy()


def overlay(face_bgr, heat, size=160, alpha=0.45):
    """Heatmap blended over the face crop, resized to size x size."""
    face = cv2.resize(face_bgr, (size, size))
    color = cv2.applyColorMap((cv2.resize(heat, (size, size)) * 255).astype("uint8"),
                              cv2.COLORMAP_JET)
    return cv2.addWeighted(color, alpha, face, 1 - alpha, 0)


def captioned(tile, text, color=(255, 255, 255)):
    bar = np.full((22, tile.shape[1], 3), 25, dtype="uint8")
    cv2.putText(bar, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def explain_faces(cam, faces, titles=None, size=160):
    """Row per face: [crop | Grad-CAM overlay], captioned with the prediction."""
    heats, probs = cam.heatmaps(faces)
    probs = apply_temperature(probs, load_temperature())    # same confidences as MaskDetector
    rows = []
    for i, (face, heat, p) in enumerate(zip(faces, heats, probs)):
        k = int(np.argmax(p))
        pred = SHORT[CLASSES[k]]
        right_ok = not titles or titles[i] == pred
        left = captioned(cv2.resize(face, (size, size)), f"true: {titles[i]}" if titles else "input")
        right = captioned(overlay(face, heat, size), f"pred: {pred} {p[k]:.2f}",
                          (140, 255, 140) if right_ok else (120, 150, 255))
        rows.append(np.hstack([left, right]))
    return rows


def grid(tiles, cols):
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * h, cols * w, 3), 25, dtype="uint8")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return canvas


def test_gallery(cam, n_per_class, seed=0):
    """n faces per class from the test split, each shown with its Grad-CAM."""
    rng = np.random.RandomState(seed)
    faces, titles = [], []
    for cls in CLASSES:
        d = os.path.join(TEST_DIR, cls)
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        for name in rng.choice(files, size=min(n_per_class, len(files)), replace=False):
            faces.append(cv2.imread(os.path.join(d, name)))
            titles.append(SHORT[cls])
    return grid(explain_faces(cam, faces, titles), cols=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM explanations for the mask classifier")
    parser.add_argument("--model", default=MASK_MODEL_PATH)
    parser.add_argument("--image", help="Photo to explain (faces are detected first)")
    parser.add_argument("--crop", action="store_true", help="--image is already a face crop")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="Build a gallery with this many test faces per class")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    cam = GradCAM(load_trained_model(args.model))

    if args.test_samples:
        out = args.out or os.path.join(MODEL_DIR, "plots", "gradcam_examples.jpg")
        img = test_gallery(cam, args.test_samples)
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        if args.crop:
            faces = [frame]
        else:
            from face_detector import FaceDetector
            _, faces = FaceDetector().detect_faces_rois(frame)
            if not faces:
                raise SystemExit("No faces detected (use --crop if the image is a face crop)")
        out = args.out or os.path.splitext(args.image)[0] + "_gradcam.jpg"
        img = grid(explain_faces(cam, faces), cols=2 if len(faces) > 3 else 1)
    else:
        parser.error("give --image or --test-samples")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, img)
    print(f"✅ Grad-CAM saved: {out}")
