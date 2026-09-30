# pyre-unsafe
"""
export.py — Export the mask classifier to TensorFlow Lite and check the result

Variants:
    fp16  weights stored as float16            (~half the size, same accuracy)
    int8  dynamic-range int8 weight quantization (~quarter the size)

Every export is scored on the held-out test split next to the Keras model
(accuracy and the largest probability difference) and timed on one face
with flip TTA, i.e. a batch of 2, which is what MaskDetector runs per face.
A variant whose test predictions disagree with Keras on more than
--max-disagreement of the faces is reported as FAILED.

Outputs:
    models/export/mask_detector_<variant>.tflite
    models/export/export_report.json

Usage:
    python export.py                    # fp16 + int8
    python export.py --variants fp16
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf

from config import MODEL_DIR, MASK_MODEL_PATH, TEST_DIR, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Export")

EXPORT_DIR = os.path.join(MODEL_DIR, "export")
VARIANTS = ("fp16", "int8")


def convert(model, variant):
    """Serialized TFLite flatbuffer for one variant."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if variant == "fp16":
        converter.target_spec.supported_types = [tf.float16]
    elif variant != "int8":
        raise ValueError(f"Unknown variant '{variant}' (options: {VARIANTS})")
    return converter.convert()


def make_interpreter(model_path=None, model_content=None, num_threads=None):
    """TFLite interpreter, from ai_edge_litert when installed (tf.lite's is deprecated)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import warnings
        warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=model_path, model_content=model_content,
                       num_threads=num_threads)


class TFLiteRunner:
    """predict_on_batch() for a TFLite model, resizing the batch dimension on demand."""

    def __init__(self, model_path=None, model_content=None, num_threads=None):
        self.interpreter = make_interpreter(model_path, model_content, num_threads)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]
        self._batch = int(self._in["shape"][0])
        self.input_shape = (None, *[int(d) for d in self._in["shape"][1:]])

    def predict_on_batch(self, batch):
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if batch.shape[0] != self._batch:
            self.interpreter.resize_tensor_input(self._in["index"], list(batch.shape))
            self.interpreter.allocate_tensors()
            self._batch = batch.shape[0]
        self.interpreter.set_tensor(self._in["index"], batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._out["index"]).copy()


def test_images(input_size, test_dir=TEST_DIR):
    from data_pipeline import make_eval_dataset
    ds, labels, _ = make_eval_dataset(test_dir, BATCH_SIZE, input_size, one_hot=False)
    images = np.concatenate([x.numpy() for x, _ in ds], axis=0)
    return images, labels


def predict_all(runner, images, batch_size=BATCH_SIZE):
    """Flip-TTA probabilities for every image."""
    out = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size]
        p = np.asarray(runner.predict_on_batch(np.concatenate([x, x[:, :, ::-1, :]], axis=0)))
        out.append((p[:len(x)] + p[len(x):]) / 2.0)
    return np.concatenate(out, axis=0)


def latency_ms(runner, input_size, runs=50, warmup=5):
    """Median milliseconds for one face with flip TTA (batch of 2)."""
    x = np.random.RandomState(0).uniform(0, 255, (2, *input_size, 3)).astype("float32")
    for _ in range(warmup):
        runner.predict_on_batch(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        runner.predict_on_batch(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main(args):
    import keras
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model = keras.models.load_model(args.model, compile=False)
    input_size = tuple(model.input_shape[1:3])

    images, labels = test_images(input_size, args.test_dir)
    ref = predict_all(model, images)
    ref_pred = ref.argmax(axis=1)
    report = {
        "keras": {
            "file": os.path.relpath(args.model, MODEL_DIR).replace("\\", "/"),
            "size_mb": round(os.path.getsize(args.model) / 2**20, 2),
            "test_accuracy": round(float((ref_pred == labels).mean()), 4),
            "latency_ms_per_face": round(latency_ms(model, input_size), 2),
        },
        "test_faces": int(len(labels)),
        "num_threads": args.threads,
    }

    ok = True
    for variant in args.variants:
        t0 = time.time()
        flatbuffer = convert(model, variant)
        path = os.path.join(EXPORT_DIR, f"mask_detector_{variant}.tflite")
        with open(path, "wb") as f:
            f.write(flatbuffer)
        runner = TFLiteRunner(model_path=path, num_threads=args.threads)
        probs = predict_all(runner, images)
        pred = probs.argmax(axis=1)
        disagreement = float((pred != ref_pred).mean())
        passed = disagreement <= args.max_disagreement
        ok &= passed
        report[variant] = {
            "file": os.path.relpath(path, MODEL_DIR).replace("\\", "/"),
            "size_mb": round(len(flatbuffer) / 2**20, 2),
            "test_accuracy": round(float((pred == labels).mean()), 4),
            "disagreement_with_keras": round(disagreement, 4),
            "max_abs_prob_diff": round(float(np.abs(probs - ref).max()), 4),
            "latency_ms_per_face": round(latency_ms(runner, input_size), 2),
            "convert_seconds": round(time.time() - t0, 1),
            "passed": passed,
        }
        logger.info(f"{variant}: {report[variant]}")

    with open(os.path.join(EXPORT_DIR, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print(f"EXPORT REPORT  ({report['test_faces']} test faces, flip TTA, latency = 1 face)")
    print("=" * 78)
    print(f"  {'model':<8}{'size MB':>9}{'accuracy':>10}{'differs':>9}{'max |Δp|':>10}{'ms/face':>9}  status")
    k = report["keras"]
    print(f"  {'keras':<8}{k['size_mb']:>9.2f}{k['test_accuracy']*100:>9.2f}%{'-':>9}{'-':>10}"
          f"{k['latency_ms_per_face']:>9.2f}  reference")
    for v in args.variants:
        r = report[v]
        print(f"  {v:<8}{r['size_mb']:>9.2f}{r['test_accuracy']*100:>9.2f}%"
              f"{r['disagreement_with_keras']*100:>8.2f}%{r['max_abs_prob_diff']:>10.4f}"
              f"{r['latency_ms_per_face']:>9.2f}  {'ok' if r['passed'] else 'FAILED'}")
    print(f"\n  Files: {EXPORT_DIR}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the mask classifier to TFLite")
    parser.add_argument("--model", default=MASK_MODEL_PATH)
    parser.add_argument("--test-dir", default=TEST_DIR)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--threads", type=int, default=None,
                        help="TFLite interpreter threads (default: TFLite's choice)")
    parser.add_argument("--max-disagreement", type=float, default=0.005,
                        help="Fail an export whose test predictions differ from Keras more often")
    raise SystemExit(0 if main(parser.parse_args()) else 1)
