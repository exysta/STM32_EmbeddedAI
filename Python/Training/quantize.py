"""
INT8 Quantization & TFLite Export — Phase 4
=============================================
Converts the trained float32 DS-CNN to fully quantized INT8 TFLite.

Automatically uses the fine-tuned model if it exists, otherwise falls
back to the base model.  Pass --model to override explicitly.

Usage:
    cd Python/Training

    # Uses gragas_dscnn_finetuned.keras if present, else gragas_dscnn.keras
    python quantize.py

    # Explicit model path
    python quantize.py --model ../models/gragas_dscnn_finetuned.keras
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from train_config import (
    FEATURES_DIR, MODELS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    FINETUNE_OUTPUT_MODEL,
)

REPR_SAMPLES = 300


def load_and_split():
    X = np.load(FEATURES_DIR / "X.npy")[..., np.newaxis]
    y = np.load(FEATURES_DIR / "y.npy")

    weights_path = FEATURES_DIR / "weights.npy"
    w = np.load(weights_path) if weights_path.exists() else np.ones(len(y))

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
        X, y, w, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval, y_trainval, w_trainval,
        test_size=val_frac, random_state=RANDOM_SEED, stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, w_test


def convert_to_tflite_int8(model, X_cal):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        idx = np.random.default_rng(42).choice(
            len(X_cal), size=min(REPR_SAMPLES, len(X_cal)), replace=False
        )
        for i in idx:
            yield [X_cal[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_tflite(tflite_path, X_test, y_test, w_test=None):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    scale = inp[0]["quantization"][0]
    zp    = inp[0]["quantization"][1]
    dtype = inp[0]["dtype"]

    preds = []
    for i in range(len(y_test)):
        sample = X_test[i:i+1].astype(np.float32)
        if dtype == np.int8:
            sample = (sample / scale + zp).astype(np.int8)
        interp.set_tensor(inp[0]["index"], sample)
        interp.invoke()
        preds.append(np.argmax(interp.get_tensor(out[0]["index"]), axis=1)[0])

    preds = np.array(preds)
    acc_overall = (preds == y_test).mean()

    # Source-stratified accuracy
    if w_test is not None and (w_test > 1.0).any():
        real_mask = w_test > 1.0
        if real_mask.sum() > 0:
            acc_real = (preds[real_mask] == y_test[real_mask]).mean()
            print(f"  INT8 real INMP441 accuracy : {acc_real:.4f} "
                  f"({acc_real*100:.1f}%,  n={real_mask.sum()})")

    return acc_overall


def resolve_model_path(override: str | None) -> Path:
    """Pick the best available model automatically."""
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Specified model not found: {p}")
        return p

    if FINETUNE_OUTPUT_MODEL.exists():
        print(f"  Auto-selected fine-tuned model: {FINETUNE_OUTPUT_MODEL.name}")
        return FINETUNE_OUTPUT_MODEL

    base = MODELS_DIR / "gragas_dscnn.keras"
    if base.exists():
        print(f"  Auto-selected base model: {base.name}")
        return base

    raise FileNotFoundError(
        "No model found. Run train.py (and optionally train.py --finetune) first."
    )


def main():
    parser = argparse.ArgumentParser(description="INT8 quantization")
    parser.add_argument("--model", default=None,
                        help="Path to .keras model (auto-detected if omitted)")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from ds_cnn_model import SpecAugment  # noqa

    print("=" * 60)
    print("  PHASE 4 — INT8 QUANTIZATION")
    print("=" * 60)

    model_path = resolve_model_path(args.model)
    X_train, X_val, X_test, y_train, y_val, y_test, w_test = load_and_split()

    print(f"\n  Loading: {model_path}")
    model = keras.models.load_model(
        str(model_path),
        custom_objects={"SpecAugment": SpecAugment},
    )

    _, acc_f32 = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Float32 accuracy : {acc_f32:.4f} ({acc_f32*100:.1f}%)")

    # Use training split for calibration
    print(f"\n  Converting to INT8 ({REPR_SAMPLES} calibration samples)…")
    tflite_model = convert_to_tflite_int8(model, X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tflite_path = MODELS_DIR / "gragas_dscnn_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    tflite_kb = len(tflite_model) / 1024
    print(f"  Saved: {tflite_path}  ({tflite_kb:.1f} KB)")

    print(f"\n  Evaluating INT8 on test set…")
    acc_int8 = evaluate_tflite(tflite_path, X_test, y_test, w_test)
    print(f"  INT8 accuracy : {acc_int8:.4f} ({acc_int8*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Size':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    f32_kb = model_path.stat().st_size / 1024
    print(f"  {model_path.name:<30} {acc_f32*100:>9.1f}% {f32_kb:>8.1f} KB")
    print(f"  {'TFLite INT8':<30} {acc_int8*100:>9.1f}% {tflite_kb:>8.1f} KB")
    drop = (acc_f32 - acc_int8) * 100
    print(f"\n  Accuracy drop: {drop:+.1f} pp")

    status = "✅ Ready for STM32Cube.AI" if acc_int8 >= 0.90 else \
             "⚠️  Below 90% — consider more real recordings"
    print(f"  {status}")
    print(f"\n  Output: {tflite_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()