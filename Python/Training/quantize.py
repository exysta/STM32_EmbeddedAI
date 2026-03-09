"""
INT8 Quantization & TFLite Export — Phase 4
=============================================
Converts the trained float32 DS-CNN to fully quantized INT8 TFLite
using post-training quantization with representative dataset calibration.

Note: QAT via tensorflow-model-optimization is incompatible with Keras 3
(TF 2.16+). Full-integer PTQ with representative calibration achieves
near-identical accuracy for models of this size.

Usage:
    cd Python/Training
    python quantize.py

Inputs:
    ../models/gragas_dscnn.keras   — Phase 3 float32 model
    ../features/X.npy / y.npy     — for calibration + validation

Outputs:
    ../models/gragas_dscnn_int8.tflite  — INT8 model for STM32Cube.AI
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from train_config import (
    FEATURES_DIR, MODELS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
)

# ── Quantization settings ─────────────────────────────────────────
REPR_SAMPLES = 300    # calibration samples (more = better range estimation)


def load_and_split():
    """Load features and reproduce the same train/val/test split."""
    X = np.load(FEATURES_DIR / "X.npy")[..., np.newaxis]  # (N,97,40,1)
    y = np.load(FEATURES_DIR / "y.npy")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac, random_state=RANDOM_SEED, stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def convert_to_tflite_int8(model, X_cal):
    """Full-integer quantization with representative dataset calibration.

    The converter profiles activation ranges by running real data through
    the model, then maps float32 weights/activations to INT8 with minimal
    range clipping.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset — converter runs these through the model
    # to determine per-layer activation ranges for INT8 mapping
    def representative_dataset():
        indices = np.random.default_rng(42).choice(
            len(X_cal), size=min(REPR_SAMPLES, len(X_cal)), replace=False
        )
        for i in indices:
            yield [X_cal[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model


def evaluate_tflite(tflite_path, X_test, y_test):
    """Run inference with the TFLite INT8 model and measure accuracy."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]["quantization"][0]
    input_zp = input_details[0]["quantization"][1]
    input_dtype = input_details[0]["dtype"]

    correct = 0
    total = len(y_test)

    for i in range(total):
        sample = X_test[i:i+1].astype(np.float32)

        # Quantize input if model expects int8
        if input_dtype == np.int8:
            sample = (sample / input_scale + input_zp).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        pred = np.argmax(output, axis=1)[0]
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / total
    return accuracy


def main():
    import tensorflow as tf
    from tensorflow import keras

    # Register custom layer for model loading
    from ds_cnn_model import SpecAugment  # noqa: F401

    print("=" * 60)
    print("  PHASE 4 — INT8 QUANTIZATION")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    print(f"\n  Calibration samples : {min(REPR_SAMPLES, len(X_train))}")
    print(f"  Test samples        : {len(y_test)}")

    # ── 2. Load float32 model ──────────────────────────────────────
    model_path = MODELS_DIR / "gragas_dscnn.keras"
    print(f"\n  Loading float32 model: {model_path}")
    model = keras.models.load_model(
        str(model_path),
        custom_objects={"SpecAugment": SpecAugment},
    )

    loss_f32, acc_f32 = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Float32 accuracy: {acc_f32:.4f} ({acc_f32*100:.1f}%)")

    # ── 3. Convert to TFLite INT8 ─────────────────────────────────
    print(f"\n  Converting to TFLite INT8 with representative calibration...")
    tflite_model = convert_to_tflite_int8(model, X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tflite_path = MODELS_DIR / "gragas_dscnn_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    tflite_kb = len(tflite_model) / 1024
    print(f"  Saved: {tflite_path}  ({tflite_kb:.1f} KB)")

    # ── 4. Validate INT8 accuracy ─────────────────────────────────
    print(f"\n  Evaluating TFLite INT8 on test set...")
    acc_int8 = evaluate_tflite(tflite_path, X_test, y_test)
    print(f"  INT8 accuracy: {acc_int8:.4f} ({acc_int8*100:.1f}%)")

    # ── 5. Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ACCURACY COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Size':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    f32_size = model_path.stat().st_size / 1024
    print(f"  {'Float32 (baseline)':<25} {acc_f32*100:>9.1f}% {f32_size:>8.1f} KB")
    print(f"  {'TFLite INT8':<25} {acc_int8*100:>9.1f}% {tflite_kb:>8.1f} KB")

    drop = (acc_f32 - acc_int8) * 100
    print(f"\n  Accuracy drop (float32 → INT8): {drop:+.1f} percentage points")
    print(f"  Size reduction: {f32_size:.0f} KB → {tflite_kb:.0f} KB "
          f"({f32_size/tflite_kb:.1f}× smaller)")

    if acc_int8 >= 0.90:
        print(f"\n  ✅ INT8 accuracy above 90% target — ready for STM32Cube.AI")
    else:
        print(f"\n  ⚠️  INT8 accuracy below 90% — model may need retraining")

    print(f"\n  Output: {tflite_path.resolve()}")
    print(f"  Next:   Import into STM32Cube.AI (Phase 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
