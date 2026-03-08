"""
Model Evaluation — Phase 3
============================
Standalone evaluation of a trained DS-CNN model.
Loads the saved model and test data, prints detailed metrics.

Usage:
    cd Python/Training
    python evaluate.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from train_config import (
    FEATURES_DIR, MODELS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
)


def main():
    import tensorflow as tf
    from tensorflow import keras

    # Register SpecAugment for loading
    from ds_cnn_model import SpecAugment  # noqa: F401

    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────
    model_path = MODELS_DIR / "gragas_dscnn.keras"
    if not model_path.exists():
        print(f"  ERROR: model not found at {model_path}")
        print(f"  Run train.py first.")
        return

    model = keras.models.load_model(
        str(model_path),
        custom_objects={"SpecAugment": SpecAugment},
    )
    print(f"\n  Loaded: {model_path}")
    model.summary()

    # ── Load & split data (same seed → same test set) ──────────────
    X = np.load(FEATURES_DIR / "X.npy")[..., np.newaxis]
    y = np.load(FEATURES_DIR / "y.npy")

    X_trainval, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"\n  Test set: {X_test.shape[0]} samples")

    # ── Evaluate ───────────────────────────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    print(f"\n  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {accuracy:.4f}  ({accuracy * 100:.1f}%)")

    # ── Confusion matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"  {'':>12} {'pred_neg':>10} {'pred_ww':>10}")
    print(f"  {'true_neg':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"  {'true_ww':>12} {cm[1][0]:>10} {cm[1][1]:>10}")

    # ── Detailed metrics ───────────────────────────────────────────
    print(f"\n  Classification report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    for line in report.split("\n"):
        print(f"  {line}")

    # ── Model size ─────────────────────────────────────────────────
    total_params = model.count_params()
    print(f"\n  Model statistics:")
    print(f"    Parameters   : {total_params:,}")
    print(f"    Float32 size : {total_params * 4 / 1024:.1f} KB")
    print(f"    INT8 (est.)  : {total_params / 1024:.1f} KB")

    # ── False positive analysis ────────────────────────────────────
    fp = cm[0][1]  # negative predicted as wakeword
    fn = cm[1][0]  # wakeword predicted as negative
    tn = cm[0][0]
    tp = cm[1][1]

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  Wake word detection:")
    print(f"    True positives  : {tp}")
    print(f"    False positives : {fp}  (FPR: {fpr:.3f})")
    print(f"    False negatives : {fn}  (FNR: {fnr:.3f})")
    print(f"    True negatives  : {tn}")

    print("=" * 60)


if __name__ == "__main__":
    main()
