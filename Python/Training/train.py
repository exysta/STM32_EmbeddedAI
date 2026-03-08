"""
DS-CNN Training Script — Phase 3
==================================
Trains a DS-CNN + SE model on the extracted MFCC features.

Usage:
    cd Python/Training
    python train.py

Inputs:
    ../features/X.npy           — (4000, 97, 40) normalised MFCCs
    ../features/y.npy           — (4000,)        labels (0=negative, 1=wakeword)

Outputs:
    ../models/gragas_dscnn.keras         — best float32 model
    ../models/training_history.npy       — loss/accuracy curves
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from train_config import (
    FEATURES_DIR, MODELS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LR_FACTOR, LR_PATIENCE, LR_MIN,
    ES_PATIENCE, ES_RESTORE_BEST,
)
from ds_cnn_model import build_ds_cnn


def load_data():
    """Load MFCC features and labels, add channel dimension."""
    X = np.load(FEATURES_DIR / "X.npy")   # (N, 97, 40)
    y = np.load(FEATURES_DIR / "y.npy")   # (N,)

    # Add channel dim for Conv2D: (N, 97, 40) → (N, 97, 40, 1)
    X = X[..., np.newaxis]

    print(f"  Loaded features : {X.shape}  dtype={X.dtype}")
    print(f"  Labels          : {y.shape}  classes={np.unique(y)}")
    return X, y


def split_data(X, y):
    """Stratified train / val / test split."""
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Second split: separate validation from training
    # VAL_SIZE is fraction of total, so adjust for remaining data
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    print(f"\n  Split:")
    print(f"    Train : {X_train.shape[0]:>5} samples")
    print(f"    Val   : {X_val.shape[0]:>5} samples")
    print(f"    Test  : {X_test.shape[0]:>5} samples")

    for name, y_subset in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = [np.sum(y_subset == i) for i in range(NUM_CLASSES)]
        dist = " / ".join(f"{CLASS_NAMES[i]}={c}" for i, c in enumerate(counts))
        print(f"    {name:>5} distribution: {dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train(X_train, X_val, y_train, y_val):
    """Build, compile, and train the DS-CNN model."""
    import tensorflow as tf
    from tensorflow import keras

    # ── Build model ────────────────────────────────────────────────
    model = build_ds_cnn()
    model.summary()

    print(f"\n  Total params    : {model.count_params():,}")
    print(f"  Float32 size    : {model.count_params() * 4 / 1024:.1f} KB")
    print(f"  INT8 size (est) : {model.count_params() / 1024:.1f} KB")

    # ── Compile ────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ──────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "gragas_dscnn.keras"

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            restore_best_weights=ES_RESTORE_BEST,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    np.save(
        MODELS_DIR / "training_history.npy",
        {k: [float(v) for v in vals] for k, vals in history.history.items()},
    )

    return model, history


def evaluate(model, X_test, y_test):
    """Evaluate model on the held-out test set."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    # Overall accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {accuracy:.4f}  ({accuracy * 100:.1f}%)")

    # Predictions
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"  {'':>12} {'pred_neg':>10} {'pred_ww':>10}")
    print(f"  {'true_neg':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"  {'true_ww':>12} {cm[1][0]:>10} {cm[1][1]:>10}")

    # Classification report
    print(f"\n  Classification report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    for line in report.split("\n"):
        print(f"  {line}")

    return accuracy


def main():
    print("=" * 60)
    print("  DS-CNN + SE ATTENTION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Features dir : {FEATURES_DIR.resolve()}")
    print(f"  Models dir   : {MODELS_DIR.resolve()}")
    print()

    # ── 1. Load data ───────────────────────────────────────────────
    X, y = load_data()

    # ── 2. Split ───────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ── 3. Train ───────────────────────────────────────────────────
    model, history = train(X_train, X_val, y_train, y_val)

    # ── 4. Evaluate ────────────────────────────────────────────────
    accuracy = evaluate(model, X_test, y_test)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  DONE — Test accuracy: {accuracy * 100:.1f}%")
    print(f"  Model saved to: {MODELS_DIR.resolve() / 'gragas_dscnn.keras'}")
    print("=" * 60)

    if accuracy >= 0.90:
        print("\n  ✅ Target accuracy (>90%) ACHIEVED")
    else:
        print("\n  ⚠️  Below 90% target — consider tuning hyperparameters")

    print(f"\n  Next step: Phase 4 — Quantization-Aware Training")
    print(f"    python evaluate.py   # detailed metrics")


if __name__ == "__main__":
    main()
