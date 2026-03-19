"""
DS-CNN Training Script — Phase 3
==================================
Trains or fine-tunes a DS-CNN + SE model on MFCC features.

Modes
-----
  scratch   (default)  Train from random init.
                       Use for first-time training on TTS-only dataset.

  finetune             Load gragas_dscnn.keras, lower LR, apply sample
                       weights so real INMP441 recordings dominate the loss.
                       Use after adding real recordings via record_dataset.py.

Usage:
    cd Python/Training

    # First training (TTS only):
    python train.py

    # After recording real samples:
    python train.py --finetune

Inputs:
    ../features/X.npy           — (N, 97, 40) normalised MFCCs
    ../features/y.npy           — (N,)        labels
    ../features/weights.npy     — (N,)        per-sample weights (optional)

Outputs (scratch):
    ../models/gragas_dscnn.keras
    ../models/training_history.npy

Outputs (finetune):
    ../models/gragas_dscnn_finetuned.keras
    ../models/finetuning_history.npy
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from train_config import (
    FEATURES_DIR, MODELS_DIR,
    NUM_CLASSES, CLASS_NAMES,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    # Scratch
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LR_FACTOR, LR_PATIENCE, LR_MIN,
    ES_PATIENCE, ES_RESTORE_BEST,
    # Fine-tune
    FINETUNE_LR, FINETUNE_EPOCHS,
    FINETUNE_LR_PATIENCE, FINETUNE_ES_PATIENCE,
    FINETUNE_BASE_MODEL, FINETUNE_OUTPUT_MODEL,
)
from ds_cnn_model import build_ds_cnn, SpecAugment


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    X = np.load(FEATURES_DIR / "X.npy")[..., np.newaxis]  # (N, 97, 40, 1)
    y = np.load(FEATURES_DIR / "y.npy")

    weights_path = FEATURES_DIR / "weights.npy"
    if weights_path.exists():
        w = np.load(weights_path)
        real_count = int((w > 1.0).sum())
        tts_count  = int((w <= 1.0).sum())
        print(f"  Sample weights loaded: real={real_count}  tts/synth={tts_count}")
    else:
        w = np.ones(len(y), dtype=np.float32)
        print(f"  No weights.npy found — using uniform weights.")
        print(f"  Tip: run extract_mfcc.py after generate_wakeword_dataset.py")

    print(f"  Features : {X.shape}  dtype={X.dtype}")
    print(f"  Labels   : {y.shape}  classes={np.unique(y)}")
    return X, y, w


def split_data(X, y, w):
    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
        X, y, w,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval, y_trainval, w_trainval,
        test_size=val_frac,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    print(f"\n  Split:")
    print(f"    Train : {X_train.shape[0]:>5}  "
          f"(real={int((w_train > 1.0).sum())}  "
          f"tts={int((w_train <= 1.0).sum())})")
    print(f"    Val   : {X_val.shape[0]:>5}  "
          f"(real={int((w_val > 1.0).sum())}  "
          f"tts={int((w_val <= 1.0).sum())})")
    print(f"    Test  : {X_test.shape[0]:>5}  "
          f"(real={int((w_test > 1.0).sum())}  "
          f"tts={int((w_test <= 1.0).sum())})")

    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val


# ── Training ──────────────────────────────────────────────────────

def run_training(X_train, X_val, y_train, y_val, w_train,
                 finetune: bool = False):
    import tensorflow as tf
    from tensorflow import keras

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if finetune:
        # ── Fine-tune mode ────────────────────────────────────────
        if not FINETUNE_BASE_MODEL.exists():
            print(f"\n  ERROR: base model not found at {FINETUNE_BASE_MODEL}")
            print(f"  Run without --finetune first to train the base model.")
            raise FileNotFoundError(FINETUNE_BASE_MODEL)

        print(f"\n  Loading base model: {FINETUNE_BASE_MODEL}")
        model = keras.models.load_model(
            str(FINETUNE_BASE_MODEL),
            custom_objects={"SpecAugment": SpecAugment},
        )
        lr       = FINETUNE_LR
        epochs   = FINETUNE_EPOCHS
        lr_pat   = FINETUNE_LR_PATIENCE
        es_pat   = FINETUNE_ES_PATIENCE
        out_path = FINETUNE_OUTPUT_MODEL
        hist_key = "finetuning_history.npy"
        print(f"  Fine-tune LR: {lr}  (10× lower than scratch to preserve TTS knowledge)")

    else:
        # ── Scratch mode ──────────────────────────────────────────
        model  = build_ds_cnn()
        lr     = LEARNING_RATE
        epochs = EPOCHS
        lr_pat = LR_PATIENCE
        es_pat = ES_PATIENCE
        out_path = MODELS_DIR / "gragas_dscnn.keras"
        hist_key = "training_history.npy"

    model.summary()
    print(f"\n  Parameters : {model.count_params():,}")
    print(f"  Float32    : {model.count_params() * 4 / 1024:.1f} KB")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=lr_pat,
            min_lr=LR_MIN,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_pat,
            restore_best_weights=ES_RESTORE_BEST,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(out_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    mode_label = "FINE-TUNING" if finetune else "TRAINING FROM SCRATCH"
    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    if finetune:
        real_n  = int((w_train > 1.0).sum())
        tts_n   = int((w_train <= 1.0).sum())
        eff_real = real_n * 4.0          # weight=4.0 for real samples
        eff_tts  = tts_n  * 1.0
        print(f"  Effective loss contribution:")
        print(f"    Real samples : {real_n} × weight 4.0 = {eff_real:.0f} effective")
        print(f"    TTS samples  : {tts_n} × weight 1.0 = {eff_tts:.0f} effective")
        print(f"    Real dominance: {eff_real / (eff_real + eff_tts) * 100:.0f}% of gradient")
    print(f"{'='*60}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=w_train,          # ← real samples count 4× in loss
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    np.save(
        MODELS_DIR / hist_key,
        {k: [float(v) for v in vals] for k, vals in history.history.items()},
    )

    print(f"\n  Model saved → {out_path}")
    return model, out_path


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, w_test):
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*60}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {accuracy:.4f}  ({accuracy * 100:.1f}%)")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion matrix:")
    print(f"  {'':>12} {'pred_neg':>10} {'pred_ww':>10}")
    print(f"  {'true_neg':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"  {'true_ww':>12} {cm[1][0]:>10} {cm[1][1]:>10}")

    print(f"\n  Classification report:")
    for line in classification_report(y_test, y_pred,
                                       target_names=CLASS_NAMES).split("\n"):
        print(f"  {line}")

    # Source-stratified accuracy (real vs TTS) if weights available
    if w_test is not None and (w_test > 1.0).any():
        real_mask = w_test > 1.0
        tts_mask  = ~real_mask
        if real_mask.sum() > 0:
            acc_real = (y_pred[real_mask] == y_test[real_mask]).mean()
            print(f"  Real INMP441 accuracy : {acc_real:.4f}  "
                  f"({acc_real*100:.1f}%,  n={real_mask.sum()})")
        if tts_mask.sum() > 0:
            acc_tts = (y_pred[tts_mask] == y_test[tts_mask]).mean()
            print(f"  TTS/synth accuracy    : {acc_tts:.4f}  "
                  f"({acc_tts*100:.1f}%,  n={tts_mask.sum()})")

    return accuracy


# ── Entry point ───────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune the DS-CNN wakeword model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help=(
            "Fine-tune from gragas_dscnn.keras using lower LR and sample "
            "weights. Use after recording real INMP441 samples."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mode = "FINE-TUNE" if args.finetune else "SCRATCH"
    print("=" * 60)
    print(f"  DS-CNN + SE — TRAINING  [{mode}]")
    print("=" * 60)
    print(f"  Features dir : {FEATURES_DIR.resolve()}")
    print(f"  Models dir   : {MODELS_DIR.resolve()}")
    print()

    X, y, w = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val = \
        split_data(X, y, w)

    model, saved_path = run_training(
        X_train, X_val, y_train, y_val, w_train,
        finetune=args.finetune,
    )

    accuracy = evaluate(model, X_test, y_test, w_test=None)

    print(f"\n{'='*60}")
    print(f"  DONE — Test accuracy: {accuracy * 100:.1f}%")
    print(f"  Model: {saved_path}")
    print("=" * 60)

    if accuracy >= 0.90:
        print("\n  ✅ Target accuracy (>90%) achieved")
    else:
        print("\n  ⚠️  Below 90% — consider more real recordings or adjusting weights")

    if args.finetune:
        print(f"\n  Next: run quantize.py (point it at gragas_dscnn_finetuned.keras)")
    else:
        print(f"\n  Next: python evaluate.py  or  python train.py --finetune")


if __name__ == "__main__":
    main()