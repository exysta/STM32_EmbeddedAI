"""
MFCC Feature Extraction — Phase 2
===================================
Converts the WAV dataset into MFCC feature arrays ready for DS-CNN training.

Manifest mode (default when dataset_manifest.csv exists):
    Reads file paths, labels, and per-sample weights from the manifest
    produced by generate_wakeword_dataset.py.  Real INMP441 recordings
    carry weight=4.0, TTS carries weight=1.0.

Fallback mode (no manifest):
    Scans wakeword/augmented/ and negative/augmented/ as before.
    All samples get weight=1.0.

Usage:
    cd Python/Preprocessing
    python extract_mfcc.py

Outputs:
    ../features/X.npy             — (N, 97, 40)  float32, normalised MFCCs
    ../features/y.npy             — (N,)          int32,   class labels
    ../features/weights.npy       — (N,)          float32, per-sample weights
    ../features/norm_stats.npz    — mean/std (40,) for on-device normalisation
"""

import csv
import sys
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from mfcc_config import (
    SAMPLE_RATE, NUM_SAMPLES, PRE_EMPHASIS,
    N_MFCC, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH,
    FMIN, FMAX, WINDOW, CENTER,
    NUM_FRAMES, FEATURE_SHAPE,
    DATASET_DIR, MANIFEST_PATH,
    WAKEWORD_DIRS, NEGATIVE_DIRS,
    FEATURES_DIR,
    LABEL_WAKEWORD, LABEL_NEGATIVE, LABEL_NAMES,
)


# ── Audio helpers ──────────────────────────────────────────────────

def load_wav(path):
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def pad_or_trim(audio):
    if len(audio) < NUM_SAMPLES:
        pad = NUM_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    elif len(audio) > NUM_SAMPLES:
        start = (len(audio) - NUM_SAMPLES) // 2
        audio = audio[start: start + NUM_SAMPLES]
    return audio


def apply_preemphasis(audio):
    if PRE_EMPHASIS <= 0:
        return audio
    return np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])


def compute_mfcc(audio):
    audio = apply_preemphasis(audio)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX, window=WINDOW, center=CENTER,
    )
    mfcc = mfcc.T  # (n_mfcc, time) → (time, n_mfcc)
    if mfcc.shape[0] > NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES, :]
    elif mfcc.shape[0] < NUM_FRAMES:
        mfcc = np.pad(mfcc, ((0, NUM_FRAMES - mfcc.shape[0]), (0, 0)), mode="edge")
    return mfcc


# ── File discovery ────────────────────────────────────────────────

def collect_from_manifest(manifest_path: Path) -> list[tuple[Path, int, float, str]]:
    """
    Read dataset_manifest.csv → list of (path, label, weight, source).
    Skips rows where the WAV file no longer exists on disk.
    """
    entries = []
    missing = 0
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(row["path"])
            if not p.exists():
                missing += 1
                continue
            entries.append((
                p,
                int(row["label"]),
                float(row["weight"]),
                row["source"],
            ))
    if missing:
        print(f"  [WARN] {missing} manifest entries point to missing files — skipped.")
    return entries


def collect_from_dirs() -> list[tuple[Path, int, float, str]]:
    """
    Fallback: scan WAKEWORD_DIRS and NEGATIVE_DIRS for *.wav files.
    All samples receive weight=1.0 and source='unknown'.
    """
    entries = []
    for d in WAKEWORD_DIRS:
        if d.exists():
            for wav in sorted(d.glob("*.wav")):
                entries.append((wav, LABEL_WAKEWORD, 1.0, "unknown"))
    for d in NEGATIVE_DIRS:
        if d.exists():
            for wav in sorted(d.glob("*.wav")):
                entries.append((wav, LABEL_NEGATIVE, 1.0, "unknown"))
    return entries


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MFCC FEATURE EXTRACTION")
    print("=" * 60)

    # ── 1. Discover files ─────────────────────────────────────────
    if MANIFEST_PATH.exists():
        print(f"  Mode     : manifest  ({MANIFEST_PATH.name})")
        entries = collect_from_manifest(MANIFEST_PATH)
    else:
        print(f"  Mode     : directory scan  (no manifest found)")
        print(f"  Tip      : run generate_wakeword_dataset.py first for")
        print(f"             weighted training with real recordings.")
        entries = collect_from_dirs()

    if not entries:
        print("  ERROR: no WAV files found. Check your dataset directory.")
        sys.exit(1)

    n_ww  = sum(1 for _, l, _, _ in entries if l == LABEL_WAKEWORD)
    n_neg = sum(1 for _, l, _, _ in entries if l == LABEL_NEGATIVE)

    # Source breakdown for manifest mode
    from collections import Counter
    src_counts = Counter(s for _, _, _, s in entries)

    print(f"  Found    : {len(entries)} WAV files  "
          f"({n_ww} wakeword, {n_neg} negative)")
    if src_counts:
        for src, cnt in sorted(src_counts.items()):
            print(f"             {src:>10}: {cnt} files")
    print(f"  Output   : {FEATURES_DIR.resolve()}")
    print()

    # ── 2. Extract MFCCs ─────────────────────────────────────────
    X_list, y_list, w_list = [], [], []
    errors = 0

    for filepath, label, weight, _ in tqdm(entries, desc="  Extracting"):
        try:
            audio = pad_or_trim(load_wav(filepath))
            mfcc  = compute_mfcc(audio)
            X_list.append(mfcc)
            y_list.append(label)
            w_list.append(weight)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  WARN: skipped {filepath.name}: {e}")

    X = np.array(X_list, dtype=np.float32)   # (N, 97, 40)
    y = np.array(y_list, dtype=np.int32)     # (N,)
    w = np.array(w_list, dtype=np.float32)   # (N,)

    print(f"\n  Extracted {X.shape[0]} samples  ({errors} errors)")

    # ── 3. Per-feature normalisation ──────────────────────────────
    # Compute mean/std over ALL samples and ALL time frames,
    # independently per MFCC coefficient → shape (N_MFCC,).
    # Weight-aware: real samples contribute more to the statistics
    # so the normalisation is calibrated toward real-mic distribution.
    w_broadcast = w[:, np.newaxis, np.newaxis]          # (N, 1, 1)
    w_sum       = w_broadcast.sum()

    mean = (X * w_broadcast).sum(axis=(0, 1)) / (w_sum / X.shape[1])
    # Simpler unweighted mean/std is fine for normalisation —
    # using unweighted keeps norm_stats.npz compatible with the C firmware
    mean = X.mean(axis=(0, 1))
    std  = X.std(axis=(0, 1))
    std[std < 1e-6] = 1e-6

    X_norm = (X - mean) / std

    # ── 4. Shuffle ───────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(len(y))
    X_norm = X_norm[idx]
    y      = y[idx]
    w      = w[idx]

    # ── 5. Save ──────────────────────────────────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "X.npy",       X_norm)
    np.save(FEATURES_DIR / "y.npy",       y)
    np.save(FEATURES_DIR / "weights.npy", w)
    np.savez(FEATURES_DIR / "norm_stats.npz", mean=mean, std=std)

    # ── Summary ──────────────────────────────────────────────────
    real_mask = w > 1.0
    print(f"\n  Saved to: {FEATURES_DIR.resolve()}")
    print(f"    X.npy        : {X_norm.shape}  float32  "
          f"({X_norm.nbytes / 1e6:.1f} MB)")
    print(f"    y.npy        : {y.shape}  int32")
    print(f"    weights.npy  : {w.shape}  float32  "
          f"(real={real_mask.sum()}, tts/synth={( ~real_mask).sum()})")
    print(f"    norm_stats   : mean + std  ({N_MFCC} coefficients)")

    print(f"\n  Class distribution:")
    for lid, name in LABEL_NAMES.items():
        mask = y == lid
        r = (w[mask] > 1.0).sum()
        t = (w[mask] <= 1.0).sum()
        print(f"    {name:>10} : {mask.sum():>5}  (real={r}, tts/synth={t})")

    print(f"\n  Norm stats (first 5 coefficients):")
    print(f"    mean : {np.array2string(mean[:5], precision=2)}")
    print(f"    std  : {np.array2string(std[:5],  precision=2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()