"""
MFCC Feature Extraction — Phase 2
===================================
Converts the WAV dataset into MFCC feature arrays ready for DS-CNN training.

Usage:
    cd Python/Preprocessing
    python extract_mfcc.py

Inputs:
    ../dataset/wakeword/augmented/*.wav   (label 1)
    ../dataset/negative/*.wav             (label 0)

Outputs:
    ../features/X.npy           — (N, 97, 40)  float32, normalised MFCCs
    ../features/y.npy           — (N,)         int32,   class labels
    ../features/norm_stats.npz  — mean/std arrays (40,) for on-device normalisation

Requirements:
    pip install librosa numpy soundfile tqdm
"""

import sys
import numpy as np
import librosa
from tqdm import tqdm

# Import config from same package
from mfcc_config import (
    SAMPLE_RATE, NUM_SAMPLES, PRE_EMPHASIS,
    N_MFCC, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH,
    FMIN, FMAX, WINDOW, CENTER,
    NUM_FRAMES, FEATURE_SHAPE,
    WAKEWORD_DIR, NEGATIVE_DIR, FEATURES_DIR,
    LABEL_WAKEWORD, LABEL_NEGATIVE, LABEL_NAMES,
)


# ── Audio helpers ──────────────────────────────────────────────────

def load_wav(path):
    """Load WAV as mono float32 at SAMPLE_RATE."""
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def pad_or_trim(audio):
    """Centre-pad or centre-trim to exactly NUM_SAMPLES.

    Uses the same logic as the dataset generator so that the audio
    content stays centred in the 1-second window.
    """
    if len(audio) < NUM_SAMPLES:
        pad = NUM_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    elif len(audio) > NUM_SAMPLES:
        start = (len(audio) - NUM_SAMPLES) // 2
        audio = audio[start : start + NUM_SAMPLES]
    return audio


def apply_preemphasis(audio):
    """Pre-emphasis filter: y[n] = x[n] - α·x[n-1].

    Compensates for the natural ~6 dB/octave spectral roll-off of
    voiced speech, boosting high-frequency energy so that the Mel
    filter bank sees a flatter spectrum.
    """
    if PRE_EMPHASIS <= 0:
        return audio
    return np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])


# ── MFCC extraction ───────────────────────────────────────────────

def compute_mfcc(audio):
    """Extract MFCC matrix from a 1-second audio signal.

    Returns
    -------
    mfcc : ndarray, shape (NUM_FRAMES, N_MFCC)
        Transposed so that rows = time, columns = coefficients,
        matching the (height, width) convention expected by a 2-D CNN.
    """
    audio = apply_preemphasis(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        window=WINDOW,
        center=CENTER,
    )
    mfcc = mfcc.T  # (n_mfcc, time) → (time, n_mfcc)

    # Guarantee output shape matches config (handles ±1 frame edge cases)
    if mfcc.shape[0] > NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES, :]
    elif mfcc.shape[0] < NUM_FRAMES:
        pad_rows = NUM_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_rows), (0, 0)), mode="edge")

    return mfcc


# ── Dataset collection ─────────────────────────────────────────────

def collect_files():
    """Discover WAV files and assign labels.

    Returns list of (Path, label) tuples.
    """
    files = []

    if not WAKEWORD_DIR.exists():
        print(f"  ERROR: wakeword directory not found: {WAKEWORD_DIR}")
        sys.exit(1)
    if not NEGATIVE_DIR.exists():
        print(f"  ERROR: negative directory not found: {NEGATIVE_DIR}")
        sys.exit(1)

    for wav in sorted(WAKEWORD_DIR.glob("*.wav")):
        files.append((wav, LABEL_WAKEWORD))

    for wav in sorted(NEGATIVE_DIR.glob("*.wav")):
        files.append((wav, LABEL_NEGATIVE))

    return files


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MFCC FEATURE EXTRACTION — Phase 2")
    print("=" * 60)
    print(f"  Feature shape   : {FEATURE_SHAPE}  (frames × coefficients)")
    print(f"  Pre-emphasis    : {PRE_EMPHASIS}")
    print(f"  Window / Hop    : {WIN_LENGTH} / {HOP_LENGTH} samples"
          f"  ({WIN_LENGTH * 1000 // SAMPLE_RATE} ms / "
          f"{HOP_LENGTH * 1000 // SAMPLE_RATE} ms)")
    print(f"  FFT / Mel bands : {N_FFT} / {N_MELS}")
    print(f"  Dataset         : {DATASET_DIR.resolve()}")
    print(f"  Output          : {FEATURES_DIR.resolve()}")
    print()

    # ── 1. Discover files ───────────────────────────────────────────
    files = collect_files()
    n_ww  = sum(1 for _, l in files if l == LABEL_WAKEWORD)
    n_neg = sum(1 for _, l in files if l == LABEL_NEGATIVE)
    print(f"  Found {len(files)} WAV files  "
          f"({n_ww} wakeword, {n_neg} negative)\n")

    # ── 2. Extract MFCCs ────────────────────────────────────────────
    X_list, y_list = [], []
    errors = 0

    for filepath, label in tqdm(files, desc="  Extracting MFCCs"):
        try:
            audio = pad_or_trim(load_wav(filepath))
            mfcc  = compute_mfcc(audio)
            X_list.append(mfcc)
            y_list.append(label)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  WARN: skipped {filepath.name}: {e}")

    X = np.array(X_list, dtype=np.float32)   # (N, 97, 40)
    y = np.array(y_list, dtype=np.int32)     # (N,)

    print(f"\n  Extracted {X.shape[0]} samples  ({errors} errors)")
    print(f"  X shape (raw)  : {X.shape}")

    # ── 3. Per-feature normalization ────────────────────────────────
    #   mean/std computed over ALL samples and ALL time frames,
    #   independently per MFCC coefficient → shape (N_MFCC,)
    mean = X.mean(axis=(0, 1))
    std  = X.std(axis=(0, 1))
    std[std < 1e-6] = 1e-6              # guard against silent coefficients

    X_norm = (X - mean) / std            # broadcasts over (N, T, C)

    # ── 4. Shuffle ──────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(len(y))
    X_norm = X_norm[idx]
    y      = y[idx]

    # ── 5. Save ─────────────────────────────────────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "X.npy", X_norm)
    np.save(FEATURES_DIR / "y.npy", y)
    np.savez(FEATURES_DIR / "norm_stats.npz", mean=mean, std=std)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n  Saved to: {FEATURES_DIR.resolve()}")
    print(f"    X.npy           : {X_norm.shape}  float32  "
          f"({X_norm.nbytes / 1e6:.1f} MB)")
    print(f"    y.npy           : {y.shape}  int32")
    print(f"    norm_stats.npz  : mean + std  ({N_MFCC} values each)")

    print(f"\n  Class distribution:")
    for lid, name in LABEL_NAMES.items():
        print(f"    {name:>10} : {np.sum(y == lid)}")

    print(f"\n  Norm stats (first 5 coefficients):")
    print(f"    mean : {np.array2string(mean[:5], precision=2)}")
    print(f"    std  : {np.array2string(std[:5],  precision=2)}")
    print("=" * 60)


if __name__ == "__main__":
    from mfcc_config import DATASET_DIR   # needed for the banner
    main()
