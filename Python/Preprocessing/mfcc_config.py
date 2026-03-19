"""
MFCC Configuration — Single source of truth
=============================================
Every parameter here must be mirrored exactly in the C firmware
(CMSIS-DSP implementation) to guarantee feature parity between
training on PC and inference on STM32.
"""

from pathlib import Path

# ── Audio ──────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000          # Hz  (matches INMP441 / SAI config)
DURATION_S      = 1               # seconds per clip
NUM_SAMPLES     = SAMPLE_RATE * DURATION_S   # 16 000

# ── Pre-emphasis ───────────────────────────────────────────────────
PRE_EMPHASIS    = 0.97

# ── MFCC parameters ───────────────────────────────────────────────
N_MFCC          = 40
N_MELS          = 40
N_FFT           = 1024
WIN_LENGTH      = 640             # 40 ms
HOP_LENGTH      = 160             # 10 ms
FMIN            = 20.0
FMAX            = 8000.0
WINDOW          = "hann"
CENTER          = False

# ── Derived geometry ───────────────────────────────────────────────
NUM_FRAMES      = 1 + (NUM_SAMPLES - WIN_LENGTH) // HOP_LENGTH   # 97
FEATURE_SHAPE   = (NUM_FRAMES, N_MFCC)                           # (97, 40)

# ── Paths (relative to Python/) ────────────────────────────────────
_PYTHON_DIR     = Path(__file__).resolve().parent.parent
DATASET_DIR     = _PYTHON_DIR / "dataset"
FEATURES_DIR    = _PYTHON_DIR / "features"

# ── Manifest (produced by generate_wakeword_dataset.py) ───────────
# extract_mfcc.py reads this to discover all WAV files + weights.
# If it does not exist, extract_mfcc.py falls back to directory scan.
MANIFEST_PATH   = DATASET_DIR / "dataset_manifest.csv"

# ── Fallback directory scan (used when no manifest exists) ─────────
# Points to the augmented real recordings + TTS subdirectories.
# When running with manifest these are ignored.
WAKEWORD_DIRS   = [
    DATASET_DIR / "wakeword" / "augmented",   # real recordings (primary)
    DATASET_DIR / "wakeword" / "tts",         # TTS diversity (supporting)
]
NEGATIVE_DIRS   = [
    DATASET_DIR / "negative" / "augmented",
    DATASET_DIR / "negative" / "tts",
]

# Legacy single-dir names kept for backward compatibility with evaluate.py
WAKEWORD_DIR    = DATASET_DIR / "wakeword" / "augmented"
NEGATIVE_DIR    = DATASET_DIR / "negative" / "augmented"

# ── Labels ─────────────────────────────────────────────────────────
LABEL_NEGATIVE  = 0
LABEL_WAKEWORD  = 1
LABEL_NAMES     = {LABEL_NEGATIVE: "negative", LABEL_WAKEWORD: "wakeword"}