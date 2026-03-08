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
# y[n] = x[n] - coeff * x[n-1]
# Boosts high frequencies to compensate for spectral tilt of speech.
# Set to 0.0 to disable.
PRE_EMPHASIS    = 0.97

# ── MFCC parameters ───────────────────────────────────────────────
N_MFCC          = 40              # cepstral coefficients to keep
N_MELS          = 40              # Mel filter banks
N_FFT           = 1024            # FFT size (must be ≥ WIN_LENGTH, power of 2 for CMSIS arm_rfft_f32)
WIN_LENGTH      = 640             # window in samples  (40 ms)
HOP_LENGTH      = 160             # hop    in samples  (10 ms)
FMIN            = 20.0            # Hz  — lower edge of first Mel filter
FMAX            = 8000.0          # Hz  — upper edge (= Nyquist at 16 kHz)
WINDOW          = "hann"          # window function
CENTER          = False           # no zero-padding — matches on-device behaviour

# ── Derived geometry ───────────────────────────────────────────────
NUM_FRAMES      = 1 + (NUM_SAMPLES - WIN_LENGTH) // HOP_LENGTH   # 97
FEATURE_SHAPE   = (NUM_FRAMES, N_MFCC)                           # (97, 40)

# ── Paths (relative to Python/) ────────────────────────────────────
_PYTHON_DIR     = Path(__file__).resolve().parent.parent
DATASET_DIR     = _PYTHON_DIR / "dataset"
WAKEWORD_DIR    = DATASET_DIR / "wakeword" / "augmented"
NEGATIVE_DIR    = DATASET_DIR / "negative"
FEATURES_DIR    = _PYTHON_DIR / "features"

# ── Labels ─────────────────────────────────────────────────────────
LABEL_NEGATIVE  = 0
LABEL_WAKEWORD  = 1
LABEL_NAMES     = {LABEL_NEGATIVE: "negative", LABEL_WAKEWORD: "wakeword"}
