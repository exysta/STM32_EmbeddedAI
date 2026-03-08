"""
Training Configuration — Phase 3
==================================
All training hyperparameters in one place.
"""

import sys
from pathlib import Path

# ── Ensure Preprocessing config is importable ──────────────────────
_PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PYTHON_DIR / "Preprocessing"))
from mfcc_config import FEATURE_SHAPE, N_MFCC, NUM_FRAMES  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────
FEATURES_DIR    = _PYTHON_DIR / "features"
MODELS_DIR      = _PYTHON_DIR / "models"

# ── Classes ────────────────────────────────────────────────────────
NUM_CLASSES     = 2
CLASS_NAMES     = ["negative", "wakeword"]

# ── Data split ─────────────────────────────────────────────────────
TEST_SIZE       = 0.15          # 15 % test
VAL_SIZE        = 0.15          # 15 % val  (of the remaining 85 %)
RANDOM_SEED     = 42

# ── Model architecture ─────────────────────────────────────────────
INPUT_SHAPE     = (NUM_FRAMES, N_MFCC, 1)    # (97, 40, 1)
FIRST_CONV_FILTERS   = 64
FIRST_CONV_KERNEL    = (4, 10)               # spans 4 time frames × 10 MFCCs
DS_CONV_BLOCKS       = 4
DS_CONV_KERNEL       = (3, 3)
SE_REDUCTION_RATIO   = 4                     # squeeze-and-excitation bottleneck
DROPOUT_RATE         = 0.4

# ── SpecAugment (training-time only) ───────────────────────────────
SPEC_AUGMENT        = True
MAX_TIME_MASK       = 10        # mask up to 10 consecutive time frames
MAX_FREQ_MASK       = 5         # mask up to 5 consecutive MFCC bands

# ── Training ───────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 100
LEARNING_RATE   = 1e-3

# ReduceLROnPlateau
LR_FACTOR       = 0.5
LR_PATIENCE     = 5
LR_MIN          = 1e-6

# EarlyStopping
ES_PATIENCE     = 10
ES_RESTORE_BEST = True
