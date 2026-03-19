"""
Training Configuration — Phase 3
==================================
All training hyperparameters in one place.

Two modes:
  scratch   — train from random init (first time, or full retrain)
  finetune  — load existing .keras, lower LR, fewer epochs
              Use this when adding real INMP441 recordings to an
              existing TTS-trained model.
"""

import sys
from pathlib import Path

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
TEST_SIZE       = 0.15
VAL_SIZE        = 0.15
RANDOM_SEED     = 42

# ── Model architecture (unchanged) ────────────────────────────────
INPUT_SHAPE          = (NUM_FRAMES, N_MFCC, 1)
FIRST_CONV_FILTERS   = 64
FIRST_CONV_KERNEL    = (4, 10)
DS_CONV_BLOCKS       = 4
DS_CONV_KERNEL       = (3, 3)
SE_REDUCTION_RATIO   = 4
DROPOUT_RATE         = 0.4

# ── SpecAugment ────────────────────────────────────────────────────
SPEC_AUGMENT        = True
MAX_TIME_MASK       = 10
MAX_FREQ_MASK       = 5

# ── Scratch training (from random init) ───────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 100
LEARNING_RATE   = 1e-3

LR_FACTOR       = 0.5
LR_PATIENCE     = 5
LR_MIN          = 1e-6

ES_PATIENCE     = 10
ES_RESTORE_BEST = True

# ── Fine-tune training (from existing model) ──────────────────────
# Lower LR prevents overwriting what was already learned from TTS.
# Fewer epochs avoids overfitting to the smaller real dataset.
FINETUNE_LR          = 1e-4       # 10× lower than scratch
FINETUNE_EPOCHS      = 40         # early stopping will kick in sooner
FINETUNE_LR_PATIENCE = 4          # slightly more aggressive LR decay
FINETUNE_ES_PATIENCE = 8          # stop earlier to avoid overfit

# Path to the base model for fine-tuning
FINETUNE_BASE_MODEL  = MODELS_DIR / "gragas_dscnn.keras"
# Fine-tuned model is saved separately so the original is preserved
FINETUNE_OUTPUT_MODEL = MODELS_DIR / "gragas_dscnn_finetuned.keras"