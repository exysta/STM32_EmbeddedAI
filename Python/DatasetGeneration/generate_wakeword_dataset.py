"""
generate_wakeword_dataset.py  —  Real-primary dataset generator
================================================================
Strategy
--------
Real INMP441 recordings are the PRIMARY source (weighted 4× in training).
TTS voices are a SUPPORTING source only — kept for voice diversity so
the model doesn't overfit to a single voice/mic combination.

Recommended recording counts (record_dataset.py → then run this):

  Wakeword (real):   80 samples  ×  20× augmentation  =  1 600 clips
  Negative (real):   80 samples  ×  15× augmentation  =  1 200 clips
  TTS wakeword:     250 synthetic                      =    250 clips
  TTS/synth neg:    250 synthetic                      =    250 clips
  ─────────────────────────────────────────────────────────────────────
  Total wakeword : ~1 850   Total negative : ~1 450
  Train weight   : real samples carry SAMPLE_WEIGHT_REAL  (default 4.0)

Minimum viable (30 real each → good enough to ship):
  Wakeword: 30 × 20 + 200 TTS = 800
  Negative: 30 × 15 + 200 TTS = 650

Directory layout expected (produced by record_dataset.py):
  dataset/
  ├── wakeword/          ← trimmed 1s real recordings  (wakeword_NNNN.wav)
  ├── negative/          ← trimmed 1s real recordings  (negative_NNNN.wav)
  └── debug/             ← raw 1.5s recordings (not used for training)

Output:
  dataset/
  ├── wakeword/augmented/   ← augmented real wakeword clips
  ├── wakeword/tts/         ← TTS wakeword clips  (kept separate)
  ├── negative/augmented/   ← augmented real negative clips
  ├── negative/tts/         ← TTS/synth negative clips
  └── dataset_manifest.csv  ← path, label, source, weight — feed to train.py

Requirements:
    pip install edge-tts pydub audiomentations numpy scipy soundfile tqdm

Usage:
    # Record first with record_dataset.py, then:
    python generate_wakeword_dataset.py

    # Skip TTS generation (use only real recordings):
    python generate_wakeword_dataset.py --no-tts

    # Override real recording directory:
    python generate_wakeword_dataset.py --dataset-dir /path/to/dataset
"""

import os
import asyncio
import random
import shutil
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16_000

# ── Target augmented counts ───────────────────────────────────────────────────
# These are the total clips written to augmented/ directories.
# If you have fewer real samples, the script augments more aggressively.
TARGET_WAKEWORD_AUG = 1600   # real wakeword clips after augmentation
TARGET_NEGATIVE_AUG = 1200   # real negative clips after augmentation

# ── TTS supporting set size ───────────────────────────────────────────────────
# Kept small on purpose — diversity only, not the primary signal.
TARGET_TTS_WAKEWORD = 300
TARGET_TTS_NEGATIVE = 300

# ── Training weights (written to manifest, used in train.py) ─────────────────
SAMPLE_WEIGHT_REAL = 4.0    # real INMP441 recordings count 4× in loss
SAMPLE_WEIGHT_TTS  = 1.0    # TTS samples count 1× (normal)

# ── Augmentation intensity ───────────────────────────────────────────────────
# Real wakeword: ~20× augmentation factor per source clip
# Real negative: ~15× augmentation factor per source clip
# TTS: light augmentation only — don't over-process already-synthetic audio
AUG_ROUNDS_WAKEWORD = 15    # augmented copies per real wakeword sample
AUG_ROUNDS_NEGATIVE = 10    # augmented copies per real negative sample
AUG_ROUNDS_TTS      = 5     # TTS used as-is, no extra augmentation

# ── TTS voices (supporting diversity only) ───────────────────────────────────
# Reduced from the original 7 voices × 7 text variants × 5 prosody = 245 clips
# to just 4 voices × 4 text variants × 2 prosody = 32 base clips → augmented
TTS_VOICES = [
    ("fr-FR-HenriNeural",    "Male",   "fr-FR"),
    ("fr-FR-DeniseNeural",   "Female", "fr-FR"),
    ("fr-CA-AntoineNeural",  "Male",   "fr-CA"),
    ("fr-CA-SylvieNeural",   "Female", "fr-CA"),
]

TTS_TEXT_VARIANTS = [
    "gragasse",
    "GRAGASSE",
    "gragace",
    "gragas's",
]

TTS_PROSODY_VARIANTS = [
    ("+0%",   "+0Hz"),
    ("-10%",  "-5Hz"),
]

# ── Negative TTS words ───────────────────────────────────────────────────────
NEGATIVE_WORDS = [
    "bonjour", "merci", "oui", "non", "salut", "au revoir", "bonsoir",
    "comment", "voila", "pardon", "excusez", "monsieur", "madame",
    "bien", "tres", "stop", "pause", "lance", "jouer", "commencer",
    "lumiere", "musique", "telephone", "heure", "meteo", "agenda",
    "rappelle", "appelle", "cherche", "allume", "eteins", "volume",
    # Phonetically close to Gragas — critical negatives
    "grâce", "garage", "fracas", "cravas", "fragile", "gras",
    "agrafe", "grabuge", "grappe", "crachat",
    # Sibilant-heavy — directly targets the ss overweighting
    "ça suffit", "sensation", "satisfaction", "association",
    "passe", "masse", "classe", "casse", "asse",
    "siffler", "sifflet", "siffle",
    "six", "seize", "soixante",
    "resse", "fesse", "baisse", "caisse","aahh","pièces","mousse","pousse"
    # "-ace" / "-asse" endings that sound like "gragasse" suffix
    "surface", "interface", "grimace", "menace", "place",
    "fesses",
]


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path))
    if sr != SAMPLE_RATE:
        import scipy.signal as sig
        data = sig.resample(data, int(len(data) * SAMPLE_RATE / sr))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def save_wav(path: Path, data: np.ndarray):
    sf.write(str(path), np.clip(data, -1.0, 1.0), SAMPLE_RATE)


def pad_or_trim_center(audio: np.ndarray, target: int = SAMPLE_RATE) -> np.ndarray:
    """Pad short clips (center-pad) or trim long clips around peak energy."""
    if len(audio) < target:
        pad = target - len(audio)
        return np.pad(audio, (pad // 2, pad - pad // 2))

    if len(audio) == target:
        return audio

    # Trim: find highest-energy 1-second window (10 ms hop)
    hop = 160
    best_rms, best_start = -1.0, 0
    for start in range(0, len(audio) - target + 1, hop):
        rms = float(np.sqrt(np.mean(audio[start:start + target] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start
    return audio[best_start:best_start + target].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def build_augmenter_real_wakeword():
    """
    Aggressive augmentation for real wakeword recordings.
    Simulates different distances, loudness, room conditions, and mic positions.
    """
    import audiomentations as A
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.020, p=0.7),
        A.TimeStretch(min_rate=0.82, max_rate=1.18, p=0.6),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
        A.Shift(min_shift=-0.25, max_shift=0.25, p=0.6),
        A.Gain(min_gain_db=-8, max_gain_db=6, p=0.6),
        A.LowPassFilter(min_cutoff_freq=2500, max_cutoff_freq=7500, p=0.35),
        A.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=300, p=0.35),
        # Room acoustics simulation
        A.RoomSimulator(
            min_size_x=3.0, max_size_x=8.0,
            min_size_y=3.0, max_size_y=8.0,
            min_size_z=2.5, max_size_z=3.5,
            min_source_x=0.1, max_source_x=0.5,
            min_source_y=0.1, max_source_y=0.5,
            p=0.4
        ),
    ])


def build_augmenter_real_negative():
    """
    Moderate augmentation for real negative recordings.
    Avoid over-augmenting silence — it should still look like silence.
    """
    import audiomentations as A
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.015, p=0.6),
        A.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        A.Shift(min_shift=-0.3, max_shift=0.3, p=0.5),
        A.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        A.LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=7500, p=0.25),
    ])


def build_augmenter_tts():
    """Light augmentation for TTS — just enough to reduce over-fitting."""
    import audiomentations as A
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.TimeStretch(min_rate=0.88, max_rate=1.12, p=0.4),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
        A.Gain(min_gain_db=-4, max_gain_db=4, p=0.4),
    ])


def augment_sample(audio: np.ndarray, augmenter) -> np.ndarray:
    """Apply augmenter and re-trim to exactly 1 second."""
    aug = augmenter(samples=audio, sample_rate=SAMPLE_RATE)
    return pad_or_trim_center(aug)


# ══════════════════════════════════════════════════════════════════════════════
#  TTS GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def synthesize_edge(text: str, voice: str, output_path: Path,
                           rate: str = "+0%", pitch: str = "+0Hz"):
    import edge_tts
    from pydub import AudioSegment
    import tempfile

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    await communicate.save(tmp)

    audio = AudioSegment.from_mp3(tmp)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    audio.export(str(output_path), format="wav")
    os.unlink(tmp)


async def generate_tts_wakeword(out_dir: Path) -> list[Path]:
    """
    Generate TTS wakeword base samples.
    4 voices × 4 text variants × 2 prosody = 32 base clips.
    Each gets AUG_ROUNDS_TTS augmentation → ~32 clips total (kept minimal).
    """
    print("\n  [TTS] Generating wakeword base clips (supporting diversity)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = build_augmenter_tts()
    paths = []
    idx = 0

    total = len(TTS_VOICES) * len(TTS_TEXT_VARIANTS) * len(TTS_PROSODY_VARIANTS)
    pbar = tqdm(total=total, desc="  TTS wakeword")

    for voice, _, _ in TTS_VOICES:
        for text in TTS_TEXT_VARIANTS:
            for rate, pitch in TTS_PROSODY_VARIANTS:
                tmp_path = out_dir / f"_tts_tmp_{idx}.wav"
                try:
                    await synthesize_edge(text, voice, tmp_path, rate=rate, pitch=pitch)
                    audio = pad_or_trim_center(load_wav(tmp_path))
                    # Save the clean base + a few augmented copies
                    for aug_i in range(max(1, TARGET_TTS_WAKEWORD // total)):
                        out_path = out_dir / f"tts_wakeword_{idx:04d}_{aug_i}.wav"
                        save_wav(out_path, augment_sample(audio, aug) if aug_i > 0 else audio)
                        paths.append(out_path)
                    if tmp_path.exists():
                        tmp_path.unlink()
                    idx += 1
                except Exception as e:
                    print(f"\n    [WARN] TTS failed ({voice}, {text!r}): {e}")
                pbar.update(1)

    pbar.close()
    print(f"  [TTS] {len(paths)} wakeword clips written to {out_dir}")
    return paths


async def generate_tts_negative(out_dir: Path) -> list[Path]:
    """
    Generate TTS negative samples. Mix of French words + synth noise.
    Phonetically close negatives (grâce, garage, fracas) are prioritised.
    """
    print("\n  [TTS] Generating negative base clips...")
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = build_augmenter_tts()
    paths = []
    idx = 0
    pbar = tqdm(total=TARGET_TTS_NEGATIVE, desc="  TTS negative")

    tts_target   = int(TARGET_TTS_NEGATIVE * 0.7)   # 70% TTS words
    synth_target = TARGET_TTS_NEGATIVE - tts_target  # 30% synth noise

    # TTS words
    while idx < tts_target:
        word  = random.choice(NEGATIVE_WORDS)
        voice = random.choice(TTS_VOICES)[0]
        rate  = random.choice(["-10%", "+0%", "+10%"])
        tmp   = out_dir / f"_neg_tmp_{idx}.wav"
        try:
            await synthesize_edge(word, voice, tmp, rate=rate)
            audio = pad_or_trim_center(load_wav(tmp))
            out_path = out_dir / f"tts_negative_{idx:04d}.wav"
            save_wav(out_path, augment_sample(audio, aug))
            paths.append(out_path)
            if tmp.exists(): tmp.unlink()
            idx += 1
            pbar.update(1)
        except Exception:
            if tmp.exists(): tmp.unlink()
            continue

    # Synthetic noise (Gaussian + silence) — fills the gap quickly
    for _ in range(synth_target):
        noise_type = random.choice(["silence", "white", "pink_approx"])
        if noise_type == "silence":
            audio = np.random.normal(0, 0.0015, SAMPLE_RATE).astype(np.float32)
        elif noise_type == "white":
            audio = np.random.normal(0, random.uniform(0.01, 0.06),
                                     SAMPLE_RATE).astype(np.float32)
        else:  # pink-ish (1/f approximation via cumsum)
            white = np.random.normal(0, 1, SAMPLE_RATE * 2).astype(np.float32)
            pink  = np.cumsum(white)
            pink  = pink[SAMPLE_RATE:]   # skip transient
            pink  = pink / (np.max(np.abs(pink)) + 1e-9) * random.uniform(0.02, 0.08)
            audio = pink[:SAMPLE_RATE]

        out_path = out_dir / f"synth_negative_{idx:04d}.wav"
        save_wav(out_path, audio)
        paths.append(out_path)
        idx += 1
        pbar.update(1)

    pbar.close()
    print(f"  [TTS] {len(paths)} negative clips written to {out_dir}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  REAL SAMPLE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_real_samples(
        source_dir: Path,
        out_dir: Path,
        aug_rounds: int,
        augmenter,
        label: str,
) -> list[Path]:
    """
    Load all WAV files from source_dir, apply augmenter aug_rounds times each,
    write results to out_dir.  Returns list of output paths.
    """
    source_files = sorted(source_dir.glob("*.wav"))
    # Exclude any debug or base subdirectory files
    source_files = [f for f in source_files if f.is_file()]

    if not source_files:
        print(f"  [WARN] No real {label} recordings found in {source_dir}")
        print(f"         Run record_dataset.py --mode {label} first.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n_src = len(source_files)
    total = n_src * aug_rounds

    print(f"\n  [REAL] Augmenting {n_src} real {label} recordings × {aug_rounds} "
          f"= {total} clips…")
    pbar = tqdm(total=total, desc=f"  Real {label}")

    idx = 0
    for src_path in source_files:
        try:
            audio = pad_or_trim_center(load_wav(src_path))
        except Exception as e:
            print(f"\n    [WARN] Could not load {src_path.name}: {e}")
            pbar.update(aug_rounds)
            continue

        # Always save one clean copy (no augmentation) as anchor
        clean_path = out_dir / f"real_{label}_{idx:05d}_clean.wav"
        save_wav(clean_path, audio)
        paths.append(clean_path)
        idx += 1
        pbar.update(1)

        # Augmented copies
        for aug_i in range(aug_rounds - 1):
            try:
                aug_audio = augment_sample(audio, augmenter)
                aug_path  = out_dir / f"real_{label}_{idx:05d}_aug{aug_i:02d}.wav"
                save_wav(aug_path, aug_audio)
                paths.append(aug_path)
                idx += 1
            except Exception:
                pass
            pbar.update(1)

    pbar.close()
    print(f"  [REAL] {len(paths)} clips written to {out_dir}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def write_manifest(dataset_dir: Path, all_clips: list[dict]):
    """
    Write dataset_manifest.csv with columns:
      path, label (0=negative / 1=wakeword), source (real/tts/synth), weight

    train.py reads this to build X, y, sample_weights arrays.
    """
    manifest_path = dataset_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "weight"])
        writer.writeheader()
        writer.writerows(all_clips)
    print(f"\n  Manifest written → {manifest_path}  ({len(all_clips)} entries)")
    return manifest_path


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_clips: list[dict], dataset_dir: Path):
    real_ww  = sum(1 for c in all_clips if c["label"] == 1 and c["source"] == "real")
    tts_ww   = sum(1 for c in all_clips if c["label"] == 1 and c["source"] == "tts")
    real_neg = sum(1 for c in all_clips if c["label"] == 0 and c["source"] == "real")
    tts_neg  = sum(1 for c in all_clips if c["label"] == 0 and c["source"] in ("tts", "synth"))
    total_ww  = real_ww + tts_ww
    total_neg = real_neg + tts_neg

    print("\n" + "═" * 60)
    print("  DATASET GENERATION COMPLETE")
    print("═" * 60)
    print(f"  Wakeword  total : {total_ww:>5}  "
          f"(real×{AUG_ROUNDS_WAKEWORD}: {real_ww}  +  TTS: {tts_ww})")
    print(f"  Negative  total : {total_neg:>5}  "
          f"(real×{AUG_ROUNDS_NEGATIVE}: {real_neg}  +  TTS/synth: {tts_neg})")
    print(f"  Real sample weight : {SAMPLE_WEIGHT_REAL}×  "
          f"(TTS: {SAMPLE_WEIGHT_TTS}×)")
    print(f"  Output directory   : {dataset_dir.resolve()}")
    print()
    print("  NEXT STEPS:")
    print("  1. Check a few files in wakeword/augmented/ and negative/augmented/")
    print("  2. Update train.py to read dataset_manifest.csv and use sample_weight")
    print("  3. python extract_mfcc.py  →  python train.py (fine-tune mode)")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

async def main(args):
    dataset_dir = Path(args.dataset_dir).resolve()

    real_ww_dir  = dataset_dir / "wakeword"
    real_neg_dir = dataset_dir / "negative"
    aug_ww_dir   = dataset_dir / "wakeword"  / "augmented"
    aug_neg_dir  = dataset_dir / "negative"  / "augmented"
    tts_ww_dir   = dataset_dir / "wakeword"  / "tts"
    tts_neg_dir  = dataset_dir / "negative"  / "tts"

    # Count existing real recordings for user feedback
    n_real_ww  = len(list(real_ww_dir.glob("*.wav")))   \
                 if real_ww_dir.exists() else 0
    n_real_neg = len(list(real_neg_dir.glob("*.wav")))  \
                 if real_neg_dir.exists() else 0

    print("\n" + "═" * 60)
    print("  Wakeword Dataset Generator  —  Real-primary edition")
    print("═" * 60)
    print(f"  Dataset dir      : {dataset_dir}")
    print(f"  Real wakeword    : {n_real_ww} recordings found")
    print(f"  Real negative    : {n_real_neg} recordings found")
    print(f"  TTS              : {'disabled (--no-tts)' if args.no_tts else 'enabled (supporting role)'}")
    print(f"  Aug rounds WW    : {AUG_ROUNDS_WAKEWORD}×  →  ~{n_real_ww * AUG_ROUNDS_WAKEWORD} clips")
    print(f"  Aug rounds neg   : {AUG_ROUNDS_NEGATIVE}×  →  ~{n_real_neg * AUG_ROUNDS_NEGATIVE} clips")
    print("═" * 60)

    if n_real_ww == 0 and n_real_neg == 0 and args.no_tts:
        print("\n[ERROR] No real recordings found and --no-tts is set. Nothing to do.")
        print("        Run record_dataset.py first to collect real samples.")
        sys.exit(1)

    if n_real_ww < 10:
        print(f"\n[WARN] Only {n_real_ww} real wakeword recordings found.")
        print("       Recommend at least 30 (ideally 80) for a good model.")
        print("       Run: python record_dataset.py --mode wakeword --count 80")

    if n_real_neg < 10:
        print(f"\n[WARN] Only {n_real_neg} real negative recordings found.")
        print("       Recommend at least 30 (ideally 80) for a good model.")
        print("       Run: python record_dataset.py --mode negative --count 80")

    all_clips = []

    # ── 1. Augment real wakeword recordings ───────────────────────────────────
    aug_wakeword = build_augmenter_real_wakeword()
    real_ww_paths = augment_real_samples(
        real_ww_dir, aug_ww_dir, AUG_ROUNDS_WAKEWORD, aug_wakeword, "wakeword"
    )
    for p in real_ww_paths:
        all_clips.append({
            "path":   str(p),
            "label":  1,
            "source": "real",
            "weight": SAMPLE_WEIGHT_REAL,
        })

    # ── 2. Augment real negative recordings ───────────────────────────────────
    aug_negative = build_augmenter_real_negative()
    real_neg_paths = augment_real_samples(
        real_neg_dir, aug_neg_dir, AUG_ROUNDS_NEGATIVE, aug_negative, "negative"
    )
    for p in real_neg_paths:
        all_clips.append({
            "path":   str(p),
            "label":  0,
            "source": "real",
            "weight": SAMPLE_WEIGHT_REAL,
        })

    # ── 3. TTS supporting set (optional) ─────────────────────────────────────
    if not args.no_tts:
        if not shutil.which("ffmpeg"):
            print("\n[WARN] ffmpeg not found — skipping TTS generation.")
            print("       Install ffmpeg or use --no-tts to suppress this warning.")
        else:
            tts_ww_paths = await generate_tts_wakeword(tts_ww_dir)
            for p in tts_ww_paths:
                all_clips.append({
                    "path":   str(p),
                    "label":  1,
                    "source": "tts",
                    "weight": SAMPLE_WEIGHT_TTS,
                })

            tts_neg_paths = await generate_tts_negative(tts_neg_dir)
            for p in tts_neg_paths:
                all_clips.append({
                    "path":   str(p),
                    "label":  0,
                    "source": "tts" if "synth" not in p.name else "synth",
                    "weight": SAMPLE_WEIGHT_TTS,
                })

    # ── 4. Write manifest ─────────────────────────────────────────────────────
    if all_clips:
        write_manifest(dataset_dir, all_clips)
        print_summary(all_clips, dataset_dir)
    else:
        print("\n[ERROR] No clips generated. Check your dataset directory and recordings.")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent / "dataset"

    parser = argparse.ArgumentParser(
        description="Generate wakeword dataset (real-primary, TTS supporting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(default_dataset),
                        help="Root dataset directory (must contain wakeword/ and negative/)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Skip TTS generation entirely — use only real recordings")
    parser.add_argument("--aug-wakeword", type=int, default=AUG_ROUNDS_WAKEWORD,
                        help="Augmentation rounds per real wakeword sample")
    parser.add_argument("--aug-negative", type=int, default=AUG_ROUNDS_NEGATIVE,
                        help="Augmentation rounds per real negative sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow CLI overrides
    AUG_ROUNDS_WAKEWORD = args.aug_wakeword
    AUG_ROUNDS_NEGATIVE = args.aug_negative
    asyncio.run(main(args))
