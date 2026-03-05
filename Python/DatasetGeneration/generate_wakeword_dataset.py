"""
Wakeword Dataset Generator - "gragas" French accent
Uses edge-tts (Microsoft Neural TTS) - free, no API key, real male + female voices

Requirements:
    pip install edge-tts pydub audiomentations numpy scipy soundfile tqdm

Also needs ffmpeg:
    Windows : https://ffmpeg.org/download.html  (add to PATH)
    Mac     : brew install ffmpeg
    Linux   : sudo apt install ffmpeg

Usage : python generate_wakeword_dataset.py
Output: ./dataset/wakeword/  and  ./dataset/negative/
Time  : ~15-25 min
"""

import os, asyncio, random, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
WORD                   = "gragasse"
OUTPUT_DIR             = Path("./dataset")
WAKEWORD_DIR           = OUTPUT_DIR / "wakeword"
NEGATIVE_DIR           = OUTPUT_DIR / "negative"
SAMPLE_RATE            = 16000
TARGET_WAKEWORD        = 2000
TARGET_NEGATIVE        = 2000
# ─────────────────────────────────────────────────────────────────────────────

# ── French voices available in edge-tts ──────────────────────────────────────
# Run `edge-tts --list-voices | grep fr-FR` to see all
FRENCH_VOICES = [
    # Male voices
    ("fr-FR-HenriNeural",              "Male",   "fr-FR"),   # Standard French male
    ("fr-FR-RemyMultilingualNeural",   "Male",   "fr-FR"),   # Modern male
    ("fr-CA-AntoineNeural",            "Male",   "fr-CA"),   # Quebec male
    # Female voices
    ("fr-FR-DeniseNeural",             "Female", "fr-FR"),   # Standard French female
    ("fr-FR-EloiseNeural",             "Female", "fr-FR"),   # Childz-like female
    ("fr-CA-SylvieNeural",             "Female", "fr-CA"),   # Quebec female
    ("fr-BE-CharlineNeural",           "Female", "fr-BE"),   # Belgian female
]

# Text variants to vary prosody
TEXT_VARIANTS = [
    "gragasse",       # forces hard final S sound
    "GRAAgasse",       # hyphen may break the liaison rule  
    "gragas's",       # apostrophe tricks prosody
    "gragace",        # your idea - keep it
    "Gragas.",
    "GRAGASSE",   # word after forces the S via liaison
    "Gragace. ", # liaison forces pronunciation of S
]

# French negative words (NOT the wakeword)
NEGATIVE_WORDS = [
    "bonjour", "merci", "oui", "non", "salut", "au revoir", "bonsoir",
    "comment", "voila", "pardon", "excusez", "monsieur", "madame",
    "bien", "tres", "stop", "pause", "lance", "jouer", "commencer",
    "lumiere", "musique", "telephone", "heure", "meteo", "agenda",
    "rappelle", "appelle", "cherche", "allume", "eteins", "volume",
    "suivant", "precedent", "repete", "aide", "annule", "retour",
]
# ─────────────────────────────────────────────────────────────────────────────

def check_deps():
    import subprocess, sys
    pkgs = ["edge_tts", "pydub", "audiomentations", "soundfile", "tqdm"]
    missing = []
    for p in pkgs:
        try: __import__(p)
        except ImportError: missing.append(p.replace("_", "-"))
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found!")
        print("   Windows: https://ffmpeg.org/download.html")
        print("   Mac:     brew install ffmpeg")
        print("   Linux:   sudo apt install ffmpeg")
        exit(1)

async def synthesize_edge(text, voice, output_path, rate="+0%", pitch="+0Hz"):
    """Generate one audio clip via edge-tts and save as 16kHz mono WAV."""
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

def load_wav(path):
    data, sr = sf.read(str(path))
    if sr != SAMPLE_RATE:
        import scipy.signal as signal
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)

def save_wav(path, data):
    sf.write(str(path), np.clip(data, -1.0, 1.0), SAMPLE_RATE)

def pad_or_trim(audio, target=SAMPLE_RATE):
    if len(audio) < target:
        pad = target - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    else:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    return audio

def augment(audio):
    import audiomentations as A
    aug = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.025, p=0.6),
        A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        A.Shift(min_shift=-0.2, max_shift=0.2, p=0.4),
        A.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        A.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7000, p=0.3),
        A.HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=400, p=0.3),
    ])
    return aug(samples=audio, sample_rate=SAMPLE_RATE)

async def generate_base_samples():
    """Generate one clip per (voice x text x speed/pitch variant)."""
    print("\n  Generating TTS base samples (male + female French voices)...")
    base_dir = WAKEWORD_DIR / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Rate / pitch combos to get more natural variation
    prosody_variants = [
        ("+0%",   "+0Hz"),
        ("-10%",  "-5Hz"),
        ("+10%",  "+5Hz"),
        ("-15%",  "+0Hz"),
        ("+15%",  "+0Hz"),
    ]

    base_samples = []
    idx = 0
    total = len(FRENCH_VOICES) * len(TEXT_VARIANTS) * len(prosody_variants)
    pbar = tqdm(total=total, desc="  TTS base")

    for voice, gender, locale in FRENCH_VOICES:
        for text in TEXT_VARIANTS:
            for rate, pitch in prosody_variants:
                out = base_dir / f"base_{idx:04d}_{gender[0]}.wav"
                try:
                    await synthesize_edge(text, voice, out, rate=rate, pitch=pitch)
                    base_samples.append((out, gender))
                    idx += 1
                except Exception:
                    pass
                pbar.update(1)

    pbar.close()
    male_c   = sum(1 for _, g in base_samples if g == "Male")
    female_c = sum(1 for _, g in base_samples if g == "Female")
    print(f"   OK: {len(base_samples)} base samples  (male: {male_c}, female: {female_c})")
    return [p for p, _ in base_samples]

def augment_to_target(base_samples):
    print(f"\n  Augmenting to {TARGET_WAKEWORD} wakeword samples...")
    aug_dir = WAKEWORD_DIR / "augmented"
    aug_dir.mkdir(exist_ok=True)

    generated = 0
    pbar = tqdm(total=TARGET_WAKEWORD)
    while generated < TARGET_WAKEWORD:
        src = random.choice(base_samples)
        try:
            audio = pad_or_trim(load_wav(src))
            save_wav(aug_dir / f"wakeword_{generated:05d}.wav", augment(audio))
            generated += 1
            pbar.update(1)
        except Exception:
            continue
    pbar.close()
    print(f"   OK: {generated} wakeword samples saved")

async def generate_negative_samples():
    print(f"\n  Generating {TARGET_NEGATIVE} negative samples...")
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    pbar = tqdm(total=TARGET_NEGATIVE)

    while generated < TARGET_NEGATIVE:
        kind = random.choice(["tts", "silence", "noise"])
        try:
            if kind == "tts":
                word  = random.choice(NEGATIVE_WORDS)
                voice = random.choice(FRENCH_VOICES)[0]
                rate  = random.choice(["-10%", "+0%", "+10%"])
                tmp   = NEGATIVE_DIR / "tmp_neg.wav"
                await synthesize_edge(word, voice, tmp, rate=rate)
                audio = pad_or_trim(load_wav(tmp))
                audio = augment(audio)
                if tmp.exists(): os.unlink(str(tmp))
            elif kind == "silence":
                audio = np.random.normal(0, 0.002, SAMPLE_RATE).astype(np.float32)
            else:
                audio = np.random.normal(0, random.uniform(0.01, 0.08), SAMPLE_RATE).astype(np.float32)

            save_wav(NEGATIVE_DIR / f"negative_{generated:05d}.wav", audio)
            generated += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()
    print(f"   OK: {generated} negative samples saved")

def print_summary():
    ww  = len(list(WAKEWORD_DIR.rglob("*.wav")))
    neg = len(list(NEGATIVE_DIR.rglob("*.wav")))
    male_voices   = sum(1 for _, g, _ in FRENCH_VOICES if g == "Male")
    female_voices = sum(1 for _, g, _ in FRENCH_VOICES if g == "Female")
    print("\n" + "="*55)
    print("DATASET GENERATION COMPLETE")
    print("="*55)
    print(f"  Wakeword samples : {ww}")
    print(f"  Negative samples : {neg}")
    print(f"  Sample rate      : {SAMPLE_RATE} Hz, mono, 1 sec clips")
    print(f"  Voices used      : {len(FRENCH_VOICES)} ({male_voices} male, {female_voices} female)")
    print(f"  Output           : {OUTPUT_DIR.resolve()}")
    print("="*55)
    print("\nNEXT STEPS:")
    print("  1. Upload ./dataset/ to Edge Impulse -> https://edgeimpulse.com")
    print("     wakeword/ = keyword,  negative/ = noise")
    print("  2. Train DS-CNN -> Export TFLite -> Import in STM32CubeAI")
    print()

async def main():
    print("Wakeword Dataset Generator - edge-tts edition")
    print(f"  Word   : '{WORD}' (French, male + female)")
    print(f"  Target : {TARGET_WAKEWORD} wakeword + {TARGET_NEGATIVE} negative")
    print(f"  Time   : ~15-25 min\n")

    check_ffmpeg()
    check_deps()

    base = await generate_base_samples()
    augment_to_target(base)
    await generate_negative_samples()
    print_summary()

if __name__ == "__main__":
    asyncio.run(main())
