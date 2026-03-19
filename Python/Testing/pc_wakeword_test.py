"""
pc_wakeword_test.py
====================
Tests the wakeword model directly on your PC microphone,
mirroring the exact STM32 pipeline:

  mic → resample → pre-emphasis
      → MFCC (librosa, same params as mfcc_config.py)
      → normalize (norm_stats.npz)
      → quantize (INPUT_SCALE / INPUT_ZP from Cube.AI report)
      → TFLite inference
      → dequantize → P(wakeword)

Capture mode: SLIDING WINDOW
  Records audio continuously. Every SLIDE_HOP_S seconds a new 1-second
  window is evaluated. This guarantees the word is fully captured in at
  least one window regardless of when you speak relative to the timer.

  Example with SLIDE_HOP_S = 0.5:
    window 1 : t=0.0 … 1.0 s
    window 2 : t=0.5 … 1.5 s
    window 3 : t=1.0 … 2.0 s   ← "gragas" centred here → detected
    ...

Requirements:
    pip install numpy librosa sounddevice tflite-runtime scipy
    (or use 'tensorflow' instead of 'tflite-runtime' if already installed)

Usage:
    python pc_wakeword_test.py

    The script streams your microphone continuously and prints a result
    line for each window evaluated.
    Press Ctrl+C to quit.
"""

import sys
import os
import time
import threading
import collections
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime

# ── Try tflite-runtime first, fall back to full tensorflow ────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("[INFO] Using tflite-runtime")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using tensorflow.lite")
    except ImportError:
        print("[ERROR] Install tflite-runtime or tensorflow:")
        print("        pip install tflite-runtime")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — must match mfcc_config.py and your Cube.AI report exactly
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = r"D:\Work\Projects\STM32_EmbeddedAI\Python\models\gragas_dscnn_int8.tflite"
NORM_STATS_PATH = r"D:\Work\Projects\STM32_EmbeddedAI\Python\features\norm_stats.npz"

# ── Audio (mfcc_config.py) ─────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
NUM_SAMPLES  = 16_000          # 1 second
PRE_EMPHASIS = 0.97

# ── MFCC (mfcc_config.py) ──────────────────────────────────────────────────
N_MFCC      = 40
N_MELS      = 40
N_FFT       = 1024
WIN_LENGTH  = 640
HOP_LENGTH  = 160
FMIN        = 20.0
FMAX        = 8000.0
WINDOW      = "hann"
CENTER      = False
NUM_FRAMES  = 1 + (NUM_SAMPLES - WIN_LENGTH) // HOP_LENGTH   # 97

# ── Quantisation (from Cube.AI report, line 22) ───────────────────────────
# After (new model values — read directly from the warning output):
INPUT_SCALE  = 0.05794549733400345
INPUT_ZP     = 9
OUTPUT_SCALE = 0.003906
OUTPUT_ZP    = -128

# ── Detection threshold ────────────────────────────────────────────────────
WW_THRESHOLD = 0.85

# ── Debug WAV output ───────────────────────────────────────────────────────
# Three versions are saved per capture so you can hear exactly what each
# pipeline stage sounds like:
#   _raw.wav          : straight from the microphone (float32 -> int16)
#   _preemph.wav      : after pre-emphasis filter
#   _norm_preview.wav : after per-coefficient normalization, reconstructed
#                       from MFCCs via Griffin-Lim (confirms the MFCC stage)
# Set to None to disable saving entirely.
DEBUG_WAV_DIR = "debug_wav"

# Only save WAVs for detections (True) or every window (False)
SAVE_DETECTIONS_ONLY = True

# ── Sliding window config ──────────────────────────────────────────────────
# How often a new 1-second window is evaluated (seconds).
# Smaller = more responsive but more CPU. 0.25 is a good balance.
# Must be < 1.0 (the window length) to have overlap.
SLIDE_HOP_S = 0.25    # evaluate 4 windows per second

# After a detection, suppress further detections for this many seconds
# to avoid the same utterance triggering multiple times.
COOLDOWN_S  = 1.5

# ── Label mapping (must match Python training labels) ─────────────────────
LABEL_NAMES = {0: "negative", 1: "wakeword (gragas)"}

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def apply_preemphasis(audio):
    """y[n] = x[n] - 0.97·x[n-1] — matches extract_mfcc.py."""
    return np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])


def compute_mfcc(audio):
    """Compute (NUM_FRAMES, N_MFCC) MFCC array — identical to extract_mfcc.py."""
    audio = apply_preemphasis(audio)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX, window=WINDOW, center=CENTER,
    )
    mfcc = mfcc.T   # (N_MFCC, time) → (time, N_MFCC)

    # Guarantee shape — same guard as extract_mfcc.py
    if mfcc.shape[0] > NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES, :]
    elif mfcc.shape[0] < NUM_FRAMES:
        mfcc = np.pad(mfcc, ((0, NUM_FRAMES - mfcc.shape[0]), (0, 0)), mode="edge")

    return mfcc   # (97, 40)


def normalize(mfcc, mean, std):
    """Per-coefficient z-score normalization — matches extract_mfcc.py."""
    return (mfcc - mean) / std   # broadcasts over (97, 40)


def quantize(mfcc_norm):
    """Float32 → INT8 affine quantization matching WW_Quantise() in C."""
    q = np.round(mfcc_norm / INPUT_SCALE).astype(np.int32) + INPUT_ZP
    q = np.clip(q, -128, 127).astype(np.int8)
    return q   # (97, 40)


def dequantize_output(raw_int8):
    """INT8 → float32 dequantization matching aiRunInference() in C."""
    return (raw_int8.astype(np.float32) - OUTPUT_ZP) * OUTPUT_SCALE


def save_debug_wavs(capture_id, audio_raw, audio_preemph, mfcc_norm):
    """Save three diagnostic WAV files for a single capture.

    Parameters
    ----------
    capture_id   : str  — timestamp string used as filename prefix
    audio_raw    : (16000,) float32  — mic signal before any processing
    audio_preemph: (16000,) float32  — after pre-emphasis
    mfcc_norm    : (97, 40) float32  — normalised MFCC matrix

    Files written to DEBUG_WAV_DIR/:
      <id>_1_raw.wav          raw mic capture
      <id>_2_preemph.wav      after pre-emphasis
      <id>_3_mfcc_heatmap.npy normalised MFCC as numpy array (load in Python
                               to verify values — audio reconstruction from
                               40-band MFCC is too lossy to be useful)
    """
    if DEBUG_WAV_DIR is None:
        return

    os.makedirs(DEBUG_WAV_DIR, exist_ok=True)

    def to_int16(sig):
        """Clip float32 signal to [-1, 1] and convert to int16."""
        sig = np.clip(sig, -1.0, 1.0)
        return (sig * 32767).astype(np.int16)

    # 1 — Raw capture
    path_raw = os.path.join(DEBUG_WAV_DIR, f"{capture_id}_1_raw.wav")
    wav.write(path_raw, SAMPLE_RATE, to_int16(audio_raw))

    # 2 — After pre-emphasis
    path_pre = os.path.join(DEBUG_WAV_DIR, f"{capture_id}_2_preemph.wav")
    wav.write(path_pre, SAMPLE_RATE, to_int16(audio_preemph))

    # 3 — Normalised MFCC matrix as .npy (more useful than audio reconstruction)
    path_npy = os.path.join(DEBUG_WAV_DIR, f"{capture_id}_3_mfcc_norm.npy")
    np.save(path_npy, mfcc_norm)

    print(f"\n  💾 Debug files saved to '{DEBUG_WAV_DIR}/':")
    print(f"       {os.path.basename(path_raw)}"
          f"   RMS={np.sqrt(np.mean(audio_raw**2)):.5f}")
    print(f"       {os.path.basename(path_pre)}"
          f"  RMS={np.sqrt(np.mean(audio_preemph**2)):.5f}")
    print(f"       {os.path.basename(path_npy)}"
          f"  shape={mfcc_norm.shape}"
          f"  min={mfcc_norm.min():.2f}  max={mfcc_norm.max():.2f}")


def run_inference(interpreter, input_index, mfcc_int8):
    """Feed one (97, 40) INT8 feature map and return output probabilities."""
    # Model expects (1, 97, 40, 1) — add batch and channel dims
    inp = mfcc_int8.reshape(1, NUM_FRAMES, N_MFCC, 1)
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    raw = interpreter.get_tensor(output_details[0]['index'])   # (1, 2) int8
    probs = dequantize_output(raw[0])   # [P(negative), P(wakeword)]
    return probs


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — continuous sliding-window inference
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load norm stats ────────────────────────────────────────────────────
    stats = np.load(NORM_STATS_PATH)
    mean  = stats["mean"].astype(np.float32)   # (40,)
    std   = stats["std"].astype(np.float32)    # (40,)
    print(f"[INFO] Norm stats loaded  mean[0]={mean[0]:.3f}  std[0]={std[0]:.3f}")

    # ── Load TFLite model ──────────────────────────────────────────────────
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    print(f"[INFO] Model loaded")
    print(f"       Input  : {input_details[0]['shape']}  "
          f"dtype={input_details[0]['dtype'].__name__}  "
          f"scale={input_details[0]['quantization'][0]:.6f}  "
          f"zp={input_details[0]['quantization'][1]}")
    print(f"       Output : {output_details[0]['shape']}  "
          f"dtype={output_details[0]['dtype'].__name__}  "
          f"scale={output_details[0]['quantization'][0]:.6f}  "
          f"zp={output_details[0]['quantization'][1]}")

    # ── Sanity-check quant params match our constants ──────────────────────
    model_in_scale = input_details[0]['quantization'][0]
    model_in_zp    = input_details[0]['quantization'][1]
    if abs(model_in_scale - INPUT_SCALE) > 1e-6 or model_in_zp != INPUT_ZP:
        print(f"\n  ⚠️  WARNING: Quantisation mismatch!")
        print(f"     Script has INPUT_SCALE={INPUT_SCALE}, INPUT_ZP={INPUT_ZP}")
        print(f"     Model reports scale={model_in_scale}, zp={model_in_zp}")
        print(f"     Update INPUT_SCALE / INPUT_ZP in this script.\n")

    input_index = input_details[0]['index']

    # ── Ring buffer — holds WINDOW + one extra hop of samples ─────────────
    # We accumulate incoming audio chunks here and slice 1-second windows
    # out of it every SLIDE_HOP_S seconds.
    hop_samples    = int(SLIDE_HOP_S * SAMPLE_RATE)   # e.g. 4000 at 0.25 s
    ring_len       = NUM_SAMPLES + hop_samples         # 16000 + 4000 = 20000
    ring           = collections.deque(maxlen=ring_len)
    ring_lock      = threading.Lock()

    # Pre-fill with silence so the first window is always complete
    ring.extend(np.zeros(ring_len, dtype=np.float32))

    # ── Audio callback — runs in a background thread ───────────────────────
    def audio_callback(indata, frames, time_info, status):
        with ring_lock:
            ring.extend(indata[:, 0])   # mono

    # ── Inference worker ───────────────────────────────────────────────────
    last_detection_time = 0.0
    window_count        = 0

    print("\n" + "="*60)
    print(f"  Sliding window: 1 s window, hop={SLIDE_HOP_S} s "
          f"→ {int(1/SLIDE_HOP_S)} windows/s")
    print(f"  Threshold: {WW_THRESHOLD}   Cooldown: {COOLDOWN_S} s")
    print(f"  Say 'gragas' at any time — Ctrl+C to quit")
    print("="*60 + "\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=hop_samples,
                        callback=audio_callback):
        try:
            while True:
                time.sleep(SLIDE_HOP_S)
                window_count += 1

                # Grab a 1-second snapshot from the ring buffer
                with ring_lock:
                    audio_raw = np.array(list(ring)[-NUM_SAMPLES:],
                                         dtype=np.float32)

                # Skip nearly-silent frames (noise gate)
                rms = float(np.sqrt(np.mean(audio_raw ** 2)))
                if rms < 0.005:
                    print(f"  [{window_count:04d}]  silence  "
                          f"(RMS={rms:.4f})", end="\r")
                    continue

                # Full pipeline
                audio_preemph = audio_raw
                mfcc_raw      = compute_mfcc(audio_raw)
                
                mfcc_norm     = normalize(mfcc_raw, mean, std)
                mfcc_int8     = quantize(mfcc_norm)
                probs         = run_inference(interp, input_index, mfcc_int8)

                p_neg = float(probs[0])
                p_ww  = float(probs[1])
                now   = time.time()

                in_cooldown = (now - last_detection_time) < COOLDOWN_S
                detected    = (p_ww >= WW_THRESHOLD) and not in_cooldown

                # Build bar
                bar = int(p_ww * 20)
                marker = "✅ DETECTED" if detected else (
                         "🔇 cooldown" if in_cooldown and p_ww >= WW_THRESHOLD
                         else "")

                print(f"  [{window_count:04d}]  "
                      f"P(ww)={p_ww:.3f} |{'█'*bar:<20}|  "
                      f"RMS={rms:.4f}  {marker}")

                if detected:
                    last_detection_time = now
                    capture_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_debug_wavs(capture_id, audio_raw,
                                    audio_preemph, mfcc_norm)

                elif not SAVE_DETECTIONS_ONLY:
                    capture_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_debug_wavs(capture_id, audio_raw,
                                    audio_preemph, mfcc_norm)

        except KeyboardInterrupt:
            print("\n\n  Stopped.")


if __name__ == "__main__":
    main()