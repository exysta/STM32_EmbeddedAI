"""
test_wav_wakeword.py
====================
Runs the exact same wakeword inference pipeline as pc_wakeword_test.py
but on a pre-recorded WAV file instead of the PC microphone.

Use this to compare:
  1. Audio from your PC mic  →  pc_wakeword_test.py     (works)
  2. Audio from the INMP441  →  test_wav_wakeword.py    (this script)

If the model fires on (1) but not on (2), the problem is in the
STM32 audio capture / MFCC pipeline, not the model.

Usage:
    python test_wav_wakeword.py --wav inmp441_capture.wav
    python test_wav_wakeword.py --wav inmp441_capture.wav --plot

The script slices the WAV into overlapping 1-second windows
(same sliding-window approach as pc_wakeword_test.py) and prints
the detection result for every window.

Requirements:
    pip install numpy librosa scipy matplotlib
    pip install tflite-runtime  (or tensorflow)
"""

import sys
import os
import argparse
import numpy as np
import librosa

try:
    import scipy.io.wavfile as wav_io
except ImportError:
    print("[ERROR] Install scipy:  pip install scipy")
    sys.exit(1)

# ── TFLite backend ──────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        print("[ERROR] Install tflite-runtime or tensorflow")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — mirrors pc_wakeword_test.py exactly
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH      = r"D:\Work\Projects\STM32_EmbeddedAI\Python\models\gragas_dscnn_int8.tflite"
NORM_STATS_PATH = r"D:\Work\Projects\STM32_EmbeddedAI\Python\features\norm_stats.npz"

SAMPLE_RATE  = 16_000
NUM_SAMPLES  = 16_000          # 1 second
PRE_EMPHASIS = 0.97

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

INPUT_SCALE  = 0.065618
INPUT_ZP     = 9
OUTPUT_SCALE = 0.003906250
OUTPUT_ZP    = -128

WW_THRESHOLD = 0.85
SLIDE_HOP_S  = 0.25

LABEL_NAMES = {0: "negative", 1: "wakeword (gragas)"}


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — identical functions from pc_wakeword_test.py
# ══════════════════════════════════════════════════════════════════════════════

def apply_preemphasis(audio):
    return np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])


def compute_mfcc(audio):
    audio = apply_preemphasis(audio)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX, window=WINDOW, center=CENTER,
    )
    mfcc = mfcc.T
    if mfcc.shape[0] > NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES, :]
    elif mfcc.shape[0] < NUM_FRAMES:
        mfcc = np.pad(mfcc, ((0, NUM_FRAMES - mfcc.shape[0]), (0, 0)),
                      mode="edge")
    return mfcc


def normalize(mfcc, mean, std):
    return (mfcc - mean) / std


def quantize(mfcc_norm):
    q = np.round(mfcc_norm / INPUT_SCALE).astype(np.int32) + INPUT_ZP
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def dequantize_output(raw_int8):
    return (raw_int8.astype(np.float32) - OUTPUT_ZP) * OUTPUT_SCALE


def run_inference(interpreter, input_index, mfcc_int8):
    inp = mfcc_int8.reshape(1, NUM_FRAMES, N_MFCC, 1)
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    raw = interpreter.get_tensor(output_details[0]['index'])
    probs = dequantize_output(raw[0])
    return probs


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test wakeword model on a pre-recorded WAV file")
    parser.add_argument("--wav", required=True,
                        help="Path to 16 kHz mono WAV file")
    parser.add_argument("--threshold", type=float, default=WW_THRESHOLD,
                        help=f"Detection threshold (default {WW_THRESHOLD})")
    parser.add_argument("--hop", type=float, default=SLIDE_HOP_S,
                        help=f"Sliding window hop in seconds (default {SLIDE_HOP_S})")
    parser.add_argument("--plot", action="store_true",
                        help="Plot waveform + detection probabilities")
    parser.add_argument("--save-mfcc", action="store_true",
                        help="Save MFCC heatmap comparisons as PNG")
    args = parser.parse_args()

    # ── Load WAV ──────────────────────────────────────────────────────────
    if args.wav.endswith('.wav') or args.wav.endswith('.WAV'):
        sr_file, data = wav_io.read(args.wav)
        if data.dtype == np.int16:
            audio_full = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            audio_full = data.astype(np.float32) / 2147483647.0
        elif data.dtype == np.float32:
            audio_full = data
        else:
            audio_full = data.astype(np.float32)

        # If stereo, take first channel
        if audio_full.ndim > 1:
            audio_full = audio_full[:, 0]
    else:
        # Use librosa for other formats
        audio_full, sr_file = librosa.load(args.wav, sr=None, mono=True)

    # Resample if needed
    if sr_file != SAMPLE_RATE:
        print(f"[INFO] Resampling from {sr_file} Hz → {SAMPLE_RATE} Hz")
        audio_full = librosa.resample(audio_full, orig_sr=sr_file,
                                       target_sr=SAMPLE_RATE)

    duration = len(audio_full) / SAMPLE_RATE
    rms_full = float(np.sqrt(np.mean(audio_full ** 2)))
    peak = float(np.max(np.abs(audio_full)))
    print(f"[INFO] Loaded: {args.wav}")
    print(f"       Duration : {duration:.2f} s  ({len(audio_full)} samples)")
    print(f"       RMS      : {rms_full:.5f}")
    print(f"       Peak     : {peak:.5f}")

    # ── Load norm stats ───────────────────────────────────────────────────
    stats = np.load(NORM_STATS_PATH)
    mean  = stats["mean"].astype(np.float32)
    std   = stats["std"].astype(np.float32)
    print(f"[INFO] Norm stats: mean[0]={mean[0]:.3f}  std[0]={std[0]:.3f}")

    # ── Load model ────────────────────────────────────────────────────────
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    input_index    = input_details[0]['index']
    print(f"[INFO] Model loaded: {os.path.basename(MODEL_PATH)}")

    # ── Slide through the file ────────────────────────────────────────────
    hop_samples = int(args.hop * SAMPLE_RATE)
    n_windows   = max(1, 1 + (len(audio_full) - NUM_SAMPLES) // hop_samples)

    print(f"\n{'='*65}")
    print(f"  Sliding window: 1.0 s, hop={args.hop} s → {n_windows} windows")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*65}\n")

    results = []  # (time_s, rms, p_ww, detected)

    best_p_ww = 0.0
    best_window = -1
    detections = 0

    for i in range(n_windows):
        start = i * hop_samples
        end   = start + NUM_SAMPLES

        # Pad with zeros if we exceed the file
        if end <= len(audio_full):
            chunk = audio_full[start:end]
        else:
            chunk = np.zeros(NUM_SAMPLES, dtype=np.float32)
            valid = len(audio_full) - start
            if valid > 0:
                chunk[:valid] = audio_full[start:start + valid]

        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Full pipeline
        mfcc_raw  = compute_mfcc(chunk)
        print("Python coef0:", mfcc_raw[:, 0])
        mfcc_norm = normalize(mfcc_raw, mean, std)
        mfcc_int8 = quantize(mfcc_norm)
        probs     = run_inference(interp, input_index, mfcc_int8)

        p_neg = float(probs[0])
        p_ww  = float(probs[1])
        detected = p_ww >= args.threshold

        t_start = start / SAMPLE_RATE
        t_end   = end / SAMPLE_RATE

        bar = int(p_ww * 40)
        marker = " ✅ DETECTED!" if detected else ""
        print(f"  [{i+1:03d}/{n_windows}]  "
              f"t={t_start:5.2f}-{t_end:5.2f}s  "
              f"RMS={rms:.4f}  "
              f"P(ww)={p_ww:.4f} |{'█'*bar:\x3c40}|{marker}")

        results.append((t_start, rms, p_ww, detected))

        if p_ww > best_p_ww:
            best_p_ww = p_ww
            best_window = i
        if detected:
            detections += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  Total windows : {n_windows}")
    print(f"  Detections    : {detections}")
    print(f"  Best P(ww)    : {best_p_ww:.4f}  "
          f"(window {best_window+1}, "
          f"t={results[best_window][0]:.2f}s)")
    print(f"  Threshold     : {args.threshold}")

    if detections > 0:
        print(f"\n  🟢 WAKEWORD DETECTED in the INMP441 recording!")
        print(f"     → The model works with INMP441 audio.")
        print(f"     → Problem is likely in the STM32 C MFCC pipeline.")
    elif best_p_ww > 0.1:
        print(f"\n  🟡 Not detected, but model shows some response "
              f"(best={best_p_ww:.4f}).")
        print(f"     → Try lowering --threshold, or check audio gain.")
    else:
        print(f"\n  🔴 Model shows no response to this recording.")
        print(f"     → The INMP441 audio may need gain/preprocessing fixes,")
        print(f"     → or the model needs retraining with real mic data.")

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

            # 1) Waveform
            t = np.arange(len(audio_full)) / SAMPLE_RATE
            axes[0].plot(t, audio_full, linewidth=0.3, color='#2196F3')
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title(f"Waveform — {os.path.basename(args.wav)}")
            axes[0].set_xlim(0, duration)

            # Shade detection windows
            for (ts, _, pw, det) in results:
                if det:
                    axes[0].axvspan(ts, ts + 1.0, alpha=0.3, color='lime')

            # 2) P(ww) over time
            times = [r[0] for r in results]
            p_wws = [r[2] for r in results]
            axes[1].bar(times, p_wws, width=args.hop * 0.9,
                       color=['#4CAF50' if p >= args.threshold
                              else '#FF9800' if p > 0.1
                              else '#9E9E9E' for p in p_wws],
                       align='edge')
            axes[1].axhline(y=args.threshold, color='red',
                          linestyle='--', label=f'threshold={args.threshold}')
            axes[1].set_ylabel("P(wakeword)")
            axes[1].set_title("Sliding Window Inference")
            axes[1].set_ylim(0, 1.05)
            axes[1].legend()

            # 3) RMS over time
            rmss = [r[1] for r in results]
            axes[2].bar(times, rmss, width=args.hop * 0.9,
                       color='#03A9F4', align='edge')
            axes[2].set_ylabel("RMS")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_title("Audio Energy (RMS per window)")

            plt.tight_layout()
            plot_path = args.wav.rsplit('.', 1)[0] + '_analysis.png'
            plt.savefig(plot_path, dpi=150)
            print(f"\n  📊 Plot saved → {plot_path}")
            plt.show()

        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot")

    # ── Save MFCC comparison ──────────────────────────────────────────────
    if args.save_mfcc and best_window >= 0:
        try:
            import matplotlib.pyplot as plt

            start = best_window * hop_samples
            chunk = audio_full[start:start + NUM_SAMPLES]
            if len(chunk) < NUM_SAMPLES:
                chunk = np.pad(chunk, (0, NUM_SAMPLES - len(chunk)))

            mfcc_raw  = compute_mfcc(chunk)
            mfcc_norm = normalize(mfcc_raw, mean, std)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            im0 = axes[0].imshow(mfcc_raw.T, aspect='auto',
                                  origin='lower', cmap='viridis')
            axes[0].set_title(f"Raw MFCC (window {best_window+1})")
            axes[0].set_xlabel("Frame")
            axes[0].set_ylabel("Coefficient")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(mfcc_norm.T, aspect='auto',
                                  origin='lower', cmap='viridis')
            axes[1].set_title(f"Normalized MFCC (window {best_window+1})")
            axes[1].set_xlabel("Frame")
            axes[1].set_ylabel("Coefficient")
            plt.colorbar(im1, ax=axes[1])

            plt.tight_layout()
            mfcc_path = args.wav.rsplit('.', 1)[0] + '_mfcc.png'
            plt.savefig(mfcc_path, dpi=150)
            print(f"  📊 MFCC heatmap saved → {mfcc_path}")

        except ImportError:
            print("[WARN] matplotlib not installed — skipping MFCC plot")


if __name__ == "__main__":
    main()
