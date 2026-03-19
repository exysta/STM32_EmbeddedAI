"""
record_dataset.py
=================
Interactive dataset recorder for the STM32 INMP441 wakeword pipeline.

Records N samples of a wakeword class and N samples of a negative class
directly from the STM32 UART stream. Each sample is saved in two forms:
  - Raw debug WAV  (full capture duration, e.g. 1.5 s) → debug/
  - Trimmed WAV    (best 1-second window by peak RMS)   → wakeword/ or negative/

The trimmed WAVs are drop-in replacements for the TTS samples in your
existing dataset and can be passed straight to extract_mfcc.py.

Usage
-----
  python record_dataset.py --port COM7 --mode wakeword --count 30
  python record_dataset.py --port COM7 --mode negative --count 30
  python record_dataset.py --port COM7 --mode wakeword --count 30 --duration 1.5

Prerequisites
-------------
  pip install pyserial numpy scipy

Firmware requirements (same as record_inmp441.py)
--------------------------------------------------
  1. In main.c UNCOMMENT:   HAL_UART_Transmit(...)
  2. In main.c COMMENT OUT: MFCC_IngestBlock(...)  (or _write() no-op)
  3. Flash, then run this script.

Frame format (must match main.c)
---------------------------------
  [0xAA 0xBB 0xCC 0xDD]        4-byte sync header
  [s0_lo s0_mid s0_hi] × 256   256 × 24-bit LE signed samples
"""

import argparse
import os
import sys
import time
import struct
import threading
import numpy as np
import scipy.io.wavfile as wav_io

try:
    import serial
except ImportError:
    print("[ERROR] Install pyserial:  pip install pyserial")
    sys.exit(1)

# ── Must match main.c ─────────────────────────────────────────────────────────
SYNC_HEADER      = bytes([0xAA, 0xBB, 0xCC, 0xDD])
BLOCK_FRAMES     = 256
SAMPLE_RATE      = 16_000
BYTES_PER_SAMPLE = 3
FRAME_SIZE       = len(SYNC_HEADER) + BLOCK_FRAMES * BYTES_PER_SAMPLE  # 772

# ── Training target ───────────────────────────────────────────────────────────
TRAIN_SAMPLES    = SAMPLE_RATE          # 16 000 — 1 second window for MFCC
TRAIN_WIN_LEN    = SAMPLE_RATE          # sliding window size for best-window search
TRAIN_WIN_HOP    = 160                  # 10 ms hop for fine search

# ── UI ────────────────────────────────────────────────────────────────────────
BEEP_DURATION_MS = 120   # cross-platform beep approximation via ASCII bell

# ANSI colours (suppressed on Windows if colorama not present)
try:
    import colorama; colorama.init()
    RED    = "\033[91m"
    GRN    = "\033[92m"
    YEL    = "\033[93m"
    CYN    = "\033[96m"
    BLD    = "\033[1m"
    RST    = "\033[0m"
except ImportError:
    RED = GRN = YEL = CYN = BLD = RST = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  UART helpers
# ═══════════════════════════════════════════════════════════════════════════════

def decode_block(payload: bytes) -> np.ndarray:
    """Decode 256 × 24-bit LE signed samples → float32 in [-1, 1]."""
    samples = np.empty(BLOCK_FRAMES, dtype=np.float32)
    for i in range(BLOCK_FRAMES):
        b = payload[i * 3: i * 3 + 3]
        val = b[0] | (b[1] << 8) | (b[2] << 16)
        if val & 0x800000:
            val -= 0x1000000
        samples[i] = val / 8_388_607.0
    return samples


def find_sync(ser: serial.Serial, timeout_s: float = 3.0) -> bool:
    """Scan byte-by-byte until the 4-byte sync header is found."""
    buf = bytearray(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf.append(b[0])
        buf = buf[-4:]
        if bytes(buf) == SYNC_HEADER:
            return True
    return False


def capture_audio(ser: serial.Serial, num_samples: int) -> np.ndarray | None:
    """
    Capture exactly num_samples from the UART stream.
    Returns float32 array or None on error.
    """
    total_blocks = (num_samples + BLOCK_FRAMES - 1) // BLOCK_FRAMES
    audio = []

    # ── Flush stale buffered data before syncing ──────────────────────────────
    # The OS serial buffer fills instantly during the countdown (~35 ms at 921600).
    # Discard everything in it, then wait one full block duration for live data
    # to arrive before attempting sync — this guarantees we lock onto live audio.
    ser.reset_input_buffer()
    time.sleep(BLOCK_FRAMES / SAMPLE_RATE * 2)   # wait ~32 ms for live blocks to arrive
    ser.reset_input_buffer()                       # flush the transitional block too

    if not find_sync(ser, timeout_s=3.0):
        return None

    for blk in range(total_blocks):
        payload = ser.read(BLOCK_FRAMES * BYTES_PER_SAMPLE)
        if len(payload) < BLOCK_FRAMES * BYTES_PER_SAMPLE:
            return None
        audio.append(decode_block(payload))
        if blk < total_blocks - 1:
            if not find_sync(ser, timeout_s=2.0):
                return None

    pcm = np.concatenate(audio)[:num_samples]
    return pcm.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Audio processing
# ═══════════════════════════════════════════════════════════════════════════════

def best_1s_window(pcm: np.ndarray) -> np.ndarray:
    """
    Extract the 1-second window from pcm that contains the most energy.
    Uses a sliding RMS with 10 ms hop over the full capture.

    If the recording is already ≤ 1 second, pad with zeros and return it.
    """
    if len(pcm) <= TRAIN_WIN_LEN:
        pad = TRAIN_WIN_LEN - len(pcm)
        return np.pad(pcm, (0, pad))

    best_rms  = -1.0
    best_start = 0

    for start in range(0, len(pcm) - TRAIN_WIN_LEN + 1, TRAIN_WIN_HOP):
        window = pcm[start: start + TRAIN_WIN_LEN]
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms > best_rms:
            best_rms  = rms
            best_start = start

    return pcm[best_start: best_start + TRAIN_WIN_LEN].copy()


def audio_stats(pcm: np.ndarray) -> dict:
    rms  = float(np.sqrt(np.mean(pcm ** 2)))
    peak = float(np.max(np.abs(pcm)))
    return {"rms": rms, "peak": peak}


def save_wav(path: str, pcm: np.ndarray):
    """Save float32 pcm as 16-bit WAV at SAMPLE_RATE."""
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm_clipped * 32767).astype(np.int16)
    wav_io.write(path, SAMPLE_RATE, pcm_i16)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def beep():
    """Cross-platform audio cue (terminal bell)."""
    print("\a", end="", flush=True)


def countdown(seconds: int, label: str = "Recording in"):
    """Blocking countdown with per-second display."""
    for i in range(seconds, 0, -1):
        bar = "█" * (seconds - i) + "░" * (i - 1)
        print(f"\r  {YEL}{label}: {BLD}{i:2d}s{RST}  [{bar}]   ", end="", flush=True)
        time.sleep(1.0)
    print(f"\r{' ' * 60}\r", end="", flush=True)


def progress_bar(elapsed: float, total: float, width: int = 40):
    """Inline recording progress bar."""
    frac  = min(elapsed / total, 1.0)
    filled = int(frac * width)
    bar   = "█" * filled + "░" * (width - filled)
    pct   = int(frac * 100)
    print(f"\r  {RED}● REC{RST}  [{GRN}{bar}{RST}] {pct:3d}%  {elapsed:.1f}/{total:.1f}s  ",
          end="", flush=True)


def print_header(mode: str, index: int, total: int):
    print()
    print(f"  {CYN}{'─'*58}{RST}")
    label = "WAKEWORD  (say: Gragas)" if mode == "wakeword" else "NEGATIVE  (ambient / other words)"
    print(f"  {BLD}Sample {index}/{total}  —  {label}{RST}")
    print(f"  {CYN}{'─'*58}{RST}")


def rms_quality_check(stats: dict, mode: str) -> tuple[bool, str]:
    """
    Heuristic quality gate on a recorded sample.
    Returns (ok, message).
    """
    rms  = stats["rms"]
    peak = stats["peak"]

    if mode == "wakeword":
        if rms < 0.005:
            return False, f"RMS={rms:.4f} — too quiet, mic may not be capturing"
        if rms < 0.012:
            return False, f"RMS={rms:.4f} — quite low, speak louder or move closer"
        if peak > 0.98:
            return False, f"Peak={peak:.3f} — clipping! move mic further away"
    else:  # negative
        if rms > 0.04:
            return False, f"RMS={rms:.4f} — negative sample is very loud"

    return True, f"RMS={rms:.4f}  Peak={peak:.3f}  ✓"


# ═══════════════════════════════════════════════════════════════════════════════
#  Recording session
# ═══════════════════════════════════════════════════════════════════════════════

def run_session(ser: serial.Serial, mode: str, count: int,
                duration: float, out_dir: str, debug_dir: str,
                countdown_s: int, start_index: int) -> list[dict]:
    """
    Record `count` samples of `mode` class.
    Returns list of result dicts for the summary table.
    """
    num_samples = int(duration * SAMPLE_RATE)
    results = []

    if mode == "wakeword":
        prompt = (
            f"  {BLD}Say {YEL}\"Gragas\"{RST}{BLD} clearly once per recording.{RST}\n"
            f"  Vary distance (20cm / 50cm / 1m) and loudness across takes.\n"
            f"  After the beep, speak immediately — you have {duration:.1f}s."
        )
    else:
        prompt = (
            f"  {BLD}Stay {YEL}SILENT{RST}{BLD} or say a random French word — NOT Gragas.{RST}\n"
            f"  Good negatives: silence, keyboard, 'bonjour', 'grâce', 'fracas'.\n"
            f"  After the beep, act naturally for {duration:.1f}s."
        )

    print(f"\n{prompt}\n")
    input(f"  Press {BLD}Enter{RST} when ready to start the session…")

    for i in range(count):
        idx = start_index + i
        print_header(mode, i + 1, count)

        # ── Countdown ────────────────────────────────────────────────────────
        countdown(countdown_s, label="Recording in")

        # ── Beep + capture ───────────────────────────────────────────────────
        beep()
        print(f"  {RED}{BLD}▶ SPEAK NOW{RST}" if mode == "wakeword"
              else f"  {YEL}{BLD}▶ RECORDING{RST}", flush=True)

        # Start a background thread just to animate the progress bar
        capture_done = threading.Event()

        def animate():
            t0 = time.time()
            while not capture_done.is_set():
                progress_bar(time.time() - t0, duration)
                time.sleep(0.05)
            progress_bar(duration, duration)
            print()

        anim_thread = threading.Thread(target=animate, daemon=True)
        anim_thread.start()

        pcm_raw = capture_audio(ser, num_samples)
        capture_done.set()
        anim_thread.join()

        # ── Validate capture ─────────────────────────────────────────────────
        if pcm_raw is None:
            print(f"  {RED}[ERROR] Capture failed — UART sync lost. Retrying…{RST}")
            results.append({"index": idx, "ok": False, "reason": "UART sync lost"})
            continue

        # ── Extract best 1-second window ─────────────────────────────────────
        pcm_train = best_1s_window(pcm_raw)
        raw_stats   = audio_stats(pcm_raw)
        train_stats = audio_stats(pcm_train)

        ok, quality_msg = rms_quality_check(train_stats, mode)

        # ── Save files ───────────────────────────────────────────────────────
        fname_base  = f"{mode}_{idx:04d}"
        train_path  = os.path.join(out_dir,   fname_base + ".wav")
        debug_path  = os.path.join(debug_dir, fname_base + "_raw.wav")

        save_wav(train_path, pcm_train)
        save_wav(debug_path, pcm_raw)

        # ── Report ───────────────────────────────────────────────────────────
        status_icon = f"{GRN}✓{RST}" if ok else f"{YEL}⚠{RST}"
        print(f"\n  {status_icon}  Train WAV : {train_path}")
        print(f"     Debug WAV : {debug_path}")
        print(f"     Raw  stats: RMS={raw_stats['rms']:.4f}  Peak={raw_stats['peak']:.3f}  ({duration:.1f}s)")
        print(f"     Train 1s  : {quality_msg}")

        if not ok:
            print(f"  {YEL}  ↳ Sample kept but flagged — review debug WAV before training.{RST}")

        results.append({
            "index":       idx,
            "ok":          ok,
            "train_path":  train_path,
            "debug_path":  debug_path,
            "rms_raw":     raw_stats["rms"],
            "rms_train":   train_stats["rms"],
            "peak_train":  train_stats["peak"],
            "reason":      quality_msg,
        })

        # ── Pause between takes ──────────────────────────────────────────────
        if i < count - 1:
            print(f"\n  {CYN}Next in {countdown_s}s — relax…{RST}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], mode: str, out_dir: str):
    ok_count   = sum(1 for r in results if r.get("ok"))
    warn_count = len(results) - ok_count

    print(f"\n{'═'*62}")
    print(f"  SESSION SUMMARY — {mode.upper()}")
    print(f"{'═'*62}")
    print(f"  Total recorded : {len(results)}")
    print(f"  Quality OK     : {GRN}{ok_count}{RST}")
    print(f"  Flagged        : {YEL}{warn_count}{RST}  (review debug WAVs)")
    print(f"  Saved to       : {out_dir}")
    print()

    if warn_count:
        print(f"  {YEL}Flagged samples:{RST}")
        for r in results:
            if not r.get("ok"):
                print(f"    [{r['index']:04d}]  {r.get('reason', '?')}")
        print()

    avg_rms = np.mean([r["rms_train"] for r in results if "rms_train" in r])
    print(f"  Mean RMS (1s train window): {avg_rms:.4f}")
    if mode == "wakeword" and avg_rms < 0.012:
        print(f"  {YEL}⚠  Average RMS is low — consider raising AUDIO_GAIN in main.c{RST}")

    print(f"\n  {GRN}Next steps:{RST}")
    print(f"  1. Listen to flagged debug WAVs in debug/ and re-record if needed.")
    print(f"  2. python extract_mfcc.py   (will pick up new WAVs automatically)")
    print(f"  3. python train.py           (fine-tune from existing .keras weights)")
    print(f"{'═'*62}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Record wakeword / negative samples from STM32 INMP441 via UART",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",       required=True,
                        help="Serial port  (e.g. COM7 or /dev/ttyACM0)")
    parser.add_argument("--baud",       type=int,   default=921_600)
    parser.add_argument("--mode",       choices=["wakeword", "negative"], required=True,
                        help="Class to record")
    parser.add_argument("--count",      type=int,   default=30,
                        help="Number of samples to record")
    parser.add_argument("--duration",   type=float, default=1.5,
                        help="Capture duration per sample (seconds). 1.5 recommended.")
    parser.add_argument("--countdown",  type=int,   default=3,
                        help="Countdown seconds before each recording")
    parser.add_argument("--dataset-dir", default=None,
                        help="Root dataset directory. Defaults to ../dataset relative to this script.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting file index (useful when appending to existing recordings)")
    args = parser.parse_args()

    # ── Resolve output directories ────────────────────────────────────────────
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    dataset_root = args.dataset_dir or os.path.join(script_dir, "..", "dataset")
    dataset_root = os.path.abspath(dataset_root)

    out_dir   = os.path.join(dataset_root, args.mode)
    debug_dir = os.path.join(dataset_root, "debug", args.mode)

    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # ── Auto-detect next available index ─────────────────────────────────────────
    # Scan existing files like wakeword_0012.wav → extract highest index → +1
    # This means re-running never overwrites previous recordings.
    if args.start_index == 0:
        existing_indices = []
    for fname in os.listdir(out_dir):
        if fname.endswith(".wav"):
            try:
                # filename format: {mode}_{index}.wav
                idx = int(fname.replace(args.mode + "_", "").replace(".wav", ""))
                existing_indices.append(idx)
            except ValueError:
                pass
    auto_start = (max(existing_indices) + 1) if existing_indices else 0
    if auto_start > 0:
        print(f"  {CYN}Auto-detected {auto_start} existing files "
              f"→ starting at index {auto_start}{RST}")
    args.start_index = auto_start
    # ── Print session info ────────────────────────────────────────────────────
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    print(f"\n{CYN}{'═'*62}{RST}")
    print(f"  {BLD}STM32 INMP441 Dataset Recorder{RST}")
    print(f"{'─'*62}")
    print(f"  Port         : {args.port}  @{args.baud} baud")
    print(f"  Class        : {BLD}{args.mode}{RST}")
    print(f"  Samples      : {args.count}  (+ {existing} already in folder)")
    print(f"  Duration     : {args.duration:.1f}s capture → 1.0s train (best window)")
    print(f"  Countdown    : {args.countdown}s before each take")
    print(f"  Train output : {out_dir}")
    print(f"  Debug output : {debug_dir}")
    print(f"{CYN}{'═'*62}{RST}\n")

    print(f"  {YEL}Firmware checklist before starting:{RST}")
    print(f"   ✓ HAL_UART_Transmit() is UNCOMMENTED in main.c")
    print(f"   ✓ MFCC_IngestBlock() is COMMENTED OUT in main.c")
    print(f"   ✓ _write() is a no-op (printf disabled)")
    print(f"   ✓ STM32 is flashed and running")

    # ── Open serial port ──────────────────────────────────────────────────────
    print(f"\n  Opening {args.port}…", end=" ", flush=True)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
    except serial.SerialException as e:
        print(f"{RED}FAILED{RST}")
        print(f"  [ERROR] {e}")
        sys.exit(1)

    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"{GRN}OK{RST}")

    # ── Wait for initial sync ─────────────────────────────────────────────────
    print(f"  Waiting for STM32 sync header…", end=" ", flush=True)
    if not find_sync(ser, timeout_s=5.0):
        print(f"{RED}TIMEOUT{RST}")
        print("  [ERROR] No sync header received. Is the firmware running?")
        ser.close()
        sys.exit(1)
    print(f"{GRN}Synced!{RST}")

    # ── Run session ───────────────────────────────────────────────────────────
    try:
        results = run_session(
            ser        = ser,
            mode       = args.mode,
            count      = args.count,
            duration   = args.duration,
            out_dir    = out_dir,
            debug_dir  = debug_dir,
            countdown_s= args.countdown,
            start_index= args.start_index,
        )
    except KeyboardInterrupt:
        print(f"\n\n  {YEL}Session interrupted by user.{RST}")
        results = []
    finally:
        ser.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print_summary(results, args.mode, out_dir)


if __name__ == "__main__":
    main()
