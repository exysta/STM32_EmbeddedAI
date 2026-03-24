import serial, csv, time, struct, numpy as np
from scipy.io import wavfile
from pathlib import Path

PORT     = "COM3"
BAUD     = 921_600
SYNC     = bytes([0xAA, 0xBB, 0xCC, 0xDD])

# 63 blocks × 256 samples = 16 128 → firmware trims to 16 000
BLOCK_FRAMES   = 256
BLOCKS_PER_WAV = 63

# Timeout per WAV:
#   63 blocks × 256 samples / 16000 Hz ≈ 1.0 s audio
#   + 716 ms inference + 200 ms serial latency margin
SCORE_TIMEOUT  = 3.0   # seconds — raise to 5.0 if you still get timeouts


def send_wav(ser: serial.Serial, wav_path: Path) -> None:
    """Send a 16 kHz mono WAV as 256-sample 24-bit LE blocks with sync headers."""
    rate, data = wavfile.read(str(wav_path))

    # Normalise to int16 regardless of source dtype
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    # Mono: take first channel if stereo
    if data.ndim > 1:
        data = data[:, 0]

    # Pad / trim to exactly BLOCKS_PER_WAV * BLOCK_FRAMES samples
    target = BLOCKS_PER_WAV * BLOCK_FRAMES
    if len(data) < target:
        data = np.pad(data, (0, target - len(data)))
    else:
        data = data[:target]

    # Send block by block
    for blk in range(BLOCKS_PER_WAV):
        chunk = data[blk * BLOCK_FRAMES : (blk + 1) * BLOCK_FRAMES]
        ser.write(SYNC)
        for s in chunk:
            # 16-bit → 24-bit left-justified (matches INMP441 SAI format)
            val = (int(s) << 8) & 0xFFFFFF
            ser.write(struct.pack('<I', val)[:3])
        # Small inter-block delay so STM32 UART RX is not overwhelmed
        time.sleep(0.005)


def read_score(ser: serial.Serial, timeout: float = SCORE_TIMEOUT) -> float | None:
    """
    Read UART lines until a [WW] p_ww= line appears.
    Returns the float p_ww value, or None on timeout.
    Prints all received lines so you can see what the STM32 is saying.
    """
    ser.reset_input_buffer()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                print(f"    STM32: {line}")
            if "p_ww=" in line:
                p_ww = float(line.split("p_ww=")[1].split()[0])
                return p_ww
        except Exception as e:
            print(f"    [WARN] readline error: {e}")
    return None


def run_evaluation(
    ser: serial.Serial,
    wakeword_dir: Path,
    negative_dir: Path,
    output_csv: Path,
) -> None:

    results = []

    ww_files  = sorted(wakeword_dir.glob("*.wav"))
    neg_files = sorted(negative_dir.glob("*.wav"))

    if not ww_files and not neg_files:
        print("[ERROR] No WAV files found. Check your test_set paths.")
        return

    print(f"\n  Wakeword samples : {len(ww_files)}")
    print(f"  Negative samples : {neg_files and len(neg_files)}")
    print(f"  Score timeout    : {SCORE_TIMEOUT} s per sample\n")

    for label, files, class_name in [
        (1, ww_files,  "wakeword"),
        (0, neg_files, "negative"),
    ]:
        for wav in files:
            print(f"\n  → [{class_name}]  {wav.name}")
            send_wav(ser, wav)
            p = read_score(ser)

            if p is None:
                print(f"    [TIMEOUT] No score received — sample skipped.")
                results.append({
                    "file":  wav.name,
                    "label": label,
                    "p_ww":  "",
                    "status": "timeout",
                })
            else:
                decision = "DETECT" if p >= 0.80 else "reject"
                print(f"    p_ww={p:.4f}  →  {decision}")
                results.append({
                    "file":   wav.name,
                    "label":  label,
                    "p_ww":   f"{p:.6f}",
                    "status": "ok",
                })

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "p_ww", "status"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    ok      = [r for r in results if r["status"] == "ok"]
    timeout = [r for r in results if r["status"] == "timeout"]
    scores  = np.array([float(r["p_ww"]) for r in ok])
    labels  = np.array([r["label"]       for r in ok])
    preds   = (scores >= 0.80).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy  = (tp + tn) / len(ok) * 100 if ok else 0
    precision = tp / (tp + fp) * 100       if (tp + fp) else 0
    recall    = tp / (tp + fn) * 100       if (tp + fn) else 0
    fpr       = fp / (fp + tn) * 100       if (fp + tn) else 0

    print(f"\n{'='*52}")
    print(f"  ON-DEVICE EVALUATION RESULTS")
    print(f"{'='*52}")
    print(f"  Samples scored   : {len(ok)}  ({len(timeout)} timed out)")
    print(f"  Accuracy         : {accuracy:.1f}%")
    print(f"  Precision        : {precision:.1f}%")
    print(f"  Recall           : {recall:.1f}%")
    print(f"  False positive % : {fpr:.1f}%")
    print(f"  Confusion matrix :")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print(f"\n  Results saved to : {output_csv}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",        default=PORT)
    parser.add_argument("--baud",        type=int, default=BAUD)
    parser.add_argument("--wakeword-dir",
                        default="../test_set/wakeword",
                        help="Directory of held-out wakeword WAVs")
    parser.add_argument("--negative-dir",
                        default="../test_set/negative",
                        help="Directory of held-out negative WAVs")
    parser.add_argument("--output",
                        default="../test_set/ondevice_results.csv")
    parser.add_argument("--timeout",     type=float, default=SCORE_TIMEOUT,
                        help="Seconds to wait for p_ww= line per sample")
    args = parser.parse_args()

    SCORE_TIMEOUT = args.timeout

    print(f"\n  Opening {args.port} @ {args.baud} baud...", end=" ")
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print("OK")

    run_evaluation(
        ser          = ser,
        wakeword_dir = Path(args.wakeword_dir),
        negative_dir = Path(args.negative_dir),
        output_csv   = Path(args.output),
    )
    ser.close()