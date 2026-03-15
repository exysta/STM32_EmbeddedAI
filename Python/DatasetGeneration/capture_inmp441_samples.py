"""
capture_inmp441_samples.py
==========================
Captures "gragas" recordings directly through the INMP441/STM32 UART bridge,
producing WAV files with the exact same spectral characteristics the model
will see at inference time.

These samples should be added to the training dataset and the model retrained.
Capturing ~100 samples with varied intonation is enough to fix the mic
frequency response mismatch.

Usage:
    python capture_inmp441_samples.py --port COM3 --out dataset/wakeword/real

Protocol (matches main.c):
    Header: AA BB CC DD   (4 bytes)
    Payload: AUDIO_BLOCK_FRAMES × 3 bytes, little-endian 24-bit signed
    AUDIO_BLOCK_FRAMES = 256, so each packet = 4 + 768 = 772 bytes
    Sample rate = 16 000 Hz → 256 samples = 16 ms per packet
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf

# ── Constants matching main.c ─────────────────────────────────────────────
SYNC_HEADER       = bytes([0xAA, 0xBB, 0xCC, 0xDD])
AUDIO_BLOCK_FRAMES = 256
BYTES_PER_SAMPLE  = 3
PACKET_PAYLOAD    = AUDIO_BLOCK_FRAMES * BYTES_PER_SAMPLE   # 768 bytes
PACKET_TOTAL      = 4 + PACKET_PAYLOAD                       # 772 bytes
SAMPLE_RATE       = 16_000
CAPTURE_DURATION  = 1.0    # seconds per sample
PACKETS_PER_CLIP  = int(CAPTURE_DURATION * SAMPLE_RATE / AUDIO_BLOCK_FRAMES)  # 62


def read_packet(ser: serial.Serial) -> np.ndarray | None:
    """Read one sync-delimited audio packet. Returns 256 float32 samples or None."""
    # Scan for sync header
    buf = b""
    while True:
        byte = ser.read(1)
        if not byte:
            return None
        buf = (buf + byte)[-4:]
        if buf == SYNC_HEADER:
            break

    # Read payload
    payload = ser.read(PACKET_PAYLOAD)
    if len(payload) != PACKET_PAYLOAD:
        return None

    # Unpack 24-bit little-endian signed samples
    samples = np.empty(AUDIO_BLOCK_FRAMES, dtype=np.float32)
    for i in range(AUDIO_BLOCK_FRAMES):
        b0, b1, b2 = payload[i*3], payload[i*3+1], payload[i*3+2]
        raw = b0 | (b1 << 8) | (b2 << 16)
        if raw & 0x800000:          # sign-extend 24-bit → int32
            raw -= 0x1000000
        samples[i] = raw / 8388608.0   # normalise to [-1, 1]

    return samples


def capture_clip(ser: serial.Serial) -> np.ndarray:
    """Capture exactly CAPTURE_DURATION seconds of audio."""
    chunks = []
    for _ in range(PACKETS_PER_CLIP):
        pkt = read_packet(ser)
        if pkt is not None:
            chunks.append(pkt)
    return np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(description="Capture INMP441 wakeword samples via UART")
    parser.add_argument("--port",  default="COM3",                        help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",  default=115200,  type=int,             help="Baud rate (default 115200)")
    parser.add_argument("--out",   default="dataset/wakeword/real",       help="Output directory")
    parser.add_argument("--count", default=100,     type=int,             help="Number of samples to capture")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.wav")))
    print(f"Output  : {out_dir.resolve()}")
    print(f"Port    : {args.port} @ {args.baud} baud")
    print(f"Target  : {args.count} samples  (starting at index {existing})")
    print(f"Duration: {CAPTURE_DURATION} s per sample\n")

    print("Tips for a good dataset:")
    print("  - Vary distance to mic: 20 cm, 50 cm, 1 m")
    print("  - Vary intonation: flat, rising, questioning")
    print("  - Vary speed: normal, slightly faster, slightly slower")
    print("  - Try different volumes: conversational, slightly louder, quieter")
    print("  - Move around: slightly left, right, above\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
        print(f"Port open: {ser.name}\n")
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}")
        return

    idx = existing
    captured = 0

    while captured < args.count:
        remaining = args.count - captured
        cmd = input(f"  [{idx:03d}]  Press ENTER to capture (say 'gragas' after beep),"
                    f"  {remaining} remaining, 'q' to quit: ").strip().lower()
        if cmd == "q":
            break

        # Short countdown
        for t in ["3...", "2...", "1...", "🎤 NOW"]:
            print(f"    {t}", end="\r")
            time.sleep(0.5 if t != "🎤 NOW" else 0.0)

        audio = capture_clip(ser)
        rms   = float(np.sqrt(np.mean(audio ** 2)))

        path = out_dir / f"gragas_real_{idx:04d}.wav"
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")

        print(f"    Saved {path.name}   RMS={rms:.4f}", end="")
        if rms < 0.005:
            print("  ⚠️  Very quiet — try again closer to mic")
        elif rms > 0.4:
            print("  ⚠️  May be clipping — try a bit quieter")
        else:
            print("  ✅")

        idx     += 1
        captured += 1

    ser.close()

    print(f"\n{'='*55}")
    print(f"  Captured {captured} samples → {out_dir.resolve()}")
    print(f"\n  Next steps:")
    print(f"    1. Run extract_mfcc.py  (WAKEWORD_DIR now includes real/)")
    print(f"    2. Retrain the DS-CNN model")
    print(f"    3. Re-quantize with Cube.AI Studio")
    print(f"    4. Regenerate network.h and flash")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
