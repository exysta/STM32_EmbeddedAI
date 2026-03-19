"""
record_inmp441.py
=================
Captures raw 24-bit audio from the INMP441 mic connected to the STM32
over the UART binary stream, and saves it as a 16 kHz mono WAV file.

Prerequisites
-------------
1. In main.c, **uncomment** the HAL_UART_Transmit line (~line 202):
       HAL_UART_Transmit(&huart3, tx_buf, sizeof(tx_buf), HAL_MAX_DELAY);

2. In main.c, **comment out** all printf() calls (they share the same
   UART and would corrupt the binary audio stream).  The easiest way is
   to temporarily make _write() a no-op:
       int _write(int file, char *ptr, int len) { return len; }

3. Flash the modified firmware.

4. Run this script:
       python record_inmp441.py --port COM7 --seconds 5 --output inmp441_capture.wav

Frame format coming from STM32
------------------------------
  [0xAA 0xBB 0xCC 0xDD]  4-byte sync header
  [s0_lo s0_mid s0_hi]   sample 0  (24-bit signed, little-endian)
  [s1_lo s1_mid s1_hi]   sample 1
  ...                     (256 samples per block at 16 kHz)

Requirements:
    pip install pyserial numpy scipy
"""

import argparse
import sys
import time
import struct
import numpy as np
import scipy.io.wavfile as wav

try:
    import serial
except ImportError:
    print("[ERROR] Install pyserial:  pip install pyserial")
    sys.exit(1)


# ── Must match main.c defines ────────────────────────────────────────────────
SYNC_HEADER      = bytes([0xAA, 0xBB, 0xCC, 0xDD])
BLOCK_FRAMES     = 256        # AUDIO_BLOCK_FRAMES
SAMPLE_RATE      = 16_000
BYTES_PER_SAMPLE = 3          # 24-bit packed
FRAME_SIZE       = len(SYNC_HEADER) + BLOCK_FRAMES * BYTES_PER_SAMPLE  # 772


def decode_block(payload: bytes) -> np.ndarray:
    """Decode 256 × 24-bit LE signed samples → float32 in [-1, 1]."""
    samples = np.empty(BLOCK_FRAMES, dtype=np.float32)
    for i in range(BLOCK_FRAMES):
        b = payload[i * 3 : i * 3 + 3]
        # Unpack 24-bit little-endian signed
        val = b[0] | (b[1] << 8) | (b[2] << 16)
        if val & 0x800000:          # sign-extend
            val -= 0x1000000
        samples[i] = val / 8388607.0   # normalize to [-1, 1]
    return samples


def find_sync(ser: serial.Serial) -> bool:
    """Scan byte-by-byte until the 4-byte sync header is found."""
    buf = bytearray(4)
    while True:
        b = ser.read(1)
        if len(b) == 0:
            return False
        buf.append(b[0])
        buf = buf[-4:]
        if bytes(buf) == SYNC_HEADER:
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Record INMP441 audio from STM32 UART stream")
    parser.add_argument("--port", required=True,
                        help="Serial port, e.g. COM7 or /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600,
                        help="Baud rate (default 921600)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Recording duration in seconds (default 5)")
    parser.add_argument("--output", default="inmp441_capture.wav",
                        help="Output WAV filename (default inmp441_capture.wav)")
    args = parser.parse_args()

    total_samples = int(args.seconds * SAMPLE_RATE)
    total_blocks  = (total_samples + BLOCK_FRAMES - 1) // BLOCK_FRAMES

    print(f"[INFO] Opening {args.port} at {args.baud} baud")
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.5)         # let STM32 boot / flush
    ser.reset_input_buffer()

    print(f"[INFO] Waiting for sync header (0xAA 0xBB 0xCC 0xDD)...")
    if not find_sync(ser):
        print("[ERROR] Timeout waiting for sync header. "
              "Is the UART transmit uncommented in firmware?")
        ser.close()
        sys.exit(1)

    print(f"[INFO] Synced! Recording {args.seconds:.1f} s "
          f"({total_blocks} blocks, {total_samples} samples)...")

    audio = []
    blocks_ok = 0
    blocks_bad = 0

    for blk in range(total_blocks):
        # Read payload (the sync was already consumed)
        payload = ser.read(BLOCK_FRAMES * BYTES_PER_SAMPLE)
        if len(payload) < BLOCK_FRAMES * BYTES_PER_SAMPLE:
            blocks_bad += 1
            continue

        audio.append(decode_block(payload))
        blocks_ok += 1

        # Find next sync for the following block
        if blk < total_blocks - 1:
            if not find_sync(ser):
                print(f"[WARN] Lost sync after block {blk+1}")
                blocks_bad += 1

        # Progress
        if (blk + 1) % 20 == 0 or blk == total_blocks - 1:
            elapsed = (blk + 1) * BLOCK_FRAMES / SAMPLE_RATE
            print(f"  [{blk+1}/{total_blocks}] "
                  f"{elapsed:.1f}s captured  "
                  f"(ok={blocks_ok} bad={blocks_bad})", end="\r")

    ser.close()
    print()

    if not audio:
        print("[ERROR] No audio captured.")
        sys.exit(1)

    # Concatenate and trim to exact requested length
    pcm = np.concatenate(audio)[:total_samples]

    # Stats
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    peak = float(np.max(np.abs(pcm)))
    print(f"\n[INFO] Captured {len(pcm)} samples ({len(pcm)/SAMPLE_RATE:.2f} s)")
    print(f"       RMS  = {rms:.5f}")
    print(f"       Peak = {peak:.5f}")

    # Save as 16-bit WAV (standard, compatible with everything)
    pcm_int16 = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
    wav.write(args.output, SAMPLE_RATE, pcm_int16)
    print(f"[INFO] Saved → {args.output}")
    print(f"\n  Next step: run the wakeword test on this file:")
    print(f"    python test_wav_wakeword.py --wav {args.output}")


if __name__ == "__main__":
    main()
