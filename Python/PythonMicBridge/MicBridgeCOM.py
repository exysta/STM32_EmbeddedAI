import serial
import wave
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
COM_PORT       = 'COM3'
BAUD_RATE      = 921600
SAMPLE_RATE    = 16000
RECORD_SECONDS = 5

# Must match STM32 defines exactly
AUDIO_BLOCK_FRAMES = 256                     # samples per block (AUDIO_BLOCK_FRAMES in main.c)
BYTES_PER_SAMPLE   = 3                       # 24-bit = 3 bytes
BLOCK_AUDIO_BYTES  = AUDIO_BLOCK_FRAMES * BYTES_PER_SAMPLE   # 768 bytes of audio per block
BLOCK_TOTAL_BYTES  = 4 + BLOCK_AUDIO_BYTES   # 4-byte header + 768 bytes audio = 772

SYNC = bytes([0xAA, 0xBB, 0xCC, 0xDD])

TOTAL_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
TOTAL_BLOCKS  = TOTAL_SAMPLES // AUDIO_BLOCK_FRAMES   # 312 blocks for 5s
# ──────────────────────────────────────────────────────────────────────────────


def unpack_24bit_le(raw_bytes: bytes, count: int) -> np.ndarray:
    """
    Vectorized unpack of `count` little-endian signed 24-bit samples
    from `raw_bytes` (must be count*3 bytes long).
    Returns int32 numpy array.
    """
    b = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(count, 3)
    # Assemble unsigned 24-bit values
    u24 = b[:, 0].astype(np.uint32) \
        | (b[:, 1].astype(np.uint32) << 8) \
        | (b[:, 2].astype(np.uint32) << 16)
    # Sign-extend: if bit 23 is set, the value is negative
    signed = u24.astype(np.int32)
    signed[u24 >= 0x800000] -= 0x1000000
    return signed


def find_sync(ser: serial.Serial) -> bool:
    """
    Scan the byte stream until we find the 4-byte sync header.
    Returns True when found, False on timeout.
    """
    buf = bytearray()
    while True:
        byte = ser.read(1)
        if not byte:
            return False   # timeout
        buf.append(byte[0])
        if len(buf) >= 4 and buf[-4:] == bytearray(SYNC):
            return True


# ── Main ──────────────────────────────────────────────────────────────────────
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=10)
print(f"Port open: {COM_PORT} @ {BAUD_RATE} baud")
print(f"Will capture {TOTAL_BLOCKS} blocks = {TOTAL_SAMPLES} samples = {RECORD_SECONDS}s")

input("\nPress ENTER to start recording...")

# Flush OS serial buffer — discard whatever STM32 sent while we were waiting
ser.reset_input_buffer()

# Wait for the very first sync header so we start at a clean block boundary
print("Waiting for first sync...")
if not find_sync(ser):
    print("ERROR: no sync found (timeout). Check wiring and baud rate.")
    ser.close()
    exit(1)
print("Synced — recording...")

# Collect blocks
all_samples = np.zeros(TOTAL_BLOCKS * AUDIO_BLOCK_FRAMES, dtype=np.int32)
blocks_received = 0

while blocks_received < TOTAL_BLOCKS:
    # Read audio payload of this block (sync was already consumed by find_sync)
    audio_bytes = ser.read(BLOCK_AUDIO_BYTES)
    if len(audio_bytes) < BLOCK_AUDIO_BYTES:
        print(f"WARNING: short read on block {blocks_received} "
              f"({len(audio_bytes)}/{BLOCK_AUDIO_BYTES} bytes) — timeout?")
        break

    # Unpack and store
    offset = blocks_received * AUDIO_BLOCK_FRAMES
    all_samples[offset: offset + AUDIO_BLOCK_FRAMES] = \
        unpack_24bit_le(audio_bytes, AUDIO_BLOCK_FRAMES)
    blocks_received += 1

    # Consume the next block's sync header before looping
    if blocks_received < TOTAL_BLOCKS:
        if not find_sync(ser):
            print(f"WARNING: lost sync after block {blocks_received}")
            break

ser.close()
print(f"Captured {blocks_received} blocks ({blocks_received * AUDIO_BLOCK_FRAMES} samples)")

# Trim to actual captured length
samples_i32 = all_samples[:blocks_received * AUDIO_BLOCK_FRAMES]
num_samples  = len(samples_i32)

# Normalize to float32 [-1.0, 1.0]  (24-bit full scale = 2^23 = 8388608)
samples_f32 = samples_i32.astype(np.float32) / 8388608.0

# ── Save WAV ──────────────────────────────────────────────────────────────────
try:
    import soundfile as sf
    sf.write("capture_24bit.wav", samples_f32, SAMPLE_RATE, subtype='PCM_24')
    print("Saved: capture_24bit.wav  (true 24-bit PCM)")
except ImportError:
    print("soundfile not installed — falling back to 16-bit WAV")
    print("  Install with: pip install soundfile")
    samples_i16 = (samples_f32 * 32767.0).astype(np.int16)
    with wave.open("capture_16bit.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_i16.tobytes())
    print("Saved: capture_16bit.wav  (16-bit fallback)")

# ── Stats ─────────────────────────────────────────────────────────────────────
rms     = np.sqrt(np.mean(samples_f32 ** 2))
peak    = np.max(np.abs(samples_f32))
rms_db  = 20 * np.log10(rms  + 1e-9)
peak_db = 20 * np.log10(peak + 1e-9)

print(f"\nDuration : {num_samples / SAMPLE_RATE:.2f}s  ({num_samples} samples)")
print(f"Peak     : {peak:.5f}  ({peak_db:.1f} dBFS)")
print(f"RMS      : {rms:.5f}  ({rms_db:.1f} dBFS)")

if peak < 0.001:
    print("WARNING: signal is nearly silent — check wiring or SAI slot config")
elif peak > 0.99:
    print("WARNING: signal is clipping — reduce AUDIO_GAIN in main.c")

# ── Plot ──────────────────────────────────────────────────────────────────────
time_axis = np.linspace(0, num_samples / SAMPLE_RATE, num_samples)
plt.figure(figsize=(12, 4))
plt.plot(time_axis, samples_f32, linewidth=0.4, color='steelblue')
plt.title(f"Captured Audio — 24-bit @ {SAMPLE_RATE} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (normalized)")
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()