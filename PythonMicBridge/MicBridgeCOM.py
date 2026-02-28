import serial
import wave
import numpy as np
import matplotlib.pyplot as plt

# ---- Settings ----
COM_PORT = 'COM3'        # change this to your actual port
BAUD_RATE = 921600
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
# ------------------

NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
BYTES_TO_READ = NUM_SAMPLES * 2  # 2 bytes per int16

# Find your COM port in Device Manager → Ports (COM & LPT)
# Look for "STMicroelectronics STLink Virtual COM Port"
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=10)

input("Press Enter to start recording...")
print(f"Recording {RECORD_SECONDS} seconds — speak now!")
ser.reset_input_buffer()  # flush garbage before recording
raw_data = ser.read(BYTES_TO_READ)
ser.close()

# Convert raw bytes to int16 samples
samples = np.frombuffer(raw_data, dtype=np.int16)

# Save as WAV
with wave.open("capture.wav", "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(raw_data)

# Quality stats
rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
peak = np.max(np.abs(samples))
print(f"RMS: {rms:.0f} | Peak: {peak} | Duration: {len(samples)/SAMPLE_RATE:.2f}s")
print("Saved as capture.wav")

# Plot waveform
time_axis = np.linspace(0, len(samples) / SAMPLE_RATE, len(samples))
plt.figure(figsize=(12, 4))
plt.plot(time_axis, samples)
plt.title("Captured Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()