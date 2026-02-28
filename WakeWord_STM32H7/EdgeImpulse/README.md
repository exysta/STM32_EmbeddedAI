# Edge Impulse Model Training Guide

This guide walks through training a wake word detection model in **Edge Impulse**
and exporting it for use with **STM32Cube.AI (X-CUBE-AI)** on the
STM32H743ZI2 Nucleo board.

---

## 1 – Create an Edge Impulse project

1. Sign up / log in at <https://studio.edgeimpulse.com/>.
2. Click **Create new project** → name it (e.g. `WakeWord_STM32H7`).
3. Under **Project type** select **Audio (keyword spotting)**.

---

## 2 – Collect training data

### 2a – Via the Edge Impulse CLI (recommended)

```bash
# Install the CLI once
npm install -g edge-impulse-cli

# Stream audio from your development machine microphone
edge-impulse-daemon --clean
```

- When prompted, select your project and choose **"Microphone"** as the device.
- In the Edge Impulse Studio, go to **Data acquisition**.
- Record samples for each class:
  | Class label | Target samples | Sample length |
  |-------------|----------------|---------------|
  | `wake_word` | ≥ 100          | 1 second each |
  | `noise`     | ≥ 100          | 1 second each |

### 2b – Upload pre-existing `.wav` files

Use **Data acquisition → Upload data** to bulk-upload 16 kHz mono WAV files.

---

## 3 – Design the Impulse

1. Go to **Create impulse**.
2. **Time series input** block:
   - Window size : **1000 ms**
   - Window increase : **500 ms**
   - Frequency : **16000 Hz**
   - Zero-pad data : ✓
3. **Processing block** → Add **MFCC**:
   - Frame length : **0.032 s** (32 ms)
   - Frame stride : **0.016 s** (16 ms)
   - Num filters  : **26**
   - Num coefficients : **13**
   - FFT length    : **512**
   - Low frequency : **0 Hz**
   - High frequency : **8000 Hz**
   - Pre-emphasis  : **0.97**
4. **Learning block** → Add **Classification (Keras)**.
5. Click **Save Impulse**.

> **Note:** The MFCC parameters above match the constants in
> `Core/Inc/audio_processing.h`.  If you change them in Edge Impulse,
> update the corresponding `#define` values in the firmware as well.

---

## 4 – Generate features and train

1. **MFCC → Generate features** – wait for processing to complete.
2. **Classifier → Configure training**:
   - Training cycles : **50**
   - Learning rate   : **0.005**
   - Architecture    : use the default 1-D Conv + Dense head,
     or switch to the **Reshape → 2-D Conv** architecture for higher accuracy.
3. Click **Start training**.
4. Inspect the confusion matrix and validation accuracy.
   Aim for ≥ 95 % accuracy before deploying.

---

## 5 – Export for STM32Cube.AI

1. Go to **Deployment**.
2. Select **STM32Cube.AI Library** (under "Other deployment options").
3. Choose **Quantized (int8)** for the smallest Flash footprint,
   or **Float32** if you need higher accuracy.
4. Click **Build** – download the `.zip` archive.
5. Extract the archive; locate the model file:
   - For TensorFlow Lite: `*.tflite`
   - For ONNX (if offered): `*.onnx`

---

## 6 – Import into STM32Cube.AI

1. Open `WakeWord_STM32H7/WakeWord_STM32H7.ioc` in **STM32CubeIDE**.
2. Navigate to:
   **Pinout & Configuration → Software Packs → X-CUBE-AI → Artificial Intelligence**.
3. Under **Network**, click **Add network** and browse to the `.tflite` or `.onnx` file.
4. Name the network **`network`** (must match the `#include "network.h"` in the code).
5. Click **Analyse** and review the memory report:
   - Flash (weights)      : should be < 200 kB for a small keyword model.
   - RAM (activations)    : should be < 50 kB.
6. Click **Generate Code** – this writes `network.c / .h` and `network_data.c / .h`
   into `X-CUBE-AI/App/`.
7. Build and flash the project.

---

## 7 – Verify on the board

Open a serial terminal (115200 8N1) on the ST-LINK COM port.
You should see:

```
====================================================
  Wake Word Detection – STM32H743ZI2 + INMP441
====================================================
[XCUBEAI] Network: network  (rev ...)
[XCUBEAI] In : 637 × float32
[XCUBEAI] Out: 2 × float32
[OK]  Audio capture started at 16000 Hz
[OK]  Say your wake word...

[WW] *** WAKE WORD DETECTED ***  (score=0.987)
```

The green LED (LD1) blinks for 500 ms each time the wake word is detected.

---

## 8 – Tuning tips

| Parameter | File | Effect |
|-----------|------|--------|
| `WAKEWORD_THRESHOLD` | `Core/Inc/main.h` | Raise to reduce false positives; lower to reduce misses |
| `WAKEWORD_LED_DURATION_MS` | `Core/Inc/main.h` | Visual feedback duration |
| `AUDIO_HOP_MS` | `Core/Inc/main.h` | Latency vs. CPU load trade-off |
| Training cycles / learning rate | Edge Impulse | Model accuracy |
| Quantization (int8 vs float32) | Edge Impulse deploy | Size / accuracy trade-off |
