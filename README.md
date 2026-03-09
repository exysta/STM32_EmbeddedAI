# STM32 Embedded AI — Wake Word Detection

> End-to-end TinyML pipeline for real-time wake word detection on a bare-metal STM32H7 microcontroller.  
> The system detects the wake word **"Gragas"** using a neural network running entirely on-device — no cloud, no OS, no external compute.

---

## Overview

This project implements a complete **keyword spotting pipeline** on a resource-constrained embedded system, covering every stage from dataset generation to on-device inference:

- **Synthetic dataset generation** via Microsoft Neural TTS with multi-voice, multi-prosody augmentation
- **Audio acquisition** via an INMP441 MEMS microphone over SAI peripheral (I2S format)
- **Feature extraction** using MFCC (Mel-Frequency Cepstral Coefficients)
- **Neural network inference** with a quantized DS-CNN model
- **Bare-metal deployment** on an STM32H7A3ZIQ (Cortex-M7, 280 MHz)

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | STM32H7A3ZIQ (Cortex-M7, 280 MHz, 1 MB RAM, 2 MB Flash) |
| Microphone | INMP441 MEMS (I2S format, 16 kHz, omnidirectional) |
| Board | NUCLEO-H7A3ZI-Q |
| Audio interface | SAI peripheral (I2S format, 32-bit frames, 24-bit data) |
| Debug interface | UART — used for raw PCM streaming to PC during validation |

---

## Project Status

```
[✅] Hardware bring-up        — SAI + INMP441 acquisition validated via UART → WAV
[✅] Dataset generation       — 2000 wakeword + 2000 negative synthetic samples
[✅] Feature extraction       — MFCC (97×40) with per-feature normalisation
[✅] Model training           — DS-CNN + SE: 98.7% accuracy, 32K params, 31 KB INT8
[🔄] INT8 quantization        — QAT via TFLite (next)
[⬜] STM32Cube.AI deployment  — C inference code generation + firmware integration (planned)
[⬜] Benchmarking             — Inference time, RAM/Flash profiling on target (planned)
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         PYTHON (Host)                            │
│                                                                  │
│  edge-tts TTS ──► augmentation ──► WAV dataset                  │
│                                         │                        │
│                               MFCC feature extraction           │
│                                         │                        │
│                               DS-CNN training (TensorFlow)       │
│                                         │                        │
│                               INT8 post-training quantization    │
│                                         │                        │
│                               STM32Cube.AI code generation       │
└─────────────────────────────────────────┼────────────────────────┘
                                          │ Generated C
┌─────────────────────────────────────────▼────────────────────────┐
│                       STM32H7A3ZIQ (Target)                      │
│                                                                  │
│  INMP441 ──SAI(I2S)──► Circular buffer ──► MFCC ──► Inference   │
│                                                        │         │
│                                Confidence threshold + debounce   │
│                                                        │         │
│                                                Wake word event   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Hardware Validation

Before training, the audio acquisition chain was validated end-to-end:

```
INMP441 ──SAI(I2S)──► STM32 buffer ──► UART ──► PC ──► WAV file
```

Raw PCM frames are streamed over UART from the STM32 and reconstructed into `.wav` files on the PC side for acoustic inspection. This confirms correct I2S framing, 24-bit data alignment in 32-bit SAI frames, and expected microphone frequency response at 16 kHz before committing to the inference firmware.

---

## Dataset Generation

The dataset is fully synthetic, generated via **Microsoft Neural TTS** (`edge-tts`) to maximize voice diversity without requiring manual recording sessions.

### Wakeword: "Gragas"

A key phonetic challenge: the French language suppresses the final **S** in "Gragas" by default. To ensure the model learns the intended pronunciation (with audible S), the TTS prompts use deliberate orthographic tricks:

| Variant | Technique |
|---|---|
| `"gragasse"` | Forces hard final S via French phonology rules |
| `"gragace"` | Alternative phonetic approximation |
| `"GRAGASSE"` | Uppercase for prosody variation |
| `"gragas's"` | Apostrophe to break liaison suppression |

### Voice diversity

| Voice | Gender | Locale |
|---|---|---|
| Henri, Remy | Male | fr-FR |
| Antoine | Male | fr-CA |
| Denise, Eloise | Female | fr-FR |
| Sylvie | Female | fr-CA |
| Charline | Female | fr-BE |

### Augmentation pipeline

Each base TTS sample is augmented with `audiomentations`:

- Gaussian noise injection
- Time stretching (0.8× – 1.25×)
- Pitch shifting (±4 semitones)
- Random time shift
- Gain variation (±6 dB)
- Low-pass / high-pass filtering (room + mic simulation)

### Dataset statistics

| Class | Count | Source |
|---|---|---|
| `wakeword` (Gragas) | 2 000 | TTS × 7 voices × 7 text variants × 5 prosody variants → augmented |
| `negative` | 2 000 | TTS (French words) + synthetic silence + Gaussian noise |

> **Known limitation:** the dataset is entirely synthetic. A real deployment will benefit from a fine-tuning pass with actual INMP441 recordings to close the domain gap between TTS audio and real microphone output.

---

## Feature Extraction (MFCC)

Each 1-second WAV clip is transformed into a compact 2D feature map via **MFCC (Mel-Frequency Cepstral Coefficients)** before being fed to the neural network.

### Pipeline

```
WAV (16 000 samples)
 → Pre-emphasis (0.97)
 → Framing (40 ms window, 10 ms hop → 97 frames)
 → Hann window
 → 1024-point FFT → Power spectrum
 → 40 Mel filter banks → Log compression
 → DCT → 40 MFCC coefficients
 → Per-feature normalization (global μ/σ)

Output: (97, 40) matrix per sample — 3.9 KB in INT8
```

### Parameter choices

| Parameter | Value | Rationale |
|---|---|---|
| Mel bands / MFCC coefficients | 40 / 40 | DS-CNN reference config (Zhang et al., 2017). Keeping all 40 preserves full spectral detail — critical for discriminating the final sibilant in "Gragas". STM32H7 has the RAM headroom (3.9 KB vs 1 MB). |
| Window / Hop | 40 ms / 10 ms | 75 % overlap. Window captures ≥ 2 pitch cycles; hop gives fine temporal resolution for phoneme transitions. |
| FFT size | 1024 | Power-of-2 ≥ window length (640). Zero-pads each frame for finer frequency resolution via CMSIS-DSP `arm_rfft_f32`. |
| Normalization | Per-feature global (z-score) | Cheapest at inference (40 sub + 40 mul per frame, ≈ 28 µs). Preserves absolute energy — important for silence/noise rejection. Only 320 B of Flash for the stored μ/σ arrays. |

> **On-device parity:** The same pipeline parameters will be reimplemented in C using CMSIS-DSP on the STM32. `librosa` is used during training only; the parameter set documented here is the single source of truth for both sides.

---

## Repository Structure

```
STM32_EmbeddedAI/
│
├── Python/
│   ├── DatasetGeneration/
│   │   └── generate_wakeword_dataset.py   # TTS + augmentation pipeline (edge-tts)
│   ├── PythonMicBridge/
│   │   └── MicBridgeCOM.py                # UART → WAV bridge for hardware validation
│   ├── Preprocessing/
│   │   ├── mfcc_config.py                 # MFCC hyperparameters (single source of truth)
│   │   └── extract_mfcc.py                # WAV → MFCC → .npy extraction pipeline
│   ├── Training/
│   │   ├── train_config.py                # Training hyperparameters
│   │   ├── ds_cnn_model.py                # DS-CNN + SE attention + SpecAugment
│   │   ├── train.py                       # Main training pipeline
│   │   └── evaluate.py                    # Metrics, confusion matrix
│   ├── dataset/                           # Generated WAV samples
│   ├── features/                          # Extracted MFCC arrays (X.npy, y.npy)
│   └── models/                            # Trained models (.keras, .tflite)
│
└── STM32/
    └── EmbeddedAI/
        ├── Core/                  # Main loop, SAI driver, audio buffer
        ├── AI/                    # Cube.AI generated inference code [planned]
        └── Drivers/               # HAL, SAI, INMP441 driver
```

---

## Roadmap

### Phase 1 — Hardware & Dataset ✅

- [x] SAI + INMP441 I2S acquisition working and validated
- [x] UART → WAV pipeline for acoustic validation on PC
- [x] Synthetic dataset: 2000 wakeword + 2000 negative samples
- [x] Multi-voice, multi-prosody augmentation pipeline

### Phase 2 — Feature Extraction ✅

- [x] MFCC parameter selection and justification documented
- [x] MFCC extraction (40 coefficients, 97 time frames, 1-second window)
- [x] Export `X.npy` / `y.npy` / `norm_stats.npz` for training

### Phase 3 — Model Training ✅

- [x] DS-CNN architecture with SE attention block (32 194 params)
- [x] 2-class classification: `wakeword` / `negative`
- [x] SpecAugment (time + frequency masking) for regularisation
- [x] Train/val/test split (70/15/15) with stratification
- [x] **Test accuracy: 98.7%** — precision 0.99, recall 0.98, FPR 1.0%
- [x] Model size: 125.8 KB float32 / ~31 KB INT8 (target was < 200 KB)

### Phase 4 — Quantization & Export

- [ ] Quantization-Aware Training (QAT) via `tensorflow-model-optimization`
- [ ] Accuracy comparison: float32 vs QAT-INT8
- [ ] `.tflite` export for STM32Cube.AI

### Phase 5 — STM32 Deployment

- [ ] STM32Cube.AI model import and memory validation
- [ ] Integrate inference C code into firmware
- [ ] 1-second sliding window with 50% overlap on SAI circular buffer
- [ ] Confidence threshold (> 85%) + 500 ms detection debounce

### Phase 6 — Benchmarking

- [ ] Inference latency via DWT cycle counter (Cortex-M7)
- [ ] RAM and Flash usage profiling
- [ ] On-device accuracy vs PC baseline
- [ ] False positive rate characterization (target: < 1 / min)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dataset generation | Python, edge-tts (Microsoft Neural TTS), audiomentations |
| Audio features | MFCC — librosa (training) / C implementation (inference) |
| Model architecture | DS-CNN + SE attention (Depthwise Separable CNN) |
| Training framework | TensorFlow / Keras |
| Quantization | Quantization-Aware Training (QAT) → TFLite INT8 |
| Deployment tool | STM32Cube.AI (X-CUBE-AI) |
| Firmware | C, ARM Cortex-M7, STM32 HAL |
| Toolchain | STM32CubeIDE, GCC ARM Embedded |

---

## Model Architecture — DS-CNN + SE Attention

Based on Google's *Hello Edge* DS-CNN (Zhang et al., 2017), enhanced with two techniques from recent KWS research (2024–2025):

```text
Input (97, 40, 1)                     ← MFCC "image": 97 time frames × 40 coefficients
  │
  ├─ SpecAugment                      ← training only: random time/freq masking (regularisation)
  │
  ├─ Conv2D 4×10, 64 filters + BN + ReLU
  │     First conv expands 1 → 64 channels.
  │     The 4×10 kernel spans 4 frames × 10 coefficients
  │     — sized to capture small phoneme-scale patterns.
  │
  ├─ 4× DS-Conv Block:
  │     DepthwiseConv2D 3×3 + BN + ReLU     (filter each channel independently)
  │     → SE Block                           (channel attention — see below)
  │     → Conv2D 1×1 + BN + ReLU            (mix channels back together)
  │
  ├─ GlobalAveragePooling2D           ← collapse spatial dims → 64-d vector
  ├─ Dropout (0.4)                    ← regularisation
  └─ Dense(2, softmax)                ← [P(negative), P(wakeword)]

Estimated: ~28K params · ~28 KB INT8 · inference < 10 ms on Cortex-M7 @ 280 MHz
```

### Why Depthwise Separable Convolutions?

A standard conv with 64 input/output channels and a 3×3 kernel costs **36 864 parameters**. A depthwise separable conv splits this into a per-channel 3×3 filter (576 params) + a 1×1 channel mixer (4 096 params) = **4 672 params** — an **8× reduction** with comparable accuracy. This keeps the model well under the 200 KB Flash target.

### Why SE Attention?

A Squeeze-and-Excitation block learns to weight each of the 64 feature maps by importance. It only adds ~2 048 parameters per block (~8 KB total) but helps the network focus on the most discriminative frequency bands — particularly useful for distinguishing the final sibilant **S** in "Gragas" from similar-sounding words.

### Why SpecAugment?

With only 4 000 synthetic samples, overfitting is a real risk. SpecAugment randomly masks bands of time frames and MFCC coefficients during training, forcing the model to learn robust patterns rather than memorising specific time-frequency regions. It costs nothing at inference — the layer is inactive on-device.

---

## Quantization — QAT (Phase 4)

The trained model is float32 (126 KB). The STM32 targets **INT8** inference — 4× smaller, 2–4× faster via CMSIS-NN SIMD optimisation.

### Why QAT instead of post-training quantization?

Post-training quantization (PTQ) naively rounds float32 weights to INT8, which can lose 2–5% accuracy on small models. **Quantization-Aware Training (QAT)** inserts fake-quantization nodes during a short fine-tuning pass (~10 epochs), so the model learns to tolerate INT8 rounding. Research (2024–2025) consistently shows +1–4% accuracy retention over PTQ at no extra inference cost.

### Pipeline

```text
gragas_dscnn.keras (float32, 126 KB)
  → QAT fine-tune (10 epochs, LR = 1e-5)
  → Convert to TFLite INT8 with representative dataset calibration
  → gragas_dscnn_int8.tflite (~31 KB)
```

### Handoff to Phase 5

The `.tflite` file is the **bridge between Python and C**. STM32Cube.AI imports it and generates:
- C arrays for weights, biases, and tensor buffers
- An inference API (`ai_run()`, `ai_create()`) linked into the firmware
- Memory layout validated against the STM32H7 RAM/Flash budget

## Target Benchmarks

| Metric | Target | Measured (PC) |
|---|---|---|
| Inference latency | < 50 ms | *pending (on-device)* |
| RAM footprint | < 512 KB | *pending (on-device)* |
| Flash (model + firmware) | < 1.5 MB | ~31 KB INT8 model |
| Test set accuracy | > 90 % | **98.7 %** |
| False positive rate | < 1 per minute | 1.0 % on test set |

*PC results on synthetic TTS data. On-device metrics will be measured in Phase 6 with real INMP441 audio.*

---

## Getting Started

### Requirements

```bash
pip install tensorflow librosa numpy scikit-learn matplotlib
pip install edge-tts pydub audiomentations soundfile tqdm
# ffmpeg required: https://ffmpeg.org/download.html
```

STM32CubeIDE + X-CUBE-AI extension pack required for firmware.

### Generate dataset

```bash
cd Python/
python generate_dataset.py
# Output: ./dataset/wakeword/  and  ./dataset/negative/
# Runtime: ~15-25 min
```

### Extract features

```bash
cd Python/Preprocessing
python extract_mfcc.py
# Output: ../features/X.npy, ../features/y.npy, ../features/norm_stats.npz
```

### Train model

```bash
cd Python/Training
python train.py
# Output: ../models/gragas_dscnn.keras
# Evaluate: python evaluate.py
```

---

## License

Apache 2.0 — see [LICENSE](./LICENSE)
