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
[🔄] Feature extraction       — MFCC preprocessing pipeline (in progress)
[⬜] Model training           — DS-CNN architecture (planned)
[⬜] INT8 quantization        — Post-training quantization via TFLite (planned)
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

## Repository Structure

```
STM32_EmbeddedAI/
│
├── Python/
│   ├── generate_dataset.py        # TTS + augmentation pipeline (edge-tts)
│   ├── preprocessing/             # MFCC feature extraction  [planned]
│   ├── training/                  # DS-CNN model definition  [planned]
│   ├── quantization/              # INT8 TFLite export       [planned]
│   └── evaluation/                # Metrics, confusion matrix [planned]
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

### Phase 2 — Feature Extraction 🔄
- [ ] WAV → 16 kHz mono normalization
- [ ] MFCC extraction (40 coefficients, 98 time frames, 1-second window)
- [ ] Export `X.npy` / `y.npy` for training

### Phase 3 — Model Training
- [ ] DS-CNN architecture (Depthwise Separable CNN)
- [ ] 3-class classification: `gragas` / `negative` / `silence`
- [ ] Train/val/test split with stratification
- [ ] Target: > 90% accuracy, < 200 KB model size

### Phase 4 — Quantization & Export
- [ ] INT8 post-training quantization via TensorFlow Lite
- [ ] Accuracy comparison: float32 vs INT8
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
| Model architecture | DS-CNN (Depthwise Separable CNN) |
| Training framework | TensorFlow / Keras |
| Quantization | TFLite INT8 post-training quantization |
| Deployment tool | STM32Cube.AI (X-CUBE-AI) |
| Firmware | C, ARM Cortex-M7, STM32 HAL |
| Toolchain | STM32CubeIDE, GCC ARM Embedded |

---

## Why DS-CNN?

Depthwise Separable CNNs are the reference architecture for keyword spotting on microcontrollers, introduced in Google's *Hello Edge* paper (Zhang et al., 2017). They factorize standard convolutions into depthwise + pointwise operations, achieving near state-of-the-art accuracy at a fraction of the compute and memory cost — well within the constraints of a Cortex-M7 running without a hardware ML accelerator.

---

## Target Benchmarks

| Metric | Target |
|---|---|
| Inference latency | < 50 ms |
| RAM footprint | < 512 KB |
| Flash (model + firmware) | < 1.5 MB |
| Test set accuracy | > 90 % |
| False positive rate | < 1 per minute |

*Based on STM32H7A3ZIQ specs (1 MB RAM, 2 MB Flash, 280 MHz Cortex-M7). Will be updated with measured values.*

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

### Extract features *(planned)*

```bash
python preprocessing/extract_mfcc.py --input dataset/ --output features/
```

### Train model *(planned)*

```bash
python training/train.py --features features/ --output models/
```

---


## License

Apache 2.0 — see [LICENSE](./LICENSE)
