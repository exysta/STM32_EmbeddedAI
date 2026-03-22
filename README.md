# STM32 Embedded AI — Wake Word Detection

> End-to-end TinyML pipeline for real-time wake word detection on a bare-metal STM32H7 microcontroller.  
> The system detects the wake word **"Gragas"** using a neural network running entirely on-device — no cloud, no OS, no external compute.

---

## Overview

This project implements a complete **keyword spotting pipeline** on a resource-constrained embedded system, covering every stage from dataset generation to on-device inference:

- **Hybrid dataset** — synthetic TTS voices (edge-tts) combined with real INMP441 mic recordings captured directly from the STM32 over UART
- **Audio acquisition** via an INMP441 MEMS microphone over SAI peripheral (I2S format)
- **Feature extraction** using MFCC (Mel-Frequency Cepstral Coefficients)
- **Neural network inference** with a quantized DS-CNN model (INT8)
- **Bare-metal deployment** on an STM32H7A3ZIQ (Cortex-M7, 280 MHz)
- **PC validation tool** for real-time microphone testing of the same pipeline

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | STM32H7A3ZIQ (Cortex-M7, 280 MHz, 1 MB RAM, 2 MB Flash) |
| Microphone | INMP441 MEMS (I2S format, 16 kHz, omnidirectional) |
| Board | NUCLEO-H7A3ZI-Q |
| Audio interface | SAI peripheral (I2S format, 32-bit frames, 24-bit data) |
| Debug interface | UART @ 921 600 baud — used for raw PCM streaming to PC (validation and dataset recording) |

---

## Project Status

```
[✅] Phase 1 — Hardware bring-up        SAI + INMP441 acquisition validated via UART → WAV
[✅] Phase 2 — Feature extraction       MFCC (97×40) with per-feature normalisation
[✅] Phase 3 — Model training           DS-CNN + SE retrained on mixed real+TTS dataset; 30 466 params
[✅] Phase 4 — INT8 quantization        Full-integer PTQ → 33 KB weights (C, via ST Edge AI Core v4)
[✅] Phase 5 — STM32 deployment         MFCC + inference running on-device in real time (re-deployed)
[🔄] Phase 6 — Benchmarking            Inference latency / RAM / Flash profiling (in progress)
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         PYTHON (Host)                            │
│                                                                  │
│  record_dataset.py (UART ← STM32 INMP441) ──► real WAVs         │
│  edge-tts TTS ──► augmentation ──► TTS WAVs                      │
│                                         │                        │
│                               dataset_manifest.csv               │
│                               (weight=4.0 real / 1.0 TTS)        │
│                                         │                        │
│                               MFCC feature extraction            │
│                                         │                        │
│                               DS-CNN training (TensorFlow)        │
│                                         │                        │
│                               INT8 post-training quantization     │
│                                         │                        │
│                               ST Edge AI Core v4 code generation  │
└─────────────────────────────────────────┼────────────────────────┘
                                          │ Generated C
┌─────────────────────────────────────────▼────────────────────────┐
│                       STM32H7A3ZIQ (Target)                      │
│                                                                  │
│  INMP441 ──SAI(I2S)──► Ring buffer ──► MFCC ──► INT8 quant       │
│                                                     │            │
│                                          ST-AI inference (DS-CNN) │
│                                                     │            │
│                                 Threshold (80%) + 1.5 s cooldown │
│                                                     │            │
│                                       LED + UART wake word event │
└──────────────────────────────────────────────────────────────────┘
```

---

## Hardware Validation

Before training, the audio acquisition chain was validated end-to-end:

```
INMP441 ──SAI(I2S)──► STM32 buffer ──► UART ──► PC ──► WAV file
```

Raw PCM frames are streamed over UART from the STM32 and reconstructed into `.wav` files on the PC side for acoustic inspection. This confirms correct I2S framing, 24-bit data alignment in 32-bit SAI frames, and expected microphone frequency response at 16 kHz before committing to the inference firmware.

The same UART pipeline is reused by `record_dataset.py` to collect labelled training samples directly from the INMP441 at 921 600 baud.

---

## Dataset Generation

The dataset combines two sources to maximise both voice diversity and domain fidelity.

### Real INMP441 Recordings

Recorded directly from the STM32 INMP441 microphone over UART using `record_dataset.py`:

| Class | Count | Method |
|---|---|---|
| `wakeword` (Gragas) | 90 | Spoken into INMP441, captured via UART @ 921 600 baud |
| `negative` | 150 | Ambient noise + non-wakeword speech, same capture path |

- Each sample is captured as a 1.5 s raw WAV; the best 1-second window (highest RMS energy) is automatically extracted for training.
- Files are written to `dataset/wakeword/` and `dataset/negative/` and registered in `dataset/dataset_manifest.csv` with `weight=4.0`.
- Real samples carry 4× the training weight of TTS samples so the normalisation statistics and loss function are calibrated toward the real-mic distribution.

### Synthetic TTS Corpus

The original TTS corpus is retained alongside the real recordings to maintain voice diversity.

**Wakeword: "Gragas"** — A key phonetic challenge: the French language suppresses the final **S** in "Gragas" by default. Orthographic tricks force the correct pronunciation:

| Variant | Technique |
|---|---|
| `"gragasse"` | Forces hard final S via French phonology rules |
| `"gragace"` | Alternative phonetic approximation |
| `"GRAGASSE"` | Uppercase for prosody variation |
| `"gragas's"` | Apostrophe to break liaison suppression |

**Voice diversity:**

| Voice | Gender | Locale |
|---|---|---|
| Henri, Remy | Male | fr-FR |
| Antoine | Male | fr-CA |
| Denise, Eloise | Female | fr-FR |
| Sylvie | Female | fr-CA |
| Charline | Female | fr-BE |

**Augmentation pipeline** — Each base TTS sample is augmented with `audiomentations`:

- Gaussian noise injection
- Time stretching (0.8× – 1.25×)
- Pitch shifting (±4 semitones)
- Random time shift
- Gain variation (±6 dB)
- Low-pass / high-pass filtering (room + mic simulation)

### Dataset statistics

| Class | TTS / synthetic | Real INMP441 | Total |
|---|---|---|---|
| `wakeword` (Gragas) | ~2 000 | 90 | ~2 090 |
| `negative` | ~2 000 | 150 | ~2 150 |

> **Domain gap note:** The real INMP441 recordings directly address the TTS→real-mic domain gap identified in the original dataset. The weighted training scheme (`weight=4.0` for real, `weight=1.0` for TTS) ensures the model prioritises the real-mic distribution while retaining the synthetic voice diversity.

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
 → Per-feature normalization (global μ/σ, weight-aware for real samples)

Output: (97, 40) matrix per sample — 3.9 KB in INT8
```

### Parameter choices

| Parameter | Value | Rationale |
|---|---|---|
| Mel bands / MFCC coefficients | 40 / 40 | DS-CNN reference config (Zhang et al., 2017). Keeping all 40 preserves full spectral detail — critical for discriminating the final sibilant in "Gragas". STM32H7 has the RAM headroom (3.9 KB vs 1 MB). |
| Window / Hop | 40 ms / 10 ms | 75 % overlap. Window captures ≥ 2 pitch cycles; hop gives fine temporal resolution for phoneme transitions. |
| FFT size | 1024 | Power-of-2 ≥ window length (640). Zero-pads each frame for finer frequency resolution via CMSIS-DSP `arm_rfft_f32`. |
| Normalization | Per-feature global (z-score) | Cheapest at inference (40 sub + 40 mul per frame, ≈ 28 µs). Preserves absolute energy — important for silence/noise rejection. Only 320 B of Flash for the stored μ/σ arrays. |

> **On-device parity:** The same pipeline parameters are reimplemented in C using CMSIS-DSP on the STM32. `librosa` is used during training only; the parameter set documented here is the single source of truth for both sides.

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

Parameters : 30 466  (32.38 KiB weights in INT8 C code)
```

### Why Depthwise Separable Convolutions?

A standard conv with 64 input/output channels and a 3×3 kernel costs **36 864 parameters**. A depthwise separable conv splits this into a per-channel 3×3 filter (576 params) + a 1×1 channel mixer (4 096 params) = **4 672 params** — an **8× reduction** with comparable accuracy. This keeps the model well under the 200 KB Flash target.

### Why SE Attention?

Squeeze-and-Excitation (SE) blocks add a lightweight channel recalibration step after each depthwise conv: global average pooling collapses the spatial map to a 64-d vector, two small FC layers (64→16→64) produce per-channel attention weights, and the output is rescaled channel-wise. This costs only ~2 100 additional parameters per block but lets the network learn which spectral features are most diagnostic for each phoneme — particularly useful for discriminating the final /s/ sibilant in "Gragas" from similar stop consonants.

---

## INT8 Quantization

Full-integer post-training quantization (PTQ) converts the float32 model to INT8 for deployment.

### Cube.AI generation report (2026-03-22)

| Field | Value |
|---|---|
| Tool | ST Edge AI Core v4.0.0-20500 |
| Model | `gragas_dscnn_int8.tflite` |
| Format | `ss/sa` per-channel |
| Parameters | 30 466 items |
| Weights (Flash, ro) | 33 160 B (32.38 KiB) |
| Activations (RAM, rw) | 516 608 B (504.50 KiB) |
| Total Flash footprint | ~78 971 B (~77 KB) |
| Total RAM footprint | ~524 548 B (~512.25 KB) |
| MACCs | 84 686 624 |

### Quantization parameters

| Tensor | Scale | Zero-point |
|---|---|---|
| Input (`serving_default_mfcc_input0`, int8 1×97×40×1) | 0.057945 | +1 |
| Output (`nl_47`, int8 1×2) | 0.003906 | −128 |

These values are referenced in `wakeword_inference.c` via the `STAI_NETWORK_IN_1_SCALE`, `STAI_NETWORK_IN_1_ZERO_POINT`, `STAI_NETWORK_OUT_1_SCALE`, and `STAI_NETWORK_OUT_1_ZERO_POINT` macros generated in `network.h`.

---

## STM32 Deployment

### Key source files

| File | Role |
|---|---|
| `Core/Src/main.c` | Main loop, SAI DMA, inference dispatch |
| `Core/Src/mfcc_processing.c` | Full MFCC pipeline in C using CMSIS-DSP (`arm_rfft_fast_f32`) |
| `Core/Src/wakeword_inference.c` | Normalisation, INT8 quantisation, ST-AI inference glue, threshold + cooldown |
| `Core/Inc/norm_stats.h` | Per-coefficient mean/std arrays exported from Python (`norm_stats.npz`) |
| `Core/Src/network.c` | ST-AI generated model weights and graph (from `.tflite`) |
| `Core/Src/network_data.c` | ST-AI generated tensor data |

---

## Repository Structure

```
STM32_EmbeddedAI/
│
├── Python/
│   ├── DatasetGeneration/
│   │   ├── generate_wakeword_dataset.py   # TTS + augmentation pipeline (edge-tts)
│   │   └── record_dataset.py             # UART recorder: real INMP441 samples from STM32
│   ├── PythonMicBridge/
│   │   └── MicBridgeCOM.py                # UART → WAV bridge for hardware validation
│   ├── Preprocessing/
│   │   ├── mfcc_config.py                 # MFCC hyperparameters (single source of truth)
│   │   └── extract_mfcc.py                # WAV → MFCC → .npy (manifest-aware, weighted)
│   ├── Training/
│   │   ├── train_config.py                # Training hyperparameters
│   │   ├── ds_cnn_model.py                # DS-CNN + SE attention + SpecAugment
│   │   ├── train.py                       # Main training pipeline
│   │   ├── evaluate.py                    # Metrics, confusion matrix
│   │   └── quantize.py                    # PTQ INT8 quantization + TFLite export
│   ├── Testing/
│   │   └── pc_wakeword_test.py            # Real-time PC mic test (mirrors STM32 pipeline)
│   ├── export_norm_stats.py               # norm_stats.npz → C header generator
│   ├── dataset/
│   │   ├── dataset_manifest.csv           # All WAV paths + labels + weights (auto-generated)
│   │   ├── wakeword/                      # Real INMP441 recordings + TTS samples
│   │   ├── negative/                      # Real INMP441 recordings + TTS samples
│   │   └── debug/                         # Raw 1.5 s captures from record_dataset.py
│   ├── features/                          # Extracted MFCC arrays (X.npy, y.npy, weights.npy)
│   └── models/                            # Trained models (.keras, .tflite)
│
└── STM32/
    └── EmbeddedAI/
        ├── Core/
        │   ├── Src/
        │   │   ├── main.c                 # Main loop, SAI DMA, inference dispatch
        │   │   ├── mfcc_processing.c      # MFCC pipeline (CMSIS-DSP)
        │   │   ├── wakeword_inference.c   # Quantise + infer + threshold + cooldown
        │   │   ├── network.c              # ST-AI generated model graph
        │   │   └── network_data.c         # ST-AI generated weights
        │   └── Inc/
        │       ├── mfcc_processing.h      # MFCC public API
        │       ├── wakeword_inference.h   # Inference public API
        │       ├── norm_stats.h           # Mean/std arrays (from Python)
        │       ├── network.h              # ST-AI generated header
        │       └── network_details.h      # ST-AI model metadata
        └── Drivers/                       # HAL, SAI, INMP441 driver
```

---

## Roadmap

### Phase 1 — Hardware & Dataset ✅

- [x] SAI + INMP441 I2S acquisition working and validated
- [x] UART → WAV pipeline for acoustic validation on PC
- [x] Synthetic dataset: ~2 000 wakeword + ~2 000 negative TTS samples
- [x] Multi-voice, multi-prosody augmentation pipeline

### Phase 2 — Feature Extraction ✅

- [x] MFCC parameter selection and justification documented
- [x] MFCC extraction (40 coefficients, 97 time frames, 1-second window)
- [x] Export `X.npy` / `y.npy` / `weights.npy` / `norm_stats.npz` for training
- [x] Manifest-aware extraction with per-sample weighting (real=4.0, TTS=1.0)

### Phase 3 — Model Training ✅

- [x] DS-CNN architecture with SE attention block (30 466 params)
- [x] 2-class classification: `wakeword` / `negative`
- [x] SpecAugment (time + frequency masking) for regularisation
- [x] Train/val/test split (70/15/15) with stratification
- [x] Weighted training: real INMP441 samples weighted 4× vs TTS
- [x] Model retrained from scratch on mixed real+TTS dataset

### Phase 4 — Quantization & Export ✅

- [x] Full-integer PTQ with representative dataset calibration
- [x] INT8 model: `gragas_dscnn_int8.tflite`
- [x] C code generated with ST Edge AI Core v4.0.0-20500
- [x] Weights: 33 160 B (32.38 KiB) · Activations: 504.50 KiB · Total Flash: ~77 KB
- [x] PC real-time test script (`pc_wakeword_test.py`) — validates TFLite model on live microphone

### Phase 5 — STM32 Deployment ✅

- [x] STM32Cube.AI model import and C code generation (`network.c/h`)
- [x] MFCC pipeline reimplemented in C with CMSIS-DSP (`mfcc_processing.c`)
- [x] Normalisation + INT8 quantisation + inference glue (`wakeword_inference.c`)
- [x] `norm_stats.h` exported from Python (`export_norm_stats.py`)
- [x] 1-second accumulation buffer on SAI circular DMA
- [x] Confidence threshold (≥ 80 %) + 1.5 s detection cooldown
- [x] LED feedback (2.5 s on trigger) + UART debug output

### Phase 6 — Benchmarking

- [ ] Inference latency via DWT cycle counter (Cortex-M7)
- [ ] RAM and Flash usage profiling
- [ ] On-device accuracy vs PC baseline
- [ ] False positive rate characterization (target: < 1 / min)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Real dataset recording | `record_dataset.py` — UART PCM stream @ 921 600 baud → WAV |
| Synthetic dataset generation | Python, edge-tts (Microsoft Neural TTS), audiomentations |
| Audio features | MFCC — librosa (training) / CMSIS-DSP C implementation (inference) |
| Model architecture | DS-CNN + SE attention (Depthwise Separable CNN) |
| Training framework | TensorFlow / Keras |
| Quantization | Full-integer PTQ → TFLite INT8 |
| Deployment tool | ST Edge AI Core v4.0.0 (STM32Cube.AI) |
| Firmware | C, ARM Cortex-M7, STM32 HAL, CMSIS-DSP |
| Toolchain | STM32CubeIDE, GCC ARM Embedded 13.3 |

---

## Target Benchmarks

| Metric | Target | Measured |
|---|---|---|
| Inference latency | < 50 ms | *pending (on-device, Phase 6)* |
| RAM footprint | < 600 KB | 504.50 KiB activations / ~512.25 KB total |
| Flash (weights) | — | 32.38 KiB (33 160 B) |
| Flash (model + RT + firmware) | < 1.5 MB | ~77 KB model+RT (well within budget) |
| Test set accuracy | > 90 % | *pending re-evaluation on mixed dataset* |
| False positive rate | < 1 per minute | *pending (on-device, Phase 6)* |

> **RAM note:** Activations alone occupy 504.50 KiB; total RAM (activations + kernel) is ~512.25 KB, marginally above the original 512 KB target. The STM32H7A3ZIQ has 1 MB RAM so there is no hard constraint — the target has been revised to **< 600 KB** to reflect the actual model footprint.

*On-device metrics will be measured in Phase 6 with real INMP441 audio.*

---

## Getting Started

### Requirements

```bash
pip install tensorflow librosa numpy scikit-learn matplotlib
pip install edge-tts pydub audiomentations soundfile tqdm
pip install sounddevice scipy          # for PC mic testing
pip install pyserial                   # for record_dataset.py
# ffmpeg required: https://ffmpeg.org/download.html
```

STM32CubeIDE + ST Edge AI Core (X-CUBE-AI v4+) required for firmware.

### (Optional) Record real INMP441 samples

Before or after TTS generation you can record labelled samples directly from the STM32 INMP441:

```bash
# 1. Flash firmware with HAL_UART_Transmit enabled and MFCC_IngestBlock disabled.
# 2. Run the recorder (921 600 baud default):
cd Python/DatasetGeneration
python record_dataset.py --port COM7 --mode wakeword --count 30
python record_dataset.py --port COM7 --mode negative --count 30
# Output: ../dataset/wakeword/*.wav  and  ../dataset/negative/*.wav
#         ../dataset/debug/{wakeword,negative}/*_raw.wav  (full 1.5 s debug captures)
# dataset_manifest.csv is created/updated automatically.
```

### Generate TTS dataset

```bash
cd Python/DatasetGeneration
python generate_wakeword_dataset.py
# Output: ../dataset/wakeword/  and  ../dataset/negative/
# Runtime: ~15-25 min
```

### Extract features

```bash
cd Python/Preprocessing
python extract_mfcc.py
# Manifest mode (preferred): reads dataset_manifest.csv, applies real-sample weights (4×)
# Fallback: directory scan if no manifest present
# Output: ../features/X.npy, ../features/y.npy, ../features/weights.npy,
#         ../features/norm_stats.npz
```

### Train model

```bash
cd Python/Training
python train.py
# Output: ../models/gragas_dscnn.keras
# Evaluate: python evaluate.py
```

### Quantize to INT8

```bash
cd Python/Training
python quantize.py
# Output: ../models/gragas_dscnn_int8.tflite
```

### Test on PC microphone

```bash
cd Python/Testing
python pc_wakeword_test.py
# Streams your microphone, prints P(wakeword) for each 1-sec sliding window
# Press Ctrl+C to quit
```

### Deploy to STM32

1. Import `gragas_dscnn_int8.tflite` into ST Edge AI Core Studio (STM32Cube.AI v4+)
2. Generate C model files → copy to `STM32/EmbeddedAI/Core/Src/`
3. Build and flash via STM32CubeIDE
4. Speak "Gragas" into the INMP441 — green LED lights up on detection

---

## License

Apache 2.0 — see [LICENSE](./LICENSE)
