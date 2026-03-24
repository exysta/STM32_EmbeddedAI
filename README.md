# STM32 Embedded AI — Wake Word Detection

> **A voice-activated AI that runs on a chip smaller than a coin — no internet, no smartphone, no cloud.**  
> Say the word. The LED lights up. Everything happens locally, in under a second.

---

## What this project is — plain English

Most voice assistants (Siri, Alexa, Google) work by sending everything you say to a remote server and waiting for a response. That requires a network connection, costs money per request, introduces latency, and raises privacy concerns.

This project does the opposite. A tiny neural network — small enough to fit in **33 KB of memory**, less than a typical JPG photo — runs entirely inside a microcontroller the size of a fingernail. It listens continuously to a microphone, and when it hears the wake word **"Gragas"**, it responds in under a second. No internet. No operating system. No external computer of any kind.

The entire pipeline was built from scratch: recording a custom voice dataset, training an AI model on a PC, compressing it to a fraction of its original size, and deploying it to the chip in C code. Every stage — from raw audio to detected wake word — was validated on real hardware.

---

## Key results at a glance

| What | Result |
|---|---|
| 🎯 **Peak detection confidence** | **99.6%** — the model was as certain as physically possible |
| ⚡ **Response time** | **716 ms** total — faster than most people clap |
| 💾 **Model size on chip** | **32 KB** — 97% smaller than the original trained model |
| 🔇 **Runs without internet** | Yes — 100% on-device, zero cloud dependency |
| 🔋 **Runs without an OS** | Yes — bare-metal C, no Linux, no RTOS |
| 📏 **Chip used** | STM32H7 microcontroller (smaller than a €2 coin) |
| 🎤 **Trained on real mic data** | Yes — recorded from the actual hardware microphone |
| 📊 **Latency stability** | < 0.5 ms variation over 1 800+ measurements |

---

## Why this is hard

Fitting a working AI on a microcontroller involves solving several compounding problems simultaneously:

**The memory problem.** A standard trained neural network is ~126 KB of floating-point weights. A microcontroller has no disk, no swap space, and often less RAM than a floppy disk. The model was compressed to INT8 format, shrinking it to 32 KB — while keeping it accurate enough to reliably detect a single word.

**The compute problem.** The chip runs at 280 MHz with no GPU. The model performs 84 million mathematical operations per inference. Getting this to complete in time required enabling hardware caches, optimising the compiler flags, and fixing a subtle memory coherency bug that occurs when the DMA controller and the CPU cache disagree about what's in RAM.

**The data problem.** There is no public dataset of people saying "Gragas" into an STM32 microphone. The dataset was built entirely from scratch: synthetic voices generated via Microsoft Neural TTS (7 French voices, multiple phonetic variants), augmented with noise and pitch shifts, then supplemented with 240 real recordings captured directly from the hardware microphone over a serial port.

**The domain gap problem.** A model trained only on synthetic speech fails on real microphone audio because the two sound different. The solution was to record real samples from the exact same microphone used at inference time, then weight those samples 4× more heavily during training so the model prioritises the real-world distribution.

---

## Performance summary

Benchmarked on **NUCLEO-H7A3ZI-Q** (STM32H7A3ZIQ, Cortex-M7 @ 280 MHz) over **1 800+ consecutive measurements** using the chip's own hardware cycle counter.

```
┌─────────────────────────────────────────────────────────────┐
│  MFCC feature extraction       3.90 ms  (ambient)           │
│                                33.42 ms  (speech detected)  │
│  Neural network inference    683    ms                       │
│  ─────────────────────────────────────                       │
│  Total pipeline              ~716   ms  ✅ fits in 1 000 ms  │
│  Headroom remaining          ~284   ms                       │
│                                                              │
│  Peak confidence              99.61%  (INT8 maximum)         │
│  Model Flash footprint        32 KB   (1.6% of 2 MB budget)  │
│  RAM footprint               512 KB   (50% of 1 MB budget)   │
│  Latency jitter              < 0.5 ms over 1 800+ runs       │
└─────────────────────────────────────────────────────────────┘
```

---

## What was built — for technical readers

A complete end-to-end TinyML keyword spotting pipeline:

- **Custom dataset** — ~4 200 samples combining Microsoft Neural TTS (7 French voices, 7 phonetic variants, 5 prosody variants, 6-transform augmentation) and 240 real INMP441 recordings streamed over UART from the STM32 at 921 600 baud
- **Feature extraction** — MFCC (97×40, 40 ms window, 10 ms hop, 1024-point FFT), identical implementation in Python/librosa for training and C/CMSIS-DSP for inference
- **Model** — DS-CNN + Squeeze-and-Excitation attention, 30 466 parameters, trained in TensorFlow/Keras with SpecAugment regularisation and weighted sampling
- **Quantization** — Full-integer post-training quantization via TFLite → 32 KB INT8 weights, C code generated by ST Edge AI Core v4.0.0
- **Firmware** — Bare-metal C on STM32H7A3ZIQ: SAI/DMA audio capture, CMSIS-DSP MFCC pipeline, DMA coherency management, 80% confidence threshold, 1.5 s cooldown, LED + UART output
- **Optimisation journey** — I-Cache + D-Cache enablement, DMA cache coherency fix (SCB_InvalidateDCache_by_Addr), -Ofast on MFCC path; latency reduced from 1907 ms → 716 ms

---

## Project status — all phases complete

```
[✅] Phase 1 — Hardware bring-up        SAI + INMP441 acquisition validated via UART → WAV
[✅] Phase 2 — Feature extraction       MFCC (97×40) with per-feature normalisation
[✅] Phase 3 — Model training           DS-CNN + SE retrained on mixed real+TTS dataset; 30 466 params
[✅] Phase 4 — INT8 quantization        Full-integer PTQ → 33 KB weights (C, via ST Edge AI Core v4)
[✅] Phase 5 — STM32 deployment         MFCC + inference running on-device in real time
[✅] Phase 6 — Benchmarking             716 ms pipeline · 3.90 ms MFCC · 683 ms inference · 99.6% peak confidence
```

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | STM32H7A3ZIQ (Cortex-M7, 280 MHz, 1 MB RAM, 2 MB Flash) |
| Microphone | INMP441 MEMS (I2S format, 16 kHz, omnidirectional) |
| Board | NUCLEO-H7A3ZI-Q |
| Audio interface | SAI peripheral (I2S format, 32-bit frames, 24-bit data) |
| Debug interface | UART @ 921 600 baud — PCM streaming, dataset recording, inference debug |

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
- Files are registered in `dataset/dataset_manifest.csv` with `weight=4.0`.
- Real samples carry 4× the training weight of TTS samples so the model prioritises the real-mic distribution.

### Synthetic TTS Corpus

**Wakeword: "Gragas"** — A key phonetic challenge: French suppresses the final **S** by default. Orthographic tricks force the correct pronunciation:

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

**Augmentation:** Gaussian noise, time stretch (0.8×–1.25×), pitch shift (±4 st), time shift, gain (±6 dB), low/high-pass filtering.

### Dataset statistics

| Class | TTS / synthetic | Real INMP441 | Total |
|---|---|---|---|
| `wakeword` (Gragas) | ~2 000 | 90 | ~2 090 |
| `negative` | ~2 000 | 150 | ~2 150 |

---

## Feature Extraction (MFCC)

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

| Parameter | Value | Rationale |
|---|---|---|
| Mel bands / MFCC coefficients | 40 / 40 | DS-CNN reference config (Zhang et al., 2017). Preserves spectral detail for final sibilant in "Gragas". |
| Window / Hop | 40 ms / 10 ms | 75% overlap. Captures ≥ 2 pitch cycles; fine temporal resolution for phoneme transitions. |
| FFT size | 1024 | Power-of-2 ≥ window length (640). Finer frequency resolution via CMSIS-DSP `arm_rfft_f32`. |
| Normalization | Per-feature z-score | 40 sub + 40 mul per frame (≈ 28 µs). Only 320 B of Flash for stored μ/σ arrays. |

> **On-device parity:** Identical parameters reimplemented in C using CMSIS-DSP. `librosa` is used during training only.

---

## Model Architecture — DS-CNN + SE Attention

Based on Google's *Hello Edge* DS-CNN (Zhang et al., 2017), enhanced with Squeeze-and-Excitation channel attention:

```
Input (97, 40, 1)
  ├─ SpecAugment                      ← training only
  ├─ Conv2D 4×10, 64 filters + BN + ReLU
  ├─ 4× DS-Conv Block:
  │     DepthwiseConv2D 3×3 + BN + ReLU
  │     → SE Block (64→16→64 channel attention)
  │     → Conv2D 1×1 + BN + ReLU
  ├─ GlobalAveragePooling2D
  ├─ Dropout (0.4)
  └─ Dense(2, softmax)   →   [P(negative), P(wakeword)]

Parameters : 30 466  ·  32.38 KiB INT8 weights
```

Depthwise separable convolutions give an **8× parameter reduction** vs standard convolutions. SE blocks add per-channel attention at ~2 100 params each, helping the model focus on spectral features diagnostic of the final /s/ in "Gragas".

---

## INT8 Quantization — Cube.AI Report (2026-03-22)

| Field | Value |
|---|---|
| Tool | ST Edge AI Core v4.0.0-20500 |
| Parameters | 30 466 |
| Weights (Flash) | 33 160 B (32.38 KiB) |
| Activations (RAM) | 516 608 B (504.50 KiB) |
| Total Flash footprint | ~78 971 B (~77 KB) |
| Total RAM | ~524 548 B (~512 KB) |
| MACCs | 84 686 624 |

| Tensor | Scale | Zero-point |
|---|---|---|
| Input (int8, 1×97×40×1) | 0.057945 | +1 |
| Output (int8, 1×2) | 0.003906 | −128 |

---

## STM32 Deployment — Key Source Files

| File | Role |
|---|---|
| `Core/Src/main.c` | Main loop, SAI DMA, inference dispatch |
| `Core/Src/mfcc_processing.c` | MFCC pipeline (CMSIS-DSP) |
| `Core/Src/wakeword_inference.c` | Normalisation, INT8 quantisation, ST-AI glue, threshold + cooldown |
| `Core/Inc/norm_stats.h` | Per-coefficient mean/std exported from Python |
| `Core/Src/network.c` / `network_data.c` | ST-AI generated model graph and weights |

---

## Benchmark Results

> Measured on **NUCLEO-H7A3ZI-Q** (STM32H7A3ZIQ, Cortex-M7 @ 280 MHz).  
> Build: Release, `-Ofast` (MFCC), `-O2` (ST-AI glue), I-Cache + D-Cache enabled, DMA coherency fix applied.  
> Sample size: **1 800+ consecutive 1-second windows** via DWT hardware cycle counter.

### Latency

| Component | Mean | Min | Max | σ |
|---|---|---|---|---|
| MFCC — ambient (gate closed) | **3.90 ms** | 3.87 ms | 3.91 ms | < 0.02 ms |
| MFCC — speech (gate open) | **33.42 ms** | 33.41 ms | 33.44 ms | < 0.02 ms |
| Inference (INT8 DS-CNN) | **683 ms** | 682.54 ms | 683.12 ms | < 0.5 ms |
| **Total pipeline (speech path)** | **~716 ms** | — | — | — |
| **Budget headroom** | **~284 ms** | — | — | — |

The pipeline is fully deterministic — sub-millisecond variance over 1 800+ windows, no jitter from cache or DMA effects.

### Detection confidence

| Scenario | p_ww | raw INT8 |
|---|---|---|
| Peak detection (best window) | **0.9961** | **127** (INT8 maximum) |
| Typical strong detection | 0.89 – 0.99 | 92 – 127 |
| Word entering / leaving window | 0.15 – 0.65 | varies |
| Ambient silence | < 0.05 | −128 (gate closed, no inference run) |

### Memory footprint

| Segment | Size | % of budget |
|---|---|---|
| Model weights (Flash, ro) | **32.38 KiB** | 1.6% of 2 MB Flash |
| ST-AI runtime (Flash) | 26.3 KiB | 1.3% of 2 MB Flash |
| Total model + runtime | **~77 KB** | 3.8% of 2 MB Flash |
| Model activations (RAM) | **504.50 KiB** | 49.3% of 1 MB RAM |
| Total RAM | **~512 KB** | 50.0% of 1 MB RAM |

### System-level summary

| Metric | Target | Measured | Status |
|---|---|---|---|
| MFCC latency | < 15 ms | **3.90 ms** (ambient) / **33.42 ms** (speech) | ✅ |
| Inference latency | < 750 ms | **683 ms** | ✅ |
| Total pipeline | < 1 000 ms | **~716 ms** (284 ms headroom) | ✅ |
| Latency determinism | — | **σ < 0.5 ms** over 1 800+ windows | ✅ |
| Flash (model + runtime) | < 1.5 MB | **~77 KB** | ✅ |
| RAM footprint | < 600 KB | **~512 KB** | ✅ |
| Peak detection confidence | — | **p_ww = 0.9961** (INT8 max) | ✅ |
| On-device accuracy (held-out) | > 90 % | *not measured — future work* | 🔄 |
| False positive rate | < 2 / hr | *not measured — future work* | 🔄 |

---

## Roadmap

### Phase 1 — Hardware & Dataset ✅
- [x] SAI + INMP441 I2S acquisition working and validated
- [x] UART → WAV pipeline for acoustic validation on PC
- [x] Synthetic dataset: ~2 000 wakeword + ~2 000 negative TTS samples
- [x] Multi-voice, multi-prosody augmentation pipeline

### Phase 2 — Feature Extraction ✅
- [x] MFCC extraction (40 coefficients, 97 time frames, 1-second window)
- [x] Export `X.npy` / `y.npy` / `weights.npy` / `norm_stats.npz`
- [x] Manifest-aware extraction with per-sample weighting (real=4.0, TTS=1.0)

### Phase 3 — Model Training ✅
- [x] DS-CNN + SE attention (30 466 params), SpecAugment, stratified 70/15/15 split
- [x] Weighted training: real INMP441 samples weighted 4× vs TTS
- [x] Model retrained from scratch on mixed real+TTS dataset

### Phase 4 — Quantization & Export ✅
- [x] Full-integer PTQ → `gragas_dscnn_int8.tflite`
- [x] C code generated with ST Edge AI Core v4.0.0-20500
- [x] Weights: 33 160 B · Activations: 504.50 KiB · Total Flash: ~77 KB

### Phase 5 — STM32 Deployment ✅
- [x] MFCC pipeline in C with CMSIS-DSP (`mfcc_processing.c`)
- [x] Normalisation + INT8 quantisation + inference glue (`wakeword_inference.c`)
- [x] 1-second accumulation buffer on SAI circular DMA
- [x] Confidence threshold ≥ 80% + 1.5 s cooldown
- [x] LED feedback + UART debug output
- [x] D-Cache coherency fix — `SCB_InvalidateDCache_by_Addr` on DMA buffers

### Phase 6 — Benchmarking ✅
- [x] DWT cycle counter instrumentation on MFCC and inference separately
- [x] 1 800+ window measurement run — latency, variance, detection confidence
- [x] Latency reduced from 1 907 ms → 716 ms (I/D-Cache + -Ofast + coherency fix)
- [x] Peak confidence p_ww = 0.9961 (INT8 saturated maximum)
- [ ] On-device accuracy on held-out real recordings — future work
- [ ] False positive rate under continuous ambient audio — future work

---

## Tech Stack

| Layer | Technology |
|---|---|
| Real dataset recording | `record_dataset.py` — UART PCM @ 921 600 baud → WAV |
| Synthetic dataset | Python, edge-tts (Microsoft Neural TTS), audiomentations |
| Audio features | MFCC — librosa (training) / CMSIS-DSP (inference) |
| Model | DS-CNN + SE attention — TensorFlow / Keras |
| Quantization | Full-integer PTQ → TFLite INT8 |
| Deployment tool | ST Edge AI Core v4.0.0 (STM32Cube.AI) |
| Firmware | C, ARM Cortex-M7, STM32 HAL, CMSIS-DSP |
| Toolchain | STM32CubeIDE, GCC ARM Embedded 13.3 |

---

## Repository Structure

```
STM32_EmbeddedAI/
│
├── Python/
│   ├── DatasetGeneration/
│   │   ├── generate_wakeword_dataset.py   # TTS + augmentation pipeline
│   │   └── record_dataset.py             # UART recorder — real INMP441 samples
│   ├── PythonMicBridge/
│   │   └── MicBridgeCOM.py                # UART → WAV bridge for hardware validation
│   ├── Preprocessing/
│   │   ├── mfcc_config.py                 # MFCC parameters (single source of truth)
│   │   └── extract_mfcc.py                # WAV → MFCC → .npy (manifest-aware)
│   ├── Training/
│   │   ├── train_config.py / ds_cnn_model.py / train.py / evaluate.py / quantize.py
│   ├── Testing/
│   │   └── pc_wakeword_test.py            # Real-time PC mic test
│   ├── export_norm_stats.py               # norm_stats.npz → C header
│   ├── dataset/
│   │   ├── dataset_manifest.csv           # All WAV paths + labels + weights
│   │   ├── wakeword/ · negative/ · debug/
│   ├── features/                          # X.npy · y.npy · weights.npy · norm_stats.npz
│   └── models/                            # .keras · .tflite
│
└── STM32/EmbeddedAI/
    └── Core/
        ├── Src/  main.c · mfcc_processing.c · wakeword_inference.c · network.c · network_data.c
        └── Inc/  mfcc_processing.h · wakeword_inference.h · norm_stats.h · network.h
```

---

## Getting Started

```bash
pip install tensorflow librosa numpy scikit-learn matplotlib
pip install edge-tts pydub audiomentations soundfile tqdm
pip install sounddevice scipy pyserial
# ffmpeg required: https://ffmpeg.org/download.html
```

STM32CubeIDE + ST Edge AI Core (X-CUBE-AI v4+) required for firmware.

```bash
# 1. Record real samples (optional but recommended)
python DatasetGeneration/record_dataset.py --port COM7 --mode wakeword --count 30
python DatasetGeneration/record_dataset.py --port COM7 --mode negative --count 30

# 2. Generate TTS dataset
python DatasetGeneration/generate_wakeword_dataset.py

# 3. Extract features
python Preprocessing/extract_mfcc.py

# 4. Train
python Training/train.py && python Training/evaluate.py

# 5. Quantize
python Training/quantize.py

# 6. Test on PC mic
python Testing/pc_wakeword_test.py

# 7. Deploy to STM32
#    Import gragas_dscnn_int8.tflite into ST Edge AI Core Studio
#    Generate C → copy to STM32/EmbeddedAI/Core/Src/ → build → flash
#    Say "Gragas" → LED on
```

---

## License

Apache 2.0 — see [LICENSE](./LICENSE)
