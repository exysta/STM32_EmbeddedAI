/**
 ******************************************************************************
 * @file    audio_processing.h
 * @brief   MFCC feature extraction for wake word detection.
 *
 * Computes a fixed-size Mel-Frequency Cepstral Coefficient (MFCC) feature
 * matrix from a window of raw 16-bit PCM audio.  The output shape matches
 * the input expected by the Edge Impulse / X-CUBE-AI model:
 *
 *   (NUM_FRAMES × NUM_MFCC_COEFFS) float32 values.
 *
 * All DSP operations use ARM CMSIS-DSP for efficiency on Cortex-M7.
 ******************************************************************************
 */

#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "main.h"

/* ── Feature dimensions ────────────────────────────────────────────────────── */

/** Number of MFCC coefficients to keep (matches model input). */
#define NUM_MFCC_COEFFS     13U

/**
 * Number of short-time frames in one inference window.
 * With AUDIO_FRAME_SAMPLES = 512, FFT_SIZE = 512, HOP = 160:
 *   num_frames = 1 + (512 - 512) / 160 = 1  (single-frame window).
 * For a 1-second window at 16 kHz with hop=160:
 *   num_frames = 1 + (16000 - 512) / 160 ≈ 97
 */
#define NUM_MFCC_FRAMES     49U   /* ≈ 800 ms window, hop = 320 samples */

/** Total number of float32 features fed to the model. */
#define FEATURE_SIZE        (NUM_MFCC_FRAMES * NUM_MFCC_COEFFS)   /* 637 */

/* ── FFT / filter-bank parameters ─────────────────────────────────────────── */

#define FFT_SIZE            512U    /* must be a power of two       */
#define NUM_MEL_FILTERS     26U     /* mel filter bank channels     */
#define PRE_EMPHASIS_COEFF  0.97f   /* first-order pre-emphasis     */

/* ── Public API ────────────────────────────────────────────────────────────── */

/**
 * @brief  Initialize MFCC tables (mel filter bank, DCT matrix, Hann window).
 *         Must be called once before AudioProcessing_ComputeMFCC().
 */
void AudioProcessing_Init(void);

/**
 * @brief  Compute the MFCC feature matrix for one inference window.
 *
 * @param  pcm_buf   Pointer to AUDIO_FRAME_SAMPLES × NUM_MFCC_FRAMES raw
 *                   16-bit PCM samples (row-major, oldest sample first).
 *                   Length = NUM_MFCC_FRAMES * AUDIO_HOP_SAMPLES +
 *                            FFT_SIZE − AUDIO_HOP_SAMPLES
 * @param  features  Output buffer of FEATURE_SIZE float32 values.
 *                   Caller must allocate at least FEATURE_SIZE floats.
 */
void AudioProcessing_ComputeMFCC(const int16_t *pcm_buf, float *features);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PROCESSING_H */
