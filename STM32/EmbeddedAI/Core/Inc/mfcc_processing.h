/**
 ******************************************************************************
 * @file    mfcc_processing.h
 * @brief   Public interface for the MFCC feature-extraction pipeline.
 *
 * Parameters are defined in mfcc_processing.c and mirror mfcc_config.py.
 * Include this header in main.c — do not include mfcc_processing.c directly.
 *
 * Typical usage in main.c
 * ------------------------
 *
 *   #include "mfcc_processing.h"
 *
 *   // 1. Once, after all peripheral inits:
 *   MFCC_Init();
 *
 *   // 2. Inside the audio_block_ready block (after extracting src):
 *   MFCC_IngestBlock(src, AUDIO_BLOCK_FRAMES);
 *
 *   // 3. In the main while(1) loop:
 *   if (g_mfcc_ready)
 *   {
 *       MFCC_Compute();
 *       // g_mfcc_out[coef * MFCC_NUM_FRAMES + frame] is now valid
 *       // Shape: (MFCC_NUM_COEFS=40, MFCC_NUM_FRAMES=97) — coef-major
 *   }
 *
 ******************************************************************************
 */

#ifndef MFCC_PROCESSING_H
#define MFCC_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "arm_math.h"   /* float32_t */

/* ── Public constants (read-only — do not modify) ────────────────────────── */

/** Number of MFCC coefficients per frame. Mirrors N_MFCC in mfcc_config.py. */
#define MFCC_NUM_COEFS   40U

/**
 * Number of time frames in one 1-second window.
 * Mirrors NUM_FRAMES = 1 + (16000 - 640) / 160 = 97 in mfcc_config.py.
 */
#define MFCC_NUM_FRAMES  97U

/**
 * Total number of float32 values in g_mfcc_out.
 * Layout: g_mfcc_out[coef * MFCC_NUM_FRAMES + frame]
 */
#define MFCC_OUT_SIZE    (MFCC_NUM_COEFS * MFCC_NUM_FRAMES)   /* 3880 */

/* ── Shared state ────────────────────────────────────────────────────────── */

/**
 * Set to 1 by MFCC_IngestBlock() when a full 1-second window has been
 * accumulated.  Cleared to 0 at the end of MFCC_Compute().
 * Poll this flag in the main loop — do not write to it from application code.
 */
extern volatile uint8_t g_mfcc_ready;

/**
 * MFCC output buffer.
 * Valid only after g_mfcc_ready was 1 and MFCC_Compute() has returned.
 *
 * Layout  : coef-major, i.e. g_mfcc_out[coef * MFCC_NUM_FRAMES + frame]
 * Shape   : (MFCC_NUM_COEFS=40, MFCC_NUM_FRAMES=97)
 *
 * To obtain time-first layout (NUM_FRAMES, NUM_COEFS) expected by the DS-CNN,
 * access as: g_mfcc_out[frame * MFCC_NUM_COEFS + coef]  — or transpose before
 * feeding the inference engine.
 */
extern float32_t g_mfcc_out[MFCC_OUT_SIZE];

/* ── Public API ──────────────────────────────────────────────────────────── */

/**
 * @brief  Initialise CMSIS-DSP structures (window, FFT, Mel, DCT, MFCC).
 *         Must be called once before MFCC_IngestBlock() or MFCC_Compute().
 *         Call after SystemClock_Config() and all HAL peripheral inits.
 */
void MFCC_Init(void);

/**
 * @brief  Feed one DMA half-transfer of raw INMP441 samples into the
 *         internal 1-second accumulation buffer.
 *
 *         Call this inside the audio_block_ready handling block in main.c,
 *         passing the same `src` pointer used for UART transmission.
 *         When AUDIO_BUF_LEN (16 000) samples have been collected,
 *         g_mfcc_ready is set to 1 and further incoming samples are dropped
 *         until MFCC_Compute() resets the accumulator.
 *
 * @param  src         Pointer to the start of a DMA half-buffer.
 *                     Interleaved stereo (L at word[i*2], R at word[i*2+1]).
 * @param  num_frames  Number of stereo frames in this half — pass
 *                     AUDIO_BLOCK_FRAMES (256) as defined in main.c.
 */
void MFCC_IngestBlock(const int32_t *src, uint32_t num_frames);

/**
 * @brief  Run the full MFCC pipeline on the accumulated 1-second buffer.
 *
 *         Applies float conversion, pre-emphasis (α=0.97), then computes
 *         MFCC_NUM_FRAMES columns of MFCC_NUM_COEFS coefficients each.
 *         Results are written to g_mfcc_out.  The accumulator and
 *         g_mfcc_ready are reset at the end so ingestion can resume.
 *
 *         Only call when g_mfcc_ready == 1.
 */
void MFCC_Compute(void);

#ifdef __cplusplus
}
#endif

#endif /* MFCC_PROCESSING_H */
