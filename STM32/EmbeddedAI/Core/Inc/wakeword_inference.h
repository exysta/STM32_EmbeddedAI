/**
 ******************************************************************************
 * @file    wakeword_inference.h
 * @brief   Public interface for the MFCC → quantise → ST-AI glue layer.
 *
 * Include this header in main.c.  The full pipeline is:
 *
 *   Startup
 *   ───────
 *   STM32CubeAI_Studio_AI_Init();   // ST-AI runtime + model init
 *   MFCC_Init();                    // CMSIS-DSP pipeline init
 *
 *   Main loop
 *   ─────────
 *   // Inside audio_block_ready block:
 *   MFCC_IngestBlock(src, AUDIO_BLOCK_FRAMES);
 *
 *   // In the while(1) body:
 *   if (g_mfcc_ready)
 *   {
 *       MFCC_Compute();             // fills g_mfcc_out (float32)
 *
 *       float p_ww;
 *       int   detected = WW_RunInference(&p_ww);
 *
 *       if (detected)
 *           HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);  // or your action
 *   }
 ******************************************************************************
 */

#ifndef WAKEWORD_INFERENCE_H
#define WAKEWORD_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * @brief  Quantise g_mfcc_out (float32 → int8) into the internal model
 *         input buffer using per-tensor affine quantisation.
 *
 *         INPUT_SCALE and INPUT_ZP must match the values in your
 *         Cube.AI quantisation report (network.h / network_config.h).
 *
 *         This is called automatically by WW_RunInference() — you only
 *         need to call it directly if you want to inspect the INT8 buffer
 *         before running the model.
 */
void WW_Quantise(void);

/**
 * @brief  Run a full wakeword detection inference:
 *           1. Quantises g_mfcc_out → INT8
 *           2. Calls aiRunInference() (ST-AI synchronous forward pass)
 *           3. Compares P(wakeword) against WW_THRESHOLD
 *
 *         Only call this after g_mfcc_ready == 1 and MFCC_Compute() has
 *         returned.
 *
 * @param  p_wakeword  Pointer to receive P(wakeword) ∈ [0, 1].
 *                     May be NULL if the raw probability is not needed.
 *                     Set to -1.0f if the inference call failed.
 *
 * @return 1  — wakeword detected  (P ≥ WW_THRESHOLD)
 *         0  — not detected or inference error
 */
int WW_RunInference(float *p_wakeword);

#ifdef __cplusplus
}
#endif

#endif /* WAKEWORD_INFERENCE_H */
