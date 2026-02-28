/**
 ******************************************************************************
 * @file    audio_capture.h
 * @brief   I2S DMA audio capture from INMP441 microphone.
 *
 * The INMP441 is an omnidirectional MEMS microphone with an I2S digital
 * output.  This driver configures I2S2 in master-receive mode with DMA
 * double-buffering so the CPU is only interrupted once every HOP_SAMPLES
 * worth of audio.
 *
 * Usage:
 *   1. Call AudioCapture_Init() once after HAL/I2S initialization.
 *   2. Call AudioCapture_Start() to begin DMA streaming.
 *   3. In the main loop, poll AudioCapture_GetFrame() to obtain a pointer
 *      to the most-recent AUDIO_FRAME_SAMPLES of 16-bit PCM data.
 *   4. Call AudioCapture_Stop() to halt DMA if needed.
 ******************************************************************************
 */

#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "main.h"

/* ── Public API ────────────────────────────────────────────────────────────── */

/**
 * @brief  Initialize the audio capture module.
 *         Must be called after MX_I2S2_Init().
 * @param  hi2s  Pointer to the I2S handle configured for INMP441.
 */
void AudioCapture_Init(I2S_HandleTypeDef *hi2s);

/**
 * @brief  Start continuous DMA capture.
 */
void AudioCapture_Start(void);

/**
 * @brief  Stop DMA capture.
 */
void AudioCapture_Stop(void);

/**
 * @brief  Obtain the latest AUDIO_HOP_SAMPLES of new 16-bit PCM audio.
 *
 * The function fills @p out_buf with exactly AUDIO_HOP_SAMPLES int16_t
 * samples captured since the previous call.  It blocks briefly (spin-waits)
 * until DMA has delivered a fresh hop since the last call.
 *
 * The caller is responsible for maintaining a longer sliding-window buffer;
 * see main.c for the typical usage pattern.
 *
 * @param  out_buf  Destination buffer (must hold AUDIO_HOP_SAMPLES int16_t).
 * @retval 0  Success.
 * @retval -1 Capture not started or overrun detected.
 */
int AudioCapture_GetHop(int16_t *out_buf);

/**
 * @brief  Called from HAL_I2S_RxHalfCpltCallback – do not call directly.
 */
void AudioCapture_HalfCpltCallback(void);

/**
 * @brief  Called from HAL_I2S_RxCpltCallback – do not call directly.
 */
void AudioCapture_CpltCallback(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_CAPTURE_H */
