/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.h
 * @brief          : Header for main.c file.
 *                   Wake word detection on STM32H743ZI2 with INMP441 microphone.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h7xx_hal.h"

/* ── Board peripherals ─────────────────────────────────────────────────────── */

/* User LED (LD1 – green) on Nucleo-H743ZI2 */
#define LED_GREEN_PIN   GPIO_PIN_0
#define LED_GREEN_PORT  GPIOB

/* User LED (LD3 – red) on Nucleo-H743ZI2 */
#define LED_RED_PIN     GPIO_PIN_14
#define LED_RED_PORT    GPIOB

/* User button (B1 – blue) on Nucleo-H743ZI2 */
#define USER_BUTTON_PIN  GPIO_PIN_13
#define USER_BUTTON_PORT GPIOC

/* ── I2S / INMP441 ─────────────────────────────────────────────────────────── */

/*
 * INMP441 wiring to Nucleo-H743ZI2 (Arduino/Morpho headers):
 *
 *   INMP441   STM32H743
 *   --------  ---------
 *   VDD       3.3 V
 *   GND       GND
 *   WS        PB12  (I2S2_WS  / CN10-pin-16)
 *   SCK       PB13  (I2S2_CK  / CN10-pin-30)
 *   SD        PB15  (I2S2_SDI / CN10-pin-26)
 *   L/R       GND   (selects left channel)
 */
#define I2S_HANDLE      hi2s2          /* defined in main.c */

/* ── Audio parameters ──────────────────────────────────────────────────────── */

#define AUDIO_SAMPLE_RATE_HZ    16000U  /* 16 kHz */
#define AUDIO_FRAME_MS          32U     /* one inference frame in ms   */
#define AUDIO_HOP_MS            16U     /* frame hop (50 % overlap)    */

/* Number of 16-bit samples in one frame and one hop */
#define AUDIO_FRAME_SAMPLES     (AUDIO_SAMPLE_RATE_HZ * AUDIO_FRAME_MS  / 1000U)
#define AUDIO_HOP_SAMPLES       (AUDIO_SAMPLE_RATE_HZ * AUDIO_HOP_MS   / 1000U)

/*
 * The I2S peripheral delivers 32-bit words (left channel only).
 * DMA double-buffer: each half = HOP_SAMPLES × 32-bit words.
 */
#define DMA_BUFFER_WORDS        (AUDIO_HOP_SAMPLES * 2U)   /* full buffer  */
#define DMA_HALF_WORDS          (AUDIO_HOP_SAMPLES)        /* half buffer  */

/* ── Wake word detection ───────────────────────────────────────────────────── */

/*
 * Detection threshold in the range [0, 255] (uint8 softmax output).
 * Tune according to your trained model's performance on validation data.
 */
#define WAKEWORD_THRESHOLD      200U

/*
 * LED blink duration after a detected wake word (in ms).
 */
#define WAKEWORD_LED_DURATION_MS  500U

/* ── Error handler ─────────────────────────────────────────────────────────── */
void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
