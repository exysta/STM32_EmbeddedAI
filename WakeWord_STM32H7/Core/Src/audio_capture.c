/**
 ******************************************************************************
 * @file    audio_capture.c
 * @brief   I2S DMA audio capture from INMP441 microphone.
 *
 * The INMP441 sends 24-bit audio data in 32-bit I2S frames at the configured
 * sample rate.  Only the left channel is used (L/R pin tied to GND).
 *
 * DMA double-buffer strategy
 * ──────────────────────────
 *   dma_buf[0 .. DMA_BUFFER_WORDS-1]   full circular DMA buffer (32-bit words)
 *   ├── First  half [0 .. DMA_HALF_WORDS-1]   → triggers HalfCplt ISR
 *   └── Second half [DMA_HALF_WORDS .. end]   → triggers Cplt    ISR
 *
 * On each ISR the 32-bit words are converted to 16-bit PCM samples (upper
 * 16 bits only) and written into a circular ring buffer sized to hold a full
 * inference window (NUM_MFCC_FRAMES hops + overlap).
 ******************************************************************************
 */

#include "audio_capture.h"
#include <string.h>

/* ── Private state ─────────────────────────────────────────────────────────── */

/** Raw 32-bit DMA receive buffer (I2S words, left-justified). */
static uint32_t dma_buf[DMA_BUFFER_WORDS];

/**
 * Ring buffer holding 16-bit PCM samples.
 * Size: enough for one full inference window (NUM_MFCC_FRAMES hops + FFT
 * overlap).  We keep it simple: 2 × AUDIO_FRAME_SAMPLES is more than enough.
 */
#define RING_BUF_SIZE   (2U * AUDIO_FRAME_SAMPLES)
static int16_t  ring_buf[RING_BUF_SIZE];
static uint32_t ring_write_idx = 0;   /* next position to write in ring_buf */
static uint32_t ring_read_idx  = 0;   /* consumer read position             */
static uint32_t samples_available = 0; /* unread samples in ring buffer      */

/** Pointer to the I2S handle set by AudioCapture_Init(). */
static I2S_HandleTypeDef *s_hi2s = NULL;

/** Flag set by the capture interrupt; cleared by AudioCapture_GetFrame(). */
static volatile uint8_t new_hop_available = 0;
static volatile uint8_t overrun_detected  = 0;

/* ── Private helpers ───────────────────────────────────────────────────────── */

/**
 * @brief  Convert DMA_HALF_WORDS I2S 32-bit words to 16-bit PCM and push into
 *         the ring buffer.  The INMP441 places the 24-bit sample MSB-first in
 *         bits [31:8]; we keep the upper 16 bits (bits [31:16]).
 */
static void push_dma_half(const uint32_t *src)
{
    for (uint32_t i = 0; i < DMA_HALF_WORDS; i++) {
        int16_t sample = (int16_t)((src[i] >> 16) & 0xFFFFU);
        ring_buf[ring_write_idx] = sample;
        ring_write_idx = (ring_write_idx + 1U) % RING_BUF_SIZE;
        if (samples_available < RING_BUF_SIZE) {
            samples_available++;
        } else {
            /* Overrun: consumer is too slow; advance read pointer. */
            ring_read_idx = (ring_read_idx + 1U) % RING_BUF_SIZE;
            overrun_detected = 1;
        }
    }
    new_hop_available = 1;
}

/* ── Public API ────────────────────────────────────────────────────────────── */

void AudioCapture_Init(I2S_HandleTypeDef *hi2s)
{
    s_hi2s = hi2s;
    memset(dma_buf,  0, sizeof(dma_buf));
    memset(ring_buf, 0, sizeof(ring_buf));
    ring_write_idx   = 0;
    ring_read_idx    = 0;
    samples_available = 0;
    new_hop_available = 0;
    overrun_detected  = 0;
}

void AudioCapture_Start(void)
{
    /*
     * Start circular DMA reception.
     * HAL expects the buffer length in number of half-words (16-bit units).
     * Each DMA_BUFFER_WORDS 32-bit word = 2 half-words.
     */
    HAL_I2S_Receive_DMA(s_hi2s,
                        (uint16_t *)dma_buf,
                        (uint16_t)(DMA_BUFFER_WORDS * 2U));
}

void AudioCapture_Stop(void)
{
    HAL_I2S_DMAStop(s_hi2s);
}

int AudioCapture_GetHop(int16_t *out_buf)
{
    if (s_hi2s == NULL) {
        return -1;
    }

    /* Spin-wait for at least one new hop since last call. */
    while (!new_hop_available) {
        __NOP();
    }
    new_hop_available = 0;

    if (overrun_detected) {
        overrun_detected = 0;
        return -1;
    }

    /*
     * Copy the most-recent AUDIO_HOP_SAMPLES from the ring buffer into
     * out_buf, in chronological order (oldest first).
     */
    uint32_t copy_len = (samples_available >= DMA_HALF_WORDS)
                        ? DMA_HALF_WORDS
                        : samples_available;

    uint32_t start = (ring_write_idx + RING_BUF_SIZE - copy_len) % RING_BUF_SIZE;

    for (uint32_t i = 0; i < copy_len; i++) {
        out_buf[i] = ring_buf[(start + i) % RING_BUF_SIZE];
    }

    /* Zero-pad if not enough samples are available yet. */
    if (copy_len < DMA_HALF_WORDS) {
        for (uint32_t i = copy_len; i < DMA_HALF_WORDS; i++) {
            out_buf[i] = 0;
        }
    }

    return 0;
}

/* ── HAL callbacks ─────────────────────────────────────────────────────────── */

void AudioCapture_HalfCpltCallback(void)
{
    push_dma_half(&dma_buf[0]);
}

void AudioCapture_CpltCallback(void)
{
    push_dma_half(&dma_buf[DMA_HALF_WORDS]);
}

/* Weak HAL callbacks – forward to our module functions. */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == s_hi2s->Instance) {
        AudioCapture_HalfCpltCallback();
    }
}

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s->Instance == s_hi2s->Instance) {
        AudioCapture_CpltCallback();
    }
}
