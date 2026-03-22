/**
 ******************************************************************************
 * @file    mfcc_processing.c
 * @brief   MFCC computation — parameters mirrored from mfcc_config.py
 *
 * Python reference (mfcc_config.py):
 *   SAMPLE_RATE  = 16 000 Hz
 *   N_FFT        = 1024        (FFT_LEN here)
 *   WIN_LENGTH   = 640         (FRAME_LEN here, 40 ms)
 *   HOP_LENGTH   = 160         (HOP_LEN  here, 10 ms)
 *   N_MELS       = 40
 *   N_MFCC       = 40
 *   FMIN         = 20.0 Hz
 *   FMAX         = 8000.0 Hz
 *   WINDOW       = "hann"
 *   CENTER       = False
 *   PRE_EMPHASIS = 0.97
 *   NUM_FRAMES   = 97          (1 + (16000 - 640) / 160)
 *
 * Audio source: INMP441 via SAI1 DMA — see main.c
 *   Raw samples arrive as int32 (24-bit left-justified, right-shifted by 8).
 *   They are accumulated into a 1-second ring buffer, then processed here.
 ******************************************************************************
 */

#include "feature_extraction.h"
#include "main.h"
#include "norm_stats.h"

#include <string.h>
#include <stdio.h>   /* printf — routed through _write() → UART in main.c */

/* ── Debug counters ───────────────────────────────────────────────────────── */
static uint32_t s_ingest_call_count  = 0U;   /* total calls to IngestBlock   */
static uint32_t s_compute_call_count = 0U;   /* total calls to MFCC_Compute  */

/* ── Energy gate ─────────────────────────────────────────────────────────── *
 *
 * Problem: the model confidently misclassifies sustained ambient noise
 * (e.g. PC fans) as the wakeword because the training data contains only
 * synthetic silence / Gaussian noise — not real-world broadband noise.
 *
 * Fix: track a slowly-adapting noise floor (exponential moving average of
 * RMS).  Only proceed to MFCC frame computation + inference when the
 * current window's RMS exceeds the noise floor by ENERGY_GATE_FACTOR.
 * A spoken word causes a distinct RMS spike above the flat noise floor.
 *
 * Tuning:
 *   ENERGY_GATE_ALPHA  — lower = slower adaptation (more stable floor)
 *   ENERGY_GATE_FACTOR — higher = less sensitive (fewer false positives)
 * ──────────────────────────────────────────────────────────────────────── */
#define ENERGY_GATE_ALPHA   0.02f    /* EMA smoothing: floor ← α·rms + (1-α)·floor */
#define ENERGY_GATE_FACTOR  3.0f     /* require rms > factor × noise_floor           */
#define ENERGY_GATE_INIT    0.0f     /* initial floor (0 = learn from first window)  */
#define ENERGY_MIN_RMS  0.008f    // below this, skip inference regardless of gate factor

static float32_t s_noise_floor    = ENERGY_GATE_INIT;  /* running EMA of RMS      */
static uint8_t   s_floor_primed   = 0U;                /* 0 until first RMS seen  */

volatile uint8_t g_energy_gate_passed = 0U;             /* 1 = speech likely       */

/* ── Parameters (must stay in sync with mfcc_config.py) ──────────────────── */

#define SAMPLE_RATE      16000U   /* Hz — matches INMP441 / SAI config        */
#define FFT_LEN           1024U   /* N_FFT   : must be power-of-2, ≥ FRAME_LEN */
#define FRAME_LEN          640U   /* WIN_LENGTH : 40 ms at 16 kHz             */
#define HOP_LEN            160U   /* HOP_LENGTH : 10 ms at 16 kHz             */
#define NUM_MELS            40U   /* N_MELS                                   */
#define NUM_MFCC            40U   /* N_MFCC                                   */
#define FMIN              20.0f   /* Hz — lower Mel filter edge               */
#define FMAX            8000.0f   /* Hz — upper Mel filter edge (= Nyquist)   */
#define PRE_EMPHASIS      0.97f   /* Pre-emphasis coefficient                 */

/* Derived — matches Python: 1 + (16000 - 640) / 160 = 97 */
#define NUM_FRAMES        (1U + (SAMPLE_RATE - FRAME_LEN) / HOP_LEN)  /* 97  */

/*
 * NUM_MEL_COEFS: total Mel filter weight count returned by MelFilterbank_Init.
 * Build once with a large sentinel, read S_MelFilter.CoefficientsLength,
 * then set this define to match. The assertion below will catch any mismatch.
 */
#define NUM_MEL_COEFS     979U   /* Adjust if the while(1) trap fires         */

/* INMP441 full-scale: 24-bit signed → 2^23 */
#define INMP441_SCALE    (1.0f / 8388608.0f)

/* ── Sliding-window ring buffer ───────────────────────────────────────────
 *
 * Audio is written continuously into a circular buffer of AUDIO_BUF_LEN
 * (16 000) samples.  Inference is triggered every SLIDE_HOP_SAMPLES new
 * samples once the buffer is full — giving overlapping 1-second windows:
 *
 *   HOP = 4 000 samples = 250 ms  →  4 windows/second
 *
 * With a 500 ms word, at least 2 consecutive windows contain it fully.
 *
 *   window 1 : t=0.00 … 1.00 s
 *   window 2 : t=0.25 … 1.25 s   ← "gragas" fully inside → detected
 *   window 3 : t=0.50 … 1.50 s   ← also detected → cooldown suppresses
 */
#define AUDIO_BUF_LEN        SAMPLE_RATE   /* 16 000 samples = 1 second      */
#define SLIDE_HOP_SAMPLES    4000U         /* 250 ms hop — 4 windows/second  */

static int32_t   s_ring_buf[AUDIO_BUF_LEN]; /* circular audio buffer         */
static uint32_t  s_ring_write_idx      = 0U; /* next write slot (mod BUF_LEN) */
static uint32_t  s_ring_fill_count     = 0U; /* capped at AUDIO_BUF_LEN       */
static uint32_t  s_samples_since_infer = 0U; /* reset in MFCC_Compute()       */

volatile uint8_t g_mfcc_ready = 0U;          /* set when new window is ready  */

/* ── CMSIS-DSP / feature-extraction handles ──────────────────────────────── */

static arm_rfft_fast_instance_f32 S_Rfft;
static MelFilterTypeDef           S_MelFilter;
static DCT_InstanceTypeDef        S_DCT;
static SpectrogramTypeDef         S_Spectr;
static MelSpectrogramTypeDef      S_MelSpectr;
static LogMelSpectrogramTypeDef   S_LogMelSpectr;
static MfccTypeDef                S_Mfcc;

/* ── Working buffers ─────────────────────────────────────────────────────── */

static float32_t pInFrame[FFT_LEN];                  /* MfccColumn zero-pads
                                                        * in-place up to FFT_LEN
                                                        * — must NOT be FRAME_LEN */
static float32_t pOutColBuffer[NUM_MFCC];              /* one MFCC column     */
static float32_t pWindowFuncBuffer[FRAME_LEN];         /* Hann window         */
static float32_t pSpectrScratchBuffer[FFT_LEN];        /* FFT scratch         */
static float32_t pDCTCoefsBuffer[NUM_MELS * NUM_MFCC]; /* DCT matrix         */
static float32_t pMfccScratchBuffer[NUM_MELS];         /* MFCC scratch        */
static float32_t pMelFilterCoefs[NUM_MEL_COEFS];       /* Mel weights         */
static uint32_t  pMelFilterStartIndices[NUM_MELS];
static uint32_t  pMelFilterStopIndices[NUM_MELS];

/* Final MFCC output — shape (NUM_MFCC, NUM_FRAMES) = (40, 97) */
float32_t g_mfcc_out[NUM_MFCC * NUM_FRAMES];

/* Float copy of the 1-second window extracted from the ring buffer */
static float32_t s_float_buf[AUDIO_BUF_LEN];

static uint8_t s_warmup_done = 0U;

/* ── Pre-emphasis ────────────────────────────────────────────────────────── */

/**
 * @brief  Apply pre-emphasis filter to a float audio buffer in-place.
 *         y[n] = x[n] - PRE_EMPHASIS * x[n-1]
 *         Matches Python: apply_preemphasis() in extract_mfcc.py
 *
 * @param  buf    float32 audio signal
 * @param  len    number of samples
 */
static void apply_preemphasis(float32_t *buf, uint32_t len)
{
    /* Process backwards to avoid needing a separate copy */
    for (uint32_t n = len - 1U; n > 0U; n--)
    {
        buf[n] = buf[n] - PRE_EMPHASIS * buf[n - 1U];
    }
    /* buf[0] is left unchanged — same behaviour as np.append(audio[0], ...) */
}

/* ── Initialisation ──────────────────────────────────────────────────────── */

/**
 * @brief  Initialise all CMSIS-DSP feature-extraction structures.
 *         Call once after SystemClock_Config() and peripheral inits.
 */
void MFCC_Init(void)
{
    /* ── Window function ──────────────────────────────────────────────────── */
    if (Window_Init(pWindowFuncBuffer, FRAME_LEN, WINDOW_HANN) != 0)
    {
        while (1); /* Window init failed */
    }

    /* ── RFFT (FFT_LEN = 1024) ────────────────────────────────────────────── */
    arm_rfft_fast_init_f32(&S_Rfft, FFT_LEN);

    /* ── Mel filterbank ───────────────────────────────────────────────────── */
    S_MelFilter.pStartIndices   = pMelFilterStartIndices;
    S_MelFilter.pStopIndices    = pMelFilterStopIndices;
    S_MelFilter.pCoefficients   = pMelFilterCoefs;
    S_MelFilter.NumMels         = NUM_MELS;
    S_MelFilter.FFTLen          = FFT_LEN;
    S_MelFilter.SampRate        = SAMPLE_RATE;
    S_MelFilter.FMin            = FMIN;    /* 20.0 Hz — matches Python FMIN   */
    S_MelFilter.FMax            = FMAX;    /* 8000.0 Hz — matches Python FMAX */
    S_MelFilter.Formula         = MEL_SLANEY;
    S_MelFilter.Normalize       = 1;
    S_MelFilter.Mel2F           = 1;
    MelFilterbank_Init(&S_MelFilter);

    /* Guard: adjust NUM_MEL_COEFS if this fires */
    if (S_MelFilter.CoefficientsLength != NUM_MEL_COEFS)
    {
        while (1); /* ← Set NUM_MEL_COEFS = S_MelFilter.CoefficientsLength */
    }

    /* ── DCT-II ortho (matches Python dct_type=2, norm='ortho') ─────────── */
    S_DCT.NumFilters    = NUM_MFCC;
    S_DCT.NumInputs     = NUM_MELS;
    S_DCT.Type          = DCT_TYPE_II_ORTHO;
    S_DCT.RemoveDCTZero = 0;
    S_DCT.pDCTCoefs     = pDCTCoefsBuffer;
    if (DCT_Init(&S_DCT) != 0)
    {
        while (1); /* DCT init failed */
    }

    /* ── Power spectrogram ────────────────────────────────────────────────── */
    S_Spectr.pRfft    = &S_Rfft;
    S_Spectr.Type     = SPECTRUM_TYPE_POWER;
    S_Spectr.pWindow  = pWindowFuncBuffer;
    S_Spectr.SampRate = SAMPLE_RATE;
    S_Spectr.FrameLen = FRAME_LEN;   /* 640 — zero-padded to FFT_LEN (1024)  */
    S_Spectr.FFTLen   = FFT_LEN;
    S_Spectr.pScratch = pSpectrScratchBuffer;

    /* ── Mel spectrogram ─────────────────────────────────────────────────── */
    S_MelSpectr.SpectrogramConf = &S_Spectr;
    S_MelSpectr.MelFilter       = &S_MelFilter;

    /* ── Log-Mel spectrogram (dB scale, matches librosa default) ─────────── */
    S_LogMelSpectr.MelSpectrogramConf = &S_MelSpectr;
    S_LogMelSpectr.LogFormula         = LOGMELSPECTROGRAM_SCALE_DB;
    S_LogMelSpectr.Ref                = 1.0f;
    S_LogMelSpectr.TopdB              = HUGE_VALF;

    /* ── MFCC ─────────────────────────────────────────────────────────────── */
    S_Mfcc.LogMelConf   = &S_LogMelSpectr;
    S_Mfcc.pDCT         = &S_DCT;
    S_Mfcc.NumMfccCoefs = NUM_MFCC;
    S_Mfcc.pScratch     = pMfccScratchBuffer;
}

/* ── DMA audio ingestion (called from main.c) ────────────────────────────── */

/**
 * @brief  Feed one DMA half-buffer of raw INMP441 samples into the accumulator.
 *
 *         Call this from main.c wherever you currently call
 *         HAL_UART_Transmit(), passing the same `src` pointer and frame count.
 *
 * @param  src         Pointer to the start of the DMA half (interleaved L/R)
 * @param  num_frames  AUDIO_BLOCK_FRAMES (256)
 */
void MFCC_IngestBlock(const int32_t *src, uint32_t num_frames)
{
    s_ingest_call_count++;

    /* Invalidate cache for this DMA half-buffer BEFORE reading it.
     * Size = num_frames stereo pairs × 2 channels × 4 bytes each.
     * Address must be 32-byte aligned for SCB_InvalidateDCache_by_Addr. */
    SCB_InvalidateDCache_by_Addr(
        (uint32_t *)src,
        num_frames * 2 * sizeof(int32_t)
    );

    /* ── CHECKPOINT 1: print every 64 calls (~1 s) ──────────────────────── */
    if ((s_ingest_call_count & 0x3F) == 1)
    {
//        printf("[MFCC] IngestBlock call #%lu  warmup=%d  "
//               "fill=%lu  since_infer=%lu\r\n",
//               (unsigned long)s_ingest_call_count,
//               (int)s_warmup_done,
//               (unsigned long)s_ring_fill_count,
//               (unsigned long)s_samples_since_infer);
    }

    /* ── Warmup: discard first 1-second window (DMA startup transient) ─── */
    if (!s_warmup_done)
    {
        s_ring_fill_count += num_frames;
        if (s_ring_fill_count >= AUDIO_BUF_LEN)
        {
            s_ring_fill_count = 0U;
            s_warmup_done     = 1U;
//            printf("[MFCC] Warmup done after %lu calls — "
//                   "ring buffer filling starts now\r\n",
//                   (unsigned long)s_ingest_call_count);
        }
        return;
    }

    /* ── Write into ring buffer continuously ─────────────────────────────── */
    for (uint32_t i = 0U; i < num_frames; i++)
    {
        int32_t raw    = src[i * 2U];   /* left channel */
        int32_t s24    = raw >> 8;
        int32_t gained = s24 * 10;      /* target RMS 0.10-0.15 when speaking
                                           increase if RMS < 0.08 when speaking
                                           decrease if RMS > 0.20 (clipping)  */
        if (gained >  8388607)  gained =  8388607;
        if (gained < -8388608)  gained = -8388608;

        s_ring_buf[s_ring_write_idx] = gained;
        s_ring_write_idx = (s_ring_write_idx + 1U) % AUDIO_BUF_LEN;
    }

    /* Track fill level (capped at one full window) */
    if (s_ring_fill_count < AUDIO_BUF_LEN)
    {
        s_ring_fill_count += num_frames;
        if (s_ring_fill_count > AUDIO_BUF_LEN)
            s_ring_fill_count = AUDIO_BUF_LEN;
    }

    s_samples_since_infer += num_frames;

    /* ── Trigger when: buffer full + enough new samples + previous consumed  */
    if (s_ring_fill_count  >= AUDIO_BUF_LEN      &&
        s_samples_since_infer >= SLIDE_HOP_SAMPLES &&
        g_mfcc_ready == 0U)
    {
        /* ── CHECKPOINT 3 ──────────────────────────────────────────────── */
//        printf("[MFCC] Window ready (hop=%lu samples)  "
//               "(ingest calls=%lu)\r\n",
//               (unsigned long)s_samples_since_infer,
//               (unsigned long)s_ingest_call_count);
        g_mfcc_ready = 1U;
    }
}

/* ── MFCC computation ────────────────────────────────────────────────────── */

/**
 * @brief  Convert the accumulated int32 audio buffer to normalised float32
 *         and apply pre-emphasis.
 *         Matches Python: load_wav → pad_or_trim → apply_preemphasis
 */
static void prepare_float_buf(void)
{
    /* s_ring_write_idx is the NEXT write slot = OLDEST sample.
     * Reading AUDIO_BUF_LEN samples from there gives chronological order.  */
    for (uint32_t n = 0U; n < AUDIO_BUF_LEN; n++)
    {
        uint32_t idx    = (s_ring_write_idx + n) % AUDIO_BUF_LEN;
        s_float_buf[n]  = (float32_t)s_ring_buf[idx] * INMP441_SCALE;
    }
    apply_preemphasis(s_float_buf, AUDIO_BUF_LEN);
}

/**
 * @brief  Run full MFCC pipeline on the 1-second float buffer.
 *
 *         Output is written to g_mfcc_out[coef * NUM_FRAMES + frame],
 *         i.e. shape (NUM_MFCC=40, NUM_FRAMES=97) — row-major, coef-first.
 *         Transpose to (NUM_FRAMES, NUM_MFCC) before feeding the CNN if your
 *         model expects time-first layout (as in extract_mfcc.py).
 *
 *         Call from the main loop when g_mfcc_ready == 1.
 */
void MFCC_Compute(void)
{
    s_compute_call_count++;
    /* ── CHECKPOINT 4: MFCC_Compute is running ──────────────────────────── */
    printf("[MFCC] Compute #%lu — processing 1-second window\r\n",
           (unsigned long)s_compute_call_count);

    prepare_float_buf();

    /* ── CHECKPOINT 4b: audio RMS — is there any signal? ────────────────── */
    float32_t rms_sum = 0.0f;
    for (uint32_t n = 0U; n < AUDIO_BUF_LEN; n++)
        rms_sum += s_float_buf[n] * s_float_buf[n];
    float32_t rms = 0.0f;
    arm_sqrt_f32(rms_sum / (float32_t)AUDIO_BUF_LEN, &rms);
    printf("[MFCC] Audio RMS after preemph = %.5f  "
           "(expect >0.01 when speaking)\r\n", rms);

    /* ── Energy gate: update noise floor and check for speech ────────────── */
    if (!s_floor_primed && s_ring_fill_count >= AUDIO_BUF_LEN)
    {
        s_noise_floor  = rms;
        s_floor_primed = 1U;
    }

    float32_t gate_threshold = s_noise_floor * ENERGY_GATE_FACTOR;
    uint8_t   gate_open = (rms > gate_threshold) && (rms > ENERGY_MIN_RMS);

    printf("[GATE] noise_floor=%.5f  threshold=%.5f  rms=%.5f  %s\r\n",
           s_noise_floor, gate_threshold, rms,
           gate_open ? "OPEN (speech?)" : "CLOSED (ambient)");

    /* Update the noise floor — only from quiet windows to avoid
     * speech pushing the floor up.  If the gate is open (speech),
     * we freeze the floor so it stays calibrated to the ambient level. */
    if (!gate_open)
    {
        s_noise_floor = ENERGY_GATE_ALPHA * rms
                      + (1.0f - ENERGY_GATE_ALPHA) * s_noise_floor;
    }

    if (!gate_open)
    {
        /* No speech detected — skip the expensive MFCC frame loop and
         * signal main.c to skip inference for this window.              */
        g_energy_gate_passed = 0U;
        s_samples_since_infer = 0U;
        g_mfcc_ready          = 0U;
        return;
    }

    g_energy_gate_passed = 1U;

    for (uint32_t frame_idx = 0U; frame_idx < NUM_FRAMES; frame_idx++)
    {
        const float32_t *frame_start = &s_float_buf[HOP_LEN * frame_idx];
        memcpy(pInFrame, frame_start, FRAME_LEN * sizeof(float32_t));
        /* Explicitly zero the padding region [FRAME_LEN..FFT_LEN-1].
         * MfccColumn pads in-place — without this, stale values from the
         * previous frame would leak into the FFT.                          */
        memset(&pInFrame[FRAME_LEN], 0,
               (FFT_LEN - FRAME_LEN) * sizeof(float32_t));

        MfccColumn(&S_Mfcc, pInFrame, pOutColBuffer);

        // Match librosa's power_to_db amin floor: 10*log10(1e-10) = -100 dB
        // MFCC coef[0] for all-floor Mel = sqrt(1/N_MELS) * (-100 * N_MELS) ≈ -632
        // A practical per-coef floor that covers this:
        for (uint32_t coef = 0U; coef < NUM_MFCC; coef++)
        {
            if (pOutColBuffer[coef] < -320.0f)
                pOutColBuffer[coef] = -320.0f;   // matches librosa's effective MFCC floor
            g_mfcc_out[coef * NUM_FRAMES + frame_idx] = pOutColBuffer[coef];
        }

    }

    /* ── CHECKPOINT 4c: first 5 raw MFCC values (coef 0, frames 0-4) ───── *
     *
     * Compare against Python: print(mfcc_raw[0:5, 0]) on a saved .npy      */

    // Full coef[0] dump — compare line-by-line to Python output below
//    printf("[DUMP_C0]");
//    for (uint32_t f = 0; f < NUM_FRAMES; f++)
//        printf(" %.1f", g_mfcc_out[0 * NUM_FRAMES + f]);
//    printf("\r\n");
//
//    // All 40 coefs at frame 48 (middle of window — most likely to contain speech)
//    printf("[DUMP_F48]");
//    for (uint32_t c = 0; c < NUM_MFCC; c++)
//        printf(" %.2f", g_mfcc_out[c * NUM_FRAMES + 48]);
//    printf("\r\n");
//    printf("[MFCC] Expected from Python norm_stats: mean[0]=%.2f  std[0]=%.2f\r\n",
//           mfcc_mean[0], mfcc_std[0]);

    /* Reset only the hop counter — ring buffer write index is NEVER reset   */
    s_samples_since_infer = 0U;
    g_mfcc_ready          = 0U;
}

/* ── Usage example (add to main.c while(1) loop) ─────────────────────────
 *
 *  // In the existing audio_block_ready block, after extracting the left
 *  // channel, add:
 *
 *  MFCC_IngestBlock(src, AUDIO_BLOCK_FRAMES);
 *
 *  // Separately, in the main while(1) loop:
 *
 *  if (g_mfcc_ready)
 *  {
 *      MFCC_Compute();
 *      // g_mfcc_out now holds (40 × 97) MFCCs — feed to your DS-CNN
 *  }
 *
 * ──────────────────────────────────────────────────────────────────────── */
