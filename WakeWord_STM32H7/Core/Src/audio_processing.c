/**
 ******************************************************************************
 * @file    audio_processing.c
 * @brief   MFCC feature extraction using ARM CMSIS-DSP.
 *
 * Pipeline per short-time frame
 * ──────────────────────────────
 *   1. Pre-emphasis filter
 *   2. Hann window
 *   3. arm_rfft_fast_f32  (real FFT, FFT_SIZE points)
 *   4. Magnitude spectrum  |X[k]|
 *   5. Mel filter bank     (NUM_MEL_FILTERS triangular filters)
 *   6. log(energy + 1e-6)  (natural log, small floor for numerical stability)
 *   7. DCT-II              (NUM_MFCC_COEFFS outputs kept)
 *
 * The feature matrix layout is row-major:
 *   features[frame * NUM_MFCC_COEFFS + coeff]
 ******************************************************************************
 */

#include "audio_processing.h"

#include <math.h>
#include <string.h>

/* CMSIS-DSP */
#define ARM_MATH_CM7
#include "arm_math.h"

/* ── Private tables ────────────────────────────────────────────────────────── */

static float hann_window[FFT_SIZE];

/*
 * Mel filter bank: stored as a (NUM_MEL_FILTERS × (FFT_SIZE/2+1)) sparse
 * matrix.  Only the non-zero triangular weights are stored per filter.
 * For simplicity on MCU we precompute and store the full dense matrix.
 */
static float mel_fb[NUM_MEL_FILTERS][FFT_SIZE / 2 + 1];

/*
 * DCT-II matrix: (NUM_MFCC_COEFFS × NUM_MEL_FILTERS)
 * dct_matrix[c][m] = cos(π/N * (m + 0.5) * c)
 */
static float dct_matrix[NUM_MFCC_COEFFS][NUM_MEL_FILTERS];

/* ARM CMSIS-DSP FFT instance. */
static arm_rfft_fast_instance_f32 fft_instance;

/* Work buffers allocated statically to avoid heap use on MCU. */
static float fft_input[FFT_SIZE];
static float fft_output[FFT_SIZE];   /* complex, interleaved Re/Im pairs */
static float mag_spectrum[FFT_SIZE / 2 + 1];
static float mel_energies[NUM_MEL_FILTERS];

/* ── Private helpers ───────────────────────────────────────────────────────── */

/** Convert Hz to mel scale. */
static inline float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/** Convert mel to Hz. */
static inline float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * @brief Build the Hann window table.
 */
static void build_hann_window(void)
{
    for (uint32_t n = 0; n < FFT_SIZE; n++) {
        hann_window[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)n
                                              / (float)(FFT_SIZE - 1)));
    }
}

/**
 * @brief Build the triangular mel filter bank.
 */
static void build_mel_filterbank(void)
{
    const float f_min  = 0.0f;
    const float f_max  = (float)AUDIO_SAMPLE_RATE_HZ / 2.0f;  /* Nyquist */
    const uint32_t num_bins = FFT_SIZE / 2 + 1;

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    /* Evenly-spaced centre frequencies in mel space
     * (NUM_MEL_FILTERS filters → NUM_MEL_FILTERS + 2 points). */
    float mel_points[NUM_MEL_FILTERS + 2];
    for (uint32_t i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        mel_points[i] = mel_min + (float)i * (mel_max - mel_min)
                        / (float)(NUM_MEL_FILTERS + 1);
    }

    /* Convert mel points back to Hz and then to FFT bin indices. */
    float hz_points[NUM_MEL_FILTERS + 2];
    uint32_t bin_points[NUM_MEL_FILTERS + 2];
    for (uint32_t i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        hz_points[i]  = mel_to_hz(mel_points[i]);
        bin_points[i] = (uint32_t)floorf((float)(FFT_SIZE + 1)
                        * hz_points[i] / (float)AUDIO_SAMPLE_RATE_HZ);
        if (bin_points[i] >= num_bins) {
            bin_points[i] = num_bins - 1;
        }
    }

    /* Build triangular filters. */
    memset(mel_fb, 0, sizeof(mel_fb));
    for (uint32_t m = 1; m <= NUM_MEL_FILTERS; m++) {
        uint32_t f_left   = bin_points[m - 1];
        uint32_t f_centre = bin_points[m];
        uint32_t f_right  = bin_points[m + 1];

        for (uint32_t k = f_left; k <= f_centre; k++) {
            if (f_centre != f_left) {
                mel_fb[m - 1][k] = (float)(k - f_left)
                                   / (float)(f_centre - f_left);
            }
        }
        for (uint32_t k = f_centre; k <= f_right; k++) {
            if (f_right != f_centre) {
                mel_fb[m - 1][k] = (float)(f_right - k)
                                   / (float)(f_right - f_centre);
            }
        }
    }
}

/**
 * @brief Build the DCT-II matrix.
 */
static void build_dct_matrix(void)
{
    for (uint32_t c = 0; c < NUM_MFCC_COEFFS; c++) {
        for (uint32_t m = 0; m < NUM_MEL_FILTERS; m++) {
            dct_matrix[c][m] = cosf((float)M_PI / (float)NUM_MEL_FILTERS
                               * ((float)m + 0.5f) * (float)c);
        }
    }
}

/* ── Public API ────────────────────────────────────────────────────────────── */

void AudioProcessing_Init(void)
{
    build_hann_window();
    build_mel_filterbank();
    build_dct_matrix();
    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
}

void AudioProcessing_ComputeMFCC(const int16_t *pcm_buf, float *features)
{
    /*
     * The input pcm_buf contains a sliding-window buffer long enough to
     * compute NUM_MFCC_FRAMES short-time frames with a hop of
     * AUDIO_HOP_SAMPLES each.
     *
     * Total length required:
     *   (NUM_MFCC_FRAMES - 1) * AUDIO_HOP_SAMPLES + FFT_SIZE
     */
    const uint32_t hop = AUDIO_HOP_SAMPLES;

    for (uint32_t frame = 0; frame < NUM_MFCC_FRAMES; frame++) {
        const int16_t *frame_start = pcm_buf + (frame * hop);

        /* 1. Convert to float and apply pre-emphasis filter. */
        fft_input[0] = (float)frame_start[0] / 32768.0f;
        for (uint32_t n = 1; n < FFT_SIZE; n++) {
            float cur  = (float)frame_start[n] / 32768.0f;
            float prev = (float)frame_start[n - 1] / 32768.0f;
            fft_input[n] = cur - PRE_EMPHASIS_COEFF * prev;
        }

        /* 2. Hann window. */
        arm_mult_f32(fft_input, hann_window, fft_input, FFT_SIZE);

        /* 3. Real FFT. */
        arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);

        /* 4. Power/magnitude spectrum.
         *    fft_output is [Re[0], Re[N/2], Re[1], Im[1], …]
         *    arm_cmplx_mag_f32 expects interleaved complex pairs. */
        /* DC and Nyquist are purely real. */
        mag_spectrum[0]           = fabsf(fft_output[0]);
        mag_spectrum[FFT_SIZE / 2] = fabsf(fft_output[1]);
        arm_cmplx_mag_f32(&fft_output[2],
                          &mag_spectrum[1],
                          (FFT_SIZE / 2) - 1);

        /* 5. Apply mel filter bank. */
        for (uint32_t m = 0; m < NUM_MEL_FILTERS; m++) {
            float energy = 0.0f;
            arm_dot_prod_f32(mag_spectrum, mel_fb[m],
                             FFT_SIZE / 2 + 1, &energy);
            /* 6. Log energy (with floor for numerical stability). */
            mel_energies[m] = logf(energy + 1e-6f);
        }

        /* 7. DCT-II → MFCC coefficients. */
        float *out = &features[frame * NUM_MFCC_COEFFS];
        for (uint32_t c = 0; c < NUM_MFCC_COEFFS; c++) {
            float sum = 0.0f;
            arm_dot_prod_f32(mel_energies, dct_matrix[c],
                             NUM_MEL_FILTERS, &sum);
            out[c] = sum;
        }
    }
}
