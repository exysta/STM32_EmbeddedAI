/**
 ******************************************************************************
 * @file    app_x-cube-ai.c
 * @brief   X-CUBE-AI application integration.
 *
 * This file wires the X-CUBE-AI generated network API (network.h /
 * network_data.h) into the STM32CubeIDE application framework.
 *
 * How to regenerate the network files
 * ─────────────────────────────────────
 *  1. Export your trained Edge Impulse project:
 *       Dashboard → Deployment → STM32Cube.AI Library → Build
 *     This downloads a .zip containing the .tflite or .onnx model.
 *
 *  2. In STM32CubeIDE, open the .ioc file and navigate to:
 *       Software Packs → X-CUBE-AI → Artificial Intelligence
 *     Add the network, point it at the downloaded model, and click "Analyse".
 *     Review the memory usage report, then click "Generate Code".
 *
 *  3. The following files will be created / updated automatically:
 *       X-CUBE-AI/App/network.c
 *       X-CUBE-AI/App/network.h
 *       X-CUBE-AI/App/network_data.c
 *       X-CUBE-AI/App/network_data.h
 *
 *  4. Build the project.  The network weights are linked into Flash; the
 *     activation buffer is allocated statically in SRAM (see
 *     wake_word_detection.c).
 ******************************************************************************
 */

#include "app_x-cube-ai.h"
#include "network.h"
#include "network_data.h"
#include <stdio.h>

/* ── MX_X_CUBE_AI_Init ─────────────────────────────────────────────────────── */

void MX_X_CUBE_AI_Init(void)
{
    /*
     * Print the network information for debug purposes.
     * In the full application the network is initialized by
     * WakeWordDetection_Init() in wake_word_detection.c.
     */
    printf("[XCUBEAI] Network: %s  (rev %s)\r\n",
           AI_NETWORK_MODEL_NAME, AI_NETWORK_MODEL_REVISION);
    printf("[XCUBEAI] In : %u × float32\r\n",  (unsigned)AI_NETWORK_IN_1_SIZE);
    printf("[XCUBEAI] Out: %u × float32\r\n",  (unsigned)AI_NETWORK_OUT_1_SIZE);
    printf("[XCUBEAI] Activations : %u bytes\r\n",
           (unsigned)AI_NETWORK_DATA_ACTIVATIONS_SIZE);
    printf("[XCUBEAI] Weights     : %u bytes\r\n",
           (unsigned)AI_NETWORK_DATA_WEIGHTS_SIZE);
}

/* ── MX_X_CUBE_AI_Process ──────────────────────────────────────────────────── */

void MX_X_CUBE_AI_Process(void)
{
    /*
     * Not used in this project – inference is driven by the main loop via
     * WakeWordDetection_Run().  This stub is kept for CubeMX compatibility.
     */
}
