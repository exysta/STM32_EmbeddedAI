/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

  /**
    * Description
    * v1.0: Minimum template to show how to use the Embedded Client API ST-AI 
    *
        */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "stai.h"



/* USER CODE BEGIN includes */
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/


/* Input defs ----------------------------------------------------------------*/

/**
STAI_ALIGNED(32) static uint8_t data_in_1[STAI_NETWORK_IN_1_SIZE_BYTES];

// Array to store the data of the input tensor
stai_ptr data_ins[] = {
  data_in_1
}; 
*/

/* Output defs ----------------------------------------------------------------*/

/**
STAI_ALIGNED(32) 
static uint8_t data_out_1[STAI_NETWORK_OUT_1_SIZE_BYTES];

// c-array to store the data of the output tensor
stai_ptr data_outs[] = {
  data_out_1
}; 
*/



/* Global byte buffer to save instantiated C-model network context */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

/* Activations buffers -------------------------------------------------------*/
STAI_ALIGNED(32) 
static uint8_t heap_overlay_pool[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];


/* Global c-array to handle the activations buffer */
stai_ptr data_activations[] = { heap_overlay_pool };

STAI_ALIGNED(32) static uint8_t states_1[4];
stai_ptr data_states[] = {
    states_1
};


/* Entry points --------------------------------------------------------------*/
/* Array of pointer to manage the model's input/output tensors */
static stai_size _in_size;
static stai_ptr stai_input[STAI_NETWORK_IN_NUM];
static stai_size _out_size;
static stai_ptr stai_output[STAI_NETWORK_OUT_NUM];

/* 
 * Bootstrap
 */
int aiInit(void) {
  stai_return_code ret_code;
  
  /* 1: Initialize runtime library */
  ret_code = stai_runtime_init();
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };

  /* 2: Initialize network model context */
  ret_code = stai_network_init(network_context);
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };

  /* 3: Set network activations buffers */
  ret_code = stai_network_set_activations(network_context, data_activations, STAI_NETWORK_ACTIVATIONS_NUM);
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };

  /* 4: Update the AI input/output buffers */
  /** Set network input/output buffers 
    * If the model uses no-inputs-allocation or no-outputs-allocation, the addresses of the input/output buffers
    * must be set before running the inference.
    * See https://stedgeai-dc.com/assets/embedded-docs/embedded_client_stai_api.html#ref_api_set_io
    * for more details
    */

  // current model uses allocate-inputs, use this part to overwrite the addresses of the input buffers
  /**
  ret_code = stai_network_set_inputs(network_context, data_ins, STAI_NETWORK_IN_NUM);
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };
   */
  // current model uses allocate-outputs, use this part to overwrite the addresses of the output buffers
  /** 
  ret_code = stai_network_set_outputs(network_context, data_outs, STAI_NETWORK_OUT_NUM);
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };
   */

  ret_code = stai_network_get_inputs(network_context, stai_input, &_in_size);
  if (ret_code != STAI_SUCCESS) {
      // handle error
  }

  ret_code = stai_network_get_outputs(network_context, stai_output, &_out_size);
  if (ret_code != STAI_SUCCESS) {
      // handle error
  }
  return 0;
}

int aiDeinit(void) {
  stai_return_code ret_code;

  /* 1: Deinitialize network model context */
  ret_code = stai_network_deinit(network_context);
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };

  /* 2: Deinitialize runtime library */
  ret_code = stai_runtime_deinit();
  if (ret_code != STAI_SUCCESS) { 
    // handle error
  };

  return 0;
}

/* 
 * Run inference
 */
int aiRun() {

  stai_return_code ret_code;

  /* Perform the inference */
  ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
  if (ret_code != STAI_SUCCESS) {
      ret_code = stai_network_get_error(network_context);
      // handle error
  };
  
  return 0;
}


/*
 * aiRunInference
 *
 * Called from main.c instead of the blocking STM32CubeAI_Studio_AI_Process().
 *
 * 1. Copies the caller's quantised INT8 input (MFCC buffer) into the model's
 *    input tensor that was retrieved by stai_network_get_inputs() during aiInit().
 * 2. Runs a single synchronous inference pass.
 * 3. Dequantises output index 1 — P(wakeword) from the Dense(2, softmax) head
 *    ( output layout: [P(negative), P(wakeword)] ).
 *
 * Returns the wake-word probability in [0, 1], or -1.0f on error.
 */
float aiRunInference(int8_t *input_data, size_t input_size,
                     float out_scale, int out_zp)
{
    stai_return_code ret;

    /* 1 — Copy caller's INT8 MFCC buffer into the model's input tensor.
     *     stai_input[0] points to the activation memory allocated by the
     *     runtime; we overwrite it with our pre-quantised features. */
    memcpy((void *)stai_input[0], input_data, input_size);

    /* 2 — Single synchronous inference pass */
    ret = stai_network_run(network_context, STAI_MODE_SYNC);
    if (ret != STAI_SUCCESS) {
        stai_network_get_error(network_context);
        return -1.0f;
    }

    /* 3 — Dequantise P(wakeword) from output tensor index 1.
     *     Formula: float = (int8_val - zero_point) * scale
     *     Caller passes OUTPUT_SCALE / OUTPUT_ZP from their quantisation report. */
    int8_t *out = (int8_t *)stai_output[0];
    float p_ww  = ((float)out[1] - (float)out_zp) * out_scale;

    return p_ww;
}

int acquire_and_process_data()
{
  /* fill the inputs of the c-model 
  for (int idx=0; idx < STAI_NETWORK_IN_NUM; idx++ )
  {
      stai_input[idx] = ....
  }

  */
  return 0;
}

int post_process()
{
  /* process the predictions
  for (int idx=0; idx < STAI_NETWORK_OUT_NUM; idx++ )
  {
      stai_output[idx] = ....
  }

  */
  return 0;
}



/* 
 * Example of main loop function
 */
void main_loop() {
  while (1) {
    /* 1 - Acquire, pre-process and fill the input buffers */
    acquire_and_process_data();

    /* 2 - Call inference engine */
    aiRun();

    /* 3 - Post-process the predictions */
    post_process();
  }
}


/* Entry points --------------------------------------------------------------*/


void STM32CubeAI_Studio_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
    aiInit();
    /* USER CODE END 5 */
}

void STM32CubeAI_Studio_AI_Process(void)
{
    main_loop();
} 

void STM32CubeAI_Studio_AI_Deinit(void)
{
    aiDeinit();
} 


#ifdef __cplusplus
}
#endif
