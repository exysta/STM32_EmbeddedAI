/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __APP_AI_H
#define __APP_AI_H
#ifdef __cplusplus
extern "C" {
#endif
/**
  ******************************************************************************
  * @file    app_x-cube-ai.h
  * @author  X-CUBE-AI C code generator
  * @brief   AI entry function definitions
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
/* Includes ------------------------------------------------------------------*/
#include "stai.h"
#include "ai_datatypes_defines.h"

#include "network.h"

/* IO buffers ----------------------------------------------------------------*/

void STM32CubeAI_Studio_AI_Init(void);
void STM32CubeAI_Studio_AI_Process(void);
float aiRunInference(int8_t *input_data, size_t input_size,
        float out_scale, int out_zp);
/* USER CODE BEGIN includes */
/* USER CODE END includes */


#ifdef __cplusplus
}
#endif
#endif /*__STMicroelectronics_ST_EDGE_AI_4.0.0-20500 359356bb0_H */
