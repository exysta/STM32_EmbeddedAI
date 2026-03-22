/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
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
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "sai.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include "wakeword_inference.h"
#include "mfcc_processing.h"
#include "app_x-cube-ai.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// SAI configured as STEREO (2 slots: L + R)
// INMP441 L/R pin = GND  → drives LEFT slot only, RIGHT slot = zero
// DMA transfers 32-bit words.
// One stereo frame = 2 words (L word then R word).
//
// AUDIO_BLOCK_FRAMES : number of stereo frames per DMA half-transfer
// AUDIO_BLOCK_WORDS  : total 32-bit words per half (L+R interleaved)
#define AUDIO_BLOCK_FRAMES   256
#define AUDIO_BLOCK_WORDS    (AUDIO_BLOCK_FRAMES * 2)   // 512 words per half

#define SAMPLE_RATE          16000
#define AUDIO_GAIN           10      // applied to 24-bit value — tune as needed

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

// Full ping-pong DMA buffer: 2 halves × AUDIO_BLOCK_WORDS words = 1024 int32
int32_t dma_rx_buffer[AUDIO_BLOCK_WORDS * 2] __attribute__((aligned(32)));

// Set to 1 (first half ready) or 2 (second half ready) by DMA callbacks
volatile uint8_t audio_block_ready = 0;

// Output: AUDIO_BLOCK_FRAMES mono samples packed as 3-byte little-endian 24-bit
// + 4-byte sync header prefix
static uint8_t tx_buf[4 + AUDIO_BLOCK_FRAMES * 3];

// Tick value at which the wakeword LED should be turned off (0 = LED idle)
static uint32_t led_off_tick = 0U;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int _write(int file, char *ptr, int len)
{
	HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, HAL_MAX_DELAY);
	return len;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
	SCB_EnableICache();   // instruction cache
	SCB_EnableDCache();   // data cache
  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SAI1_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */

  // After peripheral inits:
  STM32CubeAI_Studio_AI_Init();
  MFCC_Init();

	// Prepopulate the sync header at the start of tx_buf (never changes)
	tx_buf[0] = 0xAA;
	tx_buf[1] = 0xBB;
	tx_buf[2] = 0xCC;
	tx_buf[3] = 0xDD;

	// Start DMA circular capture
	// Element count = total words in the full ping-pong buffer
	if (HAL_SAI_Receive_DMA(&hsai_BlockA1,
			(uint8_t*)dma_rx_buffer,
			AUDIO_BLOCK_WORDS * 2) != HAL_OK)
	{
		Error_Handler();
	}

	// Inside audio_block_ready block:

	printf("STM32 audio stream ready\r\n");
	printf("[BOOT] SYSCLK = %lu Hz\r\n", HAL_RCC_GetSysClockFreq());

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1)
	{

		if (audio_block_ready != 0)
		{
			// Grab and clear atomically before processing
			uint8_t block = audio_block_ready;
			audio_block_ready = 0;

			// Select the correct half of the ping-pong buffer
			int32_t *src = (block == 1) ?
					dma_rx_buffer :
					dma_rx_buffer + AUDIO_BLOCK_WORDS;

			// Extract left channel and pack to 24-bit
			// L+R are interleaved: word[0]=L, word[1]=R, word[2]=L ...
			// → left channel is at index i*2
			// INMP441 left-justifies 24-bit audio in the 32-bit word:
			// bits[31:8] = audio data, bits[7:0] = padding zeros
			// → arithmetic right shift by 8 gives signed 24-bit in int32
			for (int i = 0; i < AUDIO_BLOCK_FRAMES; i++)
			{
				int32_t raw    = src[i * 2];
				int32_t s24    = raw >> 8;
				int32_t gained = s24 * AUDIO_GAIN;

				// Clamp to signed 24-bit range [-8388608, 8388607]
				if (gained >  8388607) gained =  8388607;
				if (gained < -8388608) gained = -8388608;

				// Pack as little-endian 24-bit into tx_buf after the 4-byte header
				uint8_t *p = &tx_buf[4 + i * 3];
				p[0] = (uint8_t)(gained        & 0xFF);
				p[1] = (uint8_t)((gained >>  8) & 0xFF);
				p[2] = (uint8_t)((gained >> 16) & 0xFF);
			}
			MFCC_IngestBlock(src, AUDIO_BLOCK_FRAMES);

			// Single transmit: header + packed audio
//			HAL_UART_Transmit(&huart3, tx_buf, sizeof(tx_buf), HAL_MAX_DELAY);
		}
		if (g_mfcc_ready) {

			DWT->CYCCNT = 0;
			uint32_t t_mfcc_start = DWT->CYCCNT;
			MFCC_Compute();
			uint32_t t_mfcc_end = DWT->CYCCNT;
			printf("[BENCH] MFCC:      %lu cycles  →  %.2f ms\r\n",
			       t_mfcc_end - t_mfcc_start,
			       (float)(t_mfcc_end - t_mfcc_start) / 280000.0f);
		    /* Only run the (expensive) neural-net inference when the energy
		     * gate determined that speech-like energy is present.            */
		    if (g_energy_gate_passed) {
		        float p_ww;
		        if (WW_RunInference(&p_ww)) {
		            HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);  /* LED on  */
		            led_off_tick = HAL_GetTick() + 2500U;                /* 2.5 s   */
		        }
		    }
		}

		/* Turn LED off after timeout — non-blocking, checked every loop */
		if (led_off_tick != 0U && HAL_GetTick() >= led_off_tick) {
		    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);    /* LED off */
		    led_off_tick = 0U;
		}

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /*AXI clock gating */
  RCC->CKGAENR = 0xE003FFFF;

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = 64;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 35;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_6) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
	if (hsai->Instance == SAI1_Block_A)
		audio_block_ready = 1;
}

void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
	if (hsai->Instance == SAI1_Block_A)
		audio_block_ready = 2;
}

void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai)
{
	if (hsai->Instance == SAI1_Block_A)
		HAL_SAI_Receive_DMA(&hsai_BlockA1,
				(uint8_t*)dma_rx_buffer,
				AUDIO_BLOCK_WORDS * 2);
}

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1)
	{
	}
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
	/* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
