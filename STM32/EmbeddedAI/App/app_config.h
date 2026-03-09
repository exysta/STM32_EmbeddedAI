/**
  ******************************************************************************
  * @file    app_config.h
  * @author  GPM/AIS Application Team
  * @brief   APP configuration
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#ifndef __APP_CONFIG_H__
#define __APP_CONFIG_H__

#ifndef NUCLEO_N6_CONFIG
#define NUCLEO_N6_CONFIG                0
#endif

#define USE_MCU_DCACHE                  1
#define USE_MCU_ICACHE                  1


#define USE_EXTERNAL_RAM                1

#define USE_OVERDRIVE                   1       /* Using overdrive, clocks:CPU@800/NPU@1GHz, no overdrive: clocks:CPU@600/NPU@800 */


/* RELOC configuration */
#ifndef USE_RELOC_MODE
#define USE_RELOC_MODE                  0
#endif /* !USE_RELOC_MODE */



#endif /* __APP_CONFIG_H__ */

