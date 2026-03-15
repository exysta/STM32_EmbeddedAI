/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-03-09T23:04:19+0100
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[37]) {
   { .size_bytes = 3880, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 1}}, .scale = {1, (const float[1]){0.08156336843967438}}, .zeropoint = {1, (const int16_t[1]){-10}}, .name = "serving_default_mfcc_input0_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.06473666429519653}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_0_output" },
   { .size_bytes = 266112, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 99, 42, 64}}, .scale = {1, (const float[1]){0.06473666429519653}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_1_pad_before_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.09375828504562378}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_1_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 64}}, .scale = {1, (const float[1]){0.010783391073346138}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_2_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {1, (const float[1]){0.00901670940220356}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_3_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.013431535102427006}}, .zeropoint = {1, (const int16_t[1]){-12}}, .name = "gemm_4_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_5_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.055396609008312225}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_10_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.07973260432481766}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_11_output" },
   { .size_bytes = 266112, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 99, 42, 64}}, .scale = {1, (const float[1]){0.07973260432481766}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_12_pad_before_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.12319736927747726}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_12_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 64}}, .scale = {1, (const float[1]){0.0045121414586901665}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_13_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {1, (const float[1]){0.007730517536401749}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_14_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.00851499568670988}}, .zeropoint = {1, (const int16_t[1]){23}}, .name = "gemm_15_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_16_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.05964299291372299}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_21_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.08547744154930115}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_22_output" },
   { .size_bytes = 266112, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 99, 42, 64}}, .scale = {1, (const float[1]){0.08547744154930115}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_23_pad_before_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.12380179762840271}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_23_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 64}}, .scale = {1, (const float[1]){0.006248328369110823}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_24_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {1, (const float[1]){0.013639613054692745}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_25_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.015937188640236855}}, .zeropoint = {1, (const int16_t[1]){38}}, .name = "gemm_26_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_27_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.08252226561307907}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_32_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.08685844391584396}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_33_output" },
   { .size_bytes = 266112, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 99, 42, 64}}, .scale = {1, (const float[1]){0.08685844391584396}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_34_pad_before_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.12345076352357864}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_34_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 64}}, .scale = {1, (const float[1]){0.006445815321058035}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_35_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {1, (const float[1]){0.014802264049649239}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_36_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.02363372966647148}}, .zeropoint = {1, (const int16_t[1]){40}}, .name = "gemm_37_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_38_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.04626816138625145}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_43_output" },
   { .size_bytes = 248320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 40, 64}}, .scale = {1, (const float[1]){0.11792958527803421}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_44_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 64}}, .scale = {1, (const float[1]){0.009350219741463661}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_45_output" },
   { .size_bytes = 2, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {1, (const float[1]){0.08059944957494736}}, .zeropoint = {1, (const int16_t[1]){7}}, .name = "gemm_46_output" },
   { .size_bytes = 2, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_47_output" }
  },
  .nodes = (const stai_node_details[36]){
    {.id = 0, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d_0 */
    {.id = 1, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv2d_1_pad_before */
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_1 */
    {.id = 2, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* pool_2 */
    {.id = 3, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* gemm_3 */
    {.id = 4, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* gemm_4 */
    {.id = 5, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* nl_5 */
    {.id = 10, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){3, 7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* eltwise_10 */
    {.id = 11, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* conv2d_11 */
    {.id = 12, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* conv2d_12_pad_before */
    {.id = 12, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv2d_12 */
    {.id = 13, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* pool_13 */
    {.id = 14, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* gemm_14 */
    {.id = 15, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* gemm_15 */
    {.id = 16, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* nl_16 */
    {.id = 21, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){11, 15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* eltwise_21 */
    {.id = 22, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* conv2d_22 */
    {.id = 23, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv2d_23_pad_before */
    {.id = 23, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* conv2d_23 */
    {.id = 24, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* pool_24 */
    {.id = 25, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* gemm_25 */
    {.id = 26, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* gemm_26 */
    {.id = 27, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* nl_27 */
    {.id = 32, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){19, 23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* eltwise_32 */
    {.id = 33, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* conv2d_33 */
    {.id = 34, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* conv2d_34_pad_before */
    {.id = 34, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* conv2d_34 */
    {.id = 35, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* pool_35 */
    {.id = 36, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* gemm_36 */
    {.id = 37, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* gemm_37 */
    {.id = 38, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* nl_38 */
    {.id = 43, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){27, 31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* eltwise_43 */
    {.id = 44, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* conv2d_44 */
    {.id = 45, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* pool_45 */
    {.id = 46, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* gemm_46 */
    {.id = 47, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} } /* nl_47 */
  },
  .n_nodes = 36
};
#endif

