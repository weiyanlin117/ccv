/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

typedef enum {
    DTYPE_FP16 = 0,
    DTYPE_FP16_MIX_FP32 = 1,
    DTYPE_FP32 = 2,
} sage_attn_pv_accum_dtype;

// Forward declaration of QuantInt8Kernel
template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, 
          bool has_sm_scale, bool sub_mean, typename T>
__global__ void QuantInt8Kernel(
    T *__restrict__ input, 
    T *__restrict__ mean, 
    int8_t *__restrict__ output, 
    float *__restrict__ scale, 
    float sm_scale, 
    const uint32_t num_tokens,
    const uint32_t stride_bz_input, 
    const uint32_t stride_seq_input, 
    const uint32_t stride_h_input,
    const uint32_t stride_bz_mean, 
    const uint32_t stride_h_mean,
    const uint32_t stride_bz_output, 
    const uint32_t stride_seq_output, 
    const uint32_t stride_h_output,
    const uint32_t stride_bz_scale, 
    const uint32_t stride_h_scale
);

template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale);

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output);

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(T *__restrict__ input, T *__restrict__ mean, half *__restrict__ output, const uint32_t num_tokens, 
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output);

                        