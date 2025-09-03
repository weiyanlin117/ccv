#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool fuse_v_mean=false, bool use_pv_fp16_accu=false>
__global__ void qk_int_sv_f8_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, float *__restrict__ V_mean,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q, 
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale);