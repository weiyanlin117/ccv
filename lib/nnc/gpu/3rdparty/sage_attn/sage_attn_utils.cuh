#pragma once

#include <cuda_fp16.h>

// Helper macro for grid size calculation  
#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
  if (block_size == 64) {                                       \
    constexpr int BLOCK_SIZE = 64;                              \
    __VA_ARGS__                                                 \
  } else if (block_size == 128) {                               \
    constexpr int BLOCK_SIZE = 128;                             \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    printf("Unsupported block_size: %d\n", (int)block_size);    \
    assert(0);                                                   \
  }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, ...)  \
  if (warp_block_size == 16) {                                           \
    constexpr int WARP_BLOCK_SIZE = 16;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 32) {                                    \
    constexpr int WARP_BLOCK_SIZE = 32;                                  \
    __VA_ARGS__                                                          \
  }  else {                                                              \
    printf("Unsupported warp_block_size: %d\n", (int)warp_block_size);   \
    assert(0);                                                   		 \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    printf("Unsupported HEAD_DIM: %d\n", (int)head_dim);        \
    assert(0);                                                  \
  }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)              \
  if (is_causal == 1) {                                         \
    constexpr bool IS_CAUSAL = true;                            \
    __VA_ARGS__                                                 \
  } else if (is_causal == 0) {                                  \
    constexpr bool IS_CAUSAL = false;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    printf("Unsupported IS_CAUSAL: %d\n", (int)is_causal);        \
    assert(0);                                                  \
  }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)              \
  if (qk_quant_gran == 2) {                                         \
    constexpr int QK_QUANT_GRAN = 2;                            \
    __VA_ARGS__                                                 \
  } else if (qk_quant_gran == 3) {                                  \
    constexpr int QK_QUANT_GRAN = 3;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    printf("Unsupported qk_quant_gran: %d\n", (int)qk_quant_gran);        \
    assert(0);                                                  \
  }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)             \
  if (return_lse == 1) {                                         \
    constexpr bool RETURN_LSE = true;                            \
    __VA_ARGS__                                                  \
  } else if (return_lse == 0) {                                  \
    constexpr bool RETURN_LSE = false;                           \
    __VA_ARGS__                                                  \
  }  else {                                                      \
    printf("Unsupported RETURN_LSE: %d\n", (int)return_lse);        \
    assert(0);                                                  \
  }


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Direct kernel wrapper for qk_int8_sv_f16_accum_f16_attn with instruction buffer optimization.
 * This function provides a low-level interface to the SageAttention kernel without CCV dependencies.
 * 
 * @param Q             INT8 query tensor data pointer
 * @param K             INT8 key tensor data pointer  
 * @param V             FP16 value tensor data pointer
 * @param O             FP16 output tensor data pointer
 * @param Q_scale       FP32 query quantization scales
 * @param K_scale       FP32 key quantization scales
 * @param qdim          Query tensor dimensions [batch, heads, seq, dim] or [batch, seq, heads, dim]
 * @param kdim          Key tensor dimensions
 * @param vdim          Value tensor dimensions
 * @param odim          Output tensor dimensions
 * @param qscale_dim    Query scale tensor dimensions
 * @param kscale_dim    Key scale tensor dimensions
 * @param qstride       Query tensor strides
 * @param kstride       Key tensor strides
 * @param vstride       Value tensor strides
 * @param ostride       Output tensor strides
 * @param qscale_stride Query scale tensor strides
 * @param kscale_stride Key scale tensor strides
 * @param tensor_layout Tensor layout: 0=NHD [batch, seq, heads, dim], 1=HND [batch, heads, seq, dim]
 * @param is_causal     Whether to apply causal masking (0=false, 1=true)
 * @param qk_quant_gran Quantization granularity: 2=per_warp, 3=per_thread
 * @param sm_scale      Softmax scale factor (typically 1/sqrt(head_dim))
 * @param return_lse    Whether to return log-sum-exp (0=false, 1=true)
 */
void qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, half *V, half *O,
    float *Q_scale, float *K_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[],
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

/**
 * Direct kernel wrapper for qk_int8_sv_f16_accum_f32_attn with FP32 accumulation.
 * This function provides a low-level interface to the SageAttention kernel without CCV dependencies.
 * 
 * @param Q             INT8 query tensor data pointer
 * @param K             INT8 key tensor data pointer  
 * @param V             FP16 value tensor data pointer
 * @param O             FP16 output tensor data pointer
 * @param Q_scale       FP32 query quantization scales
 * @param K_scale       FP32 key quantization scales
 * @param qdim          Query tensor dimensions [batch, heads, seq, dim] or [batch, seq, heads, dim]
 * @param kdim          Key tensor dimensions
 * @param vdim          Value tensor dimensions
 * @param odim          Output tensor dimensions
 * @param qscale_dim    Query scale tensor dimensions
 * @param kscale_dim    Key scale tensor dimensions
 * @param qstride       Query tensor strides
 * @param kstride       Key tensor strides
 * @param vstride       Value tensor strides
 * @param ostride       Output tensor strides
 * @param qscale_stride Query scale tensor strides
 * @param kscale_stride Key scale tensor strides
 * @param tensor_layout Tensor layout: 0=NHD [batch, seq, heads, dim], 1=HND [batch, heads, seq, dim]
 * @param is_causal     Whether to apply causal masking (0=false, 1=true)
 * @param qk_quant_gran Quantization granularity: 2=per_warp, 3=per_thread
 * @param sm_scale      Softmax scale factor (typically 1/sqrt(head_dim))
 * @param return_lse    Whether to return log-sum-exp (0=false, 1=true)
 */
void ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct(
    int8_t *Q, int8_t *K, half *V, half *O,
    float *Q_scale, float *K_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[],
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

/**
 * Direct kernel wrapper for per-warp INT8 quantization of a single tensor.
 * This function provides a low-level interface to quantize FP16 tensors to INT8 using per-warp granularity.
 */
void ccv_nnc_quant_per_warp_int8_cuda_direct(
    half *input,          // Input tensor data (FP16)
    int8_t *output,       // Output tensor data (INT8)
    float *scale,         // Scale tensor data (FP32)
    int input_dim[],      // Input dimensions [batch, seq/heads, heads/seq, dim]
    int output_dim[],     // Output dimensions
    int scale_dim[],      // Scale dimensions
    int input_stride[],   // Input strides
    int output_stride[],  // Output strides
    int scale_stride[],   // Scale strides
    int block_size,       // Block size (128 or 64)
    int warp_block_size,  // Warp block size (16 or 32)
    int tensor_layout,    // 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for per-block INT8 quantization of a single tensor.
 * This function provides a low-level interface to quantize FP16 tensors to INT8 using per-block granularity.
 */
void ccv_nnc_quant_per_block_int8_cuda_direct(
    half *input,        // Input tensor data (FP16)
    int8_t *output,       // Output tensor data (INT8)
    float *scale,         // Scale tensor data (FP32)
    int input_dim[],      // Input dimensions
    int output_dim[],     // Output dimensions
    int scale_dim[],      // Scale dimensions
    int input_stride[],   // Input strides
    int output_stride[],  // Output strides
    int scale_stride[],   // Scale strides
    int block_size,       // Block size (128 or 64)
    int tensor_layout,    // 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for per-block INT8 quantization with mean subtraction.
 * This function quantizes FP16 tensors to INT8 using per-block granularity while subtracting a mean tensor.
 */
void ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda_direct(
    half *input,        // Input tensor data (FP16)
    half *mean,         // Mean tensor data (FP16)
    int8_t *output,       // Output tensor data (INT8)
    float *scale,         // Scale tensor data (FP32)
    int input_dim[],      // Input dimensions
    int mean_dim[],       // Mean dimensions
    int output_dim[],     // Output dimensions
    int scale_dim[],      // Scale dimensions
    int input_stride[],   // Input strides
    int mean_stride[],    // Mean strides
    int output_stride[],  // Output strides
    int scale_stride[],   // Scale strides
    int block_size,       // Block size (128 or 64)
    int tensor_layout,    // 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for combined per-warp Q and per-block K quantization.
 * This is a high-level function that quantizes both Q and K tensors with appropriate granularities.
 * Q uses per-warp quantization, K uses per-block quantization (with optional mean subtraction).
 */
void ccv_nnc_per_warp_int8_direct(
    half *q,            // Input Q tensor data (FP16)
    half *k,            // Input K tensor data (FP16)
    int8_t *q_int8,       // Output Q quantized (INT8)
    int8_t *k_int8,       // Output K quantized (INT8)
    float *q_scale,       // Output Q scales (FP32)
    float *k_scale,       // Output K scales (FP32)
    half *km,           // Optional K mean tensor data (FP16, can be NULL)
    int qdim[],           // Q dimensions [batch, seq/heads, heads/seq, dim]
    int kdim[],           // K dimensions
    int q_int8_dim[],     // Q output dimensions
    int k_int8_dim[],     // K output dimensions
    int q_scale_dim[],    // Q scale dimensions
    int k_scale_dim[],    // K scale dimensions
    int km_dim[],         // K mean dimensions (can be NULL if km is NULL)
    int qstride[],        // Q strides
    int kstride[],        // K strides
    int q_int8_stride[],  // Q output strides
    int k_int8_stride[],  // K output strides
    int q_scale_stride[], // Q scale strides
    int k_scale_stride[], // K scale strides
    int km_stride[],      // K mean strides (can be NULL if km is NULL)
    int BLKQ,             // Block size for Q (128)
    int WARPQ,            // Warp size for Q (32)
    int BLKK,             // Block size for K (64)
    int tensor_layout,    // 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for qk_int8_sv_f8_accum_f16_fuse_v_scale_attn with FP8 value tensor and instruction buffer optimization.
 * This function provides a low-level interface to the SageAttention kernel with FP8 value support.
 * 
 * @param Q             INT8 query tensor data pointer
 * @param K             INT8 key tensor data pointer  
 * @param V             INT8 value tensor data pointer (FP8)
 * @param O             FP16 output tensor data pointer
 * @param Q_scale       FP32 query quantization scales
 * @param K_scale       FP32 key quantization scales
 * @param V_scale       FP32 value quantization scales
 * @param qdim          Query tensor dimensions [batch, heads, seq, dim] or [batch, seq, heads, dim]
 * @param kdim          Key tensor dimensions
 * @param vdim          Value tensor dimensions
 * @param odim          Output tensor dimensions
 * @param qscale_dim    Query scale tensor dimensions
 * @param kscale_dim    Key scale tensor dimensions
 * @param vscale_dim    Value scale tensor dimensions
 * @param qstride       Query tensor strides
 * @param kstride       Key tensor strides
 * @param vstride       Value tensor strides
 * @param ostride       Output tensor strides
 * @param qscale_stride Query scale tensor strides
 * @param kscale_stride Key scale tensor strides
 * @param vscale_stride Value scale tensor strides
 * @param tensor_layout Tensor layout: 0=NHD [batch, seq, heads, dim], 1=HND [batch, heads, seq, dim]
 * @param is_causal     Whether to apply causal masking (0=false, 1=true)
 * @param qk_quant_gran Quantization granularity: 2=per_warp, 3=per_thread
 * @param sm_scale      Softmax scale factor (typically 1/sqrt(head_dim))
 * @param return_lse    Whether to return log-sum-exp (0=false, 1=true)
 */
void qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, int8_t *V, half *O,
    float *Q_scale, float *K_scale, float *V_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[], int vscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[], int vscale_stride[],
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

/**
 * Direct kernel wrapper for complete SageAttention computation with quantized QK and FP16 V.
 * This function combines quantization and attention computation in one call.
 * 
 * The function performs:
 * 1. Quantizes Q and K tensors using per-warp/per-block quantization
 * 2. Computes attention using quantized Q, K and FP16 V tensors
 * 
 * @param query           Input Q tensor data (FP16)
 * @param key             Input K tensor data (FP16) 
 * @param k_mean          Optional K mean tensor data (FP16, can be NULL)
 * @param value           Input V tensor data (FP16)
 * @param q_int8          Output Q quantized (INT8)
 * @param k_int8          Output K quantized (INT8)
 * @param query_scale     Output Q scales (FP32)
 * @param key_scale       Output K scales (FP32)
 * @param output          Output tensor data (FP16)
 * @param qdim            Query tensor dimensions [batch, seq/heads, heads/seq, dim]
 * @param kdim            Key tensor dimensions
 * @param vdim            Value tensor dimensions
 * @param odim            Output tensor dimensions
 * @param q_int8_dim      Q quantized output dimensions
 * @param k_int8_dim      K quantized output dimensions
 * @param query_scale_dim Q scale tensor dimensions
 * @param key_scale_dim   K scale tensor dimensions
 * @param qstride         Query tensor strides
 * @param kstride         Key tensor strides
 * @param vstride         Value tensor strides
 * @param ostride         Output tensor strides
 * @param q_int8_stride   Q quantized output strides
 * @param k_int8_stride   K quantized output strides
 * @param query_scale_stride Q scale tensor strides
 * @param key_scale_stride K scale tensor strides
 * @param tensor_layout   Tensor layout: 0=NHD [batch, seq, heads, dim], 1=HND [batch, heads, seq, dim]
 * @param is_causal       Whether to apply causal masking (0=false, 1=true)
 * @param qk_quant_gran   Quantization granularity (only per_warp=2 supported)
 * @param sm_scale        Softmax scale factor (typically 1/sqrt(head_dim))
 * @param return_lse      Whether to return log-sum-exp (0=false, 1=true) - not used yet
 * @param pv_accum_dtype  PV accumulation dtype (0=FP16, 1=FP16_MIX_FP32, 2=FP32)
 * @param BLKQ            Block size for Q (default 128)
 * @param WARPQ           Warp size for Q (default 32)
 * @param BLKK            Block size for K (default 64)
 * @param km_dim          K mean dimensions (can be NULL if k_mean is NULL)
 * @param km_stride       K mean strides (can be NULL if k_mean is NULL)
 * @param cuda_stream     CUDA stream for asynchronous execution
 */
void ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
    half *query,              // Input Q tensor data (FP16)
    half *key,                // Input K tensor data (FP16) 
    half *k_mean,             // Optional K mean tensor data (FP16, can be NULL)
    half *value,              // Input V tensor data (FP16)
    int8_t *q_int8,           // Output Q quantized (INT8)
    int8_t *k_int8,           // Output K quantized (INT8)
    float *query_scale,       // Output Q scales (FP32)
    float *key_scale,         // Output K scales (FP32)
    half *output,             // Output tensor data (FP16)
    int qdim[],               // Query tensor dimensions [batch, seq/heads, heads/seq, dim]
    int kdim[],               // Key tensor dimensions
    int vdim[],               // Value tensor dimensions
    int odim[],               // Output tensor dimensions
    int q_int8_dim[],         // Q quantized output dimensions
    int k_int8_dim[],         // K quantized output dimensions
    int query_scale_dim[],    // Q scale tensor dimensions
    int key_scale_dim[],      // K scale tensor dimensions
    int qstride[],            // Query tensor strides
    int kstride[],            // Key tensor strides
    int vstride[],            // Value tensor strides
    int ostride[],            // Output tensor strides
    int q_int8_stride[],      // Q quantized output strides
    int k_int8_stride[],      // K quantized output strides
    int query_scale_stride[], // Q scale tensor strides
    int key_scale_stride[],   // K scale tensor strides
    int tensor_layout,        // 0=NHD, 1=HND
    int is_causal,            // Whether to apply causal masking
    int qk_quant_gran,        // Quantization granularity (only per_warp supported)
    float sm_scale,           // Softmax scale factor
    int return_lse,           // Whether to return log-sum-exp (not used yet)
    int pv_accum_dtype,       // PV accumulation dtype (0=FP16, 1=FP16_MIX_FP32, 2=FP32)
    int BLKQ,                 // Block size for Q
    int WARPQ,               // Warp size for Q
    int BLKK,                // Block size for K
    int km_dim[],            // K mean dimensions (can be NULL if k_mean is NULL)
    int km_stride[],         // K mean strides (can be NULL if k_mean is NULL)
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for qk_int8_sv_f8_accum_f32_fuse_v_scale_attn with FP8 value tensor, FP32 accumulation and instruction buffer optimization.
 * This function provides a low-level interface to the SageAttention kernel with FP8 value support and FP32 accumulation.
 * 
 * @param Q             INT8 query tensor data pointer
 * @param K             INT8 key tensor data pointer  
 * @param V             INT8 value tensor data pointer (FP8)
 * @param O             FP16 output tensor data pointer
 * @param Q_scale       FP32 query quantization scales
 * @param K_scale       FP32 key quantization scales
 * @param V_scale       FP32 value quantization scales
 * @param qdim          Query tensor dimensions [batch, heads, seq, dim] or [batch, seq, heads, dim]
 * @param kdim          Key tensor dimensions
 * @param vdim          Value tensor dimensions
 * @param odim          Output tensor dimensions
 * @param qscale_dim    Query scale tensor dimensions
 * @param kscale_dim    Key scale tensor dimensions
 * @param vscale_dim    Value scale tensor dimensions
 * @param qstride       Query tensor strides
 * @param kstride       Key tensor strides
 * @param vstride       Value tensor strides
 * @param ostride       Output tensor strides
 * @param qscale_stride Query scale tensor strides
 * @param kscale_stride Key scale tensor strides
 * @param vscale_stride Value scale tensor strides
 * @param tensor_layout Tensor layout: 0=NHD [batch, seq, heads, dim], 1=HND [batch, heads, seq, dim]
 * @param sm_scale      Softmax scale factor (typically 1/sqrt(head_dim))
 */
void qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, int8_t *V, half *O,
    float *Q_scale, float *K_scale, float *V_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[], int vscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[], int vscale_stride[],
    int tensor_layout,
    float sm_scale);

/**
 * Direct kernel wrapper for FP8 quantization with fused scaling.
 * This function provides a low-level interface to quantize FP16 tensors to FP8/INT8 with fused scaling.
 * 
 * @param input           Input tensor data (FP16) 
 * @param output          Output tensor data (FP8/INT8)
 * @param scale           Scale tensor data (FP32)
 * @param input_dim       Input dimensions
 * @param output_dim      Output dimensions
 * @param scale_dim       Scale dimensions
 * @param input_stride    Input tensor strides
 * @param output_stride   Output tensor strides
 * @param scale_stride    Scale tensor strides
 * @param num_tokens      Actual number of tokens (unpadded)
 * @param scale_max       Maximum scale value
 * @param tensor_layout   Tensor layout: 0=NHD, 1=HND
 * @param cuda_stream     CUDA stream for asynchronous execution
 */
void ccv_nnc_scale_fuse_quant_cuda_direct(
    half *input,              // Input tensor data (FP16) 
    int8_t *output,           // Output tensor data (FP8/INT8)
    float *scale,             // Scale tensor data (FP32)
    int input_dim[],          // Input dimensions
    int output_dim[],         // Output dimensions
    int scale_dim[],          // Scale dimensions
    int input_stride[],       // Input tensor strides
    int output_stride[],      // Output tensor strides
    int scale_stride[],       // Scale tensor strides
    int num_tokens,           // Actual number of tokens (unpadded)
    float scale_max,          // Maximum scale value
    int tensor_layout,        // Tensor layout: 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for FP8 quantization with fused scaling and mean computation.
 * This function provides a low-level interface to quantize FP16 tensors to FP8/INT8 with fused scaling
 * while computing and subtracting means.
 * 
 * @param input           Input tensor data (FP16) 
 * @param output          Output tensor data (FP8/INT8)
 * @param mean            Mean tensor data (FP32)
 * @param scale           Scale tensor data (FP32)
 * @param input_dim       Input dimensions
 * @param output_dim      Output dimensions
 * @param mean_dim        Mean dimensions
 * @param scale_dim       Scale dimensions
 * @param input_stride    Input tensor strides
 * @param output_stride   Output tensor strides
 * @param mean_stride     Mean tensor strides
 * @param scale_stride    Scale tensor strides
 * @param num_tokens      Actual number of tokens (unpadded)
 * @param scale_max       Maximum scale value
 * @param tensor_layout   Tensor layout: 0=NHD, 1=HND
 * @param cuda_stream     CUDA stream for asynchronous execution
 */
void ccv_nnc_mean_scale_fuse_quant_cuda_direct(
    half *input,              // Input tensor data (FP16) 
    int8_t *output,           // Output tensor data (FP8/INT8)
    float *mean,              // Mean tensor data (FP32)
    float *scale,             // Scale tensor data (FP32)
    int input_dim[],          // Input dimensions
    int output_dim[],         // Output dimensions
    int mean_dim[],           // Mean dimensions
    int scale_dim[],          // Scale dimensions
    int input_stride[],       // Input tensor strides
    int output_stride[],      // Output tensor strides
    int mean_stride[],        // Mean tensor strides
    int scale_stride[],       // Scale tensor strides
    int num_tokens,           // Actual number of tokens (unpadded)
    float scale_max,          // Maximum scale value
    int tensor_layout,        // Tensor layout: 0=NHD, 1=HND
    cudaStream_t cuda_stream);

/**
 * Direct kernel wrapper for SageAttention with INT8 Q/K and FP8 V quantization.
 * This function provides a complete pipeline for quantized attention computation
 * with FP32+FP32 accumulation for maximum accuracy.
 * 
 * @param query           Input Q tensor data (FP16)
 * @param key             Input K tensor data (FP16)
 * @param value           Input V tensor data (FP16)
 * @param q_int8          Output Q quantized (INT8)
 * @param k_int8          Output K quantized (INT8)
 * @param v_fp8           Output V quantized (FP8)
 * @param query_scale     Output Q scales (FP32)
 * @param key_scale       Output K scales (FP32)
 * @param value_scale     Output V scales (FP32)
 * @param output          Output tensor data (FP16)
 * @param qdim            Query tensor dimensions [batch, seq/heads, heads/seq, dim]
 * @param kdim            Key tensor dimensions
 * @param vdim            Value tensor dimensions
 * @param odim            Output tensor dimensions
 * @param q_int8_dim      Q quantized output dimensions
 * @param k_int8_dim      K quantized output dimensions
 * @param v_fp8_dim       V quantized output dimensions
 * @param query_scale_dim Q scale tensor dimensions
 * @param key_scale_dim   K scale tensor dimensions
 * @param value_scale_dim V scale tensor dimensions
 * @param qstride         Query tensor strides
 * @param kstride         Key tensor strides
 * @param vstride         Value tensor strides
 * @param ostride         Output tensor strides
 * @param q_int8_stride   Q quantized output strides
 * @param k_int8_stride   K quantized output strides
 * @param v_fp8_stride    V quantized output strides
 * @param query_scale_stride Q scale tensor strides
 * @param key_scale_stride K scale tensor strides
 * @param value_scale_stride V scale tensor strides
 * @param tensor_layout   0=NHD, 1=HND (only NHD supported currently)
 * @param is_causal       Whether to apply causal masking
 * @param qk_quant_gran   Quantization granularity (2=per_warp supported)
 * @param sm_scale        Softmax scale factor
 * @param return_lse      Whether to return log-sum-exp (0 for our case)
 * @param pv_accum_dtype  PV accumulation dtype (2=FP32+FP32 supported)
 * @param cuda_stream     CUDA stream for asynchronous execution
 * 
 * Note: BLKQ=128, WARPQ=32, BLKK=64 are fixed for FP8 version
 */
void ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct(
    half *query,
    half *key,
    half *value,
    half *v_transposed,       // Intermediate V tensor after transpose_pad_permute (FP16)
    int8_t *q_int8,
    int8_t *k_int8,
    int8_t *v_fp8,
    float *query_scale,
    float *key_scale,
    float *value_scale,
    half *output,
    int qdim[],
    int kdim[],
    int vdim[],
    int odim[],
    int q_int8_dim[],
    int k_int8_dim[],
    int v_fp8_dim[],
    int query_scale_dim[],
    int key_scale_dim[],
    int value_scale_dim[],
    int qstride[],
    int kstride[],
    int vstride[],
    int ostride[],
    int q_int8_stride[],
    int k_int8_stride[],
    int v_fp8_stride[],
    int query_scale_stride[],
    int key_scale_stride[],
    int value_scale_stride[],
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse,
    int pv_accum_dtype,
    cudaStream_t cuda_stream);

#ifdef __cplusplus
}
#endif