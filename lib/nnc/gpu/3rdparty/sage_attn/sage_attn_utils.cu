
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

#include "sage_attn_utils.cuh"
#include "math.cuh"
#include "qattn/attn_utils.cuh"
#include "qattn/qk_int_sv_f16_cuda_sm80_kernel_only.cuh"
#include "qattn/qk_int_sv_f8_cuda_sm89_kernel_only.cuh"
#include "fused.h"

extern "C" void ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct(
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
    int return_lse)
{
    // Extract dimensions based on tensor layout
    int batch_size, num_heads, seq_len, head_dim;
    int num_kv_heads, kv_len;
    
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        batch_size = qdim[0];
        num_heads = qdim[1];
        seq_len = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[1];
        kv_len = kdim[2];
    } else { // NHD layout: [batch, seq, heads, dim]
        batch_size = qdim[0];
        seq_len = qdim[1];
        num_heads = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[2];
        kv_len = kdim[1];
    }
    
    // Validate dimensions
    assert(num_heads % num_kv_heads == 0);
    const int num_kv_groups = num_heads / num_kv_heads;
    
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
        return;
    }
    
    // Extract strides based on tensor layout
    uint32_t stride_bz_q, stride_seq_q, stride_h_q;
    uint32_t stride_bz_k, stride_seq_k, stride_h_k;
    uint32_t stride_bz_v, stride_seq_v, stride_h_v;
    uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
    if (tensor_layout == 1) { // HND layout
        stride_bz_q = qstride[0];
        stride_h_q = qstride[1];
        stride_seq_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_h_k = kstride[1];
        stride_seq_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[1];
        stride_seq_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_h_o = ostride[1];
        stride_seq_o = ostride[2];
    } else { // NHD layout
        stride_bz_q = qstride[0];
        stride_seq_q = qstride[1];
        stride_h_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_seq_k = kstride[1];
        stride_h_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_seq_v = vstride[1];
        stride_h_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_seq_o = ostride[1];
        stride_h_o = ostride[2];
    }
    
    // Kernel configuration
    constexpr uint32_t CTA_Q = 128;
    constexpr uint32_t CTA_K = 64;
    constexpr uint32_t WARP_K = 64;
     
    // Launch kernel based on parameters with proper WARP_Q dispatch
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        // WARP_Q depends on HEAD_DIM for FP32 accumulation
        constexpr uint32_t WARP_Q = 32; // Use 32 for FP32 accumulation variant
        
        // Calculate shared memory requirement
        size_t smem_max = std::max(
            CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half),
            CTA_Q * HEAD_DIM * sizeof(half)
        );
        
        // Grid and block dimensions
        dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
        dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            // Use the exact same template instantiation pattern as ccv_nnc_qk_int8_sv_f16_accum_f32_attn
            // Q: QuantGranularity::kPerWarp, K: QuantGranularity::kPerBlock, float accum, use_inst_buffer: false
                    
            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
            
            // Calculate shared memory requirement  
            size_t smem_max = std::max(
                CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half),
                CTA_Q * HEAD_DIM * sizeof(half)
            );
            
            // Grid and block dimensions
            dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
            dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
            
            // Set the shared memory size attribute for the kernel
            auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, 
                DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
                float, false, half, ComputeUnit::kTensorCore, mask_mode, false, false>;
            
            cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
            
            // Direct kernel call using the exact same template as the working function
            kernel_func<<<grid_dim, block_dim, smem_max>>>(
                Q, K, V, O,
                nullptr,  // Lse (not used when return_lse=false)
                Q_scale, K_scale,
                nullptr,  // V_mean (not used)
                seq_len,  // qo_len
                kv_len,   // kv_len
                num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_seq_v, stride_h_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale
            );
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
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
    int return_lse)
{
    // Extract dimensions based on tensor layout
    int batch_size, num_heads, seq_len, head_dim;
    int num_kv_heads, kv_len;
    
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        batch_size = qdim[0];
        num_heads = qdim[1];
        seq_len = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[1];
        kv_len = kdim[2];
    } else { // NHD layout: [batch, seq, heads, dim]
        batch_size = qdim[0];
        seq_len = qdim[1];
        num_heads = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[2];
        kv_len = kdim[1];
    }
    
    // Validate dimensions
    assert(num_heads % num_kv_heads == 0);
    const int num_kv_groups = num_heads / num_kv_heads;
    
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
        return;
    }
    
    // Extract strides based on tensor layout
    uint32_t stride_bz_q, stride_seq_q, stride_h_q;
    uint32_t stride_bz_k, stride_seq_k, stride_h_k;
    uint32_t stride_bz_v, stride_seq_v, stride_h_v;
    uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
    if (tensor_layout == 1) { // HND layout
        stride_bz_q = qstride[0];
        stride_h_q = qstride[1];
        stride_seq_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_h_k = kstride[1];
        stride_seq_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[1];
        stride_seq_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_h_o = ostride[1];
        stride_seq_o = ostride[2];
    } else { // NHD layout
        stride_bz_q = qstride[0];
        stride_seq_q = qstride[1];
        stride_h_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_seq_k = kstride[1];
        stride_h_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_seq_v = vstride[1];
        stride_h_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_seq_o = ostride[1];
        stride_h_o = ostride[2];
    }
    
    // Kernel configuration
    constexpr uint32_t CTA_Q = 128;
    constexpr uint32_t CTA_K = 64;
    constexpr uint32_t WARP_K = 64;
    
    // Launch kernel based on parameters with proper WARP_Q dispatch
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        // WARP_Q depends on HEAD_DIM
        constexpr uint32_t WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
        
        // Calculate shared memory requirement
        size_t smem_max = std::max(
            CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half),
            CTA_Q * HEAD_DIM * sizeof(half)
        );
        
        // Grid and block dimensions
        dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
        dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                    
                    constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
                    
                    auto kernel_func = qk_int_sv_f16_attn_kernel<
                        CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, 
                        DataType::kInt8, 
                        static_cast<QuantGranularity>(QK_QUANT_GRAN), 
                        static_cast<QuantGranularity>(QK_QUANT_GRAN), 
                        float,  // AccumDataType
                        true,   // use_inst_buffer
                        half,   // OutputDataType (FP16)
                        ComputeUnit::kTensorCore, 
                        mask_mode, 
                        false,  // return_lse
                        false   // kInterleaved
                    >;
                    
                    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
                    
                    kernel_func<<<grid_dim, block_dim, smem_max>>>(
                        Q, K, V, O,
                        nullptr,  // Lse (not used when return_lse=false)
                        Q_scale, K_scale,
                        nullptr,  // V_mean (not used)
                        seq_len,  // qo_len
                        kv_len,   // kv_len
                        num_kv_groups,
                        stride_bz_q, stride_seq_q, stride_h_q,
                        stride_bz_k, stride_seq_k, stride_h_k,
                        stride_bz_v, stride_seq_v, stride_h_v,
                        stride_bz_o, stride_seq_o, stride_h_o,
                        sm_scale
                    );
            });
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: qk_int8_sv_f16_accum_f16_attn_inst_buf_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_quant_per_warp_int8_cuda_direct(
    half *input,        // Input tensor data (FP16)
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
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(input && output && scale);
    assert(block_size == 128 || block_size == 64);
    assert(warp_block_size == 16 || warp_block_size == 32);
    
    // Extract dimensions
    int batch_size, seq_len, num_heads, head_dim;
    if (tensor_layout == 1) { // HND layout
        batch_size = input_dim[0];
        num_heads = input_dim[1];
        seq_len = input_dim[2];
        head_dim = input_dim[3];
    } else { // NHD layout
        batch_size = input_dim[0];
        seq_len = input_dim[1];
        num_heads = input_dim[2];
        head_dim = input_dim[3];
    }
        
    // Extract input strides
    uint32_t stride_bz_input, stride_seq_input, stride_h_input;
    if (tensor_layout == 1) { // HND layout
        stride_bz_input = input_stride[0];
        stride_h_input = input_stride[1];
        stride_seq_input = input_stride[2];
    } else { // NHD layout  GPU_TENSOR_NHWC(000, 8U, batch_size, R, Hq, D)
        stride_bz_input = input_stride[0];
        stride_seq_input = input_stride[1];
        stride_h_input = input_stride[2];
    }
    
    // Extract output strides
    uint32_t stride_bz_output, stride_seq_output, stride_h_output;
    if (tensor_layout == 1) { // HND layout
        stride_bz_output = output_stride[0];
        stride_h_output = output_stride[1];
        stride_seq_output = output_stride[2];
    } else { // NHD layout
        stride_bz_output = output_stride[0];
        stride_seq_output = output_stride[1];
        stride_h_output = output_stride[2];
    }
    
    // Extract scale strides - these are crucial for correct scale writing!
    // batch_size, q_scale_blocks, Hq
    uint32_t stride_scale_bz, stride_scale_h;
    stride_scale_bz = scale_stride[1];  // batch stride for scale tensor
    stride_scale_h = scale_stride[2];   // head stride for scale tensor
    
    // Launch quantization kernel using QuantInt8Kernel
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
            constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
            dim3 grid_dim(DIV_CEIL(seq_len, WARP_BLOCK_SIZE), num_heads, batch_size);
            dim3 block_dim(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
            
            QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, half>
            <<<grid_dim, block_dim, 0, cuda_stream>>>(
                (half*)input,  // input tensor
                nullptr,       // mean (not used for per-warp)
                output,        // output tensor
                scale,         // scale tensor
                0.0f,          // sm_scale (not used)
                seq_len,       // num_tokens
                stride_bz_input, stride_seq_input, stride_h_input,    // input strides
                0, 0,                                                 // mean strides (unused)
                stride_bz_output, stride_seq_output, stride_h_output, // output strides
                stride_scale_bz, stride_scale_h                       // scale strides
            );
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_warp_int8_cuda_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_quant_per_block_int8_cuda_direct(
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
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(input && output && scale);
    assert(block_size == 128 || block_size == 64);
    
    // Extract dimensions
    int batch_size, seq_len, num_heads, head_dim;
    if (tensor_layout == 1) { // HND layout
        batch_size = input_dim[0];
        num_heads = input_dim[1];
        seq_len = input_dim[2];
        head_dim = input_dim[3];
    } else { // NHD layout
        batch_size = input_dim[0];
        seq_len = input_dim[1];
        num_heads = input_dim[2];
        head_dim = input_dim[3];
    }
    
    // Extract input strides
    uint32_t stride_bz_input, stride_seq_input, stride_h_input;
    if (tensor_layout == 1) { // HND layout
        stride_bz_input = input_stride[0];
        stride_h_input = input_stride[1];
        stride_seq_input = input_stride[2];
    } else { // NHD layout
        stride_bz_input = input_stride[0];
        stride_seq_input = input_stride[1];
        stride_h_input = input_stride[2];
    }
    
    // Extract output strides
    uint32_t stride_bz_output, stride_seq_output, stride_h_output;
    if (tensor_layout == 1) { // HND layout
        stride_bz_output = output_stride[0];
        stride_h_output = output_stride[1];
        stride_seq_output = output_stride[2];
    } else { // NHD layout
        stride_bz_output = output_stride[0];
        stride_seq_output = output_stride[1];
        stride_h_output = output_stride[2];
    }
    
    // Extract scale strides - these are crucial for correct scale writing!
    uint32_t stride_scale_bz, stride_scale_h;
    stride_scale_bz = scale_stride[1];  // batch stride for scale tensor
    stride_scale_h = scale_stride[2];   // head stride for scale tensor
    
    // Launch quantization kernel using QuantInt8Kernel
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
            int num_blocks = DIV_CEIL(seq_len, BLOCK_SIZE);
            dim3 grid_dim(num_blocks, num_heads, batch_size);
            dim3 block_dim(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
            
            QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, half>
            <<<grid_dim, block_dim, 0, cuda_stream>>>(
                (half*)input,  // input tensor
                nullptr,       // mean (not used for per-block)
                output,        // output tensor
                scale,         // scale tensor
                0.0f,          // sm_scale (not used)
                seq_len,       // num_tokens
                stride_bz_input, stride_seq_input, stride_h_input,    // input strides
                0, 0,                                                 // mean strides (unused)
                stride_bz_output, stride_seq_output, stride_h_output, // output strides
                stride_scale_bz, stride_scale_h                       // scale strides
            );
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_block_int8_cuda_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda_direct(
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
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(input && mean && output && scale);
    assert(block_size == 128 || block_size == 64);
    
    // Extract dimensions
    int batch_size, seq_len, num_heads, head_dim;
    if (tensor_layout == 1) { // HND layout
        batch_size = input_dim[0];
        num_heads = input_dim[1];
        seq_len = input_dim[2];
        head_dim = input_dim[3];
    } else { // NHD layout
        batch_size = input_dim[0];
        seq_len = input_dim[1];
        num_heads = input_dim[2];
        head_dim = input_dim[3];
    }
        
    // Extract input strides
    uint32_t stride_bz_input, stride_seq_input, stride_h_input;
    uint32_t mean_stride_bz, mean_stride_h;
    
    if (tensor_layout == 1) { // HND layout
        stride_bz_input = input_stride[0];
        stride_h_input = input_stride[1];
        stride_seq_input = input_stride[2];
        
        mean_stride_bz = mean_stride[0];
        mean_stride_h = mean_stride[1];
    } else { // NHD layout
        stride_bz_input = input_stride[0];
        stride_seq_input = input_stride[1];
        stride_h_input = input_stride[2];
        
        mean_stride_bz = mean_stride[0];
        mean_stride_h = mean_stride[1];
    }
    
    // Extract output strides
    uint32_t stride_bz_output, stride_seq_output, stride_h_output;
    if (tensor_layout == 1) { // HND layout
        stride_bz_output = output_stride[0];
        stride_h_output = output_stride[1];
        stride_seq_output = output_stride[2];
    } else { // NHD layout
        stride_bz_output = output_stride[0];
        stride_seq_output = output_stride[1];
        stride_h_output = output_stride[2];
    }
    
    // Extract scale strides - these are crucial for correct scale writing!
    uint32_t stride_scale_bz, stride_scale_h;
    stride_scale_bz = scale_stride[1];  // batch stride for scale tensor
    stride_scale_h = scale_stride[2];   // head stride for scale tensor
    
    // Launch quantization kernel with mean subtraction using QuantInt8Kernel
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
            int num_blocks = DIV_CEIL(seq_len, BLOCK_SIZE);
            dim3 grid_dim(num_blocks, num_heads, batch_size);
            dim3 block_dim(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
            
            QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, half>
            <<<grid_dim, block_dim, 0, cuda_stream>>>(
                (half*)input,  // input tensor
                (half*)mean,   // mean tensor (used for sub_mean=true)
                output,        // output tensor
                scale,         // scale tensor
                0.0f,          // sm_scale (not used)
                seq_len,       // num_tokens
                stride_bz_input, stride_seq_input, stride_h_input,    // input strides
                mean_stride_bz, mean_stride_h,                        // mean strides
                stride_bz_output, stride_seq_output, stride_h_output, // output strides
                stride_scale_bz, stride_scale_h                       // scale strides
            );
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_per_warp_int8_direct(
    half *q,            // Input Q tensor data (FP16)
    half *k,            // Input K tensor data (FP16)
    int8_t *q_int8,       // Output Q quantized (INT8)
    int8_t *k_int8,       // Output K quantized (INT8)
    float *q_scale,       // Output Q scales (FP32)
    float *k_scale,       // Output K scales (FP32)
    half *km,           // Optional K mean tensor data (FP16)
    int qdim[],           // Q dimensions [batch, seq/heads, heads/seq, dim]
    int kdim[],           // K dimensions
    int q_int8_dim[],     // Q output dimensions
    int k_int8_dim[],     // K output dimensions
    int q_scale_dim[],    // Q scale dimensions
    int k_scale_dim[],    // K scale dimensions
    int km_dim[],         // K mean dimensions
    int qstride[],        // Q strides
    int kstride[],        // K strides
    int q_int8_stride[],  // Q output strides
    int k_int8_stride[],  // K output strides
    int q_scale_stride[], // Q scale strides
    int k_scale_stride[], // K scale strides
    int km_stride[],      // K mean strides
    int BLKQ,             // Block size for Q (128)
    int WARPQ,            // Warp size for Q (32)
    int BLKK,             // Block size for K (64)
    int tensor_layout,    // 0=NHD, 1=HND
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(q && k && q_int8 && k_int8 && q_scale && k_scale);
    
    // Quantize Q using per-warp quantization
    ccv_nnc_quant_per_warp_int8_cuda_direct(
        q, q_int8, q_scale,
        qdim, q_int8_dim, q_scale_dim,
        qstride, q_int8_stride, q_scale_stride,
        BLKQ, WARPQ, tensor_layout, cuda_stream
    );
    
    // Quantize K using per-block quantization (with or without mean subtraction)
    if (km != NULL) {
        // Use mean subtraction version
        ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda_direct(
            k, km, k_int8, k_scale,
            kdim, km_dim, k_int8_dim, k_scale_dim,
            kstride, km_stride, k_int8_stride, k_scale_stride,
            BLKK, tensor_layout, cuda_stream
        );
    } else {
        // Regular per-block quantization
        ccv_nnc_quant_per_block_int8_cuda_direct(
            k, k_int8, k_scale,
            kdim, k_int8_dim, k_scale_dim,
            kstride, k_int8_stride, k_scale_stride,
            BLKK, tensor_layout, cuda_stream
        );
    }
    
}

extern "C" void qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_direct(
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
    int return_lse)
{
    // Extract dimensions based on tensor layout
    int batch_size, num_heads, seq_len, head_dim;
    int num_kv_heads, kv_len;
    
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        batch_size = qdim[0];
        num_heads = qdim[1];
        seq_len = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[1];
        kv_len = kdim[2];
    } else { // NHD layout: [batch, seq, heads, dim]
        batch_size = qdim[0];
        seq_len = qdim[1];
        num_heads = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[2];
        kv_len = kdim[1];
    }
    
    // Validate dimensions
    assert(num_heads % num_kv_heads == 0);
    const int num_kv_groups = num_heads / num_kv_heads;
    
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
        return;
    }
    
    // Extract strides based on tensor layout
    uint32_t stride_bz_q, stride_seq_q, stride_h_q;
    uint32_t stride_bz_k, stride_seq_k, stride_h_k;
    uint32_t stride_bz_v, stride_h_v, stride_d_v;
    uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
    if (tensor_layout == 1) { // HND layout
        stride_bz_q = qstride[0];
        stride_h_q = qstride[1];
        stride_seq_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_h_k = kstride[1];
        stride_seq_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[1];
        stride_d_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_h_o = ostride[1];
        stride_seq_o = ostride[2];
    } else { // NHD layout
        stride_bz_q = qstride[0];
        stride_seq_q = qstride[1];
        stride_h_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_seq_k = kstride[1];
        stride_h_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[2];
        stride_d_v = vstride[1];
        
        stride_bz_o = ostride[0];
        stride_seq_o = ostride[1];
        stride_h_o = ostride[2];
    }
    
    // Kernel configuration
    constexpr uint32_t CTA_Q = 128;
    constexpr uint32_t CTA_K = 64;
    constexpr uint32_t WARP_K = 64;
    
    // Launch kernel based on parameters with proper WARP_Q dispatch
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        // WARP_Q depends on HEAD_DIM
        constexpr uint32_t WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
        
        // Calculate shared memory requirement
        size_t smem_max = std::max(
            CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t),
            CTA_Q * HEAD_DIM * sizeof(half)
        );
        
        // Grid and block dimensions
        dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
        dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                    
                    constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
                    
                    auto kernel_func = qk_int_sv_f8_attn_kernel<
                        CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, 
                        DataType::kInt8, 
                        static_cast<QuantGranularity>(QK_QUANT_GRAN), 
                        static_cast<QuantGranularity>(QK_QUANT_GRAN), 
                        float,  // AccumDataType
                        true,   // use_inst_buffer
                        half,   // OutputDataType (FP16)
                        ComputeUnit::kCudaCore, 
                        mask_mode, 
                        false,  // return_lse
                        true,   // fuse_v_scale
                        false,  // fuse_v_mean
                        true    // use_pv_fp16_accu
                    >;
                    
                    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
                    
                    kernel_func<<<grid_dim, block_dim, smem_max>>>(
                        Q, K, V, O,
                        nullptr,  // Lse (not used when return_lse=false)
                        Q_scale, K_scale, V_scale,
                        nullptr,  // V_mean (not used)
                        seq_len,  // qo_len
                        kv_len,   // kv_len
                        num_kv_groups,
                        stride_bz_q, stride_seq_q, stride_h_q,
                        stride_bz_k, stride_seq_k, stride_h_k,
                        stride_bz_v, stride_h_v, stride_d_v,
                        stride_bz_o, stride_seq_o, stride_h_o,
                        sm_scale
                    );
            });
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}


extern "C" void qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, int8_t *V, half *O,
    float *Q_scale, float *K_scale, float *V_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[], int vscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[], int vscale_stride[],
    int tensor_layout,
    float sm_scale)
{
    // Extract dimensions based on tensor layout
    int batch_size, num_heads, seq_len, head_dim;
    int num_kv_heads, kv_len;
    
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        batch_size = qdim[0];
        num_heads = qdim[1];
        seq_len = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[1];
        kv_len = kdim[2];
    } else { // NHD layout: [batch, seq, heads, dim]
        batch_size = qdim[0];
        seq_len = qdim[1];
        num_heads = qdim[2];
        head_dim = qdim[3];
        num_kv_heads = kdim[2];
        kv_len = kdim[1];
    }
    
    // Validate dimensions
    assert(num_heads % num_kv_heads == 0);
    const int num_kv_groups = num_heads / num_kv_heads;
    
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
        return;
    }
    
    // Extract strides based on tensor layout
    uint32_t stride_bz_q, stride_seq_q, stride_h_q;
    uint32_t stride_bz_k, stride_seq_k, stride_h_k;
    uint32_t stride_bz_v, stride_h_v, stride_d_v;
    uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
    if (tensor_layout == 1) { // HND layout
        stride_bz_q = qstride[0];
        stride_h_q = qstride[1];
        stride_seq_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_h_k = kstride[1];
        stride_seq_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[1];
        stride_d_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_h_o = ostride[1];
        stride_seq_o = ostride[2];
    } else { // NHD layout
        stride_bz_q = qstride[0];
        stride_seq_q = qstride[1];
        stride_h_q = qstride[2];
        
        stride_bz_k = kstride[0];
        stride_seq_k = kstride[1];
        stride_h_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[2];
        stride_d_v = vstride[1];
        
        stride_bz_o = ostride[0];
        stride_seq_o = ostride[1];
        stride_h_o = ostride[2];
    }
    
    // Kernel configuration
    constexpr uint32_t CTA_Q = 128;
    constexpr uint32_t CTA_K = 64;
    constexpr uint32_t WARP_K = 64;
    constexpr uint32_t WARP_Q = 32;

    // Launch kernel based on parameters with proper WARP_Q dispatch
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        
        // Calculate shared memory requirement
        size_t smem_max = std::max(
            CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t),
            CTA_Q * HEAD_DIM * sizeof(half)
        );
        
        // Grid and block dimensions
        dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
        dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                    
        constexpr MaskMode mask_mode = MaskMode::kNone;
        
        auto kernel_func = qk_int_sv_f8_attn_kernel<
            CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, 
            DataType::kInt8, 
            static_cast<QuantGranularity>(QuantGranularity::kPerWarp), 
            static_cast<QuantGranularity>(QuantGranularity::kPerWarp), 
            float,  // AccumDataType
            true,   // use_inst_buffer
            half,   // OutputDataType (FP16)
            ComputeUnit::kCudaCore, 
            mask_mode, 
            false,  // return_lse
            true,   // fuse_v_scale
            false,  // fuse_v_mean
            false   // use_pv_fp16_accu
        >;
        
        cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        
        kernel_func<<<grid_dim, block_dim, smem_max>>>(
            Q, K, V, O,
            nullptr,  // Lse (not used when return_lse=false)
            Q_scale, K_scale, V_scale,
            nullptr,  // V_mean (not used)
            seq_len,  // qo_len
            kv_len,   // kv_len
            num_kv_groups,
            stride_bz_q, stride_seq_q, stride_h_q,
            stride_bz_k, stride_seq_k, stride_h_k,
            stride_bz_v, stride_h_v, stride_d_v,
            stride_bz_o, stride_seq_o, stride_h_o,
            sm_scale
        );
   
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
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
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(query && key && value && q_int8 && k_int8 && query_scale && key_scale && output);
    
    // Step 1: Quantize Q and K tensors using per-warp quantization
    ccv_nnc_per_warp_int8_direct(
        query, key,                    // Input Q and K tensors
        q_int8, k_int8,               // Output quantized tensors
        query_scale, key_scale,       // Output scale tensors
        k_mean,                       // Optional K mean tensor
        qdim, kdim,                   // Input dimensions
        q_int8_dim, k_int8_dim,       // Output quantized dimensions
        query_scale_dim, key_scale_dim, // Scale dimensions
        km_dim,                       // K mean dimensions
        qstride, kstride,             // Input strides
        q_int8_stride, k_int8_stride, // Output quantized strides
        query_scale_stride, key_scale_stride, // Scale strides
        km_stride,                    // K mean strides
        BLKQ, WARPQ, BLKK,           // Block sizes
        tensor_layout,                // Tensor layout
        cuda_stream                   // CUDA stream
    );
        
    // Step 2: Perform quantized attention computation
    if (pv_accum_dtype == 2) { // DTYPE_FP32
        ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct(
            q_int8, k_int8, value, output,      // Q, K, V, O tensors
            query_scale, key_scale,             // Q and K scales
            qdim, kdim, vdim, odim,            // Tensor dimensions
            query_scale_dim, key_scale_dim,     // Scale dimensions
            qstride, kstride, vstride, ostride, // Tensor strides
            query_scale_stride, key_scale_stride, // Scale strides
            tensor_layout,                      // Tensor layout
            is_causal,                         // Causal masking
            qk_quant_gran,                     // Quantization granularity
            sm_scale,                          // Softmax scale
            return_lse                         // Return LSE (not used)
        );
    } else if (pv_accum_dtype == 1) { // DTYPE_FP16_MIX_FP32
        qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
            q_int8, k_int8, value, output,      // Q, K, V, O tensors
            query_scale, key_scale,             // Q and K scales
            qdim, kdim, vdim, odim,            // Tensor dimensions
            query_scale_dim, key_scale_dim,     // Scale dimensions
            qstride, kstride, vstride, ostride, // Tensor strides
            query_scale_stride, key_scale_stride, // Scale strides
            tensor_layout,                      // Tensor layout
            is_causal,                         // Causal masking
            qk_quant_gran,                     // Quantization granularity
            sm_scale,                          // Softmax scale
            return_lse                         // Return LSE (not used)
        );
    } else {
        fprintf(stderr, "ERROR: Unsupported pv_accum_dtype: %d\n", pv_accum_dtype);
        return;
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct failed: %s\n", 
               cudaGetErrorString(error));
        return;
    }
}

extern "C" void ccv_nnc_scale_fuse_quant_cuda_direct(
    half *input,              // Input tensor data (FP16) 
    int8_t *output,           // Output tensor data (FP8/INT8)
    float *scale,             // Scale tensor data (FP32)
    int input_dim[],          // Input dimensions [batch, head_dim/seq, num_heads/heads, num_tokens_padded/dim]
    int output_dim[],         // Output dimensions (same as input)
    int scale_dim[],          // Scale dimensions [batch, num_heads, head_dim]
    int input_stride[],       // Input tensor strides
    int output_stride[],      // Output tensor strides
    int scale_stride[],       // Scale tensor strides
    int num_tokens,           // Actual number of tokens (unpadded)
    float scale_max,          // Maximum scale value (e.g., 448.0 for FP8)
    int tensor_layout,        // Tensor layout: 0=NHD, 1=HND
    cudaStream_t cuda_stream)
{
    // Validate inputs
    assert(input && output && scale);
    
    // Extract dimensions following the scale_fuse_quant_cuda pattern:
    // Input: [batch, head_dim, num_heads, num_tokens_padded] for NHD
    // Input: [batch, num_heads, head_dim, num_tokens_padded] for HND
    int batch_size, head_dim, num_heads, num_tokens_padded;
    if (tensor_layout == 0) { // NHD layout
        batch_size = input_dim[0];
        head_dim = input_dim[1];
        num_heads = input_dim[2]; 
        num_tokens_padded = input_dim[3];
    } else { // HND layout
        batch_size = input_dim[0];
        num_heads = input_dim[1];
        head_dim = input_dim[2];
        num_tokens_padded = input_dim[3];
    }
    
    // Extract input strides
    uint32_t stride_bz_input, stride_d_input, stride_h_input;
    if (tensor_layout == 0) { // NHD layout
        stride_bz_input = input_stride[0];
        stride_d_input = input_stride[1];   // head_dim stride
        stride_h_input = input_stride[2];   // num_heads stride
    } else { // HND layout
        stride_bz_input = input_stride[0];
        stride_h_input = input_stride[1];   // num_heads stride
        stride_d_input = input_stride[2];   // head_dim stride
    }
    
    // Extract output strides
    uint32_t stride_bz_output, stride_d_output, stride_h_output;
    if (tensor_layout == 0) { // NHD layout
        stride_bz_output = output_stride[0];
        stride_d_output = output_stride[1]; // head_dim stride
        stride_h_output = output_stride[2]; // num_heads stride
    } else { // HND layout
        stride_bz_output = output_stride[0];
        stride_h_output = output_stride[1]; // num_heads stride
        stride_d_output = output_stride[2]; // head_dim stride
    }
    
    // Extract scale strides - [batch, num_heads, head_dim]
    uint32_t stride_scale_bz = scale_stride[0];  // batch stride
    uint32_t stride_scale_h = scale_stride[1];   // num_heads stride
    
    // Validate tensor dimensions
    if (output_dim[0] != input_dim[0] || output_dim[1] != input_dim[1] || 
        output_dim[2] != input_dim[2] || output_dim[3] != input_dim[3]) {
        fprintf(stderr, "ERROR: Input and output dimensions must match\n");
        return;
    }
    
    if (scale_dim[0] != batch_size || scale_dim[1] != num_heads || scale_dim[2] != head_dim) {
        fprintf(stderr, "ERROR: Scale dimensions mismatch: expected [%d, %d, %d], got [%d, %d, %d]\n",
               batch_size, num_heads, head_dim, scale_dim[0], scale_dim[1], scale_dim[2]);
        return;
    }
    
    // Launch kernel configuration matching original scale_fuse_quant_cuda
    constexpr int CTA_SIZE = 256;
    constexpr uint32_t pad_size = 64; // Match original implementation
    
    dim3 grid(num_heads, batch_size, head_dim);
    dim3 block(CTA_SIZE);
    
    // Launch MeanScaleKernel with template parameters matching original implementation
    // pad_size=64, sub_mean=false (for scale_fuse_quant_cuda, not mean_scale_fuse_quant_cuda)
    MeanScaleKernel<pad_size, false, half><<<grid, block, 0, cuda_stream>>>(
        input,                    // Input tensor
        output,                   // Output tensor
        nullptr,                  // Mean tensor (not used for scale_fuse_quant)
        scale,                    // Scale tensor
        scale_max,                // Maximum scale value
        num_tokens,               // Actual number of tokens
        stride_bz_input, stride_d_input, stride_h_input,     // Input strides
        stride_bz_output, stride_d_output, stride_h_output,  // Output strides
        0, 0,                     // Mean strides (not used)
        stride_scale_bz, stride_scale_h  // Scale strides
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_scale_fuse_quant_cuda_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_mean_scale_fuse_quant_cuda_direct(
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
    cudaStream_t cuda_stream)
{
    // Map input dimensions based on tensor layout
    int batch_size, num_heads, head_dim, num_tokens_padded;
    uint32_t stride_bz_input, stride_d_input, stride_h_input;
    uint32_t stride_bz_output, stride_d_output, stride_h_output;
    uint32_t stride_bz_mean, stride_h_mean;
    uint32_t stride_bz_scale, stride_h_scale;
    
    batch_size = input_dim[0];
    num_tokens_padded = input_dim[3];
    
    if (tensor_layout == 0) { // NHD layout
        num_heads = input_dim[2];
        head_dim = input_dim[1];
        stride_bz_input = input_stride[0];  // batch stride
        stride_d_input = input_stride[1];   // head_dim stride  
        stride_h_input = input_stride[2];   // num_heads stride
        stride_bz_output = output_stride[0];
        stride_d_output = output_stride[1];
        stride_h_output = output_stride[2];
    } else { // HND layout
        num_heads = input_dim[1];
        head_dim = input_dim[2];
        stride_bz_input = input_stride[0];  // batch stride
        stride_d_input = input_stride[2];   // head_dim stride
        stride_h_input = input_stride[1];   // num_heads stride
        stride_bz_output = output_stride[0];
        stride_d_output = output_stride[2];
        stride_h_output = output_stride[1];
    }
    
    // Extract mean and scale strides - [batch, num_heads, head_dim]
    stride_bz_mean = mean_stride[0];   // batch stride
    stride_h_mean = mean_stride[1];    // num_heads stride
    stride_bz_scale = scale_stride[0]; // batch stride
    stride_h_scale = scale_stride[1];  // num_heads stride
    
    // Validate tensor dimensions
    if (output_dim[0] != input_dim[0] || output_dim[1] != input_dim[1] || 
        output_dim[2] != input_dim[2] || output_dim[3] != input_dim[3]) {
        fprintf(stderr, "ERROR: Input and output dimensions must match\n");
        return;
    }
    
    if (mean_dim[0] != batch_size || mean_dim[1] != num_heads || mean_dim[2] != head_dim) {
        fprintf(stderr, "ERROR: Mean dimensions mismatch: expected [%d, %d, %d], got [%d, %d, %d]\n",
               batch_size, num_heads, head_dim, mean_dim[0], mean_dim[1], mean_dim[2]);
        return;
    }
    
    if (scale_dim[0] != batch_size || scale_dim[1] != num_heads || scale_dim[2] != head_dim) {
        fprintf(stderr, "ERROR: Scale dimensions mismatch: expected [%d, %d, %d], got [%d, %d, %d]\n",
               batch_size, num_heads, head_dim, scale_dim[0], scale_dim[1], scale_dim[2]);
        return;
    }
    
    // Launch kernel configuration matching original mean_scale_fuse_quant_cuda
    constexpr int CTA_SIZE = 256;
    constexpr uint32_t pad_size = 64; // Match original implementation
    
    dim3 grid(num_heads, batch_size, head_dim);
    dim3 block(CTA_SIZE);
    
    // Launch MeanScaleKernel with template parameters matching original implementation
    // pad_size=64, sub_mean=true (for mean_scale_fuse_quant_cuda)
    MeanScaleKernel<pad_size, true, half><<<grid, block, 0, cuda_stream>>>(
        input,                    // Input tensor
        output,                   // Output tensor
        mean,                     // Mean tensor (used for mean_scale_fuse_quant)
        scale,                    // Scale tensor
        scale_max,                // Maximum scale value
        num_tokens,               // Actual number of tokens
        stride_bz_input, stride_d_input, stride_h_input,     // Input strides
        stride_bz_output, stride_d_output, stride_h_output,  // Output strides
        stride_bz_mean, stride_h_mean,                       // Mean strides
        stride_bz_scale, stride_h_scale                      // Scale strides
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_mean_scale_fuse_quant_cuda_direct kernel failed: %s\n", 
                cudaGetErrorString(error));
    }
}

extern "C" void ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct(
    half *query,              // Input Q tensor data (FP16)
    half *key,                // Input K tensor data (FP16) 
    half *k_mean,             // Optional K mean tensor data (FP16, can be NULL)
    half *value,              // Input V tensor data (FP16)
    int8_t *q_int8,           // Output Q quantized (INT8)
    int8_t *k_int8,           // Output K quantized (INT8)
    int8_t *v_fp8,            // Output V quantized (FP8)
    float *query_scale,       // Output Q scales (FP32)
    float *key_scale,         // Output K scales (FP32)
    float *value_scale,       // Output V scales (FP32)
    half *output,             // Output tensor data (FP16)
    int qdim[],               // Query tensor dimensions [batch, seq/heads, heads/seq, dim]
    int kdim[],               // Key tensor dimensions
    int vdim[],               // Value tensor dimensions
    int odim[],               // Output tensor dimensions
    int q_int8_dim[],         // Q quantized output dimensions
    int k_int8_dim[],         // K quantized output dimensions
    int v_fp8_dim[],          // V quantized output dimensions
    int query_scale_dim[],    // Q scale tensor dimensions
    int key_scale_dim[],      // K scale tensor dimensions
    int value_scale_dim[],    // V scale tensor dimensions
    int qstride[],            // Query tensor strides
    int kstride[],            // Key tensor strides
    int vstride[],            // Value tensor strides
    int ostride[],            // Output tensor strides
    int q_int8_stride[],      // Q quantized output strides
    int k_int8_stride[],      // K quantized output strides
    int v_fp8_stride[],       // V quantized output strides
    int query_scale_stride[], // Q scale tensor strides
    int key_scale_stride[],   // K scale tensor strides
    int value_scale_stride[], // V scale tensor strides
    int tensor_layout,        // 0=NHD, 1=HND
    int is_causal,            // Whether to apply causal masking
    int qk_quant_gran,        // Quantization granularity (2=per_warp for our case)
    float sm_scale,           // Softmax scale factor
    int return_lse,           // Whether to return log-sum-exp (0 for our case)
    int pv_accum_dtype,       // PV accumulation dtype (2=FP32+FP32 for our case)
    int km_dim[],             // K mean dimensions (can be NULL if k_mean is NULL)
    int km_stride[],          // K mean strides (can be NULL if k_mean is NULL)
    cudaStream_t cuda_stream)
{
    // Fixed block and warp sizes for FP8 version
    const int BLKQ = 128;
    const int WARPQ = 32;
    const int BLKK = 64;
    // Validate inputs
    assert(query && key && value && q_int8 && k_int8 && v_fp8 && 
           query_scale && key_scale && value_scale && output);
    
    // Only support specific configurations for now
    if (tensor_layout != 0) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct only supports NHD layout (tensor_layout=0)\n");
        return;
    }
    
    if (qk_quant_gran != 2) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct only supports per_warp quantization (qk_quant_gran=2)\n");
        return;
    }
    
    // if (pv_accum_dtype != 2) {
    //     fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct only supports fp32+fp32 accumulation (pv_accum_dtype=2)\n");
    //     return;
    // }
    
    // Step 1: Quantize Q and K tensors using per-warp quantization
    ccv_nnc_per_warp_int8_direct(
        query, key,                    // Input Q and K tensors
        q_int8, k_int8,               // Output quantized tensors
        query_scale, key_scale,       // Output scale tensors
        k_mean,                       // Optional K mean tensor (NULL for our case)
        qdim, kdim,                   // Input dimensions
        q_int8_dim, k_int8_dim,       // Output quantized dimensions
        query_scale_dim, key_scale_dim, // Scale dimensions
        qstride, kstride,             // Input strides
        q_int8_stride, k_int8_stride, // Output strides
        query_scale_stride, key_scale_stride, // Scale strides
        km_dim, km_stride,            // K mean dimensions and strides
        BLKQ, WARPQ, BLKK,           // Block and warp sizes
        tensor_layout,               // Tensor layout
        cuda_stream                  // CUDA stream
    );
    
    // Check for errors after Q/K quantization
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct Q/K quantization failed: %s\n",
                cudaGetErrorString(error));
        return;
    }
    
    // Step 2: Quantize V tensor to FP8 using per-channel quantization
    // For NHD layout, V is [B, S, H, D] and needs to be transformed to [B, D, H, padded_S]
    // The ccv_nnc_scale_fuse_quant_cuda_direct function handles this transformation
    
    // Extract actual sequence length from vdim
    int batch_size = vdim[0];
    int seq_len = vdim[1];     // For NHD layout
    int num_heads = vdim[2];   // For NHD layout
    int head_dim = vdim[3];
    
    // V FP8 dimensions should be [B, D, H, padded_S] for NHD layout
    // This is handled by the scale_fuse_quant function which does transpose_pad_permute
    
    const float scale_max = 448.0f;  // E4M3 max scale for FP8
    
    ccv_nnc_scale_fuse_quant_cuda_direct(
        (half*)value,                // Input V tensor (FP16)
        v_fp8,                       // Output V quantized (FP8)
        value_scale,                 // Output V scales (FP32)
        vdim,                        // V input dimensions [B, S, H, D] for NHD
        v_fp8_dim,                   // V output dimensions [B, D, H, padded_S] for NHD
        value_scale_dim,             // V scale dimensions [B, H, D]
        vstride,                     // V input strides
        v_fp8_stride,                // V output strides
        value_scale_stride,          // V scale strides
        seq_len,                     // num_tokens (actual sequence length)
        scale_max,                   // Maximum scale value for FP8
        tensor_layout,               // Tensor layout (0=NHD)
        cuda_stream                  // CUDA stream
    );
    
    // Check for errors after V quantization
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct V quantization failed: %s\n",
                cudaGetErrorString(error));
        return;
    }
    
    // Step 3: Call the FP8 attention kernel with fp32+fp32 accumulation
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
        q_int8,                      // Quantized Q (INT8)
        k_int8,                      // Quantized K (INT8)
        v_fp8,                       // Quantized V (FP8)
        output,                      // Output tensor (FP16)
        query_scale,                 // Q scales
        key_scale,                   // K scales
        value_scale,                 // V scales
        q_int8_dim,                  // Q dimensions
        k_int8_dim,                  // K dimensions
        v_fp8_dim,                   // V dimensions
        odim,                        // Output dimensions
        query_scale_dim,             // Q scale dimensions
        key_scale_dim,               // K scale dimensions
        value_scale_dim,             // V scale dimensions
        q_int8_stride,               // Q strides
        k_int8_stride,               // K strides
        v_fp8_stride,                // V strides
        ostride,                     // Output strides
        query_scale_stride,          // Q scale strides
        key_scale_stride,            // K scale strides
        value_scale_stride,          // V scale strides
        tensor_layout,               // Tensor layout (0=NHD)
        sm_scale                     // Softmax scale
    );
    
    // Check for errors after attention kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct attention kernel failed: %s\n",
                cudaGetErrorString(error));
    }
}