extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#include "sage_attn_utils.cuh"
#include "math.cuh"
#include "qattn/attn_utils.cuh"
#include "qattn/qk_int_sv_f16_cuda_sm80_kernel_only.cuh"
#include "fused.h"

#include <cuda_fp16.h>
#include <algorithm>

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
    
    // Print debug info
    printf("\n=== ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct ===\n");
    printf("Dimensions: B=%d, H=%d, S=%d, D=%d, H_kv=%d, S_kv=%d\n", 
           batch_size, num_heads, seq_len, head_dim, num_kv_heads, kv_len);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
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
        
        printf("Grid: (%d, %d, %d), Block: (%d, %d)\n", 
               grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y);
        printf("Template params: CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d, HEAD_DIM=%d\n", 
               CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
        printf("Shared memory: %zu bytes\n", smem_max);
        printf("Accumulation: FP32, use_inst_buffer: false\n");
        
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
    
    // Print debug info
    printf("\n=== qk_int8_sv_f16_accum_f16_attn_inst_buf_direct ===\n");
    printf("Dimensions: B=%d, H=%d, S=%d, D=%d, H_kv=%d, S_kv=%d\n", 
           batch_size, num_heads, seq_len, head_dim, num_kv_heads, kv_len);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
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
        
        printf("Grid: (%d, %d, %d), Block: (%d, %d)\n", 
               grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y);
        printf("Template params: CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d, HEAD_DIM=%d\n", 
               CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
        printf("Shared memory: %zu bytes\n", smem_max);
        
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
    
    printf("\n=== ccv_nnc_quant_per_warp_int8_cuda_direct ===\n");
    printf("Block size: %d, Warp block size: %d\n", block_size, warp_block_size);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
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
    
    printf("Dimensions: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    
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
            
            printf("Grid: (%d, %d, %d), Block: (%d, 1)\n", 
                   grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x);
            printf("Using QuantInt8Kernel<HEAD_DIM=%d, WARP_BLOCK_SIZE=%d>\n", HEAD_DIM, WARP_BLOCK_SIZE);
            printf("Input strides: bz=%d, seq=%d, h=%d\n", stride_bz_input, stride_seq_input, stride_h_input);
            printf("Output strides: bz=%d, seq=%d, h=%d\n", stride_bz_output, stride_seq_output, stride_h_output);
            printf("Scale strides: bz=%d, h=%d\n", stride_scale_bz, stride_scale_h);
            
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
    
    printf("\n=== ccv_nnc_quant_per_block_int8_cuda_direct ===\n");
    printf("Block size: %d\n", block_size);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
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
    
    printf("Dimensions: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    
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
            
            printf("Grid: (%d, %d, %d), Block: (%d, 1)\n", 
                   grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x);
            printf("Using QuantInt8Kernel<HEAD_DIM=%d, BLOCK_SIZE=%d>\n", HEAD_DIM, BLOCK_SIZE);
            printf("Scale strides: bz=%d, h=%d\n", stride_scale_bz, stride_scale_h);
            
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
    
    printf("\n=== ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda_direct ===\n");
    printf("Block size: %d\n", block_size);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
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
    
    printf("Dimensions: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    
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
            
            printf("Grid: (%d, %d, %d), Block: (%d, 1)\n", 
                   grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x);
            printf("Using QuantInt8Kernel<HEAD_DIM=%d, BLOCK_SIZE=%d> with mean subtraction\n", HEAD_DIM, BLOCK_SIZE);
            printf("Scale strides: bz=%d, h=%d\n", stride_scale_bz, stride_scale_h);
            
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
    
    printf("\n=== ccv_nnc_per_warp_int8_direct ===\n");
    printf("Block sizes: Q=%d (warp=%d), K=%d\n", BLKQ, WARPQ, BLKK);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
    // Quantize Q using per-warp quantization
    printf("Quantizing Q (per-warp)...\n");
    ccv_nnc_quant_per_warp_int8_cuda_direct(
        q, q_int8, q_scale,
        qdim, q_int8_dim, q_scale_dim,
        qstride, q_int8_stride, q_scale_stride,
        BLKQ, WARPQ, tensor_layout, cuda_stream
    );
    
    // Quantize K using per-block quantization (with or without mean subtraction)
    printf("Quantizing K (per-block)...\n");
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
    
    printf("✓ ccv_nnc_per_warp_int8_direct completed\n");
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
    
    printf("\n=== ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct ===\n");
    printf("Block sizes: Q=%d (warp=%d), K=%d\n", BLKQ, WARPQ, BLKK);
    printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    printf("Causal: %s, QK quant gran: %d, SM scale: %f\n", is_causal ? "true" : "false", qk_quant_gran, sm_scale);
    printf("PV accum dtype: %d (%s)\n", pv_accum_dtype, 
           pv_accum_dtype == 0 ? "FP16" : (pv_accum_dtype == 1 ? "FP16_MIX_FP32" : "FP32"));
    
    // Step 1: Quantize Q and K tensors using per-warp quantization
    printf("Step 1: Quantizing Q and K tensors...\n");
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
    
    // Debug: Print quantized outputs after quantization
    printf("\n=== After Quantization (Debug) ===\n");
    
    // Copy quantized data back to CPU for inspection
    int total_elements = qdim[0] * qdim[1] * qdim[2] * qdim[3];
    int8_t* temp_q_int8 = (int8_t*)malloc(total_elements * sizeof(int8_t));
    int8_t* temp_k_int8 = (int8_t*)malloc(total_elements * sizeof(int8_t));
    
    cudaMemcpy(temp_q_int8, q_int8, total_elements * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_k_int8, k_int8, total_elements * sizeof(int8_t), cudaMemcpyDeviceToHost);
    
    printf("Q_int8 first 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", temp_q_int8[i]);
    }
    printf("\n");
    
    printf("K_int8 first 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", temp_k_int8[i]);
    }
    printf("\n");
    
    // Copy scales back to CPU
    int q_scale_total = query_scale_dim[0] * query_scale_dim[1] * query_scale_dim[2];
    int k_scale_total = key_scale_dim[0] * key_scale_dim[1] * key_scale_dim[2];
    float* temp_q_scale = (float*)malloc(q_scale_total * sizeof(float));
    float* temp_k_scale = (float*)malloc(k_scale_total * sizeof(float));
    
    cudaMemcpy(temp_q_scale, query_scale, q_scale_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_k_scale, key_scale, k_scale_total * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Q_scale dimensions: [%d, %d, %d]\n", 
           query_scale_dim[0], query_scale_dim[1], query_scale_dim[2]);
    printf("Q_scale all %d values: ", q_scale_total);
    for (int i = 0; i < q_scale_total; i++) {
        printf("%.6f ", temp_q_scale[i]);
    }
    printf("\n");
    
    printf("K_scale dimensions: [%d, %d, %d]\n", 
           key_scale_dim[0], key_scale_dim[1], key_scale_dim[2]);
    printf("K_scale all %d values: ", k_scale_total);
    for (int i = 0; i < k_scale_total; i++) {
        printf("%.6f ", temp_k_scale[i]);
    }
    printf("\n");
    
    // Print scale strides for debugging
    printf("Q_scale strides passed to attention: [%d, %d, %d, %d]\n", 
           query_scale_stride[0], query_scale_stride[1], query_scale_stride[2], 
           (query_scale_stride[3] ? query_scale_stride[3] : -1));
    printf("K_scale strides passed to attention: [%d, %d, %d, %d]\n", 
           key_scale_stride[0], key_scale_stride[1], key_scale_stride[2],
           (key_scale_stride[3] ? key_scale_stride[3] : -1));
    
    free(temp_q_int8);
    free(temp_k_int8);
    free(temp_q_scale);
    free(temp_k_scale);
    
    printf("=== End Debug ===\n\n");
    
    // Step 2: Perform quantized attention computation
    printf("Step 2: Computing quantized attention...\n");
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
        printf("ERROR: Unsupported pv_accum_dtype: %d\n", pv_accum_dtype);
        return;
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct failed: %s\n", 
               cudaGetErrorString(error));
        return;
    }
    
    printf("✓ ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct completed successfully\n");
}