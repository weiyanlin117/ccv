extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <algorithm>  // for std::max
#include <vector>     // for std::vector

#ifdef HAVE_CUDA_SM80
#include "math.cuh"
#include "qattn/attn_utils.cuh"
#include "fused.h"
#include "qattn/qk_int_sv_f16_cuda_sm80_kernel_only.cuh"
#include "sage_attn_utils.cuh"

// Direct kernel wrapper for raw kernel testing (bypasses all CCV wrappers)
extern "C" void call_sage_attention_kernel_direct(
    int8_t *Q, int8_t *K, half *V, half *O, float *Lse,
    float *Q_scale, float *K_scale, half *V_mean,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups,
    uint32_t stride_bz_q, uint32_t stride_seq_q, uint32_t stride_h_q,
    uint32_t stride_bz_k, uint32_t stride_seq_k, uint32_t stride_h_k,
    uint32_t stride_bz_v, uint32_t stride_seq_v, uint32_t stride_h_v,
    uint32_t stride_bz_o, uint32_t stride_seq_o, uint32_t stride_h_o,
    float sm_scale, int grid_x, int grid_y, int grid_z, int block_size)
{
    dim3 grid_dim(grid_x, grid_y, grid_z);
    // PyTorch uses (32, 4) block dimensions, not (128, 1)  
    dim3 block_dim(32, 4);
    
    // Use PyTorch's exact template parameters (now available with new instantiation)
    // PyTorch: CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64, HEAD_DIM=128
    constexpr uint32_t CTA_Q = 128, CTA_K = 64;
    constexpr uint32_t WARP_Q = 32, WARP_K = 64;
    constexpr uint32_t HEAD_DIM = 128;
    
    // Calculate shared memory requirement (exactly as PyTorch does)
    // smem_Q + smem_K + smem_V vs smem_O
    size_t smem_max = std::max(
        CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half),
        CTA_Q * HEAD_DIM * sizeof(half)
    );
    
    // Debug template parameters before kernel instantiation
    printf("\n=== CCV Kernel Template Parameters ===\n");
    printf("CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d, HEAD_DIM=%d\n", CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
    printf("DataType: kInt8, Q_GRAN=%d, K_GRAN=%d\n", static_cast<int>(QuantGranularity::kPerWarp), static_cast<int>(QuantGranularity::kPerWarp));
    printf("DTypeSVAccum: float, use_inst_buffer: false, DTypeOut: half\n");
    printf("ComputeUnit: kTensorCore, MaskMode: %d, RETURN_LSE: false\n", static_cast<int>(MaskMode::kNone));
    printf("smem_max: %zu bytes\n", smem_max);
    
    printf("\n=== CCV Kernel Input Parameters (in call order) ===\n");
    printf("1. Q (int8_t*): %p\n", Q);
    printf("2. K (int8_t*): %p\n", K);
    printf("3. V (half*): %p\n", V);
    printf("4. O (half*): %p\n", O);
    printf("5. Lse (float*): %p\n", Lse);
    printf("6. Q_scale (float*): %p\n", Q_scale);
    printf("7. K_scale (float*): %p\n", K_scale);
    printf("8. V_mean (half*): %p\n", V_mean);
    printf("9. qo_len: %u\n", qo_len);
    printf("10. kv_len: %u\n", kv_len);
    printf("11. num_kv_groups: %u\n", num_kv_groups);
    printf("12-14. Q strides: bz=%u, seq=%u, h=%u\n", stride_bz_q, stride_seq_q, stride_h_q);
    printf("15-17. K strides: bz=%u, seq=%u, h=%u\n", stride_bz_k, stride_seq_k, stride_h_k);
    printf("18-20. V strides: bz=%u, seq=%u, h=%u\n", stride_bz_v, stride_seq_v, stride_h_v);
    printf("21-23. O strides: bz=%u, seq=%u, h=%u\n", stride_bz_o, stride_seq_o, stride_h_o);
    printf("24. sm_scale: %f\n", sm_scale);
    printf("25-28. grid: x=%d, y=%d, z=%d, block_size=%d\n", grid_x, grid_y, grid_z, block_size);
    fflush(stdout);
    
    // Launch kernel with shared memory (exactly as PyTorch does)
    qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
        float, false, half, ComputeUnit::kTensorCore, MaskMode::kNone, false, false>
        <<<grid_dim, block_dim, smem_max>>>(
        Q, K, V, O, Lse, Q_scale, K_scale, V_mean,
        qo_len, kv_len, num_kv_groups,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_k, stride_seq_k, stride_h_k,
        stride_bz_v, stride_seq_v, stride_h_v,
        stride_bz_o, stride_seq_o, stride_h_o,
        sm_scale
    );
}

// CCV wrapper function matching PyTorch API (CCV style with output parameter)
// Matches PyTorch qk_int8_sv_f16_accum_f32_attn with FP32 accumulation
extern "C" int ccv_nnc_qk_int8_sv_f16_accum_f32_attn(
    ccv_nnc_tensor_t* const query,
    ccv_nnc_tensor_t* const key,
    ccv_nnc_tensor_t* const value,
    ccv_nnc_tensor_t* const output,
    ccv_nnc_tensor_t* const query_scale,
    ccv_nnc_tensor_t* const key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse)
{
    // Validate input parameters
    if (!query || !key || !value || !output || !query_scale || !key_scale) {
        fprintf(stderr, "ERROR: ccv_nnc_qk_int8_sv_f16_accum_f32_attn: NULL tensor input\n");
        return CCV_NNC_EXEC_INVALID;
    }
    
    // Extract tensor views and ensure contiguous layout (CCV best practice)
    ccv_nnc_tensor_view_t* q_view = (ccv_nnc_tensor_view_t*)query;
    ccv_nnc_tensor_view_t* k_view = (ccv_nnc_tensor_view_t*)key;
    ccv_nnc_tensor_view_t* v_view = (ccv_nnc_tensor_view_t*)value;
    ccv_nnc_tensor_view_t* o_view = (ccv_nnc_tensor_view_t*)output;
    ccv_nnc_tensor_view_t* q_scale_view = (ccv_nnc_tensor_view_t*)query_scale;
    ccv_nnc_tensor_view_t* k_scale_view = (ccv_nnc_tensor_view_t*)key_scale;
    
    // Ensure tensors are contiguous (following CCV patterns)
    assert(CCV_IS_TENSOR_CONTIGUOUS(q_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(k_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(v_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(o_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(q_scale_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(k_scale_view));
    
    // Get tensor dimensions
    int qdim[CCV_NNC_MAX_DIM_ALLOC];
    int kdim[CCV_NNC_MAX_DIM_ALLOC];
    int vdim[CCV_NNC_MAX_DIM_ALLOC];
    int odim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(q_view, qdim);
    ccv_nnc_tensor_view_get_dim(k_view, kdim);
    ccv_nnc_tensor_view_get_dim(v_view, vdim);
    ccv_nnc_tensor_view_get_dim(o_view, odim);
    
    // Extract dimensions based on tensor layout
    int batch_size, num_heads, qo_len, kv_len, head_dim;
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        batch_size = qdim[0];
        num_heads = qdim[1]; 
        qo_len = qdim[2];
        kv_len = kdim[2];
        head_dim = qdim[3];
    } else { // NHD layout (tensor_layout == 0): [batch, seq, heads, dim]
        batch_size = qdim[0];
        qo_len = qdim[1];
        kv_len = kdim[1];
        num_heads = qdim[2];
        head_dim = qdim[3];
    }
    
    // Validate tensor compatibility
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
        return CCV_NNC_EXEC_INVALID;
    }
    
    // Debug parameter information
    printf("\n=== CCV SageAttention Wrapper ===\n");
    printf("Input parameters:\n");
    printf("  tensor_layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    printf("  is_causal: %d\n", is_causal);
    printf("  qk_quant_gran: %d (%s)\n", qk_quant_gran, qk_quant_gran == 2 ? "per_warp" : "per_block");
    printf("  sm_scale: %f\n", sm_scale);
    printf("  return_lse: %d\n", return_lse);
    printf("Tensor dimensions:\n");
    printf("  Query: [%d, %d, %d, %d], dtype: %d\n", qdim[0], qdim[1], qdim[2], qdim[3], q_view->info.datatype);
    printf("  Key: [%d, %d, %d, %d], dtype: %d\n", kdim[0], kdim[1], kdim[2], kdim[3], k_view->info.datatype);
    printf("  Value: [%d, %d, %d, %d], dtype: %d\n", vdim[0], vdim[1], vdim[2], vdim[3], v_view->info.datatype);
    printf("  Output: [%d, %d, %d, %d], dtype: %d\n", odim[0], odim[1], odim[2], odim[3], o_view->info.datatype);
    printf("Extracted dimensions: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, qo_len, head_dim);
    
    // Get actual tensor strides (like PyTorch does, following CCV GPU patterns)
    int qstride[CCV_NNC_MAX_DIM_ALLOC];
    int kstride[CCV_NNC_MAX_DIM_ALLOC];
    int vstride[CCV_NNC_MAX_DIM_ALLOC];
    int ostride[CCV_NNC_MAX_DIM_ALLOC];
    int qscale_stride[CCV_NNC_MAX_DIM_ALLOC];
    int kscale_stride[CCV_NNC_MAX_DIM_ALLOC];
    
    ccv_nnc_tensor_view_get_stride(q_view, qstride);
    ccv_nnc_tensor_view_get_stride(k_view, kstride);
    ccv_nnc_tensor_view_get_stride(v_view, vstride);
    ccv_nnc_tensor_view_get_stride(o_view, ostride);
    ccv_nnc_tensor_view_get_stride(q_scale_view, qscale_stride);
    ccv_nnc_tensor_view_get_stride(k_scale_view, kscale_stride);
    
    // DEBUG: Compare get_stride vs direct stride access
    printf("=== STRIDE DEBUG COMPARISON ===\n");
    printf("Q: get_stride=[%d,%d,%d,%d] vs direct=[%d,%d,%d,%d]\n",
           qstride[0], qstride[1], qstride[2], qstride[3],
           q_view->stride[0], q_view->stride[1], q_view->stride[2], q_view->stride[3]);
    printf("K: get_stride=[%d,%d,%d,%d] vs direct=[%d,%d,%d,%d]\n",
           kstride[0], kstride[1], kstride[2], kstride[3],
           k_view->stride[0], k_view->stride[1], k_view->stride[2], k_view->stride[3]);
    printf("V: get_stride=[%d,%d,%d,%d] vs direct=[%d,%d,%d,%d]\n",
           vstride[0], vstride[1], vstride[2], vstride[3],
           v_view->stride[0], v_view->stride[1], v_view->stride[2], v_view->stride[3]);
    printf("O: get_stride=[%d,%d,%d,%d] vs direct=[%d,%d,%d,%d]\n",
           ostride[0], ostride[1], ostride[2], ostride[3],
           o_view->stride[0], o_view->stride[1], o_view->stride[2], o_view->stride[3]);
    printf("=== END STRIDE DEBUG ===\n");
    fflush(stdout);
    
    // Extract strides based on tensor layout (following PyTorch approach)
    uint32_t stride_bz_q, stride_seq_q, stride_h_q;
    uint32_t stride_bz_k, stride_seq_k, stride_h_k;
    uint32_t stride_bz_v, stride_seq_v, stride_h_v;
    uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
    if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
        // Use actual tensor strides (like PyTorch tensor.stride())
        stride_bz_q = qstride[0];  // batch stride
        stride_h_q = qstride[1];   // head stride  
        stride_seq_q = qstride[2]; // seq stride
        
        stride_bz_k = kstride[0];
        stride_h_k = kstride[1];
        stride_seq_k = kstride[2];
        
        stride_bz_v = vstride[0];
        stride_h_v = vstride[1];
        stride_seq_v = vstride[2];
        
        stride_bz_o = ostride[0];
        stride_h_o = ostride[1];
        stride_seq_o = ostride[2];
    } else { // NHD layout: [batch, seq, heads, dim]
        // Use actual tensor strides (like PyTorch tensor.stride())
        stride_bz_q = qstride[0];  // batch stride
        stride_seq_q = qstride[1]; // seq stride
        stride_h_q = qstride[2];   // head stride
        
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
    
    printf("Calculated strides:\n");
    printf("  Q strides: bz=%u, seq=%u, h=%u\n", stride_bz_q, stride_seq_q, stride_h_q);
    printf("  K strides: bz=%u, seq=%u, h=%u\n", stride_bz_k, stride_seq_k, stride_h_k);
    
    // Use the exact same kernel template parameters as call_sage_attention_kernel_direct
    // PyTorch uses: CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64, HEAD_DIM=128
    constexpr uint32_t CTA_Q = 128, CTA_K = 64;
    constexpr uint32_t WARP_Q = 32, WARP_K = 64;
    // Calculate shared memory requirement (exactly as PyTorch does)
    // size_t smem_max = std::max(
    //     CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(half),
    //     CTA_Q * head_dim * sizeof(half)
    // );
    
    // Grid and block dimensions matching call_sage_attention_kernel_direct
    // dim3 grid_dim((qo_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
    // dim3 block_dim(32, 4);  // PyTorch uses (32, 4) block dimensions
    const int num_kv_groups = 1; // num_qo_heads / num_kv_heads;

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                    constexpr int CTA_Q = 128;
                    constexpr int CTA_K = 64;
                    constexpr int WARP_Q = 32;
                    constexpr int WARP_K = 64;

                    constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                    //                                     smem_Q                                     smem_K                            smem_V                     smem_O
                    size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half), CTA_Q * HEAD_DIM * sizeof(half));
                    
                    // Debug template parameters before kernel instantiation
                    printf("\n=== PyTorch Kernel Template Parameters ===\n");
                    printf("CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d, HEAD_DIM=%d\n", CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
                    printf("DTypeSVAccum: float, use_inst_buffer: false, DTypeOut: half\n");
                    printf("ComputeUnit: kTensorCore, MaskMode: %d, RETURN_LSE: %s\n", static_cast<int>(mask_mode),  "false");
                    printf("smem_max: %zu bytes\n", smem_max);
                    
                    // auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), float, false, half, ComputeUnit::kTensorCore, 
                    //                                             mask_mode, RETURN_LSE, false>;

                    // cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

                    dim3 grid_dim(div_ceil(qo_len, CTA_Q), num_heads, batch_size);
                    dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                     //     // DEBUG: Print all input and template parameters before kernel launch (HND layout)
        printf("=== KERNEL LAUNCH DEBUG INFO (HND layout) ===\n");
        printf("Input Parameters:\n");
        printf("  batch_size=%d, num_heads=%d, qo_len=%d, head_dim=%d\n", batch_size, num_heads, qo_len, head_dim);
        printf("  tensor_layout=%d (HND)\n", tensor_layout);
        printf("  q_view->data.u8=%p, k_view->data.u8=%p\n", q_view->data.u8, k_view->data.u8);
        printf("  v_view->data.f16=%p, o_view->data.f16=%p\n", v_view->data.f16, o_view->data.f16);
        printf("  q_scale_view->data.f32=%p, k_scale_view->data.f32=%p\n", q_scale_view->data.f32, k_scale_view->data.f32);
        printf("Template Parameters:\n");
        printf("  CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d\n", CTA_Q, CTA_K, WARP_Q, WARP_K);
        printf("  head_dim=%d (template param)\n", head_dim);
        printf("  DataType::kInt8, QuantGranularity::kPerWarp (Q and K)\n");
        printf("  AccumDataType=float, kSplitKV=false, OutputDataType=half\n");
        printf("  ComputeUnit::kTensorCore, MaskMode::kNone\n");
        printf("  kUseP2PCP=false, kInterleaved=false\n");
        printf("Grid/Block Configuration:\n");
        printf("  grid_dim=(%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
        printf("  block_dim=(%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
        printf("  smem_max=%zu bytes\n", smem_max);
        printf("Tensor Shapes and Strides:\n");
        printf("  q_shape=[%d,%d,%d,%d], q_stride=[%d,%d,%d,%d]\n", 
               q_view->info.dim[0], q_view->info.dim[1], q_view->info.dim[2], q_view->info.dim[3],
               qstride[0], qstride[1], qstride[2], qstride[3]);
        printf("  k_shape=[%d,%d,%d,%d], k_stride=[%d,%d,%d,%d]\n", 
               k_view->info.dim[0], k_view->info.dim[1], k_view->info.dim[2], k_view->info.dim[3],
               kstride[0], kstride[1], kstride[2], kstride[3]);
        printf("  v_shape=[%d,%d,%d,%d], v_stride=[%d,%d,%d,%d]\n", 
               v_view->info.dim[0], v_view->info.dim[1], v_view->info.dim[2], v_view->info.dim[3],
               vstride[0], vstride[1], vstride[2], vstride[3]);
        printf("  o_shape=[%d,%d,%d,%d], o_stride=[%d,%d,%d,%d]\n", 
               o_view->info.dim[0], o_view->info.dim[1], o_view->info.dim[2], o_view->info.dim[3],
               ostride[0], ostride[1], ostride[2], ostride[3]);
        printf("Scale Tensor Info:\n");
        printf("  q_scale_shape=[%d,%d,%d], k_scale_shape=[%d,%d,%d]\n",
               q_scale_view->info.dim[0], q_scale_view->info.dim[1], q_scale_view->info.dim[2],
               k_scale_view->info.dim[0], k_scale_view->info.dim[1], k_scale_view->info.dim[2]);
        printf("Stride Parameters (going to kernel):\n");
        printf("  stride_bz_q=%d, stride_seq_q=%d, stride_h_q=%d\n", stride_bz_q, stride_seq_q, stride_h_q);
        printf("  stride_bz_k=%d, stride_seq_k=%d, stride_h_k=%d\n", stride_bz_k, stride_seq_k, stride_h_k);
        printf("  stride_bz_v=%d, stride_seq_v=%d, stride_h_v=%d\n", stride_bz_v, stride_seq_v, stride_h_v);
        printf("  stride_bz_o=%d, stride_seq_o=%d, stride_h_o=%d\n", stride_bz_o, stride_seq_o, stride_h_o);
        printf("  sm_scale=%f\n", sm_scale);
        printf("=== END DEBUG INFO ===\n");
                    
                    qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
                    float, false, half, ComputeUnit::kTensorCore, mask_mode, false, false>
                    <<<grid_dim, block_dim, smem_max>>>(
                    (int8_t*)q_view->data.u8,   // Q (int8)
                    (int8_t*)k_view->data.u8,   // K (int8)
                    (half*)v_view->data.f16,     // V (fp16)
                    (half*)o_view->data.f16,     // O (fp16)
                    NULL,                       // Lse (not used)
                    (float*)q_scale_view->data.f32,  // Q_scale
                    (float*)k_scale_view->data.f32,  // K_scale
                    NULL,                       // V_mean (not used)
                    qo_len,                    // qo_len
                    kv_len,                    // kv_len (assuming self-attention)
                    num_kv_groups,                          // num_kv_groups
                    stride_bz_q, stride_seq_q, stride_h_q,  // Fixed order: batch, seq, head (matches direct function)
                    stride_bz_k, stride_seq_k, stride_h_k,  // Fixed order: batch, seq, head (matches direct function)
                    stride_bz_v, stride_seq_v, stride_h_v,  // Fixed order: batch, seq, head (matches direct function)
                    stride_bz_o, stride_seq_o, stride_h_o,  // Fixed order: batch, seq, head (matches direct function)
                    sm_scale
                    );
                    
            });
        });
    });
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: CCV SageAttention kernel failed: %s\n", cudaGetErrorString(error));
        return CCV_NNC_EXEC_INVALID;
    }
    
    // Synchronize and check for execution errors
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: CCV SageAttention kernel execution failed: %s\n", cudaGetErrorString(error));
        return CCV_NNC_EXEC_INVALID;
    }
    
    printf("✅ CCV SageAttention wrapper completed successfully\n");
    
    // Return success status (CCV style)
    return CCV_NNC_EXEC_SUCCESS;
}

// CCV wrapper for per-warp int8 quantization - direct migration from PyTorch quant_per_warp_int8_cuda
extern "C" int ccv_nnc_quant_per_warp_int8_cuda(
    ccv_nnc_tensor_t* const input,   // Input tensor (FP16)
    ccv_nnc_tensor_t* const output,  // Output tensor (INT8)
    ccv_nnc_tensor_t* const scale,   // Scale tensor (FP32)
    int block_size,                  // Block size (e.g., 128)
    int warp_block_size,             // Warp block size (e.g., 32)
    int tensor_layout,               // 0=NHD, 1=HND
    ccv_nnc_stream_context_t* const stream_context)
{
    // Validate inputs
    if (!input || !output || !scale) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_warp_int8_cuda: NULL tensor input\n");
        return CCV_NNC_EXEC_INVALID;
    }
    assert(block_size == 128 || block_size == 64);
    assert(warp_block_size == 16 || warp_block_size == 32);

    // Get tensor views
    ccv_nnc_tensor_view_t* input_view = (ccv_nnc_tensor_view_t*)input;
    ccv_nnc_tensor_view_t* output_view = (ccv_nnc_tensor_view_t*)output;
    ccv_nnc_tensor_view_t* scale_view = (ccv_nnc_tensor_view_t*)scale;
    
    // Check tensor properties
    assert(CCV_IS_TENSOR_CONTIGUOUS(input_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(output_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(scale_view));
    
    // Check data types
    assert(input_view->info.datatype == CCV_16F);
    assert(output_view->info.datatype == CCV_8U);  // INT8 (stored as 8U but treated as signed)
    assert(scale_view->info.datatype == CCV_32F);
    
    // Get dimensions
    int idim[CCV_NNC_MAX_DIM_ALLOC];
    int odim[CCV_NNC_MAX_DIM_ALLOC];
    int sdim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(input_view, idim);
    ccv_nnc_tensor_view_get_dim(output_view, odim);
    ccv_nnc_tensor_view_get_dim(scale_view, sdim);
    
    // Check dimensions
    const int input_nd = ccv_nnc_tensor_nd(input_view->info.dim);
    const int output_nd = ccv_nnc_tensor_nd(output_view->info.dim);
    const int scale_nd = ccv_nnc_tensor_nd(scale_view->info.dim);
    assert(input_nd == 4);
    assert(output_nd == 4);
    assert(scale_nd == 3);
    
    // Verify output shape matches input
    assert(odim[0] == idim[0] && odim[1] == idim[1] && odim[2] == idim[2] && odim[3] == idim[3]);
    
    const int batch_size = idim[0];
    const int head_dim = idim[3];
    assert(head_dim == 64 || head_dim == 128);

    // Get strides
    int istride[CCV_NNC_MAX_DIM_ALLOC];
    int ostride[CCV_NNC_MAX_DIM_ALLOC];
    int sstride[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_stride(input_view, istride);
    ccv_nnc_tensor_view_get_stride(output_view, ostride);
    ccv_nnc_tensor_view_get_stride(scale_view, sstride);
    
    uint32_t stride_bz_input = istride[0];
    uint32_t stride_bz_output = ostride[0];
    
    int num_tokens, num_heads;
    uint32_t stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
    
    if (tensor_layout == 0) { // NHD: [batch, seq, heads, dim]
        num_tokens = idim[1];
        num_heads = idim[2];
        stride_seq_input = istride[1];
        stride_h_input = istride[2];
        stride_seq_output = ostride[1];
        stride_h_output = ostride[2];
    } else { // HND: [batch, heads, seq, dim]
        num_tokens = idim[2];
        num_heads = idim[1];
        stride_seq_input = istride[2];
        stride_h_input = istride[1];
        stride_seq_output = ostride[2];
        stride_h_output = ostride[1];
    }
    
    // Verify scale shape
    // printf("DEBUG: Input tensor dimensions: [%d, %d, %d, %d]\n", idim[0], idim[1], idim[2], idim[3]);
    // printf("DEBUG: tensor_layout=%d, num_tokens=%d, num_heads=%d\n", tensor_layout, num_tokens, num_heads);
    // printf("DEBUG: Scale tensor dimensions: [%d, %d, %d]\n", sdim[1], sdim[2], sdim[3]);
    // printf("DEBUG: Expected dimensions: batch_size=%d, num_heads=%d, scale_blocks=%d\n", 
    //        batch_size, num_heads, ((num_tokens + block_size - 1) / block_size) * (block_size / warp_block_size));
	// printf("DEBUG: Expected dimensions: num_tokens=%d, block_size=%d, warp_block_size=%d\n", 
    //        num_tokens, block_size, warp_block_size);
	// printf("DEBUG: actual dimensions: batch_size=%d, num_heads=%d, scale_blocks=%d\n", 
    //        sdim[1], sdim[2], sdim[3]);
    
	// Verify output shape matches input
    assert(odim[0] == idim[0] && odim[1] == idim[1] && odim[2] == idim[2] && odim[3] == idim[3]);
	// check scale output SHAPE
    assert(sdim[1] == batch_size);
    assert(sdim[2] == num_heads);
    assert(sdim[3] == ((num_tokens + block_size - 1) / block_size) * (block_size / warp_block_size));
    
    // Get CUDA stream
    cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
    
    // Calculate grid and block dimensions (matching PyTorch)
    // dim3 grid((num_tokens + block_size - 1) / block_size * (block_size / warp_block_size), num_heads, batch_size);
    
    // constexpr uint32_t num_pack_per_thread = 1; // For warp_block_size=32 and head_dim=128: (32*16)/1024 = 1
    // dim3 block(warp_block_size * (head_dim / 8) / num_pack_per_thread);
    
    // printf("DEBUG: Grid dimensions: x=%d, y=%d, z=%d\n", grid.x, grid.y, grid.z);
    // printf("DEBUG: Block dimensions: x=%d\n", block.x);
    // printf("DEBUG: Scale strides: sstride[0]=%d, sstride[1]=%d, sstride[2]=%d (using [1]=%d, [2]=%d)\n", 
    //        sstride[0], sstride[1], sstride[2], sstride[1], sstride[2]);
    // printf("DEBUG:  head_dim=%d, warp_block_size=%d, block_size=%d\n", head_dim, warp_block_size, block_size);
    // Launch kernel based on parameters
	DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        //   CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        //   CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));

          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);

          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

		  QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, half><<<grid, block, 0, stream>>>(
                (half*)input_view->data.f16,
                nullptr,  // mean (not used)
                (int8_t*)output_view->data.u8,
                (float*)scale_view->data.f32,
                0.0f,  // sm_scale (not used in per_warp_int8)
                num_tokens,
                stride_bz_input, stride_seq_input, stride_h_input,
                0, 0,  // mean strides (not used)
                stride_bz_output, stride_seq_output, stride_h_output,
                sstride[1], sstride[2]
            );
        });
      });
    });
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_warp_int8_cuda kernel launch failed: %s\n", 
                cudaGetErrorString(error));
        return CCV_NNC_EXEC_INVALID;
    }
    
    return CCV_NNC_EXEC_SUCCESS;
}


// // Direct kernel wrapper for qk_int8_sv_f16_accum_f16_attn with instruction buffer (no CCV dependencies)
// extern "C" void qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
//     int8_t *Q, int8_t *K, half *V, half *O,
//     float *Q_scale, float *K_scale,
//     int qdim[], int kdim[], int vdim[], int odim[], 
//     int qscale_dim[], int kscale_dim[],
//     int qstride[], int kstride[], int vstride[], int ostride[],
//     int qscale_stride[], int kscale_stride[],
//     int tensor_layout,
//     int is_causal,
//     int qk_quant_gran,
//     float sm_scale,
//     int return_lse)
// {
//     // Extract dimensions based on tensor layout
//     int batch_size, num_heads, seq_len, head_dim;
//     int num_kv_heads, kv_len;
    
//     if (tensor_layout == 1) { // HND layout: [batch, heads, seq, dim]
//         batch_size = qdim[0];
//         num_heads = qdim[1];
//         seq_len = qdim[2];
//         head_dim = qdim[3];
//         num_kv_heads = kdim[1];
//         kv_len = kdim[2];
//     } else { // NHD layout: [batch, seq, heads, dim]
//         batch_size = qdim[0];
//         seq_len = qdim[1];
//         num_heads = qdim[2];
//         head_dim = qdim[3];
//         num_kv_heads = kdim[2];
//         kv_len = kdim[1];
//     }
    
//     // Validate dimensions
//     assert(num_heads % num_kv_heads == 0);
//     const int num_kv_groups = num_heads / num_kv_heads;
    
//     if (head_dim != 64 && head_dim != 128) {
//         fprintf(stderr, "ERROR: Unsupported head dimension %d (only 64 and 128 supported)\n", head_dim);
//         return;
//     }
    
//     // Extract strides based on tensor layout
//     uint32_t stride_bz_q, stride_seq_q, stride_h_q;
//     uint32_t stride_bz_k, stride_seq_k, stride_h_k;
//     uint32_t stride_bz_v, stride_seq_v, stride_h_v;
//     uint32_t stride_bz_o, stride_seq_o, stride_h_o;
    
//     if (tensor_layout == 1) { // HND layout
//         stride_bz_q = qstride[0];
//         stride_h_q = qstride[1];
//         stride_seq_q = qstride[2];
        
//         stride_bz_k = kstride[0];
//         stride_h_k = kstride[1];
//         stride_seq_k = kstride[2];
        
//         stride_bz_v = vstride[0];
//         stride_h_v = vstride[1];
//         stride_seq_v = vstride[2];
        
//         stride_bz_o = ostride[0];
//         stride_h_o = ostride[1];
//         stride_seq_o = ostride[2];
//     } else { // NHD layout
//         stride_bz_q = qstride[0];
//         stride_seq_q = qstride[1];
//         stride_h_q = qstride[2];
        
//         stride_bz_k = kstride[0];
//         stride_seq_k = kstride[1];
//         stride_h_k = kstride[2];
        
//         stride_bz_v = vstride[0];
//         stride_seq_v = vstride[1];
//         stride_h_v = vstride[2];
        
//         stride_bz_o = ostride[0];
//         stride_seq_o = ostride[1];
//         stride_h_o = ostride[2];
//     }
    
//     // Kernel configuration
//     constexpr uint32_t CTA_Q = 128;
//     constexpr uint32_t CTA_K = 64;
//     constexpr uint32_t WARP_K = 64;
    
//     // Print debug info
//     printf("\n=== qk_int8_sv_f16_accum_f16_attn_inst_buf_direct ===\n");
//     printf("Dimensions: B=%d, H=%d, S=%d, D=%d, H_kv=%d, S_kv=%d\n", 
//            batch_size, num_heads, seq_len, head_dim, num_kv_heads, kv_len);
//     printf("Tensor layout: %d (%s)\n", tensor_layout, tensor_layout == 1 ? "HND" : "NHD");
    
//     // Launch kernel based on parameters with proper WARP_Q dispatch
//     DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
//         // WARP_Q depends on HEAD_DIM
//         constexpr uint32_t WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
        
//         // Calculate shared memory requirement
//         size_t smem_max = std::max(
//             CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(half),
//             CTA_Q * HEAD_DIM * sizeof(half)
//         );
        
//         // Grid and block dimensions
//         dim3 grid_dim((seq_len + CTA_Q - 1) / CTA_Q, num_heads, batch_size);
//         dim3 block_dim(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        
//         printf("Grid: (%d, %d, %d), Block: (%d, %d)\n", 
//                grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y);
//         printf("Template params: CTA_Q=%d, CTA_K=%d, WARP_Q=%d, WARP_K=%d, HEAD_DIM=%d\n", 
//                CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
//         printf("Shared memory: %zu bytes\n", smem_max);
        
//         DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
//             DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                    
//                     constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
                    
//                     auto kernel_func = qk_int_sv_f16_attn_kernel<
//                         CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, 
//                         DataType::kInt8, 
//                         static_cast<QuantGranularity>(QK_QUANT_GRAN), 
//                         static_cast<QuantGranularity>(QK_QUANT_GRAN), 
//                         float,  // AccumDataType
//                         true,   // use_inst_buffer
//                         half,   // OutputDataType (FP16)
//                         ComputeUnit::kTensorCore, 
//                         mask_mode, 
//                         false, 
//                         false   // kInterleaved
//                     >;
                    
//                     cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
                    
//                     kernel_func<<<grid_dim, block_dim, smem_max>>>(
//                         Q, K, V, O,
//                         nullptr,
//                         Q_scale, K_scale,
//                         nullptr,  // V_mean (not used)
//                         seq_len,  // qo_len
//                         kv_len,   // kv_len
//                         num_kv_groups,
//                         stride_bz_q, stride_seq_q, stride_h_q,
//                         stride_bz_k, stride_seq_k, stride_h_k,
//                         stride_bz_v, stride_seq_v, stride_h_v,
//                         stride_bz_o, stride_seq_o, stride_h_o,
//                         sm_scale
//                     );
//             });
//         });
//     });
    
//     // Check for CUDA errors
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         fprintf(stderr, "ERROR: qk_int8_sv_f16_accum_f16_attn_inst_buf_direct kernel failed: %s\n", 
//                 cudaGetErrorString(error));
//     }
// }

// Per-block INT8 quantization with mean subtraction function
extern "C" int ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda(
    ccv_nnc_tensor_t* const input,   // Input tensor (FP16)
    ccv_nnc_tensor_t* const mean,    // Mean tensor (FP16)
    ccv_nnc_tensor_t* const output,  // Output tensor (INT8)
    ccv_nnc_tensor_t* const scale,   // Scale tensor (FP32)
    int block_size,                  // Block size (e.g., 64)
    int tensor_layout,               // 0=NHD, 1=HND
    ccv_nnc_stream_context_t* const stream_context)
{
    // Validate inputs
    if (!input || !mean || !output || !scale) {
        fprintf(stderr, "ERROR: ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda: NULL tensor input\n");
        return CCV_NNC_EXEC_INVALID;
    }
    
    assert(block_size == 128 || block_size == 64);

    // Get tensor views
    ccv_nnc_tensor_view_t* input_view = (ccv_nnc_tensor_view_t*)input;
    ccv_nnc_tensor_view_t* mean_view = (ccv_nnc_tensor_view_t*)mean;
    ccv_nnc_tensor_view_t* output_view = (ccv_nnc_tensor_view_t*)output;
    ccv_nnc_tensor_view_t* scale_view = (ccv_nnc_tensor_view_t*)scale;
    
    // Check tensor properties
    assert(CCV_IS_TENSOR_CONTIGUOUS(input_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(mean_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(output_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(scale_view));
    
    // Check data types
    assert(input_view->info.datatype == CCV_16F);
    assert(mean_view->info.datatype == CCV_16F);  // Mean should be same type as input
    assert(output_view->info.datatype == CCV_8U);  // INT8 (stored as 8U but treated as signed)
    assert(scale_view->info.datatype == CCV_32F);
    
    // Get dimensions
    int idim[CCV_NNC_MAX_DIM_ALLOC];
    int mdim[CCV_NNC_MAX_DIM_ALLOC];
    int odim[CCV_NNC_MAX_DIM_ALLOC];
    int sdim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(input_view, idim);
    ccv_nnc_tensor_view_get_dim(mean_view, mdim);
    ccv_nnc_tensor_view_get_dim(output_view, odim);
    ccv_nnc_tensor_view_get_dim(scale_view, sdim);
    
    // Check dimensions
    const int input_nd = ccv_nnc_tensor_nd(input_view->info.dim);
    const int mean_nd = ccv_nnc_tensor_nd(mean_view->info.dim);
    const int output_nd = ccv_nnc_tensor_nd(output_view->info.dim);
    const int scale_nd = ccv_nnc_tensor_nd(scale_view->info.dim);
    assert(input_nd == 4);
    assert(mean_nd == 3);  // Mean is 3D: [batch, heads, dim]
    assert(output_nd == 4);
    assert(scale_nd == 3);
    
    const int batch_size = idim[0];
    const int head_dim = idim[3];
    assert(head_dim == 64 || head_dim == 128);

    // Get strides from tensor views
    int istride[CCV_NNC_MAX_DIM_ALLOC];
    int mstride[CCV_NNC_MAX_DIM_ALLOC];
    int ostride[CCV_NNC_MAX_DIM_ALLOC];
    int sstride[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_stride(input_view, istride);
    ccv_nnc_tensor_view_get_stride(mean_view, mstride);
    ccv_nnc_tensor_view_get_stride(output_view, ostride);
    ccv_nnc_tensor_view_get_stride(scale_view, sstride);

    // Set strides based on tensor layout
    uint32_t stride_bz_input = istride[0];
    uint32_t stride_bz_output = ostride[0];

    int num_tokens, num_heads;
    uint32_t stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
    
    if (tensor_layout == 0) { // NHD: [batch, seq, heads, dim]
        num_tokens = idim[1];
        num_heads = idim[2];
        stride_seq_input = istride[1];
        stride_h_input = istride[2];
        stride_seq_output = ostride[1];
        stride_h_output = ostride[2];
    } else { // HND: [batch, heads, seq, dim]
        num_heads = idim[1];
        num_tokens = idim[2];
        stride_seq_input = istride[2];
        stride_h_input = istride[1];
        stride_seq_output = ostride[2];
        stride_h_output = ostride[1];
    }

    // Verify dimensions
    assert(mdim[1] == batch_size);  // Mean batch size matches
    assert(mdim[2] == num_heads);    // Mean heads matches
    assert(mdim[3] == head_dim);     // Mean dim matches head_dim
    
    // Verify output shape matches input
    assert(odim[0] == idim[0] && odim[1] == idim[1] && odim[2] == idim[2] && odim[3] == idim[3]);

    // Calculate scale dimensions - scale tensor is 3D [batch, num_heads, num_blocks]
    const size_t num_blocks = (num_tokens + block_size - 1) / block_size;
    // check scale output SHAPE
    assert(sdim[1] == batch_size);
    assert(sdim[2] == num_heads);
    assert(sdim[3] == num_blocks);

    // Use actual strides from tensors
    const uint32_t stride_bz_mean = mstride[1]; //  [1, B, H, k_scale_blocks] B index 1
    const uint32_t stride_h_mean = mstride[2]; //  [1, B, H, k_scale_blocks] B index 1
    const uint32_t stride_bz_scale = sstride[1]; //  [1, B, H, k_scale_blocks] B index 1
    const uint32_t stride_h_scale = sstride[2]; //  [1, B, H, k_scale_blocks] H index 1
    
    // Get CUDA stream
    cudaStream_t cuda_stream = ccv_nnc_stream_context_get_stream(stream_context);
    
    // Launch kernel with sub_mean=true
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
        dim3 block_dim(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
        dim3 grid_dim((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
   
        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, half><<<grid_dim, block_dim, 0, cuda_stream>>>(
            (half*)input_view->data.f16,     // input
            (half*)mean_view->data.f16,      // mean (used for sub_mean=true)
            (int8_t*)output_view->data.u8,   // output
            (float*)scale_view->data.f32,     // scale
            0.0f,                            // sm_scale (not used)
            num_tokens,                       // num_tokens
            stride_bz_input, stride_seq_input, stride_h_input,
            stride_bz_mean, stride_h_mean,    // mean strides
            stride_bz_output, stride_seq_output, stride_h_output,
            stride_bz_scale, stride_h_scale
        );
      });
    });

    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA per-block quantization with mean subtraction kernel launch failed: %s\n", 
               cudaGetErrorString(error));
        return CCV_NNC_EXEC_INVALID;
    }
    
    return CCV_NNC_EXEC_SUCCESS;
}

// Per-block INT8 quantization function (for K and V tensors)
extern "C" int ccv_nnc_quant_per_block_int8_cuda(
    ccv_nnc_tensor_t* const input,   // Input tensor (FP16)
    ccv_nnc_tensor_t* const output,  // Output tensor (INT8)
    ccv_nnc_tensor_t* const scale,   // Scale tensor (FP32)
    int block_size,                  // Block size (e.g., 64)
    int tensor_layout,               // 0=NHD, 1=HND
    ccv_nnc_stream_context_t* const stream_context)
{
	assert(block_size == 128 || block_size == 64);

    // Get tensor views
    ccv_nnc_tensor_view_t* input_view = (ccv_nnc_tensor_view_t*)input;
    ccv_nnc_tensor_view_t* output_view = (ccv_nnc_tensor_view_t*)output;
    ccv_nnc_tensor_view_t* scale_view = (ccv_nnc_tensor_view_t*)scale;
    
    // Check tensor properties
    assert(CCV_IS_TENSOR_CONTIGUOUS(input_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(output_view));
    assert(CCV_IS_TENSOR_CONTIGUOUS(scale_view));
    
    // Check data types
    assert(input_view->info.datatype == CCV_16F);
    assert(output_view->info.datatype == CCV_8U);  // INT8 (stored as 8U but treated as signed)
    assert(scale_view->info.datatype == CCV_32F);
    
    // Get dimensions
    int idim[CCV_NNC_MAX_DIM_ALLOC];
    int odim[CCV_NNC_MAX_DIM_ALLOC];
    int sdim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(input_view, idim);
    ccv_nnc_tensor_view_get_dim(output_view, odim);
    ccv_nnc_tensor_view_get_dim(scale_view, sdim);
    
    // Check dimensions
    const int input_nd = ccv_nnc_tensor_nd(input_view->info.dim);
    const int output_nd = ccv_nnc_tensor_nd(output_view->info.dim);
    const int scale_nd = ccv_nnc_tensor_nd(scale_view->info.dim);
    assert(input_nd == 4);
    assert(output_nd == 4);
    assert(scale_nd == 3);
    
    const int batch_size = input->info.dim[0];
    const int head_dim = input->info.dim[3];
    assert(head_dim == 64 || head_dim == 128);

    // Get strides from tensor views
    int istride[CCV_NNC_MAX_DIM_ALLOC];
    int ostride[CCV_NNC_MAX_DIM_ALLOC];
    int sstride[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_stride(input_view, istride);
    ccv_nnc_tensor_view_get_stride(output_view, ostride);
    ccv_nnc_tensor_view_get_stride(scale_view, sstride);

    // Set strides based on tensor layout
    uint32_t stride_bz_input = istride[0];
    uint32_t stride_bz_output = ostride[0];

    int num_tokens, num_heads;
    uint32_t stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
    
    if (tensor_layout == 0) { // NHD: [batch, seq, heads, dim]
		num_tokens = input->info.dim[1];
        num_heads = input->info.dim[2];
        stride_seq_input = istride[1];
        stride_h_input = istride[2];
        stride_seq_output = ostride[1];
        stride_h_output = ostride[2];
    } else { // HND: [batch, heads, seq, dim]
		num_heads = input->info.dim[1];
        num_tokens = input->info.dim[2];
        stride_seq_input = istride[2];
        stride_h_input = istride[1];
        stride_seq_output = ostride[2];
        stride_h_output = ostride[1];
    }

    // Verify output shape matches input
    assert(odim[0] == idim[0] && odim[1] == idim[1] && odim[2] == idim[2] && odim[3] == idim[3]);

    // Calculate scale strides - scale tensor is 3D [batch, num_heads, num_blocks]
    const size_t num_blocks = (num_tokens + block_size - 1) / block_size;
	assert(sdim[1] == batch_size && sdim[2] == num_heads && sdim[3] == num_blocks); // GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks) [1, B, H, k_scale_blocks]

    // Use actual strides from scale tensor 
    const uint32_t stride_bz_scale = sstride[1]; //  [1, B, H, k_scale_blocks] B index 1
    const uint32_t stride_h_scale = sstride[2]; //  [1, B, H, k_scale_blocks] H index 1
    
    // Get CUDA stream
    cudaStream_t cuda_stream = ccv_nnc_stream_context_get_stream(stream_context);
    
    // printf("\n=== K Per-Block Quantization Debug ===\n");
    // printf("DEBUG: Input tensor dimensions: [%d, %d, %d, %d]\n", batch_size, num_tokens, num_heads, head_dim);
    // printf("DEBUG: tensor_layout=%d (0=NHD, 1=HND)\n", tensor_layout);
    // printf("DEBUG: block_size=%d\n", block_size);
    // printf("DEBUG: Expected scale shape: [%d, %d, %d]\n", batch_size, num_heads, (num_tokens + block_size - 1) / block_size);
    // printf("DEBUG: Input strides: stride_bz=%d, stride_seq=%d, stride_h=%d\n", 
    //        stride_bz_input, stride_seq_input, stride_h_input);
    // printf("DEBUG: Output strides: stride_bz=%d, stride_seq=%d, stride_h=%d\n", 
    //        stride_bz_output, stride_seq_output, stride_h_output);
    // printf("DEBUG: Scale strides: stride_bz=%d, stride_h=%d\n", stride_bz_scale, stride_h_scale);
    // printf("DEBUG: Scale strides: 0=%d, 1=%d, 2=%d, 3=%d\n", sstride[0], sstride[1], sstride[2], sstride[3]);

    // Launch the per-block quantization kernel using the existing QuantInt8Kernel template
    // For per-block: head_dim=D, BLOCK_SIZE=block_size, has_sm_scale=false, sub_mean=false
	
    // Use the existing QuantInt8Kernel template (assuming D=128 and block_size=64)
	DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

		constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
    	dim3 block_dim(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
    	dim3 grid_dim(num_blocks, num_heads, batch_size);
   
		QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, half><<<grid_dim, block_dim, 0, cuda_stream>>>(
            (half*)input_view->data.f16,     // input
            nullptr,                        // mean (not used for per-block)
            (int8_t*)output_view->data.u8,  // output
            (float*)scale_view->data.f32,    // scale
            0.0f,                       // sm_scale (not used for per-block)
            num_tokens,                          // num_tokens
            stride_bz_input, stride_seq_input, stride_h_input,
            0, 0,                       // mean strides (not used)
            stride_bz_output, stride_seq_output, stride_h_output,
            stride_bz_scale, stride_h_scale
        );
	  });
	});
    // if (head_dim == 128 && block_size == 64 && num_pack_per_thread == 1) {
    //     QuantInt8Kernel<128, 64, 1, false, false, half><<<grid_dim, block_dim, 0, cuda_stream>>>(
    //         (half*)input_view->data.f16,     // input
    //         nullptr,                        // mean (not used for per-block)
    //         (int8_t*)output_view->data.u8,  // output
    //         (float*)scale_view->data.f32,    // scale
    //         0.0f,                       // sm_scale (not used for per-block)
    //         num_tokens,                          // num_tokens
    //         stride_bz_input, stride_seq_input, stride_h_input,
    //         0, 0,                       // mean strides (not used)
    //         stride_bz_output, stride_seq_output, stride_h_output,
    //         stride_bz_scale, stride_h_scale
    //     );
    // } else {
    //     printf("Unsupported configuration: D=%d, block_size=%d\n", head_dim, block_size);
    //     return CCV_NNC_EXEC_INVALID;
    // }
    
    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess) {
        printf("CUDA per-block quantization kernel launch failed: %s\n", 
               cudaGetErrorString(error));
        return CCV_NNC_EXEC_INVALID;
    }
    
    return CCV_NNC_EXEC_SUCCESS;
}

// CCV wrapper matching PyTorch per_warp_int8 API - quantizes Q with per-warp, K with per-block
extern "C" int ccv_nnc_per_warp_int8(
    ccv_nnc_tensor_t* const q,       // Input Q tensor (FP16)
    ccv_nnc_tensor_t* const k,       // Input K tensor (FP16)
    ccv_nnc_tensor_t* const q_int8,  // Output Q quantized (INT8)
    ccv_nnc_tensor_t* const k_int8,  // Output K quantized (INT8)
    ccv_nnc_tensor_t* const q_scale, // Output Q scales (FP32)
    ccv_nnc_tensor_t* const k_scale, // Output K scales (FP32)
    ccv_nnc_tensor_t* const km,      // Optional K mean tensor (not used currently)
    int BLKQ,                        // Block size for Q (default 128)
    int WARPQ,                       // Warp size for Q (default 32)
    int BLKK,                        // Block size for K (default 64)
    int tensor_layout,               // 0=NHD, 1=HND
    ccv_nnc_stream_context_t* const stream_context)
{   
    // Set defaults if not provided
    if (BLKQ <= 0) BLKQ = 128;
    if (WARPQ <= 0) WARPQ = 32;
    if (BLKK <= 0) BLKK = 64;
    
    // Validate required inputs
      if (!q || !k || !q_int8 || !k_int8 || !q_scale || !k_scale) {
          return CCV_NNC_EXEC_INVALID;
      }

	  int k_result;  // Declare k_result here

      // Optional parameter - check if provided
      if (km != NULL) {
          // Use mean subtraction path
          // Call ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda for K
        k_result = ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda(
              k, km, k_int8, k_scale, BLKK, tensor_layout, stream_context);
      } else {
          // Regular quantization path without mean subtraction
          // Call ccv_nnc_quant_per_block_int8_cuda for K
        k_result = ccv_nnc_quant_per_block_int8_cuda(
              k, k_int8, k_scale, BLKK, tensor_layout, stream_context);
      }

      // Always do per-warp quantization for Q
      int q_result = ccv_nnc_quant_per_warp_int8_cuda(
          q, q_int8, q_scale, BLKQ, WARPQ, tensor_layout, stream_context);

      return (q_result == 0 && k_result == 0) ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID;
}

	
// Matches sageattn_qk_int8_pv_fp16_cuda
extern "C" int ccv_nnc_sageattn_qk_int8_pv_fp16_cuda(
    ccv_nnc_tensor_t* const query,
    ccv_nnc_tensor_t* const key,
    ccv_nnc_tensor_t* const k_mean,
    ccv_nnc_tensor_t* const value,
    ccv_nnc_tensor_t* const q_int8,  // Output Q quantized (INT8)
    ccv_nnc_tensor_t* const k_int8,  // Output K quantized (INT8)
    ccv_nnc_tensor_t* const query_scale,
    ccv_nnc_tensor_t* const key_scale,
    ccv_nnc_tensor_t* const output,
    int tensor_layout, // 0 NHD, 1 HND 
    int is_causal, // default false
    int qk_quant_gran, // only support per_warp
    float sm_scale, // default // 0.088388
    int return_lse, // default false not used yet
	sage_attn_pv_accum_dtype pv_accum_dtype, // default fp32, 
    int BLKQ,                        // Block size for Q (default 128)
    int WARPQ,                       // Warp size for Q (default 32)
    int BLKK,                        // Block size for K (default 64)
    ccv_nnc_stream_context_t* const stream_context
	)
{
    int warp_result = ccv_nnc_per_warp_int8(
		query,       // Input Q tensor (FP16)
		key,       // Input K tensor (FP16)
		q_int8,  // Output Q quantized (INT8)
		k_int8,  // Output K quantized (INT8)
		query_scale, // Output Q scales (FP32)
		key_scale, // Output K scales (FP32)
		k_mean,      // Optional K mean tensor (not used currently)
		BLKQ,                        // Block size for Q (default 128)
		WARPQ,                       // Warp size for Q (default 32)
		BLKK,                        // Block size for K (default 64)
		tensor_layout,               // 0=NHD, 1=HND
        stream_context);
    
    if (warp_result != 0) {
        fprintf(stderr, "ERROR: ccv_nnc_per_warp_int8 failed\n");
        return CCV_NNC_EXEC_INVALID; 
    }
    
    if (pv_accum_dtype == DTYPE_FP32) {
        int result = ccv_nnc_qk_int8_sv_f16_accum_f32_attn(
            q_int8,        // query (int8)
            k_int8,        // key (int8)
            value,        // value (fp16)
            output,        // output (fp16) - modified in-place
            query_scale,  // query_scale (fp32)
            key_scale,  // key_scale (fp32)
            tensor_layout,                   // tensor_layout: 1=HND (matching direct test)
            is_causal,                   // is_causal: false
            qk_quant_gran,                   // qk_quant_gran: 2=per_warp
            sm_scale,            // sm_scale
            return_lse           // return_lse: false
	    );
           
	    if (result == 0)  {
            printf("✅ ccv_nnc_sageattn_qk_int8_pv_fp16_cuda completed successfully\n");
            return CCV_NNC_EXEC_SUCCESS; 
        } else {
            fprintf(stderr, "ccv_nnc_qk_int8_sv_f16_accum_f32_attn failed\n");
            return CCV_NNC_EXEC_INVALID; 
        }
    } else {
        fprintf(stderr, "ERROR: pv_accum_dtype not supported yet:%d\n", pv_accum_dtype);
        return CCV_NNC_EXEC_INVALID; 
    }

}


static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
    // SageAttention forward pass
    // Expected inputs: Q, K, V, [optional: attn_mask, weights, bias, k_mean]
    // Expected outputs: O, [optional: saved_softmax_lse, q_int8, k_int8, q_scale, k_scale]
    
    assert(input_size >= 3);
    assert(output_size >= 1);
    
    // Core inputs
    ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
    ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
    ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];
    
    // Optional inputs
    ccv_nnc_tensor_view_t* const k_mean = input_size > 6 ? (ccv_nnc_tensor_view_t*)inputs[3] : 0;
    
  
    // Core output
    ccv_nnc_tensor_view_t* const o = (ccv_nnc_tensor_view_t*)outputs[0];
    
    // Optional outputs - these should be provided by the caller if quantized outputs are needed
    ccv_nnc_tensor_view_t* const saved_softmax_lse = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
    ccv_nnc_tensor_view_t* const q_int8 = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
    ccv_nnc_tensor_view_t* const k_int8 = output_size > 3 ? (ccv_nnc_tensor_view_t*)outputs[3] : 0;
    ccv_nnc_tensor_view_t* const q_scale = output_size > 4 ? (ccv_nnc_tensor_view_t*)outputs[4] : 0;
    ccv_nnc_tensor_view_t* const k_scale = output_size > 5 ? (ccv_nnc_tensor_view_t*)outputs[5] : 0;
    
    // For SageAttention to work, we need the quantized outputs
    if (!q_int8 || !k_int8 || !q_scale || !k_scale) {
        // If quantized outputs are not provided, we cannot proceed
        return CCV_NNC_EXEC_INVALID;
    }
    
    // Validate tensor dimensions
    const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
    assert(q_nd == 3 || q_nd == 4);
    const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
    assert(k_nd == 3 || k_nd == 4);
    const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
    assert(v_nd == 3 || v_nd == 4);
    const int o_nd = ccv_nnc_tensor_nd(o->info.dim);
    assert(o_nd == 3 || o_nd == 4);
    assert(q_nd == k_nd && k_nd == v_nd && v_nd == o_nd);

    int qdim[CCV_NNC_MAX_DIM_ALLOC];
    int kdim[CCV_NNC_MAX_DIM_ALLOC];
    int vdim[CCV_NNC_MAX_DIM_ALLOC];
    int odim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(q, qdim);
    ccv_nnc_tensor_view_get_dim(k, kdim);
    ccv_nnc_tensor_view_get_dim(v, vdim);
    ccv_nnc_tensor_view_get_dim(o, odim);

    // Validate tensor formats
    assert(q->info.format == CCV_TENSOR_FORMAT_NHWC);
    assert(k->info.format == CCV_TENSOR_FORMAT_NHWC);
    assert(v->info.format == CCV_TENSOR_FORMAT_NHWC);
    assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);

    assert(CCV_IS_TENSOR_CONTIGUOUS(q));
    assert(CCV_IS_TENSOR_CONTIGUOUS(k));
    assert(CCV_IS_TENSOR_CONTIGUOUS(v));
    assert(CCV_IS_TENSOR_CONTIGUOUS(o));

    int batch_size;
    int R; // sequence length for Q
    int C; // sequence length for K/V
    int Hq; // number of heads for Q
    int Hk; // number of heads for K/V
    int D;  // head dimension
    
    if (q_nd == 3) {
        batch_size = qdim[1];
        assert(batch_size == kdim[1]);
        R = qdim[2];
        C = kdim[2];
        Hq = Hk = 1;
        D = qdim[3];
        assert(D == kdim[3]);
    } else if (q_nd == 4) {
        batch_size = qdim[0];
        assert(batch_size == kdim[0]);
        R = qdim[1];
        C = kdim[1];
        Hq = qdim[2];
        Hk = kdim[2];
        assert(Hq >= Hk);
        assert(Hq % Hk == 0);
        D = qdim[3];
        assert(D == kdim[3]);
    }

    // Check if tensors are in the correct data type
    const int is_same_dtype =
        (q->info.datatype == k->info.datatype) &&
        (q->info.datatype == v->info.datatype) &&
        (q->info.datatype == o->info.datatype) &&
        (q->info.datatype == CCV_16F); // SageAttention requires FP16
    
    if (!is_same_dtype) {
        return CCV_NNC_EXEC_INVALID;
    }

    // SageAttention only supports certain head dimensions
    if (D != 64 && D != 128) {
        return CCV_NNC_EXEC_INVALID;
    }

    // Validate quantized output tensors
    assert(q_int8->info.datatype == CCV_8U);
    assert(k_int8->info.datatype == CCV_8U);
    assert(q_scale->info.datatype == CCV_32F);
    assert(k_scale->info.datatype == CCV_32F);
    
    // SageAttention quantization parameters
    const int BLKQ = 128;  // Block size for Q quantization
    const int BLKK = 64;   // Block size for K quantization
    // Determine accumulation type and corresponding WARPQ based on head dimension and scale tensor shape
    // Following PyTorch pattern: WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32)
    sage_attn_pv_accum_dtype pv_accum_dtype; 
    if (cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_8U_32F) {
        pv_accum_dtype = DTYPE_FP32;
    } else {
        // default using mix
        pv_accum_dtype = DTYPE_FP16_MIX_FP32;
    }

    int WARPQ;    
    // Use the same logic as PyTorch SageAttention
    if (D == 128 && pv_accum_dtype == DTYPE_FP16_MIX_FP32) {
        // Head dimension 128 with 8 scales per head -> FP16_MIX_FP32 accumulation
        WARPQ = 16;
    } else {
        // Default to FP32 accumulation with WARPQ=32
        WARPQ = 32;
    }

    // Calculate quantization tensor dimensions
    const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
    const size_t warps_per_block = BLKQ / WARPQ;
    const int q_scale_blocks = q_blocks * warps_per_block;
    const int k_scale_blocks = (C + BLKK - 1) / BLKK;

    // Create internal quantization tensors on GPU
    ccv_nnc_tensor_t* q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, Hq, R, D), 0);
    ccv_nnc_tensor_t* k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, Hk, C, D), 0);
    ccv_nnc_tensor_t* q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, Hq, q_scale_blocks), 0);
    ccv_nnc_tensor_t* k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, Hk, k_scale_blocks), 0);
    // Optional k_mean tensor - create as zeros if not provided
    ccv_nnc_tensor_t* k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, Hk, D), 0);

    // Call SageAttention with proper parameters
    // tensor_layout: 1 for HND (batch, heads, seq, dim) - CCV uses NHWC which maps to HND
    const int tensor_layout = 1; // HND layout
    const int is_causal = cmd.info.scaled_dot_product_attention.is_causal;
    const int qk_quant_gran = 2; // per_warp quantization
    const float sm_scale = 1.0f / sqrtf((float)D); // scale = 1.0 / sqrt(head_dim)
    const int return_lse = saved_softmax_lse ? 1 : 0;

    // Extract tensor dimensions for quantized outputs
    int q_int8_dim[CCV_NNC_MAX_DIM_ALLOC];
    int k_int8_dim[CCV_NNC_MAX_DIM_ALLOC];
    int q_scale_dim[CCV_NNC_MAX_DIM_ALLOC];
    int k_scale_dim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(q_int8, q_int8_dim);
    ccv_nnc_tensor_view_get_dim(k_int8, k_int8_dim);
    ccv_nnc_tensor_view_get_dim(q_scale, q_scale_dim);
    ccv_nnc_tensor_view_get_dim(k_scale, k_scale_dim);

    
    printf("DEBUG: SageAttention configuration:\n");
    printf("  Head dimension: %d\n", D);
    printf("  Accumulation type: %s\n", pv_accum_dtype == DTYPE_FP32 ? "DTYPE_FP32" : "DTYPE_FP16_MIX_FP32");
    printf("  WARPQ: %d\n", WARPQ);
    // Extract tensor strides
    int qstride[CCV_NNC_MAX_DIM_ALLOC];
    int kstride[CCV_NNC_MAX_DIM_ALLOC];
    int vstride[CCV_NNC_MAX_DIM_ALLOC];
    int ostride[CCV_NNC_MAX_DIM_ALLOC];
    int q_int8_stride[CCV_NNC_MAX_DIM_ALLOC];
    int k_int8_stride[CCV_NNC_MAX_DIM_ALLOC];
    int q_scale_stride[CCV_NNC_MAX_DIM_ALLOC];
    int k_scale_stride[CCV_NNC_MAX_DIM_ALLOC];
    
    ccv_nnc_tensor_view_get_stride(q, qstride);
    ccv_nnc_tensor_view_get_stride(k, kstride);
    ccv_nnc_tensor_view_get_stride(v, vstride);
    ccv_nnc_tensor_view_get_stride(o, ostride);
    ccv_nnc_tensor_view_get_stride(q_int8, q_int8_stride);
    ccv_nnc_tensor_view_get_stride(k_int8, k_int8_stride);
    ccv_nnc_tensor_view_get_stride(q_scale, q_scale_stride);
    ccv_nnc_tensor_view_get_stride(k_scale, k_scale_stride);

    // Handle optional k_mean tensor
    int km_dim[CCV_NNC_MAX_DIM_ALLOC];
    int km_stride[CCV_NNC_MAX_DIM_ALLOC];
    if (k_mean) {
        ccv_nnc_tensor_view_get_dim(k_mean, km_dim);
        ccv_nnc_tensor_view_get_stride(k_mean, km_stride);
    }

    // Get CUDA stream
    cudaStream_t cuda_stream = ccv_nnc_stream_context_get_stream(stream_context);

    // Call direct function
    ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
        (half*)q->data.f16,           // query data
        (half*)k->data.f16,           // key data
        k_mean ? (half*)k_mean_tensor->data.f16 : NULL, // k_mean data (optional)
        (half*)v->data.f16,           // value data
        (int8_t*)q_int8->data.u8,     // q_int8 output data
        (int8_t*)k_int8->data.u8,     // k_int8 output data
        (float*)q_scale->data.f32,    // query_scale data
        (float*)k_scale->data.f32,    // key_scale data
        (half*)o->data.f16,           // output data
        qdim,                         // query dimensions
        kdim,                         // key dimensions
        vdim,                         // value dimensions
        odim,                         // output dimensions
        q_int8_dim,                   // q_int8 dimensions
        k_int8_dim,                   // k_int8 dimensions
        q_scale_dim,                  // query_scale dimensions
        k_scale_dim,                  // key_scale dimensions
        qstride,                      // query strides
        kstride,                      // key strides
        vstride,                      // value strides
        ostride,                      // output strides
        q_int8_stride,                // q_int8 strides
        k_int8_stride,                // k_int8 strides
        q_scale_stride,               // query_scale strides
        k_scale_stride,               // key_scale strides
        tensor_layout,                // tensor_layout: 1=HND
        is_causal,                    // is_causal
        qk_quant_gran,                // qk_quant_gran: 2=per_warp
        sm_scale,                     // sm_scale
        return_lse,                   // return_lse
        (int)pv_accum_dtype,          // pv_accum_dtype: FP32
        BLKQ,                         // BLKQ
        WARPQ,                        // WARPQ
        BLKK,                         // BLKK
        k_mean ? km_dim : NULL,       // k_mean dimensions (optional)
        k_mean ? km_stride : NULL,    // k_mean strides (optional)
        cuda_stream                   // CUDA stream
    );

    CUDA_ENFORCE(cudaGetLastError());

          // Clean up internal tensors
      ccv_nnc_tensor_free(q_int8_tensor);
      ccv_nnc_tensor_free(k_int8_tensor);
      ccv_nnc_tensor_free(q_scale_tensor);
      ccv_nnc_tensor_free(k_scale_tensor);
      ccv_nnc_tensor_free(k_mean_tensor);

    return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// TODO: Implement SageAttention backward pass
	// For now, return not implemented
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_8U;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
#endif
}