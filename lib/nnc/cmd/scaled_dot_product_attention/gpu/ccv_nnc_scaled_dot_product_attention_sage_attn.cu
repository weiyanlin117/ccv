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
  
    // Core output
    ccv_nnc_tensor_view_t* const o = (ccv_nnc_tensor_view_t*)outputs[0];
    
    // Optional outputs - these should be provided by the caller if quantized outputs are needed
    ccv_nnc_tensor_view_t* const saved_softmax_lse = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
    
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
        
        // NHD layout: [batch, seq, heads, dim]
        R = qdim[1];   // sequence length
        C = kdim[1];   // sequence length  
        Hq = qdim[2];  // num heads
        Hk = kdim[2];  // num heads

        // HND layout: [batch, heads, seq, dim]  for HND, although the kernel support HND, we didn't expose yet
        // Hq = qdim[1];  // num heads
        // Hk = kdim[1];  // num heads
        // R = qdim[2];   // sequence length
        // C = kdim[2];   // sequence length
        
        assert(Hq >= Hk);
        assert(Hq % Hk == 0);
        D = qdim[3];
        assert(D == kdim[3]);
    }

    const bool is_nhd_layout = true;

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
    
    // SageAttention quantization parameters
    const int BLKQ = 128;  // Block size for Q quantization
    const int BLKK = 64;   // Block size for K quantization
    // Determine accumulation type and corresponding WARPQ based on head dimension and scale tensor shape
    // Following PyTorch pattern: WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32)
    sage_attn_pv_accum_dtype pv_accum_dtype; 
    if (cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_8U_32F) {
        pv_accum_dtype = DTYPE_FP32;
    } else {
        // Default to FP32 accumulation to match PyTorch SageAttention behavior
        pv_accum_dtype = DTYPE_FP32;
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
    // Create internal quantization tensors on GPU matching the detected layout
    ccv_nnc_tensor_t* q_int8_tensor;
    ccv_nnc_tensor_t* k_int8_tensor;
    int seq_dim;
    if (is_nhd_layout) {
        // NHD layout: [batch, seq, heads, dim]
        q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, R, Hq, D), 0);
        k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, C, Hk, D), 0);
        seq_dim = 1;
    } else {
        // HND layout: [batch, heads, seq, dim]
        q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, Hq, R, D), 0);
        k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, batch_size, Hk, C, D), 0);
        seq_dim = 2;
    }

    // Scale tensors are always [batch, heads, scale_blocks] regardless of input layout
    ccv_nnc_tensor_t* q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, Hq, q_scale_blocks), 0);
    ccv_nnc_tensor_t* k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, Hk, k_scale_blocks), 0);
    ccv_nnc_tensor_t* k_mean_tensor;
    if (is_nhd_layout) {
        k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, 1, Hk, D), 0);
    } else {
        k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, Hk, 1, D), 0);
    }

    // if (smooth_k) {
        // int reduce_axis = is_nhd_layout ? 1 : 2;
        // printf("reduce!");
        // ccv_nnc_cmd_param_t reduce_params = {
        //     .size = {.dim = {1, 1, 1}},
        //     .reduce = {.axis = {reduce_axis}, .count = 1}
        // };
        // ccv_nnc_cmd_t reduce_cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MEAN_FORWARD, 0, reduce_params, 0);
        // ccv_nnc_tensor_t* reduce_inputs[] = {(ccv_nnc_tensor_t*)k};
        // ccv_nnc_tensor_t* reduce_outputs[] = {k_mean_tensor};
        // ccv_nnc_cmd_exec(reduce_cmd, ccv_nnc_no_hint, 0,
        //                 reduce_inputs, 1, reduce_outputs, 1, stream_context);
    // }
    
    ccv_nnc_tensor_view_t* q_int8 = (ccv_nnc_tensor_view_t*)q_int8_tensor;
    ccv_nnc_tensor_view_t* k_int8 = (ccv_nnc_tensor_view_t*)k_int8_tensor;
    ccv_nnc_tensor_view_t* q_scale = (ccv_nnc_tensor_view_t*)q_scale_tensor;
    ccv_nnc_tensor_view_t* k_scale = (ccv_nnc_tensor_view_t*)k_scale_tensor;
    ccv_nnc_tensor_view_t* k_mean = (ccv_nnc_tensor_view_t*)k_mean_tensor;


    // Call SageAttention with proper parameters
    // tensor_layout: 0 for NHD (batch, seq, heads, dim), 1 for HND (batch, heads, seq, dim)
    const int tensor_layout = is_nhd_layout ? 0 : 1;
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

    int km_dim[CCV_NNC_MAX_DIM_ALLOC];
    int km_stride[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(k_mean, km_dim);
    ccv_nnc_tensor_view_get_stride(k_mean, km_stride);
    

    // Get CUDA stream
    cudaStream_t cuda_stream = ccv_nnc_stream_context_get_stream(stream_context);

    // Call direct function
    ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
        (half*)q->data.f16,           // query data
        (half*)k->data.f16,           // key data
        (half*)k_mean->data.f16, // k_mean data
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