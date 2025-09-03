extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA_SM80
#include <nnc/gpu/3rdparty/flash_attn/flash_api.h>
#include <nnc/gpu/3rdparty/sage_attn/fused.h>
#include <nnc/gpu/3rdparty/sage_attn/sage_attn_utils.cuh>

// SageAttention wrapper function for INT8 quantized attention
static int _ccv_nnc_scaled_dot_product_attention_sage_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Check for any pre-existing CUDA errors at function entry
	cudaError_t entry_error = cudaGetLastError();
	if (entry_error != cudaSuccess) {
		printf("WARNING: Pre-existing CUDA error at function entry: %s\n", cudaGetErrorString(entry_error));
		// Clear the error so we can continue
		cudaGetLastError();
	}
	
	// SageAttention forward pass for INT8 quantized attention
	// Expected inputs: Q, K, V, [optional: attn_mask, weights, bias, k_mean]
	// Expected outputs: O
	
	assert(input_size >= 3);
	assert(output_size >= 1);
	
	// Core inputs
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];    
	
	// Core output
	ccv_nnc_tensor_view_t* const o = (ccv_nnc_tensor_view_t*)outputs[0];
	
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
	
	// SageAttention quantization parameters
	const int BLKQ = 128;  // Block size for Q quantization
	const int BLKK = 64;   // Block size for K quantization
	
	// Determine accumulation type and corresponding WARPQ based on head dimension and scale tensor shape
	sage_attn_pv_accum_dtype pv_accum_dtype; 
	if (cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_8U_32F) {
		pv_accum_dtype = DTYPE_FP32;
	} else {
		// Default to FP32 accumulation to match PyTorch SageAttention behavior
		pv_accum_dtype = DTYPE_FP16_MIX_FP32;
	}
	// printf("pv_accum_dtype: %d", pv_accum_dtype);

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
	
	// printf("DEBUG: R=%d, BLKQ=%d, WARPQ=%d\n", R, BLKQ, WARPQ);
	// printf("DEBUG: q_blocks=%zu, warps_per_block=%zu, q_scale_blocks=%d\n", 
	//        q_blocks, warps_per_block, q_scale_blocks);
	// printf("DEBUG: C=%d, BLKK=%d, k_scale_blocks=%d\n", C, BLKK, k_scale_blocks);
	
	// Calculate padded sequence length for V (must be multiple of 64 for SageAttention)
	const int padded_C = ((C + 63) / 64) * 64;
	
	printf("DEBUG: V transformation - C=%d, padded_C=%d\n", C, padded_C);
	printf("DEBUG: v_fp8 dimensions will be [%d, %d, %d, %d]\n", batch_size, D, Hk, padded_C);
	
	// Calculate sizes for all workspace allocations
	size_t q_int8_size = sizeof(int8_t) * batch_size * R * Hq * D;
	size_t k_int8_size = sizeof(int8_t) * batch_size * C * Hk * D;
	size_t v_transposed_size = sizeof(half) * batch_size * D * Hk * padded_C;  // [B, D, H, padded_S] for transposed V
	size_t v_fp8_size = sizeof(int8_t) * batch_size * D * Hk * padded_C;       // [B, D, H, padded_S] for FP8 V
	size_t q_scale_size = sizeof(float) * batch_size * Hq * q_scale_blocks;
	size_t k_scale_size = sizeof(float) * batch_size * Hk * k_scale_blocks;
	size_t v_scale_size = sizeof(float) * batch_size * Hk * D;          // per-channel scales for V
	// size_t k_mean_size = sizeof(half) * batch_size * 1 * Hk * D;  // FP16 data size for k_mean tensor
	
	// Set k_mean dimensions and strides manually for workspace tensor
	int km_dim[CCV_NNC_MAX_DIM_ALLOC];
	km_dim[0] = batch_size;
	km_dim[1] = 1;
	km_dim[2] = Hk;
	km_dim[3] = D;
	
	int km_stride[CCV_NNC_MAX_DIM_ALLOC];
	km_stride[3] = 1;           // dim stride
	km_stride[2] = D;           // head stride  
	km_stride[1] = Hk * D;      // seq stride (should be same as head since seq=1)
	km_stride[0] = Hk * D;      // batch stride
	
	// // Get workspace size for reduction operation
	size_t reduce_workspace_size = 0;
	// CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_mean, k_desc.descriptor, k_mean_desc, &reduce_workspace_size));
	
	// Allocate COMBINED workspace for ALL tensors including V quantization
	size_t total_workspace_size = q_int8_size + k_int8_size + v_transposed_size + v_fp8_size + 
	                             q_scale_size + k_scale_size + v_scale_size;
	
	unsigned char* workspace = (unsigned char*)ccv_nnc_stream_context_get_workspace(stream_context, total_workspace_size, CCV_TENSOR_GPU_MEMORY);
	
	// Partition the workspace for each tensor
	int8_t* q_int8_workspace = (int8_t*)workspace;
	int8_t* k_int8_workspace = (int8_t*)(workspace + q_int8_size);
	half* v_transposed_workspace = (half*)(workspace + q_int8_size + k_int8_size);
	int8_t* v_fp8_workspace = (int8_t*)(workspace + q_int8_size + k_int8_size + v_transposed_size);
	float* q_scale_workspace = (float*)(workspace + q_int8_size + k_int8_size + v_transposed_size + v_fp8_size);
	float* k_scale_workspace = (float*)(workspace + q_int8_size + k_int8_size + v_transposed_size + v_fp8_size + q_scale_size);
	float* v_scale_workspace = (float*)(workspace + q_int8_size + k_int8_size + v_transposed_size + v_fp8_size + 
	                                    q_scale_size + k_scale_size);
	//half* k_mean_workspace = (half*)(workspace + q_int8_size + k_int8_size + q_scale_size + k_scale_size);
	// void* reduce_workspace = (reduce_workspace_size > 0) ? (workspace + q_int8_size + k_int8_size + q_scale_size + k_scale_size + k_mean_size) : NULL;
	
	// // Perform the reduce mean operation
	// static const float one = 1.0f, zero = 0.0f;
	// CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_mean, 0, 0, reduce_workspace, reduce_workspace_size, 
	//                                &one, k_desc.descriptor, k_desc.data.u8, 
	//                                &zero, k_mean_desc, k_mean_workspace));

	// Call SageAttention with proper parameters
	// tensor_layout: 0 for NHD (batch, seq, heads, dim), 1 for HND (batch, heads, seq, dim)
	const int tensor_layout = 0;
	const int is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	const int qk_quant_gran = 2; // per_warp quantization
	const float sm_scale = 1.0f / sqrtf((float)D); // scale = 1.0 / sqrt(head_dim)
	const int return_lse = 0;
	
	// Extract tensor dimensions for quantized outputs (all use workspace now)
	// NHD layout: [batch, seq, heads, dim]
	int q_int8_dim[CCV_NNC_MAX_DIM_ALLOC] = {batch_size, R, Hq, D};
	int k_int8_dim[CCV_NNC_MAX_DIM_ALLOC] = {batch_size, C, Hk, D};
	int v_fp8_dim[CCV_NNC_MAX_DIM_ALLOC] = {batch_size, D, Hk, padded_C};  // V after transpose: [B, D, H, padded_S]
	int q_scale_dim[CCV_NNC_MAX_DIM_ALLOC] = {1, batch_size, Hq, q_scale_blocks};
	int k_scale_dim[CCV_NNC_MAX_DIM_ALLOC] = {1, batch_size, Hk, k_scale_blocks};
	int v_scale_dim[CCV_NNC_MAX_DIM_ALLOC] = {batch_size, Hk, D};   // V scale dimensions
	
	// printf("DEBUG: Our manual dimensions:\n");
	// printf("  q_int8_dim: [%d, %d, %d, %d]\n", q_int8_dim[0], q_int8_dim[1], q_int8_dim[2], q_int8_dim[3]);
	// printf("  k_int8_dim: [%d, %d, %d, %d]\n", k_int8_dim[0], k_int8_dim[1], k_int8_dim[2], k_int8_dim[3]);
	// printf("  q_scale_dim: [%d, %d, %d, %d]\n", q_scale_dim[0], q_scale_dim[1], q_scale_dim[2], q_scale_dim[3]);
	// printf("  k_scale_dim: [%d, %d, %d, %d]\n", k_scale_dim[0], k_scale_dim[1], k_scale_dim[2], k_scale_dim[3]);

	// Extract tensor strides
	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int ostride[CCV_NNC_MAX_DIM_ALLOC];
	
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(o, ostride);
	
	// Calculate strides manually for workspace tensors - NHD layout: [batch, seq, heads, dim]
	int q_int8_stride[CCV_NNC_MAX_DIM_ALLOC];
	q_int8_stride[3] = 1;           // dim stride
	q_int8_stride[2] = D;           // head stride  
	q_int8_stride[1] = Hq * D;      // seq stride
	q_int8_stride[0] = R * Hq * D;  // batch stride
	
	int k_int8_stride[CCV_NNC_MAX_DIM_ALLOC];
	k_int8_stride[3] = 1;           // dim stride
	k_int8_stride[2] = D;           // head stride
	k_int8_stride[1] = Hk * D;      // seq stride
	k_int8_stride[0] = C * Hk * D;  // batch stride
	
	int v_fp8_stride[CCV_NNC_MAX_DIM_ALLOC];
	v_fp8_stride[3] = 1;                      // padded_S stride
	v_fp8_stride[2] = padded_C;               // H stride  
	v_fp8_stride[1] = Hk * padded_C;          // D stride
	v_fp8_stride[0] = D * Hk * padded_C;      // batch stride
	
	// Scale tensor strides: [batch, heads, scale_blocks]
	int q_scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	q_scale_stride[3] = 1;                        // unused
	q_scale_stride[2] = q_scale_blocks;                        // scale_blocks stride
	q_scale_stride[1] = Hq * q_scale_blocks;           // head stride
	q_scale_stride[0] = batch_size * Hq * q_scale_blocks;      // batch stride
	
	int k_scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	k_scale_stride[3] = 1;                        // unused
	k_scale_stride[2] = k_scale_blocks;                        // scale_blocks stride
	k_scale_stride[1] = Hk * k_scale_blocks;           // head stride
	k_scale_stride[0] = batch_size * Hk * k_scale_blocks;      // batch stride
	
	int v_scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	v_scale_stride[3] = 1;                              // channel stride (innermost)
	v_scale_stride[2] = D;                              // head stride
	v_scale_stride[1] = Hk * D;                         // batch stride
	v_scale_stride[0] = batch_size * Hk * D;            // outer stride (unused but set for consistency)
	
	printf("DEBUG: Our manual strides:\n");
	// printf("  q_int8_stride: [%d, %d, %d, %d]\n", q_int8_stride[0], q_int8_stride[1], q_int8_stride[2], q_int8_stride[3]);
	// printf("  k_int8_stride: [%d, %d, %d, %d]\n", k_int8_stride[0], k_int8_stride[1], k_int8_stride[2], k_int8_stride[3]);
	printf("  q_scale_stride: [%d, %d, %d, %d]\n", q_scale_stride[0], q_scale_stride[1], q_scale_stride[2], q_scale_stride[3]);
	printf("  k_scale_stride: [%d, %d, %d, %d]\n", k_scale_stride[0], k_scale_stride[1], k_scale_stride[2], k_scale_stride[3]);

	// Get CUDA stream
	// printf("DEBUG: stream_context pointer: %p\n", stream_context);
	cudaStream_t cuda_stream = ccv_nnc_stream_context_get_stream(stream_context);
	// printf("DEBUG: CUDA stream in flash_attn.cu: %p\n", cuda_stream);
	
	// If stream is null, use default stream (0)
	if (cuda_stream == nullptr) {
		// printf("DEBUG: Stream is null, using cudaStreamDefault (0)\n");
		cuda_stream = 0; // Use default stream  
	}

	// Call SageAttention FP8 function using workspace memory
	ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct(
		(half*)q->data.f16,           // query data
		(half*)k->data.f16,           // key data
		(half*)v->data.f16,           // value data
		v_transposed_workspace,       // v_transposed workspace
		q_int8_workspace,             // q_int8 output data (workspace)
		k_int8_workspace,             // k_int8 output data (workspace)
		v_fp8_workspace,              // v_fp8 output data (workspace)
		q_scale_workspace,            // query_scale data (workspace)
		k_scale_workspace,            // key_scale data (workspace)
		v_scale_workspace,            // value_scale data (workspace)
		(half*)o->data.f16,           // output data
		qdim,                         // query dimensions
		kdim,                         // key dimensions
		vdim,                         // value dimensions
		odim,                         // output dimensions
		q_int8_dim,                   // q_int8 dimensions
		k_int8_dim,                   // k_int8 dimensions
		v_fp8_dim,                    // v_fp8 dimensions
		q_scale_dim,                  // query_scale dimensions
		k_scale_dim,                  // key_scale dimensions
		v_scale_dim,                  // value_scale dimensions
		qstride,                      // query strides
		kstride,                      // key strides
		vstride,                      // value strides
		ostride,                      // output strides
		q_int8_stride,                // q_int8 strides
		k_int8_stride,                // k_int8 strides
		v_fp8_stride,                 // v_fp8 strides
		q_scale_stride,               // query_scale strides
		k_scale_stride,               // key_scale strides
		v_scale_stride,               // value_scale strides
		tensor_layout,                // tensor_layout: 0=NHD
		is_causal,                    // is_causal
		qk_quant_gran,                // qk_quant_gran: 2=per_warp
		sm_scale,                     // sm_scale
		return_lse,                   // return_lse
		(int)pv_accum_dtype,          // pv_accum_dtype: FP32
		cuda_stream                   // CUDA stream
	);

	CUDA_ENFORCE(cudaGetLastError());

	// Clean up descriptors created manually
	// cudnnDestroyReduceTensorDescriptor(reduce_mean);
	// cudnnDestroyTensorDescriptor(k_mean_desc);
	
	// Clean up CuDNN tensor view descriptor (this doesn't affect workspace)
	// ccv_nnc_cudnn_deinit_tensor_view_descriptor(k_desc);

	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Check if we should use SageAttention for INT8 quantized attention
	if ((cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_8U) || 
	    (cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_8U_32F)) {
		return _ccv_nnc_scaled_dot_product_attention_sage_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	}

	// NNC notation:
	// C = sm(Q * K^T) * V
	//
	// MFA notation:
	// O = sm(Q * K^T) * V
	assert(input_size >= 3);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 3 ? (ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const weights = input_size > 4 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 5 ? (ccv_nnc_tensor_view_t*)inputs[5] : 0;
	if (bias) // bias always requires a weight matrix.
		{ assert(weights); }

	ccv_nnc_tensor_view_t* const saved_softmax_lse = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_tensor_view_t* const o = (weights) ? (ccv_nnc_tensor_view_t*)outputs[2] : (ccv_nnc_tensor_view_t*)outputs[0];
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
	int amdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(o, odim);

	assert(q->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(k->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(v->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);
	if (attn_mask) {
		// MFA does not support fused transposes on the mask.
		assert(attn_mask->info.format == CCV_TENSOR_FORMAT_NHWC);
	}

	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(o));

	if (attn_mask) {
		assert(CCV_IS_TENSOR_CONTIGUOUS(attn_mask));
	}

	int batch_size;
	int R;
	int C;
	int Hq;
	int Hk;
	int D;
	if (q_nd == 3) {
		batch_size = qdim[1];
		assert(batch_size == kdim[1]);
		R = qdim[2];
		C = kdim[2];
		Hq = Hk = 1;
		D = qdim[3];
		assert(D == kdim[3]);
	} else if (q_nd == 4) {  // B S H D
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

	if (attn_mask) {
		// MFA can support am_nd == 2 and broadcast batch=1 -> batch=batch_size, but
		// wait until that occurs in practice before doing so.
		const int am_nd = ccv_nnc_tensor_nd(attn_mask->info.dim);
		assert(am_nd == 3 || am_nd == 4); // [batch_size, R, C]

		// MFA does not support attention mask broadcasting (where the R dimension
		// of Q > 1, but the R dimension of the mask == 1).
		ccv_nnc_tensor_view_get_dim(attn_mask, amdim);
		if (am_nd == 3)
		{
			assert(amdim[1] == batch_size || amdim[1] == 1);
			amdim[0] = amdim[1];
			amdim[1] = 1;
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		} else {
			assert(amdim[0] == batch_size || amdim[0] == 1);
			assert(amdim[1] == 1);
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		}
	}
	int weights_datatype = 0;
	if (weights)
		weights_datatype = CCV_GET_DATA_TYPE(weights->info.datatype) == CCV_QX ? ((weights->info.datatype & 0xff) << 12) : weights->info.datatype;

	const int is_same_dtype =
		(q->info.datatype == k->info.datatype) &&
		(q->info.datatype == v->info.datatype) &&
		(q->info.datatype == o->info.datatype) &&
		(weights ? (q->info.datatype == weights_datatype) : 1) &&
		(bias ? (q->info.datatype == bias->info.datatype) : 1);

	assert(is_same_dtype);

	Flash_fwd_params params;
	memset(&params, 0, sizeof(params));
	params.is_bf16 = q->info.datatype == CCV_16BF;
	params.q_ptr = q->data.u8;
	params.k_ptr = k->data.u8;
	params.v_ptr = v->data.u8;
	params.q_row_stride = D * Hq;
	params.k_row_stride = D * Hk;
	params.v_row_stride = D * Hk;
	params.q_head_stride = D;
	params.k_head_stride = D;
	params.v_head_stride = D;
	params.q_batch_stride = R * Hq * D;
	params.k_batch_stride = C * Hk * D;
	params.v_batch_stride = C * Hk * D;
	auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
	params.seqlen_q = R;
	params.seqlen_q_rounded = round_multiple(R, 128);
	params.seqlen_k = C;
	params.seqlen_k_rounded = round_multiple(C, 128);
	params.d = D;
	assert(D % 8 == 0);
	params.d_rounded = round_multiple(D, 32);
	params.o_ptr = o->data.u8;
	params.o_row_stride = D * Hq;
	params.o_head_stride = D;
	params.o_batch_stride = R * Hq * D;
	params.b = batch_size;
	params.h = Hq;
	params.h_k = Hk;
	params.h_h_k_ratio = Hq / Hk;
	params.scale_softmax = cmd.info.scaled_dot_product_attention.scale;
	params.scale_softmax_log2 = cmd.info.scaled_dot_product_attention.scale * M_LOG2E;
	params.is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	params.p_dropout = 1;
	params.p_dropout_in_uint8_t = 255;
	params.rp_dropout = 1;
	params.scale_softmax_rp_dropout = params.scale_softmax;
	params.window_size_left = ccv_max(R, C);
	params.window_size_right = params.is_causal ? 0 : ccv_max(R, C);
	params.is_seqlens_k_cumulative = true;
	const int block_n = D <= 64 ? 256 : (D <= 128 ? 128 : 64);
	const int num_n_blocks = (C + block_n - 1) / block_n;
	// Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
	// In any case we don't expect seqlen_q to be larger than 64 for inference.
	const int num_m_blocks = (R + 64 - 1) / 64;
	const ccv_nnc_cuda_device_prop_t props = ccv_nnc_gpu_device_props();
	// Only enable splitkv if R is 1.
	params.num_splits = R == 1 ? num_splits_heuristic(batch_size * Hq * num_m_blocks, props.multi_processor_count * 2, num_n_blocks, 128) : 1;
	if (saved_softmax_lse)
		params.softmax_lse_ptr = saved_softmax_lse->data.u8;
	if (params.num_splits > 1)
	{
		if (saved_softmax_lse)
		{
			float* const workspace = (float*)ccv_nnc_stream_context_get_workspace(stream_context, (params.num_splits * batch_size * Hq * R + params.num_splits * batch_size * Hq * R * params.d_rounded) * sizeof(float), CCV_TENSOR_GPU_MEMORY);
			params.softmax_lseaccum_ptr = workspace;
			params.oaccum_ptr = workspace + params.num_splits * batch_size * Hq * R;
		} else {
			float* const workspace = (float*)ccv_nnc_stream_context_get_workspace(stream_context, (batch_size * Hq * R + params.num_splits * batch_size * Hq * R + params.num_splits * batch_size * Hq * R * params.d_rounded) * sizeof(float), CCV_TENSOR_GPU_MEMORY);
			params.softmax_lse_ptr = workspace;
			params.softmax_lseaccum_ptr = workspace + batch_size * Hq * R;
			params.oaccum_ptr = workspace + batch_size * Hq * R + params.num_splits * batch_size * Hq * R;
		}
	} else if (!saved_softmax_lse) {
		void* const workspace = ccv_nnc_stream_context_get_workspace(stream_context, batch_size * Hq * R * sizeof(float), CCV_TENSOR_GPU_MEMORY);
		params.softmax_lse_ptr = workspace;
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	run_mha_fwd(params, stream, false);
	CUDA_ENFORCE(cudaGetLastError());
	if (weights)
	{
		const ccv_nnc_tensor_view_t* a = o;
		const ccv_nnc_tensor_view_t* w = weights;
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
		assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
		assert(CCV_IS_TENSOR_CONTIGUOUS(b));
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == 3);
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		const int w_nd = ccv_nnc_tensor_nd(w->info.dim);
		const int transpose_w[2] = {
			w_nd - 2, w_nd - 1
		};
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, transpose_w, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		int a_rows, a_cols;
		if (o_nd == 3) {
			a_rows = odim[1] * odim[2];
			a_cols = odim[3];
		} else if (q_nd == 4) {
			a_rows = odim[0] * odim[1];
			a_cols = odim[2] * odim[3];
		}
		int b_rows, b_cols, b_rows_inc;
		b_rows = b->info.dim[0] * b->info.dim[1];
		b_cols = b->info.dim[2];
		b_rows_inc = b_cols;
		assert(a_rows == b_rows);
		assert(a_cols == w_rows);
		assert(w_cols == b_cols);

		const cublasOperation_t transa = CUBLAS_OP_T;
		const cublasOperation_t transb = CUBLAS_OP_N;
		const int lda_inc = w_cols_inc;
		const int ldb_inc = a_cols;
		size_t w_data_size = 0;
		int w_datatype = w->info.datatype;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			w_datatype = (w_params.datatype & 0xff) << 12;
			ccv_nnc_tensor_param_t depalettize_w_params = w_params;
			depalettize_w_params.datatype = w_datatype;
			depalettize_w_params.reserved = 0;
			w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
		}
		const size_t cublas_size = ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size);
		void* workspace = 0;
		if (w_data_size > 0)
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, cublas_size + w_data_size, CCV_TENSOR_GPU_MEMORY);
		unsigned char* w_data = w->data.u8;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			const size_t count = ccv_nnc_tensor_count(w_params);
			const int qbits = (w_params.datatype & 0xf00) >> 8;
			const int number_in_blocks = w_params.reserved;
			w_data = (unsigned char*)workspace + cublas_size;
			ccv_nnc_compat_depalettize(w->data.u8, w_datatype, ccv_nnc_tensor_data_size_without_padding(w_params), qbits, number_in_blocks, w_data, count, stream_context);
		}
		cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
		static const half one_f16 = 1;
		static const float one_f32 = 1;
		static const double one_f64 = 1;
		static const double zero_f64 = 0;
		const void* zero = &zero_f64;
		const void* one;
		const int is_downcast = ((cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_16F) && b->info.datatype == CCV_16F);
		switch (ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast))
		{
			case CUBLAS_COMPUTE_16F:
				one = &one_f16;
				break;
			case CUBLAS_COMPUTE_32F:
			case CUBLAS_COMPUTE_32F_FAST_TF32:
				one = &one_f32;
				break;
			case CUBLAS_COMPUTE_64F:
				one = &one_f64;
				break;
			default:
				assert(0);
		}
		ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, cublas_size);
		if (bias)
		{
			int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
			const static int no_transpose[2] = {};
			ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
			assert(bias_batch_size == 1);
			assert(bias_cols == b_cols);
			assert(CCV_IS_TENSOR_CONTIGUOUS(bias));
			const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, b_rows, b->info.datatype);
			CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, device_ones, ccv_nnc_cuda_datatype(b->info.datatype), 1, zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w_data, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, one, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w_data, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_sum_out(const int B, const int Hk, const int r, const int D, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, B * Hk * D) {
		const int j = i / D;
		const int k = i % D;
		const NUM* const arow = a + j * r * D + k;
		float accum = (float)arow[0];
		for (int l = 1; l < r; l++)
			accum += (float)arow[l * D];
		b[i] = (NUM)accum;
	}
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// NNC notation:
	// C = sm(Q * K^T) * V
	//
	// MFA notation:
	// O = sm(Q * K^T) * V
	assert(input_size >= 6);
	assert(output_size >= 3);
	ccv_nnc_tensor_view_t* const d_o = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 6 ? (ccv_nnc_tensor_view_t*)inputs[6] : 0;
	ccv_nnc_tensor_view_t* const weights = input_size > 7 ? (ccv_nnc_tensor_view_t*)inputs[7] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 8 ? (ccv_nnc_tensor_view_t*)inputs[8] : 0;
	if (bias) // bias always requires a weight matrix.
		{ assert(weights); }
	ccv_nnc_tensor_view_t* const o = input_size > 9 ? (ccv_nnc_tensor_view_t*)inputs[9] : 0;
	ccv_nnc_tensor_view_t* const saved_softmax_lse = input_size > 10 ? (ccv_nnc_tensor_view_t*)inputs[10] : 0;
	// ccv_nnc_tensor_view_t* const qkv = input_size > 11 ? (ccv_nnc_tensor_view_t*)inputs[11] : 0;
	ccv_nnc_tensor_view_t* const dq = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dk = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dv = (ccv_nnc_tensor_view_t*)outputs[2];

	// Things we don't support.
	if (weights != 0)
		return CCV_NNC_EXEC_INVALID;

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
	int amdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(o, odim);

	assert(q->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(k->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(v->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);
	if (attn_mask) {
		// MFA does not support fused transposes on the mask.
		assert(attn_mask->info.format == CCV_TENSOR_FORMAT_NHWC);
	}

	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(o));
	assert(CCV_IS_TENSOR_CONTIGUOUS(d_o));
	assert(CCV_IS_TENSOR_CONTIGUOUS(saved_softmax_lse));

	if (attn_mask) {
		assert(CCV_IS_TENSOR_CONTIGUOUS(attn_mask));
	}

	int batch_size;
	int R;
	int C;
	int Hq;
	int Hk;
	int D;
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

	if (attn_mask) {
		// MFA can support am_nd == 2 and broadcast batch=1 -> batch=batch_size, but
		// wait until that occurs in practice before doing so.
		const int am_nd = ccv_nnc_tensor_nd(attn_mask->info.dim);
		assert(am_nd == 3 || am_nd == 4); // [batch_size, R, C]

		// MFA does not support attention mask broadcasting (where the R dimension
		// of Q > 1, but the R dimension of the mask == 1).
		ccv_nnc_tensor_view_get_dim(attn_mask, amdim);
		if (am_nd == 3)
		{
			assert(amdim[1] == batch_size || amdim[1] == 1);
			amdim[0] = amdim[1];
			amdim[1] = 1;
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		} else {
			assert(amdim[0] == batch_size || amdim[0] == 1);
			assert(amdim[1] == 1);
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		}
	}
	int weights_datatype = 0;
	if (weights)
		weights_datatype = CCV_GET_DATA_TYPE(weights->info.datatype) == CCV_QX ? ((weights->info.datatype & 0xff) << 12) : weights->info.datatype;

	const int is_same_dtype =
		(q->info.datatype == k->info.datatype) &&
		(q->info.datatype == v->info.datatype) &&
		(q->info.datatype == o->info.datatype) &&
		(weights ? (q->info.datatype == weights_datatype) : 1) &&
		(bias ? (q->info.datatype == bias->info.datatype) : 1);

	assert(is_same_dtype);

	Flash_bwd_params params;
	memset(&params, 0, sizeof(params));
	params.is_bf16 = q->info.datatype == CCV_16BF;
	params.q_ptr = q->data.u8;
	params.k_ptr = k->data.u8;
	params.v_ptr = v->data.u8;
	params.q_row_stride = D * Hq;
	params.k_row_stride = D * Hk;
	params.v_row_stride = D * Hk;
	params.q_head_stride = D;
	params.k_head_stride = D;
	params.v_head_stride = D;
	params.q_batch_stride = R * Hq * D;
	params.k_batch_stride = C * Hk * D;
	params.v_batch_stride = C * Hk * D;
	auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
	params.seqlen_q = R;
	params.seqlen_q_rounded = round_multiple(R, 128);
	params.seqlen_k = C;
	params.seqlen_k_rounded = round_multiple(C, 128);
	params.d = D;
	assert(D % 8 == 0);
	params.d_rounded = round_multiple(D, 32);
	params.o_ptr = o->data.u8;
	params.o_row_stride = D * Hq;
	params.o_head_stride = D;
	params.o_batch_stride = R * Hq * D;
	params.b = batch_size;
	params.h = Hq;
	params.h_k = Hk;
	params.h_h_k_ratio = Hq / Hk;
	params.scale_softmax = cmd.info.scaled_dot_product_attention.scale;
	params.scale_softmax_log2 = cmd.info.scaled_dot_product_attention.scale * M_LOG2E;
	params.is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	params.p_dropout = 1;
	params.p_dropout_in_uint8_t = 255;
	params.rp_dropout = 1;
	params.scale_softmax_rp_dropout = params.scale_softmax;
	params.window_size_left = ccv_max(R, C);
	params.window_size_right = params.is_causal ? 0 : ccv_max(R, C);
	params.is_seqlens_k_cumulative = true;
	params.dq_ptr = dq->data.u8;
	params.dk_ptr = dk->data.u8;
	params.dv_ptr = dv->data.u8;
	params.dq_row_stride = D * Hq;
	params.dk_row_stride = D * Hq; // This is not a typo, dk / dv is expanded and we sum it later.
	params.dv_row_stride = D * Hq;
	params.dq_head_stride = D;
	params.dk_head_stride = D;
	params.dv_head_stride = D;
	params.dq_batch_stride = R * Hq * D;
	params.dk_batch_stride = C * Hq * D;
	params.dv_batch_stride = C * Hq * D;
	params.do_ptr = d_o->data.u8;
	params.do_row_stride = D * Hq;
	params.do_head_stride = D;
	params.do_batch_stride = R * Hq * D;
	params.deterministic = cmd.info.scaled_dot_product_attention.deterministic;

	size_t dq_accum_size;
	if (params.deterministic)
	{
		const ccv_nnc_cuda_device_prop_t props = ccv_nnc_gpu_device_props();
		const int nsplits = (props.multi_processor_count + batch_size * Hq - 1) / (batch_size * Hq);
		dq_accum_size = sizeof(float) * nsplits * batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
		params.dq_accum_split_stride = batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
	} else {
		dq_accum_size = sizeof(float) * batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
		params.dq_accum_split_stride = 0;
	}

	params.softmax_lse_ptr = saved_softmax_lse->data.u8;
	if (Hq != Hk)
	{
		unsigned char* const workspace = (unsigned char*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size + sizeof(short) * batch_size * Hq * C * D * 2, CCV_TENSOR_GPU_MEMORY);
		params.dsoftmax_sum = workspace;
		params.dq_accum_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded;
		params.dk_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size;
		params.dv_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size + sizeof(short) * batch_size * Hq * C * D;
	} else {
		unsigned char* const workspace = (unsigned char*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size, CCV_TENSOR_GPU_MEMORY);
		params.dsoftmax_sum = workspace;
		params.dq_accum_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded;
		params.dk_accum_ptr = 0;
		params.dv_accum_ptr = 0;
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (params.deterministic)
		cudaMemsetAsync(params.dq_accum_ptr, 0, dq_accum_size, stream);
	run_mha_bwd(params, stream);
	CUDA_ENFORCE(cudaGetLastError());
	if (Hq != Hk)
	{
		_ccv_nnc_sum_out<<<CUDA_GET_BLOCKS(batch_size * C * Hk * D), CUDA_NUM_THREADS, 0, stream>>>(batch_size * C, Hk, Hq / Hk, D, (__half*)params.dk_ptr, (__half*)dk->data.f16);
		_ccv_nnc_sum_out<<<CUDA_GET_BLOCKS(batch_size * C * Hk * D), CUDA_NUM_THREADS, 0, stream>>>(batch_size * C, Hk, Hq / Hk, D, (__half*)params.dv_ptr, (__half*)dv->data.f16);
		CUDA_ENFORCE(cudaGetLastError());
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif


REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
#endif
}



REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
#endif
}

