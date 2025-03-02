#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int wdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(w, wdim);
	int wstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(w, wstride);
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(b, bstride);
	assert(w->info.format == CCV_TENSOR_FORMAT_NCHW);
	int biasdim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int biasstride[CCV_NNC_MAX_DIM_ALLOC] = {0};
	if (bias)
	{
		assert(CCV_GET_DATA_TYPE(bias->info.datatype) != CCV_QX);
		assert(ccv_nnc_tensor_nd(bias->info.dim) == 1);
		int i;
		for (i = 0; i < CCV_NNC_MAX_DIM + 2; i++)
			biasdim[i] = 1;
		int c;
		if (b->info.format == CCV_TENSOR_FORMAT_NCHW)
			c = 1;
		else if (b->info.format == CCV_TENSOR_FORMAT_NHWC)
			c = CCV_NNC_MAX_DIM + 1;
		else
			c = 0;
		biasdim[c] = bias->info.dim[0];
		if (CCV_IS_TENSOR_VIEW(bias))
		{
			for (i = 0; i < c; i++)
				biasstride[i] = bias->info.dim[0] * bias->stride[0];
			for (i = c; i < CCV_NNC_MAX_DIM + 2; i++)
				biasstride[i] = bias->stride[0];
		}
	}
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		const int w_datatype = CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX ? ((w->info.datatype & 0xff) << 12) : w->info.datatype;
		if (use_mfa) {
			const int is_same_dtype =
				(a->info.datatype == w_datatype) &&
				(a->info.datatype == b->info.datatype) &&
				(bias ? (a->info.datatype == bias->info.datatype) : 1);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (a->info.datatype) {
				case CCV_16F: {
					mtl_data_type = 16;
					break;
				}
				case CCV_32F: {
					mtl_data_type = 3;
					break;
				}
				default: {
					use_mfa = false;
					fallback_reason = "Unsupported data type.";
					break;
				}
			}
		}

		const int a_nd = ccv_nnc_tensor_nd(adim);
		const int w_nd = ccv_nnc_tensor_nd(w->info.dim);
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		int is_batched = 0;
		int a_batch_size;
		int w_batch_size;
		int b_batch_size;
		if (use_mfa) {
			a_batch_size = a_nd < 4 ? 1 : adim[a_nd - 4];
			int i;
			for (i = 0; i < a_nd - 4; i++)
				a_batch_size *= adim[i];
			w_batch_size = w_nd < 5 ? 1 : w->info.dim[w_nd - 5];
			for (i = 0; i < w_nd - 5; i++)
				w_batch_size *= w->info.dim[i];
			b_batch_size = b_nd < 4 ? 1 : b->info.dim[b_nd - 4];
			for (i = 0; i < b_nd - 4; i++)
				b_batch_size *= b->info.dim[i];
			assert(a_batch_size == b_batch_size || a_batch_size == 1);
			assert(w_batch_size == a_batch_size || w_batch_size == 1);

			// NNC uses the convention B = A * W.
			// MFA uses the convention C = A * B.
			int is_mfa_compatible_batch = 0;
			int A_batch_size = a_batch_size;
			int B_batch_size = w_batch_size;
			int C_batch_size = b_batch_size;
			if (A_batch_size == 1 && B_batch_size == 1 && C_batch_size == 1) {
				// Not batched.
			} else if (A_batch_size <= 0 || B_batch_size <= 0 || C_batch_size <= 0) {
				// Invalid batch size.
			} else {
				// This does not check whether the D batch size matches the others. If it
				// does not match, it will crash when encoding the GEMM command.
				is_batched = 1;
				if (A_batch_size == C_batch_size) {
					if (A_batch_size == B_batch_size) {
						is_mfa_compatible_batch = 1;
					} else if (B_batch_size == 1) {
						is_mfa_compatible_batch = 1;
					}
				}
			}

			if (is_batched && !is_mfa_compatible_batch) {
				use_mfa = false;
				fallback_reason = "Unsupported batch.";
			}

			// For simplicity, omit the logic for transposing the output matrix
			// between formats.
			if (a->info.format != b->info.format) {
				use_mfa = false;
				fallback_reason = "Image layout conversion.";
			}
		}

		if (use_mfa) {
			// Height and width of the filter, not the image.
			const int W = wdim[w_nd - 1];
			const int H = wdim[w_nd - 2];

			if ((H != 1) || (W != 1)) {
				use_mfa = false;
				fallback_reason = "Kernel size not 1x1.";
			} else if (hint.stride.dim[1] != 1 || hint.stride.dim[0] != 1) {
				use_mfa = false;
				fallback_reason = "Strided filter.";
			} else if (hint.border.begin[1] != 0 ||
								 hint.border.end[1] != 0 ||
								 hint.border.begin[0] != 0 ||
								 hint.border.end[0] != 0) {
				use_mfa = false;
				fallback_reason = "Padded.";
			} else if (cmd.info.convolution.groups != 1) {
				// Groups require batched GEMM, which is available in MFA. We won't add
				// support until we encounter a production use case with groups + 1x1
				// filters.
				use_mfa = false;
				fallback_reason = "Grouped.";
			}
		}

		if (use_mfa) {
			const int is_contiguous =
				(!CCV_IS_TENSOR_VIEW(a) || ccv_nnc_tensor_view_is_contiguous(adim, astride)) &&
				(!CCV_IS_TENSOR_VIEW(w) || ccv_nnc_tensor_view_is_contiguous(w->info.dim, w->stride)) &&
				(!CCV_IS_TENSOR_VIEW(b) || ccv_nnc_tensor_view_is_contiguous(b->info.dim, b->stride)) &&
				(bias ? (!CCV_IS_TENSOR_VIEW(bias) || ccv_nnc_tensor_view_is_contiguous(bias->info.dim, bias->stride)) : 1);
			if (!is_contiguous) {
				// There is one real-world example of a Conv1x1 with non-contiguous
				// tensors, but it's 1 out of 10-100 operations in the network.
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}

		if (METAL_LOG_LEVEL(context) >= 3) {
			if (use_mfa) {
				ccv_nnc_mfa_log_message("Compatible convolution found.");
			} else {
				ccv_nnc_mfa_log_message("Incompatible convolution found. Incompatible because:");
				ccv_nnc_mfa_log_message(fallback_reason);
			}
		}

		if (use_mfa) {
			mtl_buffer_t* w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
			size_t w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
			ccv_nnc_mfa_depalettize_params_t w_depalettize_params;
			if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
			{
				ccv_nnc_tensor_param_t w_params = w->info;
				const int palette_datatype = (w_params.datatype & 0xff) << 12;
				ccv_nnc_tensor_param_t depalettize_w_params = w_params;
				depalettize_w_params.datatype = palette_datatype;
				depalettize_w_params.reserved = 0;
				const size_t w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
				const size_t count = ccv_nnc_tensor_count(w_params);
				const int qbits = (w_params.datatype & 0xf00) >> 8;
				const int number_in_blocks = w_params.reserved;
				w_depalettize_params = (ccv_nnc_mfa_depalettize_params_t){
					.data_type = palette_datatype == CCV_16F ? 16 : 3,
					.qbits = (uint32_t)qbits,
					.number_in_blocks = (uint32_t)number_in_blocks,
					.length = (uint64_t)count,
				};
				ccv_nnc_mfa_prepare_depalettize(context, w_depalettize_params);
				w_data = ccv_nnc_mfa_request_scratch(context, w_data_size);
				w_dataof = 0;
			}

			int O;
			int H;
			int W;

			// Bypass a compilation error from a header.
			int I_dim;
			assert(a->info.format == b->info.format);
			if (a->info.format == CCV_TENSOR_FORMAT_NHWC) {
				// HWxI -> MxK
				I_dim = adim[a_nd - 1];
				W = adim[a_nd - 2];
				H = adim[a_nd - 3];
			} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW) {
				// IxHW -> KxM
				W = adim[a_nd - 1];
				H = adim[a_nd - 2];
				I_dim = adim[a_nd - 3];
			} else {
				// This should never happen.
				assert(false);
			}

			// OxI -> NxK
			assert(I_dim == wdim[w_nd - 3]);
			O = wdim[w_nd - 4];

			ccv_nnc_mfa_gemm_params_t params;
			if (a->info.format == CCV_TENSOR_FORMAT_NHWC)
			{
				params = (ccv_nnc_mfa_gemm_params_t){
					.data_type = mtl_data_type,
					.M = (uint32_t)(H * W),
					.N = (uint32_t)O,
					.K = (uint32_t)I_dim,
					.A_trans = 0,
					.B_trans = 1,
					.D_trans = 0,
					.fused_bias = (bias ? 1 : 0),
					.register_float = 0,

					.batch_dimension = b_batch_size,
					.batch_stride_a = a_batch_size > 1 ? H * W * I_dim : 0,
					.batch_stride_b = w_batch_size > 1 ? O * I_dim : 0,
					.batch_stride_c = b_batch_size > 1 ? H * W * O : 0,
					.batch_stride_d = 0,
				};
			} else {
				params = (ccv_nnc_mfa_gemm_params_t){
					.data_type = mtl_data_type,
					.M = (uint32_t)O,
					.N = (uint32_t)(H * W),
					.K = (uint32_t)I_dim,
					.A_trans = 0,
					.B_trans = 0,
					.D_trans = 1,
					.fused_bias = (bias ? 1 : 0),
					.register_float = 0,

					.batch_dimension = b_batch_size,
					.batch_stride_a = w_batch_size > 1 ? O * I_dim : 0,
					.batch_stride_b = a_batch_size > 1 ? H * W * I_dim : 0,
					.batch_stride_c = b_batch_size > 1 ? H * W * O : 0,
					.batch_stride_d = 0,
				};
			}

			ccv_nnc_mfa_prepare_gemm(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
			{
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer((ccv_nnc_tensor_t*)w), // A
					(mtl_buffer_t*)w_data, // B
					NULL,
				};
				size_t tensor_offsets[2] = {
					w->dataof, // A offset
					0, // B offset
				};
				ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
			}
			mtl_buffer_t* bias_buffer = NULL;
			if (bias) {
				bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
			}
			if (a->info.format == CCV_TENSOR_FORMAT_NHWC)
			{
				mtl_buffer_t* tensors[5] = {
					mpgetbuffer((ccv_nnc_tensor_t*)a), // A
					w_data, // B
					mpgetbuffer((ccv_nnc_tensor_t*)b), // C
					bias_buffer, // D
					NULL,
				};
				size_t tensor_offsets[4] = {
					a->dataof, // A offset
					w_dataof, // B offset
					b->dataof, // C offset
					bias ? bias->dataof : 0, // D offset
				};
				ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);
			} else {
				mtl_buffer_t* tensors[5] = {
					w_data, // A
					mpgetbuffer((ccv_nnc_tensor_t*)a), // B
					mpgetbuffer((ccv_nnc_tensor_t*)b), // C
					bias_buffer, // D
					NULL,
				};
				size_t tensor_offsets[4] = {
					w_dataof, // A offset
					a->dataof, // B offset
					b->dataof, // C offset
					bias ? bias->dataof : 0, // D offset
				};
				ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);
			}
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			mtl_buffer_t* w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
			size_t w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
			MPSCommandBuffer* command_buffer;
			if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
			{
				ccv_nnc_tensor_param_t w_params = w->info;
				const int palette_datatype = (w_params.datatype & 0xff) << 12;
				ccv_nnc_tensor_param_t depalettize_w_params = w_params;
				depalettize_w_params.datatype = palette_datatype;
				depalettize_w_params.reserved = 0;
				size_t w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
				const size_t count = ccv_nnc_tensor_count(w_params);
				const int qbits = (w_params.datatype & 0xf00) >> 8;
				const int number_in_blocks = w_params.reserved;
				ccv_nnc_mfa_depalettize_params_t w_depalettize_params = {
					.data_type = palette_datatype == CCV_16F ? 16 : 3,
					.qbits = (uint32_t)qbits,
					.number_in_blocks = (uint32_t)number_in_blocks,
					.length = (uint64_t)count,
				};
				ccv_nnc_mfa_prepare_depalettize(context, w_depalettize_params);
				w_data = ccv_nnc_mfa_request_scratch(context, w_data_size);
				w_dataof = 0;
				mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer((ccv_nnc_tensor_t*)w), // A
					(mtl_buffer_t*)w_data, // B
					NULL,
				};
				size_t tensor_offsets[2] = {
					w->dataof, // A offset
					0, // B offset
				};
				ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				command_buffer = ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
			} else
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int* adim_r = adim;
			int* astride_r = astride;
			int* wdim_r = wdim;
			int* wstride_r = wstride;
			int* biasdim_r = biasdim;
			int* biasstride_r = biasstride;
			int indices[3];
			const int dilationX = ccv_max(cmd.info.convolution.dilation[1], 1);
			const int dilationY = ccv_max(cmd.info.convolution.dilation[0], 1);
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_input_w;
				MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, wdim_r, wstride_r, &mps_input_w);
				[inputTensors addObject:mps_input_w];
				MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, wdim_r, wstride_r);
				[inputShapedTypes addObject:mps_w_shape];
				MPSGraphConvolution2DOpDescriptor* descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:dilationX dilationRateInY:dilationY groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:ccv_nnc_mps_tensor_data_layout(a->info.format) weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
				MPSGraphTensor* mps_b = [graph convolution2DWithSourceTensor:mps_a weightsTensor:mps_w descriptor:descriptor name:nil];
				if (bias)
				{
					MPSGraphTensor* mps_input_bias;
					MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, bias, biasdim_r, biasstride_r, &mps_input_bias);
					[inputTensors addObject:mps_input_bias];
					MPSGraphShapedType* mps_bias_shape = ccv_nnc_mps_graph_tensor_input_shape(bias, biasdim_r, biasstride_r);
					[inputShapedTypes addObject:mps_bias_shape];
					// Add support broadcast directly.
					mps_b = [graph additionWithPrimaryTensor:mps_b secondaryTensor:mps_bias name:nil];
				}
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data_with_buffer(w, wdim, wstride, w_data, w_dataof);
			if (bias)
			{
				MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, biasdim, biasstride);
				MPSGraphTensorData* data[] = {data_a, data_w, data_bias};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1, 0);
			} else {
				MPSGraphTensorData* data[] = {data_a, data_w};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, (no bias updates yet)
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0]; // gradients input
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1]; // forward input
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2]; // weights input

	ccv_nnc_tensor_view_t* dw = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0; // weight_update
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0]; // output gradients
	ccv_nnc_tensor_view_t* db = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	const int dilationX = ccv_max(cmd.info.convolution.dilation[1], 1);
	const int dilationY = ccv_max(cmd.info.convolution.dilation[0], 1);

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = 0;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (h) {
			mtl_buffer_t* w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
			size_t w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
			if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
			{
				ccv_nnc_tensor_param_t w_params = w->info;
				const int palette_datatype = (w_params.datatype & 0xff) << 12;
				ccv_nnc_tensor_param_t depalettize_w_params = w_params;
				depalettize_w_params.datatype = palette_datatype;
				depalettize_w_params.reserved = 0;
				size_t w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
				const size_t count = ccv_nnc_tensor_count(w_params);
				const int qbits = (w_params.datatype & 0xf00) >> 8;
				const int number_in_blocks = w_params.reserved;
				ccv_nnc_mfa_depalettize_params_t w_depalettize_params = {
					.data_type = palette_datatype == CCV_16F ? 16 : 3,
					.qbits = (uint32_t)qbits,
					.number_in_blocks = (uint32_t)number_in_blocks,
					.length = (uint64_t)count,
				};
				ccv_nnc_mfa_prepare_depalettize(context, w_depalettize_params);
				w_data = ccv_nnc_mfa_request_scratch(context, w_data_size);
				w_dataof = 0;
				mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer((ccv_nnc_tensor_t*)w), // A
					(mtl_buffer_t*)w_data, // B
					NULL,
				};
				size_t tensor_offsets[2] = {
					w->dataof, // A offset
					0, // B offset
				};
				ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				command_buffer = ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
			} else
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			// [output gradient]
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];

			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_g;
				MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
				[inputTensors addObject:mps_input_g];
				MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
				[inputShapedTypes addObject:mps_g_shape];

				MPSGraphTensor* mps_input_w;
				MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, w->info.dim, w->stride, &mps_input_w);
				[inputTensors addObject:mps_input_w];
				MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, w->info.dim, w->stride);
				[inputShapedTypes addObject:mps_w_shape];

				NSMutableArray<NSNumber*>* h_shape = [NSMutableArray new];
				const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
				int i;
				for (i = 0; i < h_nd; i++)
					[h_shape addObject:@(h->info.dim[i])];
				MPSGraphConvolution2DOpDescriptor* descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:dilationX dilationRateInY:dilationY groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:ccv_nnc_mps_tensor_data_layout(g->info.format) weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
				MPSGraphTensor* mps_h = [graph convolution2DDataGradientWithIncomingGradientTensor:mps_g
																			weightsTensor:mps_w
																				outputShape:h_shape
															forwardConvolutionDescriptor:descriptor
																					name:nil];
				[h_shape release];
				[resultTensors addObject:mps_h];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data_with_buffer(w, w->info.dim, w->stride, w_data, w_dataof);
			MPSGraphTensorData* data[] = {data_g, data_w};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
		}

		if (dw) {
			if (!command_buffer)
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			// [weight updates]
			ccv_nnc_mps_graph_key_t dw_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
			int dw_indices[2];

			MPSGraphExecutable* executable_dw = ccv_nnc_mps_graph_executable_cache(dw_key, dw_indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_g;
				MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
				[inputTensors addObject:mps_input_g];
				MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
				[inputShapedTypes addObject:mps_g_shape];

				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];

				NSMutableArray<NSNumber*>* dw_shape = [NSMutableArray new];
				const int dw_nd = ccv_nnc_tensor_nd(dw->info.dim);
				int i;
				for (i = 0; i < dw_nd; i++)
					[dw_shape addObject:@(dw->info.dim[i])];
				MPSGraphConvolution2DOpDescriptor* dw_descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:dilationX dilationRateInY:dilationY groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:ccv_nnc_mps_tensor_data_layout(g->info.format) weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

				MPSGraphTensor* mps_dw = [graph convolution2DWeightsGradientWithIncomingGradientTensor:mps_g
																				sourceTensor:mps_a
																				outputShape:dw_shape
															forwardConvolutionDescriptor:dw_descriptor
																					name:nil];
				[dw_shape release];
				[resultTensors addObject:mps_dw];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data[] = {data_g, data_a};
			ccv_nnc_mps_graph_executable_result(executable_dw, command_buffer, @[data[dw_indices[0]], data[dw_indices[1]]], &dw , (int*[]){ dw->info.dim }, (int*[]){ dw->stride }, 1, 0);
		}

		if (db) {
			if (!command_buffer)
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			// [bias updates]
			ccv_nnc_mps_graph_key_t db_key = ccv_nnc_mps_graph_key_new(cmd, 2, hint, flags, inputs, input_size, outputs, output_size);
			int db_indices[1];

			MPSGraphExecutable* executable_db = ccv_nnc_mps_graph_executable_cache(db_key, db_indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_g;
				MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
				[inputTensors addObject:mps_input_g];
				MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
				[inputShapedTypes addObject:mps_g_shape];
				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
				int i;
				for (i = 0; i < g_nd; i++) {
					if (g->info.dim[i] != db->info.dim[i])
						[axes addObject:@(i)];
				}
				MPSGraphTensor* mps_db = [graph reductionSumWithTensor:mps_g axes:axes name:nil];
				[axes release];
				[resultTensors addObject:mps_db];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			ccv_nnc_mps_graph_executable_result(executable_db, command_buffer, @[data_g], &db, (int*[]){ db->info.dim }, (int*[]){ dw->info.dim }, 1, 0);
		}

		if (command_buffer)
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_back;
}
