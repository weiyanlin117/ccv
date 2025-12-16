#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_cmul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const ccv_nnc_tensor_t* const b = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(a->info.datatype == b->info.datatype) &&
				(a->info.datatype == c->info.datatype);
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

		if (use_mfa) {
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) ||
					!CCV_IS_TENSOR_CONTIGUOUS(b) ||
					!CCV_IS_TENSOR_CONTIGUOUS(c))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_cmul_params_t params = {
				.conjugate = 0,
				.data_type = mtl_data_type,
				.astride = {0, 0, 0},
				.bstride = {0, 0, 0},
				.cstride = {0, 0, 0},
				.dim = {0, 0, 0, 0}
			};
			const size_t count = ccv_nnc_tensor_count(c->info);
			if (ccv_nnc_tensor_count(a->info) == count && ccv_nnc_tensor_count(b->info) == count) {
				params.dim[0] = count;
			} else {
				int i;
				int nd = ccv_nnc_tensor_nd(a->info.dim);
				assert(nd = ccv_nnc_tensor_nd(b->info.dim));
				assert(nd = ccv_nnc_tensor_nd(c->info.dim));
				int adim[CCV_NNC_MAX_DIM_ALLOC];
				int bdim[CCV_NNC_MAX_DIM_ALLOC];
				int cdim[CCV_NNC_MAX_DIM_ALLOC];
				int squeezed_dims = 0;
				for (i = nd - 1; i >= 0; i--)
				{
					if (c->info.dim[i] == 1)
						continue;
					adim[squeezed_dims] = a->info.dim[i];
					bdim[squeezed_dims] = b->info.dim[i];
					cdim[squeezed_dims] = c->info.dim[i];
					squeezed_dims += 1;
				}
				nd = squeezed_dims;
				int astride[CCV_NNC_MAX_DIM_ALLOC];
				int bstride[CCV_NNC_MAX_DIM_ALLOC];
				int cstride[CCV_NNC_MAX_DIM_ALLOC];
				astride[0] = 1;
				bstride[0] = 1;
				cstride[0] = 1;
				for (i = 1; i < nd; i++)
				{
					astride[i] = adim[i - 1] * astride[i - 1];
					bstride[i] = bdim[i - 1] * bstride[i - 1];
					cstride[i] = cdim[i - 1] * cstride[i - 1];
				}
				for (i = 0; i < nd; i++)
				{
					if (cdim[i] == adim[i] && cdim[i] == bdim[i])
						continue;
					if (cdim[i] == adim[i])
					{
						assert(bdim[i] == 1);
						bstride[i] = 0;
					} else {
						assert(cdim[i] == bdim[i]);
						assert(adim[i] == 1);
						astride[i] = 0;
					}
				}
				assert(nd <= 4);
				params.dim[0] = cdim[0];
				params.dim[1] = cdim[1];
				params.dim[2] = cdim[2];
				params.dim[3] = cdim[3];
				for (i = nd; i < 4; i++)
					params.dim[i] = 0;
				params.astride[0] = astride[1];
				params.astride[1] = astride[2];
				params.astride[2] = astride[3];
				params.bstride[0] = bstride[1];
				params.bstride[1] = bstride[2];
				params.bstride[2] = bstride[3];
				params.cstride[0] = cstride[1];
				params.cstride[1] = cstride[2];
				params.cstride[2] = cstride[3];
			}
			ccv_nnc_mfa_prepare_cmul(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]), // gradient
				mpgetbuffer(inputs[1]), // source
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[3] = {
				a->dataof,
				b->dataof,
				c->dataof
			};
			ccv_nnc_mfa_encode_cmul(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
			const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
			ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			int nd = ccv_nnc_tensor_nd(a->info.dim);
			assert(nd == ccv_nnc_tensor_nd(b->info.dim));
			assert(nd == ccv_nnc_tensor_nd(c->info.dim));
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
				[inputShapedTypes addObject:mps_b_shape];
				int i;
				// Reshape to [..., n / 2, 2]
				NSMutableArray<NSNumber*>* a_shape = [NSMutableArray new];
				for (i = 0; i < nd - 1; i++)
					[a_shape addObject:@(a->info.dim[i])];
				[a_shape addObject: @(a->info.dim[nd - 1] / 2)];
				[a_shape addObject: @2];
				mps_a = [graph reshapeTensor:mps_a withShape:a_shape name:nil];
				[a_shape release];
				NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_a numSplits:2 axis:nd name:nil];
				NSMutableArray<NSNumber*>* b_shape = [NSMutableArray new];
				for (i = 0; i < nd - 1; i++)
					[b_shape addObject:@(b->info.dim[i])];
				[b_shape addObject: @(b->info.dim[nd - 1] / 2)];
				[b_shape addObject: @2];
				mps_b = [graph reshapeTensor:mps_b withShape:b_shape name:nil];
				[b_shape release];
				NSArray<MPSGraphTensor*>* mps_b_splits = [graph splitTensor:mps_b numSplits:2 axis:nd name:nil];
				MPSGraphTensor* mps_c_0 = [graph subtractionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
				MPSGraphTensor* mps_c_1 = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[1] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[0] name:nil] name:nil];
				NSMutableArray<NSNumber*>* c_shape = [NSMutableArray new];
				for (i = 0; i < nd; i++)
					[c_shape addObject:@(c->info.dim[i])];
				MPSGraphTensor* mps_c = [graph reshapeTensor:[graph concatTensor:mps_c_0 withTensor:mps_c_1 dimension:nd name:nil] withShape:c_shape name:nil];
				[resultTensors addObject:mps_c];
				[c_shape release];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data[] = {data_a, data_b};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_cmul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Detect if broadcasting was used in the forward pass.
	// Forward: output = a * b, where a or b may have been broadcast to match output shape.
	// inputs[0] = g (gradient, same shape as forward output)
	// inputs[1] = a (first input from forward)
	// inputs[2] = b (second input from forward)
	// outputs[0] = da (gradient w.r.t. a, same shape as a)
	// outputs[1] = db (gradient w.r.t. b, same shape as b)
	// no_broadcasting=1 means all tensors have matching dimensions (no reduction needed)
	// no_broadcasting=0 means some input was broadcast, requiring gradient reduction in MPSGraph path
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int no_broadcasting = 1;
	if (outputs[0])
	{
		assert(input_size >= 3 && inputs[2]);
		ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)outputs[0], gdim);
		ccv_nnc_tensor_view_get_broadcast_dim((ccv_nnc_tensor_view_t*)inputs[2], gdim);
		no_broadcasting = no_broadcasting && (ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[0], gdim) && ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[2], gdim));
	}
	if (no_broadcasting && output_size > 1 && outputs[1])
	{
		assert(inputs[1]);
		ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[1], gdim);
		ccv_nnc_tensor_view_get_broadcast_dim((ccv_nnc_tensor_view_t*)outputs[1], gdim);
		no_broadcasting = no_broadcasting && (ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)inputs[1], gdim) && ccv_nnc_tensor_view_check_dim((ccv_nnc_tensor_view_t*)outputs[1], gdim));
	}
	// Broadcasting is supported in MPSGraph path via reduction.
	const ccv_nnc_tensor_t* const g = inputs[0];
	if (!g)
		return CCV_NNC_EXEC_INVALID;
	assert(!g || CCV_IS_TENSOR_CONTIGUOUS(g));
	const ccv_nnc_tensor_t* const a = input_size >= 2 ? inputs[1] : 0;
	assert(!a || CCV_IS_TENSOR_CONTIGUOUS(a));
	ccv_nnc_tensor_t* const b = input_size >= 3 ? inputs[2] : 0;
	assert(!b || CCV_IS_TENSOR_CONTIGUOUS(b));
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(!c || CCV_IS_TENSOR_CONTIGUOUS(c));
	ccv_nnc_tensor_t* const d = output_size >= 2 ? outputs[1] : 0;
	assert(!d || CCV_IS_TENSOR_CONTIGUOUS(d));
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(!g || !a || g->info.datatype == a->info.datatype) &&
				(!a || !b || a->info.datatype == b->info.datatype) &&
				(!g || !b || g->info.datatype == b->info.datatype) &&
				(!a || !d || a->info.datatype == d->info.datatype) &&
				(!b || !c || b->info.datatype == c->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			int datatype = 0;
			if (a)
				datatype = a->info.datatype;
			else if (b)
				datatype = b->info.datatype;
			switch (datatype) {
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

		if (use_mfa) {
			if ((a && !CCV_IS_TENSOR_CONTIGUOUS(a)) ||
				(b && !CCV_IS_TENSOR_CONTIGUOUS(b)) ||
				(c && !CCV_IS_TENSOR_CONTIGUOUS(c)) ||
				(d && !CCV_IS_TENSOR_CONTIGUOUS(d)) ||
				(g && !CCV_IS_TENSOR_CONTIGUOUS(g)))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_cmul_params_t params = {
				.conjugate = 1,
				.data_type = mtl_data_type,
				.astride = {0, 0, 0},
				.bstride = {0, 0, 0},
				.cstride = {0, 0, 0},
				.dim = {0, 0, 0, 0}
			};
			// Check if we need reduction for c (da) or d (db) due to broadcasting
			const size_t g_count = ccv_nnc_tensor_count(g->info);
			const size_t c_count = c ? ccv_nnc_tensor_count(c->info) : 0;
			const size_t d_count = d ? ccv_nnc_tensor_count(d->info) : 0;
			const int c_needs_reduction = (b && c && c_count < g_count);
			const int d_needs_reduction = (a && d && d_count < g_count);
			// Allocate temp buffers if needed for reduction
			mtl_buffer_t* c_temp_buffer = NULL;
			mtl_buffer_t* d_temp_buffer = NULL;
			if (c_needs_reduction)
				c_temp_buffer = (mtl_buffer_t*)mpobjmalloc(0, CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * g_count);
			if (d_needs_reduction)
				d_temp_buffer = (mtl_buffer_t*)mpobjmalloc(0, CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * g_count);
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			if (g)
			{
				if (b && c)
				{
					const size_t count = ccv_nnc_tensor_count(c->info);
					if (ccv_nnc_tensor_count(g->info) == count && ccv_nnc_tensor_count(b->info) == count) {
						params.dim[0] = count;
					} else {
						int i;
						int nd = ccv_nnc_tensor_nd(g->info.dim);
						assert(nd = ccv_nnc_tensor_nd(b->info.dim));
						assert(nd = ccv_nnc_tensor_nd(c->info.dim));
						int adim[CCV_NNC_MAX_DIM_ALLOC];
						int bdim[CCV_NNC_MAX_DIM_ALLOC];
						int cdim[CCV_NNC_MAX_DIM_ALLOC];
						int squeezed_dims = 0;
						for (i = nd - 1; i >= 0; i--)
						{
							if (c->info.dim[i] == 1)
								continue;
							adim[squeezed_dims] = g->info.dim[i];
							bdim[squeezed_dims] = b->info.dim[i];
							cdim[squeezed_dims] = c->info.dim[i];
							squeezed_dims += 1;
						}
						nd = squeezed_dims;
						int astride[CCV_NNC_MAX_DIM_ALLOC];
						int bstride[CCV_NNC_MAX_DIM_ALLOC];
						int cstride[CCV_NNC_MAX_DIM_ALLOC];
						astride[0] = 1;
						bstride[0] = 1;
						cstride[0] = 1;
						for (i = 1; i < nd; i++)
						{
							astride[i] = adim[i - 1] * astride[i - 1];
							bstride[i] = bdim[i - 1] * bstride[i - 1];
							cstride[i] = cdim[i - 1] * cstride[i - 1];
						}
						for (i = 0; i < nd; i++)
						{
							if (cdim[i] == adim[i] && cdim[i] == bdim[i])
								continue;
							if (cdim[i] == adim[i])
							{
								assert(bdim[i] == 1);
								bstride[i] = 0;
							} else {
								assert(cdim[i] == bdim[i]);
								assert(adim[i] == 1);
								astride[i] = 0;
							}
						}
						assert(nd <= 4);
						params.dim[0] = cdim[0];
						params.dim[1] = cdim[1];
						params.dim[2] = cdim[2];
						params.dim[3] = cdim[3];
						for (i = nd; i < 4; i++)
							params.dim[i] = 0;
						params.astride[0] = astride[1];
						params.astride[1] = astride[2];
						params.astride[2] = astride[3];
						params.bstride[0] = bstride[1];
						params.bstride[1] = bstride[2];
						params.bstride[2] = bstride[3];
						params.cstride[0] = cstride[1];
						params.cstride[1] = cstride[2];
						params.cstride[2] = cstride[3];
					}
					// When c needs reduction, recompute params to output g's shape
					if (c_needs_reduction)
					{
						int i;
						int nd = ccv_nnc_tensor_nd(g->info.dim);
						int gdim[CCV_NNC_MAX_DIM_ALLOC];
						int bdim_arr[CCV_NNC_MAX_DIM_ALLOC];
						int squeezed_dims = 0;
						for (i = nd - 1; i >= 0; i--)
						{
							if (g->info.dim[i] == 1)
								continue;
							gdim[squeezed_dims] = g->info.dim[i];
							bdim_arr[squeezed_dims] = b->info.dim[i];
							squeezed_dims += 1;
						}
						nd = squeezed_dims;
						int gstride[CCV_NNC_MAX_DIM_ALLOC];
						int bstride_arr[CCV_NNC_MAX_DIM_ALLOC];
						gstride[0] = 1;
						bstride_arr[0] = 1;
						for (i = 1; i < nd; i++)
						{
							gstride[i] = gdim[i - 1] * gstride[i - 1];
							bstride_arr[i] = bdim_arr[i - 1] * bstride_arr[i - 1];
						}
						for (i = 0; i < nd; i++)
						{
							if (gdim[i] != bdim_arr[i])
							{
								assert(bdim_arr[i] == 1);
								bstride_arr[i] = 0;
							}
						}
						params.dim[0] = gdim[0];
						params.dim[1] = nd > 1 ? gdim[1] : 0;
						params.dim[2] = nd > 2 ? gdim[2] : 0;
						params.dim[3] = nd > 3 ? gdim[3] : 0;
						params.astride[0] = nd > 1 ? gstride[1] : 0;
						params.astride[1] = nd > 2 ? gstride[2] : 0;
						params.astride[2] = nd > 3 ? gstride[3] : 0;
						params.bstride[0] = nd > 1 ? bstride_arr[1] : 0;
						params.bstride[1] = nd > 2 ? bstride_arr[2] : 0;
						params.bstride[2] = nd > 3 ? bstride_arr[3] : 0;
						params.cstride[0] = nd > 1 ? gstride[1] : 0;
						params.cstride[1] = nd > 2 ? gstride[2] : 0;
						params.cstride[2] = nd > 3 ? gstride[3] : 0;
					}
					ccv_nnc_mfa_prepare_cmul(context, params);

					mtl_buffer_t* tensors[4] = {
						mpgetbuffer(g), // gradient
						mpgetbuffer(b), // source
						c_needs_reduction ? c_temp_buffer : mpgetbuffer(c), // destination
						NULL,
					};
					size_t tensor_offsets[3] = {
						g->dataof,
						b->dataof,
						c_needs_reduction ? 0 : c->dataof
					};
					ccv_nnc_mfa_encode_cmul(context, params, command_batch, tensors, tensor_offsets);
				}
				if (a && d)
				{
					const size_t count = ccv_nnc_tensor_count(d->info);
					if (ccv_nnc_tensor_count(g->info) == count && ccv_nnc_tensor_count(a->info) == count) {
						params.dim[0] = count;
					} else {
						int i;
						int nd = ccv_nnc_tensor_nd(g->info.dim);
						assert(nd = ccv_nnc_tensor_nd(a->info.dim));
						assert(nd = ccv_nnc_tensor_nd(d->info.dim));
						int adim[CCV_NNC_MAX_DIM_ALLOC];
						int bdim[CCV_NNC_MAX_DIM_ALLOC];
						int cdim[CCV_NNC_MAX_DIM_ALLOC];
						int squeezed_dims = 0;
						for (i = nd - 1; i >= 0; i--)
						{
							if (c->info.dim[i] == 1)
								continue;
							adim[squeezed_dims] = g->info.dim[i];
							bdim[squeezed_dims] = a->info.dim[i];
							cdim[squeezed_dims] = d->info.dim[i];
							squeezed_dims += 1;
						}
						nd = squeezed_dims;
						int astride[CCV_NNC_MAX_DIM_ALLOC];
						int bstride[CCV_NNC_MAX_DIM_ALLOC];
						int cstride[CCV_NNC_MAX_DIM_ALLOC];
						astride[0] = 1;
						bstride[0] = 1;
						cstride[0] = 1;
						for (i = 1; i < nd; i++)
						{
							astride[i] = adim[i - 1] * astride[i - 1];
							bstride[i] = bdim[i - 1] * bstride[i - 1];
							cstride[i] = cdim[i - 1] * cstride[i - 1];
						}
						for (i = 0; i < nd; i++)
						{
							if (cdim[i] == adim[i] && cdim[i] == bdim[i])
								continue;
							if (cdim[i] == adim[i])
							{
								assert(bdim[i] == 1);
								bstride[i] = 0;
							} else {
								assert(cdim[i] == bdim[i]);
								assert(adim[i] == 1);
								astride[i] = 0;
							}
						}
						assert(nd <= 4);
						params.dim[0] = cdim[0];
						params.dim[1] = cdim[1];
						params.dim[2] = cdim[2];
						params.dim[3] = cdim[3];
						for (i = nd; i < 4; i++)
							params.dim[i] = 0;
						params.astride[0] = astride[1];
						params.astride[1] = astride[2];
						params.astride[2] = astride[3];
						params.bstride[0] = bstride[1];
						params.bstride[1] = bstride[2];
						params.bstride[2] = bstride[3];
						params.cstride[0] = cstride[1];
						params.cstride[1] = cstride[2];
						params.cstride[2] = cstride[3];
					}
					// When d needs reduction, recompute params to output g's shape
					if (d_needs_reduction)
					{
						int i;
						int nd = ccv_nnc_tensor_nd(g->info.dim);
						int gdim[CCV_NNC_MAX_DIM_ALLOC];
						int adim_arr[CCV_NNC_MAX_DIM_ALLOC];
						int squeezed_dims = 0;
						for (i = nd - 1; i >= 0; i--)
						{
							if (g->info.dim[i] == 1)
								continue;
							gdim[squeezed_dims] = g->info.dim[i];
							adim_arr[squeezed_dims] = a->info.dim[i];
							squeezed_dims += 1;
						}
						nd = squeezed_dims;
						int gstride[CCV_NNC_MAX_DIM_ALLOC];
						int astride_arr[CCV_NNC_MAX_DIM_ALLOC];
						gstride[0] = 1;
						astride_arr[0] = 1;
						for (i = 1; i < nd; i++)
						{
							gstride[i] = gdim[i - 1] * gstride[i - 1];
							astride_arr[i] = adim_arr[i - 1] * astride_arr[i - 1];
						}
						for (i = 0; i < nd; i++)
						{
							if (gdim[i] != adim_arr[i])
							{
								assert(adim_arr[i] == 1);
								astride_arr[i] = 0;
							}
						}
						params.dim[0] = gdim[0];
						params.dim[1] = nd > 1 ? gdim[1] : 0;
						params.dim[2] = nd > 2 ? gdim[2] : 0;
						params.dim[3] = nd > 3 ? gdim[3] : 0;
						params.astride[0] = nd > 1 ? gstride[1] : 0;
						params.astride[1] = nd > 2 ? gstride[2] : 0;
						params.astride[2] = nd > 3 ? gstride[3] : 0;
						params.bstride[0] = nd > 1 ? astride_arr[1] : 0;
						params.bstride[1] = nd > 2 ? astride_arr[2] : 0;
						params.bstride[2] = nd > 3 ? astride_arr[3] : 0;
						params.cstride[0] = nd > 1 ? gstride[1] : 0;
						params.cstride[1] = nd > 2 ? gstride[2] : 0;
						params.cstride[2] = nd > 3 ? gstride[3] : 0;
					}
					ccv_nnc_mfa_prepare_cmul(context, params);

					mtl_buffer_t* tensors[4] = {
						mpgetbuffer(g), // gradient
						mpgetbuffer(a), // source
						d_needs_reduction ? d_temp_buffer : mpgetbuffer(d), // destination
						NULL,
					};
					size_t tensor_offsets[3] = {
						g->dataof,
						a->dataof,
						d_needs_reduction ? 0 : d->dataof
					};
					ccv_nnc_mfa_encode_cmul(context, params, command_batch, tensors, tensor_offsets);
				}
			}
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			// If we need reduction, use MPSGraph to reduce from temp buffer to final output
			if (c_needs_reduction || d_needs_reduction)
			{
				MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
				if (c_needs_reduction)
				{
					const ccv_nnc_tensor_view_t* const gv = (const ccv_nnc_tensor_view_t*)g;
					ccv_nnc_tensor_view_t* const cv = (ccv_nnc_tensor_view_t*)c;
					// Create a fake tensor_view for the temp buffer with g's shape
					ccv_nnc_tensor_view_t temp_view = *gv;
					ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 2, hint, flags, inputs, input_size, outputs, output_size);
					int indices[1];
					const int nd = ccv_nnc_tensor_nd(gv->info.dim);
					MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
						MPSGraphTensor* mps_input;
						MPSGraphTensor* mps_temp = ccv_nnc_mps_graph_tensor_input(graph, &temp_view, gv->info.dim, gv->stride, &mps_input);
						[inputTensors addObject:mps_input];
						MPSGraphShapedType* mps_temp_shape = ccv_nnc_mps_graph_tensor_input_shape(&temp_view, gv->info.dim, gv->stride);
						[inputShapedTypes addObject:mps_temp_shape];
						// Determine reduction axes
						NSMutableArray<NSNumber*>* reduction_axes = [NSMutableArray new];
						for (int i = 0; i < nd; i++)
							if (gv->info.dim[i] > cv->info.dim[i])
								[reduction_axes addObject:@(i)];
						MPSGraphTensor* mps_c = mps_temp;
						if (reduction_axes.count > 0)
							mps_c = [graph reductionSumWithTensor:mps_temp axes:reduction_axes name:nil];
						[reduction_axes release];
						[resultTensors addObject:mps_c];
					});
					MPSGraphTensorData* data_temp = ccv_nnc_mps_graph_tensor_data_with_buffer(&temp_view, gv->info.dim, gv->stride, c_temp_buffer, 0);
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_temp], &cv, (int*[]){ cv->info.dim }, (int*[]){ cv->stride }, 1, 0);
				}
				if (d_needs_reduction)
				{
					const ccv_nnc_tensor_view_t* const gv = (const ccv_nnc_tensor_view_t*)g;
					ccv_nnc_tensor_view_t* const dv = (ccv_nnc_tensor_view_t*)d;
					// Create a fake tensor_view for the temp buffer with g's shape
					ccv_nnc_tensor_view_t temp_view = *gv;
					ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 3, hint, flags, inputs, input_size, outputs, output_size);
					int indices[1];
					const int nd = ccv_nnc_tensor_nd(gv->info.dim);
					MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
						MPSGraphTensor* mps_input;
						MPSGraphTensor* mps_temp = ccv_nnc_mps_graph_tensor_input(graph, &temp_view, gv->info.dim, gv->stride, &mps_input);
						[inputTensors addObject:mps_input];
						MPSGraphShapedType* mps_temp_shape = ccv_nnc_mps_graph_tensor_input_shape(&temp_view, gv->info.dim, gv->stride);
						[inputShapedTypes addObject:mps_temp_shape];
						// Determine reduction axes
						NSMutableArray<NSNumber*>* reduction_axes = [NSMutableArray new];
						for (int i = 0; i < nd; i++)
							if (gv->info.dim[i] > dv->info.dim[i])
								[reduction_axes addObject:@(i)];
						MPSGraphTensor* mps_d = mps_temp;
						if (reduction_axes.count > 0)
							mps_d = [graph reductionSumWithTensor:mps_temp axes:reduction_axes name:nil];
						[reduction_axes release];
						[resultTensors addObject:mps_d];
					});
					MPSGraphTensorData* data_temp = ccv_nnc_mps_graph_tensor_data_with_buffer(&temp_view, gv->info.dim, gv->stride, d_temp_buffer, 0);
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_temp], &dv, (int*[]){ dv->info.dim }, (int*[]){ dv->stride }, 1, 0);
				}
				ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
			}
			// Free temp buffers
			if (c_temp_buffer)
				mpobjfree(0, c_temp_buffer);
			if (d_temp_buffer)
				mpobjfree(0, d_temp_buffer);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			if (g)
			{
				// Compute gradient w.r.t. first input (a): da = g * conj(b)
				// Forward was: output = a * b, where a had shape that may differ from output due to broadcasting
				// In backward: g has shape of output, c (da) has shape of original a
				// If a was broadcast in forward, we need to reduce (sum) gradients along those axes
				if (b && c)
				{
					const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
					const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[2];
					ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
					ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
					int indices[2];
					int nd = ccv_nnc_tensor_nd(g->info.dim);
					assert(nd == ccv_nnc_tensor_nd(b->info.dim));
					assert(nd == ccv_nnc_tensor_nd(c->info.dim));
					MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
						MPSGraphTensor* mps_input_a;
						MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_a);
						[inputTensors addObject:mps_input_a];
						MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
						[inputShapedTypes addObject:mps_a_shape];
						MPSGraphTensor* mps_input_b;
						MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
						[inputTensors addObject:mps_input_b];
						MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
						[inputShapedTypes addObject:mps_b_shape];
						int i;
						// Step 1: Reshape complex tensors to separate real/imag components
						// Example: g has shape [M, N] where N is complex pairs (N = num_complex * 2)
						// Reshape to [M, N/2, 2] to split real and imaginary parts
						// mps_a (gradient g): [M, N] -> [M, N/2, 2]
						NSMutableArray<NSNumber*>* a_shape = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							[a_shape addObject:@(g->info.dim[i])];
						[a_shape addObject: @(g->info.dim[nd - 1] / 2)];
						[a_shape addObject: @2];
						mps_a = [graph reshapeTensor:mps_a withShape:a_shape name:nil];
						[a_shape release];
						// Step 2: Split into real and imaginary parts along the last axis
						// mps_a_splits[0] = real(g), shape [M, N/2, 1]
						// mps_a_splits[1] = imag(g), shape [M, N/2, 1]
						NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_a numSplits:2 axis:nd name:nil];
						// mps_b (input b): [M, N] -> [M, N/2, 2] (or [1, N] -> [1, N/2, 2] if broadcast)
						NSMutableArray<NSNumber*>* b_shape = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							[b_shape addObject:@(b->info.dim[i])];
						[b_shape addObject: @(b->info.dim[nd - 1] / 2)];
						[b_shape addObject: @2];
						mps_b = [graph reshapeTensor:mps_b withShape:b_shape name:nil];
						[b_shape release];
						// mps_b_splits[0] = real(b), mps_b_splits[1] = imag(b)
						NSArray<MPSGraphTensor*>* mps_b_splits = [graph splitTensor:mps_b numSplits:2 axis:nd name:nil];
						// Step 3: Compute complex conjugate multiplication: g * conj(b)
						// conj(b) = real(b) - i*imag(b)
						// g * conj(b) = (real(g) + i*imag(g)) * (real(b) - i*imag(b))
						//             = (real(g)*real(b) + imag(g)*imag(b)) + i*(imag(g)*real(b) - real(g)*imag(b))
						// mps_c_0 = real part = real(g)*real(b) + imag(g)*imag(b)
						// mps_c_1 = imag part = imag(g)*real(b) - real(g)*imag(b)
						// Result shape: [M, N/2, 1] (broadcasts if b has dim 1 where g has dim M)
						MPSGraphTensor* mps_c_0 = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
						MPSGraphTensor* mps_c_1 = [graph subtractionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
						// Step 4: Concatenate real and imag parts back together
						// [M, N/2, 1] concat [M, N/2, 1] along axis nd -> [M, N/2, 2]
						MPSGraphTensor* mps_c = [graph concatTensor:mps_c_0 withTensor:mps_c_1 dimension:nd name:nil];
						// Step 5: Handle broadcasting reduction
						// If original input 'a' was broadcast (e.g., a=[1,N], output=[M,N]),
						// gradients must be summed along broadcast axes to match 'a's shape.
						// Example: g=[M,N], c(da)=[1,N] -> reduce axis 0, result [1,N]
						// Current mps_c shape: [M, N/2, 2], target c shape: [1, N]
						NSMutableArray<NSNumber*>* reduction_axes = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							if (g->info.dim[i] > c->info.dim[i])
								[reduction_axes addObject:@(i)];
						// Check the last dimension (complex pairs)
						if (g->info.dim[nd - 1] > c->info.dim[nd - 1])
							[reduction_axes addObject:@(nd - 1)];
						// After reduction: [M, N/2, 2] with axis 0 reduced -> [1, N/2, 2]
						if (reduction_axes.count > 0)
							mps_c = [graph reductionSumWithTensor:mps_c axes:reduction_axes name:nil];
						[reduction_axes release];
						// Step 6: Reshape to final output shape
						// [1, N/2, 2] -> [1, N] to match c's expected shape
						NSMutableArray<NSNumber*>* c_shape = [NSMutableArray new];
						for (i = 0; i < nd; i++)
							[c_shape addObject:@(c->info.dim[i])];
						mps_c = [graph reshapeTensor:mps_c withShape:c_shape name:nil];
						[resultTensors addObject:mps_c];
						[c_shape release];
					});
					MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
					MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
					MPSGraphTensorData* data[] = {data_a, data_b};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
				}
				// Compute gradient w.r.t. second input (b): db = g * conj(a)
				// Forward was: output = a * b, where b had shape that may differ from output due to broadcasting
				// In backward: g has shape of output, d (db) has shape of original b
				// If b was broadcast in forward, we need to reduce (sum) gradients along those axes
				if (a && d)
				{
					const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
					const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[1];
					ccv_nnc_tensor_view_t* const d = (ccv_nnc_tensor_view_t*)outputs[1];
					ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
					int indices[2];
					int nd = ccv_nnc_tensor_nd(g->info.dim);
					assert(nd == ccv_nnc_tensor_nd(a->info.dim));
					assert(nd == ccv_nnc_tensor_nd(d->info.dim));
					MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
						MPSGraphTensor* mps_input_a;
						MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_a);
						[inputTensors addObject:mps_input_a];
						MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
						[inputShapedTypes addObject:mps_a_shape];
						MPSGraphTensor* mps_input_b;
						MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_b);
						[inputTensors addObject:mps_input_b];
						MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
						[inputShapedTypes addObject:mps_b_shape];
						int i;
						// Step 1: Reshape complex tensors to separate real/imag components
						// mps_a (gradient g): [M, N] -> [M, N/2, 2]
						NSMutableArray<NSNumber*>* a_shape = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							[a_shape addObject:@(g->info.dim[i])];
						[a_shape addObject: @(g->info.dim[nd - 1] / 2)];
						[a_shape addObject: @2];
						mps_a = [graph reshapeTensor:mps_a withShape:a_shape name:nil];
						[a_shape release];
						// Step 2: Split into real and imaginary parts
						NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_a numSplits:2 axis:nd name:nil];
						// mps_b (input a): reshape similarly
						NSMutableArray<NSNumber*>* b_shape = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							[b_shape addObject:@(a->info.dim[i])];
						[b_shape addObject: @(a->info.dim[nd - 1] / 2)];
						[b_shape addObject: @2];
						mps_b = [graph reshapeTensor:mps_b withShape:b_shape name:nil];
						[b_shape release];
						NSArray<MPSGraphTensor*>* mps_b_splits = [graph splitTensor:mps_b numSplits:2 axis:nd name:nil];
						// Step 3: Compute complex conjugate multiplication: g * conj(a)
						// Same formula as above but with input 'a' instead of 'b'
						MPSGraphTensor* mps_c_0 = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
						MPSGraphTensor* mps_c_1 = [graph subtractionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
						// Step 4: Concatenate real and imag parts
						MPSGraphTensor* mps_c = [graph concatTensor:mps_c_0 withTensor:mps_c_1 dimension:nd name:nil];
						// Step 5: Handle broadcasting reduction for gradient w.r.t. b
						// If b was broadcast, reduce along those axes
						NSMutableArray<NSNumber*>* reduction_axes = [NSMutableArray new];
						for (i = 0; i < nd - 1; i++)
							if (g->info.dim[i] > d->info.dim[i])
								[reduction_axes addObject:@(i)];
						if (g->info.dim[nd - 1] > d->info.dim[nd - 1])
							[reduction_axes addObject:@(nd - 1)];
						if (reduction_axes.count > 0)
							mps_c = [graph reductionSumWithTensor:mps_c axes:reduction_axes name:nil];
						[reduction_axes release];
						// Step 6: Reshape to final output shape
						NSMutableArray<NSNumber*>* c_shape = [NSMutableArray new];
						for (i = 0; i < nd; i++)
							[c_shape addObject:@(d->info.dim[i])];
						mps_c = [graph reshapeTensor:mps_c withShape:c_shape name:nil];
						[resultTensors addObject:mps_c];
						[c_shape release];
					});
					MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
					MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
					MPSGraphTensorData* data[] = {data_a, data_b};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &d, (int*[]){ d->info.dim }, (int*[]){ d->stride }, 1, 0);
				}
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_cmul_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_cmul_back;
}
