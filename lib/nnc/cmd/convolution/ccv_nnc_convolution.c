#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_conv_forw_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (input_size == 3 && (input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	// Ignore bias.
	if (input_size == 2 && (input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_conv_back_bitmask(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// Output the propagated error, gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// Ignore bias.
	if ((input_bitmasks[0] & 7u) == ((1u << 0) | (1u << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1) | (0 << 2)))
		return 1;
	// Don't propagate error, only gradient w.r.t. w and bias.
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmasks[0] == ((0 << 0) | (1u << 1) | (1u << 2)))
		return 1;
	// Ignore bias.
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1) | (0 << 2) | (0 << 3)) && output_bitmasks[0] == ((0 << 0) | (1u << 1) | (0 << 2)))
		return 1;
	// Ignore weight.
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (0 << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (0 << 1) | (1u << 2)))
		return 1;
	// Ignore bias and weight.
	if ((input_bitmasks[0] & 5u) == ((1u << 0) | (0 << 1) | (1u << 2) | (0 << 3)) && output_bitmasks[0] == ((1u << 0) | (0 << 1) | (0 << 2)))
		return 1;
	return 0;
}

static void _ccv_nnc_conv_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	// Get the channel output from the weight matrix.
	const int count = ccv_nnc_tensor_get_n(inputs[1]);
	assert(count == cmd.convolution.count);
	ccv_nnc_tensor_set_c(outputs, ccv_nnc_tensor_nd(inputs[0].dim), count);
	ccv_nnc_tensor_set_n(outputs, ccv_nnc_tensor_get_n(inputs[0]));
	ccv_nnc_cmd_param_t modified_cmd = cmd;
	int i = 0;
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		modified_cmd.size.dim[i] = (modified_cmd.size.dim[i] - 1) * ccv_max(cmd.convolution.dilation[i], 1) + 1;
	ccv_nnc_hint_tensor_forward(modified_cmd, inputs[0], hint, outputs);
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conv_cpu_ref.c, ccv_nnc_conv_cpu_opt.c, gpu/ccv_nnc_conv_gpu_cudnn.cu, mps/ccv_nnc_conv_mps.m)
{
	registry->bitmask = _ccv_nnc_conv_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_conv_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conv_cpu_ref.c, ccv_nnc_conv_cpu_opt.c, gpu/ccv_nnc_conv_gpu_cudnn.cu, mps/ccv_nnc_conv_mps.m)
{
	registry->bitmask = _ccv_nnc_conv_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONVOLUTION_FORWARD)
#define CMD_CONVOLUTION_FORWARD(_groups, _count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count,.groups=_groups}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONVOLUTION_BACKWARD)
#define CMD_CONVOLUTION_BACKWARD(_groups, _count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count,.groups=_groups}}), 0)

static void _ccv_nnc_conv_transpose_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	assert(output_size == 1);
	outputs[0].type = inputs[0].type;
	outputs[0].format = inputs[0].format;
	outputs[0].datatype = inputs[0].datatype;
	// Get the channel output from the weight matrix.
	const int count = ccv_nnc_tensor_get_c(inputs[1]);
	assert(count == cmd.convolution_transpose.count);
	ccv_nnc_tensor_set_c(outputs, ccv_nnc_tensor_nd(inputs[0].dim), count);
	ccv_nnc_tensor_set_n(outputs, ccv_nnc_tensor_get_n(inputs[0]));
	ccv_nnc_cmd_param_t modified_cmd = cmd;
	int i = 0;
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	ccv_nnc_hint_tensor_forward(modified_cmd, inputs[0], hint, outputs);
	assert(inputs[0].format == outputs[0].format);
	const int nd = ccv_nnc_tensor_nd(inputs[0].dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw = ccv_nnc_tensor_hw(inputs[0], nd);
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		const int stride = ccv_max(1, hint.stride.dim[i]);
		const int size_dim = (modified_cmd.size.dim[i] - 1) * ccv_max(cmd.convolution_transpose.dilation[i], 1) + 1;
		outputs[0].dim[i + hw] = (inputs[0].dim[i + hw] - 1) * stride + size_dim - hint.border.begin[i] - hint.border.end[i] + cmd.convolution_transpose.output_padding;
	}
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conv_transpose_cpu_ref.c, gpu/ccv_nnc_conv_transpose_gpu_cudnn.cu, mps/ccv_nnc_conv_transpose_mps.m)
{
	registry->bitmask = _ccv_nnc_conv_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_conv_transpose_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_conv_transpose_cpu_ref.c, gpu/ccv_nnc_conv_transpose_gpu_cudnn.cu, mps/ccv_nnc_conv_transpose_mps.m)
{
	registry->bitmask = _ccv_nnc_conv_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_inputs;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD)
#define CMD_CONVOLUTION_TRANSPOSE_FORWARD(_groups, _count, _output_padding, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution_transpose={.count=_count,.groups=_groups,.output_padding=_output_padding}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD)
#define CMD_CONVOLUTION_TRANSPOSE_BACKWARD(_groups, _count, _output_padding, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution_transpose={.count=_count,.groups=_groups,.output_padding=_output_padding}}), 0)
