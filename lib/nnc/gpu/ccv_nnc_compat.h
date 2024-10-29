/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_compat_h
#define GUARD_ccv_nnc_compat_h

#ifndef HAVE_CUDA
#error "This file requires to be compiled with CUDA support."
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "nnc/ccv_nnc.h"
#include "nnc/_ccv_nnc_stream.h"
#include <cuda_runtime.h>
#include <cufile.h>  // For GPUDirect Storage

// Simple counterparts of ccmalloc / ccfree.
void* cumalloc(int device, size_t size);
void cufree(int device, void* ptr);
void cudevice(int device);
void cumemcpy(void* dest, const int dest_type, const void* src, const int src_type, size_t n);
void* cuhostalloc(size_t size);
void cuhostfree(void* ptr);
int curegister(void* ptr, size_t size);
void cuunregister(void* ptr);
typedef void(*cump_f)(int device_id, void* const context);
int curegmp(int device_id, cump_f func, void* const context); // register memory pressure handler
void cuunregmp(const int id); // un-register memory pressure handler.
void cusetprofiler(int state);

void* cuDirectFileRead(int device, size_t size, const char* const filename, const off_t offset);
void* cuDirectFileReadAsync(int device, size_t size, const char* const filename, const off_t offset, cudaStream_t stream, CUfileHandle_t file_handle, CUfileDescr_t file_descr);
cudaStream_t cuSharedFileIOStream(void);
cudaError_t cuSharedStreamSync(void);

#define MAX_FILES 100

typedef struct {
    char filename[256];             
    CUfileHandle_t file_handle;     
    CUfileDescr_t file_descr;       
    int is_used;                    
} ccv_nnc_cuda_file_entry;

ccv_nnc_cuda_file_entry ccv_nnc_get_file_entry(const char* filename);
void ccv_nnc_cleanup_all_file_entries();
void ccv_nnc_remove_cuda_file_entry(const char* filename);

// Stream context
CCV_WARN_UNUSED(ccv_nnc_stream_context_t*) ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context);
int co_stream_compat_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);
void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(void*) ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem);
void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(ccv_nnc_stream_signal_t*) ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal);

typedef struct {
	int major;
	int minor;
	int multi_processor_count;
} ccv_nnc_cuda_device_prop_t;

// Misc.
CCV_WARN_UNUSED(int) ccv_nnc_gpu_device_count(void);
CCV_WARN_UNUSED(ccv_nnc_cuda_device_prop_t) ccv_nnc_gpu_device_props(void);
void ccv_nnc_compat_depalettize(const void* input, const int datatype, const size_t input_length, const int qbits, const int number_in_blocks, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context);
#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#ifndef __cplusplus
#error "A C++ compiler is required for this header."
#endif

// CUDA objects are C++ objects, this is a C++ header.
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#if CUDNN_VERSION < 5000 // Doesn't support CUDNN with version lower than 5000 (major version 5).
#undef HAVE_CUDNN
#endif
#endif
#ifdef HAVE_NCCL
#include <nccl.h>
#if NCCL_VERSION_CODE < 2200 // Doesn't support NCCL with version lower than 2.2
#undef HAVE_NCCL
#endif
#endif

#define CUDA_NUM_THREADS (1024)
#define CUDA_1D_KERNEL_LOOP(i, n) \
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define CUDA_GET_BLOCKS(n) ccv_min(((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, 4096)

extern "C" {
CCV_WARN_UNUSED(cudaDataType_t) ccv_nnc_cuda_datatype(const int datatype); // Get the datatype corresponding to cuda datatype.
CCV_WARN_UNUSED(cublasComputeType_t) ccv_nnc_cuda_compute_datatype(const int datatype); // Get the datatype that is accurate enough to be accumulator.
// Stream context methods to get the underlying objects, note that none of these methods are thread-safe.
CCV_WARN_UNUSED(int) ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudaStream_t) ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cublasHandle_t) ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_set_cublas_workspace(cublasHandle_t cublas, const ccv_nnc_stream_context_t* const stream_context, size_t workspace_size);
#if CUDA_VERSION >= 11000
#define CUBLAS_DEFAULT_WORKSPACE_SIZE_IN_BYTES (4 * 1024 * 1024)
#else
#define CUBLAS_DEFAULT_WORKSPACE_SIZE_IN_BYTES (0)
#endif
CCV_WARN_UNUSED(size_t) ccv_nnc_cublas_workspace_size_in_bytes(const ccv_nnc_tensor_t* const* const inputs, const int input_size, const ccv_nnc_tensor_t* const* const outputs, const int output_size);

#ifdef NDEBUG
#define CUDA_ENFORCE(status) status
#else
#define CUDA_ENFORCE(status) do {                               \
	const cudaError_t __status = status;                        \
	if (__status != cudaSuccess) {                              \
		printf("[%s:%d]:CUDA - Error: %d\n",                    \
				__FILE__, __LINE__, (int)__status);             \
		cudaDeviceReset();                                      \
		assert(0);                                              \
		exit(EXIT_FAILURE);                                     \
	}                                                           \
} while (0)
#endif

#ifdef NDEBUG
#define CUBLAS_ENFORCE(status) status
#else
#define CUBLAS_ENFORCE(status) do {                               \
	cublasStatus_t __status = status;                             \
	if (__status != CUBLAS_STATUS_SUCCESS) {                      \
		printf("[%s:%d]:CUBLAS - Error: %d\n",                    \
				__FILE__, __LINE__, (int)__status);               \
		cudaDeviceReset();                                        \
		assert(0);                                                \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
} while (0)
#endif

// Return floating point one on device memory.
CCV_WARN_UNUSED(void*) ccv_nnc_stream_context_get_ones(const ccv_nnc_stream_context_t* const stream_context, const int n, const int datatype);

#ifdef HAVE_CUDNN
CCV_WARN_UNUSED(cudnnDataType_t) ccv_nnc_cudnn_datatype(const int datatype);
CCV_WARN_UNUSED(cudnnHandle_t) ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* const stream_context);
// CUDNN related descriptors.
CCV_WARN_UNUSED(cudnnActivationDescriptor_t) ccv_nnc_stream_context_get_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnConvolutionDescriptor_t) ccv_nnc_stream_context_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, const int datatype);
CCV_WARN_UNUSED(cudnnDropoutDescriptor_t) ccv_nnc_stream_context_get_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, const float p);
CCV_WARN_UNUSED(cudnnFilterDescriptor_t) ccv_nnc_stream_context_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnOpTensorDescriptor_t) ccv_nnc_stream_context_get_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnPoolingDescriptor_t) ccv_nnc_stream_context_get_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnReduceTensorDescriptor_t) ccv_nnc_stream_context_get_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnRNNDescriptor_t) ccv_nnc_stream_context_get_rnn_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnTensorDescriptor_t) ccv_nnc_stream_context_get_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnRNNDataDescriptor_t) ccv_nnc_stream_context_get_rnn_data_descriptor(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_return_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnActivationDescriptor_t activation_desc);
void ccv_nnc_stream_context_return_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnConvolutionDescriptor_t convolution_desc);
void ccv_nnc_stream_context_return_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnDropoutDescriptor_t dropout_desc);
void ccv_nnc_stream_context_return_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnFilterDescriptor_t filter_desc);
void ccv_nnc_stream_context_return_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnOpTensorDescriptor_t op_tensor_desc);
void ccv_nnc_stream_context_return_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnPoolingDescriptor_t pooling_desc);
void ccv_nnc_stream_context_return_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnReduceTensorDescriptor_t reduce_tensor_desc);
void ccv_nnc_stream_context_return_rnn_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnRNNDescriptor_t rnn_desc);
void ccv_nnc_stream_context_return_rnn_data_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnRNNDataDescriptor_t rnn_data_desc);
void ccv_nnc_stream_context_return_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnTensorDescriptor_t tensor_desc);

#ifdef NDEBUG
#define CUDNN_ENFORCE(status) status
#else
#define CUDNN_ENFORCE(status) do {                                  \
	cudnnStatus_t __status = status;                                \
	if (__status != CUDNN_STATUS_SUCCESS) {                         \
		printf("[%s:%d]:CUDNN - Error: %s\n",                       \
				__FILE__, __LINE__, cudnnGetErrorString(__status)); \
		cudaDeviceReset();                                          \
		assert(0);                                                  \
		exit(EXIT_FAILURE);                                         \
	}                                                               \
} while (0)
#endif

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnTensorDescriptor_t descriptor;
	ccv_numeric_data_t data;
} ccv_nnc_cudnn_tensor_view_descriptor_t;
ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor);
ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* tensor);
void ccv_nnc_cudnn_deinit_tensor_view_descriptor(const ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc);

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnFilterDescriptor_t descriptor;
	ccv_numeric_data_t data;
} ccv_nnc_cudnn_filter_descriptor_t;
ccv_nnc_cudnn_filter_descriptor_t ccv_nnc_cudnn_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_t* tensor);
void ccv_nnc_cudnn_deinit_filter_descriptor(const ccv_nnc_cudnn_filter_descriptor_t filter_desc);

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnConvolutionDescriptor_t descriptor;
} ccv_nnc_cudnn_convolution_descriptor_t;
ccv_nnc_cudnn_convolution_descriptor_t ccv_nnc_cudnn_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_hint_t hint, const int datatype);
void ccv_nnc_cudnn_deinit_convolution_descriptor(const ccv_nnc_cudnn_convolution_descriptor_t convolution_desc);
#endif

#ifdef HAVE_NCCL
CCV_WARN_UNUSED(ncclComm_t) ccv_nnc_nccl_get_comm(ccv_nnc_stream_context_t* const stream, const int device_count, const int device_id);
CCV_WARN_UNUSED(ncclDataType_t) ccv_nnc_nccl_datatype(const int datatype); // Get the datatype corresponding to nccl datatype.

#ifdef NDEBUG
#define NCCL_ENFORCE(status) status
#else
#define NCCL_ENFORCE(status) do {                                 \
	if (status != ncclSuccess) {                                  \
		printf("[%s:%d]:NCCL - Error: %s\n",                      \
				__FILE__, __LINE__, ncclGetErrorString(status));  \
		cudaDeviceReset();                                        \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
} while (0)
#endif

#endif
}
#endif

#endif
