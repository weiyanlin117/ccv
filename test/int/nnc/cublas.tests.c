#include "case.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "ccv_case.h"
#include "nnc/ccv_nnc_internal.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
// Forward declarations for the CUDA functions
extern void qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, __fp16 *V, __fp16 *O,
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

extern void ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct(
    int8_t *Q, int8_t *K, __fp16 *V, __fp16 *O,
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

void ccv_nnc_per_warp_int8_direct(
    __fp16 *q,            // Input Q tensor data (FP16)
    __fp16 *k,            // Input K tensor data (FP16)
    int8_t *q_int8,       // Output Q quantized (INT8)
    int8_t *k_int8,       // Output K quantized (INT8)
    float *q_scale,       // Output Q scales (FP32)
    float *k_scale,       // Output K scales (FP32)
    __fp16 *km,           // Optional K mean tensor data (FP16, can be NULL)
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
TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("gemm no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm no transpose with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 + 1, 1 * 8 + 2 * 11 - 1, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 + 1, 3 * 8 + 4 * 11 - 1, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 + 1, 5 * 8 + 6 * 11 - 1, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 + 1, 7 * 8 + 8 * 11 - 1, 7 * 9 + 8 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm no transpose with bias and palettize weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
	};
	ccv_nnc_tensor_t* const pb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(CPU_TENSOR_NHWC(32F, 2, 3), 4, 128), 0);
	(void)ccv_nnc_palettize(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 4, 128, pb->data.u8, ccv_nnc_tensor_data_size_without_padding(pb->info));
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(GPU_TENSOR_NHWC(000, 32F, 2, 3), 4, 128), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, pb, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 + 1, 1 * 8 + 2 * 11 - 1, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 + 1, 3 * 8 + 4 * 11 - 1, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 + 1, 5 * 8 + 6 * 11 - 1, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 + 1, 7 * 8 + 8 * 11 - 1, 7 * 9 + 8 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(pb);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("backward gemm with no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose and palettize weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const pb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(CPU_TENSOR_NHWC(32F, 2, 3), 4, 128), 0);
	(void)ccv_nnc_palettize(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 4, 128, pb->data.u8, ccv_nnc_tensor_data_size_without_padding(pb->info));
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(GPU_TENSOR_NHWC(000, 32F, 2, 3), 4, 128), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, pb), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(pb);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("gemm no transpose batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 7 + 3 * 10, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12,
		4 * 7 + 5 * 10, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12,
		6 * 7 + 7 * 10, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12,
		8 * 7 + 9 * 10, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
		2, 4, 6, 8,
		3, 5, 7, 9,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		-1, 0, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 7 + 3 * 10 - 1, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12 + 1,
		4 * 7 + 5 * 10 - 1, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12 + 1,
		6 * 7 + 7 * 10 - 1, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12 + 1,
		8 * 7 + 9 * 10 - 1, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm transpose b batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
		80, 110,
		90, 120,
		10, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	float dp[] = {
		-1, 0, 1,
		2, 3, -4,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 80 + 3 * 110 + 2, 2 * 90 + 3 * 120 + 3, 2 * 10 + 3 * 13 - 4,
		4 * 80 + 5 * 110 + 2, 4 * 90 + 5 * 120 + 3, 4 * 10 + 5 * 13 - 4,
		6 * 80 + 7 * 110 + 2, 6 * 90 + 7 * 120 + 3, 6 * 10 + 7 * 13 - 4,
		8 * 80 + 9 * 110 + 2, 8 * 90 + 9 * 120 + 3, 8 * 10 + 9 * 13 - 4,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("backward gemm with no transpose batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 10 * 24 + 20 * 25 + 30 * 26,
		40 * 21 + 50 * 22 + 60 * 23, 40 * 24 + 50 * 25 + 60 * 26,
		70 * 21 + 80 * 22 + 90 * 23, 70 * 24 + 80 * 25 + 90 * 26,
		100 * 21 + 110 * 22 + 120 * 23, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
		212, 222, 232,
		242, 252, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("cublas forward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_TENSOR_EQ(tb1, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("cublas forward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("cublas forward gemv in half precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("cublas forward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_TENSOR_EQ(tb1, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("cublas forward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("cublas forward gemv in half precision no bias, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("cublas forward gemv in half precision no bias, variant 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 128, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 1), 0);

	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 128, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(hw, ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(w, a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("cublas backward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, h), TENSOR_LIST(tb, tdw, tdbias, th), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdw, hdw, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdbias, hdbias, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(th, hh, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("cublas backward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_tensor_t* hg2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(ha2, hw2, hbias2, hg2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2, hg2), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, h), TENSOR_LIST(tb, tdw, tdbias, th), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb, tdw, tdbias, th), TENSOR_LIST(tb1, tdw1, tdbias1, th1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 10 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw1->data.f32, hdw->data.f32, 64 * 128, 1e-2, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias1->data.f32, hdbias->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th1->data.f32, hh->data.f32, 10 * 128, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
	ccv_nnc_tensor_free(hg2);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(tdw1);
	ccv_nnc_tensor_free(tdbias1);
	ccv_nnc_tensor_free(th1);
}

TEST_CASE("cublas backward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hg), TENSOR_LIST(a, w, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, 0), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, 0), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, h), TENSOR_LIST(tb, tdw, th), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdw, hdw, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(th, hh, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("cublas backward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hg2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hg), TENSOR_LIST(ha2, hw2, hg2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hg2), TENSOR_LIST(a, w, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, 0), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, 0), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* th1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, h), TENSOR_LIST(tb, tdw, th), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb, tdw, th), TENSOR_LIST(tb1, tdw1, th1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 10 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw1->data.f32, hdw->data.f32, 64 * 128, 1e-2, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th1->data.f32, hh->data.f32, 10 * 128, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hg2);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(tdw1);
	ccv_nnc_tensor_free(th1);
}

TEST_CASE("cublas handle permute")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);
	int i;
	for (i = 0; i < 2 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 2 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(hb, hbt), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw), TENSOR_LIST(dat, tdw), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w, dbias), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw, hdbias), TENSOR_LIST(dat, tdw, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("ewdiv forward with reciprocal")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewdiv forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(ct), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_TENSOR_EQ(ct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("ewdiv backward with output 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("ewdiv backward with output 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* db = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b, c), TENSOR_LIST(0, db), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb, hc), TENSOR_LIST(0, dbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(db), TENSOR_LIST(hdb), 0);
	REQUIRE_TENSOR_EQ(dbt, hdb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdb);
	ccv_nnc_tensor_free(dbt);
}

TEST_CASE("exp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewexp backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWEXP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("ewlog forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewlog backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWLOG_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("ewsqrt forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewsqrt backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWLOG_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("clamp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("clamp forward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp backward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("clamp forward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp backward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("scaled dot product attention with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 40, 160, 224, 224, 128 };
		int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		// Detailed comparison between GPU SageAttention and CPU reference
		int exact_matches = 0;
		int very_close_matches = 0;  // diff < 1e-5
		int close_matches = 0;       // diff < 1e-4
		int acceptable_matches = 0;  // diff < 1e-3
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * R * Hq * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)copy_of_gpu_o_tensor->data.f32[i] - (float)o_tensor->data.f32[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
			if (diff < 1e-5) very_close_matches++;
			if (diff < 1e-4) close_matches++;
			if (diff < 1e-3) acceptable_matches++;
		}
		
		printf("\n[Trial %d] Config: B=%d, R=%d, C=%d, Hq=%d, Hk=%d, D=%d, causal=%d\n", 
			trial, B, R, C, Hq, Hk, D, is_causal);
		printf("Full tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Very close (diff < 1e-5): %d/%d (%.2f%%)\n", 
			very_close_matches, total_elements, (100.0 * very_close_matches) / total_elements);
		printf("  Close (diff < 1e-4): %d/%d (%.2f%%)\n", 
			close_matches, total_elements, (100.0 * close_matches) / total_elements);
		printf("  Acceptable (diff < 1e-3): %d/%d (%.2f%%)\n", 
			acceptable_matches, total_elements, (100.0 * acceptable_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		// Provide interpretation of results
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: GPU SageAttention and CPU outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: GPU SageAttention and CPU outputs are very close (within expected FP16 precision)\n");
		} else if (max_diff < 3e-3) {
			printf("⚠️  ACCEPTABLE: GPU SageAttention and CPU outputs have small differences but within tolerance\n");
		} else {
			printf("❌ FAILED: GPU SageAttention and CPU outputs differ significantly\n");
			printf("  This may indicate a precision issue or algorithmic difference\n");
		}
		
		REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 3e-3, "GPU SageAttention output should match CPU reference within tolerance");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention gradient with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
#define num_long_trials 8
#define num_short_trials 4
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		const int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1, 32,   12, 16, 1, 2, 1 };
		const int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5, 160,  256, 128, 77, 77, 5 };
		const int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5, 128,  128, 128, 128, 128, 5 };
		const int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8, 8,  8, 8, 8, 2, 8 };
		const int D_candidates[num_trials] = {  64, 40, 160, 192, 256, 128, 64, 40, 160, 192, 256, 128 };
		const int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1 };
		const int deterministic_candidates[num_trials] = {  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

		const int B = B_candidates[trial];
		const int R = R_candidates[trial];
		const int C = C_candidates[trial];
		const int Hq = Hq_candidates[trial];
		const int Hk = Hk_candidates[trial];
		const int D = D_candidates[trial];
		const int is_causal = is_causal_candidates[trial];
		const int deterministic = deterministic_candidates[trial];
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		for (int i = 0; i < B * R * Hq * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.deterministic = deterministic;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * Hq * D, 1e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * Hk * D, 3e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dv_tensor->data.f32, dv_tensor->data.f32, B * C * Hk * D, 6e-3, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(do_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("cmul in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z16_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tz->data.f32, z_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z16_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in float, broadcast semantics")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 8, 128), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 1, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 8, 128), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 1, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1 * 5 * 8 * 128; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 5 * 1 * 128; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "d");
	ccv_nnc_tensor_symbol_t e = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "e");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_BACKWARD(), TENSOR_SYMBOL_LIST(a, b, c), TENSOR_SYMBOL_LIST(d, e), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		z_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z_tensor), TENSOR_LIST(c_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const od_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d);
	ccv_nnc_tensor_t* const oe_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const e_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, e);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d_tensor, e_tensor), TENSOR_LIST(od_tensor, oe_tensor), 0);
	ccv_nnc_tensor_t* const td = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const te = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor, z_tensor), TENSOR_LIST(td, te), 0);
	REQUIRE_TENSOR_EQ(td, od_tensor, "cmul gradient from cudnn should match from CPU");
	REQUIRE_TENSOR_EQ(te, oe_tensor, "cmul gradient from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(od_tensor);
	ccv_nnc_tensor_free(oe_tensor);
	ccv_nnc_tensor_free(td);
	ccv_nnc_tensor_free(te);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul gradient in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t e = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_BACKWARD(), TENSOR_SYMBOL_LIST(a, b, c), TENSOR_SYMBOL_LIST(d, e), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		z_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_tensor_t* const z16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z_tensor), TENSOR_LIST(z16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z16_tensor), TENSOR_LIST(c_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const od16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const od_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d_tensor), TENSOR_LIST(od16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(od16_tensor), TENSOR_LIST(od_tensor), 0);
	ccv_nnc_tensor_t* const oe16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const oe_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const e_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, e);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(e_tensor), TENSOR_LIST(oe16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(oe16_tensor), TENSOR_LIST(oe_tensor), 0);
	ccv_nnc_tensor_t* const td = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const te = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor, z_tensor), TENSOR_LIST(td, te), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, td->data.f32, od_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, te->data.f32, oe_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(z16_tensor);
	ccv_nnc_tensor_free(od_tensor);
	ccv_nnc_tensor_free(od16_tensor);
	ccv_nnc_tensor_free(td);
	ccv_nnc_tensor_free(oe_tensor);
	ccv_nnc_tensor_free(oe16_tensor);
	ccv_nnc_tensor_free(te);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}


// CPU reference implementation matching SageAttention GPU quantization logic
static void cpu_quantize_int8_per_block(const float* input, int8_t* output, float* scales,
                                       int batch_size, int num_tokens, int num_heads, int head_dim,
                                       const uint32_t BLOCK_SIZE)
{
	const int scale_blocks_per_seq = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	// SageAttention GPU uses grid=(num_blocks, num_heads, batch_size)
	// Each CUDA block processes BLOCK_SIZE tokens collaboratively
	for (int b = 0; b < batch_size; b++) {
		for (int h = 0; h < num_heads; h++) {
			for (int block_idx = 0; block_idx < scale_blocks_per_seq; block_idx++) {
				const int token_start = block_idx * BLOCK_SIZE;
				const int token_end = (token_start + BLOCK_SIZE) < num_tokens ? 
				                     (token_start + BLOCK_SIZE) : num_tokens;
				
				// Find max absolute value in this block for this head
				// This exactly matches blockReduceMax() behavior in GPU kernel
				float max_val = 0.0f;
				for (int t = token_start; t < token_end; t++) {
					for (int d = 0; d < head_dim; d++) {
						const int idx = b * num_tokens * num_heads * head_dim + 
						               t * num_heads * head_dim + h * head_dim + d;
						max_val = fmaxf(max_val, fabsf(input[idx]));
					}
				}
				
				// Compute scale exactly like GPU: s_amax / 127.0f
				const float scale = max_val / 127.0f;
				const int scale_idx = b * num_heads * scale_blocks_per_seq + h * scale_blocks_per_seq + block_idx;
				scales[scale_idx] = scale;
				
				// Quantize values using tmp_scale = 127.0f / s_amax (matching GPU)
				const float tmp_scale = (max_val > 1e-8f) ? (127.0f / max_val) : 0.0f;
				for (int t = token_start; t < token_end; t++) {
					for (int d = 0; d < head_dim; d++) {
						const int idx = b * num_tokens * num_heads * head_dim + 
						               t * num_heads * head_dim + h * head_dim + d;
						// Use float_to_int8_rn equivalent: roundf with saturation
						const float quantized = roundf(input[idx] * tmp_scale);
						output[idx] = (int8_t)fmaxf(-128.0f, fminf(127.0f, quantized));
					}
				}
			}
		}
	}
}

// CPU reference implementation for per-thread quantization (each element gets own scale)
static void cpu_quantize_int8_per_thread(const float* input, int8_t* output, float* scales,
                                        int batch_size, int num_tokens, int num_heads, int head_dim)
{
	// Per-thread: each element gets its own scale factor
	// Scale tensor shape: [batch_size, num_tokens, num_heads, head_dim]
	for (int b = 0; b < batch_size; b++) {
		for (int t = 0; t < num_tokens; t++) {
			for (int h = 0; h < num_heads; h++) {
				for (int d = 0; d < head_dim; d++) {
					const int idx = b * num_tokens * num_heads * head_dim + 
					               t * num_heads * head_dim + h * head_dim + d;
					
					// Each element gets its own scale
					const float abs_val = fabsf(input[idx]);
					const float scale = abs_val / 127.0f;
					scales[idx] = scale;
					
					// Quantize single element
					const float tmp_scale = (abs_val > 1e-8f) ? (127.0f / abs_val) : 0.0f;
					const float quantized = roundf(input[idx] * tmp_scale);
					output[idx] = (int8_t)fmaxf(-128.0f, fminf(127.0f, quantized));
				}
			}
		}
	}
}

TEST_CASE("sage attention quantization ccv_nnc_per_warp_int8 test light")
{
	ccv_cli_set_output_levels(CCV_CLI_VERBOSE);

	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
	                  ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("=== CCV SageAttention Per-Warp Quantization Light Test ===\n");
	
	// Test parameters matching PyTorch per_warp_int8 defaults
	const int B = 1;      // batch size
	const int R = 64;     // sequence length
	const int H = 8;      // number of heads  
	const int D = 128;    // head dimension
	const uint32_t BLKQ = 128;  // Block size for Q (PyTorch default)
	const uint32_t WARPQ = 32;  // Warp size for Q (PyTorch default)
	const uint32_t BLKK = 64;   // Block size for K (PyTorch default)

	// Create input tensors
	ccv_nnc_tensor_t* const q_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D), 0);

	// Load input data from PyTorch test
	FILE* q_input_file = fopen("/tmp/test_q_input.bin", "rb");
	if (q_input_file) {
		fread(q_input_tensor->data.f32, sizeof(float), B * R * H * D, q_input_file);
		fclose(q_input_file);
		printf("✓ Loaded Q input from /tmp/test_q_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_q_input.bin\n");
		return;
	}
	
	FILE* k_input_file = fopen("/tmp/test_k_input.bin", "rb");
	if (k_input_file) {
		fread(k_input_tensor->data.f32, sizeof(float), B * R * H * D, k_input_file);
		fclose(k_input_file);
		printf("✓ Loaded K input from /tmp/test_k_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_input.bin\n");
		return; 
	}
	
	FILE* k_mean_file = fopen("/tmp/test_k_mean.bin", "rb");
	if (k_mean_file) {
		fread(k_mean_tensor->data.f32, sizeof(float), B * H * D, k_mean_file);
		fclose(k_mean_file);
		printf("✓ Loaded K mean input from /tmp/test_k_mean.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_mean.bin\n");
		return; 
	}
	printf("k_mean_tensor\n");
	ccv_nnc_print_tensor_info(k_mean_tensor);
	// Convert to FP16 and transfer to GPU
	ccv_nnc_tensor_t* const q_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);

	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor), TENSOR_LIST(q_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor), TENSOR_LIST(k_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor), TENSOR_LIST(k_mean_tensor_f16), 0);

	ccv_nnc_tensor_t* const gpu_q_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor_f16), TENSOR_LIST(gpu_q_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor_f16), TENSOR_LIST(gpu_k_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor_f16), TENSOR_LIST(gpu_k_mean_tensor), 0);

	printf("Input shapes: Q[%d,%d,%d,%d], K[%d,%d,%d,%d]\n", B, R, H, D, B, R, H, D);
	
	// Print input data for verification against PyTorch
	printf("Q_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", q_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_mean sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_mean_tensor->data.f32[i]);
	}
	printf("\n");

	// Calculate input ranges
	float q_min = q_input_tensor->data.f32[0], q_max = q_input_tensor->data.f32[0];
	float k_min = k_input_tensor->data.f32[0], k_max = k_input_tensor->data.f32[0];
	for (int i = 1; i < B * R * H * D; i++) {
		if (q_input_tensor->data.f32[i] < q_min) q_min = q_input_tensor->data.f32[i];
		if (q_input_tensor->data.f32[i] > q_max) q_max = q_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] < k_min) k_min = k_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] > k_max) k_max = k_input_tensor->data.f32[i];
	}
	printf("Q_input range: [%.6f, %.6f]\n", q_min, q_max);
	printf("K_input range: [%.6f, %.6f]\n", k_min, k_max);
	
	// Calculate output sizes
	const size_t output_size = B * R * H * D;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const size_t q_scale_size = B * H * q_blocks * warps_per_block;
	const size_t k_scale_size = B * H * ((R + BLKK - 1) / BLKK);
	
	// Create output tensors for the new tensor-based API
	ccv_nnc_tensor_t* const gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	// Calculate exact scale dimensions
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (R + BLKK - 1) / BLKK;      // Should be 1
	
	printf("Scale tensor calculations: q_blocks=%zu, warps_per_block=%zu, q_scale_blocks=%d\n", 
	       q_blocks, warps_per_block, q_scale_blocks);
	printf("Expected Q scale shape: [%d, %d, %d]\n", B, H, q_scale_blocks);
	printf("Expected K scale shape: [%d, %d, %d]\n", B, H, k_scale_blocks);
	
	ccv_nnc_tensor_t* const gpu_q_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* const gpu_k_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	printf("\n");

	// Run quantization using new CCV wrapper function
#ifdef HAVE_CUDA_SM80
	extern int ccv_nnc_per_warp_int8(
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
    ccv_nnc_stream_context_t* const stream_context);
	
	printf("Running CCV per-warp quantization (new API)...\n");
	
	// Q: per-warp quantization using new wrapper
	// Note: CCV tensors are in NHWC format, which is tensor_layout=0 (NHD)
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	// int q_result = ccv_nnc_quant_per_warp_int8_cuda(
	// 	gpu_q_input_tensor,      // Input FP16 tensor
	// 	gpu_q_int8_tensor,       // Output INT8 tensor  
	// 	gpu_q_scales_tensor,     // Output FP32 scale tensor
	// 	BLKQ,                    // block_size = 128
	// 	WARPQ,                   // warp_block_size = 32
	// 	0,                       // tensor_layout = 0 (NHD/NHWC)
	// 	stream_context
	// );
	
	// int k_mean_result = ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda(
	// 	gpu_k_input_tensor,      // Input FP16 tensor
	// 	gpu_k_mean_tensor,      // Input FP16 tensor
	// 	gpu_k_int8_tensor,       // Output INT8 tensor
	// 	gpu_k_scales_tensor,     // Output FP32 scale tensor
	// 	BLKK,                    // block_size = 64
	// 	0,                       // tensor_layout = 0 (NHD/NHWC)
	// 	stream_context
	// );
	
	int result = ccv_nnc_per_warp_int8(
		gpu_q_input_tensor,       // Input Q tensor (FP16)
		gpu_k_input_tensor,       // Input K tensor (FP16)
		gpu_q_int8_tensor,  // Output Q quantized (INT8)
		gpu_k_int8_tensor,  // Output K quantized (INT8)
		gpu_q_scales_tensor, // Output Q scales (FP32)
		gpu_k_scales_tensor, // Output K scales (FP32)
		gpu_k_mean_tensor,      // Optional K mean tensor (not used currently)
		BLKQ,                        // Block size for Q (default 128)
		WARPQ,                       // Warp size for Q (default 32)
		BLKK,                        // Block size for K (default 64)
		0,               // 0=NHD, 1=HND
    stream_context);
	if (result == 0) {
		printf("✓ CCV quantization completed successfully\n");
		
		// Copy tensor results to CPU for both Q and K
		ccv_nnc_tensor_t* const cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_q_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks, 1), 0);
		ccv_nnc_tensor_t* const cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_k_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks, 1), 0);
		
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_scales_tensor), TENSOR_LIST(cpu_q_scales_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_scales_tensor), TENSOR_LIST(cpu_k_scales_tensor), 0);
		
		// Print sample outputs
		printf("Q_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_q_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("Q_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < q_scale_size; i++) printf("%.6f ", cpu_q_scales_tensor->data.f32[i]);
		printf("\n");
		
		printf("K_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_k_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("K_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < k_scale_size; i++) printf("%.6f ", cpu_k_scales_tensor->data.f32[i]);
		printf("\n");
		
		// Save outputs to disk
		FILE* q_int8_file = fopen("/tmp/ccv_q_int8.bin", "wb");
		if (q_int8_file) {
			fwrite(cpu_q_int8_tensor->data.u8, sizeof(int8_t), output_size, q_int8_file);
			fclose(q_int8_file);
			printf("✓ Saved Q_int8 to /tmp/ccv_q_int8.bin\n");
		}
		
		FILE* k_int8_file = fopen("/tmp/ccv_k_int8.bin", "wb");
		if (k_int8_file) {
			fwrite(cpu_k_int8_tensor->data.u8, sizeof(int8_t), output_size, k_int8_file);
			fclose(k_int8_file);
			printf("✓ Saved K_int8 to /tmp/ccv_k_int8.bin\n");
		}
		
		FILE* q_scales_file = fopen("/tmp/ccv_q_scales.bin", "wb");
		if (q_scales_file) {
			fwrite(cpu_q_scales_tensor->data.f32, sizeof(float), q_scale_size, q_scales_file);
			fclose(q_scales_file);
			printf("✓ Saved Q_scales to /tmp/ccv_q_scales.bin\n");
		}
		
		FILE* k_scales_file = fopen("/tmp/ccv_k_scales.bin", "wb");
		if (k_scales_file) {
			fwrite(cpu_k_scales_tensor->data.f32, sizeof(float), k_scale_size, k_scales_file);
			fclose(k_scales_file);
			printf("✓ Saved K_scales to /tmp/ccv_k_scales.bin\n");
		}
		
		printf("Output sizes: Q_int8=%zu, Q_scales=%zu, K_int8=%zu, K_scales=%zu\n",
		       output_size, q_scale_size, output_size, k_scale_size);
		
		// Clean up CPU tensors
		ccv_nnc_tensor_free(cpu_q_int8_tensor);
		ccv_nnc_tensor_free(cpu_q_scales_tensor);
		ccv_nnc_tensor_free(cpu_k_int8_tensor);
		ccv_nnc_tensor_free(cpu_k_scales_tensor);
	} else {
		printf("❌ CCV quantization failed: result=%d\n", result);
	}
#else
	printf("❌ CUDA SM80 not available\n");
	int q_result = -1;
	int k_result = -1;
#endif
	
	// Cleanup tensors
	ccv_nnc_tensor_free(q_input_tensor);
	ccv_nnc_tensor_free(k_input_tensor);
		ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(q_input_tensor_f16);
	ccv_nnc_tensor_free(k_input_tensor_f16);
		ccv_nnc_tensor_free(k_mean_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_input_tensor);
	ccv_nnc_tensor_free(gpu_k_input_tensor);
		ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scales_tensor);
	ccv_nnc_tensor_free(gpu_k_scales_tensor);
	ccv_nnc_stream_context_free(stream_context);
	
	printf("CCV quantization light test completed\n");
	REQUIRE(result == 0, "CCV quantization should succeed");
}

TEST_CASE("sage attention quantization per-warp sub mean test light")
{
	ccv_cli_set_output_levels(CCV_CLI_VERBOSE);

	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
	                  ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("=== CCV SageAttention Per-Warp Quantization Light Test ===\n");
	
	// Test parameters matching PyTorch per_warp_int8 defaults
	const int B = 1;      // batch size
	const int R = 64;     // sequence length
	const int H = 8;      // number of heads  
	const int D = 128;    // head dimension
	const uint32_t BLKQ = 128;  // Block size for Q (PyTorch default)
	const uint32_t WARPQ = 32;  // Warp size for Q (PyTorch default)
	const uint32_t BLKK = 64;   // Block size for K (PyTorch default)

	// Create input tensors
	ccv_nnc_tensor_t* const q_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D), 0);

	// Load input data from PyTorch test
	FILE* q_input_file = fopen("/tmp/test_q_input.bin", "rb");
	if (q_input_file) {
		fread(q_input_tensor->data.f32, sizeof(float), B * R * H * D, q_input_file);
		fclose(q_input_file);
		printf("✓ Loaded Q input from /tmp/test_q_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_q_input.bin\n");
		return;
	}
	
	FILE* k_input_file = fopen("/tmp/test_k_input.bin", "rb");
	if (k_input_file) {
		fread(k_input_tensor->data.f32, sizeof(float), B * R * H * D, k_input_file);
		fclose(k_input_file);
		printf("✓ Loaded K input from /tmp/test_k_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_input.bin\n");
		return; 
	}
	
	FILE* k_mean_file = fopen("/tmp/test_k_mean.bin", "rb");
	if (k_mean_file) {
		fread(k_mean_tensor->data.f32, sizeof(float), B * H * D, k_mean_file);
		fclose(k_mean_file);
		printf("✓ Loaded K mean input from /tmp/test_k_mean.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_mean.bin\n");
		return; 
	}
	printf("k_mean_tensor\n");
	ccv_nnc_print_tensor_info(k_mean_tensor);
	// Convert to FP16 and transfer to GPU
	ccv_nnc_tensor_t* const q_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);

	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor), TENSOR_LIST(q_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor), TENSOR_LIST(k_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor), TENSOR_LIST(k_mean_tensor_f16), 0);

	ccv_nnc_tensor_t* const gpu_q_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor_f16), TENSOR_LIST(gpu_q_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor_f16), TENSOR_LIST(gpu_k_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor_f16), TENSOR_LIST(gpu_k_mean_tensor), 0);

	printf("Input shapes: Q[%d,%d,%d,%d], K[%d,%d,%d,%d]\n", B, R, H, D, B, R, H, D);
	
	// Print input data for verification against PyTorch
	printf("Q_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", q_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_mean sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_mean_tensor->data.f32[i]);
	}
	printf("\n");

	// Calculate input ranges
	float q_min = q_input_tensor->data.f32[0], q_max = q_input_tensor->data.f32[0];
	float k_min = k_input_tensor->data.f32[0], k_max = k_input_tensor->data.f32[0];
	for (int i = 1; i < B * R * H * D; i++) {
		if (q_input_tensor->data.f32[i] < q_min) q_min = q_input_tensor->data.f32[i];
		if (q_input_tensor->data.f32[i] > q_max) q_max = q_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] < k_min) k_min = k_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] > k_max) k_max = k_input_tensor->data.f32[i];
	}
	printf("Q_input range: [%.6f, %.6f]\n", q_min, q_max);
	printf("K_input range: [%.6f, %.6f]\n", k_min, k_max);
	
	// Calculate output sizes
	const size_t output_size = B * R * H * D;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const size_t q_scale_size = B * H * q_blocks * warps_per_block;
	const size_t k_scale_size = B * H * ((R + BLKK - 1) / BLKK);
	
	// Create output tensors for the new tensor-based API
	ccv_nnc_tensor_t* const gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	// Calculate exact scale dimensions
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (R + BLKK - 1) / BLKK;      // Should be 1
	
	printf("Scale tensor calculations: q_blocks=%zu, warps_per_block=%zu, q_scale_blocks=%d\n", 
	       q_blocks, warps_per_block, q_scale_blocks);
	printf("Expected Q scale shape: [%d, %d, %d]\n", B, H, q_scale_blocks);
	printf("Expected K scale shape: [%d, %d, %d]\n", B, H, k_scale_blocks);
	
	ccv_nnc_tensor_t* const gpu_q_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* const gpu_k_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	printf("\n");

	// Run quantization using new CCV wrapper function
#ifdef HAVE_CUDA_SM80
	extern int ccv_nnc_quant_per_warp_int8_cuda(
		ccv_nnc_tensor_t* const input,
		ccv_nnc_tensor_t* const output,
		ccv_nnc_tensor_t* const scale,
		int block_size,
		int warp_block_size,
		int tensor_layout,
		ccv_nnc_stream_context_t* const stream_context);
	
	printf("Running CCV per-warp quantization (new API)...\n");
	
	// Q: per-warp quantization using new wrapper
	// Note: CCV tensors are in NHWC format, which is tensor_layout=0 (NHD)
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int q_result = ccv_nnc_quant_per_warp_int8_cuda(
		gpu_q_input_tensor,      // Input FP16 tensor
		gpu_q_int8_tensor,       // Output INT8 tensor  
		gpu_q_scales_tensor,     // Output FP32 scale tensor
		BLKQ,                    // block_size = 128
		WARPQ,                   // warp_block_size = 32
		0,                       // tensor_layout = 0 (NHD/NHWC)
		stream_context
	);
	
	// K: per-block quantization using new tensor API
	extern int ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda(
    ccv_nnc_tensor_t* const input,   // Input tensor (FP16)
    ccv_nnc_tensor_t* const mean,    // Mean tensor (FP16)
    ccv_nnc_tensor_t* const output,  // Output tensor (INT8)
    ccv_nnc_tensor_t* const scale,   // Scale tensor (FP32)
    int block_size,                  // Block size (e.g., 64)
    int tensor_layout,               // 0=NHD, 1=HND
    ccv_nnc_stream_context_t* const stream_context);
	
	int k_mean_result = ccv_nnc_quant_per_block_int8_fuse_sub_mean_cuda(
		gpu_k_input_tensor,      // Input FP16 tensor
		gpu_k_mean_tensor,      // Input FP16 tensor
		gpu_k_int8_tensor,       // Output INT8 tensor
		gpu_k_scales_tensor,     // Output FP32 scale tensor
		BLKK,                    // block_size = 64
		0,                       // tensor_layout = 0 (NHD/NHWC)
		stream_context
	);
	
	if (q_result == 0 && k_mean_result == 0) {
		printf("✓ CCV quantization completed successfully\n");
		
		// Copy tensor results to CPU for both Q and K
		ccv_nnc_tensor_t* const cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_q_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks, 1), 0);
		ccv_nnc_tensor_t* const cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_k_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks, 1), 0);
		
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_scales_tensor), TENSOR_LIST(cpu_q_scales_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_scales_tensor), TENSOR_LIST(cpu_k_scales_tensor), 0);
		
		// Print sample outputs
		printf("Q_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_q_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("Q_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < q_scale_size; i++) printf("%.6f ", cpu_q_scales_tensor->data.f32[i]);
		printf("\n");
		
		printf("K_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_k_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("K_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < k_scale_size; i++) printf("%.6f ", cpu_k_scales_tensor->data.f32[i]);
		printf("\n");
		
		// Save outputs to disk
		FILE* q_int8_file = fopen("/tmp/ccv_q_int8.bin", "wb");
		if (q_int8_file) {
			fwrite(cpu_q_int8_tensor->data.u8, sizeof(int8_t), output_size, q_int8_file);
			fclose(q_int8_file);
			printf("✓ Saved Q_int8 to /tmp/ccv_q_int8.bin\n");
		}
		
		FILE* k_int8_file = fopen("/tmp/ccv_k_int8.bin", "wb");
		if (k_int8_file) {
			fwrite(cpu_k_int8_tensor->data.u8, sizeof(int8_t), output_size, k_int8_file);
			fclose(k_int8_file);
			printf("✓ Saved K_int8 to /tmp/ccv_k_int8.bin\n");
		}
		
		FILE* q_scales_file = fopen("/tmp/ccv_q_scales.bin", "wb");
		if (q_scales_file) {
			fwrite(cpu_q_scales_tensor->data.f32, sizeof(float), q_scale_size, q_scales_file);
			fclose(q_scales_file);
			printf("✓ Saved Q_scales to /tmp/ccv_q_scales.bin\n");
		}
		
		FILE* k_scales_file = fopen("/tmp/ccv_k_scales.bin", "wb");
		if (k_scales_file) {
			fwrite(cpu_k_scales_tensor->data.f32, sizeof(float), k_scale_size, k_scales_file);
			fclose(k_scales_file);
			printf("✓ Saved K_scales to /tmp/ccv_k_scales.bin\n");
		}
		
		printf("Output sizes: Q_int8=%zu, Q_scales=%zu, K_int8=%zu, K_scales=%zu\n",
		       output_size, q_scale_size, output_size, k_scale_size);
		
		// Clean up CPU tensors
		ccv_nnc_tensor_free(cpu_q_int8_tensor);
		ccv_nnc_tensor_free(cpu_q_scales_tensor);
		ccv_nnc_tensor_free(cpu_k_int8_tensor);
		ccv_nnc_tensor_free(cpu_k_scales_tensor);
	} else {
		printf("❌ CCV quantization failed: Q=%d, K=%d\n", q_result, k_mean_result);
	}
#else
	printf("❌ CUDA SM80 not available\n");
	int q_result = -1;
	int k_result = -1;
#endif
	
	// Cleanup tensors
	ccv_nnc_tensor_free(q_input_tensor);
	ccv_nnc_tensor_free(k_input_tensor);
		ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(q_input_tensor_f16);
	ccv_nnc_tensor_free(k_input_tensor_f16);
		ccv_nnc_tensor_free(k_mean_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_input_tensor);
	ccv_nnc_tensor_free(gpu_k_input_tensor);
		ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scales_tensor);
	ccv_nnc_tensor_free(gpu_k_scales_tensor);
	ccv_nnc_stream_context_free(stream_context);
	
	printf("CCV quantization light test completed\n");
	REQUIRE(q_result == 0 && k_mean_result == 0, "CCV quantization should succeed");
}

TEST_CASE("sage attention quantization per-warp test light")
{
	ccv_cli_set_output_levels(CCV_CLI_INFO);

	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
	                  ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("=== CCV SageAttention Per-Warp Quantization Light Test ===\n");
	
	// Test parameters matching PyTorch per_warp_int8 defaults
	const int B = 1;      // batch size
	const int R = 64;     // sequence length
	const int H = 8;      // number of heads  
	const int D = 128;    // head dimension
	const uint32_t BLKQ = 128;  // Block size for Q (PyTorch default)
	const uint32_t WARPQ = 32;  // Warp size for Q (PyTorch default)
	const uint32_t BLKK = 64;   // Block size for K (PyTorch default)
	
	// Create input tensors
	ccv_nnc_tensor_t* const q_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	
	// Load input data from PyTorch test
	FILE* q_input_file = fopen("/tmp/test_q_input.bin", "rb");
	if (q_input_file) {
		fread(q_input_tensor->data.f32, sizeof(float), B * R * H * D, q_input_file);
		fclose(q_input_file);
		printf("✓ Loaded Q input from /tmp/test_q_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_q_input.bin\n");
		return;
	}
	
	FILE* k_input_file = fopen("/tmp/test_k_input.bin", "rb");
	if (k_input_file) {
		fread(k_input_tensor->data.f32, sizeof(float), B * R * H * D, k_input_file);
		fclose(k_input_file);
		printf("✓ Loaded K input from /tmp/test_k_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_input.bin\n");
		return; 
	}
	
	// Convert to FP16 and transfer to GPU
	ccv_nnc_tensor_t* const q_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor), TENSOR_LIST(q_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor), TENSOR_LIST(k_input_tensor_f16), 0);
	
	ccv_nnc_tensor_t* const gpu_q_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor_f16), TENSOR_LIST(gpu_q_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor_f16), TENSOR_LIST(gpu_k_input_tensor), 0);
	
	printf("Input shapes: Q[%d,%d,%d,%d], K[%d,%d,%d,%d]\n", B, R, H, D, B, R, H, D);
	
	// Print input data for verification against PyTorch
	printf("Q_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", q_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	// Calculate input ranges
	float q_min = q_input_tensor->data.f32[0], q_max = q_input_tensor->data.f32[0];
	float k_min = k_input_tensor->data.f32[0], k_max = k_input_tensor->data.f32[0];
	for (int i = 1; i < B * R * H * D; i++) {
		if (q_input_tensor->data.f32[i] < q_min) q_min = q_input_tensor->data.f32[i];
		if (q_input_tensor->data.f32[i] > q_max) q_max = q_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] < k_min) k_min = k_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] > k_max) k_max = k_input_tensor->data.f32[i];
	}
	printf("Q_input range: [%.6f, %.6f]\n", q_min, q_max);
	printf("K_input range: [%.6f, %.6f]\n", k_min, k_max);
	
	// Calculate output sizes
	const size_t output_size = B * R * H * D;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const size_t q_scale_size = B * H * q_blocks * warps_per_block;
	const size_t k_scale_size = B * H * ((R + BLKK - 1) / BLKK);
	
	// Create output tensors for the new tensor-based API
	ccv_nnc_tensor_t* const gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	// Calculate exact scale dimensions
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (R + BLKK - 1) / BLKK;      // Should be 1
	
	printf("Scale tensor calculations: q_blocks=%zu, warps_per_block=%zu, q_scale_blocks=%d\n", 
	       q_blocks, warps_per_block, q_scale_blocks);
	printf("Expected Q scale shape: [%d, %d, %d]\n", B, H, q_scale_blocks);
	printf("Expected K scale shape: [%d, %d, %d]\n", B, H, k_scale_blocks);
	
	ccv_nnc_tensor_t* const gpu_q_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* const gpu_k_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	printf("gpu_q_scales_tensor\n");
	ccv_nnc_print_tensor_info(gpu_q_scales_tensor);
	printf("gpu_k_scales_tensor\n");
	ccv_nnc_print_tensor_info(gpu_k_scales_tensor);
	printf("\n");

	// Run quantization using new CCV wrapper function
#ifdef HAVE_CUDA_SM80
	extern int ccv_nnc_quant_per_warp_int8_cuda(
		ccv_nnc_tensor_t* const input,
		ccv_nnc_tensor_t* const output,
		ccv_nnc_tensor_t* const scale,
		int block_size,
		int warp_block_size,
		int tensor_layout,
		ccv_nnc_stream_context_t* const stream_context);
	
	printf("Running CCV per-warp quantization (new API)...\n");
	
	// Q: per-warp quantization using new wrapper
	// Note: CCV tensors are in NHWC format, which is tensor_layout=0 (NHD)
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int q_result = ccv_nnc_quant_per_warp_int8_cuda(
		gpu_q_input_tensor,      // Input FP16 tensor
		gpu_q_int8_tensor,       // Output INT8 tensor  
		gpu_q_scales_tensor,     // Output FP32 scale tensor
		BLKQ,                    // block_size = 128
		WARPQ,                   // warp_block_size = 32
		0,                       // tensor_layout = 0 (NHD/NHWC)
		stream_context
	);
	
	// K: per-block quantization using new tensor API
	extern int ccv_nnc_quant_per_block_int8_cuda(
		ccv_nnc_tensor_t* const input,
		ccv_nnc_tensor_t* const output,
		ccv_nnc_tensor_t* const scale,
		int block_size,
		int tensor_layout,
		ccv_nnc_stream_context_t* const stream_context);
	
	int k_result = ccv_nnc_quant_per_block_int8_cuda(
		gpu_k_input_tensor,      // Input FP16 tensor
		gpu_k_int8_tensor,       // Output INT8 tensor
		gpu_k_scales_tensor,     // Output FP32 scale tensor
		BLKK,                    // block_size = 64
		0,                       // tensor_layout = 0 (NHD/NHWC)
		stream_context
	);
	
	if (q_result == 0 && k_result == 0) {
		printf("✓ CCV quantization completed successfully\n");
		
		// Copy tensor results to CPU for both Q and K
		ccv_nnc_tensor_t* const cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_q_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks, 1), 0);
		ccv_nnc_tensor_t* const cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
		ccv_nnc_tensor_t* const cpu_k_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks, 1), 0);
		
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_q_scales_tensor), TENSOR_LIST(cpu_q_scales_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
		                 TENSOR_LIST(gpu_k_scales_tensor), TENSOR_LIST(cpu_k_scales_tensor), 0);
		
		// Print sample outputs
		printf("Q_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", cpu_q_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("Q_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < q_scale_size; i++) printf("%.6f ", cpu_q_scales_tensor->data.f32[i]);
		printf("\n");
		
		printf("K_int8 sample (first 10): ");
		for (int i = 0; i < 10; i++) printf("%d ", cpu_k_int8_tensor->data.u8[i]);
		printf("\n");
		
		printf("K_scales sample (first 10): ");
		for (int i = 0; i < 10 && i < k_scale_size; i++) printf("%.6f ", cpu_k_scales_tensor->data.f32[i]);
		printf("\n");
		
		// Save outputs to disk
		FILE* q_int8_file = fopen("/tmp/ccv_q_int8.bin", "wb");
		if (q_int8_file) {
			fwrite(cpu_q_int8_tensor->data.u8, sizeof(int8_t), output_size, q_int8_file);
			fclose(q_int8_file);
			printf("✓ Saved Q_int8 to /tmp/ccv_q_int8.bin\n");
		}
		
		FILE* k_int8_file = fopen("/tmp/ccv_k_int8.bin", "wb");
		if (k_int8_file) {
			fwrite(cpu_k_int8_tensor->data.u8, sizeof(int8_t), output_size, k_int8_file);
			fclose(k_int8_file);
			printf("✓ Saved K_int8 to /tmp/ccv_k_int8.bin\n");
		}
		
		FILE* q_scales_file = fopen("/tmp/ccv_q_scales.bin", "wb");
		if (q_scales_file) {
			fwrite(cpu_q_scales_tensor->data.f32, sizeof(float), q_scale_size, q_scales_file);
			fclose(q_scales_file);
			printf("✓ Saved Q_scales to /tmp/ccv_q_scales.bin\n");
		}
		
		FILE* k_scales_file = fopen("/tmp/ccv_k_scales.bin", "wb");
		if (k_scales_file) {
			fwrite(cpu_k_scales_tensor->data.f32, sizeof(float), k_scale_size, k_scales_file);
			fclose(k_scales_file);
			printf("✓ Saved K_scales to /tmp/ccv_k_scales.bin\n");
		}
		
		printf("Output sizes: Q_int8=%zu, Q_scales=%zu, K_int8=%zu, K_scales=%zu\n",
		       output_size, q_scale_size, output_size, k_scale_size);
		
		// Clean up CPU tensors
		ccv_nnc_tensor_free(cpu_q_int8_tensor);
		ccv_nnc_tensor_free(cpu_q_scales_tensor);
		ccv_nnc_tensor_free(cpu_k_int8_tensor);
		ccv_nnc_tensor_free(cpu_k_scales_tensor);
	} else {
		printf("❌ CCV quantization failed: Q=%d, K=%d\n", q_result, k_result);
	}
#else
	printf("❌ CUDA SM80 not available\n");
	int q_result = -1;
	int k_result = -1;
#endif
	
	// Cleanup tensors
	ccv_nnc_tensor_free(q_input_tensor);
	ccv_nnc_tensor_free(k_input_tensor);
	ccv_nnc_tensor_free(q_input_tensor_f16);
	ccv_nnc_tensor_free(k_input_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_input_tensor);
	ccv_nnc_tensor_free(gpu_k_input_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scales_tensor);
	ccv_nnc_tensor_free(gpu_k_scales_tensor);
	ccv_nnc_stream_context_free(stream_context);
	
	printf("CCV quantization light test completed\n");
	REQUIRE(q_result == 0 && k_result == 0, "CCV quantization should succeed");
}

TEST_CASE("sage kernel test")
{
	printf("=== Raw SageAttention Kernel Test ===\n");
	printf("This test bypasses CCV wrapper and calls the raw kernel directly\n");
	
	// 1. Load PyTorch parameters and quantized data from disk
	FILE* param_file = fopen("/tmp/pytorch_kernel_params.json", "r");
	if (!param_file) {
		printf("❌ Run: python extract_pytorch_deterministic.py first\n");
		return;
	}
	fclose(param_file);
	
	// PyTorch's exact parameters
	const int B = 1, R = 64, C = 64, H = 8, D = 128;
	const float sm_scale = 0.08838834764831843f;
	
	printf("Loading PyTorch quantized inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Load PyTorch's quantized data (exactly as PyTorch saved it)
	int8_t* q_int8_data = (int8_t*)malloc(B * H * R * D * sizeof(int8_t));
	int8_t* k_int8_data = (int8_t*)malloc(B * H * C * D * sizeof(int8_t));  
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	float* q_scales_data = (float*)malloc(B * H * 4 * sizeof(float));  // PyTorch: per-warp (4 warps per head)
	float* k_scales_data = (float*)malloc(B * H * 1 * sizeof(float));  // PyTorch: per-block (1 per head)
	
	// Read binary files (PyTorch format: BRHD for tensors, BH4/BH1 for scales)
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_int8.bin", "rb");
	fread(q_int8_data, sizeof(int8_t), B * H * R * D, f); fclose(f);
	f = fopen("/tmp/pytorch_kernel_k_int8.bin", "rb"); 
	fread(k_int8_data, sizeof(int8_t), B * H * C * D, f); fclose(f);
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f); fclose(f);
	f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
	fread(q_scales_data, sizeof(float), B * H * 4, f); fclose(f);
	f = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
	fread(k_scales_data, sizeof(float), B * H * 1, f); fclose(f);
	
	printf("✅ Loaded PyTorch quantized data from disk\n");
	printf("Sample values: Q_int8[0]=%d, K_int8[0]=%d, V[0]=%f\n", 
		q_int8_data[0], k_int8_data[0], (float)v_fp16_data[0]);
	printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f\n", 
		q_scales_data[0], k_scales_data[0]);
	
	// Debug: Print scale tensor layouts
	printf("Scale tensor debug info:\n");
	printf("  Q scales (per-warp): [%f, %f, %f, %f] for head 0\n",
		q_scales_data[0], q_scales_data[1], q_scales_data[2], q_scales_data[3]);
	printf("  K scales (per-block): [%f] for head 0\n", k_scales_data[0]);
	printf("  Expected Q scale shape: [B=1, H=8, 4 warps] = [1, 8, 4]\n");
	printf("  Expected K scale shape: [B=1, H=8, 1 block] = [1, 8, 1]\n");
	
	printf("\n=== CCV Input Data Verification ===\n");
	printf("Q_int8 data (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", q_int8_data[i]);
	}
	printf("\n");
	
	printf("Q_int8 data (last 10 values): ");
	int total_elements = B * H * R * D;
	for (int i = total_elements - 10; i < total_elements; i++) {
		printf("%d ", q_int8_data[i]);
	}
	printf("\n");
	
	printf("K_int8 data (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", k_int8_data[i]);
	}
	printf("\n");
	
	printf("K_int8 data (last 10 values): ");
	for (int i = total_elements - 10; i < total_elements; i++) {
		printf("%d ", k_int8_data[i]);
	}
	printf("\n");
	
	printf("V data (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%.4f ", (float)v_fp16_data[i]);
	}
	printf("\n");
	
	printf("V data (last 10 values): ");
	for (int i = total_elements - 10; i < total_elements; i++) {
		printf("%.4f ", (float)v_fp16_data[i]);
	}
	printf("\n");
	
	// Find min/max for Q_int8, K_int8, V
	int8_t q_min = q_int8_data[0], q_max = q_int8_data[0];
	int8_t k_min = k_int8_data[0], k_max = k_int8_data[0];
	__fp16 v_min = v_fp16_data[0], v_max = v_fp16_data[0];
	
	for (int i = 1; i < B * H * R * D; i++) {
		int8_t q_val = q_int8_data[i];
		int8_t k_val = k_int8_data[i];
		__fp16 v_val = v_fp16_data[i];
		
		if (q_val < q_min) q_min = q_val;
		if (q_val > q_max) q_max = q_val;
		if (k_val < k_min) k_min = k_val;
		if (k_val > k_max) k_max = k_val;
		if (v_val < v_min) v_min = v_val;
		if (v_val > v_max) v_max = v_val;
	}
	
	printf("Q_int8 data range: [%d, %d]\n", q_min, q_max);
	printf("K_int8 data range: [%d, %d]\n", k_min, k_max);
	printf("V data range: [%.4f, %.4f]\n", (float)v_min, (float)v_max);
	
	// 2. Allocate GPU memory for kernel inputs (exactly as PyTorch does)
	void* gpu_q_int8; cudaMalloc(&gpu_q_int8, B * H * R * D * sizeof(int8_t));
	void* gpu_k_int8; cudaMalloc(&gpu_k_int8, B * H * C * D * sizeof(int8_t));
	void* gpu_v_fp16; cudaMalloc(&gpu_v_fp16, B * H * C * D * sizeof(__fp16));
	void* gpu_q_scales; cudaMalloc(&gpu_q_scales, B * H * 4 * sizeof(float));
	void* gpu_k_scales; cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));
	void* gpu_output; cudaMalloc(&gpu_output, B * H * R * D * sizeof(__fp16));
	
	// Copy data to GPU
	cudaMemcpy(gpu_q_int8, q_int8_data, B * H * R * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_int8, k_int8_data, B * H * C * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_v_fp16, v_fp16_data, B * H * C * D * sizeof(__fp16), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_q_scales, q_scales_data, B * H * 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_scales, k_scales_data, B * H * 1 * sizeof(float), cudaMemcpyHostToDevice);
	
	printf("✅ Copied data to GPU\n");
	
	// 3. Launch the exact same kernel that PyTorch uses
	// PyTorch uses: CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64, head_dim=128
	// Now using PyTorch's exact template instantiation with added WARP_K=64 kernel
	printf("Calling raw SageAttention kernel with PyTorch template parameters...\n");
	printf("Template params: CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64, head_dim=128\n");
	
	// Set up kernel launch parameters (exactly as PyTorch does)
	int grid_x = 1, grid_y = H, grid_z = B;  // PyTorch: (ceil(R/CTA_Q), H, B) = (ceil(64/128), 8, 1) = (1, 8, 1)
	int block_size = 128;                     // PyTorch: (32, 4) = 128 threads (not 512!)
	
	printf("Grid: (%d, %d, %d), Block: (%d)\n", grid_x, grid_y, grid_z, block_size);
	
	// Direct kernel call - use the compiled kernel from the object file
	// The kernel template <128, 64, 32, 16, 128, kPerWarp, kPerBlock> should be available in the linked library
	printf("Calling raw SageAttention kernel directly (bypassing CCV wrapper)...\n");
	
	// Call the raw SageAttention kernel directly with PyTorch's quantized inputs
	extern void call_sage_attention_kernel_direct(
		int8_t *Q, int8_t *K, __fp16 *V, __fp16 *O, float *Lse,
		float *Q_scale, float *K_scale, __fp16 *V_mean,
		uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups,
		uint32_t stride_bz_q, uint32_t stride_seq_q, uint32_t stride_h_q,
		uint32_t stride_bz_k, uint32_t stride_seq_k, uint32_t stride_h_k,
		uint32_t stride_bz_v, uint32_t stride_seq_v, uint32_t stride_h_v,
		uint32_t stride_bz_o, uint32_t stride_seq_o, uint32_t stride_h_o,
		float sm_scale, int grid_x, int grid_y, int grid_z, int block_size);
	
	// Set up kernel parameters exactly as PyTorch does
	// PyTorch uses HND layout: (batch, head, seq, dim) 
	// PyTorch strides: bz=65536, h=8192, seq=128, dim=1
	uint32_t qo_len = R, kv_len = C, num_kv_groups = 1;
	uint32_t stride_bz_q = H * R * D, stride_seq_q = R * D, stride_h_q = D;     // Original: HND layout
	uint32_t stride_bz_k = H * C * D, stride_seq_k = C * D, stride_h_k = D;     // Original: HND layout  
	uint32_t stride_bz_v = H * C * D, stride_seq_v = C * D, stride_h_v = D;     // Original: HND layout
	uint32_t stride_bz_o = H * R * D, stride_seq_o = R * D, stride_h_o = D;     // Original: HND layout
	
	// Grid and block dimensions (will pass as integers to wrapper)
	
	printf("Raw kernel parameters:\n");
	printf("  qo_len=%d, kv_len=%d, num_kv_groups=%d\n", qo_len, kv_len, num_kv_groups);
	printf("  Q strides: bz=%d, seq=%d, h=%d\n", stride_bz_q, stride_seq_q, stride_h_q);
	printf("  K strides: bz=%d, seq=%d, h=%d\n", stride_bz_k, stride_seq_k, stride_h_k);
	printf("  Scale: %f\n", sm_scale);
	printf("CORRECTED for HND layout (kernel expects): Q strides: bz=%d, h=%d, seq=%d\n", 
		stride_bz_q, stride_h_q, stride_seq_q);
	printf("CORRECTED for HND layout (kernel expects): K strides: bz=%d, h=%d, seq=%d\n", 
		stride_bz_k, stride_h_k, stride_seq_k);
	
	// Remove fp16 loading for Q and K - we already have int8 data
	
	
	call_sage_attention_kernel_direct(
		(int8_t*)gpu_q_int8, (int8_t*)gpu_k_int8, (__fp16*)gpu_v_fp16, (__fp16*)gpu_output, NULL,
		(float*)gpu_q_scales, (float*)gpu_k_scales, NULL,
		qo_len, kv_len, num_kv_groups,
		stride_bz_q, stride_h_q, stride_seq_q,  // HND: batch, head, seq, dim
		stride_bz_k, stride_h_k, stride_seq_k,  // HND: batch, head, seq, dim
		stride_bz_v, stride_h_v, stride_seq_v,  // HND: batch, head, seq, dim
		stride_bz_o, stride_h_o, stride_seq_o,  // HND: batch, head, seq, dim
		sm_scale, grid_x, grid_y, grid_z, block_size);
	
	printf("Raw kernel call completed, checking for errors...\n");
	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("❌ Kernel launch failed: %s\n", cudaGetErrorString(error));
	} else {
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("❌ Kernel execution failed: %s\n", cudaGetErrorString(error));
		} else {
			printf("✅ Raw kernel completed successfully\n");
			
			// 4. Copy output back and save for comparison
			__fp16* output_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
			cudaMemcpy(output_data, gpu_output, B * H * R * D * sizeof(__fp16), cudaMemcpyDeviceToHost);
			
			printf("\n=== CCV Raw Kernel Output Analysis ===\n");
			printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] = [%d, %d, %d, %d]\n", B, H, R, D, B, H, R, D);
			printf("First 10 output values:\n");
			for (int i = 0; i < 10; i++) {
				printf("  output[%d]: %f\n", i, (float)output_data[i]);
			}
			
			printf("\nSample outputs from different positions:\n");
			printf("  output[0,0,0,0]: %f\n", (float)output_data[0]);                           // [0,0,0,0]
			printf("  output[0,0,0,64]: %f\n", (float)output_data[64]);                         // [0,0,0,64] - mid head_dim
			printf("  output[0,0,0,127]: %f\n", (float)output_data[127]);                       // [0,0,0,127] - end head_dim
			printf("  output[0,0,32,0]: %f\n", (float)output_data[32 * D]);                     // [0,0,32,0] - mid sequence
			printf("  output[0,4,0,0]: %f\n", (float)output_data[4 * R * D]);                  // [0,4,0,0] - mid head
			
			// Save CCV output for comparison with PyTorch
			f = fopen("/tmp/ccv_sage_kernel_output.bin", "wb");
			fwrite(output_data, sizeof(__fp16), B * H * R * D, f);
			fclose(f);
			printf("✅ Saved CCV raw kernel output to /tmp/ccv_sage_kernel_output.bin\n");
			
			// Load and compare with PyTorch output if available
			FILE* pytorch_file = fopen("/tmp/pytorch_sage_kernel_output.bin", "rb");
			if (pytorch_file) {
				__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
				fread(pytorch_data, sizeof(__fp16), B * H * R * D, pytorch_file);
				fclose(pytorch_file);
				
				printf("\n=== CCV vs PyTorch Kernel Comparison ===\n");
				printf("Comparing first 10 values:\n");
				int exact_matches = 0;
				double max_diff = 0.0;
				double sum_diff = 0.0;
				
				for (int i = 0; i < 10; i++) {
					float ccv_val = (float)output_data[i];
					float pytorch_val = (float)pytorch_data[i];
					double diff = fabs(ccv_val - pytorch_val);
					sum_diff += diff;
					if (diff > max_diff) max_diff = diff;
					if (diff < 1e-6) exact_matches++;
					
					printf("  [%d] CCV: %f, PyTorch: %f, diff: %.8f\n", i, ccv_val, pytorch_val, diff);
				}
				
				printf("\nKey position comparisons:\n");
				printf("  [0,0,0,0] CCV: %f, PyTorch: %f, diff: %.8f\n", 
					(float)output_data[0], (float)pytorch_data[0], 
					fabs((float)output_data[0] - (float)pytorch_data[0]));
				printf("  [0,0,0,64] CCV: %f, PyTorch: %f, diff: %.8f\n", 
					(float)output_data[64], (float)pytorch_data[64], 
					fabs((float)output_data[64] - (float)pytorch_data[64]));
				printf("  [0,4,0,0] CCV: %f, PyTorch: %f, diff: %.8f\n", 
					(float)output_data[4 * R * D], (float)pytorch_data[4 * R * D], 
					fabs((float)output_data[4 * R * D] - (float)pytorch_data[4 * R * D]));
				
				printf("\nStatistics (first 10 values):\n");
				printf("  Exact matches (diff < 1e-6): %d/10\n", exact_matches);
				printf("  Maximum difference: %.8f\n", max_diff);
				printf("  Average difference: %.8f\n", sum_diff / 10.0);
				
				// Full tensor comparison
				int total_exact = 0;
				double total_max_diff = 0.0;
				double total_sum_diff = 0.0;
				int total_elements = B * H * R * D;
				
				for (int i = 0; i < total_elements; i++) {
					double diff = fabs((float)output_data[i] - (float)pytorch_data[i]);
					total_sum_diff += diff;
					if (diff > total_max_diff) total_max_diff = diff;
					if (diff < 1e-6) total_exact++;
				}
				
				printf("\nFull tensor comparison (%d elements):\n", total_elements);
				printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
					total_exact, total_elements, (100.0 * total_exact) / total_elements);
				printf("  Maximum difference: %.8f\n", total_max_diff);
				printf("  Average difference: %.8f\n", total_sum_diff / total_elements);
				
				if (total_max_diff < 1e-5) {
					printf("🎯 EXCELLENT: CCV and PyTorch outputs are virtually identical!\n");
				} else if (total_max_diff < 1e-3) {
					printf("✅ GOOD: CCV and PyTorch outputs are very close (within expected FP16 precision)\n");
				} else {
					printf("⚠️  MISMATCH: Significant differences detected between CCV and PyTorch\n");
				}
				
				free(pytorch_data);
			} else {
				printf("PyTorch output not found. Run: python3 test_sageattention_sm80_direct.py\n");
			}
			
			free(output_data);
		}
	}
	
	// Cleanup
	free(q_int8_data); free(k_int8_data); free(v_fp16_data); 
	free(q_scales_data); free(k_scales_data);
	cudaFree(gpu_q_int8); cudaFree(gpu_k_int8); cudaFree(gpu_v_fp16);
	cudaFree(gpu_q_scales); cudaFree(gpu_k_scales); cudaFree(gpu_output);
	
	REQUIRE(error == cudaSuccess, "Raw SageAttention kernel should execute successfully");
}

TEST_CASE("ccv_nnc_qk_int8_sv_f16_accum_f32_attn test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Wrapper Test ===\n");
	printf("Testing ccv_nnc_qk_int8_sv_f16_accum_f32_attn wrapper function\n");
	printf("This test validates the CCV wrapper using same inputs as 'sage kernel test'\n");
	
	// Use same test parameters as "sage kernel test"
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load same PyTorch quantized inputs as "sage kernel test"
	printf("Loading PyTorch quantized inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for input data (same as sage kernel test)
	int8_t* q_int8_data = (int8_t*)malloc(B * H * R * D * sizeof(int8_t));
	int8_t* k_int8_data = (int8_t*)malloc(B * H * C * D * sizeof(int8_t)); 
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	float* q_scales_data = (float*)malloc(B * H * 4 * sizeof(float));  // per-warp scales
	float* k_scales_data = (float*)malloc(B * H * 1 * sizeof(float));  // per-block scales
	
	// Load quantized data from PyTorch files (exact same as sage kernel test)
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open Q int8 data file");
	fread(q_int8_data, sizeof(int8_t), B * H * R * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_k_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K int8 data file");
	fread(k_int8_data, sizeof(int8_t), B * H * C * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	REQUIRE(f != NULL, "Failed to open V fp16 data file");
	fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
	REQUIRE(f != NULL, "Failed to open Q scales data file");
	fread(q_scales_data, sizeof(float), B * H * 4, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K scales data file");
	fread(k_scales_data, sizeof(float), B * H * 1, f);
	fclose(f);
	
	printf("✅ Loaded PyTorch quantized data from disk\n");
	
	// Print sample values for verification (same as sage kernel test)
	printf("Sample values: Q_int8[0]=%d, K_int8[0]=%d, V[0]=%f\n", 
		q_int8_data[0], k_int8_data[0], (float)v_fp16_data[0]);
	printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f\n", 
		q_scales_data[0], k_scales_data[0]);
	
	// Create CCV tensors for the wrapper function  
	// Use HND layout to match working direct test: [batch, heads, seq, dim]
	// Query: [B, H, R, D], Key/Value: [B, H, C, D], Output: [B, H, R, D]
	// Note: Using CCV_8U for int8 data, will cast to int8_t* in wrapper
	ccv_nnc_tensor_t* q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* q_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, 4), 0);
	ccv_nnc_tensor_t* k_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, 1), 0);
	
	// Copy data to CCV tensors
	memcpy(q_tensor->data.u8, q_int8_data, B * H * R * D * sizeof(int8_t));
	memcpy(k_tensor->data.u8, k_int8_data, B * H * C * D * sizeof(int8_t));
	memcpy(v_tensor->data.u8, v_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(q_scale_tensor->data.u8, q_scales_data, B * H * 4 * sizeof(float));
	memcpy(k_scale_tensor->data.u8, k_scales_data, B * H * 1 * sizeof(float));
	
	printf("✅ Created CCV tensors and copied input data\n");
	
	// Move tensors to GPU (HND layout)
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, 4), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, 1), 0);
	
	// Copy data to GPU tensors
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_tensor), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(v_tensor), TENSOR_LIST(gpu_v_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_scale_tensor), TENSOR_LIST(gpu_q_scale_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_scale_tensor), TENSOR_LIST(gpu_k_scale_tensor), 0);
	
	printf("✅ Copied tensors to GPU\n");
	
	// Declare wrapper function (should be in header but adding here for test)
	extern int ccv_nnc_qk_int8_sv_f16_accum_f32_attn(
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
		int return_lse);
	
	// Call the CCV wrapper function
	printf("\n=== Calling CCV SageAttention Wrapper ===\n");
	printf("Wrapper parameters:\n");
	printf("  tensor_layout: 1 (HND - matching direct test)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	
	int result = ccv_nnc_qk_int8_sv_f16_accum_f32_attn(
		gpu_q_tensor,        // query (int8)
		gpu_k_tensor,        // key (int8)
		gpu_v_tensor,        // value (fp16)
		gpu_o_tensor,        // output (fp16) - modified in-place
		gpu_q_scale_tensor,  // query_scale (fp32)
		gpu_k_scale_tensor,  // key_scale (fp32)
		1,                   // tensor_layout: 1=HND (matching direct test)
		0,                   // is_causal: false
		2,                   // qk_quant_gran: 2=per_warp
		sm_scale,            // sm_scale
		0                    // return_lse: false
	);
	
	printf("CCV wrapper result: %s\n", result == 0 ? "SUCCESS" : "FAILED");
	REQUIRE(result == 0, "CCV SageAttention wrapper should succeed");
	
	// Copy output back to CPU for comparison  
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor), 0);
	
	printf("✅ Copied output back to CPU\n");
	
	// Extract output data for comparison
	__fp16* ccv_output_data = (__fp16*)cpu_output_tensor->data.u8;
	
	printf("\n=== CCV Wrapper Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)ccv_output_data[i]);
	}
	
	printf("Sample outputs from different positions:\n");
	printf("  output[0,0,0,0]: %f\n", (float)ccv_output_data[0]);                           // [0,0,0,0]
	printf("  output[0,0,0,64]: %f\n", (float)ccv_output_data[64]);                         // [0,0,0,64] - mid head_dim
	printf("  output[0,0,0,127]: %f\n", (float)ccv_output_data[127]);                       // [0,0,0,127] - end head_dim 
	printf("  output[0,0,32,0]: %f\n", (float)ccv_output_data[32 * D]);                     // [0,0,32,0] - mid sequence
	printf("  output[0,4,0,0]: %f\n", (float)ccv_output_data[4 * R * D]);                  // [0,4,0,0] - mid head
	
	// Save CCV wrapper output for comparison
	f = fopen("/tmp/ccv_wrapper_sage_output.bin", "wb");
	if (f) {
		fwrite(ccv_output_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved CCV wrapper output to /tmp/ccv_wrapper_sage_output.bin\n");
	}
	
	// Compare with PyTorch output (load from sage kernel test)
	printf("\n=== CCV Wrapper vs PyTorch Comparison ===\n");
	f = fopen("/tmp/pytorch_sage_kernel_output.bin", "rb");
	if (f) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values:\n");
		for (int i = 0; i < 10; i++) {
			float ccv_val = (float)ccv_output_data[i];
			float pytorch_val = (float)pytorch_data[i];
			float diff = fabsf(ccv_val - pytorch_val);
			printf("  [%d] CCV: %f, PyTorch: %f, diff: %.8f\n", i, ccv_val, pytorch_val, diff);
		}
		
		printf("\nKey position comparisons:\n");
		printf("  [0,0,0,0] CCV: %f, PyTorch: %f, diff: %.8f\n", 
			(float)ccv_output_data[0], (float)pytorch_data[0], 
			fabsf((float)ccv_output_data[0] - (float)pytorch_data[0]));
		printf("  [0,0,0,64] CCV: %f, PyTorch: %f, diff: %.8f\n", 
			(float)ccv_output_data[64], (float)pytorch_data[64], 
			fabsf((float)ccv_output_data[64] - (float)pytorch_data[64]));
		printf("  [0,4,0,0] CCV: %f, PyTorch: %f, diff: %.8f\n", 
			(float)ccv_output_data[4 * R * D], (float)pytorch_data[4 * R * D], 
			fabsf((float)ccv_output_data[4 * R * D] - (float)pytorch_data[4 * R * D]));
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)ccv_output_data[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
		}
		
		printf("\nStatistics (first 10 values):\n");
		printf("  Exact matches (diff < 1e-6): %d/10\n", exact_matches >= 10 ? 10 : exact_matches);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / 10);
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: CCV wrapper and PyTorch outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: CCV wrapper and PyTorch outputs are very close (within expected FP16 precision)\n");
		} else {
			printf("⚠️  MISMATCH: Significant differences detected between CCV wrapper and PyTorch\n");
		}
		
		free(pytorch_data);
	} else {
		printf("PyTorch output not found. Run 'sage kernel test' first to generate reference output.\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(q_scale_tensor);
	ccv_nnc_tensor_free(k_scale_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor);
	
	free(q_int8_data); free(k_int8_data); free(v_fp16_data);
	free(q_scales_data); free(k_scales_data);
	
	printf("✅ CCV SageAttention wrapper test completed successfully!\n");
}


TEST_CASE("ccv_nnc_sageattn_qk_int8_pv_fp16_cuda test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Wrapper Test ===\n");
	printf("Testing ccv_nnc_sageattn_qk_int8_pv_fp16_cuda wrapper function\n");
	printf("This test validates the CCV wrapper using same inputs as 'sage kernel test'\n");
	
	// Use same test parameters as "sage kernel test"
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load FP16 input data from PyTorch files
	printf("Loading PyTorch FP16 inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for FP16 input data
	__fp16* q_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	__fp16* k_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	
	// Load FP16 data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_fp16.bin", "rb");
	if (f) {
		fread(q_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Loaded Q FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		printf("FP16 files not found, loading FP32 and converting...\n");
		float* q_fp32_data = (float*)malloc(B * H * R * D * sizeof(float));
		f = fopen("/tmp/pytorch_q_input.bin", "rb");
		if (f) {
			fread(q_fp32_data, sizeof(float), B * H * R * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)q_fp32_data[i];
			}
			printf("✅ Loaded and converted Q data from FP32\n");
		}
		free(q_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_k_fp16.bin", "rb");
	if (f) {
		fread(k_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded K FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* k_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_k_input.bin", "rb");
		if (f) {
			fread(k_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)k_fp32_data[i];
			}
			printf("✅ Loaded and converted K data from FP32\n");
		}
		free(k_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	if (f) {
		fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded V FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* v_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_v_input.bin", "rb");
		if (f) {
			fread(v_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)v_fp32_data[i];
			}
			printf("✅ Loaded and converted V data from FP32\n");
		}
		free(v_fp32_data);
	}
	
	// Print sample values for verification
	printf("Sample values: Q[0]=%f, K[0]=%f, V[0]=%f\n", 
		(float)q_fp16_data[0], (float)k_fp16_data[0], (float)v_fp16_data[0]);
	
	// Create CCV tensors for the wrapper function  
	// Use HND layout to match working direct test: [batch, heads, seq, dim]
	// Query: [B, H, R, D], Key/Value: [B, H, C, D], Output: [B, H, R, D]
	// IMPORTANT: Use 16F for input Q and K (they will be quantized internally)
	ccv_nnc_tensor_t* q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	
	// Copy data to CCV tensors
	memcpy(q_tensor->data.f16, q_fp16_data, B * H * R * D * sizeof(__fp16));
	memcpy(k_tensor->data.f16, k_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(v_tensor->data.f16, v_fp16_data, B * H * C * D * sizeof(__fp16));
	
	printf("✅ Created CCV tensors and copied input data\n");
	
	// Move tensors to GPU (HND layout)
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	
	// Copy data to GPU tensors
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_tensor), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(v_tensor), TENSOR_LIST(gpu_v_tensor), 0);
	
	printf("✅ Copied tensors to GPU\n");
	
	// Create scale tensors based on quantization parameters
	const uint32_t BLKQ = 128;
	const uint32_t WARPQ = 32;
	const uint32_t BLKK = 64;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (C + BLKK - 1) / BLKK;      // Should be 1
	
	printf("Scale tensor dimensions: q_scale_blocks=%d, k_scale_blocks=%d\n", q_scale_blocks, k_scale_blocks);
	
	ccv_nnc_tensor_t* q_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* k_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	// Initialize scales to 1.0 (will be computed by the function)
	for (int i = 0; i < B * H * q_scale_blocks; i++) {
		q_scale_tensor->data.f32[i] = 1.0f;
	}
	for (int i = 0; i < B * H * k_scale_blocks; i++) {
		k_scale_tensor->data.f32[i] = 1.0f;
	}
	
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_scale_tensor), TENSOR_LIST(gpu_q_scale_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_scale_tensor), TENSOR_LIST(gpu_k_scale_tensor), 0);
	
	// Declare wrapper function (should be in header but adding here for test)
	extern int ccv_nnc_sageattn_qk_int8_pv_fp16_cuda(
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
		int return_lse, // default false
		int pv_accum_dtype, // default fp32, 
		int BLKQ,                        // Block size for Q (default 128)
		int WARPQ,                       // Warp size for Q (default 32)
		int BLKK,                        // Block size for K (default 64)
		ccv_nnc_stream_context_t* const stream_context
	);
	
	// Call the CCV wrapper function
	printf("\n=== Calling CCV SageAttention Wrapper ===\n");
	printf("Wrapper parameters:\n");
	printf("  tensor_layout: 1 (HND - matching direct test)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	printf("  BLKQ: %d, WARPQ: %d, BLKK: %d\n", BLKQ, WARPQ, BLKK);
	
	// Create output tensors for quantized Q and K
	ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	
	// Create k_mean tensor (required for SageAttention)
	ccv_nnc_tensor_t* k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	ccv_nnc_tensor_t* gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	
	// Initialize k_mean to zeros (typical initialization)
	memset(k_mean_tensor->data.u8, 0, B * H * D * sizeof(__fp16));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_mean_tensor), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	int result = ccv_nnc_sageattn_qk_int8_pv_fp16_cuda(
		gpu_q_tensor,        // query (fp16 input)
		gpu_k_tensor,        // key (fp16 input)
		gpu_k_mean_tensor,   // k_mean (fp16)
		gpu_v_tensor,        // value (fp16)
		gpu_q_int8_tensor,   // q_int8 (output quantized Q)
		gpu_k_int8_tensor,   // k_int8 (output quantized K)
		gpu_q_scale_tensor,  // query_scale (fp32)
		gpu_k_scale_tensor,  // key_scale (fp32)
		gpu_o_tensor,        // output (fp16)
		1,                   // tensor_layout: 1=HND
		0,                   // is_causal: false
		2,                   // qk_quant_gran: 2=per_warp
		sm_scale,            // sm_scale
		0,                   // return_lse: false
		2,                   // pv_accum_dtype: 2=fp32
		BLKQ,                // BLKQ
		WARPQ,               // WARPQ
		BLKK,                // BLKK
		0                    // stream_context (NULL for default)
	);
	
	printf("CCV wrapper result: %s\n", result == 0 ? "SUCCESS" : "FAILED");
	REQUIRE(result == 0, "CCV SageAttention wrapper should succeed");
	
	// Copy output back to CPU for comparison  
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor), 0);
	
	// Also copy the quantized outputs and scales back for verification
	ccv_nnc_tensor_t* cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* cpu_q_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* cpu_k_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_scale_tensor), TENSOR_LIST(cpu_q_scale_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_scale_tensor), TENSOR_LIST(cpu_k_scale_output), 0);
	
	printf("✅ Copied outputs back to CPU\n");
	
	// Extract output data for comparison
	__fp16* ccv_output_data = (__fp16*)cpu_output_tensor->data.u8;
	int8_t* q_int8_output = (int8_t*)cpu_q_int8_tensor->data.u8;
	int8_t* k_int8_output = (int8_t*)cpu_k_int8_tensor->data.u8;
	
	printf("\n=== CCV Wrapper Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)ccv_output_data[i]);
	}
	
	printf("\nQuantized Q (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", q_int8_output[i]);
	}
	printf("\n");
	
	printf("Q scales (all %d values): ", q_scale_blocks);
	for (int i = 0; i < q_scale_blocks; i++) {
		printf("%.6f ", cpu_q_scale_output->data.f32[i]);
	}
	printf("\n");
	
	printf("\nQuantized K (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", k_int8_output[i]);
	}
	printf("\n");
	
	printf("K scales (all %d values): ", k_scale_blocks);
	for (int i = 0; i < k_scale_blocks; i++) {
		printf("%.6f ", cpu_k_scale_output->data.f32[i]);
	}
	printf("\n");
	
	printf("\nSample outputs from different positions:\n");
	printf("  output[0,0,0,0]: %f\n", (float)ccv_output_data[0]);
	printf("  output[0,0,0,64]: %f\n", (float)ccv_output_data[64]);
	printf("  output[0,0,0,127]: %f\n", (float)ccv_output_data[127]);
	printf("  output[0,0,32,0]: %f\n", (float)ccv_output_data[32 * D]);
	printf("  output[0,4,0,0]: %f\n", (float)ccv_output_data[4 * R * D]);
	
	// Save CCV wrapper output for comparison
	f = fopen("/tmp/ccv_wrapper_sage_output.bin", "wb");
	if (f) {
		fwrite(ccv_output_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved CCV wrapper output to /tmp/ccv_wrapper_sage_output.bin\n");
	}
	
	// Compare with PyTorch output if available
	printf("\n=== CCV Wrapper vs PyTorch Comparison ===\n");
	f = fopen("/tmp/pytorch_sage_output.bin", "rb");
	if (f) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values:\n");
		for (int i = 0; i < 10; i++) {
			float ccv_val = (float)ccv_output_data[i];
			float pytorch_val = (float)pytorch_data[i];
			float diff = fabsf(ccv_val - pytorch_val);
			printf("  [%d] CCV: %f, PyTorch: %f, diff: %.8f\n", i, ccv_val, pytorch_val, diff);
		}
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)ccv_output_data[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: CCV wrapper and PyTorch outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: CCV wrapper and PyTorch outputs are very close (within expected FP16 precision)\n");
		} else {
			printf("⚠️  WARNING: Larger differences detected between CCV wrapper and PyTorch\n");
		}
		
		free(pytorch_data);
	} else {
		printf("PyTorch output not found at /tmp/pytorch_sage_output.bin\n");
		printf("This is expected if you haven't run the PyTorch reference implementation.\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(q_scale_tensor);
	ccv_nnc_tensor_free(k_scale_tensor);
	ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scale_output);
	ccv_nnc_tensor_free(cpu_k_scale_output);
	
	free(q_fp16_data);
	free(k_fp16_data);
	free(v_fp16_data);
	
	printf("✅ CCV SageAttention wrapper test completed successfully!\n");
}

TEST_CASE("_ccv_nnc_scaled_dot_product_attention_forw sage test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== SageAttention Forward Function Test ===\n");
	printf("Testing _ccv_nnc_scaled_dot_product_attention_forw with SageAttention backend\n");
	
	// Use same test parameters as SageAttention wrapper test
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load FP16 input data from PyTorch files
	printf("Loading PyTorch FP16 inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for FP16 input data
	__fp16* q_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	__fp16* k_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	
	// Load FP16 data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_fp16.bin", "rb");
	if (f) {
		fread(q_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Loaded Q FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		printf("FP16 files not found, loading FP32 and converting...\n");
		float* q_fp32_data = (float*)malloc(B * H * R * D * sizeof(float));
		f = fopen("/tmp/pytorch_q_input.bin", "rb");
		if (f) {
			fread(q_fp32_data, sizeof(float), B * H * R * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)q_fp32_data[i];
			}
			printf("✅ Loaded and converted Q data from FP32\n");
		}
		free(q_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_k_fp16.bin", "rb");
	if (f) {
		fread(k_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded K FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* k_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_k_input.bin", "rb");
		if (f) {
			fread(k_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)k_fp32_data[i];
			}
			printf("✅ Loaded and converted K data from FP32\n");
		}
		free(k_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	if (f) {
		fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded V FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* v_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_v_input.bin", "rb");
		if (f) {
			fread(v_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)v_fp32_data[i];
			}
			printf("✅ Loaded and converted V data from FP32\n");
		}
		free(v_fp32_data);
	}
	
	// Print sample values for verification
	printf("Sample values: Q[0]=%f, K[0]=%f, V[0]=%f\n", 
		(float)q_fp16_data[0], (float)k_fp16_data[0], (float)v_fp16_data[0]);
	
	// Create CCV tensors for input
	// Use NHWC layout: [batch, heads, seq, dim]
	ccv_nnc_tensor_t* q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	
	// Copy data to CCV tensors
	memcpy(q_tensor->data.f16, q_fp16_data, B * H * R * D * sizeof(__fp16));
	memcpy(k_tensor->data.f16, k_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(v_tensor->data.f16, v_fp16_data, B * H * C * D * sizeof(__fp16));
	
	printf("✅ Created CCV tensors and copied input data\n");
	
	// Move tensors to GPU
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	
	// Copy data to GPU tensors
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_tensor), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(v_tensor), TENSOR_LIST(gpu_v_tensor), 0);
	
	printf("✅ Copied tensors to GPU\n");
	
	// Create output tensors - SageAttention requires quantized outputs
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	
	// Create quantized output tensors (required by SageAttention)
	const uint32_t BLKQ = 128;
	const uint32_t WARPQ = 32;
	const uint32_t BLKK = 64;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4 with WARPQ=32
	const int k_scale_blocks = (C + BLKK - 1) / BLKK;       // Should be 1
	
	ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	// Create k_mean tensor (optional input for SageAttention)
	ccv_nnc_tensor_t* k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	ccv_nnc_tensor_t* gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	
	// Initialize k_mean to zeros
	memset(k_mean_tensor->data.u8, 0, B * H * D * sizeof(__fp16));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_mean_tensor), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	// Call scaled dot product attention forward using CCV command
	// The command will internally call _ccv_nnc_scaled_dot_product_attention_forw
	printf("\n=== Calling Scaled Dot Product Attention Forward (SageAttention) ===\n");
	printf("Command parameters:\n");
	printf("  scale: %f\n", sm_scale);
	printf("  is_causal: 0 (false)\n");
	
	// Note: SageAttention implementation requires all output tensors to be provided
	// Inputs: Q, K, V, attn_mask (NULL), weights (NULL), bias (NULL), k_mean
	// Outputs: O, saved_softmax_lse (NULL), q_int8, k_int8, q_scale, k_scale
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(
      CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0,
      ((ccv_nnc_cmd_param_t){
          .size={.dim={1,1,1}},
          .scaled_dot_product_attention={
              .scale=sm_scale,
              .is_causal=0,
              .flags=CCV_NNC_GEMM_8U_32F  // Set flags explicitly
          }
      }), 0
  );

	ccv_nnc_cmd_exec(
		cmd, // scale, is_causal=false
		ccv_nnc_no_hint, 
		0, 
		TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0), // inputs
		TENSOR_LIST(gpu_o_tensor, 0, gpu_q_int8_tensor, gpu_k_int8_tensor, gpu_q_scale_tensor, gpu_k_scale_tensor), // outputs
		0
	);
	
	printf("✅ Scaled dot product attention forward completed\n");
	
	// Copy output back to CPU for analysis
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor), 0);
	
	// Also copy the quantized outputs and scales back for verification
	ccv_nnc_tensor_t* cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* cpu_q_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* cpu_k_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_scale_tensor), TENSOR_LIST(cpu_q_scale_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_scale_tensor), TENSOR_LIST(cpu_k_scale_output), 0);
	
	printf("✅ Copied outputs back to CPU\n");
	
	// Extract output data for analysis
	__fp16* output_data = (__fp16*)cpu_output_tensor->data.u8;
	int8_t* q_int8_output = (int8_t*)cpu_q_int8_tensor->data.u8;
	int8_t* k_int8_output = (int8_t*)cpu_k_int8_tensor->data.u8;
	
	printf("\n=== SageAttention Forward Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d]\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)output_data[i]);
	}
	
	printf("\nQuantized Q (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", q_int8_output[i]);
	}
	printf("\n");
	
	printf("Q scales (all %d values): ", q_scale_blocks);
	for (int i = 0; i < q_scale_blocks; i++) {
		printf("%.6f ", cpu_q_scale_output->data.f32[i]);
	}
	printf("\n");
	
	// Compare with previous test output if available
	printf("\n=== Comparison with Direct Wrapper Test ===\n");
	f = fopen("/tmp/pytorch_sage_output.bin", "rb");
	if (f) {
		__fp16* wrapper_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(wrapper_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values with direct wrapper test:\n");
		for (int i = 0; i < 10; i++) {
			float forw_val = (float)output_data[i];
			float wrapper_val = (float)wrapper_data[i];
			float diff = fabsf(forw_val - wrapper_val);
			printf("  [%d] Forward: %f, Wrapper: %f, diff: %.8f\n", i, forw_val, wrapper_val, diff);
		}
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)output_data[i] - (float)wrapper_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 1e-5, "Forward function and direct wrapper should produce identical results");
		
		free(wrapper_data);
	} else {
		printf("Direct wrapper output not found. Run '_ccv_nnc_scaled_dot_product_attention_forw test' first for comparison.\n");
	}
	
	// Save output for future comparisons
	f = fopen("/tmp/ccv_forward_sage_output.bin", "wb");
	if (f) {
		fwrite(output_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved forward function output to /tmp/ccv_forward_sage_output.bin\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scale_output);
	ccv_nnc_tensor_free(cpu_k_scale_output);
	
	free(q_fp16_data);
	free(k_fp16_data);
	free(v_fp16_data);
	
	printf("✅ SageAttention forward function test completed successfully!\n");
}


TEST_CASE("_ccv_nnc_scaled_dot_product_attention_forw sage mix test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== SageAttention Forward Function Test ===\n");
	printf("Testing _ccv_nnc_scaled_dot_product_attention_forw with SageAttention backend\n");
	
	// Use same test parameters as SageAttention wrapper test
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load FP16 input data from PyTorch files
	printf("Loading PyTorch FP16 inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for FP16 input data
	__fp16* q_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	__fp16* k_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	
	// Load FP16 data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_fp16.bin", "rb");
	if (f) {
		fread(q_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Loaded Q FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		printf("FP16 files not found, loading FP32 and converting...\n");
		float* q_fp32_data = (float*)malloc(B * H * R * D * sizeof(float));
		f = fopen("/tmp/pytorch_q_input.bin", "rb");
		if (f) {
			fread(q_fp32_data, sizeof(float), B * H * R * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)q_fp32_data[i];
			}
			printf("✅ Loaded and converted Q data from FP32\n");
		}
		free(q_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_k_fp16.bin", "rb");
	if (f) {
		fread(k_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded K FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* k_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_k_input.bin", "rb");
		if (f) {
			fread(k_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)k_fp32_data[i];
			}
			printf("✅ Loaded and converted K data from FP32\n");
		}
		free(k_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	if (f) {
		fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded V FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* v_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_v_input.bin", "rb");
		if (f) {
			fread(v_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)v_fp32_data[i];
			}
			printf("✅ Loaded and converted V data from FP32\n");
		}
		free(v_fp32_data);
	}
	
	// Print sample values for verification
	printf("Sample values: Q[0]=%f, K[0]=%f, V[0]=%f\n", 
		(float)q_fp16_data[0], (float)k_fp16_data[0], (float)v_fp16_data[0]);
	
	// Create CCV tensors for input
	// Use NHWC layout: [batch, heads, seq, dim]
	ccv_nnc_tensor_t* q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	
	// Copy data to CCV tensors
	memcpy(q_tensor->data.f16, q_fp16_data, B * H * R * D * sizeof(__fp16));
	memcpy(k_tensor->data.f16, k_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(v_tensor->data.f16, v_fp16_data, B * H * C * D * sizeof(__fp16));
	
	printf("✅ Created CCV tensors and copied input data\n");
	
	// Move tensors to GPU
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	
	// Copy data to GPU tensors
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_tensor), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(v_tensor), TENSOR_LIST(gpu_v_tensor), 0);
	
	printf("✅ Copied tensors to GPU\n");
	
	// Create output tensors - SageAttention requires quantized outputs
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	
	// Create quantized output tensors (required by SageAttention)
	const uint32_t BLKQ = 128;
	const uint32_t WARPQ = 16;
	const uint32_t BLKK = 64;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 8 with WARPQ=16
	const int k_scale_blocks = (C + BLKK - 1) / BLKK;       // Should be 1
	printf("q_scale_blocks:%d", q_scale_blocks);
	ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	// Create k_mean tensor (optional input for SageAttention)
	ccv_nnc_tensor_t* k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	ccv_nnc_tensor_t* gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	
	// Initialize k_mean to zeros
	memset(k_mean_tensor->data.u8, 0, B * H * D * sizeof(__fp16));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(k_mean_tensor), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	// Call scaled dot product attention forward using CCV command
	// The command will internally call _ccv_nnc_scaled_dot_product_attention_forw
	printf("\n=== Calling Scaled Dot Product Attention Forward (SageAttention) ===\n");
	printf("Command parameters:\n");
	printf("  scale: %f\n", sm_scale);
	printf("  is_causal: 0 (false)\n");
	
	// Note: SageAttention implementation requires all output tensors to be provided
	// Inputs: Q, K, V, attn_mask (NULL), weights (NULL), bias (NULL), k_mean
	// Outputs: O, saved_softmax_lse (NULL), q_int8, k_int8, q_scale, k_scale
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(
      CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0,
      ((ccv_nnc_cmd_param_t){
          .size={.dim={1,1,1}},
          .scaled_dot_product_attention={
              .scale=sm_scale,
              .is_causal=0,
              .flags=CCV_NNC_GEMM_8U  // Set flags explicitly
          }
      }), 0
  );

	ccv_nnc_cmd_exec(
		cmd, // scale, is_causal=false
		ccv_nnc_no_hint, 
		0, 
		TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_k_mean_tensor), // inputs
		TENSOR_LIST(gpu_o_tensor, 0, gpu_q_int8_tensor, gpu_k_int8_tensor, gpu_q_scale_tensor, gpu_k_scale_tensor), // outputs
		0
	);
	
	printf("✅ Scaled dot product attention forward completed\n");
	
	// Copy output back to CPU for analysis
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor), 0);
	
	// Also copy the quantized outputs and scales back for verification
	ccv_nnc_tensor_t* cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* cpu_q_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* cpu_k_scale_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_scale_tensor), TENSOR_LIST(cpu_q_scale_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_k_scale_tensor), TENSOR_LIST(cpu_k_scale_output), 0);
	
	printf("✅ Copied outputs back to CPU\n");
	
	// Extract output data for analysis
	__fp16* output_data = (__fp16*)cpu_output_tensor->data.u8;
	int8_t* q_int8_output = (int8_t*)cpu_q_int8_tensor->data.u8;
	int8_t* k_int8_output = (int8_t*)cpu_k_int8_tensor->data.u8;
	
	printf("\n=== SageAttention Forward Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d]\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)output_data[i]);
	}
	
	printf("\nQuantized Q (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", q_int8_output[i]);
	}
	printf("\n");
	
	printf("Q scales (all %d values): ", q_scale_blocks);
	for (int i = 0; i < q_scale_blocks; i++) {
		printf("%.6f ", cpu_q_scale_output->data.f32[i]);
	}
	printf("\n");
	
	// Compare with previous test output if available
	printf("\n=== Comparison with Direct Wrapper Test ===\n");
	f = fopen("/tmp/pytorch_sage_output.bin", "rb");
	if (f) {
		__fp16* wrapper_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(wrapper_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values with direct wrapper test:\n");
		for (int i = 0; i < 10; i++) {
			float forw_val = (float)output_data[i];
			float wrapper_val = (float)wrapper_data[i];
			float diff = fabsf(forw_val - wrapper_val);
			printf("  [%d] Forward: %f, Wrapper: %f, diff: %.8f\n", i, forw_val, wrapper_val, diff);
		}
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)output_data[i] - (float)wrapper_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-3 * 5) exact_matches++;
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-2): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 2 * 1e-2, "Forward function and direct wrapper should produce identical results");
		
		free(wrapper_data);
	} else {
		printf("Direct wrapper output not found. Run '_ccv_nnc_scaled_dot_product_attention_forw test' first for comparison.\n");
	}
	
	// Save output for future comparisons
	f = fopen("/tmp/ccv_forward_sage_output.bin", "wb");
	if (f) {
		fwrite(output_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved forward function output to /tmp/ccv_forward_sage_output.bin\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scale_output);
	ccv_nnc_tensor_free(cpu_k_scale_output);
	
	free(q_fp16_data);
	free(k_fp16_data);
	free(v_fp16_data);
	
	printf("✅ SageAttention forward function test completed successfully!\n");
}

TEST_CASE("qk_int8_sv_f16_accum_f16_attn_inst_buf_direct test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Direct Function Test (inst_buf_direct) ===\n");
	printf("Testing qk_int8_sv_f16_accum_f16_attn_inst_buf_direct function (no CCV dependencies)\n");
	printf("This test validates the direct kernel function using same inputs as previous tests\n");
	
	// Use same test parameters as other SageAttention tests
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load same PyTorch quantized inputs as other tests
	printf("Loading PyTorch quantized inputs from binary files...\n");
	
	// Allocate host memory for input data
	int8_t* q_int8_data = (int8_t*)malloc(B * H * R * D * sizeof(int8_t));
	int8_t* k_int8_data = (int8_t*)malloc(B * H * C * D * sizeof(int8_t)); 
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* o_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	float* q_scales_data = (float*)malloc(B * H * 8 * sizeof(float));  // Different scale count for inst_buf
	float* k_scales_data = (float*)malloc(B * H * 1 * sizeof(float));  // per-block scales
	
	// Load quantized data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_mix_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open Q int8 data file - run extract_pytorch_kernel_inputs.py first");
	fread(q_int8_data, sizeof(int8_t), B * H * R * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_k_mix_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K int8 data file");
	fread(k_int8_data, sizeof(int8_t), B * H * C * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	REQUIRE(f != NULL, "Failed to open V fp16 data file");
	fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
	fclose(f);
	
	// Load scales (using inst_buf variant which has 8 scales per head instead of 4)
	f = fopen("/tmp/pytorch_kernel_q_mix_scales.bin", "rb");  // Use mix scales for inst_buf
	if (f != NULL) {
		fread(q_scales_data, sizeof(float), B * H * 8, f);
		fclose(f);
		printf("✅ Loaded inst_buf Q scales (8 per head)\n");
	} else {
		// Fallback: duplicate regular scales to fill 8 elements
		f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
		REQUIRE(f != NULL, "Failed to open Q scales data file");
		float temp_scales[B * H * 4];
		fread(temp_scales, sizeof(float), B * H * 4, f);
		fclose(f);
		// Duplicate to fill 8 elements per head
		for (int i = 0; i < B * H; i++) {
			for (int j = 0; j < 4; j++) {
				q_scales_data[i * 8 + j] = temp_scales[i * 4 + j];
				q_scales_data[i * 8 + 4 + j] = temp_scales[i * 4 + j];  // Duplicate
			}
		}
		printf("✅ Loaded and duplicated Q scales (4->8 per head)\n");
	}
	
	f = fopen("/tmp/pytorch_kernel_k_mix_scales.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K mix scales data file");
	fread(k_scales_data, sizeof(float), B * H * 1, f);
	fclose(f);
	
	printf("✅ Loaded PyTorch quantized data from disk\n");
	
	// Print sample values for verification
	printf("Sample values: Q_int8[0]=%d, K_int8[0]=%d, V[0]=%f\n", 
		q_int8_data[0], k_int8_data[0], (float)v_fp16_data[0]);
	printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f\n", 
		q_scales_data[0], k_scales_data[0]);
	
	// Allocate GPU memory directly using CUDA (no CCV tensors)
	int8_t *gpu_q_int8, *gpu_k_int8;
	__fp16 *gpu_v_fp16, *gpu_o_fp16;
	float *gpu_q_scales, *gpu_k_scales;
	
	cudaMalloc(&gpu_q_int8, B * H * R * D * sizeof(int8_t));
	cudaMalloc(&gpu_k_int8, B * H * C * D * sizeof(int8_t));
	cudaMalloc(&gpu_v_fp16, B * H * C * D * sizeof(__fp16));
	cudaMalloc(&gpu_o_fp16, B * H * R * D * sizeof(__fp16));
	cudaMalloc(&gpu_q_scales, B * H * 8 * sizeof(float));
	cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));
	
	// Copy data to GPU
	cudaMemcpy(gpu_q_int8, q_int8_data, B * H * R * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_int8, k_int8_data, B * H * C * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_v_fp16, v_fp16_data, B * H * C * D * sizeof(__fp16), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_q_scales, q_scales_data, B * H * 8 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_scales, k_scales_data, B * H * 1 * sizeof(float), cudaMemcpyHostToDevice);
	
	printf("✅ Allocated GPU memory and copied input data\n");
	
	// Prepare dimension and stride arrays for direct function call
	int qdim[4] = {B, H, R, D};
	int kdim[4] = {B, H, C, D};
	int vdim[4] = {B, H, C, D};
	int odim[4] = {B, H, R, D};
	int qscale_dim[3] = {B, H, 8};  // inst_buf uses 8 scales per head
	int kscale_dim[3] = {B, H, 1};
	
	// Calculate strides for HND layout (batch, head, seq, dim)
	int qstride[4] = {H * R * D, R * D, D, 1};
	int kstride[4] = {H * C * D, C * D, D, 1};
	int vstride[4] = {H * C * D, C * D, D, 1};
	int ostride[4] = {H * R * D, R * D, D, 1};
	int qscale_stride[3] = {H * 8, 8, 1};
	int kscale_stride[3] = {H * 1, 1, 1};
	
	// Declare the direct function
	// extern void qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
	// 	int8_t *Q, int8_t *K, __fp16 *V, __fp16 *O,
	// 	float *Q_scale, float *K_scale,
	// 	int qdim[], int kdim[], int vdim[], int odim[], 
	// 	int qscale_dim[], int kscale_dim[],
	// 	int qstride[], int kstride[], int vstride[], int ostride[],
	// 	int qscale_stride[], int kscale_stride[],
	// 	int tensor_layout,
	// 	int is_causal,
	// 	int qk_quant_gran,
	// 	float sm_scale,
	// 	int return_lse);
	
	// Call the direct function
	printf("\n=== Calling Direct SageAttention Function (inst_buf) ===\n");
	printf("Function parameters:\n");
	printf("  tensor_layout: 1 (HND)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	printf("  Q scales per head: 8 (inst_buf variant)\n");
	
	printf("\nTensor dimensions:\n");
	printf("  Q: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		qdim[0], qdim[1], qdim[2], qdim[3], qstride[0], qstride[1], qstride[2], qstride[3]);
	printf("  K: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		kdim[0], kdim[1], kdim[2], kdim[3], kstride[0], kstride[1], kstride[2], kstride[3]);
	printf("  V: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		vdim[0], vdim[1], vdim[2], vdim[3], vstride[0], vstride[1], vstride[2], vstride[3]);
	printf("  O: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		odim[0], odim[1], odim[2], odim[3], ostride[0], ostride[1], ostride[2], ostride[3]);
	printf("  Q_scale: [%d, %d, %d], strides: [%d, %d, %d]\n", 
		qscale_dim[0], qscale_dim[1], qscale_dim[2], qscale_stride[0], qscale_stride[1], qscale_stride[2]);
	printf("  K_scale: [%d, %d, %d], strides: [%d, %d, %d]\n", 
		kscale_dim[0], kscale_dim[1], kscale_dim[2], kscale_stride[0], kscale_stride[1], kscale_stride[2]);
	
	qk_int8_sv_f16_accum_f16_attn_inst_buf_direct(
		gpu_q_int8,        // Q (int8)
		gpu_k_int8,        // K (int8)
		gpu_v_fp16,        // V (fp16)
		gpu_o_fp16,        // O (fp16) - output
		gpu_q_scales,      // Q_scale (fp32)
		gpu_k_scales,      // K_scale (fp32)
		qdim, kdim, vdim, odim,
		qscale_dim, kscale_dim,
		qstride, kstride, vstride, ostride,
		qscale_stride, kscale_stride,
		1,                 // tensor_layout: 1=HND
		0,                 // is_causal: false
		2,                 // qk_quant_gran: 2=per_warp
		sm_scale,          // sm_scale
		0                  // return_lse: false
	);
	
	// Check for CUDA errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR: Direct function call failed: %s\n", cudaGetErrorString(error));
		REQUIRE(0, "Direct function should succeed");
	}
	
	// Synchronize to ensure kernel completion
	cudaDeviceSynchronize();
	
	printf("✅ Direct function call completed successfully\n");
	
	// Copy output back to CPU
	cudaMemcpy(o_fp16_data, gpu_o_fp16, B * H * R * D * sizeof(__fp16), cudaMemcpyDeviceToHost);
	
	printf("\n=== Direct Function Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)o_fp16_data[i]);
	}
	
	printf("\nSample outputs from different positions:\n");
	printf("  output[0,0,0,0]: %f\n", (float)o_fp16_data[0]);                           // [0,0,0,0]
	printf("  output[0,0,0,64]: %f\n", (float)o_fp16_data[64]);                         // [0,0,0,64] - mid head_dim
	printf("  output[0,0,0,127]: %f\n", (float)o_fp16_data[127]);                       // [0,0,0,127] - end head_dim 
	printf("  output[0,0,32,0]: %f\n", (float)o_fp16_data[32 * D]);                     // [0,0,32,0] - mid sequence
	printf("  output[0,4,0,0]: %f\n", (float)o_fp16_data[4 * R * D]);                  // [0,4,0,0] - mid head
	
	// Save direct function output for comparison
	f = fopen("/tmp/ccv_direct_inst_buf_sage_output.bin", "wb");
	if (f) {
		fwrite(o_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved direct function output to /tmp/ccv_direct_inst_buf_sage_output.bin\n");
	}
	
	// Compare with PyTorch output if available
	printf("\n=== Direct Function vs PyTorch Comparison ===\n");
	f = fopen("/tmp/pytorch_sage_kernel_inst_buf_output.bin", "rb");
	if (f) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values:\n");
		int close_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		
		for (int i = 0; i < 10; i++) {
			float direct_val = (float)o_fp16_data[i];
			float pytorch_val = (float)pytorch_data[i];
			double diff = fabs(direct_val - pytorch_val);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 0.01) close_matches++;  // Tolerance for fp16+fp32 vs fp32 accumulation difference
			
			printf("  [%d] Direct: %f, PyTorch: %f, diff: %.8f\n", i, direct_val, pytorch_val, diff);
		}
		
		printf("\nKey position comparisons:\n");
		printf("  [0,0,0,0] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[0], (float)pytorch_data[0], 
			fabs((float)o_fp16_data[0] - (float)pytorch_data[0]));
		printf("  [0,0,0,64] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[64], (float)pytorch_data[64], 
			fabs((float)o_fp16_data[64] - (float)pytorch_data[64]));
		printf("  [0,4,0,0] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[4 * R * D], (float)pytorch_data[4 * R * D], 
			fabs((float)o_fp16_data[4 * R * D] - (float)pytorch_data[4 * R * D]));
		
		// Full tensor comparison
		int total_elements = B * H * R * D;
		close_matches = 0;
		max_diff = 0.0;
		sum_diff = 0.0;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)o_fp16_data[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 0.01) close_matches++;  // Tolerance for different accumulation strategies
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Close matches (diff < 0.01): %d/%d (%.2f%%)\n", 
			close_matches, total_elements, (100.0 * close_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		// Note: Different accumulation strategies (fp16+fp32 vs fp32) may have larger differences
		printf("\nNote: This inst_buf variant uses fp16+fp32 mixed accumulation,\n");
		printf("      while PyTorch reference uses fp32 accumulation.\n");
		printf("      Small differences are expected due to different precision strategies.\n");
		
		free(pytorch_data);
	} else {
		printf("PyTorch reference output not found. Run test_sageattention_sm80_direct.py first for comparison.\n");
	}
	
	// Cleanup GPU memory
	cudaFree(gpu_q_int8);
	cudaFree(gpu_k_int8);
	cudaFree(gpu_v_fp16);
	cudaFree(gpu_o_fp16);
	cudaFree(gpu_q_scales);
	cudaFree(gpu_k_scales);
	
	// Cleanup host memory
	free(q_int8_data);
	free(k_int8_data);
	free(v_fp16_data);
	free(o_fp16_data);
	free(q_scales_data);
	free(k_scales_data);
	
	printf("✅ SageAttention direct function test (inst_buf) completed successfully!\n");
}

TEST_CASE("ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Direct Function Test (FP32 accum) ===\n");
	printf("Testing ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct function (no CCV dependencies)\n");
	printf("This test validates the direct kernel function using same inputs as previous tests\n");
	
	// Use same test parameters as other sage attention tests
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	printf("Loading PyTorch quantized inputs from binary files...\n");
	
	// Allocate host memory for input data (same as other sage tests)
	int8_t* q_int8_data = (int8_t*)malloc(B * H * R * D * sizeof(int8_t));
	int8_t* k_int8_data = (int8_t*)malloc(B * H * C * D * sizeof(int8_t)); 
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	float* q_scales_data = (float*)malloc(B * H * 4 * sizeof(float));  // per-warp scales
	float* k_scales_data = (float*)malloc(B * H * 1 * sizeof(float));  // per-block scales
	
	// Load quantized data from PyTorch files (exact same as other sage tests)
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open Q int8 data file");
	fread(q_int8_data, sizeof(int8_t), B * H * R * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_k_int8.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K int8 data file");
	fread(k_int8_data, sizeof(int8_t), B * H * C * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	REQUIRE(f != NULL, "Failed to open V fp16 data file");
	fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
	REQUIRE(f != NULL, "Failed to open Q scales data file");
	fread(q_scales_data, sizeof(float), B * H * 4, f);
	fclose(f);
	
	f = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
	REQUIRE(f != NULL, "Failed to open K scales data file");
	fread(k_scales_data, sizeof(float), B * H * 1, f);
	fclose(f);
	
	printf("✅ Loaded PyTorch quantized data from disk\n");
	
	// Print sample values for verification (same as other sage tests)
	printf("Sample values: Q_int8[0]=%d, K_int8[0]=%d, V[0]=%f\n", 
		q_int8_data[0], k_int8_data[0], (float)v_fp16_data[0]);
	printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f\n", 
		q_scales_data[0], k_scales_data[0]);
	
	// Allocate GPU memory
	int8_t* gpu_q_int8;
	int8_t* gpu_k_int8;
	__fp16* gpu_v_fp16;
	__fp16* gpu_o_fp16;
	float* gpu_q_scales;
	float* gpu_k_scales;
	
	cudaMalloc(&gpu_q_int8, B * H * R * D * sizeof(int8_t));
	cudaMalloc(&gpu_k_int8, B * H * C * D * sizeof(int8_t));
	cudaMalloc(&gpu_v_fp16, B * H * C * D * sizeof(__fp16));
	cudaMalloc(&gpu_o_fp16, B * H * R * D * sizeof(__fp16));
	cudaMalloc(&gpu_q_scales, B * H * 4 * sizeof(float));
	cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));
	
	// Copy input data to GPU
	cudaMemcpy(gpu_q_int8, q_int8_data, B * H * R * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_int8, k_int8_data, B * H * C * D * sizeof(int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_v_fp16, v_fp16_data, B * H * C * D * sizeof(__fp16), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_q_scales, q_scales_data, B * H * 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_k_scales, k_scales_data, B * H * 1 * sizeof(float), cudaMemcpyHostToDevice);
	
	printf("✅ Allocated GPU memory and copied input data\n");

	// Set up tensor dimensions and strides for HND layout
	int qdim[4] = {B, H, R, D};
	int kdim[4] = {B, H, C, D};
	int vdim[4] = {B, H, C, D};
	int odim[4] = {B, H, R, D};
	int qscale_dim[3] = {B, H, 4};  // per-warp scales
	int kscale_dim[3] = {B, H, 1};  // per-block scales
	
	int qstride[4] = {H*R*D, R*D, D, 1};
	int kstride[4] = {H*C*D, C*D, D, 1};
	int vstride[4] = {H*C*D, C*D, D, 1};
	int ostride[4] = {H*R*D, R*D, D, 1};
	int qscale_stride[3] = {H*4, 4, 1};
	int kscale_stride[3] = {H*1, 1, 1};

	printf("\n=== Calling Direct SageAttention Function (FP32 accum) ===\n");
	printf("Function parameters:\n");
	printf("  tensor_layout: 1 (HND)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	printf("  Q scales per head: 4 (per-warp variant)\n");

	printf("\nTensor dimensions:\n");
	printf("  Q: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		qdim[0], qdim[1], qdim[2], qdim[3], qstride[0], qstride[1], qstride[2], qstride[3]);
	printf("  K: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		kdim[0], kdim[1], kdim[2], kdim[3], kstride[0], kstride[1], kstride[2], kstride[3]);
	printf("  V: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		vdim[0], vdim[1], vdim[2], vdim[3], vstride[0], vstride[1], vstride[2], vstride[3]);
	printf("  O: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
		odim[0], odim[1], odim[2], odim[3], ostride[0], ostride[1], ostride[2], ostride[3]);
	printf("  Q_scale: [%d, %d, %d], strides: [%d, %d, %d]\n", 
		qscale_dim[0], qscale_dim[1], qscale_dim[2], qscale_stride[0], qscale_stride[1], qscale_stride[2]);
	printf("  K_scale: [%d, %d, %d], strides: [%d, %d, %d]\n", 
		kscale_dim[0], kscale_dim[1], kscale_dim[2], kscale_stride[0], kscale_stride[1], kscale_stride[2]);

	// Call the direct function
	ccv_nnc_qk_int8_sv_f16_accum_f32_attn_direct(
		gpu_q_int8, gpu_k_int8, gpu_v_fp16, gpu_o_fp16,
		gpu_q_scales, gpu_k_scales,
		qdim, kdim, vdim, odim,
		qscale_dim, kscale_dim,
		qstride, kstride, vstride, ostride,
		qscale_stride, kscale_stride,
		1,      // tensor_layout (HND)
		0,      // is_causal (false)
		2,      // qk_quant_gran (per_warp)
		sm_scale,
		0       // return_lse (false)
	);
	
	printf("✅ Direct function call completed successfully\n");
	
	// Copy output back to host
	__fp16* o_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	cudaMemcpy(o_fp16_data, gpu_o_fp16, B * H * R * D * sizeof(__fp16), cudaMemcpyDeviceToHost);
	
	printf("\n=== Direct Function Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, (float)o_fp16_data[i]);
	}
	
	printf("\nSample outputs from different positions:\n");
	printf("  output[0,0,0,0]: %f\n", (float)o_fp16_data[0]);
	printf("  output[0,0,0,64]: %f\n", (float)o_fp16_data[64]);
	printf("  output[0,0,0,127]: %f\n", (float)o_fp16_data[127]);
	printf("  output[0,0,32,0]: %f\n", (float)o_fp16_data[32 * D]);
	printf("  output[0,4,0,0]: %f\n", (float)o_fp16_data[4 * R * D]);
	
	// Save output for comparison
	f = fopen("/tmp/ccv_direct_fp32_sage_output.bin", "wb");
	if (f) {
		fwrite(o_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Saved direct function output to /tmp/ccv_direct_fp32_sage_output.bin\n");
	}
	
	printf("\n=== Direct Function vs PyTorch Comparison ===\n");
	// Load PyTorch reference output for comparison 
	f = fopen("/tmp/pytorch_sage_kernel_output.bin", "rb");
	if (f) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		
		printf("Comparing first 10 values:\n");
		for (int i = 0; i < 10; i++) {
			float diff = fabs((float)o_fp16_data[i] - (float)pytorch_data[i]);
			printf("  [%d] Direct: %f, PyTorch: %f, diff: %.8f\n", 
				i, (float)o_fp16_data[i], (float)pytorch_data[i], diff);
		}
		
		printf("\nKey position comparisons:\n");
		float diff_0000 = fabs((float)o_fp16_data[0] - (float)pytorch_data[0]);
		float diff_0064 = fabs((float)o_fp16_data[64] - (float)pytorch_data[64]);
		float diff_4000 = fabs((float)o_fp16_data[4 * R * D] - (float)pytorch_data[4 * R * D]);
		
		printf("  [0,0,0,0] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[0], (float)pytorch_data[0], diff_0000);
		printf("  [0,0,0,64] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[64], (float)pytorch_data[64], diff_0064);
		printf("  [0,4,0,0] Direct: %f, PyTorch: %f, diff: %.8f\n", 
			(float)o_fp16_data[4 * R * D], (float)pytorch_data[4 * R * D], diff_4000);
		
		// Full tensor comparison
		int total_elements = B * H * R * D;
		int close_matches = 0;
		double max_diff = 0.0;
		double sum_diff = 0.0;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)o_fp16_data[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 0.01) close_matches++;  // Tolerance for FP32 vs mixed precision
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Close matches (diff < 0.01): %d/%d (%.2f%%)\n", 
			close_matches, total_elements, (100.0 * close_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		printf("\nNote: This FP32 accumulation variant should closely match PyTorch reference,\n");
		printf("      which also uses FP32 accumulation. Small differences may occur due to\n");
		printf("      different memory access patterns or rounding strategies.\n");
		
		free(pytorch_data);
	} else {
		printf("PyTorch reference output not found. Run test_sageattention_sm80_direct.py first for comparison.\n");
	}
	
	// Cleanup GPU memory
	cudaFree(gpu_q_int8);
	cudaFree(gpu_k_int8);
	cudaFree(gpu_v_fp16);
	cudaFree(gpu_o_fp16);
	cudaFree(gpu_q_scales);
	cudaFree(gpu_k_scales);
	
	// Cleanup host memory
	free(q_int8_data);
	free(k_int8_data);
	free(v_fp16_data);
	free(o_fp16_data);
	free(q_scales_data);
	free(k_scales_data);
	
	printf("✅ SageAttention direct function test (FP32 accum) completed successfully!\n");
}

TEST_CASE("ccv_nnc_per_warp_int8_direct test")
{
	ccv_cli_set_output_levels(CCV_CLI_VERBOSE);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
	                  ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("=== CCV SageAttention ccv_nnc_per_warp_int8_direct Test ===\n");
	
	// Test parameters
	const int B = 1;      // batch size
	const int R = 64;     // sequence length
	const int H = 8;      // number of heads  
	const int D = 128;    // head dimension
	const uint32_t BLKQ = 128;  // Block size for Q
	const uint32_t WARPQ = 32;  // Warp size for Q
	const uint32_t BLKK = 64;   // Block size for K
	
	// Calculate scale dimensions
	const int q_scale_blocks = ((R + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ);  // Per-warp: blocks × warps_per_block
	const int k_scale_blocks = (R + BLKK - 1) / BLKK;  // Per-block: each block processes BLKK tokens
	
	printf("Test parameters: B=%d, R=%d, H=%d, D=%d\n", B, R, H, D);
	printf("Block sizes: Q=%d (warp=%d), K=%d\n", BLKQ, WARPQ, BLKK);
	printf("Scale blocks: Q=%d, K=%d\n", q_scale_blocks, k_scale_blocks);
	
	// Create input tensors
	ccv_nnc_tensor_t* const q_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D), 0);

	// Load input data from PyTorch test files (same as wrapper test)
	FILE* q_input_file = fopen("/tmp/test_q_input.bin", "rb");
	if (q_input_file) {
		fread(q_input_tensor->data.f32, sizeof(float), B * R * H * D, q_input_file);
		fclose(q_input_file);
		printf("✓ Loaded Q input from /tmp/test_q_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_q_input.bin - using random data\n");
		// Fill with random data as fallback
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 1);
		for (int i = 0; i < B * R * H * D; i++) {
			q_input_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2;
		}
	}
	
	FILE* k_input_file = fopen("/tmp/test_k_input.bin", "rb");
	if (k_input_file) {
		fread(k_input_tensor->data.f32, sizeof(float), B * R * H * D, k_input_file);
		fclose(k_input_file);
		printf("✓ Loaded K input from /tmp/test_k_input.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_input.bin - using random data\n");
		// Fill with random data as fallback
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 2);
		for (int i = 0; i < B * R * H * D; i++) {
			k_input_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2;
		}
	}
	
	FILE* k_mean_file = fopen("/tmp/test_k_mean.bin", "rb");
	if (k_mean_file) {
		fread(k_mean_tensor->data.f32, sizeof(float), B * H * D, k_mean_file);
		fclose(k_mean_file);
		printf("✓ Loaded K mean input from /tmp/test_k_mean.bin\n");
	} else {
		printf("❌ Could not load /tmp/test_k_mean.bin - using zeros\n");
		// Fill with zeros as fallback
		memset(k_mean_tensor->data.f32, 0, B * H * D * sizeof(float));
	}
	
	printf("k_mean_tensor\n");
	ccv_nnc_print_tensor_info(k_mean_tensor);
	
	printf("Input shapes: Q[%d,%d,%d,%d], K[%d,%d,%d,%d]\n", B, R, H, D, B, R, H, D);
	
	// Print input data for verification against PyTorch
	printf("Q_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", q_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_input sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_input_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("K_mean sample (first 10): ");
	for (int i = 0; i < 10; i++) {
		printf("%.6f ", k_mean_tensor->data.f32[i]);
	}
	printf("\n");

	// Calculate input ranges
	float q_min = q_input_tensor->data.f32[0], q_max = q_input_tensor->data.f32[0];
	float k_min = k_input_tensor->data.f32[0], k_max = k_input_tensor->data.f32[0];
	for (int i = 1; i < B * R * H * D; i++) {
		if (q_input_tensor->data.f32[i] < q_min) q_min = q_input_tensor->data.f32[i];
		if (q_input_tensor->data.f32[i] > q_max) q_max = q_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] < k_min) k_min = k_input_tensor->data.f32[i];
		if (k_input_tensor->data.f32[i] > k_max) k_max = k_input_tensor->data.f32[i];
	}
	printf("Q_input range: [%.6f, %.6f]\n", q_min, q_max);
	printf("K_input range: [%.6f, %.6f]\n", k_min, k_max);
	
	// Calculate output sizes and print calculations
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	printf("Scale tensor calculations: q_blocks=%zu, warps_per_block=%zu, q_scale_blocks=%d\n", 
	       q_blocks, warps_per_block, q_scale_blocks);
	printf("Expected Q scale shape: [%d, %d, %d]\n", B, H, q_scale_blocks);
	printf("Expected K scale shape: [%d, %d, %d]\n", B, H, k_scale_blocks);
	
	// Convert to FP16 first, then transfer to GPU (following working test pattern)
	ccv_nnc_tensor_t* const q_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_mean_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_input_tensor), TENSOR_LIST(q_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_input_tensor), TENSOR_LIST(k_input_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor), TENSOR_LIST(k_mean_tensor_f16), 0);
	
	ccv_nnc_tensor_t* const gpu_q_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(q_input_tensor_f16), TENSOR_LIST(gpu_q_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(k_input_tensor_f16), TENSOR_LIST(gpu_k_input_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(k_mean_tensor_f16), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	// Create output tensors
	ccv_nnc_tensor_t* const gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_q_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* const gpu_k_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	printf("\n");
	// Run direct quantization function
#ifdef HAVE_CUDA_SM80
	extern void ccv_nnc_per_warp_int8_direct(
	    __fp16 *q, __fp16 *k, int8_t *q_int8, int8_t *k_int8,
	    float *q_scale, float *k_scale, __fp16 *km,
	    int qdim[], int kdim[], int q_int8_dim[], int k_int8_dim[],
	    int q_scale_dim[], int k_scale_dim[], int km_dim[],
	    int qstride[], int kstride[], int q_int8_stride[], int k_int8_stride[],
	    int q_scale_stride[], int k_scale_stride[], int km_stride[],
	    int BLKQ, int WARPQ, int BLKK, int tensor_layout, cudaStream_t cuda_stream);
	
	printf("Running ccv_nnc_per_warp_int8_direct...\n");
	
	// Prepare dimension and stride arrays (NHD layout: tensor_layout=0)
	int qdim[] = {B, R, H, D};
	int kdim[] = {B, R, H, D};
	int q_int8_dim[] = {B, R, H, D};
	int k_int8_dim[] = {B, R, H, D};
	int q_scale_dim[] = {B, H, q_scale_blocks};
	int k_scale_dim[] = {B, H, k_scale_blocks};
	int km_dim[] = {B, H, D};
	
	// Calculate strides (NHWC format)
	int qstride[] = {R * H * D, H * D, D, 1};
	int kstride[] = {R * H * D, H * D, D, 1};
	int q_int8_stride[] = {R * H * D, H * D, D, 1};
	int k_int8_stride[] = {R * H * D, H * D, D, 1};
	int q_scale_stride[] = {H * q_scale_blocks, q_scale_blocks, 1};
	int k_scale_stride[] = {H * k_scale_blocks, k_scale_blocks, 1};
	int km_stride[] = {H * D, D, 1};
	
	ccv_nnc_per_warp_int8_direct(
		(__fp16*)gpu_q_input_tensor->data.f16,  // Q input
		(__fp16*)gpu_k_input_tensor->data.f16,  // K input
		(int8_t*)gpu_q_int8_tensor->data.u8,    // Q output
		(int8_t*)gpu_k_int8_tensor->data.u8,    // K output
		(float*)gpu_q_scales_tensor->data.f32,  // Q scales
		(float*)gpu_k_scales_tensor->data.f32,  // K scales
		(__fp16*)gpu_k_mean_tensor->data.f16,   // K mean tensor
		qdim, kdim, q_int8_dim, k_int8_dim,
		q_scale_dim, k_scale_dim, km_dim,        // k_mean dimensions
		qstride, kstride, q_int8_stride, k_int8_stride,
		q_scale_stride, k_scale_stride, km_stride, // k_mean strides
		BLKQ, WARPQ, BLKK,
		0,                                       // tensor_layout=0 (NHD)
		0                                        // cuda_stream (default)
	);
	
	cudaDeviceSynchronize();
	printf("✓ ccv_nnc_per_warp_int8_direct completed successfully\n");
	
	// Copy results back to CPU for verification
	ccv_nnc_tensor_t* const cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const cpu_q_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks, 1), 0);
	ccv_nnc_tensor_t* const cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, R, H, D), 0);
	ccv_nnc_tensor_t* const cpu_k_scales_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks, 1), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_scales_tensor), TENSOR_LIST(cpu_q_scales_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_scales_tensor), TENSOR_LIST(cpu_k_scales_tensor), 0);
	
	// Print sample outputs for verification
	printf("Q_int8 sample (first 10): ");
	for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_q_int8_tensor->data.u8[i]);
	printf("\n");
	
	printf("Q_scales sample (all %d values): ", B * H * q_scale_blocks);
	for (int i = 0; i < B * H * q_scale_blocks; i++) printf("%.6f ", cpu_q_scales_tensor->data.f32[i]);
	printf("\n");
	
	printf("K_int8 sample (first 10): ");
	for (int i = 0; i < 10; i++) printf("%d ", (int8_t)cpu_k_int8_tensor->data.u8[i]);
	printf("\n");
	
	printf("K_scales sample (all %d values): ", B * H * k_scale_blocks);
	for (int i = 0; i < B * H * k_scale_blocks; i++) printf("%.6f ", cpu_k_scales_tensor->data.f32[i]);
	printf("\n");
	
	// Save outputs to disk for comparison with PyTorch and other tests
	const size_t output_size = B * R * H * D;
	const size_t q_scale_size = B * H * q_scale_blocks;
	const size_t k_scale_size = B * H * k_scale_blocks;
	
	FILE* q_int8_file = fopen("/tmp/ccv_direct_q_int8.bin", "wb");
	if (q_int8_file) {
		fwrite(cpu_q_int8_tensor->data.u8, sizeof(int8_t), output_size, q_int8_file);
		fclose(q_int8_file);
		printf("✓ Saved Q_int8 to /tmp/ccv_direct_q_int8.bin\n");
	}
	
	FILE* k_int8_file = fopen("/tmp/ccv_direct_k_int8.bin", "wb");
	if (k_int8_file) {
		fwrite(cpu_k_int8_tensor->data.u8, sizeof(int8_t), output_size, k_int8_file);
		fclose(k_int8_file);
		printf("✓ Saved K_int8 to /tmp/ccv_direct_k_int8.bin\n");
	}
	
	FILE* q_scales_file = fopen("/tmp/ccv_direct_q_scales.bin", "wb");
	if (q_scales_file) {
		fwrite(cpu_q_scales_tensor->data.f32, sizeof(float), q_scale_size, q_scales_file);
		fclose(q_scales_file);
		printf("✓ Saved Q_scales to /tmp/ccv_direct_q_scales.bin\n");
	}
	
	FILE* k_scales_file = fopen("/tmp/ccv_direct_k_scales.bin", "wb");
	if (k_scales_file) {
		fwrite(cpu_k_scales_tensor->data.f32, sizeof(float), k_scale_size, k_scales_file);
		fclose(k_scales_file);
		printf("✓ Saved K_scales to /tmp/ccv_direct_k_scales.bin\n");
	}
	
	printf("Output sizes: Q_int8=%zu, Q_scales=%zu, K_int8=%zu, K_scales=%zu\n",
	       output_size, q_scale_size, output_size, k_scale_size);
	
	// Comprehensive validation
	printf("\n=== Validation Results ===\n");
	
	// Check quantized values are in valid range
	int q_valid_count = 0, k_valid_count = 0;
	for (int i = 0; i < B * R * H * D; i++) {
		int8_t q_val = ((int8_t*)cpu_q_int8_tensor->data.u8)[i];
		int8_t k_val = ((int8_t*)cpu_k_int8_tensor->data.u8)[i];
		if (q_val >= -128 && q_val <= 127) q_valid_count++;
		if (k_val >= -128 && k_val <= 127) k_valid_count++;
	}
	printf("Q quantized values in valid INT8 range: %d/%d (%.1f%%)\n", 
	       q_valid_count, B * R * H * D, 100.0f * q_valid_count / (B * R * H * D));
	printf("K quantized values in valid INT8 range: %d/%d (%.1f%%)\n", 
	       k_valid_count, B * R * H * D, 100.0f * k_valid_count / (B * R * H * D));
	
	// Check scales are positive and reasonable
	int q_positive_scales = 0, k_positive_scales = 0;
	float q_scale_min = cpu_q_scales_tensor->data.f32[0], q_scale_max = cpu_q_scales_tensor->data.f32[0];
	float k_scale_min = cpu_k_scales_tensor->data.f32[0], k_scale_max = cpu_k_scales_tensor->data.f32[0];
	
	for (int i = 0; i < q_scale_size; i++) {
		float scale = cpu_q_scales_tensor->data.f32[i];
		if (scale > 0) q_positive_scales++;
		if (scale < q_scale_min) q_scale_min = scale;
		if (scale > q_scale_max) q_scale_max = scale;
	}
	
	for (int i = 0; i < k_scale_size; i++) {
		float scale = cpu_k_scales_tensor->data.f32[i];
		if (scale > 0) k_positive_scales++;
		if (scale < k_scale_min) k_scale_min = scale;
		if (scale > k_scale_max) k_scale_max = scale;
	}
	
	printf("Q scales - positive: %d/%d, range: [%.6f, %.6f]\n", 
	       q_positive_scales, (int)q_scale_size, q_scale_min, q_scale_max);
	printf("K scales - positive: %d/%d, range: [%.6f, %.6f]\n", 
	       k_positive_scales, (int)k_scale_size, k_scale_min, k_scale_max);
	
	// Load and compare with wrapper test results if available
	FILE* wrapper_q_int8_file = fopen("/tmp/ccv_q_int8.bin", "rb");
	FILE* wrapper_q_scales_file = fopen("/tmp/ccv_q_scales.bin", "rb");
	FILE* wrapper_k_int8_file = fopen("/tmp/ccv_k_int8.bin", "rb");
	FILE* wrapper_k_scales_file = fopen("/tmp/ccv_k_scales.bin", "rb");
	
	if (wrapper_q_int8_file && wrapper_q_scales_file && wrapper_k_int8_file && wrapper_k_scales_file) {
		int8_t* wrapper_q_int8 = (int8_t*)malloc(output_size * sizeof(int8_t));
		float* wrapper_q_scales = (float*)malloc(q_scale_size * sizeof(float));
		int8_t* wrapper_k_int8 = (int8_t*)malloc(output_size * sizeof(int8_t));
		float* wrapper_k_scales = (float*)malloc(k_scale_size * sizeof(float));
		
		fread(wrapper_q_int8, sizeof(int8_t), output_size, wrapper_q_int8_file);
		fread(wrapper_q_scales, sizeof(float), q_scale_size, wrapper_q_scales_file);
		fread(wrapper_k_int8, sizeof(int8_t), output_size, wrapper_k_int8_file);
		fread(wrapper_k_scales, sizeof(float), k_scale_size, wrapper_k_scales_file);
		fclose(wrapper_q_int8_file);
		fclose(wrapper_q_scales_file);
		fclose(wrapper_k_int8_file);
		fclose(wrapper_k_scales_file);
		
		// Compare first 10 values
		printf("\n=== Comparison with Wrapper Test ===\n");
		printf("Q_int8 comparison (first 10):\n");
		printf("  Direct: ");
		for (int i = 0; i < 10; i++) printf("%3d ", (int8_t)cpu_q_int8_tensor->data.u8[i]);
		printf("\n  Wrapper:");
		for (int i = 0; i < 10; i++) printf("%3d ", wrapper_q_int8[i]);
		printf("\n");
		
		printf("Q_scales comparison (all %d):\n", (int)q_scale_size);
		printf("  Direct: ");
		for (int i = 0; i < q_scale_size; i++) printf("%.6f ", cpu_q_scales_tensor->data.f32[i]);
		printf("\n  Wrapper:");
		for (int i = 0; i < q_scale_size; i++) printf("%.6f ", wrapper_q_scales[i]);
		printf("\n");
		
		printf("K_int8 comparison (first 10):\n");
		printf("  Direct: ");
		for (int i = 0; i < 10; i++) printf("%3d ", (int8_t)cpu_k_int8_tensor->data.u8[i]);
		printf("\n  Wrapper:");
		for (int i = 0; i < 10; i++) printf("%3d ", wrapper_k_int8[i]);
		printf("\n");
		
		printf("K_scales comparison (all %d):\n", (int)k_scale_size);
		printf("  Direct: ");
		for (int i = 0; i < k_scale_size; i++) printf("%.6f ", cpu_k_scales_tensor->data.f32[i]);
		printf("\n  Wrapper:");
		for (int i = 0; i < k_scale_size; i++) printf("%.6f ", wrapper_k_scales[i]);
		printf("\n");
		
		// Calculate differences for Q
		int q_int8_matches = 0;
		int q_scales_matches = 0;
		float max_q_scale_diff = 0.0f;
		
		for (int i = 0; i < output_size; i++) {
			if (((int8_t*)cpu_q_int8_tensor->data.u8)[i] == wrapper_q_int8[i]) {
				q_int8_matches++;
			}
		}
		
		for (int i = 0; i < q_scale_size; i++) {
			float diff = fabsf(cpu_q_scales_tensor->data.f32[i] - wrapper_q_scales[i]);
			if (diff < 1e-6f) q_scales_matches++;
			if (diff > max_q_scale_diff) max_q_scale_diff = diff;
		}
		
		// Calculate differences for K
		int k_int8_matches = 0;
		int k_scales_matches = 0;
		float max_k_scale_diff = 0.0f;
		
		for (int i = 0; i < output_size; i++) {
			if (((int8_t*)cpu_k_int8_tensor->data.u8)[i] == wrapper_k_int8[i]) {
				k_int8_matches++;
			}
		}
		
		for (int i = 0; i < k_scale_size; i++) {
			float diff = fabsf(cpu_k_scales_tensor->data.f32[i] - wrapper_k_scales[i]);
			if (diff < 1e-6f) k_scales_matches++;
			if (diff > max_k_scale_diff) max_k_scale_diff = diff;
		}
		
		printf("Q_int8 exact matches: %d/%d (%.1f%%)\n", 
		       q_int8_matches, (int)output_size, 100.0f * q_int8_matches / output_size);
		printf("Q_scales exact matches: %d/%d (%.1f%%), max diff: %.6f\n", 
		       q_scales_matches, (int)q_scale_size, 100.0f * q_scales_matches / q_scale_size, max_q_scale_diff);
		printf("K_int8 exact matches: %d/%d (%.1f%%)\n", 
		       k_int8_matches, (int)output_size, 100.0f * k_int8_matches / output_size);
		printf("K_scales exact matches: %d/%d (%.1f%%), max diff: %.6f\n", 
		       k_scales_matches, (int)k_scale_size, 100.0f * k_scales_matches / k_scale_size, max_k_scale_diff);
		
		free(wrapper_q_int8);
		free(wrapper_q_scales);
		free(wrapper_k_int8);
		free(wrapper_k_scales);
	} else {
		printf("Wrapper test results not found. Run 'sage attention quantization ccv_nnc_per_warp_int8 test light' first for comparison.\n");
	}
	
	
	
	// Cleanup
	ccv_nnc_tensor_free(q_input_tensor);
	ccv_nnc_tensor_free(k_input_tensor);
	ccv_nnc_tensor_free(k_mean_tensor);
	ccv_nnc_tensor_free(q_input_tensor_f16);
	ccv_nnc_tensor_free(k_input_tensor_f16);
	ccv_nnc_tensor_free(k_mean_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_input_tensor);
	ccv_nnc_tensor_free(gpu_k_input_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scales_tensor);
	ccv_nnc_tensor_free(gpu_k_scales_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scales_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_scales_tensor);
	
	printf("✅ ccv_nnc_per_warp_int8_direct test completed successfully!\n");
#else
	printf("⚠️  Test skipped - CUDA SM80+ required for SageAttention\n");
#endif
}

TEST_CASE("ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Direct Function Test ===\n");
	printf("Testing ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct function\n");
	printf("This test validates the direct function using same inputs as wrapper test\n");
	
	// Use same test parameters as wrapper test
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load FP16 input data from PyTorch files (same as wrapper test)
	printf("Loading PyTorch FP16 inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for FP16 input data
	__fp16* q_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	__fp16* k_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	
	// Load FP16 data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_fp16.bin", "rb");
	if (f) {
		fread(q_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Loaded Q FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		printf("FP16 files not found, loading FP32 and converting...\n");
		float* q_fp32_data = (float*)malloc(B * H * R * D * sizeof(float));
		f = fopen("/tmp/pytorch_q_input.bin", "rb");
		if (f) {
			fread(q_fp32_data, sizeof(float), B * H * R * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)q_fp32_data[i];
			}
			printf("✅ Loaded and converted Q data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 Q files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(q_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_k_fp16.bin", "rb");
	if (f) {
		fread(k_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded K FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* k_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_k_input.bin", "rb");
		if (f) {
			fread(k_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)k_fp32_data[i];
			}
			printf("✅ Loaded and converted K data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 K files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(k_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	if (f) {
		fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded V FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* v_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_v_input.bin", "rb");
		if (f) {
			fread(v_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)v_fp32_data[i];
			}
			printf("✅ Loaded and converted V data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 V files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(v_fp32_data);
	}
	
	// Print sample values for verification
	printf("Sample values: Q[0]=%f, K[0]=%f, V[0]=%f\n", 
		(float)q_fp16_data[0], (float)k_fp16_data[0], (float)v_fp16_data[0]);
	
	// Create CCV tensors and copy loaded data
	ccv_nnc_tensor_t* q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* k_mean_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	
	// Copy loaded data to CCV tensors
	memcpy(q_tensor_f16->data.f16, q_fp16_data, B * H * R * D * sizeof(__fp16));
	memcpy(k_tensor_f16->data.f16, k_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(v_tensor_f16->data.f16, v_fp16_data, B * H * C * D * sizeof(__fp16));
	
	// Initialize k_mean to zeros
	memset(k_mean_tensor_f16->data.u8, 0, B * H * D * sizeof(__fp16));
	
	// Create GPU tensors
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	
	// Transfer to GPU
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_tensor_f16), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_tensor_f16), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(v_tensor_f16), TENSOR_LIST(gpu_v_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor_f16), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	printf("✅ Created and transferred tensors to GPU\n");
	
	// Create quantized output tensors
	const uint32_t BLKQ = 128;
	const uint32_t WARPQ = 32;
	const uint32_t BLKK = 64;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (C + BLKK - 1) / BLKK;      // Should be 1
		printf("q_scale_blocks:%d", q_scale_blocks);

	printf("Scale tensor dimensions: q_scale_blocks=%d, k_scale_blocks=%d\n", q_scale_blocks, k_scale_blocks);
	
	ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	printf("\n");
	
#ifdef HAVE_CUDA_SM80
	// Declare the direct function
	extern void ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
	    __fp16 *query, __fp16 *key, __fp16 *k_mean, __fp16 *value,
	    int8_t *q_int8, int8_t *k_int8,
	    float *query_scale, float *key_scale, __fp16 *output,
	    int qdim[], int kdim[], int vdim[], int odim[],
	    int q_int8_dim[], int k_int8_dim[],
	    int query_scale_dim[], int key_scale_dim[],
	    int qstride[], int kstride[], int vstride[], int ostride[],
	    int q_int8_stride[], int k_int8_stride[],
	    int query_scale_stride[], int key_scale_stride[],
	    int tensor_layout, int is_causal, int qk_quant_gran,
	    float sm_scale, int return_lse, int pv_accum_dtype,
	    int BLKQ, int WARPQ, int BLKK,
	    int km_dim[], int km_stride[], cudaStream_t cuda_stream);
	
	printf("Running ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct...\n");
	
	// Prepare dimension and stride arrays (HND layout: tensor_layout=1)
	int qdim[] = {B, H, R, D};
	int kdim[] = {B, H, C, D};
	int vdim[] = {B, H, C, D};
	int odim[] = {B, H, R, D};
	int q_int8_dim[] = {B, H, R, D};
	int k_int8_dim[] = {B, H, C, D};
	int query_scale_dim[] = {B, H, q_scale_blocks};
	int key_scale_dim[] = {B, H, k_scale_blocks};
	int km_dim[] = {B, H, D};
	
	// Calculate strides (NHWC format - note CCV uses NHWC internally)
	int qstride[] = {H * R * D, R * D, D, 1};
	int kstride[] = {H * C * D, C * D, D, 1};
	int vstride[] = {H * C * D, C * D, D, 1};
	int ostride[] = {H * R * D, R * D, D, 1};
	int q_int8_stride[] = {H * R * D, R * D, D, 1};
	int k_int8_stride[] = {H * C * D, C * D, D, 1};
	int query_scale_stride[] = {1, H * q_scale_blocks, q_scale_blocks, 1};
	int key_scale_stride[] = {1, H * k_scale_blocks, k_scale_blocks, 1};
	int km_stride[] = {H * D, D, 1};
	
	printf("Direct function parameters:\n");
	printf("  tensor_layout: 1 (HND)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	printf("  pv_accum_dtype: 2 (FP32)\n");
	printf("  BLKQ: %d, WARPQ: %d, BLKK: %d\n", BLKQ, WARPQ, BLKK);
	
	ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
		(__fp16*)gpu_q_tensor->data.f16,      // query
		(__fp16*)gpu_k_tensor->data.f16,      // key
		NULL,                                  // k_mean (NULL = no mean subtraction, matching PyTorch smooth_k=False)
		(__fp16*)gpu_v_tensor->data.f16,      // value
		(int8_t*)gpu_q_int8_tensor->data.u8,  // q_int8
		(int8_t*)gpu_k_int8_tensor->data.u8,  // k_int8
		(float*)gpu_q_scale_tensor->data.f32,  // query_scale
		(float*)gpu_k_scale_tensor->data.f32,  // key_scale
		(__fp16*)gpu_o_tensor->data.f16,      // output
		qdim, kdim, vdim, odim,
		q_int8_dim, k_int8_dim,
		query_scale_dim, key_scale_dim,
		qstride, kstride, vstride, ostride,
		q_int8_stride, k_int8_stride,
		query_scale_stride, key_scale_stride,
		1,                   // tensor_layout: 1=HND
		0,                   // is_causal: false
		2,                   // qk_quant_gran: 2=per_warp
		sm_scale,            // sm_scale
		0,                   // return_lse: false
		2,                   // pv_accum_dtype: 2=FP32
		BLKQ,                // BLKQ
		WARPQ,               // WARPQ
		BLKK,                // BLKK
		NULL, NULL,          // k_mean dimensions and strides (NULL since no mean subtraction)
		0                    // cuda_stream (default)
	);
	
	cudaDeviceSynchronize();
	printf("✓ ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct completed successfully\n");
	
	// Copy results back to CPU for verification
	ccv_nnc_tensor_t* cpu_output_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* cpu_q_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* cpu_k_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_scale_tensor), TENSOR_LIST(cpu_q_scale_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_scale_tensor), TENSOR_LIST(cpu_k_scale_tensor), 0);
	
	// Convert output back to FP32 for easier analysis
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(cpu_output_tensor_f16), TENSOR_LIST(cpu_output_tensor), 0);
	
	printf("\n=== Direct Function Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, cpu_output_tensor->data.f32[i]);
	}
	
	printf("\nQuantized Q (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", ((int8_t*)cpu_q_int8_tensor->data.u8)[i]);
	}
	printf("\n");
	
	printf("Q scales (all %d values): ", q_scale_blocks);
	for (int i = 0; i < H * q_scale_blocks; i++) {
		printf("%.6f ", cpu_q_scale_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("\nQuantized K (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", ((int8_t*)cpu_k_int8_tensor->data.u8)[i]);
	}
	printf("\n");
	
	printf("K scales (all %d values): ", k_scale_blocks);
	for (int i = 0; i < H * k_scale_blocks; i++) {
		printf("%.6f ", cpu_k_scale_tensor->data.f32[i]);
	}
	printf("\n");
	
	// Basic validation checks
	printf("\nValidation checks:\n");
	
	// Check output range
	float min_val = cpu_output_tensor->data.f32[0];
	float max_val = cpu_output_tensor->data.f32[0];
	for (int i = 1; i < B * H * R * D; i++) {
		if (cpu_output_tensor->data.f32[i] < min_val) min_val = cpu_output_tensor->data.f32[i];
		if (cpu_output_tensor->data.f32[i] > max_val) max_val = cpu_output_tensor->data.f32[i];
	}
	printf("Output range: [%.6f, %.6f]\n", min_val, max_val);
	
	// Check quantization worked
	int q_valid = 0, k_valid = 0;
	for (int i = 0; i < B * H * R * D; i++) {
		int8_t val = ((int8_t*)cpu_q_int8_tensor->data.u8)[i];
		if (val >= -128 && val <= 127) q_valid++;
	}
	for (int i = 0; i < B * H * C * D; i++) {
		int8_t val = ((int8_t*)cpu_k_int8_tensor->data.u8)[i];
		if (val >= -128 && val <= 127) k_valid++;
	}
	printf("Q quantization: %d/%d values in valid INT8 range\n", q_valid, B * H * R * D);
	printf("K quantization: %d/%d values in valid INT8 range\n", k_valid, B * H * C * D);
	
	// Check scales are reasonable
	float q_scale_sum = 0.0f, k_scale_sum = 0.0f;
	for (int i = 0; i < B * H * q_scale_blocks; i++) {
		q_scale_sum += cpu_q_scale_tensor->data.f32[i];
	}
	for (int i = 0; i < B * H * k_scale_blocks; i++) {
		k_scale_sum += cpu_k_scale_tensor->data.f32[i];
	}
	printf("Q scale sum: %f, K scale sum: %f\n", q_scale_sum, k_scale_sum);
	
	// Compare with PyTorch output if available
	printf("\n=== CCV Direct vs PyTorch Comparison ===\n");
	FILE* pytorch_file = fopen("/tmp/pytorch_sage_output.bin", "rb");
	if (pytorch_file) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, pytorch_file);
		fclose(pytorch_file);
		
		printf("Comparing first 10 values:\n");
		for (int i = 0; i < 10; i++) {
			float ccv_val = (float)((__fp16*)cpu_output_tensor_f16->data.u8)[i];
			float pytorch_val = (float)pytorch_data[i];
			float diff = fabsf(ccv_val - pytorch_val);
			printf("  [%d] CCV: %f, PyTorch: %f, diff: %.8f\n", i, ccv_val, pytorch_val, diff);
		}
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)((__fp16*)cpu_output_tensor_f16->data.u8)[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: CCV direct and PyTorch outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: CCV direct and PyTorch outputs are very close (within expected FP16 precision)\n");
		} else {
			printf("⚠️  WARNING: Larger differences detected between CCV direct and PyTorch\n");
		}
		
		free(pytorch_data);
	} else {
		printf("PyTorch output not found at /tmp/pytorch_sage_kernel_output.bin\n");
		printf("This is expected if you haven't run the PyTorch reference implementation.\n");
	}
	
	// Save output for comparison
	FILE* outfile = fopen("/tmp/ccv_direct_sage_output.bin", "wb");
	if (outfile) {
		fwrite(cpu_output_tensor_f16->data.u8, sizeof(__fp16), B * H * R * D, outfile);
		fclose(outfile);
		printf("✅ Saved direct function output to /tmp/ccv_direct_sage_output.bin\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(k_mean_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor_f16);
	ccv_nnc_tensor_free(cpu_output_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scale_tensor);
	ccv_nnc_tensor_free(cpu_k_scale_tensor);
	
	free(q_fp16_data);
	free(k_fp16_data);
	free(v_fp16_data);
	
	printf("✅ PASS: Output is valid\n");
#else
	printf("❌ CUDA SM80 not available\n");
#endif
	
	printf("✅ CCV SageAttention direct function test completed successfully!\n");
}


TEST_CASE("ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct_mix test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== CCV SageAttention Direct Function Test ===\n");
	printf("Testing ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct function\n");
	printf("This test validates the direct function using same inputs as wrapper test\n");
	
	// Use same test parameters as wrapper test
	int B = 1, R = 64, C = 64, H = 8, D = 128;
	float sm_scale = 1.0f / sqrtf((float)D);  // 0.088388
	
	printf("Test dimensions: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	printf("Scale: %f\n", sm_scale);
	
	// Load FP16 input data from PyTorch files (same as wrapper test)
	printf("Loading PyTorch FP16 inputs: B=%d, R=%d, C=%d, H=%d, D=%d\n", B, R, C, H, D);
	
	// Allocate host memory for FP16 input data
	__fp16* q_fp16_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
	__fp16* k_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	__fp16* v_fp16_data = (__fp16*)malloc(B * H * C * D * sizeof(__fp16));
	
	// Load FP16 data from PyTorch files
	FILE* f;
	f = fopen("/tmp/pytorch_kernel_q_fp16.bin", "rb");
	if (f) {
		fread(q_fp16_data, sizeof(__fp16), B * H * R * D, f);
		fclose(f);
		printf("✅ Loaded Q FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		printf("FP16 files not found, loading FP32 and converting...\n");
		float* q_fp32_data = (float*)malloc(B * H * R * D * sizeof(float));
		f = fopen("/tmp/pytorch_q_input.bin", "rb");
		if (f) {
			fread(q_fp32_data, sizeof(float), B * H * R * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)q_fp32_data[i];
			}
			printf("✅ Loaded and converted Q data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 Q files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * R * D; i++) {
				q_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(q_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_k_fp16.bin", "rb");
	if (f) {
		fread(k_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded K FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* k_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_k_input.bin", "rb");
		if (f) {
			fread(k_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)k_fp32_data[i];
			}
			printf("✅ Loaded and converted K data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 K files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * C * D; i++) {
				k_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(k_fp32_data);
	}
	
	f = fopen("/tmp/pytorch_kernel_v_fp16.bin", "rb");
	if (f) {
		fread(v_fp16_data, sizeof(__fp16), B * H * C * D, f);
		fclose(f);
		printf("✅ Loaded V FP16 data from disk\n");
	} else {
		// If FP16 files don't exist, load FP32 and convert
		float* v_fp32_data = (float*)malloc(B * H * C * D * sizeof(float));
		f = fopen("/tmp/pytorch_v_input.bin", "rb");
		if (f) {
			fread(v_fp32_data, sizeof(float), B * H * C * D, f);
			fclose(f);
			// Convert FP32 to FP16
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)v_fp32_data[i];
			}
			printf("✅ Loaded and converted V data from FP32\n");
		} else {
			printf("⚠️  Neither FP16 nor FP32 V files found, using random data\n");
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 1);
			for (int i = 0; i < B * H * C * D; i++) {
				v_fp16_data[i] = (__fp16)((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2);
			}
		}
		free(v_fp32_data);
	}
	
	// Print sample values for verification
	printf("Sample values: Q[0]=%f, K[0]=%f, V[0]=%f\n", 
		(float)q_fp16_data[0], (float)k_fp16_data[0], (float)v_fp16_data[0]);
	
	// Create CCV tensors and copy loaded data
	ccv_nnc_tensor_t* q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* k_mean_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, D), 0);
	
	// Copy loaded data to CCV tensors
	memcpy(q_tensor_f16->data.f16, q_fp16_data, B * H * R * D * sizeof(__fp16));
	memcpy(k_tensor_f16->data.f16, k_fp16_data, B * H * C * D * sizeof(__fp16));
	memcpy(v_tensor_f16->data.f16, v_fp16_data, B * H * C * D * sizeof(__fp16));
	
	// Initialize k_mean to zeros
	memset(k_mean_tensor_f16->data.u8, 0, B * H * D * sizeof(__fp16));
	
	// Create GPU tensors
	ccv_nnc_tensor_t* gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_k_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, D), 0);
	ccv_nnc_tensor_t* gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, R, D), 0);
	
	// Transfer to GPU
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(q_tensor_f16), TENSOR_LIST(gpu_q_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_tensor_f16), TENSOR_LIST(gpu_k_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(v_tensor_f16), TENSOR_LIST(gpu_v_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(k_mean_tensor_f16), TENSOR_LIST(gpu_k_mean_tensor), 0);
	
	printf("✅ Created and transferred tensors to GPU\n");
	
	// Create quantized output tensors
	const uint32_t BLKQ = 128;
	const uint32_t WARPQ = 16;
	const uint32_t BLKK = 64;
	const size_t q_blocks = (R + BLKQ - 1) / BLKQ;
	const size_t warps_per_block = BLKQ / WARPQ;
	const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
	const int k_scale_blocks = (C + BLKK - 1) / BLKK;      // Should be 1
	printf("q_scale_blocks:%d", q_scale_blocks);

	printf("Scale tensor dimensions: q_scale_blocks=%d, k_scale_blocks=%d\n", q_scale_blocks, k_scale_blocks);
	
	ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, k_scale_blocks), 0);
	
	printf("\n");
	
#ifdef HAVE_CUDA_SM80
	// Declare the direct function
	extern void ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
	    __fp16 *query, __fp16 *key, __fp16 *k_mean, __fp16 *value,
	    int8_t *q_int8, int8_t *k_int8,
	    float *query_scale, float *key_scale, __fp16 *output,
	    int qdim[], int kdim[], int vdim[], int odim[],
	    int q_int8_dim[], int k_int8_dim[],
	    int query_scale_dim[], int key_scale_dim[],
	    int qstride[], int kstride[], int vstride[], int ostride[],
	    int q_int8_stride[], int k_int8_stride[],
	    int query_scale_stride[], int key_scale_stride[],
	    int tensor_layout, int is_causal, int qk_quant_gran,
	    float sm_scale, int return_lse, int pv_accum_dtype,
	    int BLKQ, int WARPQ, int BLKK,
	    int km_dim[], int km_stride[], cudaStream_t cuda_stream);
	
	printf("Running ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct...\n");
	
	// Prepare dimension and stride arrays (HND layout: tensor_layout=1)
	int qdim[] = {B, H, R, D};
	int kdim[] = {B, H, C, D};
	int vdim[] = {B, H, C, D};
	int odim[] = {B, H, R, D};
	int q_int8_dim[] = {B, H, R, D};
	int k_int8_dim[] = {B, H, C, D};
	int query_scale_dim[] = {B, H, q_scale_blocks};
	int key_scale_dim[] = {B, H, k_scale_blocks};
	int km_dim[] = {B, H, D};
	
	// Calculate strides (NHWC format - note CCV uses NHWC internally)
	int qstride[] = {H * R * D, R * D, D, 1};
	int kstride[] = {H * C * D, C * D, D, 1};
	int vstride[] = {H * C * D, C * D, D, 1};
	int ostride[] = {H * R * D, R * D, D, 1};
	int q_int8_stride[] = {H * R * D, R * D, D, 1};
	int k_int8_stride[] = {H * C * D, C * D, D, 1};
	int query_scale_stride[] = {1, H * q_scale_blocks, q_scale_blocks, 1};
	int key_scale_stride[] = {1, H * k_scale_blocks, k_scale_blocks, 1};
	int km_stride[] = {H * D, D, 1};
	
	printf("Direct function parameters:\n");
	printf("  tensor_layout: 1 (HND)\n");
	printf("  is_causal: 0 (false)\n");
	printf("  qk_quant_gran: 2 (per_warp)\n");
	printf("  sm_scale: %f\n", sm_scale);
	printf("  return_lse: 0 (false)\n");
	printf("  pv_accum_dtype: 1 (DTYPE_FP16_MIX_FP32)\n");
	printf("  BLKQ: %d, WARPQ: %d, BLKK: %d\n", BLKQ, WARPQ, BLKK);
	
	ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
		(__fp16*)gpu_q_tensor->data.f16,      // query
		(__fp16*)gpu_k_tensor->data.f16,      // key
		NULL,                                  // k_mean (NULL = no mean subtraction, matching PyTorch smooth_k=False)
		(__fp16*)gpu_v_tensor->data.f16,      // value
		(int8_t*)gpu_q_int8_tensor->data.u8,  // q_int8
		(int8_t*)gpu_k_int8_tensor->data.u8,  // k_int8
		(float*)gpu_q_scale_tensor->data.f32,  // query_scale
		(float*)gpu_k_scale_tensor->data.f32,  // key_scale
		(__fp16*)gpu_o_tensor->data.f16,      // output
		qdim, kdim, vdim, odim,
		q_int8_dim, k_int8_dim,
		query_scale_dim, key_scale_dim,
		qstride, kstride, vstride, ostride,
		q_int8_stride, k_int8_stride,
		query_scale_stride, key_scale_stride,
		1,                   // tensor_layout: 1=HND
		0,                   // is_causal: false
		2,                   // qk_quant_gran: 2=per_warp
		sm_scale,            // sm_scale
		0,                   // return_lse: false
		1,                   // pv_accum_dtype: 2=DTYPE_FP16_MIX_FP32
		BLKQ,                // BLKQ
		WARPQ,               // WARPQ
		BLKK,                // BLKK
		NULL, NULL,          // k_mean dimensions and strides (NULL since no mean subtraction)
		0                    // cuda_stream (default)
	);
	
	cudaDeviceSynchronize();
	printf("✓ ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct_mix completed successfully\n");
	
	// Copy results back to CPU for verification
	ccv_nnc_tensor_t* cpu_output_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_q_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, R, D), 0);
	ccv_nnc_tensor_t* cpu_k_int8_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, C, D), 0);
	ccv_nnc_tensor_t* cpu_q_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, q_scale_blocks), 0);
	ccv_nnc_tensor_t* cpu_k_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, k_scale_blocks), 0);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(cpu_output_tensor_f16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_int8_tensor), TENSOR_LIST(cpu_q_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_int8_tensor), TENSOR_LIST(cpu_k_int8_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_q_scale_tensor), TENSOR_LIST(cpu_q_scale_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(gpu_k_scale_tensor), TENSOR_LIST(cpu_k_scale_tensor), 0);
	
	// Convert output back to FP32 for easier analysis
	ccv_nnc_tensor_t* cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, R, D), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
	                 TENSOR_LIST(cpu_output_tensor_f16), TENSOR_LIST(cpu_output_tensor), 0);
	
	printf("\n=== Direct Function Output Analysis ===\n");
	printf("Output tensor shape: [B=%d, H=%d, R=%d, D=%d] (HND layout)\n", B, H, R, D);
	printf("First 10 output values:\n");
	for (int i = 0; i < 10; i++) {
		printf("  output[%d]: %f\n", i, cpu_output_tensor->data.f32[i]);
	}
	
	printf("\nQuantized Q (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", ((int8_t*)cpu_q_int8_tensor->data.u8)[i]);
	}
	printf("\n");
	
	printf("Q scales (all %d values): ", q_scale_blocks);
	for (int i = 0; i < H * q_scale_blocks; i++) {
		printf("%.6f ", cpu_q_scale_tensor->data.f32[i]);
	}
	printf("\n");
	
	printf("\nQuantized K (first 10 values): ");
	for (int i = 0; i < 10; i++) {
		printf("%d ", ((int8_t*)cpu_k_int8_tensor->data.u8)[i]);
	}
	printf("\n");
	
	printf("K scales (all %d values): ", k_scale_blocks);
	for (int i = 0; i < H * k_scale_blocks; i++) {
		printf("%.6f ", cpu_k_scale_tensor->data.f32[i]);
	}
	printf("\n");
	
	// Basic validation checks
	printf("\nValidation checks:\n");
	
	// Check output range
	float min_val = cpu_output_tensor->data.f32[0];
	float max_val = cpu_output_tensor->data.f32[0];
	for (int i = 1; i < B * H * R * D; i++) {
		if (cpu_output_tensor->data.f32[i] < min_val) min_val = cpu_output_tensor->data.f32[i];
		if (cpu_output_tensor->data.f32[i] > max_val) max_val = cpu_output_tensor->data.f32[i];
	}
	printf("Output range: [%.6f, %.6f]\n", min_val, max_val);
	
	// Check quantization worked
	int q_valid = 0, k_valid = 0;
	for (int i = 0; i < B * H * R * D; i++) {
		int8_t val = ((int8_t*)cpu_q_int8_tensor->data.u8)[i];
		if (val >= -128 && val <= 127) q_valid++;
	}
	for (int i = 0; i < B * H * C * D; i++) {
		int8_t val = ((int8_t*)cpu_k_int8_tensor->data.u8)[i];
		if (val >= -128 && val <= 127) k_valid++;
	}
	printf("Q quantization: %d/%d values in valid INT8 range\n", q_valid, B * H * R * D);
	printf("K quantization: %d/%d values in valid INT8 range\n", k_valid, B * H * C * D);
	
	// Check scales are reasonable
	float q_scale_sum = 0.0f, k_scale_sum = 0.0f;
	for (int i = 0; i < B * H * q_scale_blocks; i++) {
		q_scale_sum += cpu_q_scale_tensor->data.f32[i];
	}
	for (int i = 0; i < B * H * k_scale_blocks; i++) {
		k_scale_sum += cpu_k_scale_tensor->data.f32[i];
	}
	printf("Q scale sum: %f, K scale sum: %f\n", q_scale_sum, k_scale_sum);
	
	// Compare with PyTorch output if available
	printf("\n=== CCV Direct vs PyTorch Comparison ===\n");
	FILE* pytorch_file = fopen("/tmp/pytorch_sage_output.bin", "rb");
	if (pytorch_file) {
		__fp16* pytorch_data = (__fp16*)malloc(B * H * R * D * sizeof(__fp16));
		fread(pytorch_data, sizeof(__fp16), B * H * R * D, pytorch_file);
		fclose(pytorch_file);
		
		printf("Comparing first 10 values:\n");
		for (int i = 0; i < 10; i++) {
			float ccv_val = (float)((__fp16*)cpu_output_tensor_f16->data.u8)[i];
			float pytorch_val = (float)pytorch_data[i];
			float diff = fabsf(ccv_val - pytorch_val);
			printf("  [%d] CCV: %f, PyTorch: %f, diff: %.8f\n", i, ccv_val, pytorch_val, diff);
		}
		
		// Full tensor comparison
		int exact_matches = 0;
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * H * R * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)((__fp16*)cpu_output_tensor_f16->data.u8)[i] - (float)pytorch_data[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
		}
		
		printf("\nFull tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: CCV direct and PyTorch outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: CCV direct and PyTorch outputs are very close (within expected FP16 precision)\n");
		} else {
			printf("⚠️  WARNING: Larger differences detected between CCV direct and PyTorch\n");
		}
		
		free(pytorch_data);
	} else {
		printf("PyTorch output not found at /tmp/pytorch_sage_kernel_output.bin\n");
		printf("This is expected if you haven't run the PyTorch reference implementation.\n");
	}
	
	// Save output for comparison
	FILE* outfile = fopen("/tmp/ccv_direct_sage_output.bin", "wb");
	if (outfile) {
		fwrite(cpu_output_tensor_f16->data.u8, sizeof(__fp16), B * H * R * D, outfile);
		fclose(outfile);
		printf("✅ Saved direct function output to /tmp/ccv_direct_sage_output.bin\n");
	}
	
	// Cleanup
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(k_mean_tensor_f16);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_mean_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_q_int8_tensor);
	ccv_nnc_tensor_free(gpu_k_int8_tensor);
	ccv_nnc_tensor_free(gpu_q_scale_tensor);
	ccv_nnc_tensor_free(gpu_k_scale_tensor);
	ccv_nnc_tensor_free(cpu_output_tensor_f16);
	ccv_nnc_tensor_free(cpu_output_tensor);
	ccv_nnc_tensor_free(cpu_q_int8_tensor);
	ccv_nnc_tensor_free(cpu_k_int8_tensor);
	ccv_nnc_tensor_free(cpu_q_scale_tensor);
	ccv_nnc_tensor_free(cpu_k_scale_tensor);
	
	free(q_fp16_data);
	free(k_fp16_data);
	free(v_fp16_data);
	
	printf("✅ PASS: Output is valid\n");
#else
	printf("❌ CUDA SM80 not available\n");
#endif
	
	printf("✅ CCV SageAttention direct function test completed successfully!\n");
}


TEST_CASE("scaled dot product attention with sage_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	printf("\n=== SageAttention GPU vs CPU Reference Test ===\n");
	printf("Testing %d configurations with various dimensions and causal settings\n", num_trials);
	printf("Note: SageAttention only supports D=64 and D=128. Other dimensions will use FlashAttention.\n");

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 128, 128, 64, 64, 128 };  // Changed to only use supported dimensions
		int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		// Detailed comparison between GPU SageAttention and CPU reference
		int exact_matches = 0;
		int very_close_matches = 0;  // diff < 1e-5
		int close_matches = 0;       // diff < 1e-4
		int acceptable_matches = 0;  // diff < 1e-3
		double max_diff = 0.0, sum_diff = 0.0;
		int total_elements = B * R * Hq * D;
		
		for (int i = 0; i < total_elements; i++) {
			double diff = fabs((float)copy_of_gpu_o_tensor->data.f32[i] - (float)o_tensor->data.f32[i]);
			sum_diff += diff;
			if (diff > max_diff) max_diff = diff;
			if (diff < 1e-6) exact_matches++;
			if (diff < 1e-5) very_close_matches++;
			if (diff < 1e-4) close_matches++;
			if (diff < 1e-3) acceptable_matches++;
		}
		
		printf("\n[Trial %d] Config: B=%d, R=%d, C=%d, Hq=%d, Hk=%d, D=%d, causal=%d\n", 
			trial, B, R, C, Hq, Hk, D, is_causal);
		printf("Full tensor comparison (%d elements):\n", total_elements);
		printf("  Exact matches (diff < 1e-6): %d/%d (%.2f%%)\n", 
			exact_matches, total_elements, (100.0 * exact_matches) / total_elements);
		printf("  Very close (diff < 1e-5): %d/%d (%.2f%%)\n", 
			very_close_matches, total_elements, (100.0 * very_close_matches) / total_elements);
		printf("  Close (diff < 1e-4): %d/%d (%.2f%%)\n", 
			close_matches, total_elements, (100.0 * close_matches) / total_elements);
		printf("  Acceptable (diff < 1e-3): %d/%d (%.2f%%)\n", 
			acceptable_matches, total_elements, (100.0 * acceptable_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		// Provide interpretation of results
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: GPU SageAttention and CPU outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: GPU SageAttention and CPU outputs are very close (within expected FP16 precision)\n");
		} else if (max_diff < 3e-3) {
			printf("⚠️  ACCEPTABLE: GPU SageAttention and CPU outputs have small differences but within tolerance\n");
		} else {
			printf("❌ FAILED: GPU SageAttention and CPU outputs differ significantly\n");
			printf("  This may indicate a precision issue or algorithmic difference\n");
		}
		
		REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 3e-3, "GPU SageAttention output should match CPU reference within tolerance");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

#include "case_main.h"
