#include "case.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <time.h>
#include "nnc/ccv_nnc_internal.h"



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

TEST_CASE("cublas forward gemm in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 64), 0);

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
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("cublas forward gemv in bfloat precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 64), 0);

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
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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

TEST_CASE("ewabs forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWABS_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewabs backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWABS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWABS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
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

TEST_CASE("scaled dot product attention with flash_attn in bfloat")
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
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, B * R * Hq * D, 1e-2, "GPU computed output should be the same as CPU computed ones");

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

TEST_CASE("scaled dot product attention + unify head with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512), "bias");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(q, k, v, NO_TENSOR_SYMBOL, w, bias), TENSOR_SYMBOL_LIST(r, NO_TENSOR_SYMBOL, c), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, q);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, k);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, v);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bias);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512 * 512; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / sqrtf(512);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const w_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 512, 512), 0);
	ccv_nnc_tensor_t* const bias_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), 0);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(g_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(gq, gk, gv, NO_TENSOR_SYMBOL, gw, gbias), TENSOR_SYMBOL_LIST(gr, NO_TENSOR_SYMBOL, gc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(g_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* g_graph = 0;
	ccv_nnc_tensor_arena_t* g_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* g_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(g_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(g_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(g_symbolic_graph), &g_graph, &g_tensor_arena, &g_graph_exec_arena);
	ccv_nnc_tensor_t* const gq_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gq);
	ccv_nnc_tensor_t* const gk_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gk);
	ccv_nnc_tensor_t* const gv_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gv);
	ccv_nnc_tensor_t* const gw_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gw);
	ccv_nnc_tensor_t* const gbias_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gbias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, c);
	ccv_nnc_tensor_t* const gc_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gc);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const ho_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc_tensor, gr_tensor), TENSOR_LIST(ho_f16, hr_f16), 0);
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ho_f16, hr_f16), TENSOR_LIST(ho, hr), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, ho->data.f32, 32 * 128 * 8 * 64, 3e-3, "graph computed result should match scaled dot product attention op result");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, r_tensor->data.f32, hr->data.f32, 32 * 128 * 512, 3e-2, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(ho_f16);
	ccv_nnc_tensor_free(hr_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(w_tensor_f16);
	ccv_nnc_tensor_free(bias_tensor_f16);
}

TEST_CASE("scaled dot product attention + unify head with flash_attn in bfloat")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512), "bias");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(q, k, v, NO_TENSOR_SYMBOL, w, bias), TENSOR_SYMBOL_LIST(r, NO_TENSOR_SYMBOL, c), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, q);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, k);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, v);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bias);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512 * 512; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / sqrtf(512);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const w_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 512, 512), 0);
	ccv_nnc_tensor_t* const bias_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), 0);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(g_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(gq, gk, gv, NO_TENSOR_SYMBOL, gw, gbias), TENSOR_SYMBOL_LIST(gr, NO_TENSOR_SYMBOL, gc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(g_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* g_graph = 0;
	ccv_nnc_tensor_arena_t* g_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* g_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(g_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(g_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(g_symbolic_graph), &g_graph, &g_tensor_arena, &g_graph_exec_arena);
	ccv_nnc_tensor_t* const gq_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gq);
	ccv_nnc_tensor_t* const gk_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gk);
	ccv_nnc_tensor_t* const gv_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gv);
	ccv_nnc_tensor_t* const gw_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gw);
	ccv_nnc_tensor_t* const gbias_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gbias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, c);
	ccv_nnc_tensor_t* const gc_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gc);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const ho_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc_tensor, gr_tensor), TENSOR_LIST(ho_f16, hr_f16), 0);
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ho_f16, hr_f16), TENSOR_LIST(ho, hr), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, ho->data.f32, 32 * 128 * 8 * 64, 1e-2, "graph computed result should match scaled dot product attention op result");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, r_tensor->data.f32, hr->data.f32, 32 * 128 * 512, 1e-1, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(ho_f16);
	ccv_nnc_tensor_free(hr_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(w_tensor_f16);
	ccv_nnc_tensor_free(bias_tensor_f16);
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
		ccv_nnc_tensor_free(gpu_softmax_lse);
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

TEST_CASE("cmul in float, broadcast semantics with longer than 65535 sequence")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 8, 16), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 1, 16), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 8, 16), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 1, 16), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1 * 70000 * 8 * 16; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 70000 * 1 * 16; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
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

TEST_CASE("cmul in float, broadcast semantics with longer than 65535 sequence and more than 1 batch size")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 2, 40000, 8, 16), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 40000, 1, 16), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 2, 40000, 8, 16), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 40000, 1, 16), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 40000 * 8 * 16; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 40000 * 1 * 16; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
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
		int is_causal_candidates[num_trials] = {  0, 0, 0, 0, 0, 0 };

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

		// Generate test data with realistic ranges for SageAttention quantization
		// Use a range from -3 to +3 to ensure meaningful quantization scales
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
		// Use INT8 quantization flag to make CPU reference comparable to GPU SageAttention
		ccv_nnc_cmd_t cpu_cmd_with_int8 = ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0, 
			((ccv_nnc_cmd_param_t){
				.size={.dim={1,1,1}},
				.scaled_dot_product_attention={
					.scale=scale,
					.is_causal=is_causal,
					.flags=CCV_NNC_GEMM_8U // Request INT8 quantization for fair comparison
				}
			}), 0);
		ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
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

		ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

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
			if (diff < 4*1e-3) acceptable_matches++;
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
		printf("  Acceptable (diff < 4*1e-3): %d/%d (%.2f%%)\n", 
			acceptable_matches, total_elements, (100.0 * acceptable_matches) / total_elements);
		printf("  Maximum difference: %.8f\n", max_diff);
		printf("  Average difference: %.8f\n", sum_diff / total_elements);
		
		// Provide interpretation of results
		if (max_diff < 1e-5) {
			printf("🎯 EXCELLENT: GPU SageAttention and CPU outputs are virtually identical!\n");
		} else if (max_diff < 1e-3) {
			printf("✅ GOOD: GPU SageAttention and CPU outputs are very close (within expected FP16 precision)\n");
		} else if (max_diff < 4*1e-3) {
			printf("⚠️  ACCEPTABLE: GPU SageAttention and CPU outputs have small differences but within tolerance\n");
		} else {
			printf("❌ FAILED: GPU SageAttention and CPU outputs differ significantly\n");
			printf("  This may indicate a precision issue or algorithmic difference\n");
		}
		
		REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 4*1e-3, "GPU SageAttention output should match CPU reference within tolerance");

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

TEST_CASE("segmented gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 2;
	hindices->data.i32[2] = 1;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 3 * 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(a, indices, counts, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 2;
	hindices->data.i32[2] = 1;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 128), 0);
	ccv_nnc_tensor_t* hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(ha16, hw16), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hindices, hcounts, hw16), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb16), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 80 * 64, 1e-3, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hw16);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm with bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 3 * 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 128), 0);
	ccv_nnc_tensor_t* hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(ha16, hw16, hbias16), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hindices, hcounts, hw16, hbias16), TENSOR_LIST(a, indices, counts, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb16), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 80 * 64, 1e-3, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hw16);
	ccv_nnc_tensor_free(hbias16);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm, reuse")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 30;
	hcounts->data.i32[2] = 30;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("direct sage vs flash attention NHD performance test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	
	printf("\n=== Direct SageAttention NHD Performance Test ===\n");
	printf("Testing performance with memory reuse across 1000 runs\n");
	
	// Test configurations based on sage_attn test case
	int num_trials = 1;
	int B_candidates[6] = {  32,   12, 16, 1, 2, 1 };
	int R_candidates[6] = { 160,  256, 128, 77, 77, 5 };
	int C_candidates[6] = { 128,  128, 128, 128, 128, 5 };
	int Hq_candidates[6] = {   8,  8, 8, 8, 8, 32 };
	int Hk_candidates[6] = {   8,  8, 8, 8, 2, 8 };
	int D_candidates[6] = {  64, 128, 128, 64, 64, 128 };
	int is_causal_candidates[6] = {  0, 0, 0, 0, 0, 0 };
	
	double trial_times[6];
	
	for (int trial = 0; trial < num_trials; ++trial) {
		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		printf("\n[Trial %d] Config: B=%d, R=%d, C=%d, Hq=%d, Hk=%d, D=%d, causal=%d\n", 
			trial, B, R, C, Hq, Hk, D, is_causal);
		  // Main operations:
  // 1. Q @ K^T: [B, R, Hq, D] @ [B, D, Hk, C] -> [B, R, Hq, C]
  //    FLOPs = B * Hq * R * C * D * 2  (multiply-add operations)
  // 2. Softmax: B * Hq * R * C operations
  // 3. Result @ V: [B, R, Hq, C] @ [B, C, Hk, D] -> [B, R, Hq, D] 
  //    FLOPs = B * Hq * R * C * D * 2


		int total_Flops = B * Hq * R * C * D * 4;
		printf("Full tensor comparison (%d total_Flops):\n", total_Flops);
		// Generate test data programmatically
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		// Initialize with realistic data ranges for SageAttention quantization
		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		// Convert to FP16
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Allocate GPU tensors (reused across all runs)
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		
		// Transfer data to GPU once
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

#ifdef HAVE_CUDA_SM80
		// Create quantized output tensors for the direct function
		const uint32_t BLKQ = 128;
		const uint32_t WARPQ = 32;
		const uint32_t BLKK = 64;
		const size_t q_blocks = (R + BLKQ - 1) / BLKQ;  // Use sequence length R for NHD
		const size_t warps_per_block = BLKQ / WARPQ;
		const int q_scale_blocks = q_blocks * warps_per_block;  // Should be 4
		const int k_scale_blocks = (C + BLKK - 1) / BLKK;      // Should be 1

		ccv_nnc_tensor_t* gpu_q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* gpu_k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* gpu_q_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, q_scale_blocks), 0);
		ccv_nnc_tensor_t* gpu_k_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hk, k_scale_blocks), 0);

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

		// Prepare dimension and stride arrays (NHD layout: tensor_layout=0)
		int qdim[] = {B, R, Hq, D};      // NHD format
		int kdim[] = {B, C, Hk, D};
		int vdim[] = {B, C, Hk, D};
		int odim[] = {B, R, Hq, D};
		int q_int8_dim[] = {B, R, Hq, D};
		int k_int8_dim[] = {B, C, Hk, D};
		int query_scale_dim[] = {B, Hq, q_scale_blocks};
		int key_scale_dim[] = {B, Hk, k_scale_blocks};
		int km_dim[] = {B, Hq, D};

		// Calculate strides for NHD layout: [B, S, H, D]
		int qstride[] = {R * Hq * D, Hq * D, D, 1};
		int kstride[] = {C * Hk * D, Hk * D, D, 1};
		int vstride[] = {C * Hk * D, Hk * D, D, 1};
		int ostride[] = {R * Hq * D, Hq * D, D, 1};
		int q_int8_stride[] = {R * Hq * D, Hq * D, D, 1};
		int k_int8_stride[] = {C * Hk * D, Hk * D, D, 1};
		int query_scale_stride[] = {1, Hq * q_scale_blocks, q_scale_blocks, 1};
		int key_scale_stride[] = {1, Hk * k_scale_blocks, k_scale_blocks, 1};
		int km_stride[] = {Hq * D, D, 1};

		// Warm up runs
		int warmup_runs = 10;
		for (int i = 0; i < warmup_runs; i++) {
			ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
				(__fp16*)gpu_q_tensor->data.f16,      // query
				(__fp16*)gpu_k_tensor->data.f16,      // key
				NULL,                                 // k_mean (not using mean subtraction)
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
				0,                   // tensor_layout: 0=NHD
				is_causal,           // is_causal
				2,                   // qk_quant_gran: 2=per_warp
				scale,               // sm_scale
				0,                   // return_lse: false
				1,                   // pv_accum_dtype: 2=FP32
				BLKQ,                // BLKQ
				WARPQ,               // WARPQ
				BLKK,                // BLKK
				km_dim, km_stride,   // k_mean dimensions and strides
				0                    // cuda_stream (default)
			);
		}
		cudaDeviceSynchronize();
		
		// Performance measurement with memory reuse
		int num_runs = 10;
		struct timespec start_time, end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);
		
		for (int i = 0; i < num_runs; i++) {
			ccv_nnc_sageattn_qk_int8_pv_fp16_cuda_direct(
				(__fp16*)gpu_q_tensor->data.f16,      // query
				(__fp16*)gpu_k_tensor->data.f16,      // key
				NULL,                                 // k_mean (not using mean subtraction)
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
				0,                   // tensor_layout: 0=NHD
				is_causal,           // is_causal
				2,                   // qk_quant_gran: 2=per_warp
				scale,               // sm_scale
				0,                   // return_lse: false
				1,                   // pv_accum_dtype: 2=FP32
				BLKQ,                // BLKQ
				WARPQ,               // WARPQ
				BLKK,                // BLKK
				km_dim, km_stride,   // k_mean dimensions and strides
				0                    // cuda_stream (default)
			);
		}
		
		// Wait for GPU to complete all runs
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		
		// Cleanup additional tensors
		ccv_nnc_tensor_free(gpu_q_int8_tensor);
		ccv_nnc_tensor_free(gpu_k_int8_tensor);
		ccv_nnc_tensor_free(gpu_q_scale_tensor);
		ccv_nnc_tensor_free(gpu_k_scale_tensor);
#else
		printf("CUDA SM80 not available, skipping direct function call\n");
		return;
#endif
		
		// Calculate average elapsed time in milliseconds
		double total_elapsed_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) + 
		                          ((end_time.tv_nsec - start_time.tv_nsec) / 1000000.0);
		double elapsed_ms = total_elapsed_ms / num_runs;
		trial_times[trial] = elapsed_ms;
		
		printf("GPU Execution Time: %.4f ms (average of %d runs with memory reuse)\n", elapsed_ms, num_runs);
		
		// Cleanup
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
	}
	
	// Print performance summary
	printf("\n=== Direct SageAttention NHD Performance Summary ===\n");
	printf("Individual trial times (ms):\n");
	for (int i = 0; i < num_trials; i++) {
		printf("  Trial %d: %.4f ms\n", i, trial_times[i]);
	}
	
	// Find min and max times
	double min_time = trial_times[0], max_time = trial_times[0];
	int min_idx = 0, max_idx = 0;
	for (int i = 1; i < num_trials; i++) {
		if (trial_times[i] < min_time) {
			min_time = trial_times[i];
			min_idx = i;
		}
		if (trial_times[i] > max_time) {
			max_time = trial_times[i];
			max_idx = i;
		}
	}
	printf("Fastest trial: %d (%.4f ms)\n", min_idx, min_time);
	printf("Slowest trial: %d (%.4f ms)\n", max_idx, max_time);
	printf("====================================================\n");
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

TEST_CASE("direct sage CMD performance test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials 1

	double trial_flash_times[num_trials];

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 128, 128, 64, 64, 128 };  // Changed to only use supported dimensions
		int is_causal_candidates[num_trials] = {  0, 0, 0, 0, 0, 0 };
		
		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);
		printf("\n[Trial %d] Config: B=%d, R=%d, C=%d, Hq=%d, Hk=%d, D=%d, causal=%d\n", 
			trial, B, R, C, Hq, Hk, D, is_causal);
		int total_flops = B * Hq * R * C * D * 4;
		printf("Full tensor comparison (%d total_flops):\n", total_flops);
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
		ccv_nnc_cmd_t cpu_cmd_with_int8 = ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0, 
			((ccv_nnc_cmd_param_t){
				.size={.dim={1,1,1}},
				.scaled_dot_product_attention={
					.scale=scale,
					.is_causal=is_causal,
					.flags=CCV_NNC_GEMM_8U// Request INT8 quantization for fair comparison
				}
			}), 0);
		// Warm up runs
		printf("\n=== Starting warmup runs (with profiling) ===\n");
		int warmup_runs = 2;
		for (int i = 0; i < warmup_runs; i++) {
			ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		}
		ccv_nnc_stream_context_wait(0);
		
		// Time the GPU attention execution with multiple runs
		int num_runs = 10;
		struct timespec start_time, end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);
		
		for (int i = 0; i < num_runs; i++) {
			ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		}
		
		// Wait for GPU to complete all runs
		ccv_nnc_stream_context_wait(0);
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		
		// Calculate average elapsed time in milliseconds
		double total_elapsed_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) + 
		                          ((end_time.tv_nsec - start_time.tv_nsec) / 1000000.0);
		double elapsed_ms = total_elapsed_ms / num_runs;
		trial_flash_times[trial] = elapsed_ms;
		printf("GPU Execution Time: %.4f ms (average of %d runs with memory reuse)\n", elapsed_ms, num_runs);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
	
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
	
	// Print performance summary
	printf("\n=== SageAttention Performance Summary ===\n");
	printf("Individual trial times (ms):\n");
	for (int i = 0; i < num_trials; i++) {
		printf("  Trial %d: %.4f ms\n", i, trial_flash_times[i]);
	}
	
	// Find min and max times
	int min_time = trial_flash_times[0], max_time = trial_flash_times[0];
	int min_idx = 0, max_idx = 0;
	for (int i = 1; i < num_trials; i++) {
		if (trial_flash_times[i] < min_time) {
			min_time = trial_flash_times[i];
			min_idx = i;
		}
		if (trial_flash_times[i] > max_time) {
			max_time = trial_flash_times[i];
			max_idx = i;
		}
	}
	printf("Fastest trial: %d (%.4f ms)\n", min_idx, min_time);
	printf("Slowest trial: %d (%.4f ms)\n", max_idx, max_time);
	printf("====================================================\n");

#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("direct flash CMD performance test")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials 1

	double trial_flash_times[num_trials];

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 128, 128, 64, 64, 128 };  // Changed to only use supported dimensions
		int is_causal_candidates[num_trials] = {  0, 0, 0, 0, 0, 0 };
		
		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);
		printf("\n[Trial %d] Config: B=%d, R=%d, C=%d, Hq=%d, Hk=%d, D=%d, causal=%d\n", 
			trial, B, R, C, Hq, Hk, D, is_causal);
		int total_flops = B * Hq * R * C * D * 4;
		printf("Full tensor comparison (%d total_flops):\n", total_flops);
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
		ccv_nnc_cmd_t cpu_cmd_with_int8 = ccv_nnc_cmd(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, 0, 
			((ccv_nnc_cmd_param_t){
				.size={.dim={1,1,1}},
				.scaled_dot_product_attention={
					.scale=scale,
					.is_causal=is_causal,
					.flags=CCV_NNC_GEMM_16F// Request INT8 quantization for fair comparison
				}
			}), 0);
		// Warm up runs
		int warmup_runs = 2;
		for (int i = 0; i < warmup_runs; i++) {
			ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		}
		ccv_nnc_stream_context_wait(0);
		
		// Time the GPU attention execution with multiple runs
		int num_runs = 10;
		struct timespec start_time, end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);
		
		for (int i = 0; i < num_runs; i++) {
			ccv_nnc_cmd_exec(cpu_cmd_with_int8, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		}
		
		// Wait for GPU to complete all runs
		ccv_nnc_stream_context_wait(0);
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		
		// Calculate average elapsed time in milliseconds
		double total_elapsed_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) + 
		                          ((end_time.tv_nsec - start_time.tv_nsec) / 1000000.0);
		double elapsed_ms = total_elapsed_ms / num_runs;
		trial_flash_times[trial] = elapsed_ms;
		printf("GPU Execution Time: %.4f ms (average of %d runs with memory reuse)\n", elapsed_ms, num_runs);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
	
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
	
	// Print performance summary
	printf("\n=== FlashAttention Performance Summary ===\n");
	printf("Individual trial times (ms):\n");
	for (int i = 0; i < num_trials; i++) {
		printf("  Trial %d: %.4f ms\n", i, trial_flash_times[i]);
	}
	
	// Find min and max times
	int min_time = trial_flash_times[0], max_time = trial_flash_times[0];
	int min_idx = 0, max_idx = 0;
	for (int i = 1; i < num_trials; i++) {
		if (trial_flash_times[i] < min_time) {
			min_time = trial_flash_times[i];
			min_idx = i;
		}
		if (trial_flash_times[i] > max_time) {
			max_time = trial_flash_times[i];
			max_idx = i;
		}
	}
	printf("Fastest trial: %d (%.4f ms)\n", min_idx, min_time);
	printf("Slowest trial: %d (%.4f ms)\n", max_idx, max_time);
	printf("====================================================\n");

#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("ccv_nnc_scale_fuse_quant_cuda_direct NHD test")
{
    ccv_cli_set_output_levels(CCV_CLI_VERBOSE);
    GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
                      ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));

    printf("=== CCV SageAttention ccv_nnc_scale_fuse_quant_cuda_direct Test ===\n");

    // Test parameters matching Python reference script
    const int B = 1;      // batch size
    const int H = 8;      // number of heads  
    const int D = 128;    // head dimension
    const int S = 64;     // sequence length
    const float scale_max = 448.0f;  // E4M3 max scale
    const int tensor_layout = 0;     // 0=NHD, 1=HND

    printf("Test parameters: B=%d, H=%d, D=%d, S=%d\n", B, H, D, S);
    printf("Scale max: %.1f, tensor_layout: %d\n", scale_max, tensor_layout);
    printf("Input tensor shape: (%d, %d, %d, %d)\n", B, D, H, S);
    printf("Output tensor shape: (%d, %d, %d, %d)\n", B, D, H, S);
    printf("Scale tensor shape: (%d, %d, %d)\n", B, H, D);

    // Create input tensor (format expected by scale_fuse_quant: [B, D, H, S])
    ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, D, H, S), 0);

    // Load input data from Python reference script
    FILE* input_file = fopen("/tmp/test_scale_fuse_input.bin", "rb");
    if (input_file) {
        // Convert to FP32 for loading, then convert to FP16
        uint16_t* temp_fp16_data = (uint16_t*)malloc(B * D * H * S * sizeof(uint16_t));
        fread(temp_fp16_data, sizeof(uint16_t), B * D * H * S, input_file);
        fclose(input_file);
        
        // Convert FP16 to FP32 for CCV tensor
        for (int i = 0; i < B * D * H * S; i++) {
            union { uint16_t u16; __fp16 f16; } fp16_union;
            fp16_union.u16 = temp_fp16_data[i];
            input_tensor->data.f32[i] = (float)fp16_union.f16;
        }
        free(temp_fp16_data);
        printf("✓ Loaded input from /tmp/test_scale_fuse_input.bin\n");
    } else {
        printf("❌ Could not load /tmp/test_scale_fuse_input.bin - using random data\n");
        printf("Run: python /home/wlin1/Drawthings/ccv/test_scale_fuse_quant_cuda_reference.py\n");
        
        // Fill with random data as fallback
        dsfmt_t dsfmt;
        dsfmt_init_gen_rand(&dsfmt, 42);
        for (int i = 0; i < B * D * H * S; i++) {
            input_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2;
        }
    }

    // Calculate input range
    float input_min = input_tensor->data.f32[0], input_max = input_tensor->data.f32[0];
    for (int i = 1; i < B * D * H * S; i++) {
        if (input_tensor->data.f32[i] < input_min) input_min = input_tensor->data.f32[i];
        if (input_tensor->data.f32[i] > input_max) input_max = input_tensor->data.f32[i];
    }
    printf("Input tensor range: [%.6f, %.6f]\n", input_min, input_max);
    
    // Print input data for verification (matching Python format)
    printf("Input sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", input_tensor->data.f32[i]);
    }
    printf("\n");

    // Convert to FP16 and transfer to GPU
    ccv_nnc_tensor_t* const input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, D, H, S), 0);
    ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
                     TENSOR_LIST(input_tensor), TENSOR_LIST(input_tensor_f16), 0);

    ccv_nnc_tensor_t* const gpu_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, D, H, S), 0);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(input_tensor_f16), TENSOR_LIST(gpu_input_tensor), 0);

    // Create output tensors on GPU
    ccv_nnc_tensor_t* const gpu_output_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, D, H, S), 0);
    ccv_nnc_tensor_t* const gpu_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, D, 1), 0);

    printf("\n");
    // Run direct quantization function
#ifdef HAVE_CUDA_SM80
    extern void ccv_nnc_scale_fuse_quant_cuda_direct(
        __fp16 *input, int8_t *output, float *scale,
        int input_dim[], int output_dim[], int scale_dim[],
        int input_stride[], int output_stride[], int scale_stride[],
        int num_tokens, float scale_max, int tensor_layout, cudaStream_t cuda_stream);

    printf("Calling ccv_nnc_scale_fuse_quant_cuda_direct...\n");

    // Prepare dimension and stride arrays
    int input_dim[] = {B, D, H, S};       // Input shape: [B, D, H, S] (expected by scale_fuse_quant)
    int output_dim[] = {B, D, H, S};      // Output shape: [B, D, H, S] 
    int scale_dim[] = {B, H, D};          // Scale shape: [B, H, D]

    // Calculate strides for [B, D, H, S] layout
    int input_stride[] = {D * H * S, H * S, S, 1};
    int output_stride[] = {D * H * S, H * S, S, 1};
    int scale_stride[] = {H * D, D, 1};

    ccv_nnc_scale_fuse_quant_cuda_direct(
        (__fp16*)gpu_input_tensor->data.f16,   // Input tensor (FP16)
        (int8_t*)gpu_output_tensor->data.u8,   // Output tensor (INT8)
        (float*)gpu_scale_tensor->data.f32,    // Scale tensor (FP32)
        input_dim, output_dim, scale_dim,      // Tensor dimensions
        input_stride, output_stride, scale_stride,  // Tensor strides
        S,                                      // num_tokens (actual sequence length)
        scale_max,                             // Maximum scale value
        tensor_layout,                         // Tensor layout (0=NHD)
        0                                      // CUDA stream (default)
    );

    cudaDeviceSynchronize();
    printf("✓ scale_fuse_quant_cuda completed successfully\n");

    // Copy results back to CPU for verification
    ccv_nnc_tensor_t* const cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, D, H, S), 0);
    ccv_nnc_tensor_t* const cpu_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D, 1), 0);

    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_output_tensor), TENSOR_LIST(cpu_output_tensor), 0);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_scale_tensor), TENSOR_LIST(cpu_scale_tensor), 0);

    // Verify output
    printf("\n=== CCV Output Analysis ===\n");
    printf("Output tensor shape: [%d, %d, %d, %d]\n", B, H, D, S);
    printf("Scale tensor shape: [%d, %d, %d]\n", B, H, D);

    // Print sample outputs
    int8_t* output_data = (int8_t*)cpu_output_tensor->data.u8;
    float* scale_data = cpu_scale_tensor->data.f32;

    printf("Output sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output_data[i]);
    }
    printf("\n");

    printf("Scale sample (first 5): ");
    for (int i = 0; i < ccv_min(5, B * H * D); i++) {
        printf("%.6f ", scale_data[i]);
    }
    printf("\n");

    // Calculate output ranges
    int8_t output_min = output_data[0], output_max = output_data[0];
    float scale_min = scale_data[0], scale_max_actual = scale_data[0];
    for (int i = 1; i < B * D * H * S; i++) {
        if (output_data[i] < output_min) output_min = output_data[i];
        if (output_data[i] > output_max) output_max = output_data[i];
    }
    for (int i = 1; i < B * H * D; i++) {
        if (scale_data[i] < scale_min) scale_min = scale_data[i];
        if (scale_data[i] > scale_max_actual) scale_max_actual = scale_data[i];
    }
    printf("Output range: [%d, %d]\n", output_min, output_max);
    printf("Scale range: [%.6f, %.6f]\n", scale_min, scale_max_actual);

    // Compare with PyTorch reference results
    printf("\n=== CCV vs PyTorch Reference Comparison ===\n");
    FILE* pytorch_output_file = fopen("/tmp/pytorch_scale_fuse_output.bin", "rb");
    FILE* pytorch_scale_file = fopen("/tmp/pytorch_scale_fuse_scales.bin", "rb");

    if (pytorch_output_file && pytorch_scale_file) {
        // Load PyTorch reference results
        int8_t* pytorch_output = (int8_t*)malloc(B * H * D * S * sizeof(int8_t));
        float* pytorch_scales = (float*)malloc(B * H * D * sizeof(float));

        fread(pytorch_output, sizeof(int8_t), B * H * D * S, pytorch_output_file);
        fread(pytorch_scales, sizeof(float), B * H * D, pytorch_scale_file);

        fclose(pytorch_output_file);
        fclose(pytorch_scale_file);

        // Compare quantized output values (first 20)
        printf("Output comparison (first 20 values):\n");
        int output_matches = 0;
        for (int i = 0; i < 20; i++) {
            if (output_data[i] == pytorch_output[i]) output_matches++;
            printf("  [%d] CCV: %d, PyTorch: %d %s\n", i, output_data[i], pytorch_output[i], 
                (output_data[i] == pytorch_output[i]) ? "✓" : "✗");
        }

        // Compare scale values
        printf("Scale comparison (first 10 values):\n");
        int scale_close = 0;
        for (int i = 0; i < ccv_min(10, B * H * D); i++) {
            float diff = fabsf(scale_data[i] - pytorch_scales[i]);
            if (diff < 1e-6) scale_close++;
            printf("  [%d] CCV: %.6f, PyTorch: %.6f, diff: %.8f %s\n", i, 
                scale_data[i], pytorch_scales[i], diff, (diff < 1e-6) ? "✓" : "✗");
        }

        // Full comparison statistics
        int total_output_matches = 0, total_scale_close = 0;
        for (int i = 0; i < B * D * H * S; i++) {
            if (output_data[i] == pytorch_output[i]) total_output_matches++;
        }
        for (int i = 0; i < B * H * D; i++) {
            float diff = fabsf(scale_data[i] - pytorch_scales[i]);
            if (diff < 1e-6) total_scale_close++;
        }

        printf("\nFull comparison statistics:\n");
        printf("  Output exact matches: %d/%d (%.2f%%)\n", 
            total_output_matches, B * D * H * S, (100.0 * total_output_matches) / (B * D * H * S));
        printf("  Scale close matches: %d/%d (%.2f%%)\n", 
            total_scale_close, B * H * D, (100.0 * total_scale_close) / (B * H * D));

        if (total_output_matches == B * D * H * S && total_scale_close == B * H * D) {
            printf("🎯 PERFECT: CCV scale_fuse_quant matches PyTorch reference exactly!\n");
        } else if (total_output_matches > 0.95 * B * D * H * S && total_scale_close > 0.95 * B * H * D) {
            printf("✅ GOOD: CCV scale_fuse_quant is very close to PyTorch reference (>95%% match)\n");
        } else {
            printf("⚠️  WARNING: Significant differences between CCV and PyTorch quantization\n");
        }

        // Test quantization quality (dequantization)
        printf("\n=== Quantization Quality Analysis ===\n");
        double total_abs_error = 0.0, max_abs_error = 0.0;
        double total_rel_error = 0.0, max_rel_error = 0.0;
        int valid_samples = 0;

        // For tensor_layout=0 (NHD): output [B,D,H,S], scale [B,H,D] 
        // Need proper broadcasting: scale[b,h,d] -> scale for output[b,d,h,s]
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                for (int h = 0; h < H; h++) {
                    for (int s = 0; s < S; s++) {
                        int output_idx = b * D * H * S + d * H * S + h * S + s;
                        int scale_idx = b * H * D + h * D + d;
                        int input_idx = output_idx;  // Same indexing for [B,D,H,S] layout
                        
                        float original = input_tensor->data.f32[input_idx];
                        float dequantized = (float)output_data[output_idx] * scale_data[scale_idx];
                        
                        float abs_error = fabsf(dequantized - original);
                        float rel_error = abs_error / (fabsf(original) + 1e-8f);
                        
                        total_abs_error += abs_error;
                        total_rel_error += rel_error;
                        if (abs_error > max_abs_error) max_abs_error = abs_error;
                        if (rel_error > max_rel_error) max_rel_error = rel_error;
                        valid_samples++;
                    }
                }
            }
        }

        printf("Mean absolute error: %.6f\n", total_abs_error / valid_samples);
        printf("Max absolute error: %.6f\n", max_abs_error);
        printf("Mean relative error: %.6f\n", total_rel_error / valid_samples);
        printf("Max relative error: %.6f\n", max_rel_error);
        
        // Scale statistics
        float mean_scale = 0.0f, std_scale = 0.0f;
        for (int i = 0; i < B * H * D; i++) {
            mean_scale += scale_data[i];
        }
        mean_scale /= (B * H * D);
        for (int i = 0; i < B * H * D; i++) {
            float diff = scale_data[i] - mean_scale;
            std_scale += diff * diff;
        }
        std_scale = sqrtf(std_scale / (B * H * D));
        
        printf("Scale statistics:\n");
        printf("  Min scale: %.6f\n", scale_min);
        printf("  Max scale: %.6f\n", scale_max_actual);
        printf("  Mean scale: %.6f\n", mean_scale);
        printf("  Std scale: %.6f\n", std_scale);

        free(pytorch_output);
        free(pytorch_scales);
    } else {
        printf("PyTorch reference results not found.\n");
        printf("Run: python /home/wlin1/Drawthings/ccv/test_scale_fuse_quant_cuda_reference.py\n");
        printf("This will generate the reference files:\n");
        printf("  /tmp/test_scale_fuse_input.bin\n");
        printf("  /tmp/pytorch_scale_fuse_output.bin\n");
        printf("  /tmp/pytorch_scale_fuse_scales.bin\n");

        if (pytorch_output_file) fclose(pytorch_output_file);
        if (pytorch_scale_file) fclose(pytorch_scale_file);
    }

    // Cleanup tensors
    ccv_nnc_tensor_free(input_tensor);
    ccv_nnc_tensor_free(input_tensor_f16);
    ccv_nnc_tensor_free(gpu_input_tensor);
    ccv_nnc_tensor_free(gpu_output_tensor);
    ccv_nnc_tensor_free(gpu_scale_tensor);
    ccv_nnc_tensor_free(cpu_output_tensor);
    ccv_nnc_tensor_free(cpu_scale_tensor);

    printf("✅ ccv_nnc_scale_fuse_quant_cuda_direct test completed successfully!\n");
#else
    printf("⚠️  Test skipped - CUDA SM80+ required for SageAttention\n");
#endif
}

TEST_CASE("ccv_nnc_mean_scale_fuse_quant_cuda_direct NHD test")
{
    ccv_cli_set_output_levels(CCV_CLI_VERBOSE);
    GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
                      ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));

    printf("=== CCV SageAttention ccv_nnc_mean_scale_fuse_quant_cuda_direct Test ===\n");

    // Test parameters matching Python reference script
    const int B = 1;      // batch size
    const int H = 8;      // number of heads  
    const int D = 128;    // head dimension
    const int S = 64;     // sequence length
    const float scale_max = 448.0f;  // E4M3 max scale
    const int tensor_layout = 0;     // 0=NHD, 1=HND

    printf("Test parameters: B=%d, H=%d, D=%d, S=%d\n", B, H, D, S);
    printf("Scale max: %.1f, tensor_layout: %d\n", scale_max, tensor_layout);
    printf("Input tensor shape: (%d, %d, %d, %d)\n", B, D, H, S);
    printf("Output tensor shape: (%d, %d, %d, %d)\n", B, D, H, S);
    printf("Mean tensor shape: (%d, %d, %d)\n", B, H, D);
    printf("Scale tensor shape: (%d, %d, %d)\n", B, H, D);

    // Load or create input tensor 
    ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, D, H, S), 0);
    
    FILE* input_file = fopen("/tmp/test_mean_scale_fuse_input.bin", "rb");
    if (input_file) {
        // Load from Python reference file
        uint16_t* input_fp16_data = (uint16_t*)malloc(B * D * H * S * sizeof(uint16_t));
        fread(input_fp16_data, sizeof(uint16_t), B * D * H * S, input_file);
        fclose(input_file);
        
        // Convert FP16 to FP32 for CCV tensor
        for (int i = 0; i < B * D * H * S; i++) {
            union { uint16_t u16; __fp16 f16; } converter;
            converter.u16 = input_fp16_data[i];
            input_tensor->data.f32[i] = (float)converter.f16;
        }
        free(input_fp16_data);
        printf("✓ Loaded input from /tmp/test_mean_scale_fuse_input.bin\n");
    } else {
        // Generate test data if reference not available
        dsfmt_t dsfmt;
        dsfmt_init_gen_rand(&dsfmt, 123);  // Different seed than scale_fuse_quant
        for (int i = 0; i < B * D * H * S; ++i) {
            input_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.2f;  // Range [-0.1, 0.1]
        }
        printf("✓ Generated test input data\n");
    }

    // Print input statistics
    float input_min = input_tensor->data.f32[0], input_max = input_tensor->data.f32[0];
    for (int i = 1; i < B * D * H * S; i++) {
        if (input_tensor->data.f32[i] < input_min) input_min = input_tensor->data.f32[i];
        if (input_tensor->data.f32[i] > input_max) input_max = input_tensor->data.f32[i];
    }
    printf("Input tensor range: [%.6f, %.6f]\n", input_min, input_max);
    printf("Input sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", input_tensor->data.f32[i]);
    }
    printf("\n");

    // Convert to FP16 for GPU processing
    ccv_nnc_tensor_t* const input_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, D, H, S), 0);
    ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 
                     TENSOR_LIST(input_tensor), TENSOR_LIST(input_tensor_f16), 0);

    ccv_nnc_tensor_t* const gpu_input_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, D, H, S), 0);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(input_tensor_f16), TENSOR_LIST(gpu_input_tensor), 0);

    // Create output tensors on GPU
    ccv_nnc_tensor_t* const gpu_output_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, D, H, S), 0);
    ccv_nnc_tensor_t* const gpu_mean_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, D, 1), 0);
    ccv_nnc_tensor_t* const gpu_scale_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, D, 1), 0);

    printf("\n");
    // Run direct quantization function
#ifdef HAVE_CUDA_SM80
    extern void ccv_nnc_mean_scale_fuse_quant_cuda_direct(
        __fp16 *input, int8_t *output, float *mean, float *scale,
        int input_dim[], int output_dim[], int mean_dim[], int scale_dim[],
        int input_stride[], int output_stride[], int mean_stride[], int scale_stride[],
        int num_tokens, float scale_max, int tensor_layout, cudaStream_t cuda_stream);

    printf("Calling ccv_nnc_mean_scale_fuse_quant_cuda_direct...\n");

    // Prepare dimension and stride arrays
    int input_dim[] = {B, D, H, S};       // Input shape: [B, D, H, S] (expected by mean_scale_fuse_quant)
    int output_dim[] = {B, D, H, S};      // Output shape: [B, D, H, S] 
    int mean_dim[] = {B, H, D};           // Mean shape: [B, H, D]
    int scale_dim[] = {B, H, D};          // Scale shape: [B, H, D]

    // Calculate strides for [B, D, H, S] layout
    int input_stride[] = {D * H * S, H * S, S, 1};
    int output_stride[] = {D * H * S, H * S, S, 1};
    int mean_stride[] = {H * D, D, 1};
    int scale_stride[] = {H * D, D, 1};

    ccv_nnc_mean_scale_fuse_quant_cuda_direct(
        (__fp16*)gpu_input_tensor->data.f16,   // Input tensor (FP16)
        (int8_t*)gpu_output_tensor->data.u8,   // Output tensor (INT8)
        (float*)gpu_mean_tensor->data.f32,     // Mean tensor (FP32)
        (float*)gpu_scale_tensor->data.f32,    // Scale tensor (FP32)
        input_dim, output_dim, mean_dim, scale_dim,      // Tensor dimensions
        input_stride, output_stride, mean_stride, scale_stride,  // Tensor strides
        S,                                      // num_tokens (actual sequence length)
        scale_max,                             // Maximum scale value
        tensor_layout,                         // Tensor layout (0=NHD)
        0                                      // CUDA stream (default)
    );

    cudaDeviceSynchronize();
    printf("✓ mean_scale_fuse_quant_cuda completed successfully\n");

    // Copy results back to CPU for verification
    ccv_nnc_tensor_t* const cpu_output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, D, H, S), 0);
    ccv_nnc_tensor_t* const cpu_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D, 1), 0);
    ccv_nnc_tensor_t* const cpu_scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D, 1), 0);

    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_output_tensor), TENSOR_LIST(cpu_output_tensor), 0);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_mean_tensor), TENSOR_LIST(cpu_mean_tensor), 0);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_scale_tensor), TENSOR_LIST(cpu_scale_tensor), 0);

    // Verify output - match Python reference script format
    int8_t* output_data = (int8_t*)cpu_output_tensor->data.u8;
    float* mean_data = (float*)cpu_mean_tensor->data.f32;
    float* scale_data = (float*)cpu_scale_tensor->data.f32;
    
    int output_size = B * D * H * S;
    int mean_size = B * H * D;
    int scale_size = B * H * D;
    
    // Calculate min/max for output
    int8_t min_output = output_data[0], max_output = output_data[0];
    for (int i = 0; i < output_size; i++) {
        if (output_data[i] < min_output) min_output = output_data[i];
        if (output_data[i] > max_output) max_output = output_data[i];
    }
    
    // Calculate min/max/mean for means and scales
    float min_mean = mean_data[0], max_mean = mean_data[0], mean_mean = 0.0f;
    for (int i = 0; i < mean_size; i++) {
        if (mean_data[i] < min_mean) min_mean = mean_data[i];
        if (mean_data[i] > max_mean) max_mean = mean_data[i];
        mean_mean += mean_data[i];
    }
    mean_mean /= mean_size;
    
    float min_scale = scale_data[0], max_scale = scale_data[0], mean_scale = 0.0f;
    for (int i = 0; i < scale_size; i++) {
        if (scale_data[i] < min_scale) min_scale = scale_data[i];
        if (scale_data[i] > max_scale) max_scale = scale_data[i];
        mean_scale += scale_data[i];
    }
    mean_scale /= scale_size;
    
    printf("Output range: [%d, %d]\n", min_output, max_output);
    printf("Output sample (first 10): ");
    for (int i = 0; i < 10 && i < output_size; i++) {
        printf("%d ", output_data[i]);
    }
    printf("\n");
    
    printf("Mean range: [%.6f, %.6f]\n", min_mean, max_mean);
    printf("Mean sample (first 5): ");
    for (int i = 0; i < 5 && i < mean_size; i++) {
        printf("%.6f ", mean_data[i]);
    }
    printf("\n");
    
    printf("Scale range: [%.6f, %.6f]\n", min_scale, max_scale);
    printf("Scale sample (first 5): ");
    for (int i = 0; i < 5 && i < scale_size; i++) {
        printf("%.6f ", scale_data[i]);
    }
    printf("\n");

    // Compare with PyTorch reference results
    printf("\n=== CCV vs PyTorch Reference Comparison ===\n");
    FILE* pytorch_output_file = fopen("/tmp/pytorch_mean_scale_fuse_output.bin", "rb");
    FILE* pytorch_mean_file = fopen("/tmp/pytorch_mean_scale_fuse_means.bin", "rb");
    FILE* pytorch_scale_file = fopen("/tmp/pytorch_mean_scale_fuse_scales.bin", "rb");

    if (pytorch_output_file && pytorch_mean_file && pytorch_scale_file) {
        // Load PyTorch reference results
        int8_t* pytorch_output = (int8_t*)malloc(B * D * H * S * sizeof(int8_t));
        float* pytorch_means = (float*)malloc(B * H * D * sizeof(float));
        float* pytorch_scales = (float*)malloc(B * H * D * sizeof(float));

        fread(pytorch_output, sizeof(int8_t), B * D * H * S, pytorch_output_file);
        fread(pytorch_means, sizeof(float), B * H * D, pytorch_mean_file);
        fread(pytorch_scales, sizeof(float), B * H * D, pytorch_scale_file);

        fclose(pytorch_output_file);
        fclose(pytorch_mean_file);
        fclose(pytorch_scale_file);

        // Compare quantized output values (first 20)
        printf("Output comparison (first 20 values):\n");
        for (int i = 0; i < 20; i++) {
            printf("  [%d] CCV: %d, PyTorch: %d %s\n", i, output_data[i], pytorch_output[i], 
                (output_data[i] == pytorch_output[i]) ? "✓" : "✗");
        }

        // Compare mean values
        printf("Mean comparison (first 10 values):\n");
        for (int i = 0; i < ccv_min(10, B * H * D); i++) {
            float diff = fabsf(mean_data[i] - pytorch_means[i]);
            printf("  [%d] CCV: %.6f, PyTorch: %.6f, diff: %.8f %s\n", i, 
                mean_data[i], pytorch_means[i], diff, (diff < 1e-6) ? "✓" : "✗");
        }

        // Compare scale values
        printf("Scale comparison (first 10 values):\n");
        for (int i = 0; i < ccv_min(10, B * H * D); i++) {
            float diff = fabsf(scale_data[i] - pytorch_scales[i]);
            printf("  [%d] CCV: %.6f, PyTorch: %.6f, diff: %.8f %s\n", i, 
                scale_data[i], pytorch_scales[i], diff, (diff < 1e-6) ? "✓" : "✗");
        }

        // Full comparison statistics
        int total_output_matches = 0, total_mean_close = 0, total_scale_close = 0;
        for (int i = 0; i < B * D * H * S; i++) {
            if (output_data[i] == pytorch_output[i]) total_output_matches++;
        }
        for (int i = 0; i < B * H * D; i++) {
            float mean_diff = fabsf(mean_data[i] - pytorch_means[i]);
            float scale_diff = fabsf(scale_data[i] - pytorch_scales[i]);
            if (mean_diff < 1e-6) total_mean_close++;
            if (scale_diff < 1e-6) total_scale_close++;
        }

        printf("\nFull comparison statistics:\n");
        printf("  Output exact matches: %d/%d (%.2f%%)\n", 
            total_output_matches, B * D * H * S, (100.0 * total_output_matches) / (B * D * H * S));
        printf("  Mean close matches: %d/%d (%.2f%%)\n", 
            total_mean_close, B * H * D, (100.0 * total_mean_close) / (B * H * D));
        printf("  Scale close matches: %d/%d (%.2f%%)\n", 
            total_scale_close, B * H * D, (100.0 * total_scale_close) / (B * H * D));

        if (total_output_matches == B * D * H * S && total_mean_close == B * H * D && total_scale_close == B * H * D) {
            printf("🎯 PERFECT: CCV mean_scale_fuse_quant matches PyTorch reference exactly!\n");
        } else if (total_output_matches > 0.95 * B * D * H * S && total_mean_close > 0.95 * B * H * D && total_scale_close > 0.95 * B * H * D) {
            printf("✅ GOOD: CCV mean_scale_fuse_quant is very close to PyTorch reference (>95%% match)\n");
        } else {
            printf("⚠️  WARNING: Significant differences between CCV and PyTorch quantization\n");
        }

        // Test quantization quality (dequantization)
        printf("\n=== Quantization Quality Analysis ===\n");
        double total_abs_error = 0.0, max_abs_error = 0.0;
        double total_rel_error = 0.0, max_rel_error = 0.0;
        int valid_samples = 0;

        // For tensor_layout=0 (NHD): output [B,D,H,S], scale [B,H,D], mean [B,H,D]
        // Need proper broadcasting: scale[b,h,d] -> scale for output[b,d,h,s]
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                for (int h = 0; h < H; h++) {
                    for (int s = 0; s < S; s++) {
                        int output_idx = b * D * H * S + d * H * S + h * S + s;
                        int scale_idx = b * H * D + h * D + d;
                        int input_idx = output_idx;  // Same indexing for [B,D,H,S] layout
                        
                        float original = input_tensor->data.f32[input_idx];
                        float dequantized = (float)output_data[output_idx] * scale_data[scale_idx] + mean_data[scale_idx];
                        
                        float abs_error = fabsf(dequantized - original);
                        float rel_error = abs_error / (fabsf(original) + 1e-8f);
                        
                        total_abs_error += abs_error;
                        total_rel_error += rel_error;
                        if (abs_error > max_abs_error) max_abs_error = abs_error;
                        if (rel_error > max_rel_error) max_rel_error = rel_error;
                        valid_samples++;
                    }
                }
            }
        }

        printf("Mean absolute error: %.6f\n", total_abs_error / valid_samples);
        printf("Max absolute error: %.6f\n", max_abs_error);
        printf("Mean relative error: %.6f\n", total_rel_error / valid_samples);
        printf("Max relative error: %.6f\n", max_rel_error);
        
        // Statistics matching Python script format
        float std_mean = 0.0f, std_scale = 0.0f;
        for (int i = 0; i < mean_size; i++) {
            float diff = mean_data[i] - mean_mean;
            std_mean += diff * diff;
        }
        std_mean = sqrtf(std_mean / mean_size);
        
        for (int i = 0; i < scale_size; i++) {
            float diff = scale_data[i] - mean_scale;
            std_scale += diff * diff;
        }
        std_scale = sqrtf(std_scale / scale_size);
        
        printf("Mean statistics:\n");
        printf("  Min mean: %.6f\n", min_mean);
        printf("  Max mean: %.6f\n", max_mean);
        printf("  Mean mean: %.6f\n", mean_mean);
        printf("  Std mean: %.6f\n", std_mean);
        
        printf("Scale statistics:\n");
        printf("  Min scale: %.6f\n", min_scale);
        printf("  Max scale: %.6f\n", max_scale);
        printf("  Mean scale: %.6f\n", mean_scale);
        printf("  Std scale: %.6f\n", std_scale);

        free(pytorch_output);
        free(pytorch_means);
        free(pytorch_scales);
    } else {
        printf("PyTorch reference results not found.\n");
        printf("Run: python /home/wlin1/Drawthings/ccv/test_mean_scale_fuse_quant_cuda_reference.py\n");
        printf("This will generate the reference files:\n");
        printf("  /tmp/test_mean_scale_fuse_input.bin\n");
        printf("  /tmp/pytorch_mean_scale_fuse_output.bin\n");
        printf("  /tmp/pytorch_mean_scale_fuse_means.bin\n");
        printf("  /tmp/pytorch_mean_scale_fuse_scales.bin\n");

        if (pytorch_output_file) fclose(pytorch_output_file);
        if (pytorch_mean_file) fclose(pytorch_mean_file);
        if (pytorch_scale_file) fclose(pytorch_scale_file);
    }

    // Cleanup tensors
    ccv_nnc_tensor_free(input_tensor);
    ccv_nnc_tensor_free(input_tensor_f16);
    ccv_nnc_tensor_free(gpu_input_tensor);
    ccv_nnc_tensor_free(gpu_output_tensor);
    ccv_nnc_tensor_free(gpu_mean_tensor);
    ccv_nnc_tensor_free(gpu_scale_tensor);
    ccv_nnc_tensor_free(cpu_output_tensor);
    ccv_nnc_tensor_free(cpu_mean_tensor);
    ccv_nnc_tensor_free(cpu_scale_tensor);

    printf("✅ ccv_nnc_mean_scale_fuse_quant_cuda_direct test completed successfully!\n");
#else
    printf("⚠️  Test skipped - CUDA SM80+ required for SageAttention\n");
#endif
}

// TEST_CASE("ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct test")
// {
// #ifdef HAVE_CUDA
//     ccv_cli_set_output_levels(CCV_CLI_VERBOSE);
//     GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
//                       ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));

//     printf("=== CCV SageAttention FP8 ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct Test ===\n");

//     // Test parameters matching Python reference script (HND format)
//     const int B = 1;        // batch size
//     const int H = 8;        // number of heads  
//     const int D = 128;      // head dimension
//     const int S = 64;       // sequence length
//     const int padded_S = 64; // padded sequence length (from reference)
//     const float scale = 0.08838834764831843f; // 1/sqrt(D) from reference
    
//     printf("Test dimensions: B=%d, H=%d, S=%d, D=%d, padded_S=%d\n", B, H, S, D, padded_S);
//     printf("Scale factor: %.6f\n", scale);

//     // Create quantized input tensors matching PyTorch reference data
//     // Q, K tensors: [B, H, S, D] int8 (HND format)
//     ccv_nnc_tensor_t* const q_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, S, D), 0);
//     ccv_nnc_tensor_t* const k_int8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, S, D), 0);
//     // V tensor: [B, H, D, padded_S] int8 (FP8 data stored as int8)
//     ccv_nnc_tensor_t* const v_fp8_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 8U, B, H, D, padded_S), 0);
    
//     // Scale tensors
//     // Q scales: [B, H, 4] - per_warp quantization scales
//     ccv_nnc_tensor_t* const q_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, 4), 0);
//     // K scales: [B, H, 1] - per_block quantization scales
//     ccv_nnc_tensor_t* const k_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, 1), 0);  
//     // V scales: [B, H, D] - per-element scales for FP8
//     ccv_nnc_tensor_t* const v_scales_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, D), 0);
    
//     // Output tensor: [B, H, S, D] fp16
//     ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, H, S, D), 0);

//     // Load reference quantized data from PyTorch files
//     printf("Loading quantized reference data from PyTorch...\n");
    
//     // Load quantized tensors to CPU first, then transfer to GPU
//     ccv_nnc_tensor_t* const cpu_q_int8 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, S, D), 0);
//     ccv_nnc_tensor_t* const cpu_k_int8 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, S, D), 0);
//     ccv_nnc_tensor_t* const cpu_v_fp8 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, B, H, D, padded_S), 0);
//     ccv_nnc_tensor_t* const cpu_q_scales = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, 4), 0);
//     ccv_nnc_tensor_t* const cpu_k_scales = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, 1), 0);
//     ccv_nnc_tensor_t* const cpu_v_scales = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, H, D), 0);

//     // Load data from PyTorch binary files
//     FILE* f_q = fopen("/tmp/pytorch_kernel_q_int8.bin", "rb");
//     FILE* f_k = fopen("/tmp/pytorch_kernel_k_int8.bin", "rb");
//     FILE* f_v = fopen("/tmp/pytorch_kernel_v_fp8.bin", "rb");
//     FILE* f_q_scales = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
//     FILE* f_k_scales = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
//     FILE* f_v_scales = fopen("/tmp/pytorch_kernel_v_scales.bin", "rb");
    
//     if (!f_q || !f_k || !f_v || !f_q_scales || !f_k_scales || !f_v_scales) {
//         printf("❌ Failed to open PyTorch reference files. Run test_fp8_attention_reference.py first.\n");
//         REQUIRE(0, "Reference data files not found");
//         return;
//     }
    
//     size_t read_q = fread(cpu_q_int8->data.u8, sizeof(uint8_t), B * H * S * D, f_q);
//     size_t read_k = fread(cpu_k_int8->data.u8, sizeof(uint8_t), B * H * S * D, f_k);
//     size_t read_v = fread(cpu_v_fp8->data.u8, sizeof(uint8_t), B * H * D * padded_S, f_v);
//     size_t read_q_scales = fread(cpu_q_scales->data.f32, sizeof(float), B * H * 4, f_q_scales);
//     size_t read_k_scales = fread(cpu_k_scales->data.f32, sizeof(float), B * H * 1, f_k_scales);
//     size_t read_v_scales = fread(cpu_v_scales->data.f32, sizeof(float), B * H * D, f_v_scales);
    
//     fclose(f_q); fclose(f_k); fclose(f_v);
//     fclose(f_q_scales); fclose(f_k_scales); fclose(f_v_scales);
    
//     printf("Loaded data sizes: Q=%zu, K=%zu, V=%zu, Q_scales=%zu, K_scales=%zu, V_scales=%zu\n",
//            read_q, read_k, read_v, read_q_scales, read_k_scales, read_v_scales);
    
//     REQUIRE_EQ(read_q, B * H * S * D, "Q tensor size mismatch");
//     REQUIRE_EQ(read_k, B * H * S * D, "K tensor size mismatch");
//     REQUIRE_EQ(read_v, B * H * D * padded_S, "V tensor size mismatch");
//     REQUIRE_EQ(read_q_scales, B * H * 4, "Q scales size mismatch");
//     REQUIRE_EQ(read_k_scales, B * H * 1, "K scales size mismatch");
//     REQUIRE_EQ(read_v_scales, B * H * D, "V scales size mismatch");

//     // Transfer data to GPU
//     ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
//                      TENSOR_LIST(cpu_q_int8, cpu_k_int8, cpu_v_fp8, cpu_q_scales, cpu_k_scales, cpu_v_scales), 
//                      TENSOR_LIST(q_int8_tensor, k_int8_tensor, v_fp8_tensor, q_scales_tensor, k_scales_tensor, v_scales_tensor), 0);

//     // Print sample values for verification
//     printf("Sample quantized values (GPU tensors):\n");
//     ccv_nnc_tensor_t* const temp_q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, 1, 1, 1, 5), 0);
//     ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
//                      TENSOR_LIST(q_int8_tensor), TENSOR_LIST(temp_q), 0);
//     printf("  Q_int8[0,0,0,0:5]: [%u,%u,%u,%u,%u]\n", 
//            temp_q->data.u8[0], temp_q->data.u8[1], temp_q->data.u8[2], temp_q->data.u8[3], temp_q->data.u8[4]);
//     ccv_nnc_tensor_free(temp_q);

//     // Call the direct FP8 SageAttention kernel with FP32 accumulation
//     printf("Calling qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct...\n");
    
//     // Setup tensor dimensions and strides for the direct kernel call
//     int qdim[4] = {B, H, S, D};
//     int kdim[4] = {B, H, S, D}; 
//     int vdim[4] = {B, H, D, padded_S};  // V has transposed/padded layout
//     int odim[4] = {B, H, S, D};
//     int qscale_dim[3] = {B, H, 4};      // per_warp scales
//     int kscale_dim[3] = {B, H, 1};      // per_block scales
//     int vscale_dim[3] = {B, H, D};      // per-element scales
    
//     // For HND layout, strides are computed as [batch_stride, head_stride, seq_stride, dim_stride]
//     int qstride[4] = {H*S*D, S*D, D, 1};
//     int kstride[4] = {H*S*D, S*D, D, 1};
//     int vstride[4] = {H*D*padded_S, D*padded_S, padded_S, 1};  // V layout
//     int ostride[4] = {H*S*D, S*D, D, 1};
//     int qscale_stride[3] = {H*4, 4, 1};
//     int kscale_stride[3] = {H*1, 1, 1};
//     int vscale_stride[3] = {H*D, D, 1};
    
//     qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
//         (int8_t*)q_int8_tensor->data.u8,  // Q: [B, H, S, D] int8
//         (int8_t*)k_int8_tensor->data.u8,  // K: [B, H, S, D] int8  
//         (int8_t*)v_fp8_tensor->data.u8,   // V: [B, H, D, padded_S] int8 (FP8 data)
//         (half*)output_tensor->data.f16,   // Output: [B, H, S, D] fp16
//         q_scales_tensor->data.f32,        // Q_scales: [B, H, 4] fp32
//         k_scales_tensor->data.f32,        // K_scales: [B, H, 1] fp32
//         v_scales_tensor->data.f32,        // V_scales: [B, H, D] fp32
//         qdim, kdim, vdim, odim,           // Tensor dimensions
//         qscale_dim, kscale_dim, vscale_dim, // Scale dimensions
//         qstride, kstride, vstride, ostride, // Tensor strides
//         qscale_stride, kscale_stride, vscale_stride, // Scale strides
//         1,                                // tensor_layout (1 = HND)
//         0,                                // is_causal (0 = False)
//         2,                                // qk_quant_gran (2 = per_warp)
//         scale,                            // sm_scale
//         0                                 // return_lse (0 = False)
//     );
    
//     int result = 0;  // Assume success since direct kernel call doesn't return status
    
//     if (result != 0) {
//         printf("❌ ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct failed with code: %d\n", result);
//         REQUIRE_EQ(result, 0, "Direct FP8 kernel should execute successfully");
//         return;
//     }
    
//     printf("✅ Direct FP8 kernel executed successfully!\n");
    
//     // Transfer output back to CPU for comparison
//     ccv_nnc_tensor_t* const cpu_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, S, D), 0);
//     ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, 
//                      TENSOR_LIST(output_tensor), TENSOR_LIST(cpu_output), 0);

//     // Print sample output values
//     printf("CCV kernel output samples:\n");
//     printf("First 5 output values:\n");
//     for (int i = 0; i < 5; i++) {
//         printf("  output[%d]: %f\n", i, (float)cpu_output->data.f16[i]);
//     }
//     printf("Sample outputs from different positions:\n");
//     printf("  output[0,0,0,0]: %f\n", (float)cpu_output->data.f16[0]);
//     printf("  output[0,0,0,64]: %f\n", (float)cpu_output->data.f16[64]);
//     printf("  output[0,0,0,127]: %f\n", (float)cpu_output->data.f16[127]);

//     // Load PyTorch reference output for comparison
//     FILE* f_ref = fopen("/tmp/pytorch_sage_output_nhd.bin", "rb");
//     if (f_ref) {
//         // PyTorch output is in NHD format: [B, S, H, D], need to convert to HND: [B, H, S, D]
//         ccv_nnc_tensor_t* const ref_output_nhd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, S, H, D), 0);
//         ccv_nnc_tensor_t* const ref_output_hnd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, H, S, D), 0);
        
//         size_t read_ref = fread(ref_output_nhd->data.f16, sizeof(uint16_t), B * S * H * D, f_ref);
//         fclose(f_ref);
        
//         if (read_ref == B * S * H * D) {
//             // Convert NHD to HND: [B, S, H, D] -> [B, H, S, D]
//             for (int b = 0; b < B; b++) {
//                 for (int s = 0; s < S; s++) {
//                     for (int h = 0; h < H; h++) {
//                         for (int d = 0; d < D; d++) {
//                             int nhd_idx = ((b * S + s) * H + h) * D + d;
//                             int hnd_idx = ((b * H + h) * S + s) * D + d;
//                             ref_output_hnd->data.f16[hnd_idx] = ref_output_nhd->data.f16[nhd_idx];
//                         }
//                     }
//                 }
//             }
            
//             printf("PyTorch reference output samples (converted to HND):\n");
//             printf("First 5 PyTorch values:\n");
//             for (int i = 0; i < 5; i++) {
//                 printf("  pytorch[%d]: %f\n", i, (float)ref_output_hnd->data.f16[i]);
//             }
//             printf("PyTorch sample positions:\n");
//             printf("  pytorch[0,0,0,0]: %f\n", (float)ref_output_hnd->data.f16[0]);
//             printf("  pytorch[0,0,0,64]: %f\n", (float)ref_output_hnd->data.f16[64]);
//             printf("  pytorch[0,0,0,127]: %f\n", (float)ref_output_hnd->data.f16[127]);
            
//             // Compare outputs
//             double max_diff = 0.0, sum_diff = 0.0;
//             int total_elements = B * H * S * D;
//             int exact_matches = 0;
            
//             for (int i = 0; i < total_elements; i++) {
//                 float ccv_val = (float)cpu_output->data.f16[i];
//                 float pytorch_val = (float)ref_output_hnd->data.f16[i];
//                 double diff = fabs((double)(ccv_val - pytorch_val));
                
//                 if (diff == 0.0) exact_matches++;
//                 if (diff > max_diff) max_diff = diff;
//                 sum_diff += diff;
//             }
            
//             printf("\n=== Comparison with PyTorch SageAttention Reference ===\n");
//             printf("Total elements: %d\n", total_elements);
//             printf("Exact matches: %d (%.2f%%)\n", exact_matches, 100.0 * exact_matches / total_elements);
//             printf("Maximum difference: %.8f\n", max_diff);
//             printf("Average difference: %.8f\n", sum_diff / total_elements);
            
//             if (max_diff < 1e-6) {
//                 printf("🎯 PERFECT MATCH! CCV and PyTorch outputs are virtually identical!\n");
//             } else if (max_diff < 1e-4) {
//                 printf("✅ EXCELLENT: CCV and PyTorch outputs are very close (within expected precision)\n");
//             } else if (max_diff < 1e-3) {
//                 printf("✅ GOOD: CCV and PyTorch outputs are close (within FP8 quantization tolerance)\n");
//             } else if (max_diff < 5e-3) {
//                 printf("⚠️  ACCEPTABLE: CCV and PyTorch outputs have small differences but within tolerance\n");
//             } else {
//                 printf("❌ FAILED: CCV and PyTorch outputs differ significantly\n");
//             }
            
//             // Require reasonable match (allow for FP8 quantization differences)
//             REQUIRE_EQ_WITH_TOLERANCE(max_diff, 0, 5e-3, "CCV FP8 kernel should match PyTorch reference within tolerance");
            
//         } else {
//             printf("⚠️  Failed to load PyTorch reference output, skipping comparison\n");
//         }
        
//         ccv_nnc_tensor_free(ref_output_nhd);
//         ccv_nnc_tensor_free(ref_output_hnd);
//     } else {
//         printf("⚠️  PyTorch reference output not found, skipping comparison\n");
//     }

//     // Cleanup
//     ccv_nnc_tensor_free(q_int8_tensor);
//     ccv_nnc_tensor_free(k_int8_tensor);
//     ccv_nnc_tensor_free(v_fp8_tensor);
//     ccv_nnc_tensor_free(q_scales_tensor);
//     ccv_nnc_tensor_free(k_scales_tensor);
//     ccv_nnc_tensor_free(v_scales_tensor);
//     ccv_nnc_tensor_free(output_tensor);
//     ccv_nnc_tensor_free(cpu_q_int8);
//     ccv_nnc_tensor_free(cpu_k_int8);
//     ccv_nnc_tensor_free(cpu_v_fp8);
//     ccv_nnc_tensor_free(cpu_q_scales);
//     ccv_nnc_tensor_free(cpu_k_scales);
//     ccv_nnc_tensor_free(cpu_v_scales);
//     ccv_nnc_tensor_free(cpu_output);

//     printf("✅ ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct test completed successfully!\n");
// #else
//     printf("⚠️  Test skipped - CUDA required for SageAttention FP8\n");
// #endif
// }

TEST_CASE("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct test")
{
#ifdef HAVE_CUDA
    GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
        ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));

    printf("\n=== CCV SageAttention FP8 Direct Function Test - HND Layout (FP32 accum) ===\n");
    printf("Testing qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct function with HND layout\n");
    printf("This test validates the direct FP8 kernel function using exact quantized data from PyTorch\n");

    // Use same test parameters as other sage attention tests
    int B = 1, H = 8, S = 64, D = 128;  // HND format: Batch, Heads, Seq, Dim
    int padded_S = 64;  // V tensor padding from PyTorch quantization
    float sm_scale = 0.08838834764831843f;  // From PyTorch reference

    printf("Test dimensions (HND): B=%d, H=%d, S=%d, D=%d, padded_S=%d\n", B, H, S, D, padded_S);
    printf("Scale: %f\n", sm_scale);

    printf("Loading PyTorch quantized inputs from binary files (HND format)...\n");

    // Allocate host memory for input data
    int8_t* q_int8_hnd = (int8_t*)malloc(B * H * S * D * sizeof(int8_t));
    int8_t* k_int8_hnd = (int8_t*)malloc(B * H * S * D * sizeof(int8_t)); 
    int8_t* v_fp8_hnd = (int8_t*)malloc(B * H * D * padded_S * sizeof(int8_t));  // FP8 as int8
    float* q_scales_hnd = (float*)malloc(B * H * 4 * sizeof(float));  // per-warp scales
    float* k_scales_hnd = (float*)malloc(B * H * 1 * sizeof(float));  // per-block scales
    float* v_scales_hnd = (float*)malloc(B * H * D * sizeof(float));  // per-element scales

    // Load quantized data from PyTorch files
    FILE* f;
    f = fopen("/tmp/pytorch_kernel_q_int8.bin", "rb");
    REQUIRE(f != NULL, "Failed to open Q int8 data file");
    fread(q_int8_hnd, sizeof(int8_t), B * H * S * D, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_k_int8.bin", "rb");
    REQUIRE(f != NULL, "Failed to open K int8 data file");
    fread(k_int8_hnd, sizeof(int8_t), B * H * S * D, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_v_fp8.bin", "rb");
    REQUIRE(f != NULL, "Failed to open V fp8 data file");
    fread(v_fp8_hnd, sizeof(int8_t), B * H * D * padded_S, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open Q scales data file");
    fread(q_scales_hnd, sizeof(float), B * H * 4, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open K scales data file");
    fread(k_scales_hnd, sizeof(float), B * H * 1, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_v_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open V scales data file");
    fread(v_scales_hnd, sizeof(float), B * H * D, f);
    fclose(f);

    printf("✅ Loaded PyTorch quantized data from disk (HND format)\n");

    // Print sample values for verification
    printf("Sample values (HND): Q_int8[0]=%d, K_int8[0]=%d, V_fp8[0]=%d\n", 
        q_int8_hnd[0], k_int8_hnd[0], v_fp8_hnd[0]);
    printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f, V_scale[0]=%f\n", 
        q_scales_hnd[0], k_scales_hnd[0], v_scales_hnd[0]);

    // Allocate GPU memory
    int8_t* gpu_q_int8;
    int8_t* gpu_k_int8;
    int8_t* gpu_v_fp8;
    __fp16* gpu_o_fp16;
    float* gpu_q_scales;
    float* gpu_k_scales;
    float* gpu_v_scales;

    cudaMalloc(&gpu_q_int8, B * H * S * D * sizeof(int8_t));
    cudaMalloc(&gpu_k_int8, B * H * S * D * sizeof(int8_t));
    cudaMalloc(&gpu_v_fp8, B * H * D * padded_S * sizeof(int8_t));
    cudaMalloc(&gpu_o_fp16, B * H * S * D * sizeof(__fp16));
    cudaMalloc(&gpu_q_scales, B * H * 4 * sizeof(float));
    cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));
    cudaMalloc(&gpu_v_scales, B * H * D * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(gpu_q_int8, q_int8_hnd, B * H * S * D * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k_int8, k_int8_hnd, B * H * S * D * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v_fp8, v_fp8_hnd, B * H * D * padded_S * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_q_scales, q_scales_hnd, B * H * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k_scales, k_scales_hnd, B * H * 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v_scales, v_scales_hnd, B * H * D * sizeof(float), cudaMemcpyHostToDevice);

    printf("✅ Allocated GPU memory and copied input data (HND format)\n");

    // Set up tensor dimensions and strides for HND layout
    int qdim[4] = {B, H, S, D};  // HND format
    int kdim[4] = {B, H, S, D};
    int vdim[4] = {B, H, D, padded_S};  // V has transposed layout
    int odim[4] = {B, H, S, D};
    int qscale_dim[3] = {B, H, 4};  // per-warp scales
    int kscale_dim[3] = {B, H, 1};  // per-block scales
    int vscale_dim[3] = {B, H, D};  // per-element scales

    // HND strides: [B, H, S, D] and [B, H, D, padded_S] for V
    int qstride[4] = {H*S*D, S*D, D, 1};
    int kstride[4] = {H*S*D, S*D, D, 1};
    int vstride[4] = {H*D*padded_S, D*padded_S, padded_S, 1};
    int ostride[4] = {H*S*D, S*D, D, 1};
    int qscale_stride[3] = {H*4, 4, 1};
    int kscale_stride[3] = {H*1, 1, 1};
    int vscale_stride[3] = {H*D, D, 1};

    printf("\n=== Calling Direct FP8 SageAttention Function with HND Layout (FP32 accum) ===\n");
    printf("Function parameters:\n");
    printf("  tensor_layout: 1 (HND)\n");
    printf("  is_causal: 0 (false)\n");
    printf("  qk_quant_gran: 2 (per_warp)\n");
    printf("  sm_scale: %f\n", sm_scale);
    printf("  return_lse: 0 (false)\n");

    printf("\nTensor dimensions (HND):\n");
    printf("  Q: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        qdim[0], qdim[1], qdim[2], qdim[3], qstride[0], qstride[1], qstride[2], qstride[3]);
    printf("  K: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        kdim[0], kdim[1], kdim[2], kdim[3], kstride[0], kstride[1], kstride[2], kstride[3]);
    printf("  V: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        vdim[0], vdim[1], vdim[2], vdim[3], vstride[0], vstride[1], vstride[2], vstride[3]);
    printf("  O: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        odim[0], odim[1], odim[2], odim[3], ostride[0], ostride[1], ostride[2], ostride[3]);
	
	extern void qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, int8_t *V, __fp16 *O,
    float *Q_scale, float *K_scale, float *V_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[], int vscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[], int vscale_stride[],
    int tensor_layout,
    float sm_scale);
    // Call the direct function with HND layout
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
        gpu_q_int8, gpu_k_int8, gpu_v_fp8, gpu_o_fp16,
        gpu_q_scales, gpu_k_scales, gpu_v_scales,
        qdim, kdim, vdim, odim,
        qscale_dim, kscale_dim, vscale_dim,
        qstride, kstride, vstride, ostride,
        qscale_stride, kscale_stride, vscale_stride,
        1,      // tensor_layout (1=HND)
        sm_scale
    );

    printf("✅ Direct FP8 function call with HND layout completed successfully\n");
    // Copy output back to host
    __fp16* o_fp16_hnd = (__fp16*)malloc(B * H * S * D * sizeof(__fp16));
    cudaMemcpy(o_fp16_hnd, gpu_o_fp16, B * H * S * D * sizeof(__fp16), cudaMemcpyDeviceToHost);

    printf("✅ Kernel executed successfully!\n");
    printf("Output shape: torch.Size([%d, %d, %d, %d])\n", B, H, S, D);
    
    // Print first 5 output values matching Python test format
    printf("Output[0,0,0,:5]: tensor([");
    for (int i = 0; i < 5; i++) {
        printf("%f%s", (float)o_fp16_hnd[i], i < 4 ? ", " : "], device='cuda:0',\n       dtype=torch.float16)\n");
    }
    printf("Output[0,0,0,0]: %f\n", (float)o_fp16_hnd[0]);

    // Save output for comparison
    f = fopen("/tmp/ccv_direct_fp8_hnd_output.bin", "wb");
    if (f) {
        fwrite(o_fp16_hnd, sizeof(__fp16), B * H * S * D, f);
        fclose(f);
        printf("\nSaved direct kernel output (HND) to /tmp/ccv_direct_fp8_hnd_output.bin\n");
    }

    printf("\n=== Direct Kernel vs SageAttention Reference ===\n");
    printf("Output shape: torch.Size([%d, %d, %d, %d]) (HND format)\n", B, H, S, D);
    // Load direct PyTorch kernel output for comparison
    f = fopen("/tmp/direct_kernel_output_hnd.bin", "rb");
    if (f) {
        __fp16* pytorch_hnd_data = (__fp16*)malloc(B * H * S * D * sizeof(__fp16));
        fread(pytorch_hnd_data, sizeof(__fp16), B * H * S * D, f);
        fclose(f);

        // Skip the first 10 values comparison to match Python output

        // Full tensor comparison
        int total_elements = B * H * S * D;
        int exact_matches = 0;
        double max_diff = 0.0;
        double sum_diff = 0.0;

        for (int i = 0; i < total_elements; i++) {
            double diff = fabs((float)o_fp16_hnd[i] - (float)pytorch_hnd_data[i]);
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
            if (diff < 1e-6) exact_matches++;
        }

        printf("Max difference: %.1f\n", max_diff);
        printf("Average difference: %.1f\n", sum_diff / total_elements);

        if (max_diff < 1e-6) {
            printf("✅ PERFECT MATCH! Direct kernel produces same output as high-level SageAttention\n");
        } else if (max_diff < 0.01) {
            printf("✅ Excellent match - differences within expected FP8 quantization error\n");
        } else if (max_diff < 0.1) {
            printf("🟡 Good match - reasonable quantization differences\n");
        } else {
            printf("❌ Large differences detected\n");
        }

        // Sample position comparisons matching Python output format
        printf("\nSample position comparisons (HND format):\n");
        
        // Define the sample positions as in Python test
        int positions[][4] = {
            {0, 0, 0, 0},
            {0, 0, 0, 64},
            {0, 0, 0, 127},
            {0, 0, 32, 0},
            {0, 4, 0, 0}
        };
        
        for (int p = 0; p < 5; p++) {
            int b = positions[p][0];
            int h = positions[p][1];
            int s = positions[p][2];
            int d = positions[p][3];
            
            if (h < H && s < S && d < D) {
                int idx = b * (H * S * D) + h * (S * D) + s * D + d;
                float ccv_val = (float)o_fp16_hnd[idx];
                float pytorch_val = (float)pytorch_hnd_data[idx];
                float diff_val = fabs(ccv_val - pytorch_val);
                printf("  [%d,%d,%d,%d] Direct: %.6f, Reference: %.6f, diff: %.8f\n",
                    b, h, s, d, ccv_val, pytorch_val, diff_val);
            }
        }

        free(pytorch_hnd_data);
    } else {
        printf("PyTorch direct kernel output not found. Run test_fp8_attention_kernel_direct.py first.\n");
    }

    // Cleanup GPU memory
    cudaFree(gpu_q_int8);
    cudaFree(gpu_k_int8);
    cudaFree(gpu_v_fp8);
    cudaFree(gpu_o_fp16);
    cudaFree(gpu_q_scales);
    cudaFree(gpu_k_scales);
    cudaFree(gpu_v_scales);

    // Cleanup host memory
    free(q_int8_hnd);
    free(k_int8_hnd);
    free(v_fp8_hnd);
    free(q_scales_hnd);
    free(k_scales_hnd);
    free(v_scales_hnd);
    free(o_fp16_hnd);

    printf("✅ SageAttention FP8 direct function test (FP32 accum) completed successfully!\n");
#else
    printf("⚠️  Test skipped - CUDA required for SageAttention FP8\n");
#endif
}

TEST_CASE("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct NHD test")
{
#ifdef HAVE_CUDA
    GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
        ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));

    printf("\n=== CCV SageAttention FP8 Direct Function Test - NHD Layout (FP32 accum) ===\n");
    printf("Testing qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct function with NHD layout\n");
    printf("This test validates the direct FP8 kernel function using exact quantized data from PyTorch\n");

    // Load test parameters from PyTorch NHD test
    FILE* f = fopen("/tmp/pytorch_kernel_params_nhd.txt", "r");
    REQUIRE(f, "Parameters file not found. Run test_fp8_attention_kernel_direct_NHD.py first.");

    int B, S, H, D, padded_S;
    float scale, scale_max;
    char line[256];
    char key[128], value[128];

    while (fgets(line, sizeof(line), f)) {
        sscanf(line, "%[^=]=%s", key, value);
        if (strcmp(key, "batch_size") == 0) B = atoi(value);
        else if (strcmp(key, "seq_len") == 0) S = atoi(value);
        else if (strcmp(key, "num_heads") == 0) H = atoi(value);
        else if (strcmp(key, "head_dim") == 0) D = atoi(value);
        else if (strcmp(key, "padded_len") == 0) padded_S = atoi(value);
        else if (strcmp(key, "scale") == 0) scale = atof(value);
        else if (strcmp(key, "scale_max") == 0) scale_max = atof(value);
    }
    fclose(f);

    printf("Test dimensions (NHD): B=%d, S=%d, H=%d, D=%d, padded_S=%d\n", B, S, H, D, padded_S);
    printf("Scale: %f\n", scale);

    printf("Loading PyTorch quantized inputs from binary files (NHD format)...\n");

    // Allocate host memory for input data (NHD layout)
    int8_t* q_int8_nhd = (int8_t*)malloc(B * S * H * D * sizeof(int8_t));
    int8_t* k_int8_nhd = (int8_t*)malloc(B * S * H * D * sizeof(int8_t)); 
    int8_t* v_fp8_nhd = (int8_t*)malloc(B * D * H * padded_S * sizeof(int8_t));  // V in NHD: [B, D, H, padded_S]
    float* q_scales_nhd = (float*)malloc(B * H * 4 * sizeof(float));  // per-warp scales
    float* k_scales_nhd = (float*)malloc(B * H * 1 * sizeof(float));  // per-block scales
    float* v_scales_nhd = (float*)malloc(B * H * D * sizeof(float));  // per-element scales

    // Load quantized data from PyTorch NHD files
    f = fopen("/tmp/pytorch_kernel_q_int8_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open Q int8 NHD data file");
    fread(q_int8_nhd, sizeof(int8_t), B * S * H * D, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_k_int8_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open K int8 NHD data file");
    fread(k_int8_nhd, sizeof(int8_t), B * S * H * D, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_v_fp8_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open V fp8 NHD data file");
    fread(v_fp8_nhd, sizeof(int8_t), B * D * H * padded_S, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_q_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open Q scales data file");
    fread(q_scales_nhd, sizeof(float), B * H * 4, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_k_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open K scales data file");
    fread(k_scales_nhd, sizeof(float), B * H * 1, f);
    fclose(f);

    f = fopen("/tmp/pytorch_kernel_v_scales.bin", "rb");
    REQUIRE(f != NULL, "Failed to open V scales data file");
    fread(v_scales_nhd, sizeof(float), B * H * D, f);
    fclose(f);

    printf("✅ Loaded PyTorch quantized data from disk (NHD format)\n");

    // Print sample values for verification
    printf("Sample values (NHD): Q_int8[0]=%d, K_int8[0]=%d, V_fp8[0]=%d\n", 
        q_int8_nhd[0], k_int8_nhd[0], v_fp8_nhd[0]);
    printf("Sample scales: Q_scale[0]=%f, K_scale[0]=%f, V_scale[0]=%f\n", 
        q_scales_nhd[0], k_scales_nhd[0], v_scales_nhd[0]);

    // Allocate GPU memory
    int8_t* gpu_q_int8;
    int8_t* gpu_k_int8;
    int8_t* gpu_v_fp8;
    __fp16* gpu_o_fp16;
    float* gpu_q_scales;
    float* gpu_k_scales;
    float* gpu_v_scales;

    cudaMalloc(&gpu_q_int8, B * S * H * D * sizeof(int8_t));
    cudaMalloc(&gpu_k_int8, B * S * H * D * sizeof(int8_t));
    cudaMalloc(&gpu_v_fp8, B * D * H * padded_S * sizeof(int8_t));
    cudaMalloc(&gpu_o_fp16, B * S * H * D * sizeof(__fp16));
    cudaMalloc(&gpu_q_scales, B * H * 4 * sizeof(float));
    cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));
    cudaMalloc(&gpu_v_scales, B * H * D * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(gpu_q_int8, q_int8_nhd, B * S * H * D * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k_int8, k_int8_nhd, B * S * H * D * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v_fp8, v_fp8_nhd, B * D * H * padded_S * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_q_scales, q_scales_nhd, B * H * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k_scales, k_scales_nhd, B * H * 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v_scales, v_scales_nhd, B * H * D * sizeof(float), cudaMemcpyHostToDevice);

    printf("✅ Allocated GPU memory and copied input data (NHD format)\n");

    // Set up tensor dimensions and strides for NHD layout
    int qdim[4] = {B, S, H, D};  // NHD format: [B, S, H, D]
    int kdim[4] = {B, S, H, D};
    int vdim[4] = {B, D, H, padded_S};  // V has layout [B, D, H, padded_S]
    int odim[4] = {B, S, H, D};
    int qscale_dim[3] = {B, H, 4};  // per-warp scales
    int kscale_dim[3] = {B, H, 1};  // per-block scales
    int vscale_dim[3] = {B, H, D};  // per-element scales

    // NHD strides: [B, S, H, D] and [B, D, H, padded_S] for V
    int qstride[4] = {S*H*D, H*D, D, 1};
    int kstride[4] = {S*H*D, H*D, D, 1};
    int vstride[4] = {D*H*padded_S, H*padded_S, padded_S, 1};
    int ostride[4] = {S*H*D, H*D, D, 1};
    int qscale_stride[3] = {H*4, 4, 1};
    int kscale_stride[3] = {H*1, 1, 1};
    int vscale_stride[3] = {H*D, D, 1};

    printf("\n=== Calling Direct FP8 SageAttention Function with NHD Layout (FP32 accum) ===\n");
    printf("Function parameters:\n");
    printf("  tensor_layout: 0 (NHD)\n");
    printf("  is_causal: 0 (false)\n");
    printf("  qk_quant_gran: 2 (per_warp)\n");
    printf("  sm_scale: %f\n", scale);
    printf("  return_lse: 0 (false)\n");

    printf("\nTensor dimensions (NHD):\n");
    printf("  Q: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        qdim[0], qdim[1], qdim[2], qdim[3], qstride[0], qstride[1], qstride[2], qstride[3]);
    printf("  K: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        kdim[0], kdim[1], kdim[2], kdim[3], kstride[0], kstride[1], kstride[2], kstride[3]);
    printf("  V: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        vdim[0], vdim[1], vdim[2], vdim[3], vstride[0], vstride[1], vstride[2], vstride[3]);
    printf("  O: [%d, %d, %d, %d], strides: [%d, %d, %d, %d]\n", 
        odim[0], odim[1], odim[2], odim[3], ostride[0], ostride[1], ostride[2], ostride[3]);
    
    extern void qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
    int8_t *Q, int8_t *K, int8_t *V, __fp16 *O,
    float *Q_scale, float *K_scale, float *V_scale,
    int qdim[], int kdim[], int vdim[], int odim[], 
    int qscale_dim[], int kscale_dim[], int vscale_dim[],
    int qstride[], int kstride[], int vstride[], int ostride[],
    int qscale_stride[], int kscale_stride[], int vscale_stride[],
    int tensor_layout,
    float sm_scale);
    
    // Call the direct function with NHD layout
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_direct(
        gpu_q_int8, gpu_k_int8, gpu_v_fp8, gpu_o_fp16,
        gpu_q_scales, gpu_k_scales, gpu_v_scales,
        qdim, kdim, vdim, odim,
        qscale_dim, kscale_dim, vscale_dim,
        qstride, kstride, vstride, ostride,
        qscale_stride, kscale_stride, vscale_stride,
        0,      // tensor_layout (0=NHD)
        scale
    );

    printf("✅ Direct FP8 function call with NHD layout completed successfully\n");
    
    // Copy output back to host
    __fp16* o_fp16_nhd = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
    cudaMemcpy(o_fp16_nhd, gpu_o_fp16, B * S * H * D * sizeof(__fp16), cudaMemcpyDeviceToHost);

    printf("✅ Kernel executed successfully!\n");
    printf("Output shape: torch.Size([%d, %d, %d, %d]) (NHD format)\n", B, S, H, D);
    
    // Print first 5 output values matching Python test format
    printf("Output[0,0,0,:5]: tensor([");
    for (int i = 0; i < 5; i++) {
        printf("%f%s", (float)o_fp16_nhd[i], i < 4 ? ", " : "], device='cuda:0',\n       dtype=torch.float16)\n");
    }
    printf("Output[0,0,0,0]: %f\n", (float)o_fp16_nhd[0]);

    // Save output for comparison
    f = fopen("/tmp/ccv_direct_fp8_nhd_output.bin", "wb");
    if (f) {
        fwrite(o_fp16_nhd, sizeof(__fp16), B * S * H * D, f);
        fclose(f);
        printf("\nSaved direct kernel output (NHD) to /tmp/ccv_direct_fp8_nhd_output.bin\n");
    }

    printf("\n=== Direct Kernel vs SageAttention Reference ===\n");
    printf("Output shape: torch.Size([%d, %d, %d, %d]) (NHD format)\n", B, S, H, D);
    
    // Load direct PyTorch kernel output for comparison
    f = fopen("/tmp/direct_kernel_output_nhd.bin", "rb");
    if (f) {
        __fp16* pytorch_nhd_data = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
        fread(pytorch_nhd_data, sizeof(__fp16), B * S * H * D, f);
        fclose(f);

        // Full tensor comparison
        int total_elements = B * S * H * D;
        int exact_matches = 0;
        double max_diff = 0.0;
        double sum_diff = 0.0;

        for (int i = 0; i < total_elements; i++) {
            double diff = fabs((float)o_fp16_nhd[i] - (float)pytorch_nhd_data[i]);
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
            if (diff < 1e-6) exact_matches++;
        }

        printf("Max difference: %.1f\n", max_diff);
        printf("Average difference: %.1f\n", sum_diff / total_elements);

        if (max_diff < 1e-6) {
            printf("✅ PERFECT MATCH! Direct kernel produces same output as high-level SageAttention\n");
        } else if (max_diff < 0.01) {
            printf("✅ Excellent match - differences within expected FP8 quantization error\n");
        } else if (max_diff < 0.1) {
            printf("🟡 Good match - reasonable quantization differences\n");
        } else {
            printf("❌ Large differences detected\n");
        }

        // Sample position comparisons matching Python output format (NHD indices)
        printf("\nSample position comparisons (NHD format):\n");
        
        // Define the sample positions for NHD format [B, S, H, D]
        int positions[][4] = {
            {0, 0, 0, 0},
            {0, 0, 0, 64},
            {0, 0, 0, 127},
            {0, 32, 0, 0},
            {0, 0, 4, 0}
        };
        
        for (int p = 0; p < 5; p++) {
            int b = positions[p][0];
            int s = positions[p][1]; 
            int h = positions[p][2];
            int d = positions[p][3];
            
            if (h < H && s < S && d < D) {
                int idx = b * (S * H * D) + s * (H * D) + h * D + d;  // NHD indexing
                float ccv_val = (float)o_fp16_nhd[idx];
                float pytorch_val = (float)pytorch_nhd_data[idx];
                float diff_val = fabs(ccv_val - pytorch_val);
                printf("  [%d,%d,%d,%d] Direct: %.6f, Reference: %.6f, diff: %.8f\n",
                    b, s, h, d, ccv_val, pytorch_val, diff_val);
            }
        }

        free(pytorch_nhd_data);
    } else {
        printf("PyTorch direct kernel output not found. Run test_fp8_attention_kernel_direct_NHD.py first.\n");
    }

    // Cleanup GPU memory
    cudaFree(gpu_q_int8);
    cudaFree(gpu_k_int8);
    cudaFree(gpu_v_fp8);
    cudaFree(gpu_o_fp16);
    cudaFree(gpu_q_scales);
    cudaFree(gpu_k_scales);
    cudaFree(gpu_v_scales);

    // Cleanup host memory
    free(q_int8_nhd);
    free(k_int8_nhd);
    free(v_fp8_nhd);
    free(q_scales_nhd);
    free(k_scales_nhd);
    free(v_scales_nhd);
    free(o_fp16_nhd);

    printf("✅ SageAttention FP8 direct function test (FP32 accum, NHD layout) completed successfully!\n");
#else
    printf("⚠️  Test skipped - CUDA required for SageAttention FP8\n");
#endif
}

TEST_CASE("ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct NHD test")
{
#ifdef HAVE_CUDA
    GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));

    printf("\n=== CCV ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct Test - NHD Layout ===\n");
    printf("Testing ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct function with NHD layout\n");
    printf("This test validates the FP8 SageAttention implementation using reference data from PyTorch\n");

    // Test dimensions
    int B = 1, S = 64, H = 8, D = 128;
    float scale = 1.0f / sqrtf((float)D);
    
    printf("Test dimensions (NHD): B=%d, S=%d, H=%d, D=%d\n", B, S, H, D);
    printf("Scale (1/sqrt(d)): %f\n", scale);

    // Load input data from test_fp8_direct_vs_cpu.py output files (NHD format)
    printf("\nLoading test input data (NHD format)...\n");
    
    // Allocate host memory for FP16 input data (NHD layout: [B, S, H, D])
    __fp16* q_fp16_nhd = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
    __fp16* k_fp16_nhd = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
    __fp16* v_fp16_nhd = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
    
    FILE* f = fopen("/tmp/test_fp8_q_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open Q NHD data file. Run test_fp8_direct_vs_cpu.py first.");
    fread(q_fp16_nhd, sizeof(__fp16), B * S * H * D, f);
    fclose(f);
    
    f = fopen("/tmp/test_fp8_k_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open K NHD data file");
    fread(k_fp16_nhd, sizeof(__fp16), B * S * H * D, f);
    fclose(f);
    
    f = fopen("/tmp/test_fp8_v_nhd.bin", "rb");
    REQUIRE(f != NULL, "Failed to open V NHD data file");
    fread(v_fp16_nhd, sizeof(__fp16), B * S * H * D, f);
    fclose(f);
    
    printf("✅ Loaded input data from disk (NHD format)\n");
    printf("Sample values (NHD): Q[0]=%f, K[0]=%f, V[0]=%f\n", 
        (float)q_fp16_nhd[0], (float)k_fp16_nhd[0], (float)v_fp16_nhd[0]);

    // Allocate GPU memory for inputs
    __fp16* gpu_q_fp16;
    __fp16* gpu_k_fp16;
    __fp16* gpu_v_fp16;
    __fp16* gpu_k_mean;  // K mean for quantization
    __fp16* gpu_output;
    
    cudaMalloc(&gpu_q_fp16, B * S * H * D * sizeof(__fp16));
    cudaMalloc(&gpu_k_fp16, B * S * H * D * sizeof(__fp16));
    cudaMalloc(&gpu_v_fp16, B * S * H * D * sizeof(__fp16));
    cudaMalloc(&gpu_k_mean, B * S * H * sizeof(__fp16));  // Mean per token
    cudaMalloc(&gpu_output, B * S * H * D * sizeof(__fp16));
    
    // Allocate GPU memory for quantized tensors
    int8_t* gpu_q_int8;
    int8_t* gpu_k_int8;
    int8_t* gpu_v_fp8;
    
    // Calculate padded sequence length for V (must be multiple of 64)
    int padded_S = ((S + 63) / 64) * 64;
    
    cudaMalloc(&gpu_q_int8, B * S * H * D * sizeof(int8_t));
    cudaMalloc(&gpu_k_int8, B * S * H * D * sizeof(int8_t));
    cudaMalloc(&gpu_v_fp8, B * D * H * padded_S * sizeof(int8_t));  // V layout: [B, D, H, padded_S]
    
    // Allocate GPU memory for scales
    float* gpu_q_scales;
    float* gpu_k_scales;
    float* gpu_v_scales;
    
    cudaMalloc(&gpu_q_scales, B * H * 4 * sizeof(float));  // per-warp scales (128/32 = 4 warps)
    cudaMalloc(&gpu_k_scales, B * H * 1 * sizeof(float));  // per-block scales
    cudaMalloc(&gpu_v_scales, B * H * D * sizeof(float));  // per-channel scales
    
    // Copy input data to GPU
    cudaMemcpy(gpu_q_fp16, q_fp16_nhd, B * S * H * D * sizeof(__fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k_fp16, k_fp16_nhd, B * S * H * D * sizeof(__fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v_fp16, v_fp16_nhd, B * S * H * D * sizeof(__fp16), cudaMemcpyHostToDevice);
    
    // Initialize k_mean to zeros (will be computed by the function)
    cudaMemset(gpu_k_mean, 0, B * S * H * sizeof(__fp16));
    
    printf("✅ Allocated GPU memory and copied input data\n");
    printf("Padded sequence length for V: %d\n", padded_S);
    
    // Set up tensor dimensions and strides for NHD layout
    int qdim[4] = {B, S, H, D};  // NHD format
    int kdim[4] = {B, S, H, D};
    int vdim[4] = {B, S, H, D};  // V input still in NHD format
    int odim[4] = {B, S, H, D};
    
    int q_int8_dim[4] = {B, S, H, D};
    int k_int8_dim[4] = {B, S, H, D};
    int v_fp8_dim[4] = {B, D, H, padded_S};  // V after transpose_pad_permute: [B, D, H, padded_S]
    
    int qscale_dim[3] = {B, H, 4};  // per-warp scales
    int kscale_dim[3] = {B, H, 1};  // per-block scales
    int vscale_dim[3] = {B, H, D};  // per-channel scales
    int km_dim[3] = {B, S, H};      // K mean dimensions
    
    // NHD strides
    int qstride[4] = {S*H*D, H*D, D, 1};
    int kstride[4] = {S*H*D, H*D, D, 1};
    int vstride[4] = {S*H*D, H*D, D, 1};  // V input strides for NHD [B, S, H, D]
    int ostride[4] = {S*H*D, H*D, D, 1};
    
    int q_int8_stride[4] = {S*H*D, H*D, D, 1};
    int k_int8_stride[4] = {S*H*D, H*D, D, 1};
    int v_fp8_stride[4] = {D*H*padded_S, H*padded_S, padded_S, 1};  // V output strides for [B, D, H, padded_S]
    
    int qscale_stride[3] = {H*4, 4, 1};
    int kscale_stride[3] = {H*1, 1, 1};
    int vscale_stride[3] = {H*D, D, 1};
    int km_stride[3] = {S*H, H, 1};
    
    printf("\n=== Calling ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct ===\n");
    printf("Function parameters:\n");
    printf("  tensor_layout: 0 (NHD)\n");
    printf("  is_causal: 0 (false)\n");
    printf("  qk_quant_gran: 2 (per_warp)\n");
    printf("  pv_accum_dtype: 2 (fp32+fp32)\n");
    printf("  sm_scale: %f\n", scale);
    printf("  return_lse: 0 (false)\n");
    
    // Import the function
    extern void ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct(
        __fp16 *query, __fp16 *key, __fp16 *k_mean, __fp16 *value,
        int8_t *q_int8, int8_t *k_int8, int8_t *v_fp8,
        float *query_scale, float *key_scale, float *value_scale,
        __fp16 *output,
        int qdim[], int kdim[], int vdim[], int odim[],
        int q_int8_dim[], int k_int8_dim[], int v_fp8_dim[],
        int qscale_dim[], int kscale_dim[], int vscale_dim[],
        int qstride[], int kstride[], int vstride[], int ostride[],
        int q_int8_stride[], int k_int8_stride[], int v_fp8_stride[],
        int qscale_stride[], int kscale_stride[], int vscale_stride[],
        int tensor_layout, int is_causal, int qk_quant_gran,
        float sm_scale, int return_lse, int pv_accum_dtype,
        int km_dim[], int km_stride[], cudaStream_t cuda_stream);
    
    // Call the function
    ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct(
        gpu_q_fp16, gpu_k_fp16, gpu_k_mean, gpu_v_fp16,
        gpu_q_int8, gpu_k_int8, gpu_v_fp8,
        gpu_q_scales, gpu_k_scales, gpu_v_scales,
        gpu_output,
        qdim, kdim, vdim, odim,
        q_int8_dim, k_int8_dim, v_fp8_dim,
        qscale_dim, kscale_dim, vscale_dim,
        qstride, kstride, vstride, ostride,
        q_int8_stride, k_int8_stride, v_fp8_stride,
        qscale_stride, kscale_stride, vscale_stride,
        0,     // tensor_layout (0=NHD)
        0,     // is_causal (false)
        2,     // qk_quant_gran (2=per_warp)
        scale, // sm_scale
        0,     // return_lse (false)
        2,     // pv_accum_dtype (2=fp32+fp32)
        km_dim, km_stride,
        0      // cuda_stream (default stream)
    );
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    printf("✅ ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct executed successfully\n");
    
    // Copy output back to host
    __fp16* output_nhd = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
    cudaMemcpy(output_nhd, gpu_output, B * S * H * D * sizeof(__fp16), cudaMemcpyDeviceToHost);
    
    // DEBUG: Print intermediate quantized results to compare with PyTorch
    printf("\n=== DEBUG: Checking Intermediate Quantized Results ===\n");
    
    // Copy quantized tensors back to host for debugging
    int8_t* debug_q_int8 = (int8_t*)malloc(B * S * H * D * sizeof(int8_t));
    int8_t* debug_k_int8 = (int8_t*)malloc(B * S * H * D * sizeof(int8_t));
    int8_t* debug_v_fp8 = (int8_t*)malloc(B * D * H * padded_S * sizeof(int8_t));
    float* debug_q_scales = (float*)malloc(B * H * 4 * sizeof(float));
    float* debug_k_scales = (float*)malloc(B * H * 1 * sizeof(float));
    float* debug_v_scales = (float*)malloc(B * H * D * sizeof(float));
    
    cudaMemcpy(debug_q_int8, gpu_q_int8, B * S * H * D * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_k_int8, gpu_k_int8, B * S * H * D * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_v_fp8, gpu_v_fp8, B * D * H * padded_S * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_q_scales, gpu_q_scales, B * H * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_k_scales, gpu_k_scales, B * H * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_v_scales, gpu_v_scales, B * H * D * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Q_int8 sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", debug_q_int8[i]);
    }
    printf("\n");
    
    printf("K_int8 sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", debug_k_int8[i]);
    }
    printf("\n");
    
    printf("V_fp8 sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", debug_v_fp8[i]);
    }
    printf("\n");
    
    printf("Q_scales sample (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", debug_q_scales[i]);
    }
    printf("\n");
    
    printf("K_scales sample (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", debug_k_scales[i]);
    }
    printf("\n");
    
    printf("V_scales sample (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", debug_v_scales[i]);
    }
    printf("\n");
    
    // Save debug quantized results
    FILE* debug_file = fopen("/tmp/ccv_debug_q_int8.bin", "wb");
    if (debug_file) {
        fwrite(debug_q_int8, sizeof(int8_t), B * S * H * D, debug_file);
        fclose(debug_file);
    }
    debug_file = fopen("/tmp/ccv_debug_k_int8.bin", "wb");
    if (debug_file) {
        fwrite(debug_k_int8, sizeof(int8_t), B * S * H * D, debug_file);
        fclose(debug_file);
    }
    debug_file = fopen("/tmp/ccv_debug_v_fp8.bin", "wb");
    if (debug_file) {
        fwrite(debug_v_fp8, sizeof(int8_t), B * D * H * padded_S, debug_file);
        fclose(debug_file);
    }
    debug_file = fopen("/tmp/ccv_debug_q_scales.bin", "wb");
    if (debug_file) {
        fwrite(debug_q_scales, sizeof(float), B * H * 4, debug_file);
        fclose(debug_file);
    }
    debug_file = fopen("/tmp/ccv_debug_k_scales.bin", "wb");
    if (debug_file) {
        fwrite(debug_k_scales, sizeof(float), B * H * 1, debug_file);
        fclose(debug_file);
    }
    debug_file = fopen("/tmp/ccv_debug_v_scales.bin", "wb");
    if (debug_file) {
        fwrite(debug_v_scales, sizeof(float), B * H * D, debug_file);
        fclose(debug_file);
    }
    printf("Saved CCV debug quantized results to /tmp/ccv_debug_*.bin\n");
    
    free(debug_q_int8);
    free(debug_k_int8);
    free(debug_v_fp8);
    free(debug_q_scales);
    free(debug_k_scales);
    free(debug_v_scales);
    
    printf("\nOutput shape: [%d, %d, %d, %d] (NHD format)\n", B, S, H, D);
    
    // Print first 5 output values
    printf("Output[0,0,0,:5]: [");
    for (int i = 0; i < 5; i++) {
        printf("%f%s", (float)output_nhd[i], i < 4 ? ", " : "]\n");
    }
    printf("Output[0,0,0,0]: %f\n", (float)output_nhd[0]);
    
    // Save output for comparison
    f = fopen("/tmp/ccv_fp8_output_nhd.bin", "wb");
    if (f) {
        fwrite(output_nhd, sizeof(__fp16), B * S * H * D, f);
        fclose(f);
        printf("\nSaved CCV output to /tmp/ccv_fp8_output_nhd.bin\n");
    }
    
    // Load reference outputs for comparison
    printf("\n=== Comparison with Reference Outputs ===\n");
    
    // Compare with PyTorch SageAttention FP8 output
    f = fopen("/tmp/test_fp8_output_nhd.bin", "rb");
    if (f) {
        __fp16* pytorch_fp8_output = (__fp16*)malloc(B * S * H * D * sizeof(__fp16));
        fread(pytorch_fp8_output, sizeof(__fp16), B * S * H * D, f);
        fclose(f);
        
        double max_diff = 0.0;
        double sum_diff = 0.0;
        int total_elements = B * S * H * D;
        
        for (int i = 0; i < total_elements; i++) {
            double diff = fabs((float)output_nhd[i] - (float)pytorch_fp8_output[i]);
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
        }
        
        printf("\nvs PyTorch SageAttention FP8:\n");
        printf("  Max difference: %f\n", max_diff);
        printf("  Mean difference: %f\n", sum_diff / total_elements);
        
        if (max_diff < 0.001) {
            printf("  ✅ PERFECT MATCH with PyTorch SageAttention FP8!\n");
        } else if (max_diff < 0.01) {
            printf("  ✅ Excellent match - minimal differences\n");
        } else if (max_diff < 0.1) {
            printf("  ✅ Good match - acceptable quantization differences\n");
        } else {
            printf("  ⚠️ Larger differences detected\n");
        }
        
        // Sample position comparisons
        printf("\n  Sample comparisons (NHD [B,S,H,D]):\n");
        int positions[][4] = {{0,0,0,0}, {0,0,0,64}, {0,0,0,127}, {0,32,0,0}, {0,0,4,0}};
        
        for (int p = 0; p < 5; p++) {
            int s = positions[p][1];
            int h = positions[p][2];
            int d = positions[p][3];
            if (s < S && h < H && d < D) {
                int idx = s * H * D + h * D + d;
                printf("    [0,%d,%d,%d] CCV: %.6f, PyTorch: %.6f, diff: %.8f\n",
                    s, h, d, (float)output_nhd[idx], (float)pytorch_fp8_output[idx],
                    fabs((float)output_nhd[idx] - (float)pytorch_fp8_output[idx]));
            }
        }
        
        free(pytorch_fp8_output);
    } else {
        printf("PyTorch FP8 output not found. Run test_fp8_direct_vs_cpu.py first.\n");
    }
    
    // Compare with CPU reference
    f = fopen("/tmp/test_fp8_cpu_reference_nhd.bin", "rb");
    if (f) {
        float* cpu_reference = (float*)malloc(B * S * H * D * sizeof(float));
        fread(cpu_reference, sizeof(float), B * H * S * D, f);  // CPU ref is in HND format
        fclose(f);
        
        double max_diff = 0.0;
        double sum_diff = 0.0;
        int total_elements = B * S * H * D;
        
        // Need to compare with proper indexing since CPU ref is HND
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < H; h++) {
                    for (int d = 0; d < D; d++) {
                        int nhd_idx = b * S * H * D + s * H * D + h * D + d;
                        int hnd_idx = b * H * S * D + h * S * D + s * D + d;
                        double diff = fabs((float)output_nhd[nhd_idx] - cpu_reference[hnd_idx]);
                        sum_diff += diff;
                        if (diff > max_diff) max_diff = diff;
                    }
                }
            }
        }
        
        printf("\nvs CPU Reference (FP32):\n");
        printf("  Max difference: %f\n", max_diff);
        printf("  Mean difference: %f\n", sum_diff / total_elements);
        
        if (max_diff > 0.1) {
            printf("  ⚠️ Expected larger differences vs CPU due to quantization\n");
        } else {
            printf("  ✅ Reasonable quantization error vs CPU reference\n");
        }
        
        free(cpu_reference);
    }
    
    // Cleanup GPU memory
    cudaFree(gpu_q_fp16);
    cudaFree(gpu_k_fp16);
    cudaFree(gpu_v_fp16);
    cudaFree(gpu_k_mean);
    cudaFree(gpu_output);
    cudaFree(gpu_q_int8);
    cudaFree(gpu_k_int8);
    cudaFree(gpu_v_fp8);
    cudaFree(gpu_q_scales);
    cudaFree(gpu_k_scales);
    cudaFree(gpu_v_scales);
    
    // Cleanup host memory
    free(q_fp16_nhd);
    free(k_fp16_nhd);
    free(v_fp16_nhd);
    free(output_nhd);
    
    printf("\n✅ ccv_nnc_sageattn_qk_int8_pv_fp8_cuda_direct NHD test completed successfully!\n");
#else
    printf("⚠️  Test skipped - CUDA required for SageAttention FP8\n");
#endif
}

#include "case_main.h"
