#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <sys/time.h>
#include <ctype.h>

static double get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


int main(int argc, char** argv)
{
	ccv_nnc_init();
	// Bypass error: variable-sized object may not be initialized
#define num_trials 15 // 18
	int B_candidates[num_trials] = {  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int R_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384  };
	int C_candidates[num_trials] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024, 2048, 4096, 1024, 2048, 3072, 4096, 6144, 8192, 16384 };
	int Hq_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int Hk_candidates[num_trials] = {   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
	int D_candidates[num_trials] = {  64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 128, 128, 128, 128, 128, 128, 128 };
	int is_causal_candidates[num_trials] = {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

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
		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		for (int i = 0; i < B * R * Hq * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);
		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_t scaled_dot_product_attention = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal);
		// scaled_dot_product_attention.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
		for (int i = 0; i < 5; i++)
			ccv_nnc_cmd_exec(scaled_dot_product_attention, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
		double elapsed_time = get_current_time();
		for (int i = 0; i < 40; i++)
			ccv_nnc_cmd_exec(scaled_dot_product_attention, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
		elapsed_time = get_current_time() - elapsed_time;
		printf("%d, %d, %d, %d, %d, %d, %2.3f\n", B, R, C, Hq, Hk, D, elapsed_time);

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
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
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_trials
	return 0;
}
