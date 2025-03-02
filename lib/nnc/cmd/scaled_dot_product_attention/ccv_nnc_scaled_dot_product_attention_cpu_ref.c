#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 3 ? (ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const w = input_size > 4 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 5 ? (ccv_nnc_tensor_view_t*)inputs[5] : 0;
	if (bias) // bias always requires a weight matrix.
		{ assert(w); }
	ccv_nnc_tensor_view_t* const c = (w) ? (ccv_nnc_tensor_view_t*)outputs[2] : (ccv_nnc_tensor_view_t*)outputs[0];
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(c_nd == 3 || c_nd == 4);
	assert(q_nd == k_nd && k_nd == v_nd && v_nd == c_nd);
	// Assuming this is float 32.
	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int cdim[CCV_NNC_MAX_DIM_ALLOC];
	int amdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(c, cdim);
	if (q_nd == 3)
	{
		qdim[0] = qdim[1], qdim[1] = qdim[2], qdim[2] = 1;
		kdim[0] = kdim[1], kdim[1] = kdim[2], kdim[2] = 1;
		vdim[0] = vdim[1], vdim[1] = vdim[2], vdim[2] = 1;
		cdim[0] = cdim[1], cdim[1] = cdim[2], cdim[2] = 1;
	}
	assert(qdim[0] == kdim[0] && kdim[0] == vdim[0] && vdim[0] == cdim[0]);
	assert(qdim[2] == cdim[2]);
	assert(kdim[2] == vdim[2]);
	assert(qdim[2] % kdim[2] == 0);
	assert(qdim[2] >= kdim[2]);
	assert(qdim[3] == kdim[3]);
	assert(kdim[1] == vdim[1]);
	assert(cdim[1] == qdim[1]);
	assert(cdim[3] == vdim[3]);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	int amstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(c, cstride);
	if (q_nd == 3)
	{
		qstride[0] = qstride[1], qstride[1] = qstride[2], qstride[2] = qstride[3];
		kstride[0] = kstride[1], kstride[1] = kstride[2], kstride[2] = kstride[3];
		vstride[0] = vstride[1], vstride[1] = vstride[2], vstride[2] = vstride[3];
		cstride[0] = cstride[1], cstride[1] = cstride[2], cstride[2] = cstride[3];
	}
	if (attn_mask)
	{
		ccv_nnc_tensor_view_get_dim(attn_mask, amdim);
		ccv_nnc_tensor_view_get_stride(attn_mask, amstride);
		assert(amdim[0] == qdim[0] || amdim[0] == 1);
		assert(amdim[1] == qdim[2] || amdim[1] == 1);
		assert(amdim[2] == qdim[1]);
		assert(amdim[3] == kdim[1]);
	}
	int i[CCV_NNC_MAX_DIM + 2];
	float* qk = ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * qdim[1] * kdim[1], CCV_TENSOR_CPU_MEMORY);
	const float* const qp = q->data.f32;
	const float* const kp = k->data.f32;
	const float* const vp = v->data.f32;
	const float* const amp = attn_mask ? attn_mask->data.f32 : 0;
	float* const cp = c->data.f32;
	const float scale = cmd.info.scaled_dot_product_attention.scale;
	const int is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	const int h_h_k_ratio = qdim[2] / kdim[2];
	assert(kdim[2] == vdim[2]);
	assert(qdim[2] >= kdim[2]);
	assert(qdim[2] % kdim[2] == 0);
	for (i[0] = 0; i[0] < qdim[0]; i[0]++)
	{
		const float* const qp0 = qp + i[0] * qstride[0];
		const float* const kp0 = kp + i[0] * kstride[0];
		const float* const vp0 = vp + i[0] * vstride[0];
		const float* const amp0 = amp && amdim[0] > 1 ? amp + i[0] * amstride[0] : amp;
		float* const cp0 = cp + i[0] * cstride[0];
		for (i[1] = 0; i[1] < qdim[2]; i[1]++)
		{
			const float* const qp1 = qp0 + i[1] * qstride[2];
			const float* const kp1 = kp0 + (i[1] / h_h_k_ratio) * kstride[2];
			const float* const vp1 = vp0 + (i[1] / h_h_k_ratio) * vstride[2];
			const float* const amp1 = amp && amdim[1] > 1 ? amp0 + i[1] * amstride[1] : amp0;
			float* const cp1 = cp0 + i[1] * cstride[2];
			// Compute Q @ K^T
			parallel_for(x, qdim[1]) {
				int y, k;
				const float* const qp2 = qp1 + x * qstride[1];
				float* const cp2 = cp1 + x * cstride[1];
				float* const qk0 = qk + x * kdim[1];
				const float* const amp2 = amp1 ? amp1 + x * amstride[2] : 0;
				if (attn_mask)
				{
					for (y = 0; y < kdim[1]; y++)
					{
						const float* const kp2 = kp1 + y * kstride[1];
						float v = 0;
						for (k = 0; k < qdim[3]; k++)
							v += qp2[k * qstride[3]] * kp2[k * kstride[3]];
						qk0[y] = scale * v + amp2[y * amstride[3]];
					}
				} else {
					for (y = 0; y < kdim[1]; y++)
					{
						const float* const kp2 = kp1 + y * kstride[1];
						float v = 0;
						for (k = 0; k < qdim[3]; k++)
							v += qp2[k * qstride[3]] * kp2[k * kstride[3]];
						qk0[y] = scale * v;
					}
				}
				// Compute softmax on qk.
				if (is_causal)
				{
					const int x_end = ccv_max(x - qdim[1] + kdim[1] + 1, 0);
					for (y = x_end; y < kdim[1]; y++)
						qk0[y] = 0;
					double maxval = qk0[0];
					for (y = 1; y < x_end; y++)
						if (qk0[y] > maxval)
							maxval = qk0[y];
					double sumval = 0;
					for (y = 0; y < x_end; y++)
						sumval += (qk0[y] = expf(qk0[y] - maxval));
					sumval = 1.0 / sumval;
					for (y = 0; y < x_end; y++)
						qk0[y] *= sumval;
				} else {
					double maxval = qk0[0];
					for (y = 1; y < kdim[1]; y++)
						if (qk0[y] > maxval)
							maxval = qk0[y];
					double sumval = 0;
					for (y = 0; y < kdim[1]; y++)
						sumval += (qk0[y] = expf(qk0[y] - maxval));
					sumval = 1.0 / sumval;
					for (y = 0; y < kdim[1]; y++)
						qk0[y] *= sumval;
				}
				for (k = 0; k < vdim[3]; k++)
					cp2[k * cstride[3]] = 0;
				for (y = 0; y < kdim[1]; y++)
				{
					const float* const vp2 = vp1 + y * vstride[1];
					const float v = qk0[y];
					for (k = 0; k < vdim[3]; k++)
						cp2[k * cstride[3]] += v * vp2[k * vstride[3]];
				}
			} parallel_endfor
		}
	}
	if (w)
	{
		const int num_heads = cdim[2];
		ccv_nnc_tensor_view_t* const d = (ccv_nnc_tensor_view_t*)outputs[0];
		const int w_nd = ccv_nnc_tensor_nd(w->info.dim);
		assert(w_nd == 2);
		assert(CCV_IS_TENSOR_CONTIGUOUS(w));
		const int d_nd = ccv_nnc_tensor_nd(d->info.dim);
		assert(d_nd == 3);
		int ddim[CCV_NNC_MAX_DIM_ALLOC];
		int dstride[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_dim(d, ddim);
		ccv_nnc_tensor_view_get_stride(d, dstride);
		assert(ddim[2] == cdim[1]);
		assert(ddim[3] == num_heads * cdim[3]);
		assert(w->info.dim[1] == ddim[3]);
		assert(w->info.dim[0] == ddim[3]);
		float* const dp = d->data.f32;
		const float* const wp = w->data.f32;
		const float* const cp = c->data.f32;
		if (bias)
		{
			assert(ccv_nnc_tensor_count(bias->info) == ddim[3]);
			assert(CCV_IS_TENSOR_CONTIGUOUS(bias));
			const float* const biasp = bias->data.f32;
			for (i[0] = 0; i[0] < ddim[1]; i[0]++)
			{
				const float* const cp0 = cp + i[0] * cstride[0];
				float* const dp0 = dp + i[0] * dstride[1];
				parallel_for(y, ddim[2]) {
					int x, j, k;
					const float* const cp1 = cp0 + y * cstride[1];
					float* const dp1 = dp0 + y * dstride[2];
					for (x = 0; x < ddim[3]; x++)
					{
						const float* const wp0 = wp + x * ddim[3];
						float v = biasp[x];
						for (j = 0; j < num_heads; j++)
						{
							const float* const cp2 = cp1 + j * cstride[2];
							for (k = 0; k < cdim[3]; k++)
								v += wp0[j * cdim[3] + k] * cp2[k * cstride[3]];
						}
						dp1[x * dstride[3]] = v;
					}
				} parallel_endfor
			}
		} else {
			for (i[0] = 0; i[0] < ddim[1]; i[0]++)
			{
				const float* const cp0 = cp + i[0] * cstride[0];
				float* const dp0 = dp + i[0] * dstride[1];
				parallel_for(y, ddim[2]) {
					int x, j, k;
					const float* const cp1 = cp0 + y * cstride[1];
					float* const dp1 = dp0 + y * dstride[2];
					for (x = 0; x < ddim[3]; x++)
					{
						const float* const wp0 = wp + x * ddim[3];
						float v = 0;
						for (j = 0; j < num_heads; j++)
						{
							const float* const cp2 = cp1 + j * cstride[2];
							for (k = 0; k < cdim[3]; k++)
								v += wp0[j * cdim[3] + k] * cp2[k * cstride[3]];
						}
						dp1[x * dstride[3]] = v;
					}
				} parallel_endfor
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming no saved_softmax, we need to recompute from q, k, v.
	// We cannot do this with masks (yet).
	assert(input_size >= 6);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const dq = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dk = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dv = (ccv_nnc_tensor_view_t*)outputs[2];
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 3 || g_nd == 4);
	const int dq_nd = ccv_nnc_tensor_nd(dq->info.dim);
	assert(dq_nd == 3 || dq_nd == 4);
	assert(dq_nd == q_nd);
	const int dk_nd = ccv_nnc_tensor_nd(dk->info.dim);
	assert(dk_nd == 3 || dk_nd == 4);
	assert(dk_nd == k_nd);
	const int dv_nd = ccv_nnc_tensor_nd(dv->info.dim);
	assert(dv_nd == 3 || dv_nd == 4);
	assert(dv_nd == v_nd);
	assert(q_nd == k_nd && k_nd == v_nd && v_nd == g_nd);
	// Assuming this is float 32.
	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int dqdim[CCV_NNC_MAX_DIM_ALLOC];
	int dkdim[CCV_NNC_MAX_DIM_ALLOC];
	int dvdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_dim(dq, dqdim);
	ccv_nnc_tensor_view_get_dim(dk, dkdim);
	ccv_nnc_tensor_view_get_dim(dv, dvdim);
	if (q_nd == 3)
	{
		qdim[0] = qdim[1], qdim[1] = qdim[2], qdim[2] = 1;
		kdim[0] = kdim[1], kdim[1] = kdim[2], kdim[2] = 1;
		vdim[0] = vdim[1], vdim[1] = vdim[2], vdim[2] = 1;
		gdim[0] = gdim[1], gdim[1] = gdim[2], gdim[2] = 1;
		dqdim[0] = dqdim[1], dqdim[1] = dqdim[2], dqdim[2] = 1;
		dkdim[0] = dkdim[1], dkdim[1] = dkdim[2], dkdim[2] = 1;
		dvdim[0] = dvdim[1], dvdim[1] = dvdim[2], dvdim[2] = 1;
	}
	assert(qdim[0] == kdim[0] && kdim[0] == vdim[0] && vdim[0] == gdim[0]);
	assert(qdim[2] == gdim[2]);
	assert(kdim[2] == vdim[2]);
	assert(qdim[2] % kdim[2] == 0);
	assert(qdim[2] >= kdim[2]);
	assert(qdim[3] == kdim[3]);
	assert(kdim[1] == vdim[1]);
	assert(gdim[1] == qdim[1]);
	assert(gdim[3] == vdim[3]);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int dqstride[CCV_NNC_MAX_DIM_ALLOC];
	int dkstride[CCV_NNC_MAX_DIM_ALLOC];
	int dvstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(g, gstride);
	ccv_nnc_tensor_view_get_stride(dq, dqstride);
	ccv_nnc_tensor_view_get_stride(dk, dkstride);
	ccv_nnc_tensor_view_get_stride(dv, dvstride);
	if (q_nd == 3)
	{
		qstride[0] = qstride[1], qstride[1] = qstride[2], qstride[2] = qstride[3];
		kstride[0] = kstride[1], kstride[1] = kstride[2], kstride[2] = kstride[3];
		vstride[0] = vstride[1], vstride[1] = vstride[2], vstride[2] = vstride[3];
		gstride[0] = gstride[1], gstride[1] = gstride[2], gstride[2] = gstride[3];
		dqstride[0] = dqstride[1], dqstride[1] = dqstride[2], dqstride[2] = dqstride[3];
		dkstride[0] = dkstride[1], dkstride[1] = dkstride[2], dkstride[2] = dkstride[3];
		dvstride[0] = dvstride[1], dvstride[1] = dvstride[2], dvstride[2] = dvstride[3];
	}
	int i[CCV_NNC_MAX_DIM + 2];
	float* qk = ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * 2 * kdim[1], CCV_TENSOR_CPU_MEMORY);
	const float* const qp = q->data.f32;
	const float* const kp = k->data.f32;
	const float* const vp = v->data.f32;
	const float* const gp = g->data.f32;
	float* const dqp = dq->data.f32;
	float* const dkp = dk->data.f32;
	float* const dvp = dv->data.f32;
	const float scale = cmd.info.scaled_dot_product_attention.scale;
	const int is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	const int h_h_k_ratio = qdim[2] / kdim[2];
	for (i[0] = 0; i[0] < qdim[0]; i[0]++)
	{
		const float* const qp0 = qp + i[0] * qstride[0];
		const float* const kp0 = kp + i[0] * kstride[0];
		const float* const vp0 = vp + i[0] * vstride[0];
		const float* const gp0 = gp + i[0] * gstride[0];
		float* const dqp0 = dqp + i[0] * dqstride[0];
		float* const dkp0 = dkp + i[0] * dkstride[0];
		float* const dvp0 = dvp + i[0] * dvstride[0];
		for (i[1] = 0; i[1] < qdim[2]; i[1]++)
		{
			const float* const qp1 = qp0 + i[1] * qstride[2];
			const float* const kp1 = kp0 + (i[1] / h_h_k_ratio) * kstride[2];
			const float* const vp1 = vp0 + (i[1] / h_h_k_ratio) * vstride[2];
			const float* const gp1 = gp0 + i[1] * gstride[2];
			float* const dqp1 = dqp0 + i[1] * dqstride[2];
			float* const dkp1 = dkp0 + (i[1] / h_h_k_ratio) * dkstride[2];
			float* const dvp1 = dvp0 + (i[1] / h_h_k_ratio) * dvstride[2];
			// Compute Q @ K^T
			int x, y, k;
			for (x = 0; x < qdim[1]; x++)
			{
				float* const dqp2 = dqp1 + x * dqstride[1];
				for (k = 0; k < qdim[3]; k++)
					dqp2[k * dqstride[3]] = 0;
			}
			// Only zero out when it is at 0-index.
			if (i[1] % h_h_k_ratio == 0)
				for (y = 0; y < kdim[1]; y++)
				{
					float* const dkp2 = dkp1 + y * dkstride[1];
					for (k = 0; k < qdim[3]; k++)
						dkp2[k * dkstride[3]] = 0;
				}
			// Only zero out when it is at 0-index.
			if (i[1] % h_h_k_ratio == 0)
				for (y = 0; y < kdim[1]; y++)
				{
					float* const dvp2 = dvp1 + y * dvstride[1];
					for (k = 0; k < vdim[3]; k++)
						dvp2[k * dvstride[3]] = 0;
				}
			for (x = 0; x < qdim[1]; x++)
			{
				const float* const qp2 = qp1 + x * qstride[1];
				const float* const gp2 = gp1 + x * gstride[1];
				float* const qk0 = qk;
				float* const qks0 = qk + kdim[1];
				for (y = 0; y < kdim[1]; y++)
				{
					const float* const kp2 = kp1 + y * kstride[1];
					float v = 0;
					for (k = 0; k < qdim[3]; k++)
						v += qp2[k * qstride[3]] * kp2[k * kstride[3]];
					qk0[y] = scale * v;
				}
				// Compute softmax on qk.
				if (is_causal)
				{
					const int x_end = ccv_max(x - qdim[1] + kdim[1] + 1, 0);
					for (y = x_end; y < kdim[1]; y++)
						qk0[y] = 0;
					double maxval = qk0[0];
					for (y = 1; y < x_end; y++)
						if (qk0[y] > maxval)
							maxval = qk0[y];
					double sumval = 0;
					for (y = 0; y < x_end; y++)
						sumval += (qk0[y] = expf(qk0[y] - maxval));
					sumval = 1.0 / sumval;
					for (y = 0; y < x_end; y++)
						qk0[y] *= sumval;
				} else {
					double maxval = qk0[0];
					for (y = 1; y < kdim[1]; y++)
						if (qk0[y] > maxval)
							maxval = qk0[y];
					double sumval = 0;
					for (y = 0; y < kdim[1]; y++)
						sumval += (qk0[y] = expf(qk0[y] - maxval));
					sumval = 1.0 / sumval;
					for (y = 0; y < kdim[1]; y++)
						qk0[y] *= sumval;
				}
				for (y = 0; y < kdim[1]; y++)
				{
					float* const dvp2 = dvp1 + y * dvstride[1];
					const float v = qk0[y];
					for (k = 0; k < vdim[3]; k++)
						dvp2[k * dvstride[3]] += v * gp2[k * gstride[3]];
				}
				double sumval = 0;
				for (y = 0; y < kdim[1]; y++)
				{
					const float* const vp2 = vp1 + y * vstride[1];
					float v = 0;
					for (k = 0; k < vdim[3]; k++)
						v += gp2[k * gstride[3]] * vp2[k * vstride[3]];
					qks0[y] = v;
					sumval += v * qk0[y];
				}
				for (y = 0; y < kdim[1]; y++)
					qk0[y] = (qks0[y] - sumval) * qk0[y];
				float* const dqp2 = dqp1 + x * dqstride[1];
				for (y = 0; y < kdim[1]; y++)
				{
					const float* const kp2 = kp1 + y * kstride[1];
					float* const dkp2 = dkp1 + y * dkstride[1];
					const float v = scale * qk0[y];
					for (k = 0; k < qdim[3]; k++)
					{
						dqp2[k * dqstride[3]] += v * kp2[k * kstride[3]];
						dkp2[k * dkstride[3]] += v * qp2[k * qstride[3]];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
}
