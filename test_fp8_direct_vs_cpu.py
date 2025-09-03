#!/usr/bin/env python3

import sys
import os
# Force Python to use the local SageAttention instead of installed package
sys.path.insert(0, '/home/wlin1/Drawthings/ccv/SageAttention')

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Optional, Any
from sageattention import core
from sageattention.core import per_warp_int8_cuda, per_thread_int8_triton, per_channel_fp8

# Import SM89_ENABLED and _qattn_sm89 from core module
try:
    from sageattention.core import SM89_ENABLED, _qattn_sm89
except ImportError:
    # Fallback if not available
    SM89_ENABLED = False
    _qattn_sm89 = None
    print("Warning: SM89 kernel not available, using fallback")

def load_ccv_test_tensors():
    """Load tensors saved from CCV test case (trial 5)."""
    # Read dimensions first
    dims = {}
    try:
        with open('/tmp/test_fp8_dimensions.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                dims[key] = int(value)
    except FileNotFoundError:
        print("Warning: /tmp/test_fp8_dimensions.txt not found. Using fallback dimensions.")
        # Fallback to match the actual data size
        dims = {'B': 1, 'R': 64, 'C': 64, 'Hq': 8, 'Hk': 8, 'D': 128, 'is_causal': 0}
    
    B = dims['B']
    R = dims['R']
    C = dims['C']
    Hq = dims['Hq']
    Hk = dims['Hk']
    D = dims['D']
    is_causal = dims['is_causal']
    
    print(f"Loaded dimensions: B={B}, R={R}, C={C}, Hq={Hq}, Hk={Hk}, D={D}, is_causal={is_causal}")
    
    # Check file sizes to determine actual dimensions
    import os
    q_size = os.path.getsize('/tmp/test_fp8_q_nhd.bin') // 2  # divide by 2 for fp16
    k_size = os.path.getsize('/tmp/test_fp8_k_nhd.bin') // 2
    v_size = os.path.getsize('/tmp/test_fp8_v_nhd.bin') // 2
    
    print(f"File sizes (elements): Q={q_size}, K={k_size}, V={v_size}")
    
    # Verify dimensions match file sizes
    expected_q_size = B * R * Hq * D
    expected_k_size = B * C * Hk * D
    expected_v_size = B * C * Hk * D
    
    if q_size != expected_q_size:
        print(f"Warning: Q size mismatch. File has {q_size} elements, expected {expected_q_size}")
        # Try to infer correct dimensions based on file size
        # 65536 = 1 * 64 * 8 * 128 = B * S * H * D
        if q_size == 65536:
            print("Detected Q size = 65536, inferring dimensions: B=1, S=64, H=8, D=128")
            B, R, Hq, D = 1, 64, 8, 128
            dims['B'], dims['R'], dims['Hq'], dims['D'] = B, R, Hq, D
        if k_size == 65536:
            C, Hk = 64, 8
            dims['C'], dims['Hk'] = C, Hk
    
    # Load FP16 tensors (NHD format: [B, S, H, D])
    q_fp16 = np.fromfile('/tmp/test_fp8_q_nhd.bin', dtype=np.float16).reshape(B, R, Hq, D)
    k_fp16 = np.fromfile('/tmp/test_fp8_k_nhd.bin', dtype=np.float16).reshape(B, C, Hk, D)
    v_fp16 = np.fromfile('/tmp/test_fp8_v_nhd.bin', dtype=np.float16).reshape(B, C, Hk, D)
    
    # Convert to PyTorch tensors
    q = torch.from_numpy(q_fp16).cuda().half()
    k = torch.from_numpy(k_fp16).cuda().half()
    v = torch.from_numpy(v_fp16).cuda().half()
    
    return q, k, v, is_causal, dims

def compute_cpu_reference(q, k, v, scale, is_causal=False):
    """Compute attention using CPU in FP32 for reference. Expects HND format [B, H, S, D]."""
    q_ref = q.float().cpu()
    k_ref = k.float().cpu()
    v_ref = v.float().cpu()
    
    # Input format is [B, H, S, D] (HND)
    batch_size, num_q_heads, seq_len, head_dim = q_ref.shape
    _, num_kv_heads, _, _ = k_ref.shape
    
    if num_q_heads != num_kv_heads:
        # Repeat K and V to match Q's head count for grouped query attention
        num_groups = num_q_heads // num_kv_heads
        k_ref = k_ref.repeat_interleave(num_groups, dim=1)  # [B, H_q, S, D]
        v_ref = v_ref.repeat_interleave(num_groups, dim=1)  # [B, H_q, S, D]
    
    # Compute attention: Q @ K^T
    attn_weights = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale  # [B, H, S, S]
    
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        attn_weights = attn_weights + causal_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v_ref)  # [B, H, S, D]
    return output

def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = False,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    print(f"DEBUG: sageattn_qk_int8_pv_fp8_cuda called with layout={tensor_layout}")
    dtype = q.dtype
    assert SM89_ENABLED and _qattn_sm89 is not None, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # cuda_major_version, cuda_minor_version = get_cuda_version()
    # if(cuda_major_version, cuda_minor_version) < (12, 8) and pv_accum_dtype == 'fp32+fp16':
    #     warnings.warn("cuda version < 12.8, change pv_accum_dtype to 'fp32+fp32'")
    #     pv_accum_dtype = 'fp32+fp32'

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    # DEBUG: Print input tensor info before quantization
    print(f"\n=== Before Quantization (PyTorch Input Check) ===")
    print(f"Input Q shape: {q.shape}")
    print(f"Input K shape: {k.shape}")
    print(f"Input Q sample (first 10): {q.flatten()[:10].cpu().tolist()}")
    print(f"Input K sample (first 10): {k.flatten()[:10].cpu().tolist()}")
    if km is not None:
        print(f"K mean shape: {km.shape}")
        print(f"K mean sample (first 10): {km.flatten()[:10].cpu().tolist()}")
    else:
        print(f"K mean: None")
    print(f"tensor_layout: {tensor_layout}")
    print(f"qk_quant_gran: {qk_quant_gran}")
    print(f"=== End PyTorch Input Check ===\n")

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    # DEBUG: Save intermediate quantization results for CCV comparison
    print(f"DEBUG: Saving intermediate quantization results...")
    print(f"  q_int8 shape: {q_int8.shape}")
    print(f"Q_int8 sample (first 10): {q_int8.flatten()[:10].cpu().tolist()}")

    print(f"  k_int8 shape: {k_int8.shape}")
    print(f"k_int8 sample (first 10): {k_int8.flatten()[:10].cpu().tolist()}")

    print(f"  q_scale shape: {q_scale.shape}")
    print(f"Q_scale sample (first 10): {q_scale.flatten()[:10].cpu().tolist()}")
    print(f"q_scale input strides: {q_scale.stride()}")

    print(f"  k_scale shape: {k_scale.shape}")
    print(f"k_scale sample (first 10): {k_scale.flatten()[:10].cpu().tolist()}")
    print(f"k_scale input strides: {k_scale.stride()}")

    if km is not None:
        print(f"  km shape: {km.shape}")    
    
    # Save quantized Q and K tensors (use trial5 naming for consistency)
    q_int8.cpu().numpy().astype('int8').tofile('/tmp/pytorch_trial5_q_int8.bin')
    k_int8.cpu().numpy().astype('int8').tofile('/tmp/pytorch_trial5_k_int8.bin')
    q_scale.cpu().numpy().astype('float32').tofile('/tmp/pytorch_trial5_q_scales.bin')
    k_scale.cpu().numpy().astype('float32').tofile('/tmp/pytorch_trial5_k_scales.bin')
    
    if km is not None:
        km.cpu().numpy().astype('float16').tofile('/tmp/pytorch_trial5_k_mean.bin')
    
    # Save original inputs for reference
    q.cpu().numpy().astype('float16').tofile('/tmp/pytorch_trial5_q_input.bin')
    k.cpu().numpy().astype('float16').tofile('/tmp/pytorch_trial5_k_input.bin')
    v.cpu().numpy().astype('float16').tofile('/tmp/pytorch_trial5_v_input.bin')

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32+fp16' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
        smooth_v = False

    quant_v_scale_max = 448.0
    if pv_accum_dtype == 'fp32+fp16':
        quant_v_scale_max = 2.25

    # Step 2: Quantize V tensor using per_channel_fp8
    print(f"\n=== Before V Quantization (PyTorch) ===")
    print(f"V input shape: {v.shape}, dtype: {v.dtype}")
    print(f"V input strides: {v.stride()}")
    print(f"V input sample (first 10): {v.flatten()[:10].cpu().tolist()}")
    print(f"V input min: {v.min().item():.6f}, max: {v.max().item():.6f}, mean: {v.mean().item():.6f}")
    print(f"Quantization params: scale_max={quant_v_scale_max}, smooth_v={smooth_v}")
    
    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v)
    
    print(f"\n=== After V Quantization (PyTorch) ===")
    print(f"V_fp8 shape: {v_fp8.shape}, dtype: {v_fp8.dtype}")
    print(f"V_fp8 strides: {v_fp8.stride()}")
    print(f"V_fp8 sample (first 10 as uint8): {v_fp8.view(torch.uint8).flatten()[:10].cpu().tolist()}")
    print(f"V_scale shape: {v_scale.shape}, dtype: {v_scale.dtype}")
    print(f"V_scale sample (first 10): {v_scale.flatten()[:10].cpu().tolist()}")
    print(f"V_scale min: {v_scale.min().item():.6f}, max: {v_scale.max().item():.6f}, mean: {v_scale.mean().item():.6f}")
    if vm is not None:
        print(f"V_mean shape: {vm.shape}, dtype: {vm.dtype}")
        print(f"V_mean sample (first 10): {vm.flatten()[:10].cpu().tolist()}")
    
    # Save FP8 V as int8 bytes for CCV comparison
    v_fp8.view(torch.int8).cpu().numpy().tofile('/tmp/pytorch_trial5_v_fp8.bin')
    v_scale.cpu().numpy().astype('float32').tofile('/tmp/pytorch_trial5_v_scales.bin')
    
    if vm is not None:
        vm.cpu().numpy().astype('float32').tofile('/tmp/pytorch_trial5_v_mean.bin')

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp16":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    # DEBUG: Save final output
    print(f"  final output shape: {o.shape}")
    o.cpu().numpy().astype('float16').tofile('/tmp/pytorch_trial5_final_output.bin')
    print(f"DEBUG: All intermediate results saved to /tmp/pytorch_trial5_*.bin")

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

def test_fp8_kernel_direct():
    """Test FP8 kernel directly against CPU reference."""
    print("=== Direct FP8 Kernel vs CPU Reference Test ===\n")
    
    # Test parameters
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    
    batch_size, num_heads, seq_len, head_dim = 1, 8, 64, 128
    scale = 1.0 / (head_dim ** 0.5)
    
    # Test both HND and NHD layouts
    for layout in [ "NHD"]: ## "HND",
        print(f"\n--- Testing {layout} Layout ---")
        
        # Generate test data in appropriate layout
        if layout == "HND":
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        else:  # NHD
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
        
        print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
        
        # Compute CPU reference
        if layout == "NHD":
            # Convert to HND for CPU computation
            q_cpu = q.permute(0, 2, 1, 3).contiguous()
            k_cpu = k.permute(0, 2, 1, 3).contiguous()
            v_cpu = v.permute(0, 2, 1, 3).contiguous()
        else:
            q_cpu = q
            k_cpu = k
            v_cpu = v
        
        cpu_output = compute_cpu_reference(q_cpu, k_cpu, v_cpu, scale, is_causal=False)
        print(f"CPU reference output shape: {cpu_output.shape}")
        print(f"CPU reference sample values: {cpu_output[0,0,0,:5]}")
        
        # Call sageattn_qk_int8_pv_fp8_cuda directly
        try:
            print(f"Calling sageattn_qk_int8_pv_fp8_cuda with layout={layout}")
            output = sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout=layout,
                is_causal=False,
                sm_scale=scale,
                return_lse=False,
                pv_accum_dtype="fp32+fp32",  # Maximum precision
                qk_quant_gran="per_warp"
            )
            
            print(f"\nFP8 kernel output shape: {output.shape}")
            
            # Convert output to HND if needed for comparison
            if layout == "NHD":
                output_hnd = output.permute(0, 2, 1, 3).contiguous()
                print(f"FP8 kernel sample values: {output_hnd[0,0,0,:5]}")
            else:
                output_hnd = output
                print(f"FP8 kernel sample values: {output[0,0,0,:5]}")
            
            # Compare with CPU reference
            diff = (output_hnd.cpu().float() - cpu_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n{layout} Layout Results:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff > 0.1:
                print(f"  ⚠️ Large differences detected (expected due to quantization)")
            else:
                print(f"  ✅ Results match within tolerance")
                
            # Save inputs and outputs for CCV testing
            if layout == "HND":
                q_save = q.cpu().numpy().astype(np.float16)
                k_save = k.cpu().numpy().astype(np.float16)
                v_save = v.cpu().numpy().astype(np.float16)
                output_save = output.cpu().numpy().astype(np.float16)
            else:  # NHD
                q_save = q.cpu().numpy().astype(np.float16)
                k_save = k.cpu().numpy().astype(np.float16)
                v_save = v.cpu().numpy().astype(np.float16)
                output_save = output.cpu().numpy().astype(np.float16)
            
            cpu_output_np = cpu_output.numpy().astype(np.float32)
            
            # Save input tensors
            q_save.tofile(f'/tmp/test_fp8_q_{layout.lower()}.bin')
            k_save.tofile(f'/tmp/test_fp8_k_{layout.lower()}.bin')
            v_save.tofile(f'/tmp/test_fp8_v_{layout.lower()}.bin')
            
            # Save outputs
            output_save.tofile(f'/tmp/test_fp8_output_{layout.lower()}.bin')
            cpu_output_np.tofile(f'/tmp/test_fp8_cpu_reference_{layout.lower()}.bin')
            
            print(f"  Saved inputs to /tmp/test_fp8_{{q,k,v}}_{layout.lower()}.bin")
            print(f"  Saved FP8 output to /tmp/test_fp8_output_{layout.lower()}.bin")
            print(f"  Saved CPU reference to /tmp/test_fp8_cpu_reference_{layout.lower()}.bin")
            
        except Exception as e:
            print(f"Error calling sageattn_qk_int8_pv_fp8_cuda: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test FP8 SageAttention kernel')
    parser.add_argument('--load-ccv', action='store_true', help='Load tensors from CCV test (trial 5)')
    args = parser.parse_args()
    
    if args.load_ccv:
        print("=== Loading tensors from CCV test case (trial 5) ===\n")
        try:
            q, k, v, is_causal, dims = load_ccv_test_tensors()
            
            # Compute scale
            scale = 1.0 / (dims['D'] ** 0.5)
            
            print(f"\nTensor shapes loaded from CCV:")
            print(f"  Q: {q.shape}, K: {k.shape}, V: {v.shape}")
            print(f"  Scale: {scale}, is_causal: {is_causal}")
            
            # Compute CPU reference (convert NHD to HND for CPU)
            q_cpu = q.permute(0, 2, 1, 3).contiguous()
            k_cpu = k.permute(0, 2, 1, 3).contiguous()
            v_cpu = v.permute(0, 2, 1, 3).contiguous()
            
            print(f"CPU input shapes after NHD->HND: Q={q_cpu.shape}, K={k_cpu.shape}, V={v_cpu.shape}")
            
            cpu_output = compute_cpu_reference(q_cpu, k_cpu, v_cpu, scale, is_causal=bool(is_causal))
            print(f"\nCPU reference output shape: {cpu_output.shape}")
            print(f"CPU reference sample values: {cpu_output[0,0,0,:5]}")
            
            # Run SageAttention with the loaded tensors (NHD layout)
            print(f"\nCalling sageattn_qk_int8_pv_fp8_cuda with NHD layout")
            output = sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout="NHD",
                is_causal=bool(is_causal),
                sm_scale=scale,
                return_lse=False,
                pv_accum_dtype="fp32+fp32",
                qk_quant_gran="per_warp"
            )
            
            # Convert output to HND for comparison
            output_hnd = output.permute(0, 2, 1, 3).contiguous()
            print(f"\nFP8 kernel output shape: {output.shape}")
            print(f"FP8 kernel sample values: {output_hnd[0,0,0,:5]}")
            
            # Compare with CPU reference
            diff = (output_hnd.cpu().float() - cpu_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n=== Results Comparison ===")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 0.01:
                print(f"  ✅ Excellent match!")
            elif max_diff < 0.1:
                print(f"  ✅ Good match within tolerance")
            else:
                print(f"  ⚠️ Large differences detected (expected due to quantization)")
                
        except FileNotFoundError as e:
            print(f"Error: Could not load CCV tensors. Make sure to run the CCV test first.")
            print(f"Run: ./test/int/nnc/cublas.tests 'scaled dot product attention with sage_attn'")
            print(f"Missing file: {e}")
    else:
        test_fp8_kernel_direct()