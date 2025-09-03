#!/usr/bin/env python3

import numpy as np
import os

def load_binary_file(filepath, dtype, expected_elements=None):
    """Load binary file and check its size."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        data = np.fromfile(filepath, dtype=dtype)
        if expected_elements and len(data) != expected_elements:
            print(f"⚠️  Size mismatch in {filepath}: got {len(data)}, expected {expected_elements}")
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def compare_arrays(ccv_data, pytorch_data, name, atol=1e-6, rtol=1e-5):
    """Compare two arrays and print statistics."""
    if ccv_data is None or pytorch_data is None:
        print(f"❌ {name}: Cannot compare (missing data)")
        return False
    
    if len(ccv_data) != len(pytorch_data):
        print(f"❌ {name}: Size mismatch - CCV: {len(ccv_data)}, PyTorch: {len(pytorch_data)}")
        return False
    
    # Compute differences
    diff = np.abs(ccv_data.astype(np.float64) - pytorch_data.astype(np.float64))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Check for exact matches
    exact_matches = np.sum(diff < 1e-10)
    close_matches = np.sum(diff < atol)
    
    print(f"\n=== {name} Comparison ===")
    print(f"  Array size: {len(ccv_data)} elements")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  Exact matches (diff < 1e-10): {exact_matches}/{len(ccv_data)} ({100*exact_matches/len(ccv_data):.2f}%)")
    print(f"  Close matches (diff < {atol}): {close_matches}/{len(ccv_data)} ({100*close_matches/len(ccv_data):.2f}%)")
    
    # Show sample values for debugging
    print(f"  CCV sample (first 10): {ccv_data.flat[:10]}")
    print(f"  PyTorch sample (first 10): {pytorch_data.flat[:10]}")
    
    if max_diff < atol:
        print(f"  ✅ MATCH: Arrays are very close (max diff < {atol})")
        return True
    elif max_diff < 0.01:
        print(f"  ⚠️  ACCEPTABLE: Arrays are reasonably close (max diff < 0.01)")
        return True
    else:
        print(f"  ❌ MISMATCH: Arrays differ significantly (max diff > 0.01)")
        return False

def main():
    print("=== CCV vs PyTorch Intermediate Results Comparison ===\n")
    
    # Expected dimensions for trial 5
    B, R, C, Hq, Hk, D = 1, 5, 5, 32, 8, 128
    
    # File mappings: (CCV file, PyTorch file, data type, expected elements, description)
    comparisons = [
        # Original inputs
        ("/tmp/ccv_trial5_q_input.bin", "/tmp/pytorch_trial5_q_input.bin", np.float16, B*R*Hq*D, "Q Input (FP16)"),
        ("/tmp/ccv_trial5_k_input.bin", "/tmp/pytorch_trial5_k_input.bin", np.float16, B*C*Hk*D, "K Input (FP16)"),
        ("/tmp/ccv_trial5_v_input.bin", "/tmp/pytorch_trial5_v_input.bin", np.float16, B*C*Hk*D, "V Input (FP16)"),
        
        # Quantized Q and K
        ("/tmp/ccv_trial5_q_int8.bin", "/tmp/pytorch_trial5_q_int8.bin", np.int8, B*R*Hq*D, "Q Quantized (INT8)"),
        ("/tmp/ccv_trial5_k_int8.bin", "/tmp/pytorch_trial5_k_int8.bin", np.int8, B*C*Hk*D, "K Quantized (INT8)"),
        
        # Q and K scales
        ("/tmp/ccv_trial5_q_scales.bin", "/tmp/pytorch_trial5_q_scales.bin", np.float32, B*Hq*4, "Q Scales (per-warp)"),  # 4 warps per head
        ("/tmp/ccv_trial5_k_scales.bin", "/tmp/pytorch_trial5_k_scales.bin", np.float32, B*Hk*1, "K Scales (per-block)"),  # 1 scale per head
        
        # V quantization (more complex due to transpose_pad_permute)
        ("/tmp/ccv_trial5_v_fp8.bin", "/tmp/pytorch_trial5_v_fp8.bin", np.int8, B*D*Hk*64, "V Quantized (FP8)"),  # padded to 64
        ("/tmp/ccv_trial5_v_scales.bin", "/tmp/pytorch_trial5_v_scales.bin", np.float32, B*Hk*D, "V Scales (per-channel)"),
        
        # Final outputs
        ("/tmp/ccv_trial5_final_output.bin", "/tmp/pytorch_trial5_final_output.bin", np.float16, B*R*Hq*D, "Final Output (FP16)"),
    ]
    
    all_match = True
    
    for ccv_file, pytorch_file, dtype, expected_elements, description in comparisons:
        ccv_data = load_binary_file(ccv_file, dtype, expected_elements)
        pytorch_data = load_binary_file(pytorch_file, dtype, expected_elements)
        
        if ccv_data is not None and pytorch_data is not None:
            # Use more lenient tolerances for FP8 quantized data
            if "FP8" in description or "Quantized" in description:
                atol = 1e-3  # More lenient for quantized data
            else:
                atol = 1e-6  # Stricter for original precision data
                
            match = compare_arrays(ccv_data, pytorch_data, description, atol=atol)
            all_match &= match
        else:
            all_match = False
    
    print(f"\n=== Summary ===")
    if all_match:
        print("✅ All intermediate results match between CCV and PyTorch!")
    else:
        print("❌ Some intermediate results differ between CCV and PyTorch.")
        print("   This explains the accuracy difference in the final attention output.")
        print("   Focus on the first mismatch to identify the root cause.")

if __name__ == "__main__":
    main()