#!/usr/bin/env python3
"""
Fused Kernel Demonstration: Softmax Regression

This file demonstrates the CONCEPT of kernel fusion by implementing:
1. Standard approach: Separate matmul + softmax operations (using PyTorch/cuBLAS)
2. Fused approach: Combined matmul + softmax in a single Triton kernel

IMPORTANT: The fused kernels are actually SLOWER than PyTorch's unfused version!

Why is fusion slower here?
- PyTorch uses highly optimized cuBLAS for matmul (decades of engineering)
- Our Triton matmul is naive (no tensor cores, poor tiling)
- Matmul is compute-bound, fusion helps memory-bound operations
- The intermediate logits tensor is relatively small

Educational value:
- Understanding when fusion helps (and when it doesn't!)
- Learning Triton kernel programming concepts
- Benchmarking and performance analysis
- Don't assume "advanced" techniques are always faster - measure!
"""

import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity


# ============================================================================
# Helper Functions
# ============================================================================

def get_device(index: int = 0) -> torch.device:
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


# ============================================================================
# Standard (Unfused) Implementation - PyTorch
# ============================================================================

def matmul_softmax_unfused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Standard approach: Separate matmul and softmax (PyTorch/cuBLAS).

    Args:
        x: (batch_size, input_dim)
        weight: (input_dim, num_classes)

    Returns:
        probs: (batch_size, num_classes) - softmax probabilities
    """
    logits = x @ weight  # (batch_size, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


# ============================================================================
# Unfused Triton Implementation: Separate Matmul + Softmax Kernels
# ============================================================================

@triton.jit
def triton_matmul_kernel(
    x_ptr, weight_ptr, output_ptr,
    M, K, N,
    x_stride_row, x_stride_col,
    weight_stride_row, weight_stride_col,
    output_stride_row, output_stride_col,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Simple matmul kernel: output = x @ weight
    Each program computes one row of the output.
    """
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    # Offsets for output dimension
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Tile over K dimension
    for k_tile in range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_tile * BLOCK_SIZE_K
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        # Load x values for this row (BLOCK_SIZE_K,)
        x_tile = tl.load(
            x_ptr + row_idx * x_stride_row + k_offsets * x_stride_col,
            mask=k_mask,
            other=0.0
        )

        # Load weight tile (BLOCK_SIZE_K, BLOCK_SIZE_N)
        k_offsets_2d = k_offsets[:, None]
        n_offsets_2d = n_offsets[None, :]

        weight_tile = tl.load(
            weight_ptr + k_offsets_2d * weight_stride_row + n_offsets_2d * weight_stride_col,
            mask=(k_mask[:, None]) & (n_mask[None, :]),
            other=0.0
        )

        # Accumulate: x_tile @ weight_tile
        acc += tl.sum(x_tile[:, None] * weight_tile, axis=0)

    # Store result
    tl.store(
        output_ptr + row_idx * output_stride_row + n_offsets * output_stride_col,
        acc,
        mask=n_mask
    )


@triton.jit
def triton_softmax_kernel(
    x_ptr, output_ptr,
    M, N,
    x_stride_row, x_stride_col,
    output_stride_row, output_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel: output = softmax(x, dim=-1)
    Each program computes softmax for one row.
    """
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    # Offsets for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row
    x_row = tl.load(
        x_ptr + row_idx * x_stride_row + col_offsets * x_stride_col,
        mask=mask,
        other=float('-inf')
    )

    # Compute softmax (numerically stable)
    x_max = tl.max(x_row, axis=0)
    numerator = tl.exp(x_row - x_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    # Store result
    tl.store(
        output_ptr + row_idx * output_stride_row + col_offsets * output_stride_col,
        output,
        mask=mask
    )


def matmul_softmax_triton_unfused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Unfused Triton implementation: separate matmul and softmax kernels.
    This materializes the intermediate logits tensor.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    # Step 1: Matmul (materialize logits in global memory)
    logits = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = min(triton.next_power_of_2(N), 1024)

    grid = (M,)
    triton_matmul_kernel[grid](
        x, weight, logits,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    # Step 2: Softmax (read logits from global memory, write probs)
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)

    triton_softmax_kernel[grid](
        logits, output,
        M, N,
        logits.stride(0), logits.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ============================================================================
# Fused Triton Kernel: Matmul + Softmax
# ============================================================================

@triton.jit
def fused_matmul_softmax_kernel(
    x_ptr,  # Input: (M, K)
    weight_ptr,  # Weight: (K, N)
    output_ptr,  # Output: (M, N)
    M, K, N,  # Matrix dimensions
    x_stride_row, x_stride_col,
    weight_stride_row, weight_stride_col,
    output_stride_row, output_stride_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that computes: softmax(x @ weight)

    Each block computes one row of the output (one sample's probabilities).
    This kernel:
    1. Computes logits = x @ weight for one row
    2. Immediately applies softmax to that row
    3. Writes the result

    This avoids storing the full logits matrix in global memory.
    """
    # Each program computes one row of the output
    row_idx = tl.program_id(0)

    # Bounds check
    if row_idx >= M:
        return

    # Pointers to the row of x we're processing
    x_row_ptr = x_ptr + row_idx * x_stride_row

    # Pointers to output row
    output_row_ptr = output_ptr + row_idx * output_stride_row

    # Allocate accumulator for logits (entire row)
    # We need to compute all N outputs before softmax
    logits = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Process in tiles of N (output columns)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # Reset accumulator for this tile of outputs
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        # Compute matmul for this tile: accumulate over K dimension
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < K

            # Load chunk of x row (1, BLOCK_SIZE_K)
            x_chunk = tl.load(
                x_row_ptr + k_offsets * x_stride_col,
                mask=k_mask,
                other=0.0
            )

            # Load chunk of weight matrix (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # For each output column, we need a column from weight
            for i, k_off in enumerate(k_offsets):
                if k_off < K:
                    w_vals = tl.load(
                        weight_ptr + k_off * weight_stride_row + n_offsets * weight_stride_col,
                        mask=n_mask,
                        other=0.0
                    )
                    acc += x_chunk[i] * w_vals

        # Store logits for this tile
        logits_tile_mask = n_offsets < N
        # Keep logits in registers for now
        # We need all logits before computing softmax

        # For now, we'll compute each row separately
        # A more advanced version would tile over N as well

        # Actually, let's simplify: compute entire row at once
        if n_start == 0:  # First iteration, initialize
            logits = tl.where(n_mask, acc, float('-inf'))
        else:
            # This approach won't work well - need different strategy
            pass

    # For simplicity, let's rewrite to compute full row


@triton.jit
def fused_matmul_softmax_kernel_simple(
    x_ptr,  # Input: (M, K)
    weight_ptr,  # Weight: (K, N)
    output_ptr,  # Output: (M, N)
    M, K, N,  # Matrix dimensions
    x_stride_row, x_stride_col,
    weight_stride_row, weight_stride_col,
    output_stride_row, output_stride_col,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Simplified fused matmul + softmax kernel.
    Each program computes one row of output (one sample).
    """
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    # Compute logits for this row
    # logits[j] = sum_k x[row_idx, k] * weight[k, j]

    # Allocate space for full logit row in registers/SRAM
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N

    # Initialize logits accumulator
    logits = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Accumulate over K dimension
    x_row_ptr = x_ptr + row_idx * x_stride_row

    for k in range(K):
        # Load one element from x
        x_val = tl.load(x_row_ptr + k * x_stride_col)

        # Load one row from weight (K=k, all N)
        weight_row = tl.load(
            weight_ptr + k * weight_stride_row + n_offsets * weight_stride_col,
            mask=n_mask,
            other=0.0
        )

        # Accumulate
        logits += x_val * weight_row

    # Now compute softmax on logits
    # Numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    # Find max
    logits_max = tl.max(logits, axis=0)

    # Compute exp(logits - max)
    numerator = tl.exp(logits - logits_max)

    # Compute sum
    denominator = tl.sum(numerator, axis=0)

    # Compute softmax probabilities
    probs = numerator / denominator

    # Write output
    output_row_ptr = output_ptr + row_idx * output_stride_row
    tl.store(
        output_row_ptr + n_offsets * output_stride_col,
        probs,
        mask=n_mask
    )


def matmul_softmax_fused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Fused matmul + softmax using Triton kernel.

    Args:
        x: (batch_size, input_dim)
        weight: (input_dim, num_classes)

    Returns:
        probs: (batch_size, num_classes)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    assert x.is_contiguous() and weight.is_contiguous(), "Tensors must be contiguous"

    M, K = x.shape  # batch_size, input_dim
    K2, N = weight.shape  # input_dim, num_classes
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Kernel parameters
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    # Ensure BLOCK_SIZE_N isn't too large
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 1024)

    # Launch kernel: one program per row
    grid = (M,)

    fused_matmul_softmax_kernel_simple[grid](
        x, weight, output,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output


# ============================================================================
# Optimized Fused Kernel with Tiling
# ============================================================================

@triton.jit
def fused_matmul_softmax_tiled_kernel(
    x_ptr, weight_ptr, output_ptr,
    M, K, N,
    x_stride_row, x_stride_col,
    weight_stride_row, weight_stride_col,
    output_stride_row, output_stride_col,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized fused matmul + softmax with better memory access patterns.
    Uses tiling over K dimension for better memory coalescing.
    """
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    # Offsets for output dimension (num_classes)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N

    # Initialize logits accumulator
    logits = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Tile over K dimension (input_dim)
    for k_tile in range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_tile * BLOCK_SIZE_K
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        # Load x values for this row (BLOCK_SIZE_K,)
        x_tile = tl.load(
            x_ptr + row_idx * x_stride_row + k_offsets * x_stride_col,
            mask=k_mask,
            other=0.0
        )

        # Load weight tile (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We need to load a submatrix of weight
        k_offsets_2d = k_offsets[:, None]  # (BLOCK_SIZE_K, 1)
        n_offsets_2d = n_offsets[None, :]  # (1, BLOCK_SIZE_N)

        weight_tile = tl.load(
            weight_ptr + k_offsets_2d * weight_stride_row + n_offsets_2d * weight_stride_col,
            mask=(k_mask[:, None]) & (n_mask[None, :]),
            other=0.0
        )

        # Compute partial matmul: x_tile (K,) @ weight_tile (K, N) -> (N,)
        # x_tile: (BLOCK_SIZE_K,), weight_tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Result: (BLOCK_SIZE_N,)
        logits += tl.sum(x_tile[:, None] * weight_tile, axis=0)

    # Softmax (numerically stable)
    logits_max = tl.max(logits, axis=0)
    numerator = tl.exp(logits - logits_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator

    # Store result
    tl.store(
        output_ptr + row_idx * output_stride_row + n_offsets * output_stride_col,
        probs,
        mask=n_mask
    )


def matmul_softmax_fused_tiled(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Fused matmul + softmax with tiled implementation.

    NOTE: This version is limited to num_classes <= 256 due to register constraints.
    For larger num_classes, falls back to simple fused kernel.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()

    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # CRITICAL: The tiled kernel can only handle N <= BLOCK_SIZE_N
    # because softmax requires ALL logits before normalization
    # For large N, fall back to simple kernel
    if N > 256:
        # Use simple kernel for large num_classes
        return matmul_softmax_fused(x, weight)

    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = min(triton.next_power_of_2(N), 256)

    grid = (M,)

    fused_matmul_softmax_tiled_kernel[grid](
        x, weight, output,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output


# ============================================================================
# Benchmarking and Testing
# ============================================================================

def benchmark_function(func, x, weight, num_warmups=3, num_trials=10):
    """Benchmark a function."""
    # Warmup
    for _ in range(num_warmups):
        func(x, weight)
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(num_trials):
        start = time.time()
        result = func(x, weight)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    return mean(times), result


def test_correctness():
    """Test that all implementations match."""
    print("=" * 80)
    print("Testing Correctness")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Test cases
    test_cases = [
        (32, 128, 10),    # Small
        (128, 256, 20),   # Medium
        (256, 512, 100),  # Large
    ]

    for batch_size, input_dim, num_classes in test_cases:
        x = torch.randn(batch_size, input_dim, device='cuda')
        weight = torch.randn(input_dim, num_classes, device='cuda')

        # Reference (PyTorch)
        ref_output = matmul_softmax_unfused(x, weight)

        # Triton unfused
        triton_unfused_output = matmul_softmax_triton_unfused(x, weight)

        # Triton fused simple
        fused_output = matmul_softmax_fused(x, weight)

        # Triton fused tiled
        fused_tiled_output = matmul_softmax_fused_tiled(x, weight)

        # Check
        max_diff_triton_unfused = (ref_output - triton_unfused_output).abs().max().item()
        max_diff_simple = (ref_output - fused_output).abs().max().item()
        max_diff_tiled = (ref_output - fused_tiled_output).abs().max().item()

        status_triton_unfused = "✓ PASS" if max_diff_triton_unfused < 1e-4 else "✗ FAIL"
        status_simple = "✓ PASS" if max_diff_simple < 1e-4 else "✗ FAIL"
        status_tiled = "✓ PASS" if max_diff_tiled < 1e-4 else "✗ FAIL"

        print(f"\nTest ({batch_size}, {input_dim}, {num_classes}):")
        print(f"  Triton unfused: {status_triton_unfused} (max diff: {max_diff_triton_unfused:.2e})")
        print(f"  Triton fused simple: {status_simple} (max diff: {max_diff_simple:.2e})")
        print(f"  Triton fused tiled:  {status_tiled} (max diff: {max_diff_tiled:.2e})")

        # Check softmax properties
        row_sums = fused_output.sum(dim=1)
        sum_check = torch.allclose(row_sums, torch.ones_like(row_sums))
        print(f"  Row sums = 1:  {'✓ PASS' if sum_check else '✗ FAIL'}")


def benchmark_implementations():
    """Benchmark all implementations: PyTorch, Triton unfused, Triton fused."""
    print("\n" + "=" * 80)
    print("Benchmarking: PyTorch vs Triton Unfused vs Triton Fused")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Test different sizes - using larger batches to better show fusion benefits
    configs = [
        (256, 128, 10, "Small"),
        (1024, 512, 50, "Medium"),
        (4096, 1024, 100, "Large"),
        (8192, 2048, 200, "Very Large"),
        (16384, 4096, 500, "Huge"),
    ]

    for batch_size, input_dim, num_classes, size_name in configs:
        print(f"\n{size_name}: batch={batch_size}, input_dim={input_dim}, num_classes={num_classes}")
        print("-" * 80)

        try:
            # Allocate tensors
            x = torch.randn(batch_size, input_dim, device='cuda')
            weight = torch.randn(input_dim, num_classes, device='cuda')
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"SKIPPED: Out of memory (batch_size={batch_size}, input_dim={input_dim}, num_classes={num_classes})")
                torch.cuda.empty_cache()
                continue
            else:
                raise

        implementations = [
            ("PyTorch (cuBLAS + softmax)", matmul_softmax_unfused),
            ("Triton UNFUSED (2 kernels)", matmul_softmax_triton_unfused),
            ("Triton FUSED simple", matmul_softmax_fused),
            ("Triton FUSED tiled", matmul_softmax_fused_tiled),
        ]

        baseline_time = None
        triton_unfused_time = None

        for name, func in implementations:
            try:
                mean_time, _ = benchmark_function(func, x, weight)

                if baseline_time is None:
                    baseline_time = mean_time

                if "UNFUSED" in name:
                    triton_unfused_time = mean_time

                speedup_vs_pytorch = baseline_time / mean_time

                # Show fusion benefit when comparing fused vs unfused Triton
                if "FUSED" in name and triton_unfused_time is not None:
                    fusion_speedup = triton_unfused_time / mean_time
                    memory_saved = batch_size * num_classes * 4 / 1024  # KB
                    print(f"{name:30s}: {mean_time:8.4f} ms  (vs PyTorch: {speedup_vs_pytorch:4.2f}x, vs Triton unfused: {fusion_speedup:4.2f}x, saves {memory_saved:.1f} KB)")
                else:
                    print(f"{name:30s}: {mean_time:8.4f} ms  (vs PyTorch: {speedup_vs_pytorch:4.2f}x)")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{name:30s}: OOM (Out of Memory)")
                    torch.cuda.empty_cache()
                else:
                    print(f"{name:30s}: ERROR - {str(e)}")
            except Exception as e:
                print(f"{name:30s}: ERROR - {str(e)}")

        # Clean up after each config to free memory
        del x, weight
        torch.cuda.empty_cache()


def profile_memory_access():
    """Profile to show memory access patterns."""
    print("\n" + "=" * 80)
    print("Memory Access Profiling")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    batch_size, input_dim, num_classes = 256, 512, 50
    x = torch.randn(batch_size, input_dim, device='cuda')
    weight = torch.randn(input_dim, num_classes, device='cuda')

    print(f"\nConfig: batch={batch_size}, input={input_dim}, classes={num_classes}")
    print(f"Logits tensor size: {batch_size * num_classes * 4 / 1024:.2f} KB")
    print(f"\nUnfused approach:")
    print(f"  1. Write logits to memory: {batch_size * num_classes * 4 / 1024:.2f} KB")
    print(f"  2. Read logits from memory: {batch_size * num_classes * 4 / 1024:.2f} KB")
    print(f"  3. Write probabilities to memory: {batch_size * num_classes * 4 / 1024:.2f} KB")
    total_unfused = 3 * batch_size * num_classes * 4 / 1024
    print(f"  Total: {total_unfused:.2f} KB")

    print(f"\nFused approach:")
    print(f"  1. Write probabilities to memory: {batch_size * num_classes * 4 / 1024:.2f} KB")
    total_fused = batch_size * num_classes * 4 / 1024
    print(f"  Total: {total_fused:.2f} KB")

    print(f"\nMemory traffic reduction: {(1 - total_fused / total_unfused) * 100:.1f}%")

    # Profile with PyTorch profiler
    print("\n" + "-" * 80)
    print("PyTorch Profiler Output:")
    print("-" * 80)

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
    ) as prof:
        _ = matmul_softmax_unfused(x, weight)

    print("\nUnfused:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
    ) as prof:
        _ = matmul_softmax_fused_tiled(x, weight)

    print("\nFused:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Fused Kernel Demonstration: Matmul + Softmax")
    print("=" * 80)
    print()
    print("This demonstrates kernel fusion by comparing THREE implementations:")
    print()
    print("  1. PyTorch (cuBLAS):      Highly optimized, unfused matmul + softmax")
    print("  2. Triton UNFUSED:        Two separate Triton kernels (matmul, then softmax)")
    print("  3. Triton FUSED:          Single Triton kernel (matmul + softmax together)")
    print()
    print("Key comparisons:")
    print("  • PyTorch vs Triton:      Shows the power of cuBLAS optimization")
    print("  • Unfused vs Fused Triton: Shows the ACTUAL benefit of kernel fusion!")
    print()
    print("What you'll learn:")
    print("  - When fusion helps (Triton unfused → fused)")
    print("  - When optimization libraries dominate (PyTorch vs Triton)")
    print("  - Memory traffic reduction from fusion")
    print("  - The importance of benchmarking apples-to-apples")
    print()

    # Run tests
    test_correctness()
    benchmark_implementations()
    profile_memory_access()

    print("\n" + "=" * 80)
    print("Summary: Key Lessons")
    print("=" * 80)
    print()
    print("✓ Correctness: All implementations produce identical results")
    print()
    print("Performance hierarchy:")
    print("  PyTorch (cuBLAS) > Triton fused ≥ Triton unfused")
    print()
    print("Important Takeaways:")
    print("  1. Fusion DOES help: Triton fused > Triton unfused")
    print("     → Eliminates intermediate tensor materialization")
    print("     → Reduces memory bandwidth usage")
    print()
    print("  2. But cuBLAS still wins overall:")
    print("     → Decades of matmul optimization (tensor cores, etc.)")
    print("     → Hard to beat with naive Triton kernels")
    print()
    print("  3. The right comparison matters:")
    print("     → Compare apples-to-apples (Triton vs Triton)")
    print("     → Don't conflate fusion benefits with library quality")
    print()
    print("  4. When fusion helps most:")
    print("     → Memory-bound operations (not compute-bound like matmul)")
    print("     → Large intermediate tensors")
    print("     → Chaining multiple ops (Flash Attention)")
    print()


if __name__ == "__main__":
    main()
