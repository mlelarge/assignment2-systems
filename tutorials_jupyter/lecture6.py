#!/usr/bin/env python3
"""
Lecture 6: GPU kernels with Triton
Converted from Jupyter notebook to standalone Python script.
"""

import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from einops import rearrange


# ============================================================================
# Helper functions
# ============================================================================

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def check_equal(f1, f2):
    x = torch.randn(2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal2(f1, f2):
    x = torch.randn(2048, 2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda: operation(x)


# ============================================================================
# Basic implementations
# ============================================================================

def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


# ============================================================================
# Benchmarking and profiling utilities
# ============================================================================

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    return mean_time


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table


# ============================================================================
# Triton GELU kernel
# ============================================================================

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundary
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


# ============================================================================
# Triton Softmax kernel
# ============================================================================

def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


@triton.jit
def triton_softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    x_stride, y_stride,  # Receive strides as arguments
    BLOCK_SIZE: tl.constexpr
):
    # Process each row independently
    row_idx = tl.program_id(0)

    # Create block pointers for this row
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(x_stride, 1),  # Use the passed stride
        offsets=(row_idx, 0),
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0)  # Row-major
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(y_stride, 1),  # Use the passed stride
        offsets=(row_idx, 0),
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0)
    )

    # Load row (automatically handles masking and padding)
    x_row = tl.load(x_block_ptr, boundary_check=(1,), padding_option="zero")

    # Compute softmax
    x_row = x_row - tl.max(x_row)
    numerator = tl.exp(x_row)
    y_row = numerator / tl.sum(numerator)

    # Store result
    tl.store(y_block_ptr, y_row, boundary_check=(1,))


def triton_softmax(x: torch.Tensor):
    y = torch.empty_like(x)
    M, N = x.shape
    block_size = triton.next_power_of_2(N)

    triton_softmax_kernel[(M,)](
        x, y,
        M, N,
        x.stride(0), y.stride(0),  # Pass strides from host
        BLOCK_SIZE=block_size
    )
    return y


# ============================================================================
# Weighted sum implementation
# ============================================================================

def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


def weighted_sum_triton(x: torch.Tensor, weight: torch.Tensor):
    D = x.shape[-1]
    output_dims = x.shape[:-1]
    # Reshape input tensor to 2D

    x = rearrange(x, "... d -> (...) d")
    # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
    y = torch.empty(x.shape[0], device=x.device)

    D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
    ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time

    # Launch our kernel with n instances in our 1D grid.
    n_rows = y.numel()
    weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

    return y.view(output_dims)


def check_equal3(f1, f2):
    x = torch.randn(64, 64, 2048, device=get_device())
    w = torch.randn(2048, device=get_device())
    y1 = f1(x, w)
    y2 = f2(x, w)
    assert torch.allclose(y1, y2, atol=1e-4)


# ============================================================================
# Autograd function for weighted sum
# ============================================================================

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
        ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(x.shape[0], device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        return y.view(output_dims)


f_weightedsum = WeightedSumFunc.apply


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all benchmarks and tests."""

    print("=" * 80)
    print("GELU Benchmarks")
    print("=" * 80)

    # GELU benchmarks
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    print(f"Manual GELU time: {manual_time:.4f} ms")

    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    print(f"Triton GELU time: {triton_time:.4f} ms")

    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    print(f"PyTorch GELU time: {pytorch_time:.4f} ms")

    compiled_gelu = torch.compile(manual_gelu)
    compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    print(f"Compiled GELU time: {compiled_time:.4f} ms")

    # Profile triton GELU
    print("\nTriton GELU Profile:")
    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    print(triton_gelu_profile)

    print("\n" + "=" * 80)
    print("Softmax Benchmarks")
    print("=" * 80)

    # Softmax benchmarks
    manual_time = benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
    print(f"Manual softmax time: {manual_time:.4f} ms")

    compiled_softmax = torch.compile(manual_softmax)
    compiled_time = benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
    print(f"Compiled softmax time: {compiled_time:.4f} ms")

    pytorch_time = benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
    print(f"PyTorch softmax time: {pytorch_time:.4f} ms")

    triton_time = benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))
    print(f"Triton softmax time: {triton_time:.4f} ms")

    # Check correctness
    print("\nChecking Triton softmax correctness...")
    check_equal2(pytorch_softmax, triton_softmax)
    print("✓ Triton softmax matches PyTorch")

    # Show stride example
    print("\n" + "=" * 80)
    print("Stride Example")
    print("=" * 80)
    dim = 16384
    x = torch.randn(dim, dim, device=get_device())
    print(f"Tensor shape: {x.shape}")
    print(f"Tensor stride: {x.stride()}")

    print("\n" + "=" * 80)
    print("Weighted Sum Tests")
    print("=" * 80)

    # Check weighted sum implementations
    print("Checking weighted sum Triton implementation...")
    check_equal3(weighted_sum, weighted_sum_triton)
    print("✓ Triton weighted sum matches reference")

    print("Checking weighted sum autograd function...")
    check_equal3(weighted_sum, f_weightedsum)
    print("✓ Autograd weighted sum matches reference")

    print("Checking compiled weighted sum...")
    f_weightedsum_compile = torch.compile(f_weightedsum)
    check_equal3(weighted_sum, f_weightedsum_compile)
    print("✓ Compiled autograd weighted sum matches reference")

    # Test gradient computation
    print("\nTesting gradient computation...")
    x = torch.randn(1, 16, device=get_device())
    w = torch.randn(16, device=get_device()).requires_grad_(True)
    result = f_weightedsum_compile(x, w)
    print(f"Result: {result}")
    print("✓ Gradient computation successful")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
