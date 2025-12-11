# Weighted Sum Benchmarking Guide

This guide explains how to benchmark and profile the different weighted sum implementations.

## Available Implementations

1. **PyTorch native (matmul)**: `x @ w` - Standard PyTorch matrix multiplication
2. **Manual PyTorch**: `(weight * x).sum(axis=-1)` - Element-wise multiply and sum
3. **Triton forward only**: Custom Triton kernel for forward pass only
4. **Triton w/ PyTorch backward**: Triton forward + PyTorch operations for backward
5. **Triton w/ Triton backward**: Both forward and backward using Triton kernels

## Running Benchmarks

### Option 1: Standalone Benchmark Script (Recommended)

Run only benchmarks and profiling without training:

```bash
python tutorials_jupyter/benchmark_weighted_sum.py
```

### Option 2: Using the Main Script

Run benchmarks along with training:

```bash
python tutorials_jupyter/weighted_sum.py
```

Run benchmarks only (no training):

```bash
python tutorials_jupyter/weighted_sum.py --benchmark
```

## What Gets Benchmarked

### Performance Benchmarking

The `benchmark_all_implementations()` function tests three different problem sizes:

- **Small**: Batch size 64, Input dim 128
- **Medium**: Batch size 256, Input dim 512
- **Large**: Batch size 1024, Input dim 2048

For each size, it:
1. Runs 3 warmup iterations (to compile kernels)
2. Runs 10 timed iterations
3. Reports mean execution time in milliseconds
4. Calculates speedup relative to PyTorch native (matmul)

### Correctness Testing

After benchmarking, each implementation is verified to produce the same output as the baseline (PyTorch matmul) within numerical tolerance (1e-4).

### Profiling

The `profile_implementations()` function uses PyTorch's profiler to show:
- CPU time
- CUDA kernel time
- Memory operations
- Detailed breakdown of operations

Profiling uses medium-sized inputs (256 x 512) and shows:
- Self CPU time
- Self CUDA time
- Total CPU/CUDA time
- Number of calls

## Understanding the Results

### Expected Performance Characteristics

**Forward Pass Performance:**
- PyTorch matmul is highly optimized (uses cuBLAS)
- Triton kernels may be competitive for medium sizes
- Overhead matters more for small inputs
- Large batch sizes favor parallelism

**Backward Pass Performance:**
- Triton backward should be faster than PyTorch operations
- Fused kernels reduce memory traffic
- Reduction operations (sum) are critical

### Reading Benchmark Output

Example output:
```
PyTorch native (matmul)            :   0.1234 ms (speedup: 1.00x)
Manual PyTorch                     :   0.2345 ms (speedup: 0.53x)
Triton forward only                :   0.1456 ms (speedup: 0.85x)
Triton w/ PyTorch backward         :   0.1567 ms (speedup: 0.79x)
Triton w/ Triton backward          :   0.1234 ms (speedup: 1.00x)
```

**Interpreting:**
- Lower ms = faster
- Speedup > 1.0 = faster than baseline
- Speedup < 1.0 = slower than baseline
- Speedup = 1.0 = same speed as baseline

### Reading Profile Output

Example profiler table:
```
Name                    Self CPU %   Self CPU   CPU total %   CPU total   CUDA total
----------------------  -----------  ---------  ------------  ----------  -----------
weighted_sum_fwd               0.00%    0.000us         0.00%    0.000us     0.600ms
aten::empty_like              28.25%    0.614ms        71.65%    1.559ms     0.000ms
cuLaunchKernel                 1.67%    0.036ms         1.67%    0.036ms     0.000ms
```

**Key columns:**
- **Self CUDA**: Time spent in CUDA kernels (this is what matters for GPU performance)
- **CPU total**: Time spent in CPU operations
- **CUDA total**: Total CUDA time including nested operations

## Performance Tips

1. **Kernel Launch Overhead**: Small tensors suffer from launch overhead. Triton kernels are better for medium-to-large inputs.

2. **Memory Bandwidth**: The weighted sum is memory-bound. Performance depends on:
   - Coalesced memory accesses
   - Minimizing global memory traffic
   - Effective use of shared memory

3. **Tiling Strategy**: Our kernels use:
   - `ROWS_TILE_SIZE = 16` (process 16 rows at a time)
   - `D_TILE_SIZE = next_power_of_2(D) // 16` (tile dimensions)

4. **Backward Pass**: The Triton backward pass should excel because:
   - It fuses operations that PyTorch does separately
   - Reduces intermediate tensor materialization
   - Better memory locality

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors, reduce the problem sizes in `benchmark_all_implementations()`:

```python
sizes = [
    (32, 64),    # Smaller
    (128, 256),
    (512, 1024),
]
```

### Slow First Run

First run is always slower due to:
- Kernel compilation (Triton)
- CUDA initialization
- Cache warming

This is why we use warmup iterations.

### Inconsistent Timings

GPU timing can vary due to:
- Other processes using the GPU
- GPU clock throttling
- Cache effects

Run multiple times and look at trends, not absolute numbers.

## Comparing with Lecture 6

This benchmarking setup is similar to `lecture6.py` but focused specifically on weighted sum operations:

**Similarities:**
- Uses `benchmark()` and `profile()` helper functions
- Runs multiple warmup and trial iterations
- Uses PyTorch profiler for detailed analysis
- Reports mean times and speedups

**Differences:**
- Tests vector-matrix operations instead of element-wise ops
- Includes backward pass implementations
- Tests multiple tensor sizes
- Includes correctness verification

## Example Use Cases

### Quick Check

Just verify correctness:
```python
test_triton_backward()
```

### Performance Comparison

Compare all implementations:
```python
benchmark_all_implementations()
```

### Deep Dive

Understand where time is spent:
```python
profile_implementations()
```

### Custom Benchmark

Test specific sizes:
```python
from weighted_sum import benchmark, weighted_sum_triton

x = torch.randn(1024, 4096, device='cuda')
w = torch.randn(4096, device='cuda')

time, result = benchmark("custom", weighted_sum_triton, x, w)
print(f"Time: {time:.4f} ms")
```
