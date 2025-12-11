# Triton Backward Pass Implementation

This document explains the new `WeightedSumFunc_wtritonBackward` autograd function that uses Triton kernels for both forward and backward passes.

## Overview

We now have three implementations of the weighted sum operation in `weighted_sum.py`:

1. **`WeightedSumFunc`** - Forward only (no explicit backward)
2. **`WeightedSumFunc_wBackward`** - Forward with Triton, backward with PyTorch
3. **`WeightedSumFunc_wtritonBackward`** - Forward AND backward with Triton ⭐

## Triton Kernels

### Forward Pass Kernel

The forward kernel `weighted_sum_fwd` computes:
```
y[i] = sum_j (weight[j] * x[i, j])
```

This is a weighted sum over the feature dimension for each sample.

### Backward Pass Kernels

We need two separate backward kernels because we compute gradients with respect to two different tensors:

#### 1. `weighted_sum_bwd_dx` - Gradient w.r.t. input x

Computes:
```
grad_x[i, j] = grad_output[i] * weight[j]
```

**Algorithm:**
- Each thread block processes `ROWS_TILE_SIZE` rows
- For each row, broadcast the gradient output to all dimensions
- Multiply by the weight vector
- Write result to grad_x

**Grid configuration:** `(ROWS / ROWS_TILE_SIZE,)` blocks

#### 2. `weighted_sum_bwd_dw` - Gradient w.r.t. weight

Computes:
```
grad_weight[j] = sum_i (grad_output[i] * x[i, j])
```

**Algorithm:**
- Each thread block processes `D_TILE_SIZE` dimensions
- Loop over all rows and accumulate contributions
- For each row tile, load grad_output and x values
- Compute elementwise product and sum over rows
- Accumulate into grad_weight

**Grid configuration:** `(D / D_TILE_SIZE,)` blocks

## Key Implementation Details

### Memory Layout

Both kernels use block pointers (`tl.make_block_ptr`) for efficient memory access:
- Handles boundary conditions automatically
- Supports strided tensors
- Optimized memory coalescing

### Tiling Strategy

**Forward and dx kernels:**
- Process rows in tiles of size `ROWS_TILE_SIZE = 16`
- Process dimensions in tiles of size `D_TILE_SIZE = next_power_of_2(D) // 16`

**dw kernel:**
- Processes dimensions in tiles (outer loop over dimensions)
- Accumulates over all rows (inner loop over rows)
- This maximizes reuse and minimizes atomic operations

### Gradient Accumulation

The `weighted_sum_bwd_dw` kernel uses a local accumulator:
```python
grad_w_acc = tl.zeros((D_TILE_SIZE,), dtype=tl.float32)
```

This accumulator is updated in the loop over rows, and only written once at the end, avoiding repeated global memory writes.

## Usage

```python
from weighted_sum import f_weightedsum_triton_backward

# Use in your model
class MyModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_dim, device='cuda'))

    def forward(self, x):
        return f_weightedsum_triton_backward(x, self.weight)

# Train normally - gradients are computed with Triton kernels
model = MyModel(128)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(64, 128, device='cuda')
y = model(x)
loss = y.sum()
loss.backward()  # Uses Triton backward kernels!
optimizer.step()
```

## Testing

Run the test function to verify correctness:

```python
test_triton_backward()
```

This compares:
- Forward pass output between PyTorch and Triton versions
- Gradients (grad_x and grad_weight) between PyTorch and Triton versions

Both should match within numerical tolerance (atol=1e-5).

## Performance Considerations

**Advantages of Triton backward:**
1. **Fused operations** - No intermediate tensor materialization
2. **Optimized memory access** - Block pointers and tiling
3. **Reduced Python overhead** - Entire backward pass in compiled kernel

**When to use:**
- Large batch sizes (takes advantage of parallelism)
- Large feature dimensions (better tiling efficiency)
- Memory-constrained scenarios (less intermediate storage)

**When PyTorch might be faster:**
- Very small tensors (kernel launch overhead dominates)
- Irregular tensor shapes (less efficient tiling)

## Comparison Table

| Feature | WeightedSumFunc | WeightedSumFunc_wBackward | WeightedSumFunc_wtritonBackward |
|---------|----------------|---------------------------|--------------------------------|
| Forward | Triton kernel | Triton kernel | Triton kernel |
| Backward | Auto (PyTorch) | PyTorch operations | Triton kernels |
| Performance | Good | Good | Best |
| Flexibility | High | High | Medium |
| Complexity | Low | Medium | High |

## Mathematical Derivation

Given the forward pass: `y = Xw` where:
- `X` is (ROWS, D)
- `w` is (D,)
- `y` is (ROWS,)

**Gradient w.r.t. X:**
```
∂L/∂X[i,j] = ∂L/∂y[i] * ∂y[i]/∂X[i,j]
           = ∂L/∂y[i] * w[j]
           = grad_output[i] * weight[j]
```

**Gradient w.r.t. w:**
```
∂L/∂w[j] = Σ_i (∂L/∂y[i] * ∂y[i]/∂w[j])
         = Σ_i (∂L/∂y[i] * X[i,j])
         = Σ_i (grad_output[i] * x[i,j])
```

These are exactly what our Triton kernels compute!
