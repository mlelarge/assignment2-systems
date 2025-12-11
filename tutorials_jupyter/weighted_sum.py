import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange
from torch.profiler import ProfilerActivity


def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row,
    x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
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
        row = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )  # Move by D_TILE_SIZE

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

    D_TILE_SIZE = (
        triton.next_power_of_2(D) // 16
    )  # Roughly 16 loops through the embedding dimension
    ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time

    # Launch our kernel with n instances in our 1D grid.
    n_rows = y.numel()
    weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
        x,
        weight,
        y,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        y.stride(0),
        ROWS=n_rows,
        D=D,
        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        D_TILE_SIZE=D_TILE_SIZE,
    )

    return y.view(output_dims)


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        x_reshaped = rearrange(x, "... d -> (...) d")

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert (
            x_reshaped.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous x"

        D_TILE_SIZE = (
            triton.next_power_of_2(D) // 16
        )  # Roughly 16 loops through the embedding dimension
        ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(x_reshaped.shape[0], device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x_reshaped,
            weight,
            y,
            x_reshaped.stride(0),
            x_reshaped.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        return y.view(output_dims)


class WeightedSumFunc_wBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        x_reshaped = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x_reshaped, weight)
        ctx.output_dims = output_dims

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert (
            x_reshaped.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous x"

        D_TILE_SIZE = (
            triton.next_power_of_2(D) // 16
        )  # Roughly 16 loops through the embedding dimension
        ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(x_reshaped.shape[0], device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x_reshaped,
            weight,
            y,
            x_reshaped.stride(0),
            x_reshaped.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        return y.view(output_dims)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, weight = ctx.saved_tensors

        # Reshape grad_output to match forward pass
        grad_output_flat = grad_output.reshape(-1)

        # Gradient wrt weight: sum over all samples
        # d/dw (w^T x) = x
        # So grad_weight = sum_i grad_output[i] * x[i]
        grad_weight = (grad_output_flat[:, None] * x).sum(dim=0)

        # Gradient wrt x: broadcast weight
        # d/dx (w^T x) = w
        # So grad_x = grad_output * w
        grad_x = grad_output_flat[:, None] * weight[None, :]

        # Reshape grad_x back to original shape
        grad_x = grad_x.view(*ctx.output_dims, -1)

        return grad_x, grad_weight


@triton.jit
def weighted_sum_bwd_dx(
    grad_output_ptr,  # Input: gradient from upstream (ROWS,)
    weight_ptr,  # Input: weight vector (D,)
    grad_x_ptr,  # Output: gradient wrt x (ROWS, D)
    grad_output_stride,
    weight_stride,
    grad_x_stride_row,
    grad_x_stride_dim,
    ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Backward pass for computing gradient wrt input x.
    grad_x = grad_output[:, None] * weight[None, :]
    Each row of grad_x is grad_output[i] * weight
    """
    row_tile_idx = tl.program_id(0)

    # Block pointer for grad_output (1D vector)
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(ROWS,),
        strides=(grad_output_stride,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Block pointer for grad_x (2D matrix)
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(ROWS, D),
        strides=(grad_x_stride_row, grad_x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # Block pointer for weight (1D vector)
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # Load grad_output for this row tile (ROWS_TILE_SIZE,)
    grad_out = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

    # Process weight in tiles
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load weight tile (D_TILE_SIZE,)
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # Compute grad_x = grad_output[:, None] * weight[None, :]
        # grad_out: (ROWS_TILE_SIZE,), weight_tile: (D_TILE_SIZE,)
        # Result: (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_x_tile = grad_out[:, None] * weight_tile[None, :]

        # Store result
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))

        # Advance pointers
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


@triton.jit
def weighted_sum_bwd_dw(
    grad_output_ptr,  # Input: gradient from upstream (ROWS,)
    x_ptr,  # Input: input matrix (ROWS, D)
    grad_weight_ptr,  # Output: gradient wrt weight (D,)
    grad_output_stride,
    x_stride_row,
    x_stride_dim,
    grad_weight_stride,
    ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Backward pass for computing gradient wrt weight.
    grad_weight = sum_i grad_output[i] * x[i]
    This kernel processes one dimension tile at a time.
    """
    dim_tile_idx = tl.program_id(0)

    # Block pointer for grad_weight (1D vector)
    grad_weight_block_ptr = tl.make_block_ptr(
        grad_weight_ptr,
        shape=(D,),
        strides=(grad_weight_stride,),
        offsets=(dim_tile_idx * D_TILE_SIZE,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # Initialize accumulator for this dimension tile
    grad_w_acc = tl.zeros((D_TILE_SIZE,), dtype=tl.float32)

    # Loop over rows
    for row_tile_idx in range(tl.cdiv(ROWS, ROWS_TILE_SIZE)):
        # Block pointer for grad_output
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(ROWS,),
            strides=(grad_output_stride,),
            offsets=(row_tile_idx * ROWS_TILE_SIZE,),
            block_shape=(ROWS_TILE_SIZE,),
            order=(0,),
        )

        # Block pointer for x
        x_block_ptr = tl.make_block_ptr(
            x_ptr,
            shape=(ROWS, D),
            strides=(x_stride_row, x_stride_dim),
            offsets=(row_tile_idx * ROWS_TILE_SIZE, dim_tile_idx * D_TILE_SIZE),
            block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )

        # Load grad_output (ROWS_TILE_SIZE,)
        grad_out = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

        # Load x (ROWS_TILE_SIZE, D_TILE_SIZE)
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute contribution: grad_output[:, None] * x
        # Result shape: (ROWS_TILE_SIZE, D_TILE_SIZE)
        contrib = grad_out[:, None] * x_tile

        # Sum over rows to accumulate into grad_weight
        grad_w_acc += tl.sum(contrib, axis=0)

    # Store final result
    tl.store(grad_weight_block_ptr, grad_w_acc, boundary_check=(0,))


class WeightedSumFunc_wtritonBackward(torch.autograd.Function):
    """
    Weighted sum autograd function with Triton kernels for both forward and backward passes.
    """

    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        x_reshaped = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x_reshaped, weight)
        ctx.output_dims = output_dims

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x_reshaped.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ROWS_TILE_SIZE = 16

        # Need to initialize empty result tensor
        y = torch.empty(x_reshaped.shape[0], device=x.device)

        # Launch forward kernel
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x_reshaped,
            weight,
            y,
            x_reshaped.stride(0),
            x_reshaped.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        return y.view(output_dims)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, weight = ctx.saved_tensors

        # Reshape grad_output to match forward pass
        grad_output_flat = grad_output.reshape(-1).contiguous()

        # Get dimensions
        ROWS, D = x.shape
        D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ROWS_TILE_SIZE = 16

        # Allocate output gradients
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)

        # Launch backward kernel for grad_x
        weighted_sum_bwd_dx[(triton.cdiv(ROWS, ROWS_TILE_SIZE),)](
            grad_output_flat,
            weight,
            grad_x,
            grad_output_flat.stride(0),
            weight.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            ROWS=ROWS,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # Launch backward kernel for grad_weight
        weighted_sum_bwd_dw[(triton.cdiv(D, D_TILE_SIZE),)](
            grad_output_flat,
            x,
            grad_weight,
            grad_output_flat.stride(0),
            x.stride(0),
            x.stride(1),
            grad_weight.stride(0),
            ROWS=ROWS,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # Reshape grad_x back to original shape
        grad_x = grad_x.view(*ctx.output_dims, -1)

        return grad_x, grad_weight


# Use the version with PyTorch backward by default
f_weightedsum = WeightedSumFunc_wBackward.apply

# Use the version with Triton backward (more efficient)
f_weightedsum_triton_backward = WeightedSumFunc_wtritonBackward.apply

# Version without backward (will fail if used in training)
f_weightedsum_no_backward = torch.compile(WeightedSumFunc.apply)

# f_weightedsum_compile = (f_weightedsum)


# ============================================================================
# Linear Regression Example
# ============================================================================


class LinearRegressionTriton(torch.nn.Module):
    """
    Linear regression using the custom Triton weighted sum kernel.

    Model: y = w^T x + b
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_dim, device="cuda") * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(1, device="cuda"))

    def forward(self, x):
        # x: (batch_size, input_dim)
        # Use our custom weighted sum function
        return f_weightedsum(x, self.weight) + self.bias


def generate_regression_data(n_samples=1000, input_dim=128, noise_std=0.1, seed=42):
    """
    Generate synthetic linear regression data.

    Returns:
        X: (n_samples, input_dim) feature matrix
        y: (n_samples,) continuous target values
        true_weight: (input_dim,) true weight vector used for generation
        true_bias: (1,) true bias value used for generation
    """
    torch.manual_seed(seed)

    # Generate random features
    X = torch.randn(n_samples, input_dim, device="cuda")

    # Create true weights for data generation
    true_weight = torch.randn(input_dim, device="cuda")
    true_bias = torch.randn(1, device="cuda")

    # Generate target values: y = w^T x + b + noise
    y = X @ true_weight + true_bias

    # Add Gaussian noise
    noise = torch.randn(n_samples, device="cuda") * noise_std
    y = y + noise

    return X, y, true_weight, true_bias


def train_linear_regression(
    model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, batch_size=64
):
    """
    Train the linear regression model.

    Args:
        model: LinearRegressionTriton model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training

    Returns:
        train_losses: List of training losses (MSE) per epoch
        val_losses: List of validation losses (MSE) per epoch
        val_r2_scores: List of validation R² scores per epoch
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    val_r2_scores = []

    n_batches = (len(X_train) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        # Shuffle data
        perm = torch.randperm(len(X_train), device="cuda")
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val).item()
            val_losses.append(val_loss)

            # Calculate R² score
            ss_res = ((y_val - y_val_pred) ** 2).sum()
            ss_tot = ((y_val - y_val.mean()) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot)
            val_r2_scores.append(r2_score.item())

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val R² = {r2_score.item():.4f}"
            )

    return train_losses, val_losses, val_r2_scores


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def benchmark(description: str, func, x, weight, num_warmups: int = 3, num_trials: int = 10):
    """Benchmark a weighted sum function by running it multiple times."""
    # Warmup: first times might be slower due to compilation
    for _ in range(num_warmups):
        func(x, weight)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time it for real now
    times: list[float] = []
    for trial in range(num_trials):
        start_time = time.time()
        result = func(x, weight)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    mean_time = mean(times)
    return mean_time, result


def profile(description: str, func, x, weight, num_warmups: int = 1):
    """Profile a weighted sum function using PyTorch profiler."""
    # Warmup
    for _ in range(num_warmups):
        func(x, weight)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Run with profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        func(x, weight)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Print table
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=80,
        row_limit=10
    )
    return table


def benchmark_all_implementations():
    """Benchmark all weighted sum implementations."""
    print("=" * 80)
    print("Benchmarking Weighted Sum Implementations")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return

    # Test different sizes
    sizes = [
        (64, 128),      # Small
        (256, 512),     # Medium
        (1024, 2048),   # Large
    ]

    for batch_size, input_dim in sizes:
        print(f"\n{'='*80}")
        print(f"Batch size: {batch_size}, Input dim: {input_dim}")
        print(f"{'='*80}\n")

        # Create test data
        x = torch.randn(batch_size, input_dim, device='cuda')
        weight = torch.randn(input_dim, device='cuda')

        # Benchmark each implementation
        implementations = [
            ("PyTorch native (matmul)", lambda x, w: x @ w),
            ("Manual PyTorch", weighted_sum),
            ("Triton forward only", weighted_sum_triton),
            ("Triton w/ PyTorch backward", lambda x, w: WeightedSumFunc_wBackward.apply(x, w)),
            ("Triton w/ Triton backward", lambda x, w: WeightedSumFunc_wtritonBackward.apply(x, w)),
        ]

        results = {}
        baseline_time = None

        for name, func in implementations:
            try:
                mean_time, result = benchmark(name, func, x, weight)
                results[name] = mean_time

                if baseline_time is None:
                    baseline_time = mean_time

                speedup = baseline_time / mean_time
                print(f"{name:35s}: {mean_time:8.4f} ms (speedup: {speedup:.2f}x)")

            except Exception as e:
                print(f"{name:35s}: ERROR - {str(e)}")

        # Verify all implementations produce the same result
        print(f"\n{'='*80}")
        print("Correctness Check")
        print(f"{'='*80}")

        baseline_result = x @ weight
        for name, func in implementations[1:]:  # Skip first (it's the baseline)
            try:
                _, result = benchmark(name, func, x, weight, num_warmups=1, num_trials=1)
                max_diff = (result - baseline_result).abs().max().item()
                status = "✓ PASS" if max_diff < 1e-4 else "✗ FAIL"
                print(f"{name:35s}: {status} (max diff: {max_diff:.2e})")
            except Exception as e:
                print(f"{name:35s}: ✗ ERROR - {str(e)}")


def profile_implementations():
    """Profile weighted sum implementations with PyTorch profiler."""
    print("\n" + "=" * 80)
    print("Profiling Weighted Sum Implementations")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        return

    # Use medium size for profiling
    batch_size, input_dim = 256, 512
    x = torch.randn(batch_size, input_dim, device='cuda')
    weight = torch.randn(input_dim, device='cuda')

    implementations = [
        ("PyTorch matmul", lambda x, w: x @ w),
        ("Triton forward", weighted_sum_triton),
        ("Triton w/ Triton backward", lambda x, w: WeightedSumFunc_wtritonBackward.apply(x, w)),
    ]

    for name, func in implementations:
        print(f"\n{'='*80}")
        print(f"Profile: {name}")
        print(f"{'='*80}\n")
        try:
            table = profile(name, func, x, weight)
            print(table)
        except Exception as e:
            print(f"ERROR: {str(e)}")


def test_triton_backward():
    """
    Test that the Triton backward pass produces the same gradients as PyTorch.
    """
    print("=" * 80)
    print("Testing Triton Backward Pass")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    # Test dimensions
    batch_size = 64
    input_dim = 128

    # Create test inputs
    x1 = torch.randn(batch_size, input_dim, device='cuda', requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    weight1 = torch.randn(input_dim, device='cuda', requires_grad=True)
    weight2 = weight1.clone().detach().requires_grad_(True)

    # Forward pass with PyTorch backward
    y1 = WeightedSumFunc_wBackward.apply(x1, weight1)
    loss1 = y1.sum()
    loss1.backward()

    # Forward pass with Triton backward
    y2 = WeightedSumFunc_wtritonBackward.apply(x2, weight2)
    loss2 = y2.sum()
    loss2.backward()

    # Check that outputs match
    print(f"\nForward pass comparison:")
    print(f"  Max difference in output: {(y1 - y2).abs().max().item():.2e}")
    assert torch.allclose(y1, y2, atol=1e-5), "Forward outputs don't match!"
    print(f"  ✓ Forward outputs match")

    # Check that gradients match
    print(f"\nBackward pass comparison:")
    print(f"  Max difference in grad_x: {(x1.grad - x2.grad).abs().max().item():.2e}")
    print(f"  Max difference in grad_weight: {(weight1.grad - weight2.grad).abs().max().item():.2e}")

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), "grad_x doesn't match!"
    assert torch.allclose(weight1.grad, weight2.grad, atol=1e-5), "grad_weight doesn't match!"

    print(f"  ✓ Gradients match!")
    print()


def main():
    """
    Main function to demonstrate linear regression with custom Triton kernel.
    """
    print("=" * 80)
    print("Linear Regression with Custom Triton Weighted Sum Kernel")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return

    # Hyperparameters
    n_train = 800
    n_val = 200
    input_dim = 128
    epochs = 100
    lr = 0.01
    batch_size = 64
    noise_std = 0.1

    print(f"\nDataset configuration:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Noise std: {noise_std}")
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print()

    # Generate data
    print("Generating synthetic linear regression data...")
    X, y, true_weight, true_bias = generate_regression_data(
        n_samples=n_train + n_val, input_dim=input_dim, noise_std=noise_std
    )

    # Split into train and validation
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print(f"Training set: X shape = {X_train.shape}, y shape = {y_train.shape}")
    print(f"Validation set: X shape = {X_val.shape}, y shape = {y_val.shape}")
    print(f"\nTrue parameters:")
    print(f"  True weight norm: {true_weight.norm().item():.6f}")
    print(f"  True bias: {true_bias.item():.6f}")
    print()

    # Initialize model
    print("Initializing linear regression model with Triton kernel...")
    model = LinearRegressionTriton(input_dim=input_dim)
    print(
        f"Model parameters: weight shape = {model.weight.shape}, bias shape = {model.bias.shape}"
    )
    print()

    # Train model
    print("Starting training...\n")
    train_losses, val_losses, val_r2_scores = train_linear_regression(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Training Loss (MSE): {train_losses[-1]:.4f}")
    print(f"Final Validation Loss (MSE): {val_losses[-1]:.4f}")
    print(f"Final Validation R²: {val_r2_scores[-1]:.4f}")
    print()

    # Compare learned parameters with true parameters
    print("=" * 80)
    print("Parameter Comparison: Learned vs True")
    print("=" * 80)

    learned_weight = model.weight.data
    learned_bias = model.bias.data

    # Compute various comparison metrics
    weight_diff = learned_weight - true_weight
    weight_mse = (weight_diff**2).mean().item()
    weight_mae = weight_diff.abs().mean().item()
    weight_cos_sim = torch.nn.functional.cosine_similarity(
        learned_weight.unsqueeze(0), true_weight.unsqueeze(0)
    ).item()

    bias_diff = (learned_bias - true_bias).abs().item()

    # Compute correlation
    weight_corr = torch.corrcoef(torch.stack([learned_weight, true_weight]))[
        0, 1
    ].item()

    print(f"\nWeight Statistics:")
    print(f"  True weight norm:        {true_weight.norm().item():.6f}")
    print(f"  Learned weight norm:     {learned_weight.norm().item():.6f}")
    print(f"  Weight MSE:              {weight_mse:.6f}")
    print(f"  Weight MAE:              {weight_mae:.6f}")
    print(f"  Weight cosine similarity: {weight_cos_sim:.6f}")
    print(f"  Weight correlation:      {weight_corr:.6f}")

    print(f"\nBias Statistics:")
    print(f"  True bias:               {true_bias.item():.6f}")
    print(f"  Learned bias:            {learned_bias.item():.6f}")
    print(f"  Bias absolute difference: {bias_diff:.6f}")

    # Show top 5 weight components comparison
    print(f"\nTop 5 Weight Components (by absolute value of true weight):")
    top_indices = true_weight.abs().argsort(descending=True)[:5]
    print(f"  {'Index':<8} {'True':<12} {'Learned':<12} {'Difference':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for idx in top_indices:
        idx_val = idx.item()
        true_val = true_weight[idx].item()
        learned_val = learned_weight[idx].item()
        diff = learned_val - true_val
        print(f"  {idx_val:<8} {true_val:>12.6f} {learned_val:>12.6f} {diff:>12.6f}")

    print()

    # Test gradient computation
    print("=" * 80)
    print("Gradient Computation Test")
    print("=" * 80)
    model.eval()

    # Clear any existing gradients
    model.zero_grad()

    X_test = torch.randn(10, input_dim, device="cuda", requires_grad=True)
    y_test_target = torch.randn(10, device="cuda")
    y_pred = model(X_test)
    loss = torch.nn.MSELoss()(y_pred, y_test_target)
    loss.backward()

    print(f"✓ Gradients computed successfully!")
    print(f"  Test loss: {loss.item():.6f}")
    print(f"  Weight gradient norm: {model.weight.grad.norm().item():.6f}")
    print(f"  Bias gradient norm: {model.bias.grad.norm().item():.6f}")
    if X_test.grad is not None:
        print(f"  Input gradient norm: {X_test.grad.norm().item():.6f}")
    print()

    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Check if user wants to run benchmarks/profiling or training
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Run benchmarks and profiling only
        test_triton_backward()
        benchmark_all_implementations()
        profile_implementations()
    else:
        # Default: run training example
        # First test the Triton backward pass
        test_triton_backward()

        # Optionally run benchmarks (comment out if you want to skip)
        print("\nRunning quick benchmark...")
        benchmark_all_implementations()

        # Then run the full training example
        main()
