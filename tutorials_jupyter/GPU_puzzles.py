#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numba
import numpy as np
from numba import cuda


# In[2]:


import warnings

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)


# In[3]:


def map_spec(a):
    return a + 10


# In[4]:


# Define the CUDA kernel
@cuda.jit
def map_kernel(out, a):
    # Get the thread index
    i = cuda.threadIdx.x
    # Each thread adds 10 to one element
    out[i] = a[i] + 10


# Size of our array
SIZE = 4

# Create input and output arrays
a = np.arange(SIZE, dtype=np.float32)  # [0, 1, 2, 3]
out = np.zeros(SIZE, dtype=np.float32)

# Copy arrays to GPU
a_device = cuda.to_device(a)
out_device = cuda.to_device(out)

# kernel[grid, block](args)
# Launch kernel: grid = 1 block, block = SIZE threads
map_kernel[1, SIZE](out_device, a_device)

# Copy result back to CPU
result = out_device.copy_to_host()

# Verify result
expected = map_spec(a)
print(f"Input:    {a}")
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[5]:


def zip_spec(a, b):
    return a + b


# In[6]:


out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(SIZE)
zip_spec(a, b)


# In[7]:


# Define the CUDA kernel
@cuda.jit
def zip_kernel(out, a, b):
    # Get the thread index
    i = cuda.threadIdx.x

    out[i] = a[i] + b[i]


def init_pb(a=a, b=b, out=out):
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    out_device = cuda.to_device(out)
    return a_device, b_device, out_device


a_device, b_device, out_device = init_pb()

# Launch kernel: 1 block, SIZE threads
zip_kernel[1, SIZE](out_device, a_device, b_device)

# Copy result back to CPU
result = out_device.copy_to_host()

# Verify result
expected = zip_spec(a, b)
print(f"Input a:  {a}")
print(f"Input b:  {b}")
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[8]:


# CUDA kernel with Guard
@cuda.jit
def zip_guard_kernel(out, a, b, size):
    # Get the thread index
    i = cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + b[i]


a_device, b_device, out_device = init_pb()

NUM_TRHEADS = 2 * SIZE
zip_guard_kernel[1, NUM_TRHEADS](out_device, a_device, b_device, SIZE)

# Copy result back to CPU
result = out_device.copy_to_host()

# Verify result
expected = zip_spec(a, b)
print(f"Input a:  {a}")
print(f"Input b:  {b}")
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[9]:


a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
out = map_spec(a)
out


# In[10]:


@cuda.jit
def map_2d_kernel(out, a, size):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10


a_device, b_device, out_device = init_pb(a=a, out=np.zeros_like(out))

NUM_TRHEADS = (SIZE, SIZE)
map_2d_kernel[1, NUM_TRHEADS](out_device, a_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = map_spec(a)
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[11]:


a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)
out = a + b
out


# In[12]:


@cuda.jit
def broadcast_kernel(out, a, b, size):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]


a_device, b_device, out_device = init_pb(a=a, b=b, out=np.zeros_like(out))

THREADS = (2 * SIZE, SIZE)
broadcast_kernel[1, THREADS](out_device, a_device, b_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = a + b
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[13]:


@cuda.jit
def broadcast_grid_kernel(out, a, b, size):
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]


a_device, b_device, out_device = init_pb(a=a, b=b, out=np.zeros_like(out))

# 1D threads, 2D grid
THREADS = 1
GRID = (SIZE, SIZE)
broadcast_grid_kernel[GRID, THREADS](out_device, a_device, b_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = a + b
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[14]:


@cuda.jit
def broadcast_grid_kernel(out, a, b, size):
    i = cuda.threadIdx.x
    j = cuda.blockIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]


a_device, b_device, out_device = init_pb(a=a, b=b, out=np.zeros_like(out))

THREADS = SIZE
GRID = (1, SIZE)
broadcast_grid_kernel[GRID, THREADS](out_device, a_device, b_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = a + b
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# In[15]:


SIZE // 2


# In[16]:


@cuda.jit
def broadcast_grid_kernel(out, a, b, size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]


a_device, b_device, out_device = init_pb(a=a, b=b, out=np.zeros_like(out))

THREADS = (SIZE // 2, SIZE // 2)
GRID = (SIZE // 2, SIZE // 2)
broadcast_grid_kernel[GRID, THREADS](out_device, a_device, b_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = a + b
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# `blockDim.x = blockDim.y =2 `
#
# | blockIdx.x | blockIdx.y  | threadIdx.x | threadIdx.y | **i** | **j** | Computes |
# |------------|---------------|-------------|-------------|-------|-------|----------|
# | 0 | 0 | 0 | 0 | **0** | **0** | out[0,0] |
# | 0 | 0 | 1 | 0 | **1** | **0** | out[1,0] |
# | 0 | 0 | 0 | 1 | **0** | **1** | out[0,1] |
# | 0 | 0 | 1 | 1 | **1** | **1** | out[1,1] |
# | 1 | 0 | 0 | 0 | **2** | **0** | out[2,0] |
# | 1 | 0 | 1 | 0 | **3** | **0** | out[3,0] |
# | 0 | 1 | 0 | 0 | **0** | **2** | out[0,2] |
# | 0 | 1 | 1 | 1 | **1** | **3** | out[1,3] |
# | 1 | 1 | 0 | 0 | **2** | **2** | out[2,2] |
# | 1 | 1 | 1 | 1 | **3** | **3** | out[3,3] |

# In[17]:


def pool_spec(a):
    out = np.zeros(a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


SIZE = 8
a = np.arange(SIZE)
out = pool_spec(a)
out


# ```
# Memory hierarchy:
# ┌─────────────────────────────────────┐
# │  Global Memory (a, out)             │  ← ALL threads can access
# │  - Slow                             │
# │  - Accessible across all blocks     │
# └─────────────────────────────────────┘
#          ↓                    ↓
#    ┌─────────┐          ┌─────────┐
#    │ Block 0 │          │ Block 1 │
#    │ Shared  │          │ Shared  │      ← Only threads in THIS block
#    │ Memory  │          │ Memory  │         can access
#    │ (fast)  │          │ (fast)  │
#    └─────────┘          └─────────┘
# ```

# In[18]:


@cuda.jit
def pool_kernel(out, a, size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < size:
        # Manually compute sum - can't use slicing in CUDA!
        temp_sum = 0.0
        for k in range(max(i - 2, 0), i + 1):
            temp_sum += a[k]  # Use global memory to handle cross-block boundaries
        out[i] = temp_sum


a_device, b_device, out_device = init_pb(a=a, out=np.zeros_like(out))

THREADS = SIZE // 2
GRID = (2, 1)
pool_kernel[GRID, THREADS](out_device, a_device, SIZE)

result = out_device.copy_to_host()

# Verify result
expected = pool_spec(a)
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")
r = 3
print(
    f"number of access to global memory: 1 + 2 + {THREADS-2} threads x {r} reads = {1+2+(THREADS-2)*r} global reads per block -> {2*(1+2+(THREADS-2)*r)} global reads in total"
)


# The method below allows for 8+4=12 global reads in total.
#
# **How it works:**
# ```
# Global:      [0, 1, 2, 3, 4, 5, 6, 7]
# Block 0 loads:              Block 1 loads:
#         ↓                          ↓
# Shared: [0, 0, 0, 1, 2, 3] Shared: [2, 3, 4, 5, 6, 7]
#          └─┘  └──────────┘         └──┘  └──────────┘
#         halo    main data          halo    main data
# ```

# Each block can only have a constant amount of shared memory that threads in that block can read and write to. This needs to be a literal python constant not a variable. After writing to shared memory you need to call `cuda.syncthreads` to ensure that threads do not cross.

# In[19]:


TPB = 4  # Threads per block
SharedMem = TPB + 2  # cannot computed at runtime


@cuda.jit
def pool_kernel_shared(out, a, size):
    # Allocate shared memory with HALO (extra elements for boundary)
    # Need TPB + 2 extra elements (for the 2-element lookback)
    shared = cuda.shared.array(SharedMem, numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    # Each thread loads its own element into shared memory (offset by 2 for halo)
    if i < size:
        shared[local_i + 2] = a[i]

    # First 2 threads also load the HALO (left boundary elements)
    if local_i == 0:
        # Load 2 elements before the block starts
        start_idx = cuda.blockIdx.x * cuda.blockDim.x
        if start_idx >= 2:
            shared[1] = a[start_idx - 1]
            shared[0] = a[start_idx - 2]
        else:
            shared[1] = 0.0
            shared[0] = 0.0  # Padding for out-of-bounds

    # Wait for all threads to finish loading
    cuda.syncthreads()

    # Now compute using shared memory
    if i < size:
        temp_sum = 0.0
        # Look back up to 2 elements in shared memory
        for k in range(max(0, 3 - (i + 1)), 3):  # At most 3 elements
            temp_sum += shared[local_i + 2 - (2 - k)]
        out[i] = temp_sum


a_device, b_device, out_device = init_pb(a=a, out=np.zeros_like(out))


THREADS = TPB
GRID = (SIZE // TPB, 1)  # (2, 1) for SIZE=8, TPB=4
pool_kernel_shared[GRID, THREADS](out_device, a_device, SIZE)

result = out_device.copy_to_host()

expected = pool_spec(a)
print(f"Output:   {result}")
print(f"Expected: {expected}")
print(f"Correct:  {np.allclose(result, expected)}")


# ![](https://www.cs.uaf.edu/2012/fall/cs441/lecture/tree_sum_16td.png)

# In[20]:


@cuda.jit
def dot_kernel_numba(a, b, out, size):
    # ← At this point, a, b, out are ALREADY on the device (GPU)
    # This kernel executes on the GPU
    shared = cuda.shared.array(256, numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    if i < size:
        shared[local_i] = a[i] * b[i]
    else:
        shared[local_i] = 0.0

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        cuda.syncthreads()
        stride //= 2

    if local_i == 0:
        cuda.atomic.add(
            out, 0, shared[0]
        )  # To avoid RACE CONDITION! Multiple blocks write to out[0] see below


# Test
SIZE = 8
a = np.arange(SIZE, dtype=np.float32)
b = np.arange(SIZE, dtype=np.float32)

expected = np.dot(a, b)

a_device, b_device, out_device = init_pb(a=a, b=b, out=np.zeros_like([expected]))


size = a_device.shape[0]
threads_per_block = 256
blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

# Launch kernel - a_device, b_device, out_device are all on GPU
dot_kernel_numba[blocks_per_grid, threads_per_block](
    a_device, b_device, out_device, size
)

result = out_device.copy_to_host()

print(f"CUDA result: {result[0]}")
print(f"NumPy result: {expected}")
print(f"Match: {np.allclose(result[0], expected)}")


# ## The Problem Without Atomics
# When you write out[0] = out[0] + value, it's actually 3 separate steps:
# ```
# # out[0] = out[0] + value breaks down to:
# 1. READ:   temp = out[0]      # Read current value
# 2. MODIFY: temp = temp + value # Add to it
# 3. WRITE:  out[0] = temp       # Write back
# ```
#
# **With multiple threads, these steps can interleave and lose updates:**
# ```
# Initial: out[0] = 0
#
# Thread A (Block 0):              Thread B (Block 1):
# 1. READ: temp_A = 0
# 2. MODIFY: temp_A = 0 + 5
#                                  1. READ: temp_B = 0      ← Still sees 0!
# 3. WRITE: out[0] = 5
#                                  2. MODIFY: temp_B = 0 + 3 ← Uses old value!
#                                  3. WRITE: out[0] = 3      ← Overwrites 5!
#
# Final: out[0] = 3  ❌ Should be 8!
# ```
#
# ## What Atomics Do
#
# `cuda.atomic.add(out, 0, value)` **locks the memory location** so the entire operation completes before another thread can access it:
# ```
# Initial: out[0] = 0
#
# Thread A (Block 0):              Thread B (Block 1):
# 🔒 LOCK out[0]
# 1. READ: temp_A = 0
# 2. MODIFY: temp_A = 0 + 5
# 3. WRITE: out[0] = 5
# 🔓 UNLOCK out[0]
#                                  🔒 LOCK out[0]  ← Must wait for unlock
#                                  1. READ: temp_B = 5      ← Sees updated value!
#                                  2. MODIFY: temp_B = 5 + 3
#                                  3. WRITE: out[0] = 8
#                                  🔓 UNLOCK out[0]
#
# Final: out[0] = 8  ✅ Correct!
# ```

# ## Numba CUDA: Grid and Block Dimensions
#
# In **Numba CUDA**, you launch kernels with explicit grid and block dimensions:
# ```python
# from numba import cuda
# import numpy as np
#
# @cuda.jit
# def kernel(output):
#     # Thread indices within the block
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     tz = cuda.threadIdx.z
#
#     # Block indices within the grid
#     bx = cuda.blockIdx.x
#     by = cuda.blockIdx.y
#     bz = cuda.blockIdx.z
#
#     # Block dimensions
#     block_dim_x = cuda.blockDim.x
#     block_dim_y = cuda.blockDim.y
#
#     # Global thread index
#     i = bx * block_dim_x + tx
#     j = by * block_dim_y + ty
#
#     output[i, j] = i * 1000 + j
#
# # Launch configuration
# threads_per_block = (16, 16)  # Block dimensions: 16x16 = 256 threads per block
# blocks_per_grid = (4, 8)       # Grid dimensions: 4x8 = 32 blocks total
#
# output = np.zeros((64, 128), dtype=np.int32)
# kernel[blocks_per_grid, threads_per_block](output)
# ```
#
# **Key concepts:**
# - `kernel[grid, block](args)` - Launch syntax
# - **Grid** = `(blocks_x, blocks_y, blocks_z)` - How many blocks
# - **Block** = `(threads_x, threads_y, threads_z)` - How many threads per block
# - Total threads = `grid_x × grid_y × grid_z × block_x × block_y × block_z`
#
# ---
#
# ## Triton: Program Grid
#
# In **Triton**, you specify a **program grid** and work with **program IDs**:
# ```python
# import triton
# import triton.language as tl
#
# @triton.jit
# def kernel(output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
#     # Program ID (like block index in CUDA)
#     pid_x = tl.program_id(0)
#     pid_y = tl.program_id(1)
#
#     # Compute offsets for this program
#     row_start = pid_x * BLOCK_SIZE
#     col_start = pid_y * BLOCK_SIZE
#
#     # Create a block of indices
#     rows = row_start + tl.arange(0, BLOCK_SIZE)
#     cols = col_start + tl.arange(0, BLOCK_SIZE)
#
#     # Triton handles the actual thread mapping automatically
#     output = rows[:, None] * 1000 + cols[None, :]
#
#     # Store results
#     mask = (rows[:, None] < M) & (cols[None, :] < N)
#     tl.store(output_ptr + rows[:, None] * N + cols[None, :], output, mask=mask)
#
# # Launch configuration
# M, N = 64, 128
# BLOCK_SIZE = 16
#
# output = torch.zeros((M, N), dtype=torch.int32, device='cuda')
#
# # Grid: number of programs in each dimension
# grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
# kernel[grid](output, M, N, BLOCK_SIZE=BLOCK_SIZE)
# ```
#
# **Key concepts:**
# - `kernel[grid](args, BLOCK_SIZE=...)` - Launch syntax
# - **Grid** = `(programs_x, programs_y, programs_z)` - Number of program instances
# - **No explicit block/thread dimensions** - Triton handles threading automatically
# - Work with **blocks of data** using `tl.arange()` and vectorized operations
#
# ---
#
# ## Comparison Table
#
# | Aspect | Numba CUDA | Triton |
# |--------|------------|--------|
# | **Launch syntax** | `kernel[grid, block](args)` | `kernel[grid](args, BLOCK=...)` |
# | **Grid represents** | Number of **blocks** | Number of **programs** |
# | **Block/Thread control** | Explicit: `(tx, ty, tz)` per block | Abstracted: work on data blocks |
# | **Thread indexing** | Manual: `blockIdx`, `threadIdx` | Automatic: `tl.program_id()` + `tl.arange()` |
# | **Typical grid** | `(n_blocks_x, n_blocks_y, n_blocks_z)` | `(n_programs_x, n_programs_y, n_programs_z)` |
# | **Typical block** | `(threads_x, threads_y, threads_z)` | N/A (implicit in `BLOCK_SIZE`) |
# | **Memory access** | Per-thread indexing | Vectorized block operations |
# | **Abstraction level** | Low-level (like CUDA C) | High-level (compiler optimizes) |
#
# ---
#
# ## Practical Example: Vector Addition
#
# ### Numba CUDA Version:
# ```python
# @cuda.jit
# def vector_add_numba(a, b, c, n):
#     i = cuda.grid(1)  # Global thread index
#     if i < n:
#         c[i] = a[i] + b[i]
#
# # Launch
# n = 1_000_000
# threads_per_block = 256
# blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
# vector_add_numba[blocks_per_grid, threads_per_block](a, b, c, n)
# ```
#
# ### Triton Version:
# ```python
# @triton.jit
# def vector_add_triton(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offset < n
#
#     a = tl.load(a_ptr + offset, mask=mask)
#     b = tl.load(b_ptr + offset, mask=mask)
#     c = a + b
#     tl.store(c_ptr + offset, c, mask=mask)
#
# # Launch
# BLOCK_SIZE = 1024
# grid = (triton.cdiv(n, BLOCK_SIZE),)
# vector_add_triton[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
# ```
#
# ---
#
# ## Key Takeaway
#
# - **Numba CUDA**: You think in terms of **blocks of threads** (2-level hierarchy: grid → blocks → threads)
# - **Triton**: You think in terms of **programs operating on data blocks** (1-level: grid → programs, with implicit vectorization)
#
# Triton is conceptually **one level of "blocks"** in CUDA terms - each Triton program is roughly equivalent to a CUDA block, but Triton automatically handles the thread-level parallelism within that program.

# In[ ]:

print("GPU puzzles completed successfully!")
