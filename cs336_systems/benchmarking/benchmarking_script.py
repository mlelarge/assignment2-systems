import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
from contextlib import nullcontext

import cs336_basics.model as models
import cs336_basics.nn_utils as nn_utils
import cs336_basics.optimizer as optimizer

import torch
import timeit
import argparse
import numpy as np

# Parse arguments (or define them)
parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=1024)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--d_ff", type=int, default=3072)
parser.add_argument("--rope_theta", type=float, default=10000.0)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--num_warmup", type=int, default=5)
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--mixed_precision", action="store_true", default=False)
parser.add_argument("--memory_profile", action="store_true", default=False)
args = parser.parse_args()

model = models.BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)

# Move model to device
device = torch.device(args.device)
model = model.to(device)
optim = optimizer.AdamW(model.parameters())

random_input = torch.randint(
    low=0,
    high=args.vocab_size,
    size=(args.batch_size, args.context_length),
    device=device,
)
random_target = torch.randint(
    low=0,
    high=args.vocab_size,
    size=(args.batch_size, args.context_length),
    device=device,
)


def return_dtype():
    if args.mixed_precision:
        return torch.bfloat16
    else:
        return torch.float32


def compute_forward_and_loss(context):
    model.eval()
    with torch.no_grad():
        with context:
            logits = model(random_input)
            loss = nn_utils.cross_entropy(
                logits.view(-1, args.vocab_size), random_target.view(-1)
            )
        torch.cuda.synchronize() if device.type == "cuda" else None
    return loss


def compute_forward_and_backward(context):
    with context:
        model.train()
        model.zero_grad()
        logits = model(random_input)
        loss = nn_utils.cross_entropy(
            logits.view(-1, args.vocab_size), random_target.view(-1)
        )
        loss.backward()
        optim.step()
        # Add this line to clear grad tensors
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
    return loss


def run_benchmark():
    dtype = return_dtype()
    print(f"dtype: {dtype}")
    context = (
        torch.amp.autocast(device_type=device.type, dtype=return_dtype())
        if args.mixed_precision
        else nullcontext()
    )
    # Warm-up
    for _ in range(args.num_warmup):
        compute_forward_and_loss(context)

    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    # Benchmark for forward only
    avg_time_benchmarks = []
    for i in range(args.num_steps):
        # print(f"Iteration {i}/{args.num_steps}")
        start_time = timeit.default_timer()
        compute_forward_and_loss(context)
        end_time = timeit.default_timer()
        avg_time_benchmarks.append(end_time - start_time)
    print(
        f"Average time per step (forward only): {sum(avg_time_benchmarks) / args.num_steps:.6f} seconds"
    )
    print(f"Standard deviation: {np.std(np.array(avg_time_benchmarks)):.6f} seconds")
    # Save snapshot
    if args.memory_profile:
        torch.cuda.memory._dump_snapshot("memory_forward_only.pickle")

    # Stop recording (important to avoid overhead)
    if args.memory_profile:
        torch.cuda.memory._record_memory_history(enabled=False)

    # Warm-up for forward and backward
    for _ in range(args.num_warmup):
        compute_forward_and_backward(context)

    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    # Benchmark for forward and backward
    avg_time_benchmarks = []
    for i in range(args.num_steps):
        # print(f"Iteration {i}/{args.num_steps}")
        start_time = timeit.default_timer()
        compute_forward_and_backward(context)
        end_time = timeit.default_timer()
        avg_time_benchmarks.append(end_time - start_time)
    print(
        f"Average time per step (forward + backward): {sum(avg_time_benchmarks) / args.num_steps:.6f} seconds"
    )
    print(f"Standard deviation: {np.std(np.array(avg_time_benchmarks)):.6f} seconds")
    if args.memory_profile:
        torch.cuda.memory._dump_snapshot("memory_backward.pickle")
        torch.cuda.memory._record_memory_history(enabled=False)


if __name__ == "__main__":
    run_benchmark()
