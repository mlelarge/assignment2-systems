import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import cs336_basics.model as models
import cs336_basics.nn_utils as nn_utils

# import cs336_basics.optimizer as optimizer

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


def compute_forward_and_loss():
    with torch.no_grad():
        logits = model(random_input)
        loss = nn_utils.cross_entropy(
            logits.view(-1, args.vocab_size), random_target.view(-1)
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    return loss


def compute_forward_and_backward():
    model.train()
    model.zero_grad()
    logits = model(random_input)
    loss = nn_utils.cross_entropy(
        logits.view(-1, args.vocab_size), random_target.view(-1)
    )
    loss.backward()
    torch.cuda.synchronize() if device.type == "cuda" else None
    return loss


def run_benchmark():
    # Warm-up
    for _ in range(args.num_warmup):
        compute_forward_and_loss()

    # Benchmark
    avg_time_benchmarks = []
    for i in range(args.num_steps):
        print(f"Iteration {i}/{args.num_steps}")
        start_time = timeit.default_timer()
        compute_forward_and_loss()
        end_time = timeit.default_timer()
        avg_time_benchmarks.append(end_time - start_time)
    print(
        f"Average time per step (forward only): {sum(avg_time_benchmarks) / args.num_steps:.6f} seconds"
    )
    print(f"Standard deviation: {np.std(np.array(avg_time_benchmarks)):.6f} seconds")

    # Warm-up for forward and backward
    for _ in range(args.num_warmup):
        compute_forward_and_backward()

    # Benchmark for forward and backward
    avg_time_benchmarks = []
    for i in range(args.num_steps):
        print(f"Iteration {i}/{args.num_steps}")
        start_time = timeit.default_timer()
        compute_forward_and_backward()
        end_time = timeit.default_timer()
        avg_time_benchmarks.append(end_time - start_time)
    print(
        f"Average time per step (forward + backward): {sum(avg_time_benchmarks) / args.num_steps:.6f} seconds"
    )
    print(f"Standard deviation: {np.std(np.array(avg_time_benchmarks)):.6f} seconds")


if __name__ == "__main__":
    run_benchmark()
