#!/usr/bin/env python3
"""
Standalone benchmarking script for weighted sum implementations.
This script only runs benchmarks and profiling without the training example.
"""

import time
import torch
import triton
from weighted_sum import (
    weighted_sum,
    weighted_sum_triton,
    WeightedSumFunc_wBackward,
    WeightedSumFunc_wtritonBackward,
    test_triton_backward,
    benchmark_all_implementations,
    profile_implementations,
)


def main():
    print("=" * 80)
    print("Weighted Sum Implementation Benchmarks")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Benchmarks require a GPU.")
        return

    # Test correctness first
    test_triton_backward()

    # Run comprehensive benchmarks
    benchmark_all_implementations()

    # Run profiling
    profile_implementations()

    print("\n" + "=" * 80)
    print("Benchmarking completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
