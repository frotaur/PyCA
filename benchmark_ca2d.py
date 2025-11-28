"""
Benchmark script for CA2D automaton.
Tests FPS performance of:
1. step + draw
2. step + draw + access worldmap
"""

import time
import torch
from pyca.automata.models.ca2d import CA2D

def benchmark_operation(ca, operation_name, iterations=100, warmup=10):
    """
    Benchmark a specific operation.

    Args:
        ca: CA2D instance
        operation_name: Name of the operation being benchmarked
        iterations: Number of iterations to run
        warmup: Number of warmup iterations before timing

    Returns:
        Average FPS
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {operation_name}")
    print(f"{'='*60}")

    is_cuda = ca.device == "cuda" or (hasattr(ca.device, 'type') and ca.device.type == "cuda")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        if "step" in operation_name:
            ca.step()
        if "draw" in operation_name:
            ca.draw()
        if "worldmap" in operation_name:
            wm = ca.worldmap
            _ = wm[0, 0, 0]  # Force actual data access to ensure transfer completes
        # Synchronize GPU after each warmup iteration
        if is_cuda:
            torch.cuda.synchronize()

    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")

    # Synchronize before starting timer
    if is_cuda:
        torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(iterations):
        if "step" in operation_name:
            ca.step()
        if "draw" in operation_name:
            ca.draw()
        if "worldmap" in operation_name:
            wm = ca.worldmap
            _ = wm[0, 0, 0]  # Force actual data access to ensure transfer completes
        # Synchronize GPU after each iteration to ensure accurate timing
        if is_cuda:
            torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    avg_fps = iterations / elapsed_time

    print(f"Elapsed time: {elapsed_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Time per iteration: {(elapsed_time/iterations)*1000:.3f}ms")

    return avg_fps

def main():
    # Configuration
    size = (1000, 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 100
    warmup = 10

    print("="*60)
    print("CA2D Performance Benchmark")
    print("="*60)
    print(f"Size: {size[0]}x{size[1]}")
    print(f"Device: {device}")
    print(f"Iterations: {iterations}")
    print(f"Warmup iterations: {warmup}")

    # Initialize CA2D
    print("\nInitializing CA2D...")
    ca = CA2D(size=size, s_num="23", b_num="3", dot=False, device=device)
    print(f"CA2D initialized successfully")
    print(f"Rule: s:{ca.get_rule_from_num(ca.s_num)}, b:{ca.get_rule_from_num(ca.b_num)}")

    # Benchmark 1: step + draw
    fps_step_draw = benchmark_operation(
        ca,
        "step + draw",
        iterations=iterations,
        warmup=warmup
    )

    # Reset for next benchmark
    ca.reset()

    # Benchmark 2: step + draw + access worldmap
    fps_step_draw_worldmap = benchmark_operation(
        ca,
        "step + draw + worldmap access",
        iterations=iterations,
        warmup=warmup
    )

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Operation':<35} {'FPS':>10}")
    print("-"*60)
    print(f"{'step + draw':<35} {fps_step_draw:>10.2f}")
    print(f"{'step + draw + worldmap access':<35} {fps_step_draw_worldmap:>10.2f}")
    print("="*60)

    # Additional info
    print(f"\nDevice used: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()