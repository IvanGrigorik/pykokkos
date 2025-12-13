#!/usr/bin/env python3
"""
Benchmark script for gradual testing dataset.
Tests all fusion strategies: trace (no fusion), naive (vertical), horizontal, barrier
"""

import os
import sys
import subprocess
import time
import json
import statistics
from pathlib import Path

# Add pykokkos to path
sys.path.insert(0, "/home/sifi/pykokkos-fuse")


def run_test(test_file, strategy, iterations=100, skip_first=True):
    """
    Run a test with a specific fusion strategy and measure execution time.
    First run compiles, subsequent runs measure actual runtime.

    Args:
        test_file: Path to test file
        strategy: Fusion strategy ('trace', 'naive', 'horizontal', 'barrier')
        iterations: Number of iterations to average (after first compile run)
        skip_first: If True, skip the first run (compilation) and only measure runtime

    Returns:
        tuple: (success, avg_time, kernel_count)
    """
    env = os.environ.copy()
    env["PK_FUSION"] = strategy
    env["PYTHONPATH"] = f"/home/sifi/pykokkos-fuse:{env.get('PYTHONPATH', '')}"
    env["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
    env["PK_EXEC_SPACE"] = "CUDA"
    # Ensure nvcc is in PATH
    cuda_paths = [
        "/usr/local/cuda-12.9/bin",
        "/usr/local/cuda-12.8/bin",
        "/usr/local/cuda/bin",
    ]
    current_path = env.get("PATH", "")
    for cuda_path in cuda_paths:
        if os.path.exists(f"{cuda_path}/nvcc") and cuda_path not in current_path:
            env["PATH"] = f"{cuda_path}:{current_path}"
            break

    # Get test directory and pk_cpp path
    test_dir = Path(test_file).parent.resolve()  # Use resolve() to get absolute path
    pk_cpp_dir = test_dir / "pk_cpp"

    # First run: compilation (skip timing)
    if skip_first:
        print(f"    Compiling... ", end="", flush=True)
        try:
            # Set cwd explicitly to test directory to avoid directory issues
            result = subprocess.run(
                [sys.executable, test_file],
                env=env,
                cwd=str(test_dir),  # Set working directory explicitly
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for compilation
            )
            if result.returncode != 0:
                print(f"FAILED")
                print(f"    Error: {result.stderr[:500]}")
                if result.stdout:
                    print(f"    Stdout: {result.stdout[:300]}")
                return False, None, None
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            return False, None, None

    # Subsequent runs: measure runtime
    times = []
    success = False

    for i in range(iterations):
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                env=env,
                cwd=str(test_dir),  # Set working directory explicitly
                capture_output=True,
                text=True,
                timeout=60,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                times.append(elapsed)
                success = True
                if (i + 1) % 10 == 0:
                    print(
                        f"    Progress: {i+1}/{iterations} iterations completed",
                        flush=True,
                    )
            else:
                print(f"  Strategy {strategy} iteration {i+1} failed:")
                print(f"    {result.stderr[:200]}")
                return False, None, None
        except subprocess.TimeoutExpired:
            print(f"  Strategy {strategy} iteration {i+1} timed out")
            return False, None, None
        except Exception as e:
            print(f"  Strategy {strategy} iteration {i+1} error: {e}")
            return False, None, None

    if not times:
        return False, None, None

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

    # Count kernels generated before cleanup
    kernel_count = 0
    if pk_cpp_dir.exists():
        kernel_count = len(list(pk_cpp_dir.rglob("bindings.cpp")))
    
    # Remove pk_cpp after running the test to clean up for next test
    if pk_cpp_dir.exists():
        import shutil
        try:
            shutil.rmtree(pk_cpp_dir, ignore_errors=True)
            time.sleep(0.1)  # Small delay to ensure deletion completes
        except Exception:
            pass  # Continue even if cleanup fails

    return success, (avg_time, min_time, max_time, std_dev), kernel_count


def benchmark_all(strategies=None):
    """Run benchmarks for all test files with specified strategies."""

    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))

    if strategies is None:
        # Default: compare horizontal vs barrier (true horizontal fusion with barriers)
        strategies = ["horizontal", "barrier"]

    results = {}

    print("=" * 70)
    print("GRADUAL TESTING BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {', '.join(strategies)}")
    print("Note: First run compiles, subsequent runs measure runtime (100 iterations)")
    print()

    for test_file in test_files:
        test_name = test_file.stem
        print(f"Testing: {test_name}")
        print("-" * 70)

        results[test_name] = {}

        # Remove pk_cpp before starting this test to ensure fresh compilation
        import shutil
        test_pk_cpp = test_dir / "pk_cpp"
        if test_pk_cpp.exists():
            try:
                shutil.rmtree(test_pk_cpp, ignore_errors=True)
                time.sleep(0.2)  # Wait for deletion to complete
            except Exception:
                pass
        
        for strategy in strategies:
            print(f"  Strategy: {strategy:12s}")
            success, avg_time, kernel_count = run_test(
                test_file, strategy, iterations=100, skip_first=True
            )

            if success:
                avg_time, min_time, max_time, std_dev = avg_time
                print(
                    f"    Runtime: {avg_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s, std: {std_dev:.4f}s), Kernels: {kernel_count}"
                )
                results[test_name][strategy] = {
                    "success": True,
                    "time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "std_dev": std_dev,
                    "kernels": kernel_count,
                }
            else:
                print(f"    FAILED")
                results[test_name][strategy] = {
                    "success": False,
                    "time": None,
                    "kernels": None,
                }

        print()

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Test':<30} {'Trace':<12} {'Naive':<12} {'Horizontal':<12} {'Barrier':<12}"
    )
    print("-" * 70)

    for test_name, test_results in results.items():
        row = f"{test_name:<30}"
        for strategy in strategies:
            if strategy in test_results and test_results[strategy]["success"]:
                time_str = f"{test_results[strategy]['time']:.4f}s"
                kernels = test_results[strategy]["kernels"]
                row += f" {time_str:>6} ({kernels:>2})"
            else:
                row += " " + "FAILED".center(11)
        print(row)

    # Save results to JSON
    json_file = test_dir / "benchmark_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {json_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        strategies = sys.argv[1:]
        benchmark_all(strategies)
    else:
        benchmark_all()
