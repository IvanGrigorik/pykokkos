#!/usr/bin/env python3
"""
Benchmark script to measure compilation time for naive vs barrier fusion.
Each test is compiled once with each strategy to measure compilation time.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Add pykokkos to path
sys.path.insert(0, '/home/sifi/pykokkos-fuse')

def compile_test(test_file, strategy):
    """
    Compile a test with a specific fusion strategy and measure compilation time.
    
    Args:
        test_file: Path to test file
        strategy: Fusion strategy ('naive' or 'barrier')
    
    Returns:
        tuple: (success, compile_time)
    """
    env = os.environ.copy()
    env["PK_FUSION"] = strategy
    env["PYTHONPATH"] = f"/home/sifi/pykokkos-fuse:{env.get('PYTHONPATH', '')}"
    env["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
    env["PK_EXEC_SPACE"] = "CUDA"
    
    # Ensure nvcc is in PATH
    cuda_paths = ["/usr/local/cuda-12.9/bin", "/usr/local/cuda-12.8/bin", "/usr/local/cuda/bin"]
    current_path = env.get("PATH", "")
    for cuda_path in cuda_paths:
        if os.path.exists(f"{cuda_path}/nvcc") and cuda_path not in current_path:
            env["PATH"] = f"{cuda_path}:{current_path}"
            break
    
    # Clean pk_cpp before compilation
    test_dir = Path(test_file).parent
    pk_cpp_dir = test_dir / "pk_cpp"
    if pk_cpp_dir.exists():
        import shutil
        shutil.rmtree(pk_cpp_dir)
    
    # Measure compilation time
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout for compilation
        )
        compile_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, compile_time
        else:
            print(f"    Compilation failed: {result.stderr[:200]}")
            return False, None
    except subprocess.TimeoutExpired:
        compile_time = time.time() - start_time
        print(f"    Compilation timed out after {compile_time:.2f}s")
        return False, None
    except Exception as e:
        compile_time = time.time() - start_time
        print(f"    Compilation error: {e}")
        return False, None

def benchmark_compile_times():
    """Run compilation time benchmarks for all test files."""
    
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))
    
    strategies = ['naive', 'barrier']
    results = {}
    
    print("="*70)
    print("COMPILATION TIME BENCHMARK")
    print("="*70)
    print("Comparing: Naive (Vertical) vs Barrier (True Horizontal)")
    print("Note: Each test is compiled once per strategy")
    print()
    
    for test_file in test_files:
        test_name = test_file.stem
        print(f"Testing: {test_name}")
        print("-" * 70)
        
        results[test_name] = {}
        
        for strategy in strategies:
            print(f"  Strategy: {strategy:12s} ", end="", flush=True)
            success, compile_time = compile_test(test_file, strategy)
            
            if success:
                print(f"Compile time: {compile_time:.2f}s")
                results[test_name][strategy] = {
                    'success': True,
                    'compile_time': compile_time
                }
            else:
                print(f"FAILED")
                results[test_name][strategy] = {
                    'success': False,
                    'compile_time': None
                }
        
        print()
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Test':<30} {'Naive (s)':<15} {'Barrier (s)':<15} {'Speedup':<15}")
    print("-" * 70)
    
    for test_name, test_results in results.items():
        row = f"{test_name:<30}"
        naive_time = test_results.get('naive', {}).get('compile_time')
        barrier_time = test_results.get('barrier', {}).get('compile_time')
        
        if naive_time:
            row += f" {naive_time:>12.2f}  "
        else:
            row += " " + "FAILED".center(13)
        
        if barrier_time:
            row += f" {barrier_time:>12.2f}  "
        else:
            row += " " + "FAILED".center(13)
        
        if naive_time and barrier_time:
            speedup = naive_time / barrier_time
            row += f" {speedup:>6.2f}x"
            if speedup > 1.0:
                row += " (barrier faster)"
            else:
                row += " (naive faster)"
        else:
            row += " " + "N/A".center(13)
        
        print(row)
    
    # Calculate averages
    print()
    print("-" * 70)
    naive_times = [r.get('naive', {}).get('compile_time') for r in results.values() 
                   if r.get('naive', {}).get('compile_time')]
    barrier_times = [r.get('barrier', {}).get('compile_time') for r in results.values() 
                     if r.get('barrier', {}).get('compile_time')]
    
    if naive_times and barrier_times:
        avg_naive = sum(naive_times) / len(naive_times)
        avg_barrier = sum(barrier_times) / len(barrier_times)
        avg_speedup = avg_naive / avg_barrier
        
        print(f"Average Compilation Time:")
        print(f"  Naive:   {avg_naive:.2f}s")
        print(f"  Barrier: {avg_barrier:.2f}s")
        print(f"  Average Speedup: {avg_speedup:.2f}x {'(barrier faster)' if avg_speedup > 1.0 else '(naive faster)'}")
    
    # Save results to JSON
    json_file = test_dir / "compile_time_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {json_file}")
    
    return results

if __name__ == "__main__":
    benchmark_compile_times()



