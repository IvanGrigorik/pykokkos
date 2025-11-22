#!/usr/bin/env python3
"""
Run all fusion examples and demonstrate the differences between strategies.

This script runs the same computations with different fusion strategies
and compares the results.
"""

import os
import sys
import subprocess
import time


def run_with_strategy(script_name, strategy):
    """Run a script with a specific fusion strategy."""
    env = os.environ.copy()
    env["PK_FUSION"] = strategy
    
    print(f"\n{'='*70}")
    print(f"Running {script_name} with PK_FUSION={strategy}")
    print('='*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\n  Execution time: {elapsed:.3f} seconds")
        
        return result.returncode == 0, elapsed
        
    except subprocess.TimeoutExpired:
        print("[WARNING] TIMEOUT: Script took too long to execute")
        return False, 0
    except Exception as e:
        print(f"[WARNING] ERROR: {e}")
        return False, 0


def main():
    print("\n" + "="*70)
    print("FUSION STRATEGY DEMONSTRATION")
    print("="*70)
    print("\nThis script runs all examples with different fusion strategies")
    print("to demonstrate the differences between them.")
    
    examples = [
        ("example_horizontal.py", "Horizontal Fusion Examples"),
        ("example_vertical.py", "Vertical Fusion Examples"),
        ("example_comparison.py", "Strategy Comparison"),
        ("example_no_fusion_case.py", "Fusion Limitations"),
    ]
    
    strategies = ["trace", "naive", "horizontal"]
    
    results = {}
    
    for script, description in examples:
        print(f"\n\n{'='*70}")
        print(f"TESTING: {description}")
        print('='*70)
        
        script_results = {}
        
        for strategy in strategies:
            success, elapsed = run_with_strategy(script, strategy)
            script_results[strategy] = (success, elapsed)
        
        results[script] = script_results
    
    # Print summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for script, description in examples:
        print(f"\n{description} ({script}):")
        script_results = results[script]
        
        for strategy in strategies:
            success, elapsed = script_results[strategy]
            status = "[OK]" if success else "[X]"
            print(f"  {status} {strategy:12s}: {elapsed:6.3f}s")
    
    print("\n" + "="*70)
    print("NOTES")
    print("="*70)
    print("""
The execution times show the overhead of different strategies:
  * 'trace' = No fusion (baseline)
  * 'naive' = Vertical fusion (consecutive operations)
  * 'horizontal' = Horizontal fusion (independent operations)

Performance gains depend on:
  1. Number of kernels that can be fused
  2. Kernel launch overhead vs computation time
  3. Whether operations are independent or dependent

For compute-bound kernels, fusion may show minimal improvement.
For launch-overhead-bound scenarios, fusion can provide significant speedup.
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


