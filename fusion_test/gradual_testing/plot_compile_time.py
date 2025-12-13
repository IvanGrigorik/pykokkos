#!/usr/bin/env python3
"""
Plot compilation time comparison between naive and barrier fusion.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file):
    """Load compilation time results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)

def plot_compile_time_comparison(results_file="compile_time_results.json", output_file="compile_time_comparison.png"):
    """Create comparison plot of compilation times."""
    
    results = load_results(results_file)
    
    test_names = []
    test_numbers = []
    naive_times = []
    barrier_times = []
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"Test {test_num}")
        test_numbers.append(test_num)
        
        naive_data = results[test_name].get('naive', {})
        barrier_data = results[test_name].get('barrier', {})
        
        if naive_data.get('success'):
            naive_times.append(naive_data['compile_time'])
        else:
            naive_times.append(None)
            
        if barrier_data.get('success'):
            barrier_times.append(barrier_data['compile_time'])
        else:
            barrier_times.append(None)
    
    if not test_names:
        print("No test data found!")
        return None
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(test_names))
    
    # Compilation time comparison (line chart)
    naive_t = [t if t is not None else None for t in naive_times]
    barrier_t = [t if t is not None else None for t in barrier_times]
    
    # Filter out None values for plotting
    naive_valid = [(i, t) for i, t in enumerate(naive_t) if t is not None]
    barrier_valid = [(i, t) for i, t in enumerate(barrier_t) if t is not None]
    
    if naive_valid:
        naive_x = [i for i, _ in naive_valid]
        naive_y = [t for _, t in naive_valid]
        ax.plot(naive_x, naive_y, marker='o', markersize=8, linewidth=2.5, 
                label='Naive (Vertical)', color='#3498db', alpha=0.8)
    
    if barrier_valid:
        barrier_x = [i for i, _ in barrier_valid]
        barrier_y = [t for _, t in barrier_valid]
        ax.plot(barrier_x, barrier_y, marker='s', markersize=8, linewidth=2.5, 
                label='Barrier (True Horizontal)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Test Case', fontsize=11, fontweight='bold')
    ax.set_ylabel('Compilation Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Compilation Time Comparison: Naive vs Barrier', fontsize=12, fontweight='bold')
    if len(x) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    valid_tests = [i for i, (n, b) in enumerate(zip(naive_times, barrier_times)) 
                   if n is not None and b is not None]
    
    if valid_tests:
        naive_avg = np.mean([naive_times[i] for i in valid_tests])
        barrier_avg = np.mean([barrier_times[i] for i in valid_tests])
        
        # Calculate speedups
        speedups_list = []
        for i in valid_tests:
            n = naive_times[i]
            b = barrier_times[i]
            if n and b and b > 0:
                speedups_list.append(n / b)
        
        speedup_avg = np.mean(speedups_list) if speedups_list else 0
        
        print(f"Average Compilation Time:")
        print(f"  Naive:   {naive_avg:.2f}s")
        print(f"  Barrier: {barrier_avg:.2f}s")
        if speedup_avg > 0:
            print(f"  Average Speedup: {speedup_avg:.2f}x {'(barrier faster)' if speedup_avg > 1.0 else '(naive faster)'}")
        print()
        print(f"Total Compilation Time:")
        print(f"  Naive:   {sum([naive_times[i] for i in valid_tests]):.2f}s")
        print(f"  Barrier: {sum([barrier_times[i] for i in valid_tests]):.2f}s")
    
    return fig

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'compile_time_results.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'compile_time_comparison.png'
    
    plot_compile_time_comparison(results_file, output_file)

