#!/usr/bin/env python3
"""
Multiple plot variants for fusion benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)

def variant1_grouped_bars(results_file="benchmark_results.json", output_file="variant1_grouped_bars.png"):
    """Variant 1: Grouped bar chart for runtime and kernel count."""
    results = load_results(results_file)
    
    test_names = []
    test_numbers = []
    strategies = ["trace", "naive", "barrier"]
    strategy_labels = ["Trace", "Naive", "Barrier"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    
    runtime_data = {s: [] for s in strategies}
    kernel_data = {s: [] for s in strategies}
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"T{test_num}")
        test_numbers.append(test_num)
        
        for strategy in strategies:
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                runtime_data[strategy].append(data["time"])
                kernel_data[strategy].append(data["kernels"])
            else:
                runtime_data[strategy].append(0)
                kernel_data[strategy].append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(test_names))
    width = 0.25
    
    # Runtime plot
    for i, strategy in enumerate(strategies):
        offset = (i - 1) * width
        ax1.bar(x + offset, runtime_data[strategy], width, label=strategy_labels[i], 
                color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Test Case", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Runtime (seconds)", fontsize=11, fontweight='bold')
    ax1.set_title("Runtime Comparison", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Kernel count plot
    for i, strategy in enumerate(strategies):
        offset = (i - 1) * width
        ax2.bar(x + offset, kernel_data[strategy], width, label=strategy_labels[i], 
                color=colors[i], alpha=0.8)
    
    ax2.set_xlabel("Test Case", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Number of Kernels", fontsize=11, fontweight='bold')
    ax2.set_title("Kernel Count Comparison", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=0)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 1 saved: {output_file}")

def variant2_heatmap(results_file="benchmark_results.json", output_file="variant2_heatmap.png"):
    """Variant 2: Heatmap showing kernel counts."""
    results = load_results(results_file)
    
    test_names = []
    strategies = ["trace", "naive", "barrier"]
    strategy_labels = ["Trace", "Naive", "Barrier"]
    
    kernel_matrix = []
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"Test {test_num}")
        row = []
        for strategy in strategies:
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                row.append(data["kernels"])
            else:
                row.append(0)
        kernel_matrix.append(row)
    
    kernel_matrix = np.array(kernel_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(kernel_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(strategy_labels)))
    ax.set_yticks(np.arange(len(test_names)))
    ax.set_xticklabels(strategy_labels)
    ax.set_yticklabels(test_names)
    
    # Add text annotations
    for i in range(len(test_names)):
        for j in range(len(strategies)):
            text = ax.text(j, i, kernel_matrix[i, j], ha="center", va="center", 
                          color="black" if kernel_matrix[i, j] < kernel_matrix.max()/2 else "white",
                          fontweight='bold')
    
    ax.set_title("Kernel Count Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Number of Kernels')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 2 saved: {output_file}")

def variant3_speedup_normalized(results_file="benchmark_results.json", output_file="variant3_speedup.png"):
    """Variant 3: Speedup normalized to trace (no fusion)."""
    results = load_results(results_file)
    
    test_names = []
    test_numbers = []
    strategies = ["naive", "barrier"]
    strategy_labels = ["Naive", "Barrier"]
    colors = ["#3498db", "#e74c3c"]
    
    speedup_data = {s: [] for s in strategies}
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"T{test_num}")
        test_numbers.append(test_num)
        
        trace_time = results[test_name].get("trace", {}).get("time")
        if not trace_time:
            for strategy in strategies:
                speedup_data[strategy].append(0)
            continue
        
        for strategy in strategies:
            data = results[test_name].get(strategy, {})
            if data.get("success") and data.get("time"):
                speedup = trace_time / data["time"]
                speedup_data[strategy].append(speedup)
            else:
                speedup_data[strategy].append(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(test_names))
    width = 0.255
    
    for i, strategy in enumerate(strategies):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, speedup_data[strategy], width, 
                     label=strategy_labels[i], color=colors[i], alpha=0.8)
        # Add value labels
        for j, val in enumerate(speedup_data[strategy]):
            if val > 0:
                ax.text(j + offset, val + 0.02, f'{val:.2f}x', 
                       ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (Trace)')
    ax.set_xlabel("Test Case", fontsize=11, fontweight='bold')
    ax.set_ylabel("Speedup (vs Trace)", fontsize=11, fontweight='bold')
    ax.set_title("Speedup Comparison (Normalized to Trace)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 3 saved: {output_file}")

def variant4_line_with_errorbars(results_file="benchmark_results.json", output_file="variant4_line_errorbars.png"):
    """Variant 4: Line plot with error bars showing min/max."""
    results = load_results(results_file)
    
    test_numbers = []
    strategies = ["trace", "naive", "barrier"]
    strategy_labels = ["Trace", "Naive", "Barrier"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    markers = ["o", "s", "^", "D"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for strategy_idx, strategy in enumerate(strategies):
        times = []
        kernels = []
        mins = []
        maxs = []
        test_nums = []
        
        for test_name in sorted(results.keys()):
            test_num = int(test_name.split("_")[1])
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                test_nums.append(test_num)
                times.append(data["time"])
                kernels.append(data["kernels"])
                mins.append(data.get("min_time", data["time"]))
                maxs.append(data.get("max_time", data["time"]))
        
        if test_nums:
            # Runtime plot with error bars
            errors = [[t - m for t, m in zip(times, mins)], 
                     [M - t for t, M in zip(times, maxs)]]
            ax1.errorbar(test_nums, times, yerr=errors, marker=markers[strategy_idx],
                        label=strategy_labels[strategy_idx], color=colors[strategy_idx],
                        linewidth=2, markersize=6, capsize=3, capthick=1.5, alpha=0.8)
            
            # Kernel count plot
            ax2.plot(test_nums, kernels, marker=markers[strategy_idx],
                    label=strategy_labels[strategy_idx], color=colors[strategy_idx],
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax1.set_ylabel("Runtime (seconds)", fontsize=11, fontweight='bold')
    ax1.set_title("Runtime Comparison with Min/Max Error Bars", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Test Number", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Number of Kernels", fontsize=11, fontweight='bold')
    ax2.set_title("Kernel Count Comparison", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sorted([int(k.split("_")[1]) for k in results.keys()]))
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 4 saved: {output_file}")

def variant5_stacked_area(results_file="benchmark_results.json", output_file="variant5_stacked_area.png"):
    """Variant 5: Stacked area chart showing kernel count distribution."""
    results = load_results(results_file)
    
    test_numbers = []
    strategies = ["trace", "naive", "barrier"]
    strategy_labels = ["Trace", "Naive", "Barrier"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    
    kernel_data = {s: [] for s in strategies}
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_numbers.append(test_num)
        
        for strategy in strategies:
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                kernel_data[strategy].append(data["kernels"])
            else:
                kernel_data[strategy].append(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Stack the data
    bottom = np.zeros(len(test_numbers))
    for i, strategy in enumerate(strategies):
        ax.fill_between(test_numbers, bottom, bottom + kernel_data[strategy],
                       label=strategy_labels[i], color=colors[i], alpha=0.7, linewidth=1.5)
        bottom += kernel_data[strategy]
    
    ax.set_xlabel("Test Number", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative Kernel Count", fontsize=11, fontweight='bold')
    ax.set_title("Stacked Kernel Count Comparison", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(test_numbers)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 5 saved: {output_file}")

def variant6_side_by_side(results_file="benchmark_results.json", output_file="variant6_side_by_side.png"):
    """Variant 6: Side-by-side comparison with separate y-axes."""
    results = load_results(results_file)
    
    test_names = []
    test_numbers = []
    strategies = ["trace", "naive", "barrier"]
    strategy_labels = ["Trace", "Naive", "Barrier"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    
    runtime_data = {s: [] for s in strategies}
    kernel_data = {s: [] for s in strategies}
    
    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"T{test_num}")
        test_numbers.append(test_num)
        
        for strategy in strategies:
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                runtime_data[strategy].append(data["time"])
                kernel_data[strategy].append(data["kernels"])
            else:
                runtime_data[strategy].append(0)
                kernel_data[strategy].append(0)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Runtime
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(test_names))
    width = 0.25
    for i, strategy in enumerate(strategies):
        offset = (i - 1) * width
        ax1.bar(x + offset, runtime_data[strategy], width, label=strategy_labels[i], 
                color=colors[i], alpha=0.8)
    ax1.set_ylabel("Runtime (s)", fontsize=10, fontweight='bold')
    ax1.set_title("Runtime", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=0, fontsize=8)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top right: Kernels
    ax2 = fig.add_subplot(gs[0, 1])
    for i, strategy in enumerate(strategies):
        offset = (i - 1) * width
        ax2.bar(x + offset, kernel_data[strategy], width, label=strategy_labels[i], 
                color=colors[i], alpha=0.8)
    ax2.set_ylabel("Kernels", fontsize=10, fontweight='bold')
    ax2.set_title("Kernel Count", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=0, fontsize=8)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Speedup
    ax3 = fig.add_subplot(gs[1, :])
    speedup_data = {s: [] for s in ["naive", "barrier"]}
    for i, test_name in enumerate(sorted(results.keys())):
        trace_time = results[test_name].get("trace", {}).get("time")
        if trace_time:
            for strategy in ["naive", "barrier"]:
                data = results[test_name].get(strategy, {})
                if data.get("success") and data.get("time"):
                    speedup_data[strategy].append(trace_time / data["time"])
                else:
                    speedup_data[strategy].append(0)
        else:
            for strategy in ["naive", "barrier"]:
                speedup_data[strategy].append(0)
    
    x_speedup = np.arange(len(test_names))
    for i, strategy in enumerate(["naive", "barrier"]):
        offset = (i - 1) * width
        ax3.bar(x_speedup + offset, speedup_data[strategy], width,
               label=strategy_labels[strategies.index(strategy)], 
               color=colors[strategies.index(strategy)], alpha=0.8)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel("Test Case", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Speedup (vs Trace)", fontsize=10, fontweight='bold')
    ax3.set_title("Speedup Comparison", fontsize=11, fontweight='bold')
    ax3.set_xticks(x_speedup)
    ax3.set_xticklabels(test_names, rotation=0, fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Variant 6 saved: {output_file}")

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    
    print("Generating plot variants...")
    print("=" * 70)
    
    variant1_grouped_bars(results_file, "variant1_grouped_bars.png")
    variant2_heatmap(results_file, "variant2_heatmap.png")
    variant3_speedup_normalized(results_file, "variant3_speedup.png")
    variant4_line_with_errorbars(results_file, "variant4_line_errorbars.png")
    variant5_stacked_area(results_file, "variant5_stacked_area.png")
    variant6_side_by_side(results_file, "variant6_side_by_side.png")
    
    print("=" * 70)
    print("All variants generated!")

