#!/usr/bin/env python3
"""
Plot benchmark results with average/low/high visualization.
Shows average as a line, low/high as shaded area around the average.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def plot_comparison(
    results_file="benchmark_results.json", output_file="fusion_comparison.png"
):
    """Create comparison plot showing average/low/high for each strategy."""

    results = load_results(results_file)

    # Extract test names and times for all strategies
    test_names = []
    test_numbers = []

    # Data structure: strategy -> list of (avg, min, max) tuples
    strategy_data = {
        "trace": {"avg": [], "min": [], "max": []},
        "naive": {"avg": [], "min": [], "max": []},
        "horizontal": {"avg": [], "min": [], "max": []},
        "barrier": {"avg": [], "min": [], "max": []},
    }

    for test_name in sorted(results.keys()):
        test_num = int(test_name.split("_")[1])
        test_names.append(f"Test {test_num}")
        test_numbers.append(test_num)

        for strategy in strategy_data.keys():
            data = results[test_name].get(strategy, {})
            if data.get("success"):
                strategy_data[strategy]["avg"].append(data["time"])
                strategy_data[strategy]["min"].append(
                    data.get("min_time", data["time"])
                )
                strategy_data[strategy]["max"].append(
                    data.get("max_time", data["time"])
                )
            else:
                strategy_data[strategy]["avg"].append(None)
                strategy_data[strategy]["min"].append(None)
                strategy_data[strategy]["max"].append(None)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors for each strategy
    colors = {
        "trace": "#95a5a6",
        "naive": "#3498db",
        "horizontal": "#2ecc71",
        "barrier": "#e74c3c",
    }

    labels = {
        "trace": "Trace (No Fusion)",
        "naive": "Naive (Vertical)",
        "horizontal": "Horizontal",
        "barrier": "Barrier (True Horizontal)",
    }

    # Plot each strategy
    for strategy in ["trace", "naive", "horizontal", "barrier"]:
        avg_times = strategy_data[strategy]["avg"]
        min_times = strategy_data[strategy]["min"]
        max_times = strategy_data[strategy]["max"]

        # Filter out None values for plotting
        valid_indices = [
            i for i in range(len(test_numbers)) if avg_times[i] is not None
        ]

        if not valid_indices:
            continue

        x_vals = [test_numbers[i] for i in valid_indices]
        avg_vals = [avg_times[i] for i in valid_indices]
        min_vals = [min_times[i] for i in valid_indices]
        max_vals = [max_times[i] for i in valid_indices]

        # Plot shaded area for min-max range
        ax.fill_between(
            x_vals,
            min_vals,
            max_vals,
            alpha=0.2,
            color=colors[strategy],
            label=f"{labels[strategy]} (min-max range)",
        )

        # Plot average line
        ax.plot(
            x_vals,
            avg_vals,
            marker="o",
            markersize=4,
            linewidth=2,
            color=colors[strategy],
            label=labels[strategy],
            linestyle="-",
        )

    ax.set_xlabel("Test Number", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(
        "Benchmark Results - Runtime Comparison with Min/Max Range",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Set x-axis to show test numbers
    ax.set_xticks(test_numbers)
    ax.set_xticklabels([f"Test {n}" for n in test_numbers], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for strategy in ["trace", "naive", "horizontal", "barrier"]:
        avg_times = [t for t in strategy_data[strategy]["avg"] if t is not None]
        if avg_times:
            overall_avg = np.mean(avg_times)
            overall_min = min(
                [t for t in strategy_data[strategy]["min"] if t is not None]
            )
            overall_max = max(
                [t for t in strategy_data[strategy]["max"] if t is not None]
            )
            print(
                f"{labels[strategy]:<25} Avg: {overall_avg:.4f}s  Min: {overall_min:.4f}s  Max: {overall_max:.4f}s"
            )

    return fig


if __name__ == "__main__":
    import sys

    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "fusion_comparison.png"

    plot_comparison(results_file, output_file)
