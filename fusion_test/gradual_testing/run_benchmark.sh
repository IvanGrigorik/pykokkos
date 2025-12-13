#!/bin/bash
# Script to run benchmark comparing horizontal vs barrier fusion (true horizontal with barriers)

cd "$(dirname "$0")"

echo "Running benchmark: Horizontal vs Barrier Fusion (True Horizontal with Barriers)"
echo "================================================================================"
echo ""

# Set environment
export PYTHONPATH=/home/sifi/pykokkos-fuse:$PYTHONPATH
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PK_EXEC_SPACE=CUDA

# Run benchmark with trace, naive, and barrier strategies
python3 benchmark.py trace naive barrier

# Generate plot if results exist
if [ -f "benchmark_results.json" ]; then
    echo ""
    echo "Generating plot..."
    python3 plot_results.py benchmark_results.json fusion_comparison.png
    echo ""
    echo "Benchmark complete! Check fusion_comparison.png for results."
else
    echo ""
    echo "Warning: benchmark_results.json not found. Tests may have failed."
    echo "Please ensure PyKokkos environment is properly set up."
fi

