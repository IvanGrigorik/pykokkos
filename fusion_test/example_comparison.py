#!/usr/bin/env python3
"""
Side-by-side comparison of horizontal vs vertical fusion strategies.

This script demonstrates scenarios where the choice of fusion strategy matters
and shows what gets fused under each strategy.

Run with different strategies:
  PK_FUSION=horizontal python example_comparison.py
  PK_FUSION=naive python example_comparison.py
  PK_FUSION=trace python example_comparison.py  # No fusion
"""

import os
import sys
import pykokkos as pk
import numpy as np


# ============================================================================
# WORKUNIT DEFINITIONS
# ============================================================================


@pk.workunit
def init_A(i: int, A: pk.View1D[float], val: float):
    A[i] = val


@pk.workunit
def init_B(i: int, B: pk.View1D[float], val: float):
    B[i] = val


@pk.workunit
def compute_C_from_A(i: int, A: pk.View1D[float], C: pk.View1D[float]):
    """C depends on A."""
    C[i] = A[i] * 2.0


@pk.workunit
def compute_D_from_B(i: int, B: pk.View1D[float], D: pk.View1D[float]):
    """D depends on B."""
    D[i] = B[i] * 3.0


@pk.workunit
def compute_E_from_C(i: int, C: pk.View1D[float], E: pk.View1D[float]):
    """E depends on C (which depends on A)."""
    E[i] = C[i] + 10.0


@pk.workunit
def modify_inplace(i: int, X: pk.View1D[float], factor: float):
    """Modify array in place."""
    X[i] = X[i] * factor


# ============================================================================
# SCENARIO 1: Independent Branches
# ============================================================================


def scenario_1_independent_branches():
    """
    Two independent computation branches.

    Branch 1: A -> C -> E  (A produces C, C produces E)
    Branch 2: B -> D       (B produces D)

    Horizontal fusion:
      - CAN fuse: init_A + init_B (independent initializations)
      - CANNOT fuse: compute_C_from_A + compute_D_from_B (both depend on earlier kernels)
      - CANNOT fuse: compute_E_from_C with others (depends on C)

    Vertical fusion:
      - CAN fuse within each branch if consecutive
      - Branch 1: init_A -> compute_C_from_A -> compute_E_from_C (sequential)
      - Branch 2: init_B -> compute_D_from_B (sequential)

    Result: Horizontal is better here because branches are independent!
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Two Independent Computation Branches")
    print("=" * 70)

    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)
    E = pk.View([N], dtype=float)

    # Branch 1: A -> C -> E
    pk.parallel_for(N, init_A, A=A, val=5.0)
    pk.parallel_for(N, compute_C_from_A, A=A, C=C)
    pk.parallel_for(N, compute_E_from_C, C=C, E=E)

    # Branch 2: B -> D
    pk.parallel_for(N, init_B, B=B, val=7.0)
    pk.parallel_for(N, compute_D_from_B, B=B, D=D)
    pk.flush()

    print(f"A[0] = {A[0]}, C[0] = {C[0]}, E[0] = {E[0]}")
    print(f"B[0] = {B[0]}, D[0] = {D[0]}")

    print("\nHorizontal fusion strategy:")
    print("  [OK] Can fuse some independent operations across branches")
    print("  -> Better for this pattern!")
    print("\nVertical fusion strategy:")
    print("  [OK] Can fuse within each branch")
    print("  -> Misses opportunity to parallelize branches")


# ============================================================================
# SCENARIO 2: Long Sequential Chain
# ============================================================================


def scenario_2_sequential_chain():
    """
    Long chain of dependent operations on same array.

    Horizontal fusion: CANNOT fuse (all operations dependent)
    Vertical fusion: CAN fuse entire chain

    Result: Vertical fusion is clearly better!
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Long Sequential Chain")
    print("=" * 70)

    N = 1000
    X = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=X, val=2.0)

    # Long chain of dependent operations
    for i in range(5):
        pk.parallel_for(N, modify_inplace, X=X, factor=1.1)

    expected = 2.0 * (1.1**5)
    print(f"X[0] = {X[0]:.6f} (expected: {expected:.6f})")

    print("\nHorizontal fusion strategy:")
    print("  [X] Cannot fuse (each operation depends on previous)")
    print("\nVertical fusion strategy:")
    print("  [OK] Can fuse entire chain")
    print("  -> Clearly better for this pattern!")


# ============================================================================
# SCENARIO 3: Mixed Pattern
# ============================================================================


@pk.workunit
def independent_op_1(i: int, X: pk.View1D[float]):
    X[i] = float(i)


@pk.workunit
def independent_op_2(i: int, Y: pk.View1D[float]):
    Y[i] = float(i * 2)


@pk.workunit
def dependent_op(i: int, X: pk.View1D[float], Y: pk.View1D[float], Z: pk.View1D[float]):
    """Depends on both X and Y."""
    Z[i] = X[i] + Y[i]


def scenario_3_mixed_pattern():
    """
    Mixed pattern with both independent and dependent operations.
    
    independent_op_1 (X)  \\
                            --> dependent_op (X, Y -> Z)
    independent_op_2 (Y)  /
    
    Horizontal fusion: CAN fuse independent_op_1 + independent_op_2
    Vertical fusion: CAN fuse if operations are consecutive
    
    Result: Both strategies have merits depending on execution order!
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Mixed Independent and Dependent Operations")
    print("=" * 70)

    N = 1000
    X = pk.View([N], dtype=float)
    Y = pk.View([N], dtype=float)
    Z = pk.View([N], dtype=float)

    # Two independent operations
    pk.parallel_for(N, independent_op_1, X=X)
    pk.parallel_for(N, independent_op_2, Y=Y)

    # One dependent operation that needs both results
    pk.parallel_for(N, dependent_op, X=X, Y=Y, Z=Z)

    print(f"X[5] = {X[5]}, Y[5] = {Y[5]}, Z[5] = {Z[5]}")

    print("\nHorizontal fusion strategy:")
    print("  [OK] Can fuse independent_op_1 + independent_op_2")
    print("  -> Good for the initialization phase")
    print("\nVertical fusion strategy:")
    print("  [OK] Can fuse if operations are consecutive")
    print("  -> May miss parallelization opportunity")


# ============================================================================
# SCENARIO 4: Multiple Independent Pipelines
# ============================================================================


def scenario_4_multiple_pipelines():
    """
    Multiple independent pipelines running in parallel.

    Pipeline 1: A1 -> B1 -> C1
    Pipeline 2: A2 -> B2 -> C2

    Horizontal fusion: CAN fuse operations at same stage across pipelines
                       (A1 + A2, B1 + B2, C1 + C2)
    Vertical fusion: CAN fuse within each pipeline
                     (A1 -> B1 -> C1, A2 -> B2 -> C2)

    Result: Horizontal fusion is better for balancing the pipelines!
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Multiple Independent Pipelines")
    print("=" * 70)

    N = 1000

    # Pipeline 1
    A1 = pk.View([N], dtype=float)
    B1 = pk.View([N], dtype=float)
    C1 = pk.View([N], dtype=float)

    # Pipeline 2
    A2 = pk.View([N], dtype=float)
    B2 = pk.View([N], dtype=float)
    C2 = pk.View([N], dtype=float)

    # Stage 1: Initialize both pipelines
    pk.parallel_for(N, init_A, A=A1, val=1.0)
    pk.parallel_for(N, init_A, A=A2, val=2.0)

    # Stage 2: First transformation
    pk.parallel_for(N, compute_C_from_A, A=A1, C=B1)
    pk.parallel_for(N, compute_C_from_A, A=A2, C=B2)

    # Stage 3: Second transformation
    pk.parallel_for(N, compute_E_from_C, C=B1, E=C1)
    pk.parallel_for(N, compute_E_from_C, C=B2, E=C2)

    print(f"Pipeline 1: A1[0]={A1[0]}, B1[0]={B1[0]}, C1[0]={C1[0]}")
    print(f"Pipeline 2: A2[0]={A2[0]}, B2[0]={B2[0]}, C2[0]={C2[0]}")

    print("\nHorizontal fusion strategy:")
    print("  [OK] Can fuse operations at each stage across pipelines")
    print("  -> Excellent for balancing multiple pipelines!")
    print("\nVertical fusion strategy:")
    print("  [OK] Can fuse within each pipeline")
    print("  -> Processes one pipeline at a time (less parallel)")


# ============================================================================
# MAIN
# ============================================================================


def print_current_strategy():
    strategy = os.environ.get("PK_FUSION", "trace")
    print("\n" + "=" * 70)
    print(f"FUSION STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\nCurrent strategy: {strategy.upper()}")

    if strategy == "horizontal":
        print("-> Fuses INDEPENDENT operations that can run in parallel")
    elif strategy == "naive":
        print("-> Fuses CONSECUTIVE operations in execution order")
    elif strategy == "trace":
        print("-> NO FUSION (baseline for comparison)")

    print("\nTip: Run with different PK_FUSION values to compare:")
    print("  PK_FUSION=horizontal python example_comparison.py")
    print("  PK_FUSION=naive python example_comparison.py")
    print("  PK_FUSION=trace python example_comparison.py")


def main():
    print_current_strategy()

    scenario_1_independent_branches()
    # scenario_2_sequential_chain()
    # scenario_3_mixed_pattern()
    # scenario_4_multiple_pipelines()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBest fusion strategy depends on the computation pattern:")
    print("")
    print("Use HORIZONTAL fusion when:")
    print("  * Operations are independent (no data dependencies)")
    print("  * Multiple parallel pipelines or streams")
    print("  * Want to maximize parallelism")
    print("")
    print("Use VERTICAL fusion when:")
    print("  * Operations form a sequential pipeline")
    print("  * Read-after-write dependencies exist")
    print("  * Single stream of dependent operations")
    print("")
    print("Ideal: Hybrid strategy that chooses based on dependencies!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
