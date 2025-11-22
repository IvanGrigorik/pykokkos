#!/usr/bin/env python3
"""
Examples of cases where fusion CANNOT occur under either strategy.

These examples demonstrate the limitations and safety checks of both
horizontal and vertical fusion strategies.

Run with: PK_FUSION=horizontal python example_no_fusion_case.py
      or: PK_FUSION=naive python example_no_fusion_case.py
"""

import os
import pykokkos as pk
import numpy as np


# ============================================================================
# CASE 1: Different Execution Ranges
# ============================================================================

@pk.workunit
def kernel_range_N(i: int, A: pk.View1D[float]):
    A[i] = float(i)


@pk.workunit
def kernel_range_M(i: int, B: pk.View1D[float]):
    B[i] = float(i * 2)


def case_1_different_ranges():
    """
    Operations with different execution ranges cannot be fused.
    
    NEITHER horizontal NOR vertical fusion can occur!
    
    Reason: The operations loop over different iteration spaces.
    """
    print("\n" + "="*70)
    print("CASE 1: Different Execution Ranges")
    print("="*70)
    
    N = 1000
    M = 500  # Different size!
    
    A = pk.View([N], dtype=float)
    B = pk.View([M], dtype=float)
    
    # These have different ranges and CANNOT be fused
    pk.parallel_for(N, kernel_range_N, A=A)
    pk.parallel_for(M, kernel_range_M, B=B)
    
    print(f"A has {len(A)} elements, B has {len(B)} elements")
    print(f"A[100] = {A[100]}, B[100] = {B[100]}")
    
    print("\n[X] Horizontal fusion: CANNOT fuse (different ranges)")
    print("[X] Vertical fusion: CANNOT fuse (different ranges)")
    print("\nFusion requires same iteration space (begin and end)!")


# ============================================================================
# CASE 2: Unsafe Access Patterns
# ============================================================================

@pk.workunit
def write_with_offset(i: int, A: pk.View1D[float], N: int):
    """Writes to A[i] and also accesses A[i+1]."""
    if i < N - 1:
        A[i] = float(i)
        A[i + 1] = A[i + 1] + 1.0  # Potential race condition!


@pk.workunit
def simple_write(i: int, A: pk.View1D[float]):
    """Simple write to A[i]."""
    A[i] = float(i * 2)


def case_2_unsafe_access():
    """
    Operations with unsafe/complex access patterns may not fuse.
    
    Fusion safety analysis may reject these operations.
    
    The safety checker looks at how arrays are indexed and determines
    if fusion could cause race conditions.
    """
    print("\n" + "="*70)
    print("CASE 2: Unsafe/Complex Access Patterns")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    A.fill(0.0)
    
    # The first kernel has a complex access pattern
    # The safety analysis might prevent fusion
    pk.parallel_for(N, write_with_offset, A=A, N=N)
    pk.parallel_for(N, simple_write, A=A)
    
    print(f"A[10] = {A[10]}")
    
    print("\n[X] Fusion may be prevented by safety analysis")
    print("  Reason: Complex indexing pattern (A[i+1]) detected")
    print("  Safety: Prevents potential race conditions")


# ============================================================================
# CASE 3: Reductions Cannot Be Fused (typically)
# ============================================================================

@pk.workunit
def reduction_sum(i: int, A: pk.View1D[float], acc: float):
    acc += A[i]


@pk.workunit
def simple_for(i: int, B: pk.View1D[float]):
    B[i] = float(i)


def case_3_reductions():
    """
    Reductions are typically not fused with other operations.
    
    Reason: Reductions have special semantics (accumulator) and
    different execution patterns than parallel_for.
    """
    print("\n" + "="*70)
    print("CASE 3: Reductions Not Fused")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    A.fill(1.0)
    
    # Reduction followed by parallel_for
    # These will NOT be fused
    result = pk.parallel_reduce(N, reduction_sum, A=A)
    pk.parallel_for(N, simple_for, B=B)
    
    print(f"Reduction result: {result}")
    print(f"B[10] = {B[10]}")
    
    print("\n[X] Reductions are not fused with parallel_for")
    print("  Reason: Different operation semantics")
    print("  Note: Current fusion strategies focus on parallel_for")


# ============================================================================
# CASE 4: Write-Write Conflicts
# ============================================================================

@pk.workunit
def write_all_A(i: int, A: pk.View1D[float]):
    A[i] = 1.0


@pk.workunit
def write_all_A_different(i: int, A: pk.View1D[float]):
    A[i] = 2.0


def case_4_write_conflicts():
    """
    Two operations writing to the same locations.
    
    Horizontal fusion: CANNOT fuse (write-write conflict)
    Vertical fusion: CAN fuse (sequential execution preserves semantics)
    
    This shows a key difference between the strategies!
    """
    print("\n" + "="*70)
    print("CASE 4: Write-Write Conflicts")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    A.fill(0.0)
    
    # Both write to same array locations
    pk.parallel_for(N, write_all_A, A=A)
    pk.parallel_for(N, write_all_A_different, A=A)
    
    print(f"A[0] = {A[0]} (should be 2.0 from second kernel)")
    
    print("\n[X] Horizontal fusion: CANNOT fuse")
    print("  Reason: Both write to same locations (conflict)")
    print("[OK] Vertical fusion: CAN fuse")
    print("  Reason: Sequential execution maintains correctness")


# ============================================================================
# CASE 5: Team Policy Operations
# ============================================================================

@pk.workunit
def team_kernel(team: pk.TeamMember, A: pk.View1D[float]):
    """Team policy kernel (more complex than RangePolicy)."""
    # Simplified team kernel
    i = team.league_rank()
    if i < len(A):
        A[i] = float(i)


@pk.workunit
def range_kernel(i: int, B: pk.View1D[float]):
    """Simple range policy kernel."""
    B[i] = float(i)


def case_5_team_policy():
    """
    TeamPolicy operations are not fused with RangePolicy operations.
    
    Reason: Different execution models (teams vs simple range).
    """
    print("\n" + "="*70)
    print("CASE 5: TeamPolicy vs RangePolicy")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    
    league_size = 100
    team_policy = pk.TeamPolicy(league_size, pk.AUTO)
    
    # TeamPolicy kernel
    pk.parallel_for(team_policy, team_kernel, A=A)
    
    # RangePolicy kernel
    pk.parallel_for(N, range_kernel, B=B)
    
    print(f"A[10] = {A[10]}, B[10] = {B[10]}")
    
    print("\n[X] Cannot fuse TeamPolicy with RangePolicy")
    print("  Reason: Different execution models")
    print("  Note: Current fusion focuses on RangePolicy operations")


# ============================================================================
# MAIN
# ============================================================================

def main():
    strategy = os.environ.get("PK_FUSION", "trace")
    
    print("\n" + "="*70)
    print("FUSION LIMITATION EXAMPLES")
    print("="*70)
    print(f"\nCurrent strategy: {strategy.upper()}")
    print("\nThese cases show when fusion CANNOT occur, regardless of strategy.")
    
    case_1_different_ranges()
    case_2_unsafe_access()
    case_3_reductions()
    case_4_write_conflicts()
    case_5_team_policy()
    
    print("\n" + "="*70)
    print("SUMMARY: Fusion Limitations")
    print("="*70)
    print("\nFusion is prevented by:")
    print("  * Different execution ranges (different N)")
    print("  * Unsafe memory access patterns")
    print("  * Reductions and scans (special semantics)")
    print("  * Write-write conflicts (horizontal fusion)")
    print("  * Different execution policies (Team vs Range)")
    print("\nThese safety checks ensure correctness!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


