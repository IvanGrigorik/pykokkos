#!/usr/bin/env python3
"""
Example demonstrating horizontal fusion.

These kernels are INDEPENDENT - they can be fused horizontally but NOT vertically
because they don't have data dependencies. Each kernel operates on different data.

Run with: PK_FUSION=horizontal python example_horizontal.py
"""

import os
os.environ["PK_FUSION"] = "horizontal"

import pykokkos as pk
import numpy as np


# ============================================================================
# WORKUNIT DEFINITIONS
# ============================================================================

@pk.workunit
def initialize_array_A(i: int, A: pk.View1D[float], value: float):
    """Initialize array A with a constant value."""
    A[i] = value


@pk.workunit
def initialize_array_B(i: int, B: pk.View1D[float], value: float):
    """Initialize array B with a constant value."""
    B[i] = value


@pk.workunit
def initialize_array_C(i: int, C: pk.View1D[float], value: float):
    """Initialize array C with a constant value."""
    C[i] = value


@pk.workunit
def compute_squares(i: int, input: pk.View1D[float], output: pk.View1D[float]):
    """Compute squares of input array into output array."""
    output[i] = input[i] * input[i]


@pk.workunit
def compute_cubes(i: int, input: pk.View1D[float], output: pk.View1D[float]):
    """Compute cubes of input array into output array."""
    output[i] = input[i] * input[i] * input[i]


@pk.workunit
def apply_transform_1(i: int, input: pk.View1D[float], output: pk.View1D[float], scale: float):
    """Apply transformation: output = input * scale."""
    output[i] = input[i] * scale


@pk.workunit
def apply_transform_2(i: int, input: pk.View1D[float], output: pk.View1D[float], offset: float):
    """Apply transformation: output = input + offset."""
    output[i] = input[i] + offset


# ============================================================================
# EXAMPLE 1: Independent Initializations
# ============================================================================

def example_1_independent_initializations():
    """
    Three arrays are initialized independently.
    
    Horizontal fusion: CAN fuse all three (they're independent)
    Vertical fusion: CAN also fuse (no dependencies, consecutive operations)
    
    BUT: This is a better fit for horizontal fusion conceptually since
    these are truly parallel, independent operations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Independent Initializations")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    
    # These three operations are completely independent
    # They can all be fused horizontally
    pk.parallel_for(N, initialize_array_A, A=A, value=1.0)
    pk.parallel_for(N, initialize_array_B, B=B, value=2.0)
    pk.parallel_for(N, initialize_array_C, C=C, value=3.0)
    
    # Force execution
    print(f"A[0] = {A[0]} (expected: 1.0)")
    print(f"B[0] = {B[0]} (expected: 2.0)")
    print(f"C[0] = {C[0]} (expected: 3.0)")
    
    print("\n[OK] All three kernels can be fused horizontally")
    print("  (No data dependencies between them)")


# ============================================================================
# EXAMPLE 2: Independent Computations on Different Data
# ============================================================================

def example_2_independent_computations():
    """
    Two independent computations on different input/output pairs.
    
    Horizontal fusion: CAN fuse (independent operations)
    Vertical fusion: CAN also fuse if consecutive
    
    This is ideal for horizontal fusion.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Independent Computations on Different Data")
    print("="*70)
    
    N = 1000
    
    # Input arrays
    X = pk.View([N], dtype=float)
    Y = pk.View([N], dtype=float)
    X.fill(2.0)
    Y.fill(3.0)
    
    # Output arrays
    X_squared = pk.View([N], dtype=float)
    Y_cubed = pk.View([N], dtype=float)
    
    # Two completely independent computations
    # One computes squares, other computes cubes
    pk.parallel_for(N, compute_squares, input=X, output=X_squared)
    pk.parallel_for(N, compute_cubes, input=Y, output=Y_cubed)
    
    # Force execution
    print(f"X_squared[0] = {X_squared[0]} (expected: 4.0)")
    print(f"Y_cubed[0] = {Y_cubed[0]} (expected: 27.0)")
    
    print("\n[OK] Both kernels can be fused horizontally")
    print("  (No shared data, completely independent)")


# ============================================================================
# EXAMPLE 3: Multiple Independent Transformations
# ============================================================================

def example_3_multiple_transformations():
    """
    Multiple transformations applied to different data streams.
    
    Horizontal fusion: CAN fuse (all independent)
    Vertical fusion: CAN also fuse if consecutive
    
    Classic horizontal fusion scenario.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Independent Transformations")
    print("="*70)
    
    N = 1000
    
    # Multiple data streams
    stream1 = pk.View([N], dtype=float)
    stream2 = pk.View([N], dtype=float)
    stream1.fill(10.0)
    stream2.fill(20.0)
    
    # Output streams
    result1 = pk.View([N], dtype=float)
    result2 = pk.View([N], dtype=float)
    
    # Apply different transformations to each stream
    pk.parallel_for(N, apply_transform_1, input=stream1, output=result1, scale=2.0)
    pk.parallel_for(N, apply_transform_2, input=stream2, output=result2, offset=5.0)
    
    # Force execution
    print(f"result1[0] = {result1[0]} (expected: 20.0)")
    print(f"result2[0] = {result2[0]} (expected: 25.0)")
    
    print("\n[OK] Both transformations can be fused horizontally")
    print("  (Operating on independent data streams)")


# ============================================================================
# EXAMPLE 4: Cannot be Vertically Fused (but can be horizontally fused)
# ============================================================================

@pk.workunit
def read_and_modify_A(i: int, A: pk.View1D[float], B: pk.View1D[float]):
    """Read from A, write to B."""
    B[i] = A[i] * 2.0


@pk.workunit  
def read_and_modify_C(i: int, C: pk.View1D[float], D: pk.View1D[float]):
    """Read from C, write to D."""
    D[i] = C[i] * 3.0


def example_4_horizontal_only():
    """
    Scenario where operations access some shared read-only data
    but write to different outputs.
    
    Horizontal fusion: CAN fuse (writes to different outputs)
    Vertical fusion: Depends on safety analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Horizontal-Friendly Pattern")
    print("="*70)
    
    N = 1000
    
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)
    
    A.fill(5.0)
    C.fill(7.0)
    
    # Two operations that read from different sources
    # and write to different destinations
    pk.parallel_for(N, read_and_modify_A, A=A, B=B)
    pk.parallel_for(N, read_and_modify_C, C=C, D=D)
    
    # Force execution
    print(f"B[0] = {B[0]} (expected: 10.0)")
    print(f"D[0] = {D[0]} (expected: 21.0)")
    
    print("\n[OK] These operations are independent and can be fused horizontally")
    print("  (No data dependencies between them)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("HORIZONTAL FUSION EXAMPLES")
    print("="*70)
    print("\nThese examples show kernels that are INDEPENDENT and can be")
    print("fused horizontally because they don't have data dependencies.")
    print("\nFusion strategy: HORIZONTAL (PK_FUSION=horizontal)")
    
    example_1_independent_initializations()
    example_2_independent_computations()
    example_3_multiple_transformations()
    example_4_horizontal_only()
    
    print("\n" + "="*70)
    print("SUMMARY: Horizontal Fusion Benefits")
    print("="*70)
    print("* Fuses independent operations that can run in parallel")
    print("* Reduces kernel launch overhead")
    print("* Increases GPU occupancy by combining independent work")
    print("* Best for operations with no data dependencies")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

