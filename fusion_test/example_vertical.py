#!/usr/bin/env python3
"""
Example demonstrating vertical/naive fusion.

These kernels have DEPENDENCIES - they must execute in order and can be
fused vertically but NOT horizontally because they have read-after-write
data dependencies.

Run with: PK_FUSION=naive python example_vertical.py
"""

import os
os.environ["PK_FUSION"] = "naive"

import pykokkos as pk
import numpy as np


# ============================================================================
# WORKUNIT DEFINITIONS
# ============================================================================

@pk.workunit
def initialize(i: int, A: pk.View1D[float], value: float):
    """Initialize array with a value."""
    A[i] = value


@pk.workunit
def square_inplace(i: int, A: pk.View1D[float]):
    """Square each element in-place."""
    A[i] = A[i] * A[i]


@pk.workunit
def add_constant(i: int, A: pk.View1D[float], constant: float):
    """Add a constant to each element."""
    A[i] = A[i] + constant


@pk.workunit
def multiply_constant(i: int, A: pk.View1D[float], factor: float):
    """Multiply each element by a constant."""
    A[i] = A[i] * factor


@pk.workunit
def copy_array(i: int, src: pk.View1D[float], dst: pk.View1D[float]):
    """Copy from src to dst."""
    dst[i] = src[i]


@pk.workunit
def transform_1(i: int, A: pk.View1D[float], B: pk.View1D[float]):
    """First transformation: B = A * 2."""
    B[i] = A[i] * 2.0


@pk.workunit
def transform_2(i: int, B: pk.View1D[float], C: pk.View1D[float]):
    """Second transformation: C = B + 3 (depends on B)."""
    C[i] = B[i] + 3.0


@pk.workunit
def transform_3(i: int, C: pk.View1D[float], D: pk.View1D[float]):
    """Third transformation: D = C * C (depends on C)."""
    D[i] = C[i] * C[i]


# ============================================================================
# EXAMPLE 1: Sequential In-Place Operations
# ============================================================================

def example_1_sequential_inplace():
    """
    Sequential operations on the same array, each modifying it in-place.
    
    Vertical fusion: CAN fuse (consecutive operations, same range)
    Horizontal fusion: CANNOT fuse (operations depend on previous results)
    
    Classic vertical fusion scenario - pipeline of transformations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Sequential In-Place Operations")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    
    # Pipeline of operations on the same array
    # Each operation depends on the result of the previous one
    pk.parallel_for(N, initialize, A=A, value=2.0)        # A = 2.0
    pk.parallel_for(N, square_inplace, A=A)               # A = 4.0
    pk.parallel_for(N, add_constant, A=A, constant=1.0)   # A = 5.0
    pk.parallel_for(N, multiply_constant, A=A, factor=2.0) # A = 10.0
    
    # Force execution
    print(f"A[0] = {A[0]} (expected: 10.0)")
    
    print("\n[OK] All operations can be fused vertically")
    print("  (Sequential pipeline on same array)")
    print("[X] Cannot be fused horizontally")
    print("  (Each operation depends on the previous result)")


# ============================================================================
# EXAMPLE 2: Producer-Consumer Chain
# ============================================================================

def example_2_producer_consumer():
    """
    Chain of operations where each produces data for the next.
    
    Vertical fusion: CAN fuse (sequential data flow)
    Horizontal fusion: CANNOT fuse (read-after-write dependencies)
    
    This represents a data processing pipeline.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Producer-Consumer Chain")
    print("="*70)
    
    N = 1000
    
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)
    
    A.fill(5.0)
    
    # Chain: A -> B -> C -> D
    # Each operation reads what the previous one wrote
    pk.parallel_for(N, transform_1, A=A, B=B)  # B depends on A
    pk.parallel_for(N, transform_2, B=B, C=C)  # C depends on B
    pk.parallel_for(N, transform_3, C=C, D=D)  # D depends on C
    
    # Force execution
    expected = ((5.0 * 2.0) + 3.0) ** 2  # ((A*2)+3)^2 = (13)^2 = 169
    print(f"D[0] = {D[0]} (expected: {expected})")
    
    print("\n[OK] Operations can be fused vertically")
    print("  (Sequential producer-consumer chain)")
    print("[X] Cannot be fused horizontally")
    print("  (Each operation needs result from previous)")


# ============================================================================
# EXAMPLE 3: Read-After-Write Dependencies
# ============================================================================

@pk.workunit
def write_pattern(i: int, A: pk.View1D[float]):
    """Write a pattern to array."""
    A[i] = float(i)


@pk.workunit
def accumulate(i: int, A: pk.View1D[float]):
    """Accumulate values (each element = element + index)."""
    A[i] = A[i] + float(i)


@pk.workunit
def normalize(i: int, A: pk.View1D[float], total: float):
    """Normalize by dividing by total."""
    A[i] = A[i] / total


def example_3_read_after_write():
    """
    Operations with clear read-after-write dependencies.
    
    Vertical fusion: CAN fuse (respects dependencies)
    Horizontal fusion: CANNOT fuse (dependencies prevent it)
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Read-After-Write Dependencies")
    print("="*70)
    
    N = 1000
    A = pk.View([N], dtype=float)
    
    # Each operation reads what was written by the previous one
    pk.parallel_for(N, write_pattern, A=A)         # Write initial pattern
    pk.parallel_for(N, accumulate, A=A)            # Read and modify
    pk.parallel_for(N, normalize, A=A, total=100.0) # Read and normalize
    
    # Force execution
    expected = (0.0 + 0.0) / 100.0  # First element: (i + i) / 100
    print(f"A[0] = {A[0]} (expected: {expected})")
    print(f"A[10] = {A[10]} (expected: {(10.0 + 10.0) / 100.0})")
    
    print("\n[OK] Operations can be fused vertically")
    print("  (Sequential modifications with dependencies)")
    print("[X] Cannot be fused horizontally")
    print("  (Read-after-write dependencies)")


# ============================================================================
# EXAMPLE 4: Iterative Refinement
# ============================================================================

@pk.workunit
def apply_iteration(i: int, x: pk.View1D[float], factor: float):
    """Apply one iteration of refinement."""
    x[i] = x[i] * factor + 0.1


def example_4_iterative():
    """
    Multiple iterations of the same operation.
    
    Vertical fusion: CAN fuse iterations
    Horizontal fusion: CANNOT fuse (each depends on previous)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Iterative Refinement")
    print("="*70)
    
    N = 1000
    x = pk.View([N], dtype=float)
    x.fill(1.0)
    
    # Multiple iterations, each depending on the previous
    num_iterations = 5
    for iteration in range(num_iterations):
        pk.parallel_for(N, apply_iteration, x=x, factor=0.9)
    
    # Force execution
    # After 5 iterations: ((((1*0.9+0.1)*0.9+0.1)*0.9+0.1)*0.9+0.1)*0.9+0.1
    result = 1.0
    for _ in range(num_iterations):
        result = result * 0.9 + 0.1
    print(f"x[0] = {x[0]} (expected: {result:.6f})")
    
    print("\n[OK] Iterations can be fused vertically")
    print("  (Sequential refinement steps)")
    print("[X] Cannot be fused horizontally")
    print("  (Each iteration depends on the previous)")


# ============================================================================
# EXAMPLE 5: Cannot Be Horizontally Fused - Shared Writes
# ============================================================================

@pk.workunit
def contribute_to_output(i: int, input: pk.View1D[float], output: pk.View1D[float]):
    """Add contribution from input to output."""
    output[i] = output[i] + input[i]


def example_5_shared_output():
    """
    Multiple operations writing to the same output array.
    
    Vertical fusion: CAN fuse if safe
    Horizontal fusion: CANNOT fuse (write conflicts)
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Sequential Contributions to Shared Output")
    print("="*70)
    
    N = 1000
    
    input1 = pk.View([N], dtype=float)
    input2 = pk.View([N], dtype=float)
    output = pk.View([N], dtype=float)
    
    input1.fill(1.0)
    input2.fill(2.0)
    output.fill(0.0)
    
    # Both operations modify the same output array
    # Must execute sequentially
    pk.parallel_for(N, contribute_to_output, input=input1, output=output)
    pk.parallel_for(N, contribute_to_output, input=input2, output=output)
    
    # Force execution
    print(f"output[0] = {output[0]} (expected: 3.0)")
    
    print("\n[OK] Can potentially be fused vertically (same array access pattern)")
    print("[X] Cannot be fused horizontally")
    print("  (Both write to same output, must be sequential)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("VERTICAL/NAIVE FUSION EXAMPLES")
    print("="*70)
    print("\nThese examples show kernels with DATA DEPENDENCIES that must")
    print("execute in order. They can be fused vertically but NOT horizontally.")
    print("\nFusion strategy: NAIVE/VERTICAL (PK_FUSION=naive)")
    
    example_1_sequential_inplace()
    example_2_producer_consumer()
    example_3_read_after_write()
    example_4_iterative()
    example_5_shared_output()
    
    print("\n" + "="*70)
    print("SUMMARY: Vertical Fusion Benefits")
    print("="*70)
    print("* Fuses consecutive operations that execute in order")
    print("* Reduces kernel launch overhead for sequential operations")
    print("* Respects data dependencies (read-after-write)")
    print("* Best for pipeline-style computations")
    print("* Preserves correctness of dependent operations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


