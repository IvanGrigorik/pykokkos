#!/usr/bin/env python3
"""
Test 13: Mixed Independent Operations with Different Ranges and Dependencies
Some operations have dependencies, some are independent with different ranges.
Expected: Naive fusion creates multiple kernels (due to ranges and dependencies),
          horizontal creates multiple kernels (due to dependencies),
          barrier fuses everything into 1 kernel with barriers.
"""

import os
import sys

sys.path.insert(0, "/home/sifi/pykokkos-fuse")

import pykokkos as pk


@pk.workunit
def init_A(i: int, A: pk.View1D[float], val: float):
    A[i] = val


@pk.workunit
def init_B(i: int, B: pk.View1D[float], val: float):
    B[i] = val


@pk.workunit
def compute_C(i: int, A: pk.View1D[float], C: pk.View1D[float]):
    C[i] = A[i] * 2.0


@pk.workunit
def init_D(i: int, D: pk.View1D[float], val: float):
    D[i] = val


def run_test():
    N1 = 800
    N2 = 1200
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    A = pk.View([N1], dtype=float)
    B = pk.View([N2], dtype=float)
    C = pk.View([N1], dtype=float)
    D = pk.View([N2], dtype=float)

    # Independent A and B (different ranges)
    # Then C depends on A (same range as A)
    # Then independent D (different range)
    pk.parallel_for(N1, init_A, A=A, val=5.0)
    pk.parallel_for(N2, init_B, B=B, val=10.0)
    pk.parallel_for(N1, compute_C, A=A, C=C)
    pk.parallel_for(N2, init_D, D=D, val=15.0)

    pk.flush()

    assert A[0] == 5.0, f"A[0] = {A[0]}, expected 5.0"
    assert B[0] == 10.0, f"B[0] = {B[0]}, expected 10.0"
    assert C[0] == 10.0, f"C[0] = {C[0]}, expected 10.0"
    assert D[0] == 15.0, f"D[0] = {D[0]}, expected 15.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 13: {'PASSED' if success else 'FAILED'}")
