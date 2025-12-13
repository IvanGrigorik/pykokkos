#!/usr/bin/env python3
"""
Test 11: Independent Operations with Different Ranges
Independent operations that operate on different ranges.
Expected: Naive fusion cannot fuse (different ranges), but horizontal/barrier can fuse into 1 kernel.
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
def init_C(i: int, C: pk.View1D[float], val: float):
    C[i] = val


def run_test():
    N1 = 500
    N2 = 1000
    N3 = 750
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    A = pk.View([N1], dtype=float)
    B = pk.View([N2], dtype=float)
    C = pk.View([N3], dtype=float)

    # Different ranges - naive fusion cannot fuse these
    pk.parallel_for(N1, init_A, A=A, val=1.0)
    pk.parallel_for(N2, init_B, B=B, val=2.0)
    pk.parallel_for(N3, init_C, C=C, val=3.0)

    pk.flush()

    assert A[0] == 1.0, f"A[0] = {A[0]}, expected 1.0"
    assert B[0] == 2.0, f"B[0] = {B[0]}, expected 2.0"
    assert C[0] == 3.0, f"C[0] = {C[0]}, expected 3.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 11: {'PASSED' if success else 'FAILED'}")
