#!/usr/bin/env python3
"""
Test 2: Three Independent Operations
Three completely independent kernels.
Expected: All strategies should fuse into 1 kernel.
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
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=1.0)
    pk.parallel_for(N, init_B, B=B, val=2.0)
    pk.parallel_for(N, init_C, C=C, val=3.0)

    pk.flush()

    assert A[0] == 1.0, f"A[0] = {A[0]}, expected 1.0"
    assert B[0] == 2.0, f"B[0] = {B[0]}, expected 2.0"
    assert C[0] == 3.0, f"C[0] = {C[0]}, expected 3.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 2: {'PASSED' if success else 'FAILED'}")
