#!/usr/bin/env python3
"""
Test 3: Simple Sequential Chain
Two operations with a simple dependency: A -> B
Expected: Vertical/Barrier fuse into 1, Horizontal creates 2 kernels.
"""

import os
import sys

sys.path.insert(0, "/home/sifi/pykokkos-fuse")

import pykokkos as pk


@pk.workunit
def init_A(i: int, A: pk.View1D[float], val: float):
    A[i] = val


@pk.workunit
def compute_B(i: int, A: pk.View1D[float], B: pk.View1D[float]):
    B[i] = A[i] * 2.0


def run_test():
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=5.0)
    pk.parallel_for(N, compute_B, A=A, B=B)

    pk.flush()

    assert A[0] == 5.0, f"A[0] = {A[0]}, expected 5.0"
    assert B[0] == 10.0, f"B[0] = {B[0]}, expected 10.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 3: {'PASSED' if success else 'FAILED'}")
