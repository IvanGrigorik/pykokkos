#!/usr/bin/env python3
"""
Test 6: Mixed Independent and Dependent Operations
Some independent operations mixed with a dependency chain.
Expected: Horizontal fuses independent ops, creates separate kernels for chain.
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
def compute_D(i: int, C: pk.View1D[float], D: pk.View1D[float]):
    D[i] = C[i] + 5.0


def run_test():
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=3.0)
    pk.parallel_for(N, init_B, B=B, val=7.0)
    pk.parallel_for(N, compute_C, A=A, C=C)
    pk.parallel_for(N, compute_D, C=C, D=D)

    pk.flush()

    assert A[0] == 3.0, f"A[0] = {A[0]}, expected 3.0"
    assert B[0] == 7.0, f"B[0] = {B[0]}, expected 7.0"
    assert C[0] == 6.0, f"C[0] = {C[0]}, expected 6.0"
    assert D[0] == 11.0, f"D[0] = {D[0]}, expected 11.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 6: {'PASSED' if success else 'FAILED'}")
