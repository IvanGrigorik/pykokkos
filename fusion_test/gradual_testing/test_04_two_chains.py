#!/usr/bin/env python3
"""
Test 4: Two Independent Chains
Two parallel dependency chains: A->B and C->D
Expected: Horizontal fuses across chains, Vertical fuses within chains.
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


@pk.workunit
def init_C(i: int, C: pk.View1D[float], val: float):
    C[i] = val


@pk.workunit
def compute_D(i: int, C: pk.View1D[float], D: pk.View1D[float]):
    D[i] = C[i] * 3.0


def run_test():
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=2.0)
    pk.parallel_for(N, compute_B, A=A, B=B)
    pk.parallel_for(N, init_C, C=C, val=3.0)
    pk.parallel_for(N, compute_D, C=C, D=D)

    pk.flush()

    assert A[0] == 2.0, f"A[0] = {A[0]}, expected 2.0"
    assert B[0] == 4.0, f"B[0] = {B[0]}, expected 4.0"
    assert C[0] == 3.0, f"C[0] = {C[0]}, expected 3.0"
    assert D[0] == 9.0, f"D[0] = {D[0]}, expected 9.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 4: {'PASSED' if success else 'FAILED'}")
