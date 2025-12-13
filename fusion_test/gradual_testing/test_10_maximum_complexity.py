#!/usr/bin/env python3
"""
Test 10: Maximum Complexity (HARDEST)
Multiple independent sources, complex intermediate computations,
multiple sinks. Tests all fusion strategies to their limits.
A->B, A->C, D->E, B->F, C->F, E->F, F->G, F->H
Expected: Most complex dependency pattern, hardest to fuse optimally.
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
def compute_C(i: int, A: pk.View1D[float], C: pk.View1D[float]):
    C[i] = A[i] * 3.0


@pk.workunit
def init_D(i: int, D: pk.View1D[float], val: float):
    D[i] = val


@pk.workunit
def compute_E(i: int, D: pk.View1D[float], E: pk.View1D[float]):
    E[i] = D[i] * 4.0


@pk.workunit
def compute_F(
    i: int,
    B: pk.View1D[float],
    C: pk.View1D[float],
    E: pk.View1D[float],
    F: pk.View1D[float],
):
    F[i] = B[i] + C[i] + E[i]


@pk.workunit
def compute_G(i: int, F: pk.View1D[float], G: pk.View1D[float]):
    G[i] = F[i] * 2.0


@pk.workunit
def compute_H(i: int, F: pk.View1D[float], H: pk.View1D[float]):
    H[i] = F[i] * 3.0


def run_test():
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)
    E = pk.View([N], dtype=float)
    F = pk.View([N], dtype=float)
    G = pk.View([N], dtype=float)
    H = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=1.0)
    pk.parallel_for(N, compute_B, A=A, B=B)
    pk.parallel_for(N, compute_C, A=A, C=C)
    pk.parallel_for(N, init_D, D=D, val=2.0)
    pk.parallel_for(N, compute_E, D=D, E=E)
    pk.parallel_for(N, compute_F, B=B, C=C, E=E, F=F)
    pk.parallel_for(N, compute_G, F=F, G=G)
    pk.parallel_for(N, compute_H, F=F, H=H)

    pk.flush()

    assert A[0] == 1.0, f"A[0] = {A[0]}, expected 1.0"
    assert B[0] == 2.0, f"B[0] = {B[0]}, expected 2.0"
    assert C[0] == 3.0, f"C[0] = {C[0]}, expected 3.0"
    assert D[0] == 2.0, f"D[0] = {D[0]}, expected 2.0"
    assert E[0] == 8.0, f"E[0] = {E[0]}, expected 8.0"
    assert F[0] == 13.0, f"F[0] = {F[0]}, expected 13.0"
    assert G[0] == 26.0, f"G[0] = {G[0]}, expected 26.0"
    assert H[0] == 39.0, f"H[0] = {H[0]}, expected 39.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 10: {'PASSED' if success else 'FAILED'}")
