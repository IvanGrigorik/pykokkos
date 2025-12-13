#!/usr/bin/env python3
"""
Test 9: Complex DAG Pattern
Multiple sources, intermediate nodes, and sinks.
A->B, A->C, B->D, C->D, D->E
Expected: Tests fusion with complex dependency graph.
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
def compute_D(i: int, B: pk.View1D[float], C: pk.View1D[float], D: pk.View1D[float]):
    D[i] = B[i] + C[i]


@pk.workunit
def compute_E(i: int, D: pk.View1D[float], E: pk.View1D[float]):
    E[i] = D[i] * 2.0


def run_test():
    N = 1000
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    D = pk.View([N], dtype=float)
    E = pk.View([N], dtype=float)

    pk.parallel_for(N, init_A, A=A, val=2.0)
    pk.parallel_for(N, compute_B, A=A, B=B)
    pk.parallel_for(N, compute_C, A=A, C=C)
    pk.parallel_for(N, compute_D, B=B, C=C, D=D)
    pk.parallel_for(N, compute_E, D=D, E=E)

    pk.flush()

    assert A[0] == 2.0, f"A[0] = {A[0]}, expected 2.0"
    assert B[0] == 4.0, f"B[0] = {B[0]}, expected 4.0"
    assert C[0] == 6.0, f"C[0] = {C[0]}, expected 6.0"
    assert D[0] == 10.0, f"D[0] = {D[0]}, expected 10.0"
    assert E[0] == 20.0, f"E[0] = {E[0]}, expected 20.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 9: {'PASSED' if success else 'FAILED'}")
