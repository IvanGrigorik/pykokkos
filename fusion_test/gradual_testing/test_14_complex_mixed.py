#!/usr/bin/env python3
"""
Test 14: Complex Mixed Scenario
Multiple independent operations with different ranges, separated by dependencies.
Expected: Naive fusion creates many kernels, barrier fuses into 1 kernel.
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


@pk.workunit
def compute_E(i: int, B: pk.View1D[float], E: pk.View1D[float]):
    E[i] = B[i] * 3.0


@pk.workunit
def init_F(i: int, F: pk.View1D[float], val: float):
    F[i] = val


def run_test():
    N1 = 600
    N2 = 900
    N3 = 1100
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    A = pk.View([N1], dtype=float)
    B = pk.View([N2], dtype=float)
    C = pk.View([N1], dtype=float)
    D = pk.View([N3], dtype=float)
    E = pk.View([N2], dtype=float)
    F = pk.View([N1], dtype=float)

    # Pattern: Independent (A, B) -> Dependent (C on A) -> Independent (D) -> Dependent (E on B) -> Independent (F)
    pk.parallel_for(N1, init_A, A=A, val=1.0)
    pk.parallel_for(N2, init_B, B=B, val=2.0)
    pk.parallel_for(N1, compute_C, A=A, C=C)
    pk.parallel_for(N3, init_D, D=D, val=3.0)
    pk.parallel_for(N2, compute_E, B=B, E=E)
    pk.parallel_for(N1, init_F, F=F, val=4.0)

    pk.flush()

    assert A[0] == 1.0, f"A[0] = {A[0]}, expected 1.0"
    assert B[0] == 2.0, f"B[0] = {B[0]}, expected 2.0"
    assert C[0] == 2.0, f"C[0] = {C[0]}, expected 2.0"
    assert D[0] == 3.0, f"D[0] = {D[0]}, expected 3.0"
    assert E[0] == 6.0, f"E[0] = {E[0]}, expected 6.0"
    assert F[0] == 4.0, f"F[0] = {F[0]}, expected 4.0"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 14: {'PASSED' if success else 'FAILED'}")
