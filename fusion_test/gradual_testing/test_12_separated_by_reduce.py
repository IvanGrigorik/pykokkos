#!/usr/bin/env python3
"""
Test 12: Independent Operations Separated by Reduce
Independent operations that are separated by a reduce operation.
Expected: Naive fusion creates separate kernels (reduce breaks fusion),
          but horizontal/barrier can fuse independent ops into fewer kernels.
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


@pk.workunit
def sum_reduce(i: int, acc: pk.Acc[float], A: pk.View1D[float]):
    acc += A[i]


def run_test():
    N = 1000
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)

    # Independent operations separated by reduce
    # Naive fusion will create: [init_A, init_B] fused, then reduce, then init_C separate
    pk.parallel_for(N, init_A, A=A, val=1.0)
    pk.parallel_for(N, init_B, B=B, val=2.0)
    pk.flush()  # Flush before reduce to ensure A is written
    result = pk.parallel_reduce(N, sum_reduce, A=A)
    pk.parallel_for(N, init_C, C=C, val=3.0)

    pk.flush()

    assert A[0] == 1.0, f"A[0] = {A[0]}, expected 1.0"
    assert B[0] == 2.0, f"B[0] = {B[0]}, expected 2.0"
    assert C[0] == 3.0, f"C[0] = {C[0]}, expected 3.0"
    assert abs(result - N * 1.0) < 0.01, f"Sum = {result}, expected {N * 1.0}"

    return True


if __name__ == "__main__":
    success = run_test()
    print(f"Test 12: {'PASSED' if success else 'FAILED'}")
