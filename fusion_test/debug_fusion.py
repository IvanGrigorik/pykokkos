#!/usr/bin/env python3
"""Debug script to understand what horizontal fusion is doing."""

import os
os.environ["PK_FUSION"] = "horizontal"

import sys
sys.path.insert(0, '/home/sifi/pykokkos-fuse')

import pykokkos as pk

@pk.workunit
def init_A(i: int, A: pk.View1D[float], val: float):
    A[i] = val

@pk.workunit
def compute_C_from_A(i: int, A: pk.View1D[float], C: pk.View1D[float]):
    """C depends on A."""
    C[i] = A[i] * 2.0

@pk.workunit
def compute_E_from_C(i: int, C: pk.View1D[float], E: pk.View1D[float]):
    """E depends on C (which depends on A)."""
    E[i] = C[i] + 10.0

@pk.workunit
def init_B(i: int, B: pk.View1D[float], val: float):
    B[i] = val

@pk.workunit
def compute_D_from_B(i: int, B: pk.View1D[float], D: pk.View1D[float]):
    """D depends on B."""
    D[i] = B[i] * 3.0

N = 100
A = pk.View([N], dtype=float)
B = pk.View([N], dtype=float)
C = pk.View([N], dtype=float)
D = pk.View([N], dtype=float)
E = pk.View([N], dtype=float)

print("Operations in order:")
print("1. init_A: writes A")
print("2. compute_C_from_A: reads A, writes C")
print("3. compute_E_from_C: reads C, writes E")
print("4. init_B: writes B")
print("5. compute_D_from_B: reads B, writes D")
print()

# Branch 1: A -> C -> E
pk.parallel_for(N, init_A, A=A, val=5.0)
pk.parallel_for(N, compute_C_from_A, A=A, C=C)
pk.parallel_for(N, compute_E_from_C, C=C, E=E)

# Branch 2: B -> D
pk.parallel_for(N, init_B, B=B, val=7.0)
pk.parallel_for(N, compute_D_from_B, B=B, D=D)

# Check what was traced
from pykokkos import runtime_singleton
tracer = runtime_singleton.runtime.tracer

print(f"Number of operations in trace: {len(tracer.operations)}")
for i, op in enumerate(tracer.operations):
    print(f"Op {i}: {op.name}, dependencies: {[dep.name for dep in op.dependencies]}")

pk.flush()

print(f"\nResults:")
print(f"A[0] = {A[0]} (expected 5.0)")
print(f"C[0] = {C[0]} (expected 10.0)")
print(f"E[0] = {E[0]} (expected 20.0)")
print(f"B[0] = {B[0]} (expected 7.0)")
print(f"D[0] = {D[0]} (expected 21.0)")


