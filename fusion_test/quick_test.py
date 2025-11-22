#!/usr/bin/env python3
"""
Quick test to verify fusion is working correctly.

This is a minimal example to quickly check if the fusion strategies
are functioning as expected.
"""

import os
import sys

# Test with horizontal fusion
os.environ["PK_FUSION"] = "horizontal"

import pykokkos as pk
import numpy as np


@pk.workunit
def set_A(i: int, A: pk.View1D[float]):
    A[i] = 1.0


@pk.workunit
def set_B(i: int, B: pk.View1D[float]):
    B[i] = 2.0


@pk.workunit
def set_C(i: int, C: pk.View1D[float]):
    C[i] = 3.0


@pk.workunit
def add_AB(i: int, A: pk.View1D[float], B: pk.View1D[float], C: pk.View1D[float]):
    C[i] = A[i] + B[i]


def test_horizontal():
    """Test horizontal fusion with independent operations."""
    print("\n" + "="*70)
    print("TEST: Horizontal Fusion")
    print("="*70)
    
    N = 100
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    C = pk.View([N], dtype=float)
    
    # Three independent operations - should fuse horizontally
    pk.parallel_for(N, set_A, A=A)
    pk.parallel_for(N, set_B, B=B)
    pk.parallel_for(N, set_C, C=C)
    
    # Force execution
    assert A[0] == 1.0, f"Expected A[0]=1.0, got {A[0]}"
    assert B[0] == 2.0, f"Expected B[0]=2.0, got {B[0]}"
    assert C[0] == 3.0, f"Expected C[0]=3.0, got {C[0]}"
    
    print("[OK] Three independent initializations executed correctly")
    print(f"  A[0]={A[0]}, B[0]={B[0]}, C[0]={C[0]}")
    
    return True


def test_vertical():
    """Test vertical fusion with dependent operations."""
    print("\n" + "="*70)
    print("TEST: Vertical Fusion")
    print("="*70)
    
    # Change to vertical fusion
    os.environ["PK_FUSION"] = "naive"
    
    N = 100
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    Result = pk.View([N], dtype=float)
    
    # Sequential operations
    pk.parallel_for(N, set_A, A=A)
    pk.parallel_for(N, set_B, B=B)
    pk.parallel_for(N, add_AB, A=A, B=B, C=Result)
    
    # Force execution
    expected = 1.0 + 2.0
    assert Result[0] == expected, f"Expected Result[0]={expected}, got {Result[0]}"
    
    print("[OK] Sequential operations executed correctly")
    print(f"  Result[0] = {Result[0]} (expected: {expected})")
    
    return True


def test_no_fusion():
    """Test with no fusion for comparison."""
    print("\n" + "="*70)
    print("TEST: No Fusion (Baseline)")
    print("="*70)
    
    # Change to trace (no fusion)
    os.environ["PK_FUSION"] = "trace"
    
    N = 100
    A = pk.View([N], dtype=float)
    B = pk.View([N], dtype=float)
    
    pk.parallel_for(N, set_A, A=A)
    pk.parallel_for(N, set_B, B=B)
    
    # Force execution
    assert A[0] == 1.0, f"Expected A[0]=1.0, got {A[0]}"
    assert B[0] == 2.0, f"Expected B[0]=2.0, got {B[0]}"
    
    print("[OK] Operations executed correctly without fusion")
    print(f"  A[0]={A[0]}, B[0]={B[0]}")
    
    return True


def main():
    print("\n" + "="*70)
    print("QUICK FUSION TEST")
    print("="*70)
    print("\nThis script quickly tests that fusion is working correctly.")
    
    tests = [
        ("Horizontal Fusion", test_horizontal),
        ("Vertical Fusion", test_vertical),
        ("No Fusion", test_no_fusion),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"[X] Test failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, success, error in results:
        if success:
            print(f"[OK] {name}: PASSED")
        else:
            print(f"[X] {name}: FAILED")
            if error:
                print(f"  Error: {error}")
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n All tests passed!\n")
        return 0
    else:
        print("\n[WARNING] Some tests failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())


