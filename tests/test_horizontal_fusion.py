import os
import unittest
import pykokkos as pk


# Workunits for horizontal fusion testing
@pk.workunit
def init_view_a(i: int, view_a: pk.View1D[int], value: int):
    view_a[i] = value

@pk.workunit
def init_view_b(i: int, view_b: pk.View1D[int], value: int):
    view_b[i] = value

@pk.workunit
def init_view_c(i: int, view_c: pk.View1D[int], value: int):
    view_c[i] = value

@pk.workunit
def compute_independent(i: int, view_a: pk.View1D[int], view_b: pk.View1D[int], result: pk.View1D[int]):
    result[i] = view_a[i] + view_b[i]

@pk.workunit
def init_view_dependent(i: int, view_a: pk.View1D[int], view_b: pk.View1D[int]):
    # This depends on view_a, so it shouldn't fuse horizontally with init_view_a
    view_b[i] = view_a[i] * 2


class TestHorizontalFusion(unittest.TestCase):
    def setUp(self):
        self.iterations: int = 10
        # Set horizontal fusion strategy
        os.environ["PK_FUSION"] = "horizontal"

    def tearDown(self):
        # Clean up environment variable
        if "PK_FUSION" in os.environ:
            del os.environ["PK_FUSION"]
        # Flush trace to execute any pending operations
        pk.flush()

    def test_independent_kernels_different_sizes(self):
        """Test that kernels with different iteration spaces can be fused horizontally"""
        size_a = 5
        size_b = 10
        size_c = 7
        
        v_a = pk.View((size_a,), int)
        v_b = pk.View((size_b,), int)
        v_c = pk.View((size_c,), int)
        
        # These kernels are independent and have different sizes
        # They should be able to fuse horizontally
        pk.parallel_for(size_a, init_view_a, view_a=v_a, value=1)
        pk.parallel_for(size_b, init_view_b, view_b=v_b, value=2)
        pk.parallel_for(size_c, init_view_c, view_c=v_c, value=3)
        
        # Flush to execute fused operations
        pk.flush()
        
        # Verify results
        self.assertEqual(v_a[0], 1)
        self.assertEqual(v_b[0], 2)
        self.assertEqual(v_c[0], 3)

    def test_independent_kernels_same_size(self):
        """Test that independent kernels with same size can be fused horizontally"""
        size = 10
        
        v_a = pk.View((size,), int)
        v_b = pk.View((size,), int)
        v_c = pk.View((size,), int)
        
        # These kernels are independent
        pk.parallel_for(size, init_view_a, view_a=v_a, value=10)
        pk.parallel_for(size, init_view_b, view_b=v_b, value=20)
        pk.parallel_for(size, init_view_c, view_c=v_c, value=30)
        
        # Flush to execute fused operations
        pk.flush()
        
        # Verify results
        self.assertEqual(v_a[0], 10)
        self.assertEqual(v_b[0], 20)
        self.assertEqual(v_c[0], 30)
        self.assertEqual(v_a[size-1], 10)
        self.assertEqual(v_b[size-1], 20)
        self.assertEqual(v_c[size-1], 30)

    def test_dependent_kernels_not_fused(self):
        """Test that dependent kernels are not fused horizontally"""
        size = 10
        
        v_a = pk.View((size,), int)
        v_b = pk.View((size,), int)
        
        # First kernel initializes v_a
        pk.parallel_for(size, init_view_a, view_a=v_a, value=5)
        
        # Second kernel depends on v_a (reads it), so they shouldn't fuse
        pk.parallel_for(size, init_view_dependent, view_a=v_a, view_b=v_b)
        
        # Flush to execute operations
        pk.flush()
        
        # Verify results
        self.assertEqual(v_a[0], 5)
        self.assertEqual(v_b[0], 10)  # v_b[i] = v_a[i] * 2

    def test_mixed_independent_and_dependent(self):
        """Test a mix of independent and dependent kernels"""
        size = 10
        
        v_a = pk.View((size,), int)
        v_b = pk.View((size,), int)
        v_c = pk.View((size,), int)
        v_d = pk.View((size,), int)
        
        # Independent kernels - should fuse horizontally
        pk.parallel_for(size, init_view_a, view_a=v_a, value=1)
        pk.parallel_for(size, init_view_b, view_b=v_b, value=2)
        
        # This depends on v_a, so it won't fuse with the above
        pk.parallel_for(size, init_view_dependent, view_a=v_a, view_b=v_c)
        
        # This is independent of the above
        pk.parallel_for(size, init_view_c, view_c=v_d, value=4)
        
        # Flush to execute operations
        pk.flush()
        
        # Verify results
        self.assertEqual(v_a[0], 1)
        self.assertEqual(v_b[0], 2)
        self.assertEqual(v_c[0], 2)  # v_c[i] = v_a[i] * 2
        self.assertEqual(v_d[0], 4)

    def test_write_conflict_not_fused(self):
        """Test that kernels writing to the same view are not fused horizontally"""
        size = 10
        
        v_a = pk.View((size,), int)
        
        # Both kernels write to v_a - should not fuse horizontally
        pk.parallel_for(size, init_view_a, view_a=v_a, value=1)
        pk.parallel_for(size, init_view_a, view_a=v_a, value=2)
        
        # Flush to execute operations
        pk.flush()
        
        # The last write should win
        self.assertEqual(v_a[0], 2)


if __name__ == '__main__':
    unittest.main()

