"""
Unit tests for horizontal fusion logic without requiring full PyKokkos runtime
"""
import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykokkos.core.fusion.trace import Tracer, TracerOperation, DataDependency
from pykokkos.core.fusion.access_modes import AccessIndex, AccessMode
from pykokkos.interface import RangePolicy, ExecutionSpace


class TestHorizontalFusionLogic(unittest.TestCase):
    """Test horizontal fusion logic directly"""

    def setUp(self):
        self.tracer = Tracer()

    def test_fuse_horizontal_empty_list(self):
        """Test fuse_horizontal with empty list"""
        result = self.tracer.fuse_horizontal([])
        self.assertEqual(result, [])

    def test_fuse_horizontal_single_operation(self):
        """Test fuse_horizontal with single operation"""
        # Create a mock operation
        op = Mock(spec=TracerOperation)
        op.op_id = 0
        op.operation = "for"
        op.dependencies = set()
        op.access_indices = {}
        
        result = self.tracer.fuse_horizontal([op])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], op)

    def test_fuse_horizontal_independent_operations(self):
        """Test that independent operations can be fused horizontally"""
        # Create two independent operations
        op1 = Mock(spec=TracerOperation)
        op1.op_id = 0
        op1.operation = "for"
        op1.dependencies = set()
        op1.access_indices = {}
        op1.name = "op1"
        op1.workunit = Mock()
        op1.workunit.__name__ = "workunit1"
        op1.policy = RangePolicy(ExecutionSpace.Default, 0, 10)
        op1.parser = Mock()
        op1.entity_name = "op1"
        op1.args = {"view_a": Mock()}
        op1.future = None
        
        op2 = Mock(spec=TracerOperation)
        op2.op_id = 1
        op2.operation = "for"
        op2.dependencies = set()
        op2.access_indices = {}
        op2.name = "op2"
        op2.workunit = Mock()
        op2.workunit.__name__ = "workunit2"
        op2.policy = RangePolicy(ExecutionSpace.Default, 0, 20)
        op2.parser = Mock()
        op2.entity_name = "op2"
        op2.args = {"view_b": Mock()}
        op2.future = None
        
        # Mock get_operation_views to return different views
        def get_views_op1(op):
            if op == op1:
                return {op1.args["view_a"]}
            return {op2.args["view_b"]}
        
        self.tracer.get_operation_views = get_views_op1
        
        # Mock is_safe_to_fuse_horizontal to return True (independent)
        original_is_safe = self.tracer.is_safe_to_fuse_horizontal
        self.tracer.is_safe_to_fuse_horizontal = lambda *args: True
        
        result = self.tracer.fuse_horizontal([op1, op2])
        
        # Should fuse into one operation
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0], TracerOperation))
        
        # Restore original method
        self.tracer.is_safe_to_fuse_horizontal = original_is_safe

    def test_is_safe_to_fuse_horizontal_no_common_views(self):
        """Test is_safe_to_fuse_horizontal with no common views"""
        # Create operations with no common views
        op1 = Mock(spec=TracerOperation)
        op1.dependencies = set()
        op1.access_indices = {}
        op1.future = None
        
        op2 = Mock(spec=TracerOperation)
        op2.dependencies = set()
        op2.access_indices = {}
        op2.future = None
        
        view1 = Mock()
        view1.rank.return_value = 1
        view2 = Mock()
        view2.rank.return_value = 1
        
        # Mock get_operation_views
        def get_views(op):
            if op == op1:
                return {view1}
            return {view2}
        
        self.tracer.get_operation_views = get_views
        
        result = self.tracer.is_safe_to_fuse_horizontal(
            [op1], {view1}, set(), op2, {view2}, set()
        )
        
        # Should be safe to fuse (no common views)
        self.assertTrue(result)

    def test_is_safe_to_fuse_horizontal_write_conflict(self):
        """Test is_safe_to_fuse_horizontal detects write conflicts"""
        # Create operations that both write to the same view
        op1 = Mock(spec=TracerOperation)
        op1.dependencies = set()
        op1.future = None
        
        op2 = Mock(spec=TracerOperation)
        op2.dependencies = set()
        op2.future = None
        
        view = Mock()
        view.rank.return_value = 1
        
        # Both operations write to the same view
        op1.access_indices = {(id(view), 0): (AccessIndex.TID, AccessMode.Write, "i")}
        op2.access_indices = {(id(view), 0): (AccessIndex.TID, AccessMode.Write, "i")}
        
        # Mock get_operation_views
        def get_views(op):
            return {view}
        
        self.tracer.get_operation_views = get_views
        
        result = self.tracer.is_safe_to_fuse_horizontal(
            [op1], {view}, set(), op2, {view}, set()
        )
        
        # Should NOT be safe to fuse (write conflict)
        self.assertFalse(result)

    def test_is_safe_to_fuse_horizontal_future_dependency(self):
        """Test is_safe_to_fuse_horizontal detects Future dependencies"""
        # Create operations where op2 depends on op1's future
        future = Mock()
        future_id = id(future)
        
        op1 = Mock(spec=TracerOperation)
        op1.dependencies = set()
        op1.access_indices = {}
        op1.future = future
        
        op2 = Mock(spec=TracerOperation)
        op2.dependencies = {DataDependency(None, future_id, 0)}
        op2.access_indices = {}
        op2.future = None
        
        view1 = Mock()
        view1.rank.return_value = 1
        view2 = Mock()
        view2.rank.return_value = 1
        
        # Mock get_operation_views
        def get_views(op):
            if op == op1:
                return {view1}
            return {view2}
        
        self.tracer.get_operation_views = get_views
        
        result = self.tracer.is_safe_to_fuse_horizontal(
            [op1], {view1}, set(), op2, {view2}, op2.dependencies
        )
        
        # Should NOT be safe to fuse (Future dependency)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

