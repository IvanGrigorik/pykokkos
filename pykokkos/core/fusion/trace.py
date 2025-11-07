import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pykokkos.core.parsers import Parser, PyKokkosEntity
from pykokkos.interface import ExecutionPolicy, RangePolicy, Subview, ViewType

from .access_modes import AccessIndex, AccessMode, get_view_access_modes, get_view_write_indices_and_modes
from .future import Future


@dataclass
class DataDependency:
    """
    Represents data + version
    """

    name: Optional[str] # for debugging purposes
    data_id: int
    version: int

    def __hash__(self) -> int:
        return hash((self.data_id, self.version))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataDependency):
            return False

        return self.data_id == other.data_id and self.version == other.version


@dataclass
class TracerOperation:
    """
    A single operation in a trace
    """

    op_id: Optional[int] # None for fused operations
    future: Optional[Future]
    name: Optional[str]
    policy: ExecutionPolicy
    workunit: Callable[..., None]
    operation: str
    parser: Parser
    entity_name: str
    args: Dict[str, Any]
    dependencies: Set[DataDependency]
    access_indices: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]

    def __hash__(self) -> int:
        return self.op_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TracerOperation):
            return False

        return self.op_id == other.op_id

    def __repr__(self) -> str:
        if self.name is None:
            return self.workunit.__name__

        return self.name


class Tracer:
    """
    Holds traces of operations
    """

    def __init__(self) -> None:
        self.op_id: int = 0

        # This functions as an ordered set
        self.operations: Dict[TracerOperation, None] = {}

        # Map from each data object id (future or array) to the current version
        self.data_version: Dict[int, int] = {}

        # Map from data version to tracer operation
        self.data_operation: Dict[DataDependency, TracerOperation] = {}

        # Cache expensive operations that require traversing the AST
        self.access_modes_cache: Dict[Tuple[str, str], Dict[str, AccessMode]] = {}
        self.safety_cache: Dict[Tuple[str, str], Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]] = {}

    def log_operation(
        self,
        future: Optional[Future],
        name: Optional[str],
        policy: ExecutionPolicy,
        workunit: Callable[..., None],
        operation: str,
        parser: Parser,
        entity_name: str,
        **kwargs
    ) -> None:
        """
        Log the workunit and its arguments in the trace

        :param future: the future object corresponding to the output of reductions and scans
        :param name: the name of the kernel
        :param policy: the execution policy of the operation
        :param workunit: the workunit function object
        :param kwargs: the keyword arguments passed to the workunit
        :param operation: the name of the operation "for", "reduce", or "scan"
        :param parser: the parser containing the AST of the workunit
        :param entity_name: the name of the workunit entity
        """

        entity: PyKokkosEntity = parser.get_entity(entity_name)
        AST: ast.FunctionDef = entity.AST

        cache_key: Tuple[str, str] = (parser.path, entity_name)

        dependencies: Set[DataDependency]
        access_modes: Dict[str, AccessMode]
        dependencies, access_modes = self.get_data_dependencies(kwargs, AST, cache_key)

        access_indices: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]] = self.get_safety_info(kwargs, AST, cache_key)
        tracer_op = TracerOperation(self.op_id, future, name, policy, workunit, operation, parser, entity_name, dict(kwargs), dependencies, access_indices)
        self.op_id += 1

        self.update_output_data_operations(kwargs, access_modes, tracer_op, future, operation)

        self.operations[tracer_op] = None

    def get_safety_info(self, kwargs: Dict[str, Any], AST: ast.FunctionDef, cache_key: Tuple[str, str]) -> Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]:
        """
        Get the view access indices needed to check for safety

        :param kwargs: the keyword arguments passed to the workunit
        :param AST: the AST of the input workunit
        :param cache_key: used to cache the safety info extracted from the AST
        :returns: the set of data dependencies and the access modes of the views
        """

        # Map from view name to the object id
        view_args: Dict[str, int] = {}
        # Map from view name to the rank
        view_name_and_rank: Dict[str, int] = {}

        for arg, value in kwargs.items():
            if isinstance(value, ViewType):
                view_args[arg] = id(value)
                view_name_and_rank[arg] = value.rank()

        # Map from view name (str) + dimension (int) to the type of
        # access to that view's dimension
        write_indices: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]

        if cache_key in self.safety_cache:
            write_indices = self.safety_cache[cache_key]
        else:
            write_indices = get_view_write_indices_and_modes(AST, view_name_and_rank)
            self.safety_cache[cache_key] = write_indices

        # Now need to convert view name to view ID
        safety_info: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]] = {}
        for (name, dim), access_index in write_indices.items():
            view_id: int = view_args[name]
            safety_info[(view_id, dim)] = access_index

        return safety_info

    def get_operations(self, data: Union[Future, ViewType]) -> List[TracerOperation]:
        """
        Get all the operations needed to update the data of a future
        or view and remove them from the trace

        :param data: the Future or View corresponding to the value that needs to be updated
        :returns: the list of operations to be executed
        """

        if isinstance(data, Subview):
            data = data.base_view

        version: int = self.data_version.get(id(data), 0)
        dependency = DataDependency(None, id(data), version)

        if dependency not in self.data_operation:
            # The data does not depend on any prior operation
            return []

        operation: TracerOperation = self.data_operation[dependency]
        if operation not in self.operations:
            # This means that the dependency was already updated
            return []

        operations: List[TracerOperation] = [operation]
        del self.operations[operation]

        # Ideally, we would not have to do this. By adding an
        # operation to this list, its dependencies should be
        # automatically updated when the operation is executed.
        # However, since the operations are not executed in Python, we
        # cannot trigger the flush. We could also potentially iterate
        # over kwargs prior to invoking a kernel and call flush_data()
        # for all futures and views. We should implement both and
        # benchmark them
        i: int = 0
        while i < len(operations):
            current_op = operations[i]

            for dep in current_op.dependencies:
                if dep not in self.data_operation:
                    assert dep.version == 0
                    continue

                dependency_op: TracerOperation = self.data_operation[dep]
                if dependency_op not in self.operations:
                    # This means that the dependency was already updated
                    continue

                operations.append(dependency_op)
                del self.operations[dependency_op]

            i += 1

        operations.sort(key=lambda op: op.op_id)

        return operations

    def fuse(self, operations: List[TracerOperation], strategy: str) -> List[TracerOperation]:
        """
        Apply the specified fusion strategy to the given list of operations

        :param operations: the TracerOperations to be fused
        :param strategy: the fusion strategy to follow ("trace", "naive", "horizontal")
        """

        if strategy == "trace":
            return operations

        if strategy == "naive":
            return self.fuse_naive(operations)

        if strategy == "horizontal":
            return self.fuse_horizontal(operations)

        raise RuntimeError(f"Unrecognized fusion strategy '{strategy}'")

    def is_safe_to_fuse(self, current: List[TracerOperation], current_views: Set[ViewType], current_safety_info: Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]], next: TracerOperation, next_views: Set[ViewType]) -> bool:
        """
        Check whether the next operation is safe to fuse with the
        current operations

        :param current: the current list of tracer operations
        :param current_views: the combined set of views used by each operation to be fused
        :param next: the next potential operation to be added
        :param next_views: the set of views in the operation to be added
        :returns: whether the next operation can be added
        """

        common_views = current_views.intersection(next_views)
        next_safety_info = next.access_indices

        for view in common_views:
            for dim in range(view.rank()):
                key: Tuple[int, int] = (id(view), dim)

                assert key in current_safety_info
                assert key in next_safety_info

                current_access_index, current_access_mode, current_index_str = current_safety_info[key]
                next_access_index, next_access_mode, next_index_str = next_safety_info[key]

                if current_access_mode == AccessMode.Read and next_access_mode == AccessMode.Read:
                    continue

                # If the same function on the thread index is used to
                # index both views then this will not prevent fusion.
                if current_access_index == AccessIndex.TIDFunc and next_access_index == AccessIndex.TIDFunc and current_index_str == next_index_str:
                    continue

                if current_access_index.value > AccessIndex.TID.value or next_access_index.value > AccessIndex.TID.value:
                    return False

        return True

    def get_operation_views(self, operation: TracerOperation) -> Set[ViewType]:
        """
        Get all views from a TracerOperation's arguments

        :param operation: the input tracer operation
        :returns: the set of views used in that operation
        """

        views: Set[ViewType] = set()

        for key, value in operation.args.items():
            if isinstance(value, ViewType):
                views.add(value)

        return views

    def fuse_naive(self, operations: List[TracerOperation]) -> List[TracerOperation]:
        """
        Fuse a list of operations naively: combine all consecutive
        parallel fors and leave reductions and scans alone

        :param operations: the TracerOperations to be fused
        :returns: the list of TracerOperations post fusion
        """

        fused_ops: List[TracerOperation] = []
        ops_to_fuse: List[TracerOperation] = []
        ops_to_fuse_views: Set[ViewType] = set()
        fused_safety_info: Dict[Tuple[int, int], AccessIndex] = {}

        if len(operations) == 0:
            return []

        if len(operations) == 1:
            return operations

        fused_range: Optional[Tuple[int, int]]
        if isinstance(operations[-1].policy, RangePolicy):
            fused_range = (operations[-1].policy.begin, operations[-1].policy.end)
        else:
            fused_range = None

        while len(operations) > 0:
            op: TracerOperation = operations.pop()
            op_views: Set[ViewType] = self.get_operation_views(op)

            if not isinstance(op.policy, RangePolicy):
                if len(ops_to_fuse) > 0:
                    ops_to_fuse.reverse()
                    fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))
                    ops_to_fuse.clear()
                    ops_to_fuse_views.clear()
                    fused_safety_info = {}
                    fused_range = None

                # Can't fuse team policies now
                fused_ops.append(op)
                continue

            if op.operation in {"reduce", "scan"}:
                if len(ops_to_fuse) > 0:
                    ops_to_fuse.reverse()
                    fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))
                    ops_to_fuse.clear()
                    ops_to_fuse_views.clear()
                    fused_safety_info = {}
                    fused_range = None

                # Don't fuse reduce or scan now because that might
                # cause slowdowns.
                fused_ops.append(op)
                continue

            current_range: Tuple[int, int] = (op.policy.begin, op.policy.end)
            if fused_range is None:
                fused_range = current_range

            # Cannot fuse the incoming op with the current ops. Fuse
            # everything in ops_to_fuse.
            if fused_range != current_range or not self.is_safe_to_fuse(ops_to_fuse, ops_to_fuse_views, fused_safety_info, op, op_views):
                ops_to_fuse.reverse()
                fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))
                ops_to_fuse.clear()
                ops_to_fuse_views.clear()

                ops_to_fuse.append(op)
                ops_to_fuse_views.update(op_views)
                fused_safety_info = op.access_indices
                fused_range = current_range
                continue

            if op.operation == "for":
                ops_to_fuse.append(op)
                ops_to_fuse_views.update(op_views)
                fused_safety_info = self.fuse_safety_info(fused_safety_info, op.access_indices)

            elif op.operation == "reduce":
                if len(ops_to_fuse) == 0:
                    ops_to_fuse.append(op)
                    ops_to_fuse_views.update(op_views)
                    fused_safety_info = self.fuse_safety_info(fused_safety_info, op.access_indices)

                else:
                    ops_to_fuse.reverse()
                    fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))
                    ops_to_fuse.clear()
                    ops_to_fuse_views.clear()

                    ops_to_fuse.append(op)
                    ops_to_fuse_views.update(op_views)
                    fused_safety_info = op.access_indices
                    fused_range = current_range
            else:
                ops_to_fuse.reverse()
                fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))
                ops_to_fuse.clear()
                ops_to_fuse_views.clear()

                ops_to_fuse.append(op)
                ops_to_fuse_views.update(op_views)
                fused_safety_info = op.access_indices
                fused_range = current_range

        # Fuse anything left over
        if len(ops_to_fuse) > 0:
            ops_to_fuse.reverse()
            fused_ops.append(self.fuse_operations(ops_to_fuse, fused_safety_info))

        fused_ops.reverse()

        return fused_ops

    def fuse_horizontal(self, operations: List[TracerOperation]) -> List[TracerOperation]:
        """
        Fuse a list of operations horizontally: combine kernels that can
        run in parallel (no data dependencies, can have different iteration spaces)

        :param operations: the TracerOperations to be fused
        :returns: the list of TracerOperations post fusion
        """

        if len(operations) == 0:
            return []

        if len(operations) == 1:
            return operations

        # Sort operations by op_id to maintain order
        sorted_ops = sorted(operations, key=lambda op: op.op_id if op.op_id is not None else 0)

        fused_ops: List[TracerOperation] = []
        ops_to_fuse: List[TracerOperation] = []
        ops_to_fuse_views: Set[ViewType] = set()
        ops_to_fuse_dependencies: Set[DataDependency] = set()

        for op in sorted_ops:
            op_views: Set[ViewType] = self.get_operation_views(op)

            # Skip reduce and scan operations for now (can be added later)
            if op.operation in {"reduce", "scan"}:
                # Fuse any pending operations first
                if len(ops_to_fuse) > 0:
                    fused_ops.append(self.fuse_operations_horizontal(ops_to_fuse))
                    ops_to_fuse.clear()
                    ops_to_fuse_views.clear()
                    ops_to_fuse_dependencies.clear()

                fused_ops.append(op)
                continue

            # Check if this operation can be fused horizontally with current ops
            can_fuse = self.is_safe_to_fuse_horizontal(
                ops_to_fuse,
                ops_to_fuse_views,
                ops_to_fuse_dependencies,
                op,
                op_views,
                op.dependencies
            )

            if can_fuse and len(ops_to_fuse) > 0:
                # Can fuse with existing operations
                ops_to_fuse.append(op)
                ops_to_fuse_views.update(op_views)
                ops_to_fuse_dependencies.update(op.dependencies)
            else:
                # Cannot fuse - finalize current group and start new one
                if len(ops_to_fuse) > 0:
                    fused_ops.append(self.fuse_operations_horizontal(ops_to_fuse))
                    ops_to_fuse.clear()
                    ops_to_fuse_views.clear()
                    ops_to_fuse_dependencies.clear()

                # Start new fusion group
                ops_to_fuse.append(op)
                ops_to_fuse_views.update(op_views)
                ops_to_fuse_dependencies.update(op.dependencies)

        # Fuse anything left over
        if len(ops_to_fuse) > 0:
            fused_ops.append(self.fuse_operations_horizontal(ops_to_fuse))

        return fused_ops

    def is_safe_to_fuse_horizontal(
        self,
        current: List[TracerOperation],
        current_views: Set[ViewType],
        current_dependencies: Set[DataDependency],
        next: TracerOperation,
        next_views: Set[ViewType],
        next_dependencies: Set[DataDependency]
    ) -> bool:
        """
        Check whether the next operation is safe to fuse horizontally with the
        current operations. Horizontal fusion allows different iteration spaces
        but requires no data dependencies.

        :param current: the current list of tracer operations
        :param current_views: the combined set of views used by each operation to be fused
        :param current_dependencies: the combined set of data dependencies
        :param next: the next potential operation to be added
        :param next_views: the set of views in the operation to be added
        :param next_dependencies: the set of data dependencies of the next operation
        :returns: whether the next operation can be added
        """

        # Check for data dependencies between operations
        # For horizontal fusion, operations must be independent (no data dependencies)
        # Check if next operation depends on any data written by current operations
        
        # First check Future dependencies
        for next_dep in next_dependencies:
            # Check if any current operation produces this dependency
            for current_op in current:
                # Check if current_op has a future that matches next_dep
                if current_op.future is not None and id(current_op.future) == next_dep.data_id:
                    # Next operation depends on current operation's future - cannot fuse
                    return False
        
        # Check view dependencies
        for current_op in current:
            # Get views written by current_op
            current_op_views = self.get_operation_views(current_op)
            for view in current_op_views:
                # Check if current_op writes to this view
                writes_to_view = False
                for dim in range(view.rank()):
                    key = (id(view), dim)
                    if key in current_op.access_indices:
                        _, access_mode, _ = current_op.access_indices[key]
                        if access_mode in {AccessMode.Write, AccessMode.ReadWrite}:
                            writes_to_view = True
                            break
                if writes_to_view:
                    # Check if next operation depends on this view
                    if view in next_views:
                        # Check if next operation reads from this view
                        for dim in range(view.rank()):
                            key = (id(view), dim)
                            if key in next.access_indices:
                                _, access_mode, _ = next.access_indices[key]
                                if access_mode in {AccessMode.Read, AccessMode.ReadWrite}:
                                    # Next operation reads what current writes - dependency exists
                                    # For horizontal fusion, we want independent operations
                                    return False

        # Check for write conflicts on common views
        # If both operations write to the same view, they cannot run in parallel
        common_views = current_views.intersection(next_views)
        next_safety_info = next.access_indices

        for view in common_views:
            # Get write access information for this view
            view_written_in_current = False
            view_written_in_next = False

            # Check if current operations write to this view
            for current_op in current:
                current_views_op = self.get_operation_views(current_op)
                if view in current_views_op:
                    for dim in range(view.rank()):
                        key = (id(view), dim)
                        if key in current_op.access_indices:
                            _, access_mode, _ = current_op.access_indices[key]
                            if access_mode in {AccessMode.Write, AccessMode.ReadWrite}:
                                view_written_in_current = True
                                break
                    if view_written_in_current:
                        break

            # Check if next operation writes to this view
            for dim in range(view.rank()):
                key = (id(view), dim)
                if key in next_safety_info:
                    _, access_mode, _ = next_safety_info[key]
                    if access_mode in {AccessMode.Write, AccessMode.ReadWrite}:
                        view_written_in_next = True
                        break

            # If both write to the same view, cannot fuse horizontally
            if view_written_in_current and view_written_in_next:
                return False

        return True

    def fuse_operations_horizontal(self, operations: List[TracerOperation]) -> TracerOperation:
        """
        Fuse a list of TracerOperations horizontally into one.
        Unlike vertical fusion, these operations can have different policies.

        :param operations: the TracerOperations to be fused
        :returns: the fused operation
        """

        if len(operations) == 1:
            return operations[0]

        names: List[str] = []
        workunits: List[Callable[..., None]] = []

        # For horizontal fusion, we use the first operation's policy as a placeholder
        # In practice, each operation will execute with its own policy
        policy: ExecutionPolicy = operations[0].policy

        # The last operation determines the type of the fused operation
        operation: str = operations[-1].operation
        future: Optional[Future] = operations[-1].future

        parsers: List[Parser] = []
        args: Dict[str, Dict[str, Any]] = {}
        dependencies: Set[DataDependency] = set()
        safety_info: Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]] = {}

        for index, op in enumerate(operations):
            names.append(op.name if op.name is not None else op.workunit.__name__)
            workunits.append(op.workunit)
            parsers.append(op.parser)
            args[f"args_{index}"] = op.args
            dependencies.update(op.dependencies)
            safety_info = self.fuse_safety_info(safety_info, op.access_indices)

        fused_name: str
        if len(names) < 5:
            fused_name = "_".join(names)
        else:
            # Avoid long names
            fused_name = "_".join(names[:5]) + hashlib.md5(("".join(names)).encode()).hexdigest()

        # For horizontal fusion, we store the list of policies in args
        # This allows each operation to execute with its own policy
        policies: List[ExecutionPolicy] = [op.policy for op in operations]
        args["_horizontal_policies"] = policies

        return TracerOperation(None, future, fused_name, policy, workunits, operation, parsers, fused_name, args, dependencies, safety_info)

    def fuse_safety_info(self, info_0: Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]], info_1: Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]]) -> Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]]:
        """
        Fuse the safety info of two separate operations

        :param info_0: the safety info of the first op
        :param info_1: the safety info of the second op
        :returns: the fused safety info
        """

        fused_info: Dict[Tuple[int, int], Tuple[AccessIndex, AccessMode, str]] = {}
        for key, value in info_0.items():
            if key not in info_1:
                fused_info[key] = value
            else:
                other_index, other_mode, other_index_str = info_1[key]
                current_index, current_mode, current_index_str = value

                index_to_set: AccessIndex
                mode_to_set: AccessMode

                if other_index.value > current_index.value:
                    index_to_set = other_index
                else:
                    index_to_set = current_index

                if other_mode == current_mode:
                    mode_to_set = other_mode
                else:
                    mode_to_set = AccessMode.ReadWrite

                fused_info[key] = (index_to_set, mode_to_set, other_index_str)

        for key, value in info_1.items():
            # Already handled in the previous loop
            if key in fused_info:
                continue

            fused_info[key] = value

        return fused_info

    def fuse_operations(self, operations: List[TracerOperation], fused_safety_info: Dict[Tuple[int, int], AccessIndex]) -> TracerOperation:
        """
        Fuse a list of TracerOperations into one

        :param operations: the TracerOperations to be fused
        :param fused_safety_info: the fused safety information
        :returns: the fused operation
        """

        if len(operations) == 1:
            return operations[0]

        names: List[str] = []
        policy: RangePolicy = operations[0].policy
        workunits: List[Callable[..., None]] = []

        # The last operation determines the type of the fused
        # operation since it can be a reduce
        operation: str = operations[-1].operation
        future: Optional[Future] = operations[-1].future

        parsers: List[Parser] = []
        args: Dict[str, Dict[str, Any]] = {}
        dependencies: Set[DataDependency] = set()
        safety_info: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]] = {}

        for index, op in enumerate(operations):
            assert isinstance(op.policy, RangePolicy) and policy.begin == op.policy.begin and policy.end == op.policy.end

            names.append(op.name if op.name is not None else op.workunit.__name__)
            workunits.append(op.workunit)
            parsers.append(op.parser)
            args[f"args_{index}"] = op.args
            dependencies.update(op.dependencies)
            safety_info = self.fuse_safety_info(safety_info, op.access_indices)

        fused_name: str
        if len(names) < 5:
            fused_name = "_".join(names)
        else:
            # Avoid long names
            fused_name = "_".join(names[:5]) + hashlib.md5(("".join(names)).encode()).hexdigest()

        return TracerOperation(None, future, fused_name, policy, workunits, operation, parsers, fused_name, args, dependencies, fused_safety_info)

    def get_data_dependencies(self, kwargs: Dict[str, Any], AST: ast.FunctionDef, cache_key: Tuple[str, str]) -> Tuple[Set[DataDependency], Dict[str, AccessMode]]:
        """
        Get the data dependencies of an operation from its input arguments

        :param kwargs: the keyword arguments passed to the workunit
        :param AST: the AST of the input workunit
        :param cache_key: the key used to cache the results of traversing the AST
        :returns: the set of data dependencies and the access modes of the views
        """

        dependencies: Set[DataDependency] = set()
        view_args: Set[str] = set()

        # First pass to get the Future dependencies and record all the views
        for arg, value in kwargs.items():
            if isinstance(value, Subview):
                value = value.base_view

            if isinstance(value, Future):
                version: int = self.data_version.get(id(value), 0)
                dependency = DataDependency(arg, id(value), version)

                dependencies.add(dependency)

            if isinstance(value, ViewType):
                view_args.add(arg)

        access_modes: Dict[str, AccessMode]
        if cache_key in self.access_modes_cache:
            access_modes = self.access_modes_cache[cache_key]
        else:
            access_modes = get_view_access_modes(AST, view_args)
            self.access_modes_cache[cache_key] = access_modes

        # Second pass to check if the views are dependencies
        for arg, value in kwargs.items():
            if isinstance(value, Subview):
                value = value.base_view

            if isinstance(value, ViewType) and access_modes[arg] in {AccessMode.Read, AccessMode.ReadWrite}:
                version: int = self.data_version.get(id(value), 0)
                dependency = DataDependency(arg, id(value), version)

                dependencies.add(dependency)

        return dependencies, access_modes

    def update_output_data_operations(
        self,
        kwargs: Dict[str, Any],
        access_modes: Dict[str, AccessMode],
        tracer_op: TracerOperation,
        future: Optional[Future],
        operation: str
    ) -> None:
        """
        Update the data versions and operations of all data being written to

        :param kwargs: the keyword arguments passed to the workunit
        :param access_modes: how the passed views are being accessed
        :param tracer_op: the current tracer operation being logged
        :param future: the future object corresponding to the output of reductions and scans
        :param operation: the name of the operation "for", "reduce", or "scan"
        """

        for arg, value in kwargs.items():
            if isinstance(value, Subview):
                value = value.base_view

            if isinstance(value, ViewType) and access_modes[arg] in {AccessMode.Write, AccessMode.ReadWrite}:
                version: int = self.data_version.get(id(value), 0)
                self.data_version[id(value)] = version + 1
                dependency = DataDependency(arg, id(value), version + 1)

                self.data_operation[dependency] = tracer_op

        if operation in {"reduce", "scan"}:
            assert future is not None
            self.data_operation[DataDependency(None, id(future), 0)] = tracer_op
