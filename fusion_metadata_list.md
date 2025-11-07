# PyKokkos Kernel Fusion Metadata List

## Metadata Currently Collected for Vertical Fusion

1. **TracerOperation Fields**
   - op_id (Optional[int])
   - future (Optional[Future])
   - name (Optional[str])
   - policy (ExecutionPolicy)
   - workunit (Callable[..., None])
   - operation (str: "for", "reduce", or "scan")
   - parser (Parser)
   - entity_name (str)
   - args (Dict[str, Any])
   - dependencies (Set[DataDependency])
   - access_indices (Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]])

2. **DataDependency Fields**
   - name (Optional[str])
   - data_id (int)
   - version (int)

3. **View Access Information**
   - AccessMode (Read, Write, ReadWrite)
   - AccessIndex (Empty, Constant, TID, TIDFunc, Iter, All)
   - Index string representation (str)
   - View rank/dimensionality (int)
   - View object ID (int)

4. **EntityMetadata**
   - entity (Union[Callable, type, None])
   - name (str)
   - path (str)

5. **AST Information**
   - AST (ast.FunctionDef)
   - Full AST (ast.Module)
   - Source code (Tuple[List[str], int])
   - Function parameters (List[ast.arg])
   - Function body (List[ast.stmt])
   - Decorators (List[ast.Call])

6. **Type Information**
   - Inferred types (UpdatedTypes)
   - Type signature (Optional[str])
   - Parameter annotations
   - Updated decorator (UpdatedDecorator)

7. **Execution Policy**
   - Policy type (RangePolicy, MDRangePolicy, TeamPolicy, etc.)
   - Range bounds (begin, end)
   - Multi-dimensional ranges
   - Team size

8. **View Arguments**
   - View objects (ViewType instances)
   - View IDs (Set[int])
   - View names (Dict[str, int])
   - View ranks (Dict[str, int])

9. **Fusion State Tracking**
   - data_version (Dict[int, int])
   - data_operation (Dict[DataDependency, TracerOperation])
   - access_modes_cache (Dict[Tuple[str, str], Dict[str, AccessMode]])
   - safety_cache (Dict[Tuple[str, str], Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]])

10. **PyKokkos Import Information**
    - pk_import (str)
    - classtypes (List[PyKokkosEntity])

11. **Compilation Metadata**
    - types_signature (Optional[str])
    - restrict_signature (Optional[str])
    - execution_space (ExecutionSpace)

---

## Additional Metadata Needed for Horizontal Fusion

1. **Resource Utilization Metrics**
   - Compute intensity (ratio of compute to memory operations)
   - Memory intensity (number and type of memory accesses per thread)
   - Register usage (number of registers per thread)
   - Shared memory usage (amount per block)
   - Constant memory usage (amount used)
   - Local memory usage (spill memory)

2. **Thread and Block Configuration**
   - Thread block size (threads per block)
   - Grid dimensions (grid size configuration)
   - Warp occupancy (warps per block)
   - Thread occupancy (percentage of max threads per SM)
   - Block occupancy (blocks per SM)

3. **Synchronization Requirements**
   - Barrier locations (positions in kernel)
   - Synchronization type (barrier, memory fence, etc.)
   - Shared memory synchronization points
   - Atomic operations (locations and types)
   - Memory consistency requirements

4. **Instruction-Level Characteristics**
   - Instruction mix (distribution of instruction types)
   - Instruction latency (average latency)
   - Pipeline utilization (pipeline efficiency)
   - Branch divergence (likelihood and impact)
   - Memory coalescing (degree of coalescing)

5. **Execution Characteristics**
   - Kernel execution time (estimated/measured)
   - Memory bandwidth utilization (percentage)
   - Compute throughput (operations per second)
   - Power consumption (estimated)
   - Resource contention (potential conflicts)

6. **Enhanced Data Access Patterns**
   - Memory access stride (stride patterns)
   - Cache locality (cache hit/miss patterns)
   - Memory bank conflicts (shared memory conflicts)
   - Coalescing opportunities (potential for coalescing)
   - Prefetching opportunities (potential for prefetching)

7. **Enhanced Dependency Analysis**
   - Cross-kernel dependencies (dependencies between kernels)
   - Data flow graph (graph of dependencies)
   - Parallelism opportunities (potential for parallel execution)
   - Critical path (longest dependency chain)

8. **Hardware-Specific Information**
   - SM utilization (per-SM resource usage)
   - Warp scheduling efficiency
   - Memory hierarchy usage (L1, L2, shared, global)
   - Device capabilities (required GPU/device capabilities)

9. **Fusion Compatibility Metrics**
   - Resource compatibility (can kernels share resources)
   - Scheduling compatibility (can kernels be scheduled together)
   - Memory compatibility (are memory patterns compatible)
   - Performance impact estimate (estimated gain/loss)

10. **Dynamic Runtime Information**
    - Actual execution time (measured from profiling)
    - Actual resource usage (measured from profiling)
    - Profiling data (CUDA/OpenCL profiling info)
    - Performance counters (hardware counter values)

---

## References
- PyFuser: https://users.ece.utexas.edu/~gligoric/papers/AlAwarETAL25PyFuser.pdf
- ICS 2025 Paper: https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-43.pdf
- Kernel Fusion Paper: https://arxiv.org/pdf/2007.01277

