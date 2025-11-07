# PyKokkos Kernel Fusion Metadata Analysis

## Overview
PyKokkos currently supports **vertical kernel fusion** (fusing consecutive kernels with the same iteration space). This document lists the metadata currently collected for vertical fusion and identifies additional metadata needed for **horizontal fusion** (fusing kernels that execute in parallel).

---

## Metadata Currently Collected for Vertical Fusion

### 1. **TracerOperation Metadata** (from `pykokkos/core/fusion/trace.py`)
   - **op_id**: Operation identifier (Optional[int])
   - **future**: Future object for reductions/scans (Optional[Future])
   - **name**: Kernel name (Optional[str])
   - **policy**: Execution policy (ExecutionPolicy - RangePolicy, MDRangePolicy, etc.)
   - **workunit**: Workunit function object (Callable[..., None])
   - **operation**: Operation type - "for", "reduce", or "scan" (str)
   - **parser**: Parser containing the AST (Parser)
   - **entity_name**: Name of the workunit entity (str)
   - **args**: Keyword arguments passed to the workunit (Dict[str, Any])
   - **dependencies**: Set of data dependencies (Set[DataDependency])
   - **access_indices**: View access patterns per dimension (Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]])

### 2. **DataDependency Metadata** (from `pykokkos/core/fusion/trace.py`)
   - **name**: Dependency name for debugging (Optional[str])
   - **data_id**: Object ID of the data (int)
   - **version**: Version number of the data (int)

### 3. **View Access Information** (from `pykokkos/core/fusion/access_modes.py`)
   - **AccessMode**: Read, Write, or ReadWrite (AccessMode enum)
   - **AccessIndex**: Type of index used:
     - Empty (0)
     - Constant (1)
     - TID (2) - Thread ID
     - TIDFunc (3) - Function of thread ID
     - Iter (4) - Loop iterator
     - All (5) - All elements
   - **Index string**: String representation of the index expression (str)
   - **View rank**: Dimensionality of the view (int)
   - **View ID**: Object ID of the view (int)

### 4. **EntityMetadata** (from `pykokkos/core/module_setup.py`)
   - **entity**: The functor/workunit/workload object (Union[Callable, type, None])
   - **name**: Name of the entity (str)
   - **path**: File path containing the entity (str)

### 5. **AST Information** (from `pykokkos/core/compiler.py` and `pykokkos/core/fusion/fuse.py`)
   - **AST**: Abstract Syntax Tree of the workunit (ast.FunctionDef)
   - **Full AST**: Complete module AST (ast.Module)
   - **Source**: Source code lines (Tuple[List[str], int]])
   - **Function parameters**: Parameter list with annotations (List[ast.arg])
   - **Function body**: Statements in the workunit body (List[ast.stmt])
   - **Decorators**: Workunit decorators (List[ast.Call])

### 6. **Type Information** (from `pykokkos/core/type_inference/args_type_inference.py`)
   - **Inferred types**: Type annotations for parameters (UpdatedTypes)
   - **Type signature**: Hash/string identifying workunit signature (Optional[str])
   - **Parameter annotations**: Type annotations for each parameter
   - **Updated decorator**: Decorator specifiers (UpdatedDecorator)

### 7. **Execution Policy Information** (from `pykokkos/interface/execution_policy.py`)
   - **Policy type**: RangePolicy, MDRangePolicy, TeamPolicy, etc.
   - **Range bounds**: begin and end for RangePolicy (int, int)
   - **Multi-dimensional ranges**: For MDRangePolicy
   - **Team size**: For TeamPolicy

### 8. **View Arguments**
   - **View objects**: ViewType instances passed as arguments
   - **View IDs**: Object IDs to track view aliasing (Set[int])
   - **View names**: Parameter names for views (Dict[str, int])
   - **View ranks**: Dimensionality of each view (Dict[str, int])

### 9. **Fusion State Tracking** (from `pykokkos/core/fusion/trace.py`)
   - **data_version**: Map from data object ID to current version (Dict[int, int])
   - **data_operation**: Map from DataDependency to TracerOperation (Dict[DataDependency, TracerOperation])
   - **access_modes_cache**: Cached access modes per entity (Dict[Tuple[str, str], Dict[str, AccessMode]])
   - **safety_cache**: Cached safety information per entity (Dict[Tuple[str, str], Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]])

### 10. **PyKokkos Import Information**
   - **pk_import**: PyKokkos import alias used (str)
   - **classtypes**: List of classtypes used by the entity (List[PyKokkosEntity])

### 11. **Compilation Metadata**
   - **types_signature**: Hash identifying type signature (Optional[str])
   - **restrict_signature**: Hash for restricted views (Optional[str])
   - **execution_space**: Execution space (ExecutionSpace)

---

## Additional Metadata Needed for Horizontal Fusion

### 1. **Resource Utilization Metrics**
   - **Compute intensity**: Ratio of compute operations to memory operations
   - **Memory intensity**: Number and type of memory accesses per thread
   - **Register usage**: Number of registers used per thread
   - **Shared memory usage**: Amount of shared memory used per block
   - **Constant memory usage**: Amount of constant memory used
   - **Local memory usage**: Amount of local memory (spill) used

### 2. **Thread and Block Configuration**
   - **Thread block size**: Number of threads per block (int)
   - **Grid dimensions**: Grid size configuration (Tuple[int, ...])
   - **Warp occupancy**: Number of warps per block
   - **Thread occupancy**: Percentage of maximum threads per SM
   - **Block occupancy**: Number of blocks that can run concurrently per SM

### 3. **Synchronization Requirements**
   - **Barrier locations**: Positions of synchronization barriers in the kernel
   - **Synchronization type**: Type of synchronization (barrier, memory fence, etc.)
   - **Shared memory synchronization**: Points where shared memory is synchronized
   - **Atomic operations**: Locations and types of atomic operations
   - **Memory consistency requirements**: Memory ordering constraints

### 4. **Instruction-Level Characteristics**
   - **Instruction mix**: Distribution of instruction types (arithmetic, memory, control)
   - **Instruction latency**: Average latency of instructions
   - **Pipeline utilization**: How well the pipeline is utilized
   - **Branch divergence**: Likelihood and impact of branch divergence
   - **Memory coalescing**: Degree of memory access coalescing

### 5. **Execution Characteristics**
   - **Kernel execution time**: Estimated or measured execution time
   - **Memory bandwidth utilization**: Percentage of memory bandwidth used
   - **Compute throughput**: Operations per second
   - **Power consumption**: Estimated power consumption
   - **Resource contention**: Potential for resource conflicts with other kernels

### 6. **Data Access Patterns (Enhanced)**
   - **Memory access stride**: Stride patterns for memory accesses
   - **Cache locality**: Cache hit/miss patterns
   - **Memory bank conflicts**: Shared memory bank conflicts
   - **Coalescing opportunities**: Potential for memory access coalescing
   - **Prefetching opportunities**: Potential for data prefetching

### 7. **Dependency Analysis (Enhanced)**
   - **Cross-kernel dependencies**: Dependencies between different kernels
   - **Data flow graph**: Graph of data dependencies across kernels
   - **Parallelism opportunities**: Potential for parallel execution
   - **Critical path**: Longest dependency chain

### 8. **Hardware-Specific Information**
   - **SM (Streaming Multiprocessor) utilization**: Per-SM resource usage
   - **Warp scheduling efficiency**: How efficiently warps are scheduled
   - **Memory hierarchy usage**: Usage of L1, L2, shared, global memory
   - **Device capabilities**: GPU/device-specific capabilities required

### 9. **Fusion Compatibility Metrics**
   - **Resource compatibility**: Whether kernels can share resources
   - **Scheduling compatibility**: Whether kernels can be scheduled together
   - **Memory compatibility**: Whether memory access patterns are compatible
   - **Performance impact estimate**: Estimated performance gain/loss from fusion

### 10. **Dynamic Runtime Information**
   - **Actual execution time**: Measured execution time from profiling
   - **Actual resource usage**: Measured resource usage from profiling
   - **Profiling data**: CUDA/OpenCL profiling information
   - **Performance counters**: Hardware performance counter values

---

## Summary

**Current Vertical Fusion Metadata (11 categories):**
- TracerOperation metadata
- DataDependency metadata
- View access information
- EntityMetadata
- AST information
- Type information
- Execution policy information
- View arguments
- Fusion state tracking
- PyKokkos import information
- Compilation metadata

**Additional Metadata for Horizontal Fusion (10 categories):**
- Resource utilization metrics
- Thread and block configuration
- Synchronization requirements
- Instruction-level characteristics
- Execution characteristics
- Enhanced data access patterns
- Enhanced dependency analysis
- Hardware-specific information
- Fusion compatibility metrics
- Dynamic runtime information

---

## References
- PyFuser: https://users.ece.utexas.edu/~gligoric/papers/AlAwarETAL25PyFuser.pdf
- ICS 2025 Paper: https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-43.pdf
- Kernel Fusion Paper: https://arxiv.org/pdf/2007.01277

