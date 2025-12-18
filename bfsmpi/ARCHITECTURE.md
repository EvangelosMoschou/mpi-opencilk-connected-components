# MPI BFS Architecture

This document provides a high-level overview of the distributed Connected Components (CC) implementation using MPI + OpenCilk.

## Algorithm Overview

The implementation provides **three algorithms** for finding connected components in distributed graphs:

```
               ┌─────────────────────────────────────┐
               │         mpi_coloring_cc()           │
               │        (Adaptive Dispatcher)        │
               └─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│ Label Propag. │   │  Frontier BFS │   │ Streaming Frontier│
│   (Default)   │   │  BFS_USE_     │   │ BFS_USE_STREAMING_│
│               │   │  FRONTIER=1   │   │ FRONTIER=1        │
└───────────────┘   └───────────────┘   └───────────────────┘
```

### 1. Label Propagation (Default)
- **Best for**: High-density graphs (social networks) with small diameter
- **Iterations**: O(diameter) but with dense communication
- **Key optimization**: Bitset-based frontier tracking

### 2. Frontier BFS (`BFS_USE_FRONTIER=1`)
- **Best for**: Low-density graphs, multiple components
- **Iterations**: O(diameter) per component
- **Key optimization**: Only communicate frontier vertices

### 3. Streaming Frontier BFS (`BFS_USE_STREAMING_FRONTIER=1`)
- **Best for**: Large-scale graphs with communication bottlenecks
- **Key feature**: Overlapped computation and communication
- **Termination**: Distributed send/recv counting

---

## Data Distribution

### Graph Partitioning
- Vertices are distributed across MPI ranks using **edge-balanced partitioning**
- Each rank owns a contiguous range of vertices: `[vtx_dist[rank], vtx_dist[rank+1])`
- The `get_owner()` function uses **binary search** on `vtx_dist` (O(log P))

```
Rank 0:  vertices [0, 1000)
Rank 1:  vertices [1000, 2500)    ← More vertices = more edges
Rank 2:  vertices [2500, 3200)
...
```

### Edge Partitioning (Internal vs Remote)
During `setup_communication()`, each vertex's adjacency list is **partitioned in-place**:

```
local_col_idx[row_ptr[i] ... split_indices[i]-1]  → Internal edges (local neighbors)
local_col_idx[split_indices[i] ... row_ptr[i+1]-1] → Remote edges (cross-rank neighbors)
```

This allows skipping remote edges during local propagation phases.

---

## Key Data Structures

### `MPI_Graph` (defined in `bfs_mpi.h`)
Core graph structure containing:
- `local_row_ptr`, `local_col_idx`: CSR adjacency for local vertices
- `local_labels`: Component label for each local vertex
- `vtx_dist`: Global vertex distribution array
- `split_indices`: Partition point for internal/remote edges

### `CommInfo` (static, internal)
Communication buffers and metadata:
- `boundary_indices`: Vertices with remote neighbors
- `send_buf`, `recv_buf`: Message buffers
- `send_counts`, `recv_counts`: Per-rank message sizes

### `StreamBuffer` (for Streaming BFS)
Double-buffered async communication:
- `primary`: Currently filling
- `secondary`: Currently sending
- `req`: MPI_Request for async ops

---

## Communication Patterns

### Label Propagation
```
1. Local Propagation (parallel via cilk_for)
2. Pack messages for remote neighbors
3. MPI_Alltoall → exchange counts
4. MPI_Irecv + MPI_Isend → non-blocking exchange
5. MPI_Waitall
6. Process received labels
7. Repeat until global convergence
```

### Streaming BFS
```
1. Process local frontier (parallel)
2. For each remote neighbor:
   - Buffer in TLS (thread-local)
   - Flush to global buffer when threshold reached
   - Send via MPI_Isend when buffer fills
3. Periodically poll (MPI_Iprobe) for incoming messages
4. Termination: global_sent == global_recv
```

---

## Optimizations

### Zero Planting
Find the highest-degree vertex globally and use it as the starting point (label 1). This accelerates convergence for single-component graphs.

### Path Compression
During propagation, follow label chains to find the minimum reachable label:
```c
while (val > my_offset && val <= my_offset + local_n) {
    int parent = labels[val - my_offset - 1];
    if (parent < val) val = parent;
    else break;
}
```

### Sampling Rounds
During early iterations, limit neighbor exploration to `sampling_k` neighbors per vertex. This reduces communication in dense graphs.

### Software Prefetching
Two-stage prefetch for memory access patterns:
1. Prefetch neighbor index from `col_idx`
2. Prefetch label from `labels`

### Thread-Local Buffers (Hybrid Packing)
When using multiple OpenCilk workers, each thread packs messages into its own buffer to avoid contention, then merges into the global send buffer.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BFS_USE_FRONTIER` | 0 | Use Frontier BFS instead of Label Propagation |
| `BFS_USE_STREAMING_FRONTIER` | 0 | Use Streaming Frontier BFS |
| `BFS_SAMPLING_ROUNDS` | 0 | Number of initial sampling rounds |
| `BFS_SAMPLING_K` | 2 | Neighbors to sample per vertex |
| `BFS_PREFETCH_DIST` | 16 | Prefetch distance for memory operations |
| `BFS_ZERO_PLANTING` | 1 | Enable zero planting optimization |
| `BFS_HYBRID_PACKING` | 0 | Enable thread-local buffer packing |

---

## File Structure

```
bfsmpi/
├── bfs_mpi.h         # Types, macros, and API declarations
├── BFSmpi.c          # Algorithm implementations
├── partition.c       # Graph initialization and distribution
└── ARCHITECTURE.md   # This file
```
