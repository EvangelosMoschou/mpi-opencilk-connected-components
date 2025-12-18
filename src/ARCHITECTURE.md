# MAT to CSR Architecture

This document provides a high-level overview of the graph loading and preprocessing pipeline.

## Overview

The `mat_to_csr` module handles:
1. **Reading** MATLAB `.mat` files (both v5 and v7.3/HDF5 formats)
2. **Converting** sparse matrix data to CSR format
3. **Preprocessing** for optimal BFS performance
4. **Caching** preprocessed graphs as `.csrbin` files

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  .mat file  │───▶│  Read MAT   │───▶│ Symmetrize  │───▶│  Reorder    │
│ (MATLAB)    │    │  (matio/H5) │    │  A = A∪A^T  │    │ (by degree) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                   ┌─────────────┐    ┌─────────────┐          ▼
                   │  .csrbin    │◀───│  Write      │◀───┌─────────────┐
                   │  (cache)    │    │  Cache      │    │  CSR_info   │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

---

## CSR Format

**Compressed Sparse Row (CSR)** stores a sparse graph efficiently:

```c
typedef struct {
    int nrows;           // Number of vertices (N)
    long long nzcount;   // Number of edges (NNZ)
    long long *row_ptr;  // Row pointers [0..N], size: N+1
    int *col_idx;        // Column indices, size: NNZ
} CSR_info;
```

For vertex `i`:
- Neighbors are in `col_idx[row_ptr[i] .. row_ptr[i+1]-1]`
- Degree is `row_ptr[i+1] - row_ptr[i]`

---

## Preprocessing Steps

### 1. Symmetrization
Undirected graphs require A = A ∪ A^T. If edge (u→v) exists, ensure (v→u) also exists.

```c
// For each edge (r,c) in original:
//   Add (r,c) to new graph
//   Add (c,r) to new graph
// Then deduplicate each row
```

### 2. Degree-Based Reordering
Vertices are renumbered by descending degree. This improves cache locality because:
- High-degree vertices (hubs) are accessed first
- Hub vertices are contiguous in memory

```c
// Original: vertex 0 has degree 5, vertex 1 has degree 100
// Reordered: vertex 0 has degree 100, vertex 1 has degree 5
```

### 3. Row Sorting & Deduplication
Each adjacency list is sorted and duplicate edges removed:

```c
// For small rows (<32 entries): insertion sort
// For large rows: qsort
// Then linear scan to remove duplicates
```

---

## HDF5 Streaming (for Large Graphs)

MATLAB v7.3 files use HDF5 format. For graphs like Friendster (65M vertices, 1.8B edges), loading everything into RAM at once is infeasible.

**Two-Pass Chunked Streaming:**

1. **Pass 1** - Calculate degrees (stream `ir` array)
2. **Pass 2** - Build graph (stream `ir` array again)

```
┌──────────────────────────────────────────────────────────┐
│                    HDF5 File                             │
│  ┌────────┐ ┌────────────────────────────────────────┐   │
│  │   jc   │ │                   ir                   │   │
│  │ 65M×8B │ │              1.8B × 4B                 │   │
│  │= 520MB │ │              = 7.2GB                   │   │
│  └────────┘ └────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼ (stream 128MB chunks)
              ┌─────────────────────────────┐
              │   H5Streamer (32M int buf)  │
              └─────────────────────────────┘
```

---

## Binary Cache Format

Preprocessed graphs are cached to `.csrbin` files:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 | nrows (int64) |
| 8 | 8 | nzcount (int64) |
| 16 | (nrows+1)×8 | row_ptr (long long[]) |
| ... | nzcount×4 | col_idx (int[]) |

Cache is used when:
1. `.csrbin` file exists
2. `.csrbin` is newer than `.mat` file

---

## Distributed Loading (MPI)

For MPI applications, graphs can be loaded in parallel:

### Edge-Balanced Partitioning
Instead of equal vertices per rank, distribute equal edges:

```c
int *compute_balanced_partition(fname, nprocs, &total_n);
// Returns: vtx_dist[nprocs+1] where rank p owns [vtx_dist[p], vtx_dist[p+1])
```

### Partial CSR Reading
Each rank reads only its portion:

```c
CSR_info read_csr_bin_part(fname, start_row, end_row, &global_n);
```

---

## Memory Optimization

### Aligned Allocation
All arrays use 64-byte alignment for AVX-512 and cache efficiency:

```c
posix_memalign(&ptr, 64, size);
```

### Combined Build
The `build_reordered_symmetric_csr_from_arrays` function combines reordering + symmetrization in one pass, avoiding intermediate large allocations.

---

## File Structure

```
src/
├── mat_to_csr.h        # API declarations, CSR_info struct
├── mat_to_csr.c        # Implementation
└── ARCHITECTURE.md     # This file
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `matio` | Read MATLAB v5/v7 files |
| `hdf5` | Read MATLAB v7.3 (HDF5) files |
| `OpenCilk` | Parallel preprocessing (optional) |
