# OpenCilk BFS Architecture

This document provides a high-level overview of the shared-memory Connected Components (CC) implementation using OpenCilk.

## Algorithm Overview

The implementation provides **two algorithms** with an **adaptive dispatcher**:

```
                   ┌────────────────────┐
                   │    coloring_cc()   │
                   │ (Adaptive Dispatch)│
                   └────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌──────────────────┐        ┌──────────────────┐
   │ coloring_cc_bfs  │        │ coloring_cc_lp   │
   │  (Frontier BFS)  │        │(Label Propagation)│
   │ AvgDeg ≤ 4.0     │        │ AvgDeg > 4.0     │
   └──────────────────┘        └──────────────────┘
```

### Frontier-Based BFS (`coloring_cc_bfs`)
- **Best for**: Sparse graphs with large diameter (e.g., k-mer graphs)
- **Algorithm**: Classic BFS with frontier expansion
- **Parallelism**: `cilk_for` over frontier vertices
- **Iterations**: O(diameter) per component

### Label Propagation (`coloring_cc_lp`)
- **Best for**: Dense graphs with small diameter (e.g., social networks)
- **Algorithm**: Iteratively propagate minimum labels until convergence
- **Parallelism**: `cilk_for` over all vertices
- **Iterations**: O(diameter) total

---

## Key Data Structures

### CSR Graph Format (from `mat_to_csr.h`)
```c
typedef struct {
    long long nzcount;     // Number of edges
    int nrows;             // Number of vertices
    long long *row_ptr;    // Row pointers (size: n+1)
    int *col_idx;          // Column indices (size: nzcount)
} CSR_info;
```

### Labels Array
- Size: `n` integers
- Values: 1-based component IDs (1..n)
- Convention: `labels[v] = component_id` where smaller ID = earlier discovered

---

## OpenCilk Features Used

### Parallel Loops
```c
cilk_for(int v = 0; v < n; v++) {
    // Process vertex v in parallel
}
```

### Custom Reducer
For tracking if any label changed during an iteration:
```c
void bool_or_identity(void *view) { *(int *)view = 0; }
void bool_or_reduce(void *left, void *right) {
    *(int *)left = *(int *)left | *(int *)right;
}

int cilk_reducer(bool_or_identity, bool_or_reduce) changed = 1;
```

### Atomic Operations
Race-free label updates:
```c
__atomic_compare_exchange_n(&labels[v], &expected, new_val, ...);
__atomic_fetch_min(&labels[v], new_val, __ATOMIC_SEQ_CST);
__atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
```

### Cilkscale Instrumentation
Performance analysis via work-span measurements:
```c
wsp_t start = wsp_getworkspan();
// ... work ...
wsp_t end = wsp_getworkspan();
wsp_dump(wsp_sub(end, start), "BFS-OpenCilk-LP");
```

---

## Optimizations

### Zero Planting
Find the highest-degree vertex and give it label 1. This accelerates convergence because the minimum label (1) will propagate outward from the most connected vertex.

```c
// Find max degree vertex
for (int i = 0; i < n; i++) {
    if (row_ptr[i+1] - row_ptr[i] > max_deg) { ... }
}
// Swap with vertex 0 to give it label 1
```

### Hub Push
During zero planting, if a neighbor is also a hub (degree > average), recursively push the label to its neighbors. This creates a "ripple effect" for faster convergence.

### Path Compression
Follow label chains to find the true minimum:
```c
while (best > labels[best - 1]) {
    best = labels[best - 1];
}
```

### Sampling Rounds
In early iterations, only examine `sampling_k` neighbors per vertex:
```c
if (iterations <= sampling_rounds) {
    limit = start_edge + sampling_k;
}
```

### Software Prefetching
Two-stage prefetch to hide memory latency:
```c
// Stage 1: Prefetch next neighbor index
__builtin_prefetch(&col_idx[e + prefetch_dist * 2], 0, 3);
// Stage 2: Prefetch label for upcoming neighbor
__builtin_prefetch(&labels[col_idx[e + prefetch_dist]], 0, 3);
```

### Memory Alignment
Inform compiler about 64-byte aligned arrays for better vectorization:
```c
row_ptr = __builtin_assume_aligned(row_ptr, 64);
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BFS_SAMPLING_ROUNDS` | 0 | Sampling rounds at start |
| `BFS_SAMPLING_K` | 2 | Neighbors per vertex during sampling |
| `BFS_PREFETCH_DIST` | 16 | Prefetch distance |
| `BFS_ZERO_PLANTING` | 1 | Enable zero planting |
| `BFS_HUB_PUSH` | 1 | Enable hub push optimization |

---

## File Structure

```
bfscilk/
├── bfs.h            # API declarations and constants
├── BFScilk.c        # Algorithm implementations
└── ARCHITECTURE.md  # This file
```

---

## Performance Characteristics

| Algorithm | Work | Span | Parallelism |
|-----------|------|------|-------------|
| Frontier BFS | O(V + E) | O(D × max_deg) | O(frontier_size) |
| Label Prop | O(D × E) | O(D × V/P) | O(V) |

Where:
- V = vertices, E = edges, D = diameter, P = processors
