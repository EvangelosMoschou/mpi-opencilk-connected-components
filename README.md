# Parallel Connected Components — Hybrid MPI + OpenCilk

A high-performance hybrid parallel implementation of graph connected components using **MPI** (distributed memory) and **OpenCilk** (shared memory).

This project is designed to run efficiently on clusters and multi-core machines, leveraging **Profile Guided Optimization (PGO)**, **Link Time Optimization (LTO)**, and state-of-the-art algorithmic optimizations to achieve maximum performance.

---

## Overview

The application processes large undirected graphs stored in MATLAB `.mat` format. It employs a hybrid parallel strategy:
1.  **MPI (Message Passing Interface):** Distributes the graph vertices across multiple nodes/processes.
2.  **OpenCilk:** Parallelizes local computation within each process using a work-stealing scheduler.

### Key Features

- **Hybrid Parallelism:** Combines MPI for inter-node communication and OpenCilk for intra-node parallelism.
- **Advanced Optimizations:** See the "Optimizations" section below.
- **Autotuning:** Includes a script to automatically find the optimal balance between MPI processes and Cilk workers.
- **Robust Build System:** `Makefile` handles dependency management, PGO workflows, and directory structures.

---

## Optimizations Implemented

We have implemented a suite of optimizations to maximize performance and memory efficiency:

### 1. Neighbor Sampling
Limits the number of neighbors processed during the first few rounds of BFS. This reduces work when the frontier is small.
- **Control:** `BFS_SAMPLING_ROUNDS` (Default: 0/OFF), `BFS_SAMPLING_K` (Default: 2)

### 2. Software Prefetching
Uses `__builtin_prefetch` to hide memory latency during indirect graph accesses (`col_idx` -> `labels`).
- **Control:** `BFS_PREFETCH_DIST` (Default: 16)

### 3. Zero Planting
Identifies the highest-degree vertex (hub) and assigns it the global minimum label (1) at the start. This helps the "winning" label propagate faster.
- **Control:** `BFS_ZERO_PLANTING` (Default: 1/ON)

### 4. Hub Push (Heuristic)
An extension of Zero Planting. The max-degree node pushes its label to neighbors. If those neighbors are also hubs (degree > average), they immediately push to *their* neighbors.
- **Impact:** Massive speedup for social networks (LiveJournal: 0.7s -> 0.03s).
- **Control:** `BFS_HUB_PUSH` (Default: 1/ON). *Tip: Disable for mesh graphs like kmer_V1r.*

### 5. Graph Reordering (State of the Art)
A preprocessing step in `mat_to_csr` that sorts vertices by descending degree.
- **Impact:** Drastically improves cache locality by clustering hubs together.
- **Benefit:** Permanent optimization for the dataset (cached in `.csrbin`).

### 6. Memory Optimization
Both MPI and Cilk implementations free the redundant full graph copy immediately after initialization/BFS.
- **Impact:** Reduces peak memory usage by **~2.7 GB** for `kmer_V1r`.

---

## Configuration

You can control all optimizations via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BFS_SAMPLING_ROUNDS` | 0 | Number of initial rounds to use sampling. |
| `BFS_SAMPLING_K` | 2 | Max neighbors to visit during sampling rounds. |
| `BFS_PREFETCH_DIST` | 16 | Lookahead distance for prefetching. |
| `BFS_ZERO_PLANTING` | 1 | Enable Zero Planting (1=ON, 0=OFF). |
| `BFS_HUB_PUSH` | 1 | Enable Hub Push (1=ON, 0=OFF). |
| `BFS_DISTRIBUTED` | 0 | Enable Edge-Balanced Distributed Loading (1=ON, 0=OFF). Saves RAM. |
| `BFS_HYBRID_PACKING`| 0 | Enable Multi-threaded Communication Packing (1=ON, 0=OFF). for Hybrid. |

### 7. Hybrid Parallelism & Distributed Loading
- **Distributed Loading (`BFS_DISTRIBUTED=1`)**: Instead of Rank 0 processing the whole graph, all ranks load their partition directly. Crucial for massive graphs (reduces Rank 0 peak memory).
- **Hybrid Packing (`BFS_HYBRID_PACKING=1`)**: Fully parallelizes the MPI communication layer using OpenCilk threads inside each MPI rank. 
  - *Note:* Requires `CILK_NWORKERS > 1` and `mpirun --bind-to none` to be effective.

### Usage Examples

**OpenCilk (Single Node):**
```bash
# Run with default settings
./bin/cilk/main "Mat file/kmer_V1r.mat"

# Run with Hub Push disabled (recommended for mesh graphs)
BFS_HUB_PUSH=0 ./bin/cilk/main "Mat file/kmer_V1r.mat"
```

**MPI (Distributed):**
```bash
# Run with 4 MPI processes, passing environment variables
mpirun -np 4 -x BFS_HUB_PUSH=0 ./bin/mpi/main "Mat file/kmer_V1r.mat"

# Run Hybrid (4 MPI x 4 Threads) with Max Optimization
BFS_DISTRIBUTED=1 BFS_HYBRID_PACKING=1 CILK_NWORKERS=4 \
mpirun -np 4 --bind-to none ./bin/mpi/main "Mat file/kmer_V1r.mat"
```

---

## Performance

**System:** 16-core machine (Ryzen 5800H approx equivalent)

| Dataset | Implementation | Time | Notes |
| :--- | :--- | :--- | :--- |
| **com-LiveJournal** | OpenCilk | **~0.03s** | Instant convergence due to Hub Push. |
| **com-LiveJournal** | MPI (8 procs) | **~1.3s** | Communication overhead dominates. |
| **kmer_V1r** | OpenCilk | **~9.15s** | Graph Reordering + Early Retirement. |
| **kmer_V1r** | MPI (8 procs) | **~31.28s** | Hub Push OFF + Graph Reordering. |

---

## Project Structure

```
.
├── src/                   # Core source files
│   ├── main_mpi.c         # MPI entry point (Memory Optimized)
│   ├── main.c             # Cilk-only entry point (Memory Optimized)
│   └── mat_to_csr.c       # Matrix conversion + Graph Reordering
├── bfsmpi/                # MPI Implementation
│   ├── BFSmpi.c           # Distributed BFS logic (Hub Push, Sampling, etc.)
│   └── partition.c        # Graph partitioning helpers
├── bfscilk/               # OpenCilk Implementation
│   └── BFScilk.c          # Shared memory BFS logic
├── scripts/               # Automation scripts
├── logs/                  # Performance logs and PGO profiles
├── bin/                   # Compiled executables
├── Makefile               # Build configuration
└── README.md              # This file
```

---

## License

Academic project for AUTH Parallel Computing course.
