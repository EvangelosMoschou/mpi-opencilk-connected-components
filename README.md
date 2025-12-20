
# Parallel Connected Components — Hybrid MPI + OpenCilk

A high-performance hybrid parallel implementation of graph connected components using **MPI** (distributed memory) and **OpenCilk** (shared memory).

This project is designed to run efficiently on clusters and multi-core machines, leveraging **Profile Guided Optimization (PGO)**, **Link Time Optimization (LTO)**, and state-of-the-art algorithmic optimizations to achieve maximum performance.

---

## Overview

The application processes large undirected graphs stored in MATLAB `.mat` format. It employs a hybrid parallel strategy:
1.  **MPI (Message Passing Interface):** Distributes the graph vertices across multiple nodes/processes.
2.  **OpenCilk:** Parallelizes local computation within each process using a work-stealing scheduler.

### Prerequisites

- **Clang with OpenCilk:** For shared-memory parallelism.
- **MPI:** For distributed-memory parallelism.
- **Libraries:** `libmatio`, `libhdf5`.

For detailed building and running instructions, see [usage.md](usage.md).

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

### 7. Streaming Frontier (RAM Efficient Execution)
Replaces the standard bulk-synchronous frontier exchange with a streaming approach. Instead of buffering all messages (~GBs), we stream small chunks (~KBs) and process them asynchronously.
- **Benefit:** Reduces per-rank memory usage from **5GB+ to ~200MB** on Friendster.
- **Control:** `BFS_USE_STREAMING_FRONTIER=1`

### 8. Memory-Efficient Matrix Traversal (Distributed Loading)
Instead of loading the entire graph into RAM on a single node (which causes OOM for ~15GB+ graphs), each MPI process traverses the matrix file on the SSD and loads *only* its assigned partition.
- **Benefit:** Enables processing massive graphs (like Friendster) on commodity hardware with limited RAM.
- **Control:** `BFS_DISTRIBUTED=1`


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
| `BFS_DISTRIBUTED` | 0 | Enable Memory-Efficient Matrix Traversal (1=ON, 0=OFF). Loads graph parts from SSD. |
| `BFS_HYBRID_PACKING`| 0 | Enable Multi-threaded Communication Packing (1=ON, 0=OFF). for Hybrid. |
| `BFS_USE_STREAMING_FRONTIER`| 0 | Enable Streaming BFS (1=ON). Crucial for Friendster to avoid OOM. |

### 9. Hybrid Parallelism & Distributed Loading
- **Distributed Loading (`BFS_DISTRIBUTED=1`)**: Uses **Matrix Traversal** on SSD to load partitions directly to each rank. Crucial for massive graphs (reduces Rank 0 peak memory).
- **Hybrid Packing (`BFS_HYBRID_PACKING=1`)**: Fully parallelizes the MPI communication layer using OpenCilk threads inside each MPI rank. 
  - *Note:* Requires `CILK_NWORKERS > 1` and `mpirun --bind-to none` to be effective.

### 10. HDF5 Converter Optimizations (`mat_to_csr.c`)
The matrix conversion tool uses a highly optimized **2-Pass Chunked Streaming** approach to handle massive graphs (like Friendster) on commodity hardware:
- **Chunked HDF5 Streaming:** Reads the `.mat` file in 32MB chunks to avoid loading the full 14GB+ file into RAM.
- **Parallel Construction:** Uses `cilk_for` to parallelize degree calculation, graph construction, and row sorting.
- **On-the-fly Symmetrization:** Constructs $A \cup A^T$ directly during the build phase without ever allocating the transpose matrix, saving 50% memory.
- **Result:** Successfully converts and reorders the **65M vertex / 3.6B edge** Friendster graph on a single 32GB node in minutes.

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

---

## Future Optimizations

### Graph Compression for Low RAM Machines
For machines with limited RAM (e.g., 16GB) where OOM is a risk.
- **Proposed Solution**: Implement **Delta Compression** (e.g., Varint/VByte) for the CSR column indices.
- **Benefit**: Can reduce graph size by 2x-4x (e.g., Friendster down to ~4-6GB), fitting easily in small RAM.
- **Trade-off**: Increases CPU usage significantly as neighbors must be decoded on-the-fly during traversal. Ideal for memory-constrained environments where feasibility is more important than raw speed.

