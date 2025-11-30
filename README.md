<<<<<<< HEAD
# PkD_Project2
=======
# Parallel Connected Components — Hybrid MPI + OpenCilk

A high-performance hybrid parallel implementation of graph connected components using **MPI** (distributed memory) and **OpenCilk** (shared memory).

This project is designed to run efficiently on clusters and multi-core machines, leveraging **Profile Guided Optimization (PGO)** and **Link Time Optimization (LTO)** to achieve maximum performance.

---

## Overview

The application processes large undirected graphs stored in MATLAB `.mat` format. It employs a hybrid parallel strategy:
1.  **MPI (Message Passing Interface):** Distributes the graph vertices across multiple nodes/processes.
2.  **OpenCilk:** Parallelizes local computation within each process using a work-stealing scheduler.

### Key Features

- **Hybrid Parallelism:** Combines MPI for inter-node communication and OpenCilk for intra-node parallelism.
- **Advanced Optimizations:**
    - **O(1) Owner Lookup:** Efficiently determines vertex ownership without binary search.
    - **Memory Optimization:** Uses `int8_t` for frontier tracking to minimize memory bandwidth.
    - **PGO & LTO:** Automated build targets for Profile Guided Optimization and Link Time Optimization.
- **Autotuning:** Includes a script to automatically find the optimal balance between MPI processes and Cilk workers for the host machine.
- **Robust Build System:** `Makefile` handles dependency management, PGO workflows, and directory structures.

---

## Project Structure

```
.
├── src/                   # Core source files
│   ├── main_mpi.c         # MPI entry point
│   ├── main.c             # Cilk-only entry point
│   └── mat_to_csr.c       # Matrix conversion utilities
├── bfsmpi/                # MPI Implementation
│   ├── BFSmpi.c           # Distributed BFS logic
│   └── partition.c        # Graph partitioning helpers
├── bfscilk/               # OpenCilk Implementation
├── scripts/               # Automation scripts
│   ├── bench_mpi.sh       # Basic benchmarking script
│   └── autotune_mpi.sh    # Advanced autotuner
├── logs/                  # Performance logs and PGO profiles
├── bin/                   # Compiled executables
├── Makefile               # Build configuration
└── README.md              # This file
```

---

## Quick Start

### 1. Prerequisites
- **Clang/LLVM** with OpenCilk support.
- **OpenMPI** (or compatible MPI implementation).
- **libmatio** (for reading `.mat` files).

### 2. Build & Run (Fastest Method)

The easiest way to get the best performance is to use the autotuner, which detects your hardware, generates a PGO profile, and finds the best configuration.

```bash
make autotune
```

This will:
1.  Detect available cores.
2.  Test various `MPI Process x Cilk Worker` combinations.
3.  Generate a PGO profile for each.
4.  Save the fastest binary to `./main_best`.

### 3. Manual Build

To build the standard MPI version (using an existing profile if available):

```bash
make mpi
```

To manually regenerate the PGO profile:

```bash
make profile
```

### 4. Running Manually

```bash
# Run with 1 MPI process and 16 Cilk workers
mpirun -np 1 -x CILK_NWORKERS=16 ./bin/mpi/main "Mat file/com-LiveJournal.mat"
```

---

## Performance

On a 16-core machine, this implementation achieves **~0.72s** for the LiveJournal dataset (~69M edges), utilizing a 1x16 configuration with PGO enabled.
2. For each vertex in parallel (`cilk_for`):
   - Find the minimum label among all neighbors
   - If smaller than current label, update atomically
3. Use **Cilk reducer** to track if any label changed
4. Repeat until convergence (no labels change)

### Parallelism

- Uses `cilk_for` for work-stealing parallel iteration
- Atomic compare-and-swap for lock-free updates
- Custom reducer for parallel OR reduction of `changed` flag

---

## Sample Output

```
Number of nonzeros: 69362378
Your .mat to CSR format in .bin files conversion complete.
Loaded matrix:
 number of rows:3997962,
 number of nonzeros:69362378
Using Cilk coloring implementation
OpenCilk using 16 workers(cores)
tag,work (seconds),span (seconds),parallelism,burdened_span (seconds),burdened_parallelism
BFS-OpenCilk,0.443468,0.00500878,88.5382,0.00507753,87.3394
Coloring CC: iterations=7 time=0.201595 s components=1
First 20 labels (or fewer):
1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 11:1 12:1 13:1 14:1 15:1 16:1 17:1 18:1 19:1 20:1
```

---

## Future Work

See `MPI_IMPLEMENTATION_PLAN.md` for the roadmap to extend this to distributed computing with MPI.

---

## License

Academic project for AUTH Parallel Computing course.
>>>>>>> origin/master
