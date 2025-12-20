# Detailed Usage Guide

This guide covers building, running, and optimizing the Hybrid MPI + OpenCilk Connected Components application.

---

## 1. Prerequisites

Ensure your environment is set up with the following:

*   **Compiler:** Clang with OpenCilk support (e.g., OpenCilk 2.0+).
*   **MPI:** OpenMPI or MPICH.
*   **Libraries:** `libmatio` (for reading MATLAB files), `libhdf5`.
*   **Tools:** `llvm-profdata` (for PGO).

---

## 2. Building the Project

The `Makefile` provides several targets for different build needs.

### Standard Build (MPI + Cilk)
To build the MPI version (main entry point for clusters):
```bash
make mpi
```
*   **Output:** `bin/mpi/main`

To build the Cilk-only version (for single-node execution):
```bash
make all
```
*   **Output:** `bin/cilk/main`

### Profile Guided Optimization (PGO)
To get maximum performance, you should generate a profile based on your specific workload.

**For MPI:**
```bash
make profile
```
*   **What it does:** Compiles with instrumentation, runs a benchmark, merges profile data into `logs/bfsmpi.profdata`, and cleans up.

**For Cilk:**
```bash
make profile_cilk
```

### Autotuning (Recommended)
If you are on a new machine, use the autotuner to find the best configuration.

```bash
make autotune
```
*   **What it does:** Detects cores, tests all valid configurations, generates PGO profiles, and saves the fastest binary to `./main_best`.

---

## 3. Running the Application

### Using the Autotuned Binary
If you ran `make autotune`, use the generated `main_best`.

```bash
# The best configuration is determined during autotuning
mpirun -np <BEST_NP> -x CILK_NWORKERS=<BEST_THREADS> ./main_best "Mat file/your_graph.mat"
```

### Manual Execution (MPI)
**Syntax:**
```bash
mpirun -np <NUM_PROCS> -x CILK_NWORKERS=<NUM_WORKERS> ./bin/mpi/main <MATRIX_FILE>
```

**Examples:**

*   **Pure MPI (16 processes, 1 thread each):**
    ```bash
    mpirun -np 16 -x CILK_NWORKERS=1 ./bin/mpi/main "Mat file/graph.mat"
    ```

*   **Hybrid (4 processes, 4 threads each):**
    ```bash
    mpirun -np 4 -x CILK_NWORKERS=4 ./bin/mpi/main "Mat file/graph.mat"
    ```

### Advanced Hybrid Execution (Optimized)
For massive graphs like Friendster, use the following flags:

1.  **`BFS_DISTRIBUTED=1`**: Enables memory-efficient distributed loading from SSD.
2.  **`BFS_HYBRID_PACKING=1`**: Parallelizes message packing using Cilk.
3.  **`BFS_USE_STREAMING_FRONTIER=1`**: Crucial for avoiding OOM on large frontiers (e.g. Friendster).

**Full Command:**
```bash
BFS_DISTRIBUTED=1 BFS_HYBRID_PACKING=1 BFS_USE_STREAMING_FRONTIER=1 CILK_NWORKERS=4 \
mpirun -np 4 --bind-to none ./bin/mpi/main "Mat file/friendster.mat"
```

---

## 4. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CILK_NWORKERS` | `16` | Cilk worker count (intra-node parallelism) |
| `BFS_DISTRIBUTED` | `0` | Enable memory-efficient matrix traversal (1=ON) |
| `BFS_HYBRID_PACKING` | `0` | Enable parallel communication packing (1=ON) |
| `BFS_USE_STREAMING_FRONTIER` | `0` | Use streaming instead of bulk frontier exchange (1=ON) |
| `BFS_SAMPLING_ROUNDS` | `0` | Number of initial rounds to use neighbor sampling |
| `BFS_HUB_PUSH` | `1` | Enable hub-based label propagation (1=ON) |

---

## 5. Troubleshooting

*   **Cannot open .mat file:** Ensure the file exists in `Mat file/` and you have `libmatio` installed.
*   **Profile not found:** You need to run `make profile` or `make profile_cilk` first.
*   **Out of Memory (OOM):** For large graphs, ensure `BFS_DISTRIBUTED=1` and `BFS_USE_STREAMING_FRONTIER=1` are set.
*   **Performance issues in Hybrid:** Ensure you use `--bind-to none` with `mpirun` to allow Cilk to manage threads efficiently.

---

## Scaling Study

To test scaling on a single node:
```bash
for t in 1 2 4 8 16; do
  CILK_NWORKERS=$t ./bin/cilk/main "Mat file/graph.mat"
done
```
