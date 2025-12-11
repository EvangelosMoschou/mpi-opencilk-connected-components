# Detailed Usage Guide

This guide covers building, running, and optimizing the Hybrid MPI + OpenCilk Connected Components application.

---

## 1. Prerequisites

Ensure your environment is set up with the following:

*   **Compiler:** Clang with OpenCilk support (e.g., OpenCilk 2.0+).
*   **MPI:** OpenMPI or MPICH.
*   **Libraries:** `libmatio` (for reading MATLAB files).
*   **Tools:** `llvm-profdata` (for PGO).

---

## 2. Building the Project

The `Makefile` provides several targets for different build needs.

### Standard Build (MPI + Cilk)
Builds the MPI version. If a PGO profile exists in `logs/bfsmpi.profdata`, it will be used automatically.

```bash
make mpi
```
*   **Output:** `bin/mpi/main`

### Profile Guided Optimization (PGO)
To get maximum performance, you should generate a profile based on your specific workload.

```bash
make profile
```
*   **What it does:**
    1.  Compiles with instrumentation.
    2.  Runs a benchmark on `Mat file/com-LiveJournal.mat`.
    3.  Merges the profile data into `logs/bfsmpi.profdata`.
    4.  Cleans the build so you can run `make mpi` next.

### Autotuning (Recommended)
If you are on a new machine, use the autotuner to find the best configuration.

```bash
make autotune
```
*   **What it does:**
    1.  Detects available cores.
    2.  Tests all valid `Process x Worker` combinations.
    3.  Generates PGO profiles for each.
    4.  Saves the fastest binary to `./main_best`.

---

## 3. Running the Application

### Using the Autotuned Binary
If you ran `make autotune`, the best binary is in the root directory.

```bash
# Example: If the best config was 1 process x 16 workers
mpirun -np 1 -x CILK_NWORKERS=16 ./main_best "Mat file/com-LiveJournal.mat"
```

### Manual Execution
You can run the standard binary located in `bin/mpi/main`.

**Syntax:**
```bash
mpirun -np <NUM_PROCS> -x CILK_NWORKERS=<NUM_WORKERS> ./bin/mpi/main <MATRIX_FILE>
```

**Examples:**

*   **Pure MPI (16 processes, 1 thread each):**
    ```bash
    mpirun -np 16 -x CILK_NWORKERS=1 ./bin/mpi/main "Mat file/com-LiveJournal.mat"
    ```

*   **Pure Cilk (1 process, 16 threads):**
    ```bash
    mpirun -np 1 -x CILK_NWORKERS=16 ./bin/mpi/main "Mat file/com-LiveJournal.mat"
    ```

    mpirun -np 4 -x CILK_NWORKERS=4 ./bin/mpi/main "Mat file/com-LiveJournal.mat"
    ```

### Advanced Hybrid Execution (Optimized)
To fully utilize modern multi-core clusters, use the Distributed Loading and Hybrid Packing extensions.

1.  **Enable Distributed Loading (`BFS_DISTRIBUTED=1`)**:
    Reduces memory pressure on Rank 0 by letting every node load its own chunk of the graph.
    
2.  **Enable Parallel Packing (`BFS_HYBRID_PACKING=1`)**:
    Allows all threads within a process to pack MPI messages concurrently.

3.  **Unbind Threads (`--bind-to none`)**:
    Crucial for OpenCilk threads to roam freely across cores when running inside MPI.

**Full Command:**
```bash
BFS_DISTRIBUTED=1 BFS_HYBRID_PACKING=1 CILK_NWORKERS=4 \
mpirun -np 4 --bind-to none ./bin/mpi/main "Mat file/kmer_V1r.mat"
```

---

## 4. Benchmarking Scripts

### `scripts/bench_mpi.sh`
A simple wrapper to run the benchmark with a specific matrix.

```bash
./scripts/bench_mpi.sh "Mat file/com-LiveJournal.mat" [REPEATS]
```

### `scripts/autotune_mpi.sh`
The advanced autotuner script.

```bash
./scripts/autotune_mpi.sh "Mat file/com-LiveJournal.mat"
```

---

## 5. Troubleshooting

*   **"Missing binary ... run make mpi first":**
    You need to compile the code. Run `make mpi`.

*   **"Profile not found":**
    The compiler is looking for `logs/bfsmpi.profdata`. Run `make profile` to generate it, or remove `-fprofile-instr-use` from the Makefile if you don't want PGO.

*   **"There are not enough slots available":**
    OpenMPI thinks you are oversubscribing the machine. Use `mpirun --oversubscribe ...` or ensure your hostfile is correct. The autotuner handles this automatically.


### Save Output to Log

```bash
./bench.sh 2>&1 | tee my_benchmark_$(date +%Y%m%d_%H%M).log
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | `16` | OpenMP/Cilk worker count (runtime) |
| `CILK_NWORKERS` | `16` | Cilk worker count (runtime) |
| `PTHREADS` | `16` | PThreads count (compile-time for `make pthr`) |
| `REPEATS` | `5` | Number of benchmark repeats |
| `SKIP_SEQ` | `0` | Set to `1` to skip 1-thread sequential run |
| `BFS_ALGO` | `parallel` | Algorithm: `unionfind`, `parallel`, or `inplace` |

---

## Understanding Output

Each run prints:

```
Coloring CC: iterations=7 time=0.185322 s components=1
```

- **iterations**: Number of iterations to converge
- **time**: Wall-clock execution time
- **components**: Number of connected components found

**Cilk** also prints work/span analysis:

```
OpenCilk using 16 workers(cores)
BFS-OpenCilk,0.38253,0.00211979,180.457,0.00218854,174.788
```

---

## Common Workflows

### Quick Test Run

```bash
make omp
OMP_NUM_THREADS=4 ./bin/omp/main "Mat file/test_graph.mat" -a unionfind
```

### Compare Algorithms (OpenMP)

```bash
export OMP_NUM_THREADS=16
MAT="Mat file/my_graph.mat"

echo "=== Union-Find ==="
./bin/omp/main "$MAT" -a unionfind

echo "=== Label Propagation ==="
./bin/omp/main "$MAT" -a parallel
```

### Full Benchmark Suite

```bash
# Clean and rebuild
make clean
make all

# Run with 16 workers, skip sequential, save log
OMP_THREADS=16 PTHREADS=16 SKIP_SEQ=1 ./bench.sh 2>&1 | tee results_16t.log
```

### Scaling Study

```bash
for t in 1 4 8 16 32; do
  echo "=== Testing with $t threads ==="
  OMP_NUM_THREADS=$t ./bin/omp/main "Mat file/graph.mat" -a parallel
done
```

---

## Finding Your CPU Core Count

```bash
nproc                              # Quick count
lscpu | grep "^CPU(s):"            # Detailed info
grep -c ^processor /proc/cpuinfo   # /proc method
```

Use this value for `OMP_NUM_THREADS` or `CILK_NWORKERS`.

---

## Dependencies

**Required:**
- GCC with OpenMP support
- libmatio (`sudo apt-get install libmatio-dev`)
- pthread (usually included)

**Optional:**
- OpenCilk clang (for Cilk builds)

---

## Troubleshooting

**"cannot open .mat file"**
- Check that the file exists in `Mat file/` directory
- Verify file path doesn't have special characters

**Cilk build fails**
- Update `CC2` path in `Makefile` to your OpenCilk clang location
- Or skip Cilk: `make seq omp pthr`

**Out of memory**
- Large graphs need substantial RAM (8+ GB for 200M+ vertices)
- Try a smaller graph or increase system swap

**Thread count mismatch (PThreads)**
- Rebuild: `PTHREADS=<desired_count> make pthr`
- Thread count is compile-time, not runtime
