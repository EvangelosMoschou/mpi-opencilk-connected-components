#!/bin/bash

# Configuration
MAT_FILE="com-Friendster.mat"  # ~65M vertices, ~1.8B edges
PROCS=8                       # Number of MPI processes
CPUS_PER_PROC=1               # OpenCilk workers per process
BIN_DIR="bin/local_fast"
OBJ_DIR="obj/local_fast"
LOG_DIR="logs"

# Ensure directories exist
mkdir -p "$BIN_DIR" "$OBJ_DIR" "$LOG_DIR"

# --- Optimizations ---
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_DISTRIBUTED=1
export BFS_HYBRID_PACKING=1
export BFS_PREFETCH_DIST=32
export BFS_ASYNC=1
export BFS_ASSUME_SYMMETRIC=1 # Skip symmetrization (saves 3x RAM)
export BFS_SKIP_REORDER=1     # Skip reordering (saves 2x RAM)
# export BFS_USE_FRONTIER=1   # DISABLED: OOM due to large buffers

# Setup Local OpenCilk (Adjust path as needed, assuming project root)
PROJECT_ROOT="$(pwd)"
if [ -d "$PROJECT_ROOT/opencilk" ]; then
    echo "Using Local OpenCilk from $PROJECT_ROOT/opencilk"
    export PATH="$PROJECT_ROOT/opencilk/bin:$PATH"
    export LD_LIBRARY_PATH="$PROJECT_ROOT/opencilk/lib:$PROJECT_ROOT/opencilk/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
else
    echo "Warning: Local 'opencilk' directory not found. Using system compiler."
fi

# Detect Matrix
FOUND_MAT=$(find . -name "$MAT_FILE" | head -n 1)
if [ -z "$FOUND_MAT" ]; then
    echo "Error: $MAT_FILE not found in $(pwd) or subdirectories."
    exit 1
fi
echo "Using Matrix: $FOUND_MAT"

echo "=========================================="
echo "       LOCAL PGO BENCHMARK RUN            "
echo "=========================================="

# --- PGO Workflow ---
echo ""
echo ">>> Starting PGO Workflow (ASYNC MODE) <<<"

# 1. Build for Profiling
echo "[PGO 1/4] Building Generation Binary..."
rm -f "$LOG_DIR"/bfsmpi_local_fast-*.profraw "$LOG_DIR"/bfsmpi_local_fast.profdata
rm -rf "$BIN_DIR" "$OBJ_DIR" # Force clean build

make mpi \
    CFLAGS_MPI="-fopencilk -O3 -march=native -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi $(pkg-config --cflags matio) -I/usr/include/hdf5/serial" \
    LDFLAGS_MPI="$(pkg-config --libs matio) -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -fopencilk -fprofile-instr-generate" \
    BIN_DIR="$BIN_DIR" OBJ_DIR="$OBJ_DIR"

if [ ! -f "$BIN_DIR/mpi/main" ]; then
    echo "Error: Build failed!"
    exit 1
fi

# 2. Run Profiling (Generate .profraw)
echo "[PGO 2/4] Running Profiling Phase..."
export LLVM_PROFILE_FILE="$LOG_DIR/bfsmpi_local_fast-%p.profraw"
export CILK_NWORKERS=$CPUS_PER_PROC

# Run once to generate profile
mpirun -np $PROCS -x LLVM_PROFILE_FILE -x CILK_NWORKERS "$BIN_DIR/mpi/main" "$FOUND_MAT"

if ! ls "$LOG_DIR"/bfsmpi_local_fast-*.profraw >/dev/null 2>&1; then
    echo "Error: No profile data generated!"
    exit 1
fi

# 3. Merge Profiles
echo "[PGO 3/4] Merging Profiles..."
llvm-profdata merge -output="$LOG_DIR/bfsmpi_local_fast.profdata" "$LOG_DIR"/bfsmpi_local_fast-*.profraw
echo "Profile saved to $LOG_DIR/bfsmpi_local_fast.profdata"

# 4. Build Optimized
echo "[PGO 4/4] Building Optimized Binary..."
# Clean objects to force rebuild
rm -rf "$BIN_DIR" "$OBJ_DIR"

make mpi \
    CFLAGS_MPI="-fopencilk -O3 -march=native -Wall -Wno-deprecated-declarations -I. -Isrc -Ibfsmpi $(pkg-config --cflags matio) -I/usr/include/hdf5/serial" \
    LDFLAGS_MPI="$(pkg-config --libs matio) -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -fopencilk" \
    BIN_DIR="$BIN_DIR" OBJ_DIR="$OBJ_DIR" \
    PGO_MPI_FLAG="-fprofile-instr-use=$LOG_DIR/bfsmpi_local_fast.profdata"

if [ ! -f "$BIN_DIR/mpi/main" ]; then
    echo "Error: Optimized build failed!"
    exit 1
fi

echo ">>> PGO Workflow Complete <<<"
echo ""

# --- Final Benchmark Run ---
echo "Running Optimized Benchmark (Async PGO)..."
mpirun -np $PROCS "$BIN_DIR/mpi/main" "$FOUND_MAT"
