#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=00:04:00
#SBATCH --job-name=MPI_PGO_Test10_2Node_4Tasks
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --output="Slurm output/test10_pgo_2node_4tasks_%j.out"

# --- Optimizations ---
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_DISTRIBUTED=1
export BFS_HYBRID_PACKING=1
export BFS_PREFETCH_DIST=16

# --- Environment Setup (Relative to Script) ---
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

echo "=========================================="
echo "       SLURM JOB CONFIGURATION            "
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Description:   PGO ENABLED Test (Based on Test 7)"
echo "Mode:          PGO (Profile-Guided Optimization)"
echo "=========================================="

# Load modules
module load gcc/13.2.0-iqpfkya
module load openmpi/5.0.3-rhzbeym
module load matio/1.5.26-xzs7hxf

# Setup Local OpenCilk
if [ -d "$PROJECT_ROOT/opencilk" ]; then
    echo "Using Local OpenCilk from $PROJECT_ROOT/opencilk"
    export PATH="$PROJECT_ROOT/opencilk/bin:$PATH"
    export LD_LIBRARY_PATH="$PROJECT_ROOT/opencilk/lib:$PROJECT_ROOT/opencilk/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
else
    echo "Error: Local 'opencilk' directory not found at $PROJECT_ROOT/opencilk"
    exit 1
fi

cd "$PROJECT_ROOT"

# Detect Matrix (Needed for Profiling)
MAT_FILE=$(find . -name "*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: No .mat file found!"
    exit 1
fi
echo "Using Matrix: $MAT_FILE"

# --- PGO Workflow ---
echo ""
echo ">>> Starting PGO Workflow <<<"

# 1. Build for Profiling
echo "[PGO 1/4] Building Generation Binary..."
make clean BIN_DIR=bin/test10 OBJ_DIR=obj/test10
# Use flags for instrumentation
make mpi \
    CFLAGS_MPI="-fopencilk -O3 -march=native -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi $(pkg-config --cflags matio)" \
    LDFLAGS_MPI="$(pkg-config --libs matio) -fopencilk -fprofile-instr-generate" \
    BIN_DIR=bin/test10 OBJ_DIR=obj/test10

if [ ! -f "bin/test10/mpi/main" ]; then
    echo "Error: Build failed!"
    exit 1
fi

# 2. Run Profiling (Generate .profraw)
echo "[PGO 2/4] Running Profiling Phase..."
# Clean old logs
rm -f logs/bfsmpi_test10-*.profraw
export LLVM_PROFILE_FILE="logs/bfsmpi_test10-%p.profraw"
export CILK_NWORKERS=$SLURM_CPUS_PER_TASK

# Run once to generate profile
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/test10/mpi/main "$MAT_FILE"

if ! ls logs/bfsmpi_test10-*.profraw >/dev/null 2>&1; then
    echo "Error: No profile data generated!"
    # Fallback or continue? better exit
    exit 1
fi

# 3. Merge Profiles
echo "[PGO 3/4] Merging Profiles..."
llvm-profdata merge -output=logs/bfsmpi_test10.profdata logs/bfsmpi_test10-*.profraw
echo "Profile saved to logs/bfsmpi_test10.profdata"

# 4. Build Optimized
echo "[PGO 4/4] Building Optimized Binary..."
rm -rf bin/test10 obj/test10
# Standard make mpi will detect logs/bfsmpi.profdata and use -fprofile-instr-use
make mpi BIN_DIR=bin/test10 OBJ_DIR=obj/test10 PGO_MPI_FLAG="-fprofile-instr-use=logs/bfsmpi_test10.profdata"

if [ ! -f "bin/test10/mpi/main" ]; then
    echo "Error: Optimized build failed!"
    exit 1
fi

echo ">>> PGO Workflow Complete <<<"
echo ""

# --- Final Benchmark Run ---
echo "Running Optimized Benchmark..."
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/test10/mpi/main "$MAT_FILE"
