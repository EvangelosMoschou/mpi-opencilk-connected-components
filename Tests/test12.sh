#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=00:20:00
#SBATCH --job-name=MPI_PGO_Async_Test12_mycielskian20
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --output="Slurm output/test12_pgo_async_mycielskian20_%j.out"

# --- Optimizations ---
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_DISTRIBUTED=1
export BFS_HYBRID_PACKING=1
export BFS_PREFETCH_DIST=32 # Aggressive prefetch for speed

# --- ASYNC MODE ---
export BFS_ASYNC=1

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
echo "Description:   ABSOLUTE SPEED Test (Async + PGO) for Mycielskian20"
echo "Mode:          PGO + ASYNC"
echo "Matrix:        mycielskian20.mat"
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

# Detect Matrix (Specific Target)
MAT_FILE=$(find . -name "mycielskian20.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: mycielskian20.mat not found! Searching recursively..."
    MAT_FILE=$(find . -name "mycielskian20.mat" | head -n 1)
    if [ -z "$MAT_FILE" ]; then
         echo "Critcal Error: Matrix file not found."
         # Fallback to verify logic, but preferably fail.
         exit 1
    fi
fi
echo "Using Matrix: $MAT_FILE"

# --- PGO Workflow ---
echo ""
echo ">>> Starting PGO Workflow (ASYNC MODE) <<<"

# 1. Build for Profiling
echo "[PGO 1/4] Building Generation Binary..."
rm -rf bin/test12 obj/test12 logs/bfsmpi_test12-*.profraw logs/bfsmpi_test12.profdata
# Use flags for instrumentation (NO LTO, NO LLD)
make mpi \
    CFLAGS_MPI="-fopencilk -O3 -march=native -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi $(pkg-config --cflags matio)" \
    LDFLAGS_MPI="$(pkg-config --libs matio) -fopencilk -fprofile-instr-generate" \
    BIN_DIR=bin/test12 OBJ_DIR=obj/test12

if [ ! -f "bin/test12/mpi/main" ]; then
    echo "Error: Build failed!"
    exit 1
fi

# 2. Run Profiling (Generate .profraw)
echo "[PGO 2/4] Running Profiling Phase..."
# Clean old logs
rm -f logs/bfsmpi_test12-*.profraw
export LLVM_PROFILE_FILE="logs/bfsmpi_test12-%p.profraw"
export CILK_NWORKERS=$SLURM_CPUS_PER_TASK

# Run once to generate profile
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/test12/mpi/main "$MAT_FILE"

if ! ls logs/bfsmpi_test12-*.profraw >/dev/null 2>&1; then
    echo "Error: No profile data generated!"
    exit 1
fi

# 3. Merge Profiles
echo "[PGO 3/4] Merging Profiles..."
llvm-profdata merge -output=logs/bfsmpi_test12.profdata logs/bfsmpi_test12-*.profraw
echo "Profile saved to logs/bfsmpi_test12.profdata"

# 4. Build Optimized
echo "[PGO 4/4] Building Optimized Binary..."
# Manual clean to preserve logs
rm -rf bin/test12 obj/test12
# Standard make mpi will detect logs/bfsmpi.profdata and use -fprofile-instr-use
# We pass the specific profile file to the make command via CFLAGS/LDFLAGS override if needed, 
# or we use the custom PGO_MPI_FLAG support if we added it to Makefile, or we can just pass it directly.
# The Makefile supports PGO_MPI_FLAG.
make mpi BIN_DIR=bin/test12 OBJ_DIR=obj/test12 PGO_MPI_FLAG="-fprofile-instr-use=logs/bfsmpi_test12.profdata"

if [ ! -f "bin/test12/mpi/main" ]; then
    echo "Error: Optimized build failed!"
    exit 1
fi

echo ">>> PGO Workflow Complete <<<"
echo ""

# --- Final Benchmark Run ---
echo "Running Optimized Benchmark (Async PGO)..."
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/test12/mpi/main "$MAT_FILE"
