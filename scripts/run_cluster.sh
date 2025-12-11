#!/bin/bash
# scripts/run_cluster.sh
# Runs the MPI binary on the cluster with correct library paths.
# Usage: ./scripts/run_cluster.sh <NP> <MAT_FILE> [WORKERS]

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CILK_DIR="$PROJECT_ROOT/opencilk"
MATIO_DIR="$PROJECT_ROOT/libs_matio"

# Auto-detect matrix if needed
if [ -z "${2:-}" ]; then
     DETECTED_MAT=$(find "$PROJECT_ROOT/Mat file" -name "*.mat" 2>/dev/null | head -n 1)
     if [ -n "$DETECTED_MAT" ]; then
         MAT_FILE="$DETECTED_MAT"
         echo "Auto-detected matrix: $MAT_FILE"
     fi
else
     MAT_FILE="$2"
fi

# Validate Args
if [ -z "${1:-}" ] || [ -z "${MAT_FILE:-}" ]; then
    echo "Usage: $0 <NP> [MAT_FILE] [WORKERS]"
    echo "  <NP>       : Number of MPI processes"
    echo "  [MAT_FILE] : Path to matrix file (optional if auto-detectable)"
    echo "  [WORKERS]  : OpenCilk workers per process (default: 1)"
    exit 1
fi

NP=$1
WORKERS=${3:-1}

# --- Setup Environment ---
# Only append to paths if directories exist (handles cluster vs local differences)
if [ -d "$MATIO_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$MATIO_DIR/lib:$LD_LIBRARY_PATH"
fi
if [ -d "$CILK_DIR" ]; then
    export LD_LIBRARY_PATH="$CILK_DIR/lib:$CILK_DIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    export PATH="$CILK_DIR/bin:$PATH"
fi

# --- Run ---
echo "=================================================="
echo " Running on $NP MPI processes"
echo " OpenCilk Workers per process: $WORKERS"
echo " Graph: $MAT_FILE"
echo "=================================================="

# Check for scheduler
if command -v srun &> /dev/null; then
    RUNNER="srun -n $NP"
    echo "Using 'srun' (Slurm)..."
else
    RUNNER="mpirun -np $NP -x LD_LIBRARY_PATH -x CILK_NWORKERS"
    echo "Using 'mpirun'..."
fi

export CILK_NWORKERS=$WORKERS

$RUNNER ./bin/mpi/main "$MAT_FILE"
