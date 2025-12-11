#!/bin/bash
# scripts/run_cluster.sh
# Runs the MPI binary on the cluster with correct library paths.
# Usage: ./scripts/run_cluster.sh <NP> <MAT_FILE> [WORKERS]

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CILK_DIR="$PROJECT_ROOT/opencilk"
MATIO_DIR="$PROJECT_ROOT/libs_matio"

# Validate Args
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <NP> <MAT_FILE> [WORKERS]"
    echo "Example: $0 4 data/graph.mat 8"
    exit 1
fi

NP=$1
MAT_FILE=$2
WORKERS=${3:-1} # Default to 1 worker if not specified

# --- Setup Environment ---
export LD_LIBRARY_PATH="$MATIO_DIR/lib:$CILK_DIR/lib:$CILK_DIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$CILK_DIR/bin:$PATH"

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
