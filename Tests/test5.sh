#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=00:02:00
#SBATCH --job-name=MPI_Sequential_Test5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --output="Slurm output/test5_seq_%j.out"

# --- Optimizations ---
# Keeping optimizations enabled for consistent comparison
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_DISTRIBUTED=0
export BFS_HYBRID_PACKING=0
export BFS_PREFETCH_DIST=16

# Exit on error
set -e

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
echo "Node Count:    $SLURM_JOB_NUM_NODES"
echo "Tasks/Node:    $SLURM_NTASKS_PER_NODE"
echo "CPUs/Task:     $SLURM_CPUS_PER_TASK"
echo "Memory/Node:   32GB"
echo "Date:          $(date)"
echo "Description:   Sequential Run (1 Node, 1 Core used, 10 Reserved)"
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

# Build Project
cd "$PROJECT_ROOT"
echo "Building project in $PROJECT_ROOT..."
make clean
make mpi

# Detect Matrix
MAT_FILE=$(find . -name "*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    MAT_FILE=$(find "Mat file" -name "*.mat" | head -n 1)
fi

if [ -z "$MAT_FILE" ]; then
    echo "Error: No .mat file found!"
    exit 1
fi

echo "Using Matrix: $MAT_FILE"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Force sequential execution even if SLURM assigns more resources by accident
export CILK_NWORKERS=1

# Run without mpirun/srun to ensure single process (or srun -n 1)
# We reserve 10 cores but explicitly use only 1 (-c 1)
mpirun -np 1 ./bin/mpi/main "$MAT_FILE"
