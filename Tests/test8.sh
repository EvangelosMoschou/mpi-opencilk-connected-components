#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=00:02:00
#SBATCH --job-name=MPI_Async_Test8_2Node_2Tasks
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --output="Slurm output/test8_async_2node_2tasks_%j.out"

# --- Optimizations ---
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_DISTRIBUTED=1
export BFS_HYBRID_PACKING=1
export BFS_PREFETCH_DIST=16

# --- ASYNC MODE ---
export BFS_ASYNC=1

# --- Environment Setup (Relative to Script) ---
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # If running via sbatch, use the submission directory (Project2/Tests) to find root
    PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"
else
    # Fallback for local execution
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
echo "Description:   Async Medium Scaling (2 Nodes, 2 Tasks/Node, 12 Core/Task, 32GB)"
echo "Mode:          ASYNC"
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
make clean BIN_DIR=bin/test8 OBJ_DIR=obj/test8
make mpi BIN_DIR=bin/test8 OBJ_DIR=obj/test8

# Detect Matrix
MAT_FILE=$(find . -name "*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: No .mat file found!"
    exit 1
fi

echo "Using Matrix: $MAT_FILE"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

export CILK_NWORKERS=$SLURM_CPUS_PER_TASK

# Run
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/test8/mpi/main "$MAT_FILE"
