#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --job-name=Friendster_1Node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output="../Slurm output/friendster_1node_%j.out"

# --- Optimizations ---
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_USE_STREAMING_FRONTIER=1
export BFS_DISTRIBUTED=1
export BFS_HYBRID_PACKING=1
export BFS_PREFETCH_DIST=16
export BFS_ASSUME_SYMMETRIC=1  # Friendster is symmetric, skip expensive check
export BFS_SKIP_REORDER=0      # Use reordering for better BFS performance

# --- Environment Setup ---
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SLURM_SUBMIT_DIR")")")"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

echo "=========================================="
echo "       SLURM JOB CONFIGURATION            "
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node Count:    $SLURM_JOB_NUM_NODES"
echo "Memory/Node:   64GB"
echo "Description:   Friendster 1 Node (Hybrid: 8 MPI x 16 Cilk)"
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

# Find Friendster Mat file
# Adjust searching path if necessary, or assume it's in Mat file/
MAT_FILE=$(find "$PROJECT_ROOT" -name "*Friendster*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: No Friendster .mat file found!"
    # Fallback to local search just in case
    MAT_FILE=$(find . -name "*.mat" | head -n 1)
fi

echo "Using Matrix: $MAT_FILE"
export CILK_NWORKERS=$SLURM_CPUS_PER_TASK

# Run
mpirun -np $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE)) ./bin/mpi/main "$MAT_FILE"
