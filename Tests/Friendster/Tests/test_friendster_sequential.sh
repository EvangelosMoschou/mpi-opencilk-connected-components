#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --job-name=Friendster_Sequential
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --output="../Slurm output/friendster_sequential_%j.out"

# --- Optimizations ---
# Sequential run typically uses standard Label Propagation or Frontier BFS
# We disable specific parallel optimizations
export BFS_ZERO_PLANTING=1
export BFS_HUB_PUSH=1
export BFS_USE_STREAMING_FRONTIER=0 # Disable parallel streaming for sequential baseline
export BFS_DISTRIBUTED=0
export BFS_HYBRID_PACKING=0
export BFS_PREFETCH_DIST=16
export BFS_ASSUME_SYMMETRIC=1
export BFS_SKIP_REORDER=0

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
echo "Execution:     Sequential (1 Node, 1 Rank, 1 Thread)"
echo "Memory/Node:   64GB"
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

# Build Project with PGO
cd "$PROJECT_ROOT"
echo "Building project in $PROJECT_ROOT..."
make clean_all
make profile # Generate PGO data using Mawi
make mpi     # Build optimized binary using PGO data

# Find Friendster Mat file
MAT_FILE=$(find "$PROJECT_ROOT" -name "*Friendster*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: No Friendster .mat file found!"
    MAT_FILE=$(find . -name "*.mat" | head -n 1)
fi

echo "Using Matrix: $MAT_FILE"
export CILK_NWORKERS=1 # Enforce single threaded execution

# Run
mpirun -np 1 ./bin/mpi/main "$MAT_FILE"
