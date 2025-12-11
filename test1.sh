#!/bin/bash

#SBATCH --partition=rome
#SBATCH --time=00:05:00 
#SBATCH --job-name=MPI+OMP_test 
#SBATCH --nodes=2
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10 



# Load modules in dependency order
# OpenCilk v2.0 requires gcc/9.4.0
module load gcc/9.4.0-eewq4j6
module load OpenCilk/v2.0

# Load other dependencies (may swap GCC runtime but hopefully compatible)
module load openmpi/5.0.3-rhzbeym
module load matio/1.5.26-xzs7hxf


# Compile the project first (since we ran make clean)
echo "Building project..."
make mpi

# Detect Matrix File (similar to our other scripts)
MAT_FILE=$(find "Mat file" -name "*.mat" | head -n 1)
if [ -z "$MAT_FILE" ]; then
    echo "Error: No .mat file found!"
    exit 1
fi

echo "Using Matrix: $MAT_FILE"
export CILK_NWORKERS=$SLURM_CPUS_PER_TASK

# Run the correct binary with arguments
srun ./bin/mpi/main "$MAT_FILE"