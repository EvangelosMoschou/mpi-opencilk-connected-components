#!/bin/bash

#SBATCH --partition=rome
#SBATCH --time=00:05:00 
#SBATCH --job-name=MPI+OMP_test 
#SBATCH --nodes=2
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10 



# Load modules in dependency order
# Load modules
# We need GCC for dependencies. OpenCilk v2.0 is based on LLVM 14.
# A newer GCC is generally fine and often preferred for better standard lib support.
module load gcc/13.2.0-iqpfkya
# module load OpenCilk/v2.0 # Using local opencilk folder instead

# Load other dependencies
module load openmpi/5.0.3-rhzbeym
module load matio/1.5.26-xzs7hxf

# Setup Local OpenCilk Environment
PROJECT_ROOT=$(pwd)
if [ -d "$PROJECT_ROOT/opencilk" ]; then
    echo "Using Local OpenCilk from $PROJECT_ROOT/opencilk"
    export PATH="$PROJECT_ROOT/opencilk/bin:$PATH"
    export LD_LIBRARY_PATH="$PROJECT_ROOT/opencilk/lib:$PROJECT_ROOT/opencilk/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
else
    echo "Error: Local 'opencilk' directory not found!"
    exit 1
fi

# Load other dependencies (may swap GCC runtime but hopefully compatible)
module load openmpi/5.0.3-rhzbeym
module load matio/1.5.26-xzs7hxf


# Compile the project first (since we ran make clean)
echo "Building project..."
make clean
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