#!/usr/bin/env bash
# =============================================================================
# bench_mpi.sh - MPI+OpenCilk Hybrid Benchmark Script
# =============================================================================
#
# Purpose:
#   Benchmarks the MPI+OpenCilk hybrid implementation with different
#   process/worker configurations to find optimal parallelization strategy.
#
# Usage:
#   ./scripts/bench_mpi.sh                    # Auto-detect matrix
#   ./scripts/bench_mpi.sh matrix.mat         # Use specific matrix
#   ./scripts/bench_mpi.sh matrix.mat 5       # 5 repeats per config
#
# Default Configurations Tested (16 cores total):
#   1×16  - 1 MPI process,  16 Cilk workers  (shared-memory only)
#   2×8   - 2 MPI processes, 8 Cilk workers
#   4×4   - 4 MPI processes, 4 Cilk workers
#   8×2   - 8 MPI processes, 2 Cilk workers
#
# Output:
#   logs/benchmark_mpi.log - Timing results for each configuration
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGFILE="$PROJECT_ROOT/logs/benchmark_mpi.log"
BIN_MPI="$PROJECT_ROOT/bin/mpi/main"

# Auto-detect matrix if not provided
if [ -z "${1:-}" ]; then
    # Try finding in specific folders first
    DETECTED_MAT=$(find "$PROJECT_ROOT/Mat file" -name "*.mat" | head -n 1)
    if [ -z "$DETECTED_MAT" ]; then
        # Try current directory
        DETECTED_MAT=$(find "$PROJECT_ROOT" -maxdepth 1 -name "*.mat" | head -n 1)
    fi
    
    if [ -n "$DETECTED_MAT" ]; then
        echo "Auto-detected matrix: $DETECTED_MAT"
        MATRIX="$DETECTED_MAT"
    else
        echo "Error: No .mat file provided and none found in 'Mat file/' or project root."
        exit 1
    fi
else
    MATRIX="$1"
fi

# Resolve absolute path for MATRIX if it's relative
if [[ "$MATRIX" != /* ]]; then
  MATRIX="$(pwd)/$MATRIX"
fi

if [ ! -f "$MATRIX" ]; then
    echo "Error: Matrix file not found at $MATRIX"
    exit 1
fi

REPEATS=${2:-3}

# Ensure the binary exists
if [ ! -x "$BIN_MPI" ]; then
  echo "Missing binary $BIN_MPI, run 'make mpi' first"
  exit 1
fi

# Configurations to test: <num_procs>x<num_workers> -> total cores used
CONFIGS=("1x16" "2x8" "4x4" "8x2")
REPEATS=${2:-1}

# Set OpenCilk to use a fixed worker policy per process via CILK_NWORKERS env var
# We'll set it for each mpirun invocation using -x CILK_NWORKERS

# A helper to parse config
parse_config() {
  local cfg="$1"
  num_procs=${cfg%x*}
  num_workers=${cfg#*x}
}

echo "MPI+Cilk benchmark: matrix=$MATRIX" > "$LOGFILE"

for cfg in "${CONFIGS[@]}"; do
  parse_config "$cfg"
  echo "===== Config: ${cfg} (mpirun -np ${num_procs}, CILK_NWORKERS=${num_workers}) =====" | tee -a "$LOGFILE"

  # Use mpirun with provided env var and pin processes using --map-by ppr
  # The ppr option: ppr:${num_procs}:node will create num_procs per node. On a single machine, this ensures good mapping
  # Use --bind-to core to set bindings, and --oversubscribe only when necessary
  for r in $(seq 1 $REPEATS); do
    echo "Run $r" | tee -a "$LOGFILE"
    /usr/bin/time -f "%e" -o /tmp/mptime.txt \
      mpirun -np ${num_procs} --map-by ppr:${num_procs}:node -x CILK_NWORKERS=${num_workers} "$BIN_MPI" "$MATRIX" 2>&1 | tee -a "$LOGFILE"
    # Append measured time
    echo "walltime: $(cat /tmp/mptime.txt)" | tee -a "$LOGFILE"
  done
  echo "" | tee -a "$LOGFILE"
done

echo "All done. Results written to $LOGFILE"

# Note: If you want 16 MPI ranks (each with 1 worker), add "16x1" to CONFIGS
