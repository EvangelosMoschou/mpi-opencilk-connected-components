#!/usr/bin/env bash
# bench_mpi.sh - Run MPI+Cilk benchmarks for different process/worker configurations
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGFILE="$PROJECT_ROOT/logs/benchmark_mpi.log"
BIN_MPI="$PROJECT_ROOT/bin/mpi/main"

MATRIX="$1"
# Resolve absolute path for MATRIX if it's relative
if [[ "$MATRIX" != /* ]]; then
  MATRIX="$(pwd)/$MATRIX"
fi

REPEATS=${2:-3}

# Ensure the binary exists
if [ ! -x "$BIN_MPI" ]; then
  echo "Missing binary $BIN_MPI, run 'make mpi' first"
  exit 1
fi

# Configurations to test: <num_procs>x<num_workers> -> total cores used
CONFIGS=("1x16")
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
