#!/usr/bin/env bash
# =============================================================================
# bench.sh - OpenCilk BFS/CC Benchmark Script
# =============================================================================
#
# Purpose:
#   Runs benchmarks for the OpenCilk (shared-memory) BFS/Connected Components
#   implementation. Measures execution time across multiple runs and logs results.
#
# Usage:
#   ./scripts/bench.sh                  # Benchmark all .mat files in 'Mat file/'
#   ./scripts/bench.sh matrix.mat       # Benchmark a specific matrix file
#
# Environment Variables:
#   REPEATS        - Number of benchmark runs per matrix (default: 5)
#   CILK_NWORKERS  - Number of OpenCilk workers/threads (default: nproc)
#
# Output:
#   logs/benchmark.log - CSV with Matrix, Algorithm, Average_Time(s)
#
# Notes:
#   - Automatically generates PGO profile if not found
#   - Uses adaptive algorithm selection (Frontier BFS vs Label Propagation)
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$PROJECT_ROOT/bin"
REPEATS=${REPEATS:-5}

echo "Checking PGO profile..."
PROF_DATA="$PROJECT_ROOT/logs/bfscilk.profdata"
if [ ! -f "$PROF_DATA" ]; then
    echo "PGO profile ($PROF_DATA) not found. Generating using 'make profile_cilk'..."
    make -C "$PROJECT_ROOT" profile_cilk || { echo "Profile generation failed"; exit 1; }
else
    echo "Found existing PGO profile."
fi

echo "Cleaning previous build artifacts..."
make -C "$PROJECT_ROOT" clean

echo "Building Cilk (OpenCilk) with PGO..."
make -C "$PROJECT_ROOT" || { echo "Cilk build failed"; exit 1; }

MAIN_CILK="$BIN_DIR/cilk/main"
MAT_DIR="$PROJECT_ROOT/Mat file"
LOG_FILE="$PROJECT_ROOT/logs/benchmark.log"

if [ ! -d "$MAT_DIR" ] || [ -z "$(ls -A "$MAT_DIR")" ]; then
    echo "Matrix file directory '$MAT_DIR' not found or is empty."
    exit 1
fi

# Number of workers for Cilk (defaults to all available cores)
CILK_NWORKERS=${CILK_NWORKERS:-$(nproc)}
export CILK_NWORKERS

echo ""
echo "================================================================="
echo "Benchmark Configuration:"
echo "  Cilk Workers: $CILK_NWORKERS"
echo "  Repeats per matrix: $REPEATS"
echo "  Log File: $LOG_FILE"
echo "================================================================="
echo ""

# Initialize log file
echo "Matrix,Algorithm,Average_Time(s)" > "$LOG_FILE"

# Determine which files to process
if [ $# -ge 1 ]; then
    TARGET_MAT="$1"
    if [ -f "$TARGET_MAT" ]; then
        MAT_FILES=("$TARGET_MAT")
    elif [ -f "$MAT_DIR/$TARGET_MAT" ]; then
        MAT_FILES=("$MAT_DIR/$TARGET_MAT")
    else
        echo "Error: Matrix file '$TARGET_MAT' not found."
        exit 1
    fi
else
    # Use nullglob to handle case where no .mat files exist gracefully
    shopt -s nullglob
    MAT_FILES=("$MAT_DIR"/*.mat)
    shopt -u nullglob
fi

# Run benchmarks for each .mat file
for mat in "${MAT_FILES[@]}"; do
    if [ ! -f "$mat" ]; then continue; fi
    
    base=$(basename "$mat")
    echo "================================================================="
    echo "Benchmarking: $base"
    echo "================================================================="
    
    # Arrays to store times
    CILK_TIMES=()
    
    # Cilk Runs
    echo "Running Cilk ($REPEATS repeats)..."
    ALGO_DETECTED=""
    for i in $(seq 1 $REPEATS); do
        OUTPUT=$(BFS_ALGO=cilk "$MAIN_CILK" "$mat")
        
        # Capture algorithm selection from the first run
        if [ -z "$ALGO_DETECTED" ]; then
            ALGO_LINE=$(echo "$OUTPUT" | grep "Adaptive BFS:" || true)
            if [ -n "$ALGO_LINE" ]; then
                ALGO_DETECTED=$(echo "$ALGO_LINE" | sed -E 's/Adaptive BFS: (.*)/\1/')
                echo "  Algorithm: $ALGO_DETECTED"
            fi
        fi

        TIME_LINE=$(echo "$OUTPUT" | grep "Coloring CC:" || true)
        if [ -n "$TIME_LINE" ]; then
            TIME=$(echo "$TIME_LINE" | sed -E 's/.*time=([0-9.]+) s.*/\1/')
            CILK_TIMES+=($TIME)
            echo "  Run $i: $TIME s"
        else
            echo "  Run $i: Failed to extract time"
        fi
    done
    
    # Calculate and Log Averages
    if [ ${#CILK_TIMES[@]} -gt 0 ]; then
        CILK_AVG=$(echo "${CILK_TIMES[@]}" | LC_NUMERIC=C awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        echo "$base,Cilk,$CILK_AVG" >> "$LOG_FILE"
        echo "Cilk Average: $CILK_AVG s"
    fi
    
    echo ""
done

echo "Benchmark complete. Results saved to $LOG_FILE"
