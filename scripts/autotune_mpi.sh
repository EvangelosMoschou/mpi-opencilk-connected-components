#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
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

# Resolve absolute path for MATRIX
if [[ "$MATRIX" != /* ]]; then
  MATRIX="$PROJECT_ROOT/$MATRIX"
fi

if [ ! -f "$MATRIX" ]; then
    echo "Error: Matrix file not found at $MATRIX"
    exit 1
fi

# 1. Detect Resources
# Try to detect total cores. If in a job, nproc might return the slot count.
TOTAL_CORES=$(nproc)
echo "Detected $TOTAL_CORES available cores."

# 2. Generate Configurations (P x W = TOTAL_CORES)
# We prefer P to be powers of 2 for MPI, but we'll try all divisors to be thorough.
CONFIGS=()
for (( p=1; p<=TOTAL_CORES; p++ )); do
    if (( TOTAL_CORES % p == 0 )); then
        w=$(( TOTAL_CORES / p ))
        CONFIGS+=("$p $w")
    fi
done

echo "Testing configurations (Procs x Workers):"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $cfg"
done
echo "---------------------------------------------------"

BEST_TIME=999999.0
BEST_CONFIG=""

# Helper for float comparison
is_smaller() {
    awk -v n1="$1" -v n2="$2" 'BEGIN {if (n1<n2) exit 0; else exit 1}'
}

cd "$PROJECT_ROOT"

for cfg in "${CONFIGS[@]}"; do
    read p w <<< "$cfg"
    echo ">>> Testing Config: $p MPI Procs x $w Cilk Workers"
    
    # --- Step 1: Profile Generation ---
    echo "    [1/3] Generating Profile..."
    make clean > /dev/null 2>&1
    
    # Build with PGO Generation flags
    # We use the same flags as the Makefile's 'profile' target but manually specified
    make mpi \
        CFLAGS_MPI="-fopencilk -O3 -march=native -flto -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi" \
        LDFLAGS_MPI="-lmatio -fopencilk -flto -fuse-ld=lld -fprofile-instr-generate" \
        > /dev/null 2>&1
    
    if [ ! -f "bin/mpi/main" ]; then
        echo "    Build failed! Skipping..."
        continue
    fi

    # Run to generate profile data
    # We run just once for profiling
    export LLVM_PROFILE_FILE="logs/autotune_%p.profraw"
    # Clean old profiles
    rm -f logs/autotune_*.profraw
    
    mpirun --oversubscribe -np "$p" -x CILK_NWORKERS="$w" ./bin/mpi/main "$MATRIX" > /dev/null 2>&1
    
    # Merge profiles
    llvm-profdata merge -output=logs/autotune.profdata logs/autotune_*.profraw > /dev/null 2>&1
    rm -f logs/autotune_*.profraw
    
    # --- Step 2: Optimized Build ---
    echo "    [2/3] Building Optimized Binary..."
    make clean > /dev/null 2>&1
    
    make mpi \
        CFLAGS_MPI="-fopencilk -O3 -march=native -flto -fprofile-instr-use=logs/autotune.profdata -Wall -I. -Isrc -Ibfsmpi" \
        LDFLAGS_MPI="-lmatio -fopencilk -flto -fuse-ld=lld" \
        > /dev/null 2>&1

    if [ ! -f "bin/mpi/main" ]; then
        echo "    Optimized build failed! Skipping..."
        continue
    fi

    # --- Step 3: Benchmark ---
    echo "    [3/3] Benchmarking (3 runs)..."
    MIN_RUN_TIME=999999.0
    
    for i in {1..3}; do
        # Run benchmark
        OUTPUT=$(mpirun --oversubscribe -np "$p" -x CILK_NWORKERS="$w" ./bin/mpi/main "$MATRIX" 2>&1)
        
        # Parse time: "MPI CC: procs=... time=0.749078 s ..."
        # We use grep and awk to extract the float value
        TIME=$(echo "$OUTPUT" | grep "MPI CC:" | sed -E 's/.*time=([0-9.]+) s.*/\1/')
        
        if [ -z "$TIME" ]; then
             echo "      Run $i: Failed to parse time."
             echo "      --- Output Head ---"
             echo "$OUTPUT" | head -n 10
             echo "      -------------------"
             continue
        fi
        
        echo "      Run $i: $TIME s"
        
        if is_smaller "$TIME" "$MIN_RUN_TIME"; then
            MIN_RUN_TIME=$TIME
        fi
    done
    
    echo "    Best for ($p x $w): $MIN_RUN_TIME s"
    
    if is_smaller "$MIN_RUN_TIME" "$BEST_TIME"; then
        BEST_TIME=$MIN_RUN_TIME
        BEST_CONFIG="$p x $w"
        # Save the best binary safely (outside bin/ which gets cleaned)
        cp bin/mpi/main ./main_best
        cp logs/autotune.profdata logs/best.profdata
    fi
    echo "---------------------------------------------------"
done

echo ""
echo "==================================================="
echo "Optimization Complete."
echo "Best Configuration: $BEST_CONFIG"
echo "Best Time: $BEST_TIME s"
echo "The optimized binary is saved as './main_best'"
echo "The PGO profile is saved as 'logs/best.profdata'"
echo "==================================================="
