#!/bin/bash
# =============================================================================
# build_fix_all.sh - Robust Cluster Build Script (Safe Mode)
# =============================================================================
#
# Purpose:
#   Builds the MPI binary with conservative flags that work on all systems.
#   Use this when standard build fails due to LTO, PGO, or linker issues.
#
# What It Does:
#   1. Sets up environment (OpenCilk, Matio library paths)
#   2. Cleans previous builds (removes potentially corrupted object files)
#   3. Compiles with SAFE flags:
#      - NO -flto (avoids linker plugin errors)
#      - NO -fuse-ld=lld (works without LLD)
#      - NO -fprofile-instr-use (avoids PGO version mismatches)
#
# Usage:
#   ./scripts/build_fix_all.sh
#
# Requirements:
#   - opencilk/ directory with OpenCilk compiler
#   - libs_matio/ directory with Matio library (or run install_matio.sh)
#
# Output:
#   bin/mpi/main - Built MPI binary
#
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}/..")" && pwd)"
CILK_DIR="$PROJECT_ROOT/opencilk"
MATIO_DIR="$PROJECT_ROOT/libs_matio"

echo "=========================================="
echo "   Robust Cluster Build (Safe Mode)"
echo "=========================================="

# --- 1. Environment Setup ---
EnvironmentErrors=0

# OpenCilk
if [ -d "$CILK_DIR" ]; then
    export PATH="$CILK_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CILK_DIR/lib:$CILK_DIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    echo "[OK] OpenCilk found."
else
    echo "[ERR] OpenCilk missing at $CILK_DIR"
    EnvironmentErrors=1
fi

# Matio
if [ -d "$MATIO_DIR" ]; then
    export CPATH="$MATIO_DIR/include:$CPATH"
    export LIBRARY_PATH="$MATIO_DIR/lib:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="$MATIO_DIR/lib:$LD_LIBRARY_PATH"
    echo "[OK] Matio found."
else
    echo "[ERR] Matio missing at $MATIO_DIR. Run scripts/install_matio.sh first."
    EnvironmentErrors=1
fi

if [ $EnvironmentErrors -ne 0 ]; then
    echo "Aborting due to missing dependencies."
    exit 1
fi

# --- 2. Clean Previous Builds ---
echo "--- Cleaning old object files ---"
# We run make clean to remove the files that caused 'file format not recognized'
make clean
rm -rf obj_mpi bin/mpi

# --- 3. Build with Safe Flags ---
# We override flags to remove potential failure points:
# x No -flto (Causes Linker plugin error)
# x No -fuse-ld=lld (Missing on cluster)
# x No -fprofile-instr-use (Avoids PGO mismatch version errors)
# o Includes Matio explicitly

echo "--- Building MPI Version ---"
make mpi \
    CC=clang \
    CFLAGS_MPI="-fopencilk -O3 -march=native -Wall -I. -Isrc -Ibfsmpi -I$MATIO_DIR/include" \
    LDFLAGS_MPI="-lmatio -fopencilk -L$MATIO_DIR/lib"

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "   BUILD SUCCESSFUL!"
    echo "   Binary: bin/mpi/main"
    echo "=========================================="
else
    echo "Build Failed."
    exit 1
fi
