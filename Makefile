# ==============================================================================
# Makefile for Parallel Connected Components Project
# ==============================================================================

# ------------------------------------------------------------------------------
# Cilk (Sequential/Shared Memory) Configuration
# ------------------------------------------------------------------------------

# Compiler: Using Clang with OpenCilk support
# Auto-locate OpenCilk: try 'cilk' command first (as requested), fallback to 'clang'
CILK_PATH := $(shell which cilk 2>/dev/null)
ifeq ($(CILK_PATH),)
    CC = clang
else
    CC = $(CILK_PATH)
endif

# Compilation Flags:
# -fopencilk: Enable OpenCilk parallelism
# -O3: Maximum optimization level
# -march=native: Optimize for the current CPU architecture
# -flto: Enable link-time optimization
# -fprofile-instr-use: Use profile-guided optimization (PGO) data
# -Wall: Enable all warnings
# -pthread: Enable POSIX threads support
# -I.: Include current directory for headers
CFLAGS_CILK = -fopencilk -O3 -march=native -flto -fprofile-instr-use=logs/bfscilk.profdata -Wall -pthread -I. -Isrc -Ibfscilk

# Linker Flags:
# -lmatio: Link against matio library (for .mat file reading)
# -fuse-ld=lld: Use LLVM's lld linker for faster linking
LDFLAGS_CILK = -lmatio -fopencilk -flto -fuse-ld=lld -pthread

# Source files for Cilk version
SRCS = src/main.c bfscilk/BFScilk.c src/mat_to_csr.c

# Output directories
BIN_DIR := bin
BIN_CILK := $(BIN_DIR)/cilk
OBJDIR_CILK := obj_cilk

# Object files and target executable for Cilk version
OBJS_CILK = $(OBJDIR_CILK)/src/main.o $(OBJDIR_CILK)/src/mat_to_csr.o $(OBJDIR_CILK)/bfscilk/BFScilk.o
OBJS = $(OBJS_CILK)
TARGET = $(BIN_CILK)/main

# Declare phony targets (not actual files)
.PHONY: all clean dirs mpi dirs_mpi profile_cilk

# ------------------------------------------------------------------------------
# Build Targets (Cilk)
# ------------------------------------------------------------------------------

# Default target: Build the Cilk version
all: dirs $(TARGET)

# Create necessary directories for Cilk build
dirs:
	@mkdir -p $(OBJDIR_CILK)/src $(OBJDIR_CILK)/bfscilk $(BIN_CILK)

# Link object files into final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS_CILK) -o $@ $(OBJS) $(LDFLAGS_CILK)

# Pattern rule: Compile .c files to .o files
$(OBJDIR_CILK)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_CILK) -c $< -o $@

# Clean all build artifacts
clean:
	rm -rf $(OBJDIR_CILK) $(OBJDIR_MPI) $(BIN_DIR)
	rm -f *.o logs/*.profraw

# ------------------------------------------------------------------------------
# MPI (Distributed Memory) Configuration
# ------------------------------------------------------------------------------

# Force MPI to use Clang as the underlying C compiler
export OMPI_CC=$(CC)

# MPI compiler wrapper
MPICC = mpicc

# MPI Compilation Flags (includes PGO, Cilk for nested parallelism)
CFLAGS_MPI = -fopencilk -O3 -march=native -flto -fprofile-instr-use=logs/bfsmpi.profdata -Wall -I. -Isrc -Ibfsmpi

# MPI Linker Flags
LDFLAGS_MPI = -lmatio -fopencilk -flto -fuse-ld=lld

# Source files for MPI version
SRCS_MPI = src/main_mpi.c bfsmpi/BFSmpi.c bfsmpi/partition.c src/mat_to_csr.c

# Output directories and files
OBJDIR_MPI = obj_mpi
OBJS_MPI = $(patsubst %.c, $(OBJDIR_MPI)/%.o, $(SRCS_MPI))
TARGET_MPI = bin/mpi/main

# ------------------------------------------------------------------------------
# Build Targets (MPI)
# ------------------------------------------------------------------------------

# Build the MPI version
mpi: dirs_mpi $(TARGET_MPI)

# Create necessary directories for MPI build
dirs_mpi:
	@mkdir -p $(OBJDIR_MPI)/src $(OBJDIR_MPI)/bfsmpi bin/mpi

# Link MPI object files into final executable
$(TARGET_MPI): $(OBJS_MPI)
	$(MPICC) $(CFLAGS_MPI) -o $@ $(OBJS_MPI) $(LDFLAGS_MPI)

# Pattern rule: Compile .c files to .o files for MPI
$(OBJDIR_MPI)/%.o: %.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

# ------------------------------------------------------------------------------
# Profile-Guided Optimization (PGO) Targets
# ------------------------------------------------------------------------------

# Generate PGO profile data for MPI version
# Steps:
#   1. Clean previous builds
#   2. Build with instrumentation (-fprofile-instr-generate)
#   3. Run benchmark to collect profile data
#   4. Merge all .profraw files into a single .profdata file
#   5. Clean instrumented build (next 'make mpi' will use the profile)
profile:
	$(MAKE) clean
	$(MAKE) mpi CFLAGS_MPI="-fopencilk -O3 -march=native -flto -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi" LDFLAGS_MPI="-lmatio -fopencilk -flto -fuse-ld=lld -fprofile-instr-generate"
	LLVM_PROFILE_FILE="logs/bfsmpi-%p.profraw" ./scripts/bench_mpi.sh "Mat file/com-LiveJournal.mat" 1
	llvm-profdata merge -output=logs/bfsmpi.profdata logs/bfsmpi-*.profraw
	$(MAKE) clean
	@echo "Profile generated in logs/bfsmpi.profdata. Run 'make mpi' to use it."

# Generate PGO profile data for Cilk version
# Similar to MPI profile, but for the shared-memory Cilk version
profile_cilk:
	$(MAKE) clean
	$(MAKE) dirs
	$(MAKE) $(TARGET) CFLAGS_CILK="-fopencilk -O3 -march=native -flto -fprofile-instr-generate -Wall -pthread -I. -Isrc -Ibfscilk" LDFLAGS_CILK="-lmatio -fopencilk -flto -fuse-ld=lld -fprofile-instr-generate -pthread"
	LLVM_PROFILE_FILE="logs/bfscilk-%p.profraw" ./$(TARGET) "Mat file/com-LiveJournal.mat"
	llvm-profdata merge -output=logs/bfscilk.profdata logs/bfscilk-*.profraw
	$(MAKE) clean
	@echo "Profile generated in logs/bfscilk.profdata. Run 'make' to use it."

# ------------------------------------------------------------------------------
# Autotune Target
# ------------------------------------------------------------------------------

# Run autotuning script to find optimal MPI/worker configurations
autotune:
	@chmod +x scripts/autotune_mpi.sh
	./scripts/autotune_mpi.sh "Mat file/com-LiveJournal.mat"
