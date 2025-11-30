# ==============================================================================
# Makefile for Parallel Connected Components Project
# ==============================================================================

CC = clang
# Include src/ for headers like mat_to_csr.h
# Default Cilk Flags (PGO Use) - REMOVED cilkscale for performance
CFLAGS_CILK = -fopencilk -O3 -march=native -flto -fprofile-instr-use=logs/bfscilk.profdata -Wall -pthread -I. -Isrc -Ibfscilk
LDFLAGS_CILK = -lmatio -fopencilk -flto -fuse-ld=lld -pthread

# Source files
SRCS = src/main.c bfscilk/BFScilk.c src/mat_to_csr.c

BIN_DIR := bin
BIN_CILK := $(BIN_DIR)/cilk
OBJDIR_CILK := obj_cilk

OBJS_CILK = $(OBJDIR_CILK)/src/main.o $(OBJDIR_CILK)/src/mat_to_csr.o $(OBJDIR_CILK)/bfscilk/BFScilk.o
OBJS = $(OBJS_CILK)
TARGET = $(BIN_CILK)/main

.PHONY: all clean dirs mpi dirs_mpi profile_cilk

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJDIR_CILK)/src $(OBJDIR_CILK)/bfscilk $(BIN_CILK)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS_CILK) -o $@ $(OBJS) $(LDFLAGS_CILK)

$(OBJDIR_CILK)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_CILK) -c $< -o $@

clean:
	rm -rf $(OBJDIR_CILK) $(OBJDIR_MPI) $(BIN_DIR)
	rm -f *.o logs/*.profraw

# MPI Configuration
export OMPI_CC=clang

MPICC = mpicc
# Use profile from logs/ folder
CFLAGS_MPI = -fopencilk -O3 -march=native -flto -fprofile-instr-use=logs/bfsmpi.profdata -Wall -I. -Isrc -Ibfsmpi
LDFLAGS_MPI = -lmatio -fopencilk -flto -fuse-ld=lld

SRCS_MPI = src/main_mpi.c bfsmpi/BFSmpi.c bfsmpi/partition.c src/mat_to_csr.c
OBJDIR_MPI = obj_mpi
OBJS_MPI = $(patsubst %.c, $(OBJDIR_MPI)/%.o, $(SRCS_MPI))
TARGET_MPI = bin/mpi/main

mpi: dirs_mpi $(TARGET_MPI)

dirs_mpi:
	@mkdir -p $(OBJDIR_MPI)/src $(OBJDIR_MPI)/bfsmpi bin/mpi

$(TARGET_MPI): $(OBJS_MPI)
	$(MPICC) $(CFLAGS_MPI) -o $@ $(OBJS_MPI) $(LDFLAGS_MPI)

$(OBJDIR_MPI)/%.o: %.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

# Profile Generation Target (MPI)
profile:
	$(MAKE) clean
	$(MAKE) mpi CFLAGS_MPI="-fopencilk -O3 -march=native -flto -fprofile-instr-generate -Wall -I. -Isrc -Ibfsmpi" LDFLAGS_MPI="-lmatio -fopencilk -flto -fuse-ld=lld -fprofile-instr-generate"
	LLVM_PROFILE_FILE="logs/bfsmpi-%p.profraw" ./scripts/bench_mpi.sh "Mat file/com-LiveJournal.mat" 1
	llvm-profdata merge -output=logs/bfsmpi.profdata logs/bfsmpi-*.profraw
	$(MAKE) clean
	@echo "Profile generated in logs/bfsmpi.profdata. Run 'make mpi' to use it."

# Profile Generation Target (Cilk)
profile_cilk:
	$(MAKE) clean
	$(MAKE) dirs
	$(MAKE) $(TARGET) CFLAGS_CILK="-fopencilk -O3 -march=native -flto -fprofile-instr-generate -Wall -pthread -I. -Isrc -Ibfscilk" LDFLAGS_CILK="-lmatio -fopencilk -flto -fuse-ld=lld -fprofile-instr-generate -pthread"
	LLVM_PROFILE_FILE="logs/bfscilk-%p.profraw" ./$(TARGET) "Mat file/com-LiveJournal.mat"
	llvm-profdata merge -output=logs/bfscilk.profdata logs/bfscilk-*.profraw
	$(MAKE) clean
	@echo "Profile generated in logs/bfscilk.profdata. Run 'make' to use it."

# Autotune Target
autotune:
	@chmod +x scripts/autotune_mpi.sh
	./scripts/autotune_mpi.sh "Mat file/com-LiveJournal.mat"
