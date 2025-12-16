# ==============================================================================
# Makefile for Parallel Connected Components Project
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration & Compiler Detection
# ------------------------------------------------------------------------------

# Detect OpenCilk
LOCAL_CILK_CHECK := $(shell test -x ./opencilk/bin/clang && echo yes)

ifeq ($(LOCAL_CILK_CHECK),yes)
    CILK_BASE := $(abspath ./opencilk)
    CC := $(CILK_BASE)/bin/clang
    export PATH := $(CILK_BASE)/bin:$(PATH)
    $(info [Config] Local OpenCilk: $(CC))
else
    CILK_PATH := $(shell which cilk 2>/dev/null)
    ifeq ($(CILK_PATH),)
        CC = clang
        $(warning [Config] OpenCilk not found. Defaulting to 'clang'.)
    else
        CC = $(CILK_PATH)
        $(info [Config] System OpenCilk: $(CC))
    endif
endif

# MPI Configuration
MPICC = mpicc
export OMPI_CC=$(CC)
export MPICH_CC=$(CC)
export I_MPI_CC=$(CC)

# ------------------------------------------------------------------------------
# Build Flags
# ------------------------------------------------------------------------------

# Common Flags
# Use pkg-config for matio if available, otherwise fallback to -lmatio
MATIO_FLAGS := $(shell pkg-config --cflags matio 2>/dev/null)
MATIO_LIBS  := $(shell pkg-config --libs matio 2>/dev/null || echo "-lmatio")

# HDF5 Detection (Debian uses hdf5-serial, RHEL/Rocky uses hdf5)
# If HDF5_DIR or HDF5_ROOT is set (common in modules), use it.
HDF5_LDFLAGS :=
HDF5_CFLAGS :=
ifneq ($(HDF5_DIR),)
    HDF5_LDFLAGS := -L$(HDF5_DIR)/lib
    HDF5_CFLAGS  := -I$(HDF5_DIR)/include
endif
ifneq ($(HDF5_ROOT),)
    HDF5_LDFLAGS := -L$(HDF5_ROOT)/lib
    HDF5_CFLAGS  := -I$(HDF5_ROOT)/include
endif

HDF5_LIBS := $(shell pkg-config --libs hdf5-serial 2>/dev/null || pkg-config --libs hdf5 2>/dev/null || echo "-lhdf5")
HDF5_INCS := $(shell pkg-config --cflags hdf5-serial 2>/dev/null || pkg-config --cflags hdf5 2>/dev/null)

CFLAGS_COMMON  = -fopencilk -O3 -march=native -Wall -Wno-deprecated-declarations -I. -Isrc $(MATIO_FLAGS) $(HDF5_CFLAGS) $(HDF5_INCS)
LDFLAGS_COMMON = $(MATIO_LIBS) $(HDF5_LDFLAGS) $(HDF5_LIBS) -fopencilk


# Cilk-specific
# Only use PGO if the profile data exists
PGO_CILK_FLAG = $(if $(wildcard logs/bfscilk.profdata),-fprofile-instr-use=logs/bfscilk.profdata,)
CFLAGS_CILK    = $(CFLAGS_COMMON) -pthread -Ibfscilk $(PGO_CILK_FLAG)
LDFLAGS_CILK   = $(LDFLAGS_COMMON) -pthread

# MPI-specific
PGO_MPI_FLAG = $(if $(wildcard logs/bfsmpi.profdata),-fprofile-instr-use=logs/bfsmpi.profdata,)
CFLAGS_MPI     = $(CFLAGS_COMMON) -Ibfsmpi $(PGO_MPI_FLAG)
LDFLAGS_MPI    = $(LDFLAGS_COMMON)

# ------------------------------------------------------------------------------
# Sources & Objects
# ------------------------------------------------------------------------------

# Output Directories
BIN_DIR     := bin
OBJ_DIR     := obj
BIN_CILK    := $(BIN_DIR)/cilk
BIN_MPI     := $(BIN_DIR)/mpi
OBJ_CILK    := $(OBJ_DIR)/cilk
OBJ_MPI     := $(OBJ_DIR)/mpi

# Cilk Targets
SRCS_CILK   := src/main.c bfscilk/BFScilk.c src/mat_to_csr.c
OBJS_CILK   := $(SRCS_CILK:%.c=$(OBJ_CILK)/%.o)
TARGET_CILK := $(BIN_CILK)/main

# MPI Targets
SRCS_MPI    := src/main_mpi.c bfsmpi/BFSmpi.c bfsmpi/partition.c src/mat_to_csr.c
OBJS_MPI    := $(SRCS_MPI:%.c=$(OBJ_MPI)/%.o)
TARGET_MPI  := $(BIN_MPI)/main

# ------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------

.PHONY: all clean mpi profile profile_cilk autotune

all: $(TARGET_CILK)

mpi: $(TARGET_MPI)

# Cilk Build Rules
$(TARGET_CILK): $(OBJS_CILK)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_CILK) -o $@ $^ $(LDFLAGS_CILK)

$(OBJ_CILK)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_CILK) -c $< -o $@

# MPI Build Rules
$(TARGET_MPI): $(OBJS_MPI)
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS_MPI) -o $@ $^ $(LDFLAGS_MPI)

$(OBJ_MPI)/%.o: %.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

# Clean build artifacts (keep profiles)
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) logs/*.profraw
	rm -rf obj_mpi obj_cilk

# Clean everything including profiles
clean_all: clean
	rm -rf logs/*.profdata

# ------------------------------------------------------------------------------
# Profile-Guided Optimization (PGO)
# ------------------------------------------------------------------------------

# Helper for PGO
PGO_GEN_FLAGS_CILK_C = $(filter-out -fprofile-instr-use=%,$(CFLAGS_CILK)) -fprofile-instr-generate
PGO_GEN_FLAGS_CILK_L = $(filter-out -fprofile-instr-use=%,$(LDFLAGS_CILK)) -fprofile-instr-generate
PGO_GEN_FLAGS_MPI_C  = $(filter-out -fprofile-instr-use=%,$(CFLAGS_MPI)) -fprofile-instr-generate
PGO_GEN_FLAGS_MPI_L  = $(filter-out -fprofile-instr-use=%,$(LDFLAGS_MPI)) -fprofile-instr-generate

profile_cilk: clean
	@mkdir -p $(BIN_CILK) logs
	$(MAKE) $(TARGET_CILK) CFLAGS_CILK="$(PGO_GEN_FLAGS_CILK_C)" LDFLAGS_CILK="$(PGO_GEN_FLAGS_CILK_L)"
	LLVM_PROFILE_FILE="logs/bfscilk-%p.profraw" ./$(TARGET_CILK) "Mat file/com-LiveJournal.mat"
	llvm-profdata merge -output=logs/bfscilk.profdata logs/bfscilk-*.profraw
	$(MAKE) clean
	@echo "Profile generated: logs/bfscilk.profdata"

profile: clean
	@mkdir -p $(BIN_MPI) logs
	$(MAKE) $(TARGET_MPI) CFLAGS_MPI="$(PGO_GEN_FLAGS_MPI_C)" LDFLAGS_MPI="$(PGO_GEN_FLAGS_MPI_L)"
	# Run on Mawi sample for profiling (Friendster is too big for profiling step)
	LLVM_PROFILE_FILE="logs/bfsmpi-%p.profraw" ./scripts/bench_mpi.sh "Mat file/mawi_201512020330.mat" 1
	llvm-profdata merge -output=logs/bfsmpi.profdata logs/bfsmpi-*.profraw
	# Clean build artifacts but KEEP .profdata
	$(MAKE) clean
	@echo "Profile generated: logs/bfsmpi.profdata"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

# Prepare: Detect .mat file and pre-generate .csrbin
prepare: $(TARGET_CILK)
	@echo "Detected Matrix: $$(./scripts/bench_mpi.sh --detect-only 2>/dev/null | grep 'Auto-detected' | awk '{print $$3}')"
	@# We use the bench script to find the file, then run conversion
	@MAT_FILE=$$(find "Mat file" -name "*.mat" 2>/dev/null | head -n 1); \
	if [ -z "$$MAT_FILE" ]; then \
		MAT_FILE=$$(find . -maxdepth 1 -name "*.mat" 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$MAT_FILE" ]; then \
		echo "Preparing cache for $$MAT_FILE ..."; \
		./$(TARGET_CILK) "$$MAT_FILE" --convert-only; \
	else \
		echo "No .mat file found to prepare."; \
	fi

autotune: prepare
	@chmod +x scripts/autotune_mpi.sh
	./scripts/autotune_mpi.sh
