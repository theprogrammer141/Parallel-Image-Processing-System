# ─────────────────────────────────────────────────────────────────────────────
# Parallel Image Processing System — Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Targets:
#   make all        — build all four binaries
#   make sequential — build sequential version
#   make openmp     — build OpenMP version
#   make mpi        — build MPI version
#   make hybrid     — build Hybrid MPI+OpenMP version
#   make run        — run all versions with default settings
#   make benchmark  — run full benchmark suite via shell script
#   make plot       — generate performance plots (requires Python)
#   make clean      — remove build artefacts
#   make test_image — download/generate a test image
# ─────────────────────────────────────────────────────────────────────────────

CXX        := g++
MPICXX     := mpic++
CXXFLAGS   := -O2 -std=c++17 -Wall -Wextra
OMP_FLAGS  := -fopenmp
# Silence spurious cast warning from Open MPI's own C++ headers (op_inln.h)
MPI_WFLAGS := -Wno-cast-function-type

# OpenCV flags (pkg-config preferred, falls back to manual)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv 2>/dev/null)

INCLUDE    := -Iinclude

BUILD_DIR  := build
SRC_SEQ    := src/sequential/sequential.cpp
SRC_OMP    := src/openmp/openmp_proc.cpp
SRC_MPI    := src/mpi/mpi_proc.cpp
SRC_HYB    := src/hybrid/hybrid_proc.cpp
SRC_OCL    := src/opencl/opencl_proc.cpp

BIN_SEQ    := $(BUILD_DIR)/sequential
BIN_OMP    := $(BUILD_DIR)/openmp_proc
BIN_MPI    := $(BUILD_DIR)/mpi_proc
BIN_HYB    := $(BUILD_DIR)/hybrid_proc
BIN_OCL    := $(BUILD_DIR)/opencl_proc

OCL_LIBS   := -lOpenCL

IMAGE      ?= test_images/sample.jpg
OMP_THREADS ?= 4
MPI_PROCS   ?= 4

.PHONY: all sequential openmp mpi hybrid opencl run benchmark plot clean test_image dirs

all: dirs sequential openmp mpi hybrid opencl

dirs:
	@mkdir -p $(BUILD_DIR) results/images results/data results/plots

# ── Sequential ────────────────────────────────────────────────────────────────
sequential: dirs
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OPENCV_CFLAGS) \
		$(SRC_SEQ) -o $(BIN_SEQ) $(OPENCV_LIBS)
	@echo "Built: $(BIN_SEQ)"

# ── OpenMP ────────────────────────────────────────────────────────────────────
openmp: dirs
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDE) $(OPENCV_CFLAGS) \
		$(SRC_OMP) -o $(BIN_OMP) $(OPENCV_LIBS)
	@echo "Built: $(BIN_OMP)"

# ── MPI ───────────────────────────────────────────────────────────────────────
mpi: dirs
	$(MPICXX) $(CXXFLAGS) $(MPI_WFLAGS) $(INCLUDE) $(OPENCV_CFLAGS) \
		$(SRC_MPI) -o $(BIN_MPI) $(OPENCV_LIBS)
	@echo "Built: $(BIN_MPI)"

# ── Hybrid MPI + OpenMP ───────────────────────────────────────────────────────
hybrid: dirs
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) $(MPI_WFLAGS) $(INCLUDE) $(OPENCV_CFLAGS) \
		$(SRC_HYB) -o $(BIN_HYB) $(OPENCV_LIBS)
	@echo "Built: $(BIN_HYB)"

# ── OpenCL GPU ────────────────────────────────────────────────────────────────
opencl: dirs
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OPENCV_CFLAGS) \
		$(SRC_OCL) -o $(BIN_OCL) $(OPENCV_LIBS) $(OCL_LIBS) \
		-Wno-deprecated-declarations
	@echo "Built: $(BIN_OCL)"

# ── Quick Run (defaults) ─────────────────────────────────────────────────────
run: all test_image
	@echo "\n=== Sequential ==="
	$(BIN_SEQ) $(IMAGE)
	@echo "\n=== OpenMP ($(OMP_THREADS) threads) ==="
	$(BIN_OMP) $(IMAGE) $(OMP_THREADS)
	@echo "\n=== MPI ($(MPI_PROCS) processes) ==="
	mpirun -np $(MPI_PROCS) $(BIN_MPI) $(IMAGE)
	@echo "\n=== Hybrid ($(MPI_PROCS) procs × $(OMP_THREADS) threads) ==="
	mpirun -np $(MPI_PROCS) $(BIN_HYB) $(IMAGE) $(OMP_THREADS)
	@echo "\n=== OpenCL GPU ==="
	$(BIN_OCL) $(IMAGE)

# ── Full Benchmark Suite ──────────────────────────────────────────────────────
benchmark: all test_image
	bash scripts/benchmark.sh $(IMAGE)

# ── Performance Plots ─────────────────────────────────────────────────────────
plot:
	python3 scripts/plot_results.py

# ── Generate / download test image ───────────────────────────────────────────
# Accepts either JPEG or PPM output from gen_test_image.py
test_image:
	@mkdir -p test_images
	@if [ ! -f test_images/sample.jpg ] && [ ! -f test_images/sample.ppm ]; then \
		echo "Generating synthetic test image (4096x4096) ..."; \
		python3 scripts/gen_test_image.py; \
	fi
	$(eval IMAGE := $(shell \
		if [ -f test_images/sample.jpg ]; then echo test_images/sample.jpg; \
		else echo test_images/sample.ppm; fi))

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) results/images/* results/data/* results/plots/*
	@echo "Cleaned build artefacts."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  all          Build all binaries"
	@echo "  sequential   Build sequential binary"
	@echo "  openmp       Build OpenMP binary"
	@echo "  mpi          Build MPI binary"
	@echo "  hybrid       Build Hybrid MPI+OpenMP binary"
	@echo "  opencl       Build OpenCL GPU binary"
	@echo "  run          Quick run with defaults"
	@echo "  benchmark    Full benchmark suite"
	@echo "  plot         Generate performance plots"
	@echo "  test_image   Create synthetic test image"
	@echo "  clean        Remove build artefacts"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE=$(IMAGE)"
	@echo "  OMP_THREADS=$(OMP_THREADS)"
	@echo "  MPI_PROCS=$(MPI_PROCS)"
