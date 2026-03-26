# Parallel Image Processing System

A high-performance image processing system demonstrating three levels of parallelism using **OpenMP**, **MPI**, and a **Hybrid MPI+OpenMP** approach, developed as a final project for the Parallel & Distributed Computing (PDC) Lab.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Operations Implemented](#operations-implemented)
5. [Parallelism Strategy](#parallelism-strategy)
6. [Prerequisites](#prerequisites)
7. [Build](#build)
8. [Usage](#usage)
9. [Benchmark & Plot](#benchmark--plot)
10. [Performance Results](#performance-results)
11. [Technical Details](#technical-details)

---

## Problem Statement

Image processing tasks (filtering, edge detection, colour transforms) require per-pixel computation over millions of pixels. A 4 K image has ~12 million pixels — perfect for data-parallel acceleration.

This project implements and compares:

| Version    | Parallelism           | Best for                  |
| ---------- | --------------------- | ------------------------- |
| Sequential | None                  | Correctness baseline      |
| OpenMP     | Shared-memory threads | Single multi-core machine |
| MPI        | Distributed processes | Multi-node clusters       |
| Hybrid     | MPI × OpenMP          | Full cluster utilisation  |

---

## Architecture Overview

```
Input Image
     │
     ▼
┌────────────┐         ┌──────────────────────────────────────┐
│  Rank 0    │ Scatter │  Rank 1 … N–1                        │
│  (master)  ├────────►│  Process row-strips independently    │
│            │ Gather  │  (OpenMP threads within each rank)   │
│            │◄────────│                                      │
└────────────┘         └──────────────────────────────────────┘
     │
     ▼
Output Image / Timing CSV
```

For operations requiring context from neighbouring rows (Gaussian blur, Sobel), each rank exchanges **halo rows** with its neighbours before computing.

---

## Project Structure

```
Parallel-Image-Processing-System/
├── include/
│   └── image_processing.h       # Shared types, function declarations
├── src/
│   ├── sequential/
│   │   └── sequential.cpp       # Pure sequential implementation
│   ├── openmp/
│   │   └── openmp_proc.cpp      # OpenMP parallelised implementation
│   ├── mpi/
│   │   └── mpi_proc.cpp         # MPI distributed implementation
│   └── hybrid/
│       └── hybrid_proc.cpp      # Hybrid MPI + OpenMP implementation
├── scripts/
│   ├── benchmark.sh             # Full benchmark sweep script
│   ├── gui_app.py               # Desktop GUI to run all versions
│   └── plot_results.py          # Performance visualisation (Python)
├── test_images/                 # Input images placed here
├── results/
│   ├── images/                  # Output images written here
│   ├── data/                    # CSV timing files
│   └── plots/                   # Generated performance charts
├── Makefile
└── README.md
```

---

## Operations Implemented

| #   | Operation                  | Description                                       |
| --- | -------------------------- | ------------------------------------------------- |
| 1   | **Grayscale**              | ITU-R BT.601 luma: `Y = 0.299R + 0.587G + 0.114B` |
| 2   | **Gaussian Blur**          | 5×5 kernel, σ≈1.4, border-replicated              |
| 3   | **Sobel Edge Detection**   | Gradient magnitude, 3×3 kernel                    |
| 4   | **Brightness/Contrast**    | `dst = α·src + β` (α=1.2, β=30)                   |
| 5   | **Histogram Equalization** | Global CDF-based pixel remapping                  |

---

## Parallelism Strategy

### OpenMP

- `#pragma omp parallel for schedule(static)` over the outer row loop
- Thread count controlled by `OMP_NUM_THREADS` or the CLI `-t` argument
- Histogram equalization uses thread-local partial histograms + `#pragma omp critical` reduction

### MPI

- **Scatter**: Rank 0 loads image, splits into row-strips, distributes with `MPI_Scatterv`
- **Halo exchange**: Neighbours send 1–2 border rows for stencil operations
- **Gather**: Results collected on rank 0 with `MPI_Gatherv`
- **Histogram**: Each rank builds a local histogram; `MPI_Reduce(MPI_SUM)` merges them; rank 0 computes LUT, then `MPI_Bcast` distributes it

### Hybrid MPI+OpenMP

- MPI distributes row-strips across nodes
- Within each MPI rank, OpenMP threads process pixels in parallel
- Total parallelism = `P × T` (processes × threads per process)

---

## Prerequisites

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y \
    build-essential \
    libopencv-dev \
    libopenmpi-dev \
    openmpi-bin \
    python3-pip

pip3 install pandas matplotlib numpy
```

Verify installations:

```bash
g++ --version
mpic++ --version
pkg-config --modversion opencv4
python3 -c "import cv2; print(cv2.__version__)"
```

---

## Build

```bash
# Clone / navigate to project
cd Parallel-Image-Processing-System

# Build all four binaries
make all

# Or build individually
make sequential
make openmp
make mpi
make hybrid
```

Binaries are placed in `build/`.

---

## Usage

### Provide a test image

```bash
# Auto-generate a 4096×4096 synthetic image
make test_image

# Or copy your own image
cp /path/to/your/photo.jpg test_images/sample.jpg
```

### Run individually

```bash
# Sequential
./build/sequential test_images/sample.jpg

# OpenMP (8 threads)
./build/openmp_proc test_images/sample.jpg 8

# MPI (4 processes)
mpirun -np 4 ./build/mpi_proc test_images/sample.jpg

# Hybrid (4 processes, 4 threads each = 16 total workers)
mpirun -np 4 ./build/hybrid_proc test_images/sample.jpg 4
```

### Quick run with defaults

```bash
make run                            # uses OMP_THREADS=4, MPI_PROCS=4
make run OMP_THREADS=8 MPI_PROCS=4 IMAGE=test_images/sample.jpg
```

### Desktop GUI

```bash
# Build all binaries and launch the GUI
make gui

# Or run directly
python3 scripts/gui_app.py
```

GUI features:
- Image picker for your custom photo
- Per-version toggles (Sequential, OpenMP, MPI, Hybrid, OpenCL)
- Runtime parameters (OpenMP threads, MPI processes, Hybrid threads)
- One-click build, run selected/all, and plot generation
- Live execution logs and timing table
- Quick open buttons for `results/images` and `results/plots`

Output images appear in `results/images/` and timing CSVs in `results/data/`.

---

## Benchmark & Plot

```bash
# Full sweep (threads: 1,2,4,8  processes: 1,2,4,8)
make benchmark

# Generate all performance charts
make plot
```

Charts saved to `results/plots/`:

| Chart                        | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `speedup_by_operation.png`   | Speedup vs. workers, one subplot per operation |
| `speedup_by_version.png`     | Best speedup grouped bar chart                 |
| `elapsed_time_heatmap.png`   | Hybrid elapsed time heatmap (P × T)            |
| `scalability_efficiency.png` | Parallel efficiency % vs. workers              |
| `summary_table.png`          | Side-by-side timing & speedup table            |

---

## Performance Results

Expected speedup trends on an 8-core machine with a 4 K image:

| Version    | Workers     | Typical Speedup |
| ---------- | ----------- | --------------- |
| Sequential | 1           | 1× (baseline)   |
| OpenMP     | 4 threads   | ~3–4×           |
| OpenMP     | 8 threads   | ~5–7×           |
| MPI        | 4 processes | ~2.5–3.5×       |
| MPI        | 8 processes | ~4–6×           |
| Hybrid     | 4P×2T       | ~5–7×           |
| Hybrid     | 2P×4T       | ~5–7×           |

> Speedups vary by operation. Gaussian blur (stencil, high arithmetic intensity) scales best. Histogram equalization is limited by memory bandwidth and the serial LUT build step.

---

## Technical Details

### Halo / Ghost Rows (Stencil Operations)

For Gaussian blur (5×5 kernel, 2-row halo) and Sobel (3×3 kernel, 1-row halo):

```
Rank k receives:
  ┌─────────────────┐  ← halo from rank k-1 (or border-replicated)
  │  local strip    │
  └─────────────────┘  ← halo from rank k+1 (or border-replicated)
```

### Load Balancing

Row distribution: `rows_per_rank[i] = H/P + (i < H%P ? 1 : 0)`  
Distributes remainder rows to the first `H mod P` ranks — maximum 1-row imbalance.

### Timing

- Sequential: `std::chrono::steady_clock`
- OpenMP: `omp_get_wtime()` (wall-clock)
- MPI / Hybrid: `MPI_Wtime()` (wall-clock, barrier-synchronised before starting)

### CSV Output Schema

```
version, operation, width, height, threads, processes, elapsed_sec, speedup
```

---

## Authors

Developed for PDC Lab — 6th Semester BSSE  
Technology: C++17 · OpenMP · Open MPI · OpenCV 4
