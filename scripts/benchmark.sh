#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# benchmark.sh — Full benchmark suite for Parallel Image Processing System
#
# Runs each binary across a range of thread/process counts and collects timing.
# Results are merged into results/data/all_results.csv for plotting.
#
# Usage:  bash scripts/benchmark.sh [image_path]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE="${1:-test_images/sample.jpg}"
BUILD="build"
DATA="results/data"
mkdir -p "$DATA"

SEQ_BIN="$BUILD/sequential"
OMP_BIN="$BUILD/openmp_proc"
MPI_BIN="$BUILD/mpi_proc"
HYB_BIN="$BUILD/hybrid_proc"

# Thread / process counts to sweep
THREAD_COUNTS=(1 2 4 8)
PROC_COUNTS=(1 2 4 8)

echo "=============================================="
echo " Parallel Image Processing — Benchmark Suite"
echo "=============================================="
echo " Image : $IMAGE"
echo " Threads tested : ${THREAD_COUNTS[*]}"
echo " Processes tested: ${PROC_COUNTS[*]}"
echo "----------------------------------------------"

# ── Check binaries exist ───────────────────────────────────────────────────
for bin in "$SEQ_BIN" "$OMP_BIN" "$MPI_BIN" "$HYB_BIN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found. Run 'make all' first."
        exit 1
    fi
done

# ── 1. Sequential (single run — baseline) ─────────────────────────────────
echo ""
echo ">>> Sequential"
"$SEQ_BIN" "$IMAGE"

# ── 2. OpenMP sweep ────────────────────────────────────────────────────────
echo ""
echo ">>> OpenMP"
for t in "${THREAD_COUNTS[@]}"; do
    echo "  threads = $t"
    "$OMP_BIN" "$IMAGE" "$t"
    # Rename CSV so each run doesn't overwrite the last
    if [[ -f "$DATA/openmp_results.csv" ]]; then
        cp "$DATA/openmp_results.csv" "$DATA/openmp_t${t}.csv"
    fi
done

# ── 3. MPI sweep ───────────────────────────────────────────────────────────
echo ""
echo ">>> MPI"
for p in "${PROC_COUNTS[@]}"; do
    echo "  processes = $p"
    mpirun --oversubscribe -np "$p" "$MPI_BIN" "$IMAGE"
    if [[ -f "$DATA/mpi_results.csv" ]]; then
        cp "$DATA/mpi_results.csv" "$DATA/mpi_p${p}.csv"
    fi
done

# ── 4. Hybrid sweep ────────────────────────────────────────────────────────
echo ""
echo ">>> Hybrid (MPI × OpenMP)"
for p in 1 2 4; do
    for t in 1 2 4; do
        echo "  processes=$p  threads=$t"
        mpirun --oversubscribe -np "$p" "$HYB_BIN" "$IMAGE" "$t"
        if [[ -f "$DATA/hybrid_results.csv" ]]; then
            cp "$DATA/hybrid_results.csv" "$DATA/hybrid_p${p}_t${t}.csv"
        fi
    done
done

# ── 5. Merge all CSVs ──────────────────────────────────────────────────────
echo ""
echo ">>> Merging results …"
ALL="$DATA/all_results.csv"
echo "version,operation,width,height,threads,processes,elapsed_sec,speedup" > "$ALL"

# Read sequential baseline timings for speedup computation
declare -A SEQ_TIME
while IFS=',' read -r ver op w h th pr el sp; do
    [[ "$ver" == "version" ]] && continue
    SEQ_TIME["$op"]="$el"
done < "$DATA/sequential_results.csv"

# Append sequential (speedup = 1.0)
tail -n +2 "$DATA/sequential_results.csv" >> "$ALL"

# Append parallel runs with computed speedup
for csv in "$DATA"/openmp_t*.csv "$DATA"/mpi_p*.csv "$DATA"/hybrid_p*_t*.csv; do
    [[ -f "$csv" ]] || continue
    while IFS=',' read -r ver op w h th pr el sp; do
        [[ "$ver" == "version" ]] && continue
        base="${SEQ_TIME[$op]:-1}"
        speedup=$(python3 -c "print(round($base/$el,4))" 2>/dev/null || echo "0")
        echo "$ver,$op,$w,$h,$th,$pr,$el,$speedup"
    done < "$csv"
done >> "$ALL"

echo "All results merged → $ALL"
echo ""
echo "Run 'make plot' to generate performance graphs."
