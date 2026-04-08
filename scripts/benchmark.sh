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
TMP_RUNS="$DATA/.runs"

cleanup_tmp_runs() {
    rm -rf "$TMP_RUNS"
}

trap cleanup_tmp_runs EXIT

mkdir -p "$DATA"
cleanup_tmp_runs
mkdir -p "$TMP_RUNS"

SEQ_BIN="$BUILD/sequential"
OMP_BIN="$BUILD/openmp_proc"
MPI_BIN="$BUILD/mpi_proc"
HYB_BIN="$BUILD/hybrid_proc"
OCL_BIN="$BUILD/opencl_proc"

# Thread / process counts to sweep
THREAD_COUNTS=(1 2 4 8)
PROC_COUNTS=(1 2 4 8)
HYB_PROC_COUNTS=(1 2 4)
HYB_THREAD_COUNTS=(1 2 4)

# Benchmark rigor controls (override via env)
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT_RUNS="${REPEAT_RUNS:-3}"

# MPI launcher behavior
MPI_OVERSUBSCRIBE="${MPI_OVERSUBSCRIBE:-1}"
OPENCL_REQUIRED="${OPENCL_REQUIRED:-0}"

echo "=============================================="
echo " Parallel Image Processing — Benchmark Suite"
echo "=============================================="
echo " Image : $IMAGE"
echo " Threads tested : ${THREAD_COUNTS[*]}"
echo " Processes tested: ${PROC_COUNTS[*]}"
echo " Warmups per config: $WARMUP_RUNS"
echo " Repeats per config: $REPEAT_RUNS"
echo " MPI oversubscribe: $MPI_OVERSUBSCRIBE"
echo " OpenCL required: $OPENCL_REQUIRED"
echo "----------------------------------------------"

if [[ "$WARMUP_RUNS" -lt 0 || "$REPEAT_RUNS" -lt 1 ]]; then
    echo "ERROR: WARMUP_RUNS must be >= 0 and REPEAT_RUNS must be >= 1"
    exit 1
fi

mpirun_args() {
    local np="$1"
    if [[ "$MPI_OVERSUBSCRIBE" == "1" ]]; then
        echo "--oversubscribe -np $np"
    else
        echo "-np $np"
    fi
}

aggregate_median_csv() {
    local out_csv="$1"
    shift
    local files=("$@")

    python3 - "$out_csv" "${files[@]}" <<'PY'
import csv
import statistics
import sys

out_csv = sys.argv[1]
files = sys.argv[2:]
groups = {}

for fpath in files:
    with open(fpath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["version"],
                row["operation"],
                row["width"],
                row["height"],
                row["threads"],
                row["processes"],
            )
            groups.setdefault(key, []).append(float(row["elapsed_sec"]))

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "version", "operation", "width", "height",
        "threads", "processes", "elapsed_sec", "speedup"
    ])

    def key_sort(item):
        k = item[0]
        return (k[0], k[1], int(k[4]), int(k[5]))

    for key, values in sorted(groups.items(), key=key_sort):
        version, op, w, h, th, pr = key
        median_elapsed = statistics.median(values)
        writer.writerow([version, op, w, h, th, pr, f"{median_elapsed:.6f}", "0"])
PY
}

run_with_repeats() {
    local result_csv="$1"
    local aggregate_csv="$2"
    shift 2
    local cmd=("$@")
    local files=()
    local warm
    local rep

    for ((warm = 1; warm <= WARMUP_RUNS; ++warm)); do
        echo "    warmup $warm/$WARMUP_RUNS"
        "${cmd[@]}" >/dev/null
    done

    for ((rep = 1; rep <= REPEAT_RUNS; ++rep)); do
        echo "    run $rep/$REPEAT_RUNS"
        "${cmd[@]}"
        local run_csv="$TMP_RUNS/$(basename "${aggregate_csv%.csv}")_r${rep}.csv"
        cp "$result_csv" "$run_csv"
        files+=("$run_csv")
    done

    aggregate_median_csv "$aggregate_csv" "${files[@]}"
}

# ── Check binaries exist ───────────────────────────────────────────────────
for bin in "$SEQ_BIN" "$OMP_BIN" "$MPI_BIN" "$HYB_BIN" "$OCL_BIN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found. Run 'make all' first."
        exit 1
    fi
done

# ── 1. Sequential (single run — baseline) ─────────────────────────────────
echo ""
echo ">>> Sequential"
run_with_repeats "$DATA/sequential_results.csv" "$DATA/sequential_results.csv" \
    "$SEQ_BIN" "$IMAGE"

# ── 2. OpenMP sweep ────────────────────────────────────────────────────────
echo ""
echo ">>> OpenMP"
for t in "${THREAD_COUNTS[@]}"; do
    echo "  threads = $t"
    run_with_repeats "$DATA/openmp_results.csv" "$DATA/openmp_t${t}.csv" \
        "$OMP_BIN" "$IMAGE" "$t"
done

# ── 3. MPI sweep ───────────────────────────────────────────────────────────
echo ""
echo ">>> MPI"
for p in "${PROC_COUNTS[@]}"; do
    echo "  processes = $p"
    # shellcheck disable=SC2206
    local_mpi_args=( $(mpirun_args "$p") )
    run_with_repeats "$DATA/mpi_results.csv" "$DATA/mpi_p${p}.csv" \
        mpirun "${local_mpi_args[@]}" "$MPI_BIN" "$IMAGE"
done

# ── 4. Hybrid sweep ────────────────────────────────────────────────────────
echo ""
echo ">>> Hybrid (MPI × OpenMP)"
for p in "${HYB_PROC_COUNTS[@]}"; do
    for t in "${HYB_THREAD_COUNTS[@]}"; do
        echo "  processes=$p  threads=$t"
        # shellcheck disable=SC2206
        local_mpi_args=( $(mpirun_args "$p") )
        run_with_repeats "$DATA/hybrid_results.csv" "$DATA/hybrid_p${p}_t${t}.csv" \
            mpirun "${local_mpi_args[@]}" "$HYB_BIN" "$IMAGE" "$t"
    done
done

# ── 5. OpenCL sweep ────────────────────────────────────────────────────────
echo ""
echo ">>> OpenCL"
if run_with_repeats "$DATA/opencl_results.csv" "$DATA/opencl.csv" \
    "$OCL_BIN" "$IMAGE"; then
    :
else
    rm -f "$DATA/opencl.csv"
    if [[ "$OPENCL_REQUIRED" == "1" ]]; then
        echo "ERROR: OpenCL benchmark failed and OPENCL_REQUIRED=1"
        exit 1
    fi
    echo "WARNING: OpenCL benchmark failed; continuing without OpenCL rows."
fi

# ── 6. Merge all CSVs ──────────────────────────────────────────────────────
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
for csv in "$DATA"/openmp_t*.csv "$DATA"/mpi_p*.csv "$DATA"/hybrid_p*_t*.csv "$DATA"/opencl*.csv; do
    [[ -f "$csv" ]] || continue
    while IFS=',' read -r ver op w h th pr el sp; do
        [[ "$ver" == "version" ]] && continue
        base="${SEQ_TIME[$op]:-1}"
        speedup=$(awk -v b="$base" -v e="$el" 'BEGIN{if(e<=0){print 0}else{printf "%.4f", b/e}}')
        echo "$ver,$op,$w,$h,$th,$pr,$el,$speedup"
    done < "$csv"
done >> "$ALL"

echo "All results merged → $ALL"
echo ""
echo "Run 'make plot' to generate performance graphs."
