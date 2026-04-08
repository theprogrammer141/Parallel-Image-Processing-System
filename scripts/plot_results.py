#!/usr/bin/env python3
"""
plot_results.py — Generate performance analysis plots for the
Parallel Image Processing System benchmark results.

Reads:  results/data/all_results.csv
Writes: results/plots/
  - speedup_by_operation.png   (line chart per operation)
  - speedup_by_version.png     (grouped bar chart)
  - elapsed_time_heatmap.png   (heatmap: processes × threads for hybrid)
  - scalability_efficiency.png (parallel efficiency vs. workers)
  - summary_table.png          (table image)

Usage:  python3 scripts/plot_results.py [csv_path]
"""

import sys
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/data/all_results.csv"
OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "sequential": "#555555",
    "openmp": "#2196F3",
    "mpi": "#4CAF50",
    "hybrid": "#FF5722",
    "opencl": "#9C27B0",
}
OPERATIONS = ["grayscale", "gaussian_blur", "sobel_edge", "brightness", "histogram_eq"]
OP_LABELS = {
    "grayscale": "Grayscale",
    "gaussian_blur": "Gaussian Blur",
    "sobel_edge": "Sobel Edge",
    "brightness": "Brightness",
    "histogram_eq": "Histogram Eq.",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    }
)

# ─── Load Data ────────────────────────────────────────────────────────────────
DATA_DIR = Path("results/data")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load all_results.csv if it exists, otherwise merge individual CSVs."""
    p = Path(csv_path)
    if p.exists():
        return pd.read_csv(p)

    # Fall back: prefer full benchmark sweep outputs when present.
    seq_csv = DATA_DIR / "sequential_results.csv"
    sweep_parts = []
    sweep_parts.extend(sorted(DATA_DIR.glob("openmp_t*.csv")))
    sweep_parts.extend(sorted(DATA_DIR.glob("mpi_p*.csv")))
    sweep_parts.extend(sorted(DATA_DIR.glob("hybrid_p*_t*.csv")))

    if (DATA_DIR / "opencl.csv").exists():
        sweep_parts.append(DATA_DIR / "opencl.csv")
    elif (DATA_DIR / "opencl_results.csv").exists():
        sweep_parts.append(DATA_DIR / "opencl_results.csv")

    if sweep_parts:
        parts = [seq_csv] if seq_csv.exists() else []
        parts.extend(sweep_parts)
    else:
        # Final fallback: merge any direct per-version outputs.
        parts = [
            f for f in DATA_DIR.glob("*_results.csv") if f.name != "all_results.csv"
        ]

    if not parts:
        print("ERROR: No result CSVs found. Run 'make run' or 'make benchmark' first.")
        sys.exit(1)

    frames = [pd.read_csv(f) for f in parts]
    combined = pd.concat(frames, ignore_index=True)

    # Compute speedup relative to the best sequential baseline per operation.
    seq = (
        combined[combined["version"] == "sequential"]
        .groupby("operation", as_index=True)["elapsed_sec"]
        .min()
    )

    def add_speedup(row):
        base = seq.get(row["operation"], None)
        if row["version"] == "sequential":
            return 1.0
        if base is None or row["elapsed_sec"] <= 0:
            return row.get("speedup", 0)
        return round(base / row["elapsed_sec"], 4)

    combined["speedup"] = combined.apply(add_speedup, axis=1)
    combined = combined.sort_values(
        ["version", "operation", "processes", "threads"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Save merged file for future use
    out = DATA_DIR / "all_results.csv"
    combined.to_csv(out, index=False)
    print(f"Merged {len(parts)} CSV files → {out}")
    return combined


df = load_data(CSV_PATH)
df["workers"] = df["threads"] * df["processes"]
print(f"Loaded {len(df)} rows")
print("Rows by version:\n" + df.groupby("version")["operation"].count().to_string())

# ─── 1. Speedup Line Chart per Operation ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, op in enumerate(OPERATIONS):
    ax = axes[idx]
    sub = df[df["operation"] == op]

    # sequential baseline (workers=1)
    for ver in ["openmp", "mpi", "hybrid", "opencl"]:
        vdf = sub[sub["version"] == ver].copy()
        if vdf.empty:
            continue
        # Use best time for each worker count
        vdf = vdf.groupby("workers")["speedup"].max().reset_index()
        vdf = vdf.sort_values("workers")
        ax.plot(
            vdf["workers"],
            vdf["speedup"],
            marker="o",
            label=ver.capitalize(),
            color=COLORS[ver],
        )

    # Ideal speedup reference
    max_w = sub[sub["version"] != "sequential"]["workers"].max() if len(sub) > 1 else 8
    if pd.notna(max_w) and max_w > 0:
        ws = np.arange(1, int(max_w) + 1)
        ax.plot(ws, ws, "k--", alpha=0.4, linewidth=1, label="Ideal")

    ax.set_title(OP_LABELS.get(op, op))
    ax.set_xlabel("Total Workers (processes × threads)")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

# Hide unused subplot
for i in range(len(OPERATIONS), len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Speedup vs. Workers — All Operations", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = OUT_DIR / "speedup_by_operation.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─── 2. Grouped Bar Chart (best speedup per version per operation) ─────────────
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(OPERATIONS))
bar_w = 0.18
versions = ["openmp", "mpi", "hybrid", "opencl"]
offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]

for ver, offset in zip(versions, offsets):
    speedups = []
    for op in OPERATIONS:
        sub = df[(df["version"] == ver) & (df["operation"] == op)]
        speedups.append(sub["speedup"].max() if not sub.empty else 0)
    ax.bar(
        x + offset,
        speedups,
        bar_w,
        label=ver.capitalize(),
        color=COLORS[ver],
        alpha=0.85,
        edgecolor="white",
    )

ax.set_xticks(x)
ax.set_xticklabels([OP_LABELS[o] for o in OPERATIONS], rotation=15, ha="right")
ax.set_ylabel("Best Speedup (×)")
ax.set_title("Best Speedup per Version and Operation", fontweight="bold")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

out = OUT_DIR / "speedup_by_version.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─── 3. Hybrid Heatmap (elapsed time: processes × threads for Sobel) ──────────
hybrid = df[(df["version"] == "hybrid") & (df["operation"] == "sobel_edge")]
if not hybrid.empty:
    pivot = hybrid.pivot_table(
        index="processes", columns="threads", values="elapsed_sec", aggfunc="min"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("OMP Threads per Process")
    ax.set_ylabel("MPI Processes")
    ax.set_title(
        "Hybrid: Elapsed Time (s) — Sobel Edge\n(lower = faster)", fontweight="bold"
    )
    fig.colorbar(im, ax=ax, label="Elapsed (s)")
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
    out = OUT_DIR / "elapsed_time_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ─── 4. Parallel Efficiency ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for ver in ["openmp", "mpi", "hybrid", "opencl"]:
    sub = df[(df["version"] == ver)].copy()
    if sub.empty:
        continue
    eff = sub.groupby("workers", as_index=False)["speedup"].max()
    eff["efficiency"] = (eff["speedup"] / eff["workers"]) * 100
    eff = eff[["workers", "efficiency"]]
    eff = eff.sort_values("workers")
    ax.plot(
        eff["workers"],
        eff["efficiency"],
        marker="s",
        label=ver.capitalize(),
        color=COLORS[ver],
    )

ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="100% (ideal)")
ax.set_xlabel("Total Workers (processes × threads)")
ax.set_ylabel("Parallel Efficiency (%)")
ax.set_title("Parallel Efficiency vs. Workers", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=1)
ax.set_ylim(0, 120)

out = OUT_DIR / "scalability_efficiency.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─── 5. Summary Table ─────────────────────────────────────────────────────────
summary_rows = []
for op in OPERATIONS:
    row = {"Operation": OP_LABELS.get(op, op)}
    for ver in ["sequential", "openmp", "mpi", "hybrid", "opencl"]:
        sub = df[(df["version"] == ver) & (df["operation"] == op)]
        if not sub.empty:
            t = sub["elapsed_sec"].min()
            sp = sub["speedup"].max()
            row[ver.capitalize()] = f"{t:.4f}s\n(×{sp:.2f})"
        else:
            row[ver.capitalize()] = "—"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
cols = ["Operation", "Sequential", "Openmp", "Mpi", "Hybrid", "Opencl"]
cols = [c for c in cols if c in summary_df.columns]
summary_df = summary_df[cols]

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 2.0)

# Header colour
for j in range(len(summary_df.columns)):
    tbl[0, j].set_facecolor("#263238")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title(
    "Performance Summary — All Versions & Operations",
    fontweight="bold",
    pad=10,
    fontsize=13,
)

out = OUT_DIR / "summary_table.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nAll plots saved to results/plots/")
