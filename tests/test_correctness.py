#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownReturnType=false
"""Correctness regression tests for all image-processing backends.

This script:
1) Builds a deterministic small test image.
2) Runs each backend executable.
3) Compares each output image against sequential output with tolerances.

OpenCL execution is optional because some machines do not expose an OpenCL runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
RESULTS_IMG = ROOT / "results" / "images"
TEST_INPUT = ROOT / "test_images" / "ci_small.png"

OPS = ["grayscale", "gaussian_blur", "sobel_edge", "brightness", "histogram_eq"]

BASE_TOL = {
    "grayscale": 1,
    "gaussian_blur": 3,
    "sobel_edge": 4,
    "brightness": 1,
    "histogram_eq": 1,
}

OPENCL_TOL = {
    "grayscale": 2,
    "gaussian_blur": 4,
    "sobel_edge": 6,
    "brightness": 2,
    "histogram_eq": 4,
}


def run_cmd(cmd: list[str], label: str, required: bool = True) -> bool:
    print(f"\n==> {label}")
    print("$", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())

    if proc.returncode != 0:
        msg = f"{label} failed with exit code {proc.returncode}"
        if required:
            raise RuntimeError(msg)
        print(f"[optional] {msg}")
        return False
    return True


def generate_input_image() -> None:
    TEST_INPUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(TEST_INPUT), img)
    if not ok:
        raise RuntimeError(f"Failed to write test input image: {TEST_INPUT}")


def output_path(version: str, op: str) -> Path:
    if version == "sequential":
        return RESULTS_IMG / f"seq_{op}.png"
    if version == "openmp":
        return RESULTS_IMG / f"omp_{op}_t2.png"
    if version == "mpi":
        return RESULTS_IMG / f"mpi_{op}_p2.png"
    if version == "hybrid":
        return RESULTS_IMG / f"hybrid_{op}_p2_t2.png"
    if version == "opencl":
        return RESULTS_IMG / f"ocl_{op}.png"
    raise ValueError(f"Unknown version: {version}")


def cleanup_expected_outputs() -> None:
    versions = ["sequential", "openmp", "mpi", "hybrid", "opencl"]
    for ver in versions:
        for op in OPS:
            p = output_path(ver, op)
            if p.exists():
                p.unlink()


def load_image(path: Path) -> Any:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def max_abs_diff(a: Any, b: Any) -> int:
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    if diff.size == 0:
        return 0
    return int(diff.max())


def ensure_required_binaries() -> None:
    required = ["sequential", "openmp_proc", "mpi_proc", "hybrid_proc"]
    for name in required:
        p = BUILD_DIR / name
        if not p.exists():
            raise RuntimeError(f"Missing binary: {p}. Run 'make all' first.")


def main() -> int:
    if shutil.which("mpirun") is None:
        print("ERROR: mpirun not found in PATH")
        return 1

    try:
        ensure_required_binaries()
        RESULTS_IMG.mkdir(parents=True, exist_ok=True)
        generate_input_image()
        cleanup_expected_outputs()

        run_cmd([str(BUILD_DIR / "sequential"), str(TEST_INPUT)], "Sequential", required=True)
        run_cmd([str(BUILD_DIR / "openmp_proc"), str(TEST_INPUT), "2"], "OpenMP (2 threads)", required=True)
        run_cmd(
            ["mpirun", "--oversubscribe", "-np", "2", str(BUILD_DIR / "mpi_proc"), str(TEST_INPUT)],
            "MPI (2 processes)",
            required=True,
        )
        run_cmd(
            [
                "mpirun",
                "--oversubscribe",
                "-np",
                "2",
                str(BUILD_DIR / "hybrid_proc"),
                str(TEST_INPUT),
                "2",
            ],
            "Hybrid (2 processes x 2 threads)",
            required=True,
        )

        ran_opencl = False
        opencl_bin = BUILD_DIR / "opencl_proc"
        if opencl_bin.exists():
            ran_opencl = run_cmd([str(opencl_bin), str(TEST_INPUT)], "OpenCL", required=False)
        else:
            print("[optional] OpenCL binary not found; skipping OpenCL checks")

        failures: list[str] = []

        for op in OPS:
            ref_path = output_path("sequential", op)
            ref = load_image(ref_path)

            for version in ["openmp", "mpi", "hybrid"]:
                cand_path = output_path(version, op)
                if not cand_path.exists():
                    failures.append(f"{version}:{op} missing output {cand_path}")
                    continue

                cand = load_image(cand_path)
                try:
                    diff = max_abs_diff(ref, cand)
                except RuntimeError as e:
                    failures.append(f"{version}:{op} {e}")
                    continue

                tol = BASE_TOL[op]
                print(f"[check] {version:7s} {op:13s} max_abs_diff={diff} tol={tol}")
                if diff > tol:
                    failures.append(
                        f"{version}:{op} exceeded tolerance (diff={diff}, tol={tol})"
                    )

            if ran_opencl:
                cand_path = output_path("opencl", op)
                if not cand_path.exists():
                    failures.append(f"opencl:{op} missing output {cand_path}")
                    continue

                cand = load_image(cand_path)
                try:
                    diff = max_abs_diff(ref, cand)
                except RuntimeError as e:
                    failures.append(f"opencl:{op} {e}")
                    continue

                tol = OPENCL_TOL[op]
                print(f"[check] opencl  {op:13s} max_abs_diff={diff} tol={tol}")
                if diff > tol:
                    failures.append(
                        f"opencl:{op} exceeded tolerance (diff={diff}, tol={tol})"
                    )

        if failures:
            print("\nCorrectness test failures:")
            for f in failures:
                print(" -", f)
            return 1

        print("\nAll correctness checks passed.")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
