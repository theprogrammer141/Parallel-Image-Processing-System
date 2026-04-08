#!/usr/bin/env python3
"""
GUI controller for the Parallel Image Processing System.

Features:
- Select input image
- Choose versions to run (sequential, openmp, mpi, hybrid, opencl)
- Configure OpenMP threads and MPI processes
- Build binaries from GUI
- Run selected versions and stream logs
- Show timing table per operation
- Side-by-side preview with click-to-zoom
- Hover pixel inspector (with Pillow)
- Generate performance plots
- Open output folders

This script uses only Python standard library + tkinter.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

PILImage: Any = None
PILImageTk: Any = None
PIL_LANCZOS: Any = None
try:
    from PIL import Image as _PILImage  # type: ignore[import-not-found]
    from PIL import ImageTk as _PILImageTk  # type: ignore[import-not-found]

    PILImage = _PILImage
    PILImageTk = _PILImageTk
    PIL_LANCZOS = getattr(_PILImage, "Resampling", _PILImage).LANCZOS
    _pil_available = True
except Exception:
    _pil_available = False

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
IMAGES_DIR = RESULTS_DIR / "images"
PLOTS_DIR = RESULTS_DIR / "plots"

VERSIONS = ["sequential", "openmp", "mpi", "hybrid", "opencl"]
OPERATIONS = ["grayscale", "gaussian_blur", "sobel_edge", "brightness", "histogram_eq"]
PREFIX_BY_VERSION = {
    "sequential": "seq_",
    "openmp": "omp_",
    "mpi": "mpi_",
    "hybrid": "hybrid_",
    "opencl": "ocl_",
}

TIME_LINE_RE = re.compile(
    r"^\s*(grayscale|gaussian_blur|sobel_edge|brightness|histogram_eq)\s+.*?([0-9]+\.[0-9]+)\s+s\s*$",
    re.IGNORECASE,
)


class Tooltip:
    """Hover tooltip for tkinter widgets."""
    def __init__(self, widget: tk.Widget, text: str, delay: int = 1000) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow: tk.Toplevel | None = None
        self.id: str | None = None
        self.x = self.y = 0
        
        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _event: tk.Event) -> None:
        if self.tipwindow or not self.text:
            return
        self.id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, _event: tk.Event) -> None:
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        self._hide_tooltip()

    def _show_tooltip(self) -> None:
        if self.tipwindow:
            return
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.configure(bg="#2d2d2d", relief=tk.SOLID, borderwidth=1)
        
        label = tk.Label(
            tw,
            text=self.text,
            bg="#2d2d2d",
            fg="#e0e0e0",
            wraplength=300,
            justify=tk.LEFT,
            font=("Segoe UI", 8),
            padx=8,
            pady=4
        )
        label.pack()
        
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        tw.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ImageProcessingGUI(tk.Tk):
    # Enhanced Color Theme - Professional Dark Mode
    BG_DARK = "#0f1419"
    BG_DARKER = "#0a0e13"
    BG_PANEL = "#1a1f2e"
    BG_SECTION = "#252d3d"
    BG_LIGHT = "#323d52"
    BG_ACCENT = "#1d293d"

    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a8b5c9"
    TEXT_MUTED = "#6b7a8f"

    ACCENT_BLUE = "#0d88ff"
    ACCENT_CYAN = "#00d4ff"
    ACCENT_PURPLE = "#9d47ff"
    ACCENT_GREEN = "#1de9b6"
    ACCENT_ORANGE = "#ff9500"

    SUCCESS = "#00cc88"
    WARNING = "#ff9500"
    ERROR = "#ff3333"

    # Border and shadow colors
    BORDER = "#3a4556"
    SHADOW = "#000000"

    def __init__(self) -> None:
        super().__init__()
        self.title("🚀 Parallel Image Processing System")
        self.geometry("1400x900")
        self.minsize(1100, 750)

        # Configure dark theme
        self.configure(bg=self.BG_DARK)

        # Configure ttk style for dark theme
        self._setup_styles()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False
        self.results: dict[str, dict[str, float]] = {v: {} for v in VERSIONS}
        self._animation_id: int | None = None
        self._pulse_state = 0.0

        self.image_var = tk.StringVar(value=str(ROOT / "test_images" / "sample.jpg"))
        self.omp_threads_var = tk.StringVar(value="4")
        self.mpi_procs_var = tk.StringVar(value="4")
        self.hybrid_threads_var = tk.StringVar(value="4")

        self.version_vars = {v: tk.BooleanVar(value=True) for v in VERSIONS}
        self.preview_version_var = tk.StringVar(value="sequential")
        self.preview_operation_var = tk.StringVar(value="grayscale")
        self._orig_photo: tk.PhotoImage | object | None = None
        self._proc_photo: tk.PhotoImage | object | None = None
        self._preview_meta: dict[str, dict[str, Any] | None] = {"original": None, "processed": None}
        self._zoom_state: dict[str, dict[str, Any]] = {}
        self.original_pixel_var = tk.StringVar(value="x:- y:- value:-")
        self.processed_pixel_var = tk.StringVar(value="x:- y:- value:-")

        self._build_ui()
        self._pump_log_queue()
        self._animate()
        self.after(250, self._refresh_previews)

    def _setup_styles(self) -> None:
        """Configure ttk style for professional dark theme with advanced styling."""
        style = ttk.Style()
        style.theme_use("clam")

        # ===== FRAME STYLES =====
        style.configure("TFrame", background=self.BG_DARK, relief="flat", borderwidth=0)
        style.configure("Panel.TFrame", background=self.BG_PANEL, borderwidth=1, relief="solid")
        style.configure("Section.TFrame", background=self.BG_SECTION, borderwidth=0)

        # ===== LABEL STYLES =====
        style.configure("TLabel", background=self.BG_DARK, foreground=self.TEXT_PRIMARY, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.BG_DARK, foreground=self.ACCENT_CYAN, font=("Segoe UI", 14, "bold"))
        style.configure("Heading.TLabel", background=self.BG_DARK, foreground=self.ACCENT_CYAN, font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background=self.BG_DARK, foreground=self.TEXT_MUTED, font=("Segoe UI", 8))

        # ===== LABELFRAME STYLES =====
        style.configure("TLabelFrame", background=self.BG_PANEL, foreground=self.ACCENT_CYAN, relief="solid", borderwidth=1, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelFrame.Label", background=self.BG_PANEL, foreground=self.ACCENT_CYAN)

        # ===== BUTTON STYLES =====
        # Primary buttons (blue)
        style.configure(
            "TButton",
            background=self.ACCENT_BLUE,
            foreground=self.TEXT_PRIMARY,
            relief="flat",
            padding=10,
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "TButton",
            background=[
                ("pressed", self.ACCENT_PURPLE),
                ("active", self.ACCENT_CYAN),
                ("disabled", self.BG_LIGHT)
            ],
            foreground=[("disabled", self.TEXT_MUTED)]
        )

        # Accent buttons (cyan/green)
        style.configure(
            "Accent.TButton",
            background=self.ACCENT_GREEN,
            foreground=self.BG_DARK,
            relief="flat",
            padding=10,
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "Accent.TButton",
            background=[
                ("pressed", self.ACCENT_CYAN),
                ("active", self.SUCCESS),
                ("disabled", self.BG_LIGHT)
            ]
        )

        # Secondary buttons (outline style)
        style.configure(
            "Secondary.TButton",
            background=self.BG_LIGHT,
            foreground=self.ACCENT_BLUE,
            relief="solid",
            padding=8,
            font=("Segoe UI", 9),
            borderwidth=1
        )
        style.map(
            "Secondary.TButton",
            background=[
                ("pressed", self.ACCENT_BLUE),
                ("active", self.BG_SECTION)
            ],
            foreground=[
                ("pressed", self.TEXT_PRIMARY),
                ("active", self.ACCENT_CYAN)
            ]
        )

        # ===== CHECKBOX & RADIO STYLES =====
        style.configure("TCheckbutton", background=self.BG_DARK, foreground=self.TEXT_PRIMARY, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", self.BG_SECTION)])

        # ===== INPUT STYLES =====
        style.configure(
            "TEntry",
            fieldbackground=self.BG_LIGHT,
            foreground=self.TEXT_PRIMARY,
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 10),
            padding=6
        )
        style.map("TEntry", fieldbackground=[("focus", self.BG_ACCENT)])

        style.configure(
            "TCombobox",
            fieldbackground=self.BG_LIGHT,
            foreground=self.TEXT_PRIMARY,
            arrowcolor=self.ACCENT_CYAN,
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 10),
            padding=6
        )
        style.map("TCombobox", fieldbackground=[("focus", self.BG_ACCENT)])

        # ===== TREEVIEW STYLES =====
        style.configure(
            "Treeview",
            background=self.BG_PANEL,
            foreground=self.TEXT_PRIMARY,
            fieldbackground=self.BG_PANEL,
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 9),
            rowheight=24
        )
        style.configure(
            "Treeview.Heading",
            background=self.BG_SECTION,
            foreground=self.ACCENT_CYAN,
            borderwidth=1,
            relief="raised",
            font=("Segoe UI", 9, "bold"),
            padding=6
        )
        style.map("Treeview", background=[("selected", self.BG_ACCENT)])
        style.map("Treeview.Heading", background=[("active", self.BG_LIGHT)])

        # ===== PROGRESS BAR STYLES =====
        style.configure(
            "Horizontal.TProgressbar",
            background=self.ACCENT_GREEN,
            troughcolor=self.BG_LIGHT,
            borderwidth=1,
            relief="solid",
            height=8
        )

        # ===== SCROLLBAR STYLES =====
        style.configure(
            "Horizontal.TScrollbar",
            background=self.BG_LIGHT,
            troughcolor=self.BG_PANEL,
            borderwidth=1
        )
        style.configure(
            "Vertical.TScrollbar",
            background=self.BG_LIGHT,
            troughcolor=self.BG_PANEL,
            borderwidth=1
        )

    def _animate(self) -> None:
        """Pulse animation for progress bar effect."""
        if self.is_running:
            self._pulse_state = (self._pulse_state + 0.05) % 1.0
        self._animation_id = self.after(50, self._animate)

    def _build_ui(self) -> None:
        """Build the professional UI with enhanced layout and tooltips."""
        main = ttk.Frame(self, padding=0)
        main.pack(fill=tk.BOTH, expand=True)

        # ===== HEADER SECTION =====
        header = tk.Canvas(main, height=70, bg=self.BG_DARKER, highlightthickness=0)
        header.pack(fill=tk.X)

        title_text = tk.Label(
            header,
            text="🚀 Parallel Image Processing System",
            font=("Segoe UI", 16, "bold"),
            bg=self.BG_DARKER,
            fg=self.ACCENT_CYAN
        )
        title_text.pack(pady=12)

        # Divider line
        divider1 = tk.Canvas(main, height=1, bg=self.BORDER, highlightthickness=0)
        divider1.pack(fill=tk.X)

        # ===== MAIN SCROLLABLE AREA =====
        main_frame = ttk.Frame(main, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== CONFIGURATION SECTION =====
        config_section = tk.Frame(main_frame, bg=self.BG_SECTION, borderwidth=0)
        config_section.pack(fill=tk.X, pady=(0, 12))

        config_title = tk.Label(
            config_section,
            text="⚙️ Input Configuration",
            font=("Segoe UI", 11, "bold"),
            bg=self.BG_SECTION,
            fg=self.ACCENT_CYAN,
            pady=8,
            padx=12
        )
        config_title.pack(fill=tk.X)

        config_content = tk.Frame(config_section, bg=self.BG_PANEL, borderwidth=1, relief=tk.SOLID)
        config_content.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Image input row
        img_row = ttk.Frame(config_content)
        img_row.pack(fill=tk.X, padx=12, pady=12)

        img_label = tk.Label(img_row, text="Input Image:", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        img_label.pack(side=tk.LEFT, padx=(0, 8))

        img_entry = ttk.Entry(img_row, textvariable=self.image_var, width=70)
        img_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        browse_btn = ttk.Button(img_row, text="📂 Browse", command=self._browse_image, style="Accent.TButton")
        browse_btn.pack(side=tk.LEFT)
        Tooltip(browse_btn, "Select an input image file\n(supports JPG, PNG, BMP, TIFF, PPM)")

        # Parameters row
        param_row = ttk.Frame(config_content)
        param_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        param_configs = [
            ("Threads", self.omp_threads_var, "🔷", "OpenMP thread count"),
            ("Processes", self.mpi_procs_var, "🔶", "MPI process count"),
            ("Hybrid", self.hybrid_threads_var, "🔹", "Hybrid mode threads"),
        ]

        for i, (label, var, icon, tooltip_text) in enumerate(param_configs):
            if i > 0:
                sep = tk.Label(param_row, text="│", bg=self.BG_PANEL, fg=self.BORDER, font=("Segoe UI", 10))
                sep.pack(side=tk.LEFT, padx=12)

            lbl = tk.Label(param_row, text=f"{icon} {label}:", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY, font=("Segoe UI", 9))
            lbl.pack(side=tk.LEFT, padx=(0, 6))

            entry = ttk.Entry(param_row, textvariable=var, width=5)
            entry.pack(side=tk.LEFT)
            Tooltip(entry, tooltip_text)

        # ===== VERSIONS SECTION =====
        versions_section = tk.Frame(main_frame, bg=self.BG_SECTION, borderwidth=0)
        versions_section.pack(fill=tk.X, pady=(0, 12))

        versions_title = tk.Label(
            versions_section,
            text="✨ Processing Versions",
            font=("Segoe UI", 11, "bold"),
            bg=self.BG_SECTION,
            fg=self.ACCENT_GREEN,
            pady=8,
            padx=12
        )
        versions_title.pack(fill=tk.X)

        versions_content = tk.Frame(versions_section, bg=self.BG_PANEL, borderwidth=1, relief=tk.SOLID)
        versions_content.pack(fill=tk.X, padx=8, pady=(0, 8))

        version_configs = [
            ("sequential", "⚪", "Single-threaded baseline"),
            ("openmp", "🔵", "Shared-memory parallelism"),
            ("mpi", "🟣", "Distributed memory"),
            ("hybrid", "🟠", "MPI + OpenMP combined"),
            ("opencl", "💠", "GPU/heterogeneous compute"),
        ]

        for idx, (vname, icon, tooltip_text) in enumerate(version_configs):
            cb = ttk.Checkbutton(
                versions_content,
                text=f"{icon} {vname.capitalize()}",
                variable=self.version_vars[vname],
                style="TCheckbutton"
            )
            cb.pack(side=tk.LEFT, padx=12, pady=8)
            Tooltip(cb, tooltip_text)

        # ===== CONTROL PANEL SECTION =====
        control_section = tk.Frame(main_frame, bg=self.BG_SECTION, borderwidth=0)
        control_section.pack(fill=tk.X, pady=(0, 12))

        control_title = tk.Label(
            control_section,
            text="🎮 Control Panel",
            font=("Segoe UI", 11, "bold"),
            bg=self.BG_SECTION,
            fg=self.ACCENT_ORANGE,
            pady=8,
            padx=12
        )
        control_title.pack(fill=tk.X)

        control_content = tk.Frame(control_section, bg=self.BG_PANEL, borderwidth=1, relief=tk.SOLID)
        control_content.pack(fill=tk.X, padx=8, pady=(0, 8))

        btns_main = ttk.Frame(control_content)
        btns_main.pack(fill=tk.X, padx=12, pady=12)

        self.btn_build = ttk.Button(btns_main, text="🔨 Build All", command=self._start_build_all)
        self.btn_build.pack(side=tk.LEFT, padx=(0, 6))
        Tooltip(self.btn_build, "Compile all processing executables")

        self.btn_run_selected = ttk.Button(btns_main, text="▶️ Run Selected", command=self._start_run_selected, style="Secondary.TButton")
        self.btn_run_selected.pack(side=tk.LEFT, padx=(0, 6))
        Tooltip(self.btn_run_selected, "Execute only checked versions")

        self.btn_run_all = ttk.Button(btns_main, text="▶️▶️ Run All 5", command=self._start_run_all)
        self.btn_run_all.pack(side=tk.LEFT, padx=(0, 6))
        Tooltip(self.btn_run_all, "Execute all processing versions")

        self.btn_plot = ttk.Button(btns_main, text="📊 Generate Plots", command=self._start_plot, style="Accent.TButton")
        self.btn_plot.pack(side=tk.LEFT)
        Tooltip(self.btn_plot, "Create performance comparison charts")

        btns_secondary = ttk.Frame(control_content)
        btns_secondary.pack(fill=tk.X, padx=12, pady=(0, 12))

        ttk.Button(btns_secondary, text="📁 Open Images", command=lambda: self._open_folder(IMAGES_DIR), style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns_secondary, text="📈 Open Plots", command=lambda: self._open_folder(PLOTS_DIR), style="Secondary.TButton").pack(side=tk.LEFT)

        # ===== STATUS AND PROGRESS =====
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        status_label = tk.Label(status_frame, text="Status:", bg=self.BG_DARK, fg=self.TEXT_MUTED, font=("Segoe UI", 9))
        status_label.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready ✓")
        status_display = tk.Label(status_frame, textvariable=self.status_var, bg=self.BG_DARK, fg=self.SUCCESS, font=("Segoe UI", 10, "bold"))
        status_display.pack(side=tk.LEFT, padx=(8, 0))

        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X)

        # Divider line
        divider2 = tk.Canvas(main_frame, height=1, bg=self.BORDER, highlightthickness=0)
        divider2.pack(fill=tk.X, pady=12)

        # ===== MAIN CONTENT AREA =====
        body = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        # ===== EXECUTION LOG =====
        logs_frame = ttk.LabelFrame(left, text="📝 Execution Log", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            logs_frame,
            height=16,
            wrap="word",
            bg=self.BG_PANEL,
            fg=self.TEXT_PRIMARY,
            font=("Courier New", 9),
            insertbackground=self.ACCENT_CYAN,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(logs_frame, orient="vertical", command=lambda *a: self.log_text.yview(*a))
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.configure(state="disabled")

        self.log_text.tag_configure("success", foreground=self.SUCCESS, font=("Courier New", 9, "bold"))
        self.log_text.tag_configure("error", foreground=self.ERROR, font=("Courier New", 9, "bold"))
        self.log_text.tag_configure("warning", foreground=self.WARNING)
        self.log_text.tag_configure("info", foreground=self.ACCENT_CYAN)

        # ===== RIGHT PANED AREA (TABLE + PREVIEW) =====
        right_paned = ttk.Panedwindow(right, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # Timings table
        table_frame = ttk.LabelFrame(right_paned, text="⏱️ Operation Timings (seconds)", padding=10)
        right_paned.add(table_frame, weight=1)

        cols = ["operation"] + VERSIONS
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, anchor="center", width=115 if c != "operation" else 130)

        for op in OPERATIONS:
            self.table.insert("", tk.END, values=[op] + ["-" for _ in VERSIONS])

        self.table.pack(fill=tk.BOTH, expand=True)

        # Image preview
        preview_frame = ttk.LabelFrame(right_paned, text="🖼️ Image Preview", padding=10)
        right_paned.add(preview_frame, weight=5)

        control_row = ttk.Frame(preview_frame)
        control_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(control_row, text="Version:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ver_box = ttk.Combobox(
            control_row,
            textvariable=self.preview_version_var,
            values=VERSIONS,
            state="readonly",
            width=11,
        )
        ver_box.pack(side=tk.LEFT, padx=(0, 14))
        ver_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_previews())

        ttk.Label(control_row, text="Op:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        op_box = ttk.Combobox(
            control_row,
            textvariable=self.preview_operation_var,
            values=OPERATIONS,
            state="readonly",
            width=12,
        )
        op_box.pack(side=tk.LEFT, padx=(0, 14))
        op_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_previews())

        ttk.Button(control_row, text="🔄 Refresh", command=self._refresh_previews, style="Secondary.TButton").pack(side=tk.LEFT)

        panes = ttk.Frame(preview_frame)
        panes.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        panes.grid_columnconfigure(0, weight=1, uniform="preview")
        panes.grid_columnconfigure(1, weight=1, uniform="preview")
        panes.grid_rowconfigure(0, weight=1)

        left_preview = ttk.LabelFrame(panes, text="Original 🔍", padding=6)
        left_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        right_preview = ttk.LabelFrame(panes, text="Processed 🎨", padding=6)
        right_preview.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.original_label = tk.Label(
            left_preview,
            text="No image",
            anchor="center",
            bg=self.BG_PANEL,
            fg=self.TEXT_SECONDARY,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.original_path_label = tk.Label(
            left_preview,
            text="",
            bg=self.BG_DARK,
            fg=self.TEXT_MUTED,
            anchor="center",
            font=("Segoe UI", 7)
        )
        self.original_path_label.pack(fill=tk.X, padx=2, pady=(0, 1))
        ttk.Label(
            left_preview,
            textvariable=self.original_pixel_var,
            foreground=self.ACCENT_BLUE,
            anchor="center",
            font=("Segoe UI", 8),
            style="Muted.TLabel"
        ).pack(fill=tk.X, padx=2, pady=(0, 2))

        self.processed_label = tk.Label(
            right_preview,
            text="No output",
            anchor="center",
            bg=self.BG_PANEL,
            fg=self.TEXT_SECONDARY,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.processed_path_label = tk.Label(
            right_preview,
            text="",
            bg=self.BG_DARK,
            fg=self.TEXT_MUTED,
            anchor="center",
            font=("Segoe UI", 7)
        )
        self.processed_path_label.pack(fill=tk.X, padx=2, pady=(0, 1))
        ttk.Label(
            right_preview,
            textvariable=self.processed_pixel_var,
            foreground=self.ACCENT_GREEN,
            anchor="center",
            font=("Segoe UI", 8),
            style="Muted.TLabel"
        ).pack(fill=tk.X, padx=2, pady=(0, 2))

        self.original_label.bind("<Button-1>", lambda _e: self._open_zoom("original"))
        self.processed_label.bind("<Button-1>", lambda _e: self._open_zoom("processed"))
        self.original_label.bind("<Motion>", lambda e: self._on_preview_motion("original", e))
        self.processed_label.bind("<Motion>", lambda e: self._on_preview_motion("processed", e))
        self.original_label.bind("<Leave>", lambda _e: self.original_pixel_var.set("x:- y:- value:-"))
        self.processed_label.bind("<Leave>", lambda _e: self.processed_pixel_var.set("x:- y:- value:-"))

    def _append_log(self, text: str, tag: str = "info") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _pump_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._pump_log_queue)

    def _browse_image(self) -> None:
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.ppm"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self.image_var.set(p)
            self._refresh_previews()

    def _load_preview_photo(self, path: Path, max_w: int = 650, max_h: int = 520) -> tuple[Any, dict[str, Any]]:
        if not path.exists() or not path.is_file():
            return None, {}

        if _pil_available and PILImage is not None and PILImageTk is not None:
            try:
                full = PILImage.open(path).convert("RGB")
                disp = full.copy()
                disp.thumbnail((max_w, max_h), PIL_LANCZOS)
                photo = PILImageTk.PhotoImage(disp)
                meta = {
                    "path": path,
                    "backend": "pil",
                    "orig_w": full.width,
                    "orig_h": full.height,
                    "disp_w": disp.width,
                    "disp_h": disp.height,
                    "full_img": full,
                }
                return photo, meta
            except Exception:
                return None, {}

        if path.suffix.lower() not in {".png", ".gif", ".ppm", ".pgm"}:
            return None, {}

        try:
            source = tk.PhotoImage(file=str(path))
            w, h = source.width(), source.height()
            scale = max(w / max_w, h / max_h, 1)
            factor = int(math.ceil(scale))
            photo = source
            if factor > 1:
                photo = source.subsample(factor, factor)
            meta = {
                "path": path,
                "backend": "tk",
                "orig_w": w,
                "orig_h": h,
                "disp_w": photo.width(),
                "disp_h": photo.height(),
                "display_img": photo,
            }
            return photo, meta
        except Exception:
            return None, {}

    def _pixel_text(self, panel: str, px: int, py: int, rgb: tuple[int, ...]) -> str:
        return f"{panel} x:{px} y:{py} value:({rgb[0]}, {rgb[1]}, {rgb[2]})"

    def _on_preview_motion(self, panel: str, event: tk.Event) -> None:
        meta = self._preview_meta.get(panel)
        if not meta:
            return

        widget = self.original_label if panel == "original" else self.processed_label
        disp_w = int(meta.get("disp_w", 0))
        disp_h = int(meta.get("disp_h", 0))
        if disp_w <= 0 or disp_h <= 0:
            return

        ww = widget.winfo_width()
        wh = widget.winfo_height()
        ox = max((ww - disp_w) // 2, 0)
        oy = max((wh - disp_h) // 2, 0)

        if event.x < ox or event.y < oy or event.x >= ox + disp_w or event.y >= oy + disp_h:
            return

        local_x = event.x - ox
        local_y = event.y - oy
        orig_w = int(meta.get("orig_w", disp_w))
        orig_h = int(meta.get("orig_h", disp_h))
        px = min(max(int(local_x * orig_w / disp_w), 0), orig_w - 1)
        py = min(max(int(local_y * orig_h / disp_h), 0), orig_h - 1)

        rgb = None
        if meta.get("backend") == "pil" and meta.get("full_img") is not None:
            img = meta["full_img"]
            rgb = img.getpixel((px, py))
            if not isinstance(rgb, tuple):
                rgb = (int(rgb), int(rgb), int(rgb))
            elif len(rgb) > 3:
                rgb = rgb[:3]
        elif meta.get("backend") == "tk" and meta.get("display_img") is not None:
            timg = meta["display_img"]
            sx = min(max(int(local_x), 0), disp_w - 1)
            sy = min(max(int(local_y), 0), disp_h - 1)
            val = timg.get(sx, sy)
            if isinstance(val, tuple):
                rgb = tuple(int(v) for v in val[:3])

        if rgb is None:
            return

        if panel == "original":
            self.original_pixel_var.set(self._pixel_text(panel, px, py, rgb))
        else:
            self.processed_pixel_var.set(self._pixel_text(panel, px, py, rgb))

    def _render_zoom(self, panel: str) -> None:
        state = self._zoom_state.get(panel)
        if not state:
            return
        if not _pil_available or PILImageTk is None:
            return
        base = state.get("base_img")
        if base is None:
            return

        scale = float(state.get("scale", 1.0))
        new_w = max(int(base.width * scale), 1)
        new_h = max(int(base.height * scale), 1)
        try:
            resized = base.resize((new_w, new_h), PIL_LANCZOS)
            photo = PILImageTk.PhotoImage(resized)
            state["photo"] = photo
            lbl = state["label"]
            lbl.configure(image=photo, text="")
            state["info"].set(f"{base.width}×{base.height}  |  zoom: {scale:.2f}x  |  display: {new_w}×{new_h}")
        except Exception as e:
            state["info"].set(f"Error rendering zoom: {e}")

    def _open_zoom(self, panel: str) -> None:
        meta = self._preview_meta.get(panel)
        if not meta:
            messagebox.showinfo("Zoom", "No image available for zoom.")
            return
        if not _pil_available or PILImage is None:
            messagebox.showinfo("Zoom", "Install Pillow to use zoom view.")
            return

        path = meta.get("path")
        if not path:
            return
        path = Path(path)

        existing = self._zoom_state.get(panel)
        if existing and existing.get("win") and existing["win"].winfo_exists():
            if existing.get("path") == path:
                existing["win"].deiconify()
                existing["win"].lift()
                return

        win = tk.Toplevel(self)
        title_icon = "🔍" if panel == "original" else "🎨"
        win.title(f"{title_icon} Zoom - {panel.capitalize()}")
        win.geometry("950x750")
        win.configure(bg=self.BG_DARK)

        # Toolbar with buttons
        toolbar = ttk.Frame(win, padding=8)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="🔍− Zoom Out", command=lambda: self._zoom_adjust(panel, 0.8)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="🔍+ Zoom In", command=lambda: self._zoom_adjust(panel, 1.25)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="🎯 Fit", command=lambda: self._zoom_reset(panel)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="💾 Save", command=lambda: self._save_zoom_image(panel, path)).pack(side=tk.LEFT)

        info_var = tk.StringVar(value="Loading...")
        info_label = ttk.Label(toolbar, textvariable=info_var, foreground=self.ACCENT_CYAN)
        info_label.pack(side=tk.LEFT, padx=(20, 0))

        # Canvas with image
        canvas_frame = ttk.Frame(win, padding=4)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            canvas_frame,
            background=self.BG_PANEL,
            highlightthickness=1,
            highlightbackground=self.BG_LIGHT
        )
        canvas.pack(fill=tk.BOTH, expand=True)

        # Image label inside canvas
        img_label = tk.Label(
            canvas,
            bg=self.BG_PANEL,
            fg=self.TEXT_SECONDARY
        )
        window_id = canvas.create_window(0, 0, anchor="nw", window=img_label)

        # Load base image
        try:
            base_img = PILImage.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Zoom", f"Failed to load image: {e}")
            win.destroy()
            return

        # Store state
        self._zoom_state[panel] = {
            "win": win,
            "label": img_label,
            "canvas": canvas,
            "window_id": window_id,
            "base_img": base_img,
            "scale": 1.0,
            "photo": None,
            "info": info_var,
            "path": path,
        }

        # Canvas resize handler
        def on_configure(_e: tk.Event) -> None:
            st = self._zoom_state.get(panel)
            if not st:
                return
            c = st.get("canvas")
            if not c:
                return
            try:
                w = max(c.winfo_width(), 1)
                h = max(c.winfo_height(), 1)
                c.coords(st["window_id"], w // 2, h // 2)
            except Exception:
                pass

        canvas.bind("<Configure>", on_configure)

        # Mouse wheel zoom (cross-platform)
        def on_mousewheel(e: tk.Event) -> None:
            # Windows / macOS
            delta = getattr(e, "delta", 0)
            if delta > 0:
                self._zoom_adjust(panel, 1.1)
            elif delta < 0:
                self._zoom_adjust(panel, 0.9)

        def on_scroll_up(_e: tk.Event) -> None:
            # Linux scroll up
            self._zoom_adjust(panel, 1.1)

        def on_scroll_down(_e: tk.Event) -> None:
            # Linux scroll down
            self._zoom_adjust(panel, 0.9)

        win.bind("<MouseWheel>", on_mousewheel)
        win.bind("<Button-4>", on_scroll_up)
        win.bind("<Button-5>", on_scroll_down)
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_scroll_up)
        canvas.bind("<Button-5>", on_scroll_down)

        # Keyboard shortcuts
        def on_key(e: tk.Event) -> None:
            if e.keysym == "plus" or e.keysym == "equal":
                self._zoom_adjust(panel, 1.2)
            elif e.keysym == "minus":
                self._zoom_adjust(panel, 0.8)
            elif e.keysym == "Home":
                self._zoom_reset(panel)

        win.bind("<KeyPress>", on_key)

        self._render_zoom(panel)

    def _zoom_adjust(self, panel: str, factor: float) -> None:
        st = self._zoom_state.get(panel)
        if not st:
            return
        cur = float(st.get("scale", 1.0))
        st["scale"] = min(max(cur * factor, 0.1), 8.0)
        self._render_zoom(panel)

    def _zoom_reset(self, panel: str) -> None:
        st = self._zoom_state.get(panel)
        if not st:
            return
        st["scale"] = 1.0
        self._render_zoom(panel)

    def _save_zoom_image(self, panel: str, original_path: Path) -> None:
        """Save the current zoom view to a file."""
        st = self._zoom_state.get(panel)
        if not st or not st.get("base_img"):
            messagebox.showwarning("Save", "No image to save.")
            return

        try:
            base = st["base_img"]
            scale = float(st.get("scale", 1.0))
            new_w = max(int(base.width * scale), 1)
            new_h = max(int(base.height * scale), 1)
            resized = base.resize((new_w, new_h), PIL_LANCZOS)

            # Generate filename
            suffix = f"_zoom_{int(scale*100)}pct.png"
            save_name = original_path.stem + suffix
            save_path = IMAGES_DIR / save_name

            resized.save(str(save_path))
            messagebox.showinfo("Save", f"✅ Saved to:\n{save_name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {e}")

    def _find_processed_image(self, version: str, operation: str) -> Path | None:
        prefix = PREFIX_BY_VERSION.get(version)
        if not prefix:
            return None
        candidates = sorted(
            IMAGES_DIR.glob(f"{prefix}{operation}*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _refresh_previews(self) -> None:
        input_img = Path(self.image_var.get().strip())
        original, original_meta = self._load_preview_photo(input_img)
        if original is None:
            if not input_img.exists():
                msg = "Input image not found"
            elif not _pil_available:
                msg = "Install Pillow for JPG/BMP/TIFF preview"
            else:
                msg = "Preview unavailable"
            self.original_label.configure(image="", text=msg)
            self.original_path_label.configure(text=str(input_img))
            self._orig_photo = None
            self._preview_meta["original"] = None
            self.original_pixel_var.set("x:- y:- value:-")
        else:
            self.original_label.configure(image=original, text="")
            self.original_path_label.configure(text=input_img.name)
            self._orig_photo = original
            self._preview_meta["original"] = original_meta

        ver = self.preview_version_var.get()
        op = self.preview_operation_var.get()
        out_path = self._find_processed_image(ver, op)
        if out_path is None:
            self.processed_label.configure(image="", text="No output for current selection")
            self.processed_path_label.configure(text=f"{ver}:{op}")
            self._proc_photo = None
            self._preview_meta["processed"] = None
            self.processed_pixel_var.set("x:- y:- value:-")
            return

        processed, processed_meta = self._load_preview_photo(out_path)
        if processed is None:
            self.processed_label.configure(image="", text="Could not load output preview")
            self.processed_path_label.configure(text=str(out_path.name))
            self._proc_photo = None
            self._preview_meta["processed"] = None
            self.processed_pixel_var.set("x:- y:- value:-")
            return

        self.processed_label.configure(image=processed, text="")
        self.processed_path_label.configure(text=out_path.name)
        self._proc_photo = processed
        self._preview_meta["processed"] = processed_meta

    def _selected_versions(self) -> list[str]:
        return [v for v in VERSIONS if self.version_vars[v].get()]

    def _set_busy(self, busy: bool, status: str) -> None:
        self.is_running = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_build.configure(state=state)
        self.btn_run_selected.configure(state=state)
        self.btn_run_all.configure(state=state)
        self.btn_plot.configure(state=state)

        # Update status with colors
        if busy:
            self.status_var.set(f"⏳ {status}")
            self.progress.start(10)
        else:
            if "failed" in status.lower() or "error" in status.lower():
                icon = "❌"
                self.progress.configure(value=100)
            elif "completed" in status.lower() or "finished" in status.lower() or "ready" in status.lower():
                icon = "✅"
                self.progress.configure(value=100)
            else:
                icon = "ℹ️"
            self.status_var.set(f"{icon} {status}")
            self.progress.stop()

    def _int_or_error(self, value: str, label: str) -> int | None:
        try:
            v = int(value)
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            messagebox.showerror("Invalid parameter", f"{label} must be a positive integer.")
            return None

    def _validate_inputs(self) -> tuple[Path, int, int, int] | None:
        image = Path(self.image_var.get().strip())
        if not image.exists() or not image.is_file():
            messagebox.showerror("Invalid image", "Please select a valid image file.")
            return None

        omp_t = self._int_or_error(self.omp_threads_var.get().strip(), "OpenMP threads")
        mpi_p = self._int_or_error(self.mpi_procs_var.get().strip(), "MPI processes")
        hyb_t = self._int_or_error(self.hybrid_threads_var.get().strip(), "Hybrid threads")
        if omp_t is None or mpi_p is None or hyb_t is None:
            return None

        return image, omp_t, mpi_p, hyb_t

    def _build_command(self, version: str, image: Path, omp_t: int, mpi_p: int, hyb_t: int) -> list[str]:
        if version == "sequential":
            return [str(BUILD_DIR / "sequential"), str(image)]
        if version == "openmp":
            return [str(BUILD_DIR / "openmp_proc"), "-i", str(image), "-t", str(omp_t)]
        if version == "mpi":
            return ["mpirun", "-np", str(mpi_p), str(BUILD_DIR / "mpi_proc"), str(image)]
        if version == "hybrid":
            return [
                "mpirun",
                "-np",
                str(mpi_p),
                str(BUILD_DIR / "hybrid_proc"),
                "-i",
                str(image),
                "-t",
                str(hyb_t),
            ]
        if version == "opencl":
            return [str(BUILD_DIR / "opencl_proc"), str(image)]
        raise ValueError(f"Unsupported version: {version}")

    def _run_subprocess(self, cmd: list[str], label: str) -> int:
        self.log_queue.put(f"\n{'='*60}\n")
        self.log_queue.put(f"▶️ {label}\n")
        self.log_queue.put(f"{'='*60}\n")
        self.log_queue.put(f"$ {shlex.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(line)
            m = TIME_LINE_RE.match(line.strip())
            if m:
                op = m.group(1)
                val = float(m.group(2))
                low_label = label.lower()
                if "openmp" in low_label:
                    self.results["openmp"][op] = val
                elif "hybrid" in low_label:
                    self.results["hybrid"][op] = val
                elif "opencl" in low_label:
                    self.results["opencl"][op] = val
                elif "mpi" in low_label:
                    self.results["mpi"][op] = val
                elif "sequential" in low_label:
                    self.results["sequential"][op] = val

        proc.wait()
        rc = proc.returncode
        if rc == 0:
            self.log_queue.put(f"✅ {label} finished successfully.\n")
        else:
            self.log_queue.put(f"❌ {label} failed with code {rc}.\n")
        return rc

    def _refresh_table(self) -> None:
        rows = self.table.get_children()
        for i, op in enumerate(OPERATIONS):
            vals = [op]
            for v in VERSIONS:
                t = self.results.get(v, {}).get(op)
                vals.append(f"{t:.4f}" if t is not None else "-")
            self.table.item(rows[i], values=vals)

    def _load_timings_from_csv(self, versions: list[str]) -> None:
        for v in versions:
            csv_path = DATA_DIR / f"{v}_results.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    op = row.get("operation", "").strip()
                    if op in OPERATIONS:
                        try:
                            self.results[v][op] = float(row.get("elapsed_sec", ""))
                        except ValueError:
                            pass

    def _start_build_all(self) -> None:
        if self.is_running:
            return
        self._set_busy(True, "Building binaries...")
        threading.Thread(target=self._worker_build_all, daemon=True).start()

    def _worker_build_all(self) -> None:
        rc = self._run_subprocess(["make", "all"], "🔨 Build all")
        self.after(0, lambda: self._set_busy(False, "Build completed successfully" if rc == 0 else "Build failed"))

    def _start_run_selected(self) -> None:
        if self.is_running:
            return
        sel = self._selected_versions()
        if not sel:
            messagebox.showwarning("No versions selected", "Select at least one version to run.")
            return
        validated = self._validate_inputs()
        if validated is None:
            return
        image, omp_t, mpi_p, hyb_t = validated
        self._set_busy(True, "Running selected versions...")
        threading.Thread(
            target=self._worker_run_versions,
            args=(sel, image, omp_t, mpi_p, hyb_t),
            daemon=True,
        ).start()

    def _start_run_all(self) -> None:
        if self.is_running:
            return
        validated = self._validate_inputs()
        if validated is None:
            return
        image, omp_t, mpi_p, hyb_t = validated
        self._set_busy(True, "Running all versions...")
        threading.Thread(
            target=self._worker_run_versions,
            args=(VERSIONS, image, omp_t, mpi_p, hyb_t),
            daemon=True,
        ).start()

    def _worker_run_versions(self, versions: list[str], image: Path, omp_t: int, mpi_p: int, hyb_t: int) -> None:
        missing: list[str] = []
        mapping = {
            "sequential": BUILD_DIR / "sequential",
            "openmp": BUILD_DIR / "openmp_proc",
            "mpi": BUILD_DIR / "mpi_proc",
            "hybrid": BUILD_DIR / "hybrid_proc",
            "opencl": BUILD_DIR / "opencl_proc",
        }
        for v in versions:
            if not mapping[v].exists():
                missing.append(v)

        if missing:
            self.log_queue.put("\n⚠️ Binaries missing; running build first...\n")
            brc = self._run_subprocess(["make", "all"], "🔨 Auto-build")
            if brc != 0:
                self.after(0, lambda: self._set_busy(False, "Build failed"))
                return

        has_error = False
        for v in versions:
            cmd = self._build_command(v, image, omp_t, mpi_p, hyb_t)
            version_labels = {
                "sequential": "⚪ Sequential",
                "openmp": "🔵 OpenMP",
                "mpi": "🟣 MPI",
                "hybrid": "🟠 Hybrid",
                "opencl": "💠 OpenCL"
            }
            label = version_labels.get(v, v)
            rc = self._run_subprocess(cmd, label)
            if rc != 0:
                has_error = True
                break

        self._load_timings_from_csv(versions)
        self.after(0, self._refresh_table)
        self.after(0, self._refresh_previews)
        self.after(
            0,
            lambda: self._set_busy(False, "All runs completed successfully" if not has_error else "Run finished with errors"),
        )

    def _start_plot(self) -> None:
        if self.is_running:
            return
        self._set_busy(True, "Generating plots...")
        threading.Thread(target=self._worker_plot, daemon=True).start()

    def _worker_plot(self) -> None:
        self.log_queue.put(
            "📊 Keeping existing benchmark CSV files and regenerating plots.\n"
        )

        rc = self._run_subprocess([sys.executable, "scripts/plot_results.py"], "📊 Generating plots")
        status = "Plots generated successfully" if rc == 0 else "Plot generation failed"
        self.after(0, lambda: self._set_busy(False, status))

    def _open_folder(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(folder)], cwd=ROOT)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)], cwd=ROOT)
        elif os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Path", str(folder))


def main() -> int:
    app = ImageProcessingGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
