"""Desktop GUI shell for FluxForge.

The GUI is intentionally CLI-first:
- Every button maps to an existing ``fluxforge`` subcommand handler.
- The equivalent CLI command is shown in the run log.
- Users can copy the last generated CLI command for scripting/reproducibility.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    Figure = None

from fluxforge.analysis.peak_finders import PEAK_FINDER_METHODS
from fluxforge.cli import app as cli_app
from fluxforge.data.efficiency import CALIBRATION_SOURCES
from fluxforge.physics.stopping_power import Projectile, STANDARD_MATERIALS
from fluxforge_gui.constants import (
    ALLOWED_REACTION_CATEGORIES,
    GUI_BUFFER_OPERATIONS,
    GUI_PEAK_COUNTING_METHODS,
    GUI_PEAK_IDENTIFICATION_METHODS,
    GUI_RAFM_COUNTING_METHODS,
    GUI_UNFOLD_METHODS,
    GUI_UNFOLD_MLEM_CONVERGENCE_MODES,
)
from fluxforge_gui.presets import (
    get_gui_data_source_choices,
    get_gui_profile_choices,
    get_standards_gui_presets,
)


class UiBuilderMixin:
    def _configure_styles(self) -> None:
        """Apply a sleeker ttk look while staying cross-platform and open-source."""

        self.root.configure(background="#edf2f7")
        style = ttk.Style(self.root)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        style.configure("TFrame", background="#edf2f7")
        style.configure("Toolbar.TFrame", background="#dbe7f3")
        style.configure("TLabelframe", background="#edf2f7", padding=10)
        style.configure(
            "TLabelframe.Label",
            background="#edf2f7",
            foreground="#1f2937",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TLabel", background="#edf2f7", foreground="#1f2937")
        style.configure("Hint.TLabel", background="#edf2f7", foreground="#4b5563")
        style.configure("Status.TLabel", background="#dbe7f3", foreground="#111827")
        style.configure("Offline.TLabel", background="#dbe7f3", foreground="#8b1e3f")
        style.configure("TNotebook", background="#edf2f7", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff")],
            foreground=[("selected", "#111827")],
        )
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 9, "bold"))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self.root, padding=10, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(2, weight=1)

        ttk.Button(
            toolbar, text="Open Project...", command=self._choose_project_dir
        ).grid(row=0, column=0, padx=(0, 8))
        self.copy_btn = ttk.Button(
            toolbar, text="Copy Last CLI", command=self._copy_last_cli, state="disabled"
        )
        self.copy_btn.grid(row=0, column=1, padx=(0, 8))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(toolbar, textvariable=self.status_var).grid(
            row=0, column=2, sticky="e"
        )
        mode_text = (
            "Offline mode: remote sources/downloads disabled"
            if getattr(self, "offline_mode", False)
            else "Offline-ready workflow"
        )
        mode_style = (
            "Offline.TLabel"
            if getattr(self, "offline_mode", False)
            else "Status.TLabel"
        )
        ttk.Label(toolbar, text=mode_text, style=mode_style).grid(
            row=0, column=3, sticky="e", padx=(12, 0)
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        log_frame = ttk.LabelFrame(self.root, text="Run Log", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_box = ScrolledText(log_frame, wrap="word", height=14)
        self.log_box.grid(row=0, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")

    def _build_scrollable_controls_panel(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        column: int,
        width: int = 430,
    ) -> tuple[ttk.Frame, tk.Canvas]:
        """Create a vertically scrollable side panel for dense workflow controls."""

        shell = ttk.Frame(parent)
        shell.grid(row=row, column=column, sticky="nsw", padx=(0, 12))
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            shell,
            width=width,
            highlightthickness=0,
            bd=0,
            background="#edf2f7",
        )
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")

        interior = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=interior, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event) -> None:
            delta = getattr(event, "delta", 0)
            if delta:
                canvas.yview_scroll(int(-delta / 120), "units")
                return
            num = getattr(event, "num", None)
            if num == 4:
                canvas.yview_scroll(-3, "units")
            elif num == 5:
                canvas.yview_scroll(3, "units")

        def _bind_wheel(_event=None) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event=None) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        interior.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_width)
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        return interior, canvas

    def _build_ingest_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="1. Ingest")
        frame.columnconfigure(1, weight=1)

        self.ingest_input = tk.StringVar()
        self.ingest_output = tk.StringVar(value=str(self.project_dir / "spectrum.json"))
        self.ingest_profile = tk.StringVar(value="")
        self.ingest_background_file = tk.StringVar(value="")
        self.ingest_background_scale_mode = tk.StringVar(value="live")
        self.ingest_background_scale_factor = tk.StringVar(value="")
        self.ingest_energy_calibration = tk.StringVar(value="")
        self.ingest_efficiency_coefficients = tk.StringVar(value="")
        self.ingest_validate = tk.BooleanVar(value=True)

        self._path_row(frame, 0, "Input spectrum:", self.ingest_input, save=False)
        self._path_row(frame, 1, "Output artifact:", self.ingest_output, save=True)
        ttk.Label(frame, text="Bundled profile:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.ingest_profile,
            values=get_gui_profile_choices(),
            state="readonly",
            width=28,
        ).grid(row=2, column=1, sticky="w")
        self._path_row(
            frame, 3, "Background spectrum:", self.ingest_background_file, save=False
        )
        ttk.Label(frame, text="Background scale mode:").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.ingest_background_scale_mode,
            values=["live", "real", "manual"],
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w")
        self._entry_row(
            frame, 5, "Background scale factor:", self.ingest_background_scale_factor
        )
        self._entry_row(
            frame, 6, "Energy calibration CSV:", self.ingest_energy_calibration
        )
        self._entry_row(
            frame,
            7,
            "Efficiency coefficients CSV:",
            self.ingest_efficiency_coefficients,
        )
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.ingest_validate
        ).grid(row=8, column=1, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Run Ingest", command=self._run_ingest).grid(
            row=9, column=1, sticky="w"
        )

        ingest_batch_frame = ttk.LabelFrame(
            frame, text="Batch ingest workflow", padding=10
        )
        ingest_batch_frame.grid(
            row=10, column=0, columnspan=3, sticky="nsew", pady=(14, 0)
        )
        ingest_batch_frame.columnconfigure(1, weight=1)

        self.ingest_batch_input_dir = tk.StringVar(value=str(self.project_dir))
        self.ingest_batch_output_dir = tk.StringVar(
            value=str(self.project_dir / "batch_artifacts")
        )
        self.ingest_batch_profile = tk.StringVar(value="")
        self.ingest_batch_background_file = tk.StringVar(value="")
        self.ingest_batch_background_scale_mode = tk.StringVar(value="live")
        self.ingest_batch_background_scale_factor = tk.StringVar(value="")
        self.ingest_batch_energy_calibration = tk.StringVar(value="")
        self.ingest_batch_efficiency_coefficients = tk.StringVar(value="")
        self.ingest_batch_background_adjusted_dir = tk.StringVar(
            value=str(self.project_dir / "background_adjusted")
        )
        self.ingest_batch_final_corrected_dir = tk.StringVar(
            value=str(self.project_dir / "final_corrected")
        )
        self.ingest_batch_validate = tk.BooleanVar(value=True)

        self._directory_row(
            ingest_batch_frame, 0, "Input directory:", self.ingest_batch_input_dir
        )
        self._directory_row(
            ingest_batch_frame, 1, "Output directory:", self.ingest_batch_output_dir
        )
        ttk.Label(ingest_batch_frame, text="Bundled profile:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            ingest_batch_frame,
            textvariable=self.ingest_batch_profile,
            values=get_gui_profile_choices(),
            state="readonly",
            width=28,
        ).grid(row=2, column=1, sticky="w")
        self._path_row(
            ingest_batch_frame,
            3,
            "Background spectrum:",
            self.ingest_batch_background_file,
            save=False,
        )
        ttk.Label(ingest_batch_frame, text="Background scale mode:").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            ingest_batch_frame,
            textvariable=self.ingest_batch_background_scale_mode,
            values=["live", "real", "manual"],
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w")
        self._entry_row(
            ingest_batch_frame,
            5,
            "Background scale factor:",
            self.ingest_batch_background_scale_factor,
        )
        self._entry_row(
            ingest_batch_frame,
            6,
            "Energy calibration CSV:",
            self.ingest_batch_energy_calibration,
        )
        self._entry_row(
            ingest_batch_frame,
            7,
            "Efficiency coefficients CSV:",
            self.ingest_batch_efficiency_coefficients,
        )
        self._directory_row(
            ingest_batch_frame,
            8,
            "Background-adjusted CSV dir:",
            self.ingest_batch_background_adjusted_dir,
        )
        self._directory_row(
            ingest_batch_frame,
            9,
            "Final-corrected CSV dir:",
            self.ingest_batch_final_corrected_dir,
        )
        ttk.Checkbutton(
            ingest_batch_frame,
            text="Validate artifact schema",
            variable=self.ingest_batch_validate,
        ).grid(row=10, column=1, sticky="w", pady=(0, 8))
        ttk.Button(
            ingest_batch_frame,
            text="Run Batch Ingest",
            command=self._run_ingest_batch,
        ).grid(row=11, column=1, sticky="w")

    def _build_spectrum_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="2. Spectrum")
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        controls, self.preview_controls_canvas = self._build_scrollable_controls_panel(
            frame,
            row=0,
            column=0,
            width=430,
        )
        controls.columnconfigure(1, weight=1)

        self.preview_input = tk.StringVar(value=str(self.project_dir / "spectrum.json"))
        self.preview_overlay_inputs = tk.StringVar(value="")
        self.preview_peaks = tk.StringVar(value=str(self.project_dir / "peaks.json"))
        self.preview_y_scale = tk.StringVar(value="linear")
        self.preview_x_min = tk.StringVar(value="")
        self.preview_x_max = tk.StringVar(value="")
        self.preview_png_output = tk.StringVar(
            value=str(self.project_dir / "gui_preview.png")
        )
        self.preview_plot_title = tk.StringVar(value="")
        self.preview_manual_peak_report = tk.StringVar(
            value=str(self.project_dir / "manual_peak_report.json")
        )
        self.preview_plot_background_subtracted = tk.BooleanVar(value=False)
        self.preview_status = tk.StringVar(
            value="Load a spectrum artifact or raw file to preview it."
        )
        self.preview_selected_buffer = tk.StringVar(value="")
        self.preview_manual_roi_mode = tk.BooleanVar(value=False)
        self.preview_roi_label = tk.StringVar(value="ROI 1")
        self.preview_roi_left = tk.StringVar(value="")
        self.preview_roi_right = tk.StringVar(value="")
        self.preview_roi_file = tk.StringVar(
            value=str(self.project_dir / "manual_rois.json")
        )
        self.preview_calibration_c0 = tk.StringVar(value="0.0")
        self.preview_calibration_c1 = tk.StringVar(value="1.0")
        self.preview_calibration_c2 = tk.StringVar(value="0.0")
        self.preview_calibration_c3 = tk.StringVar(value="0.0")
        self.preview_inspector = tk.StringVar(
            value="Load peaks, then select a peak row to inspect fit/deconvolution context."
        )
        self.preview_peak_finder_method = tk.StringVar(value="segmented")
        self.preview_peak_threshold_sigma = tk.StringVar(value="3.0")
        self.preview_peak_min_distance = tk.StringVar(value="6")
        self.preview_peak_identification_method = tk.StringVar(value="line_match")
        self.preview_peak_tolerance_keV = tk.StringVar(value="1.5")
        self.preview_peak_min_matches = tk.StringVar(value="2")
        self.preview_peak_min_intensity = tk.StringVar(value="0.0")
        self.preview_database_source = tk.StringVar(value="decay_2012")
        self.preview_custom_database_path = tk.StringVar(value="")
        self.preview_database_summary = tk.StringVar(
            value="Select a nuclear-data source for peak identification and calibration references."
        )
        self.preview_peak_counting_method = tk.StringVar(value="gaussian_fit")
        self.preview_count_summary = tk.StringVar(
            value="Select a peak, then run a counting method to see net/gross counts and uncertainty."
        )
        self.preview_fit_mode = tk.StringVar(value="multiplet")
        self.preview_fit_window = tk.StringVar(value="8")
        self.preview_fit_neighbor_window = tk.StringVar(value="5.0")
        self.preview_fit_share_sigma = tk.BooleanVar(value=True)
        self.preview_fit_lock_centroids = tk.BooleanVar(value=False)
        self.preview_fit_constraint_template = tk.StringVar(value="identity")
        self.preview_fit_summary = tk.StringVar(
            value="Select one or more nearby peaks to run constrained multiplet or Hypermet fits."
        )
        self.preview_buffer_operation = tk.StringVar(value="sum")
        self.preview_buffer_output = tk.StringVar(
            value=str(self.project_dir / "combined_buffer.json")
        )
        self.preview_calibration_point_channel = tk.StringVar(value="")
        self.preview_calibration_point_observed = tk.StringVar(value="")
        self.preview_calibration_point_reference = tk.StringVar(value="")
        self.preview_calibration_point_label = tk.StringVar(value="")
        self.preview_calibration_order = tk.StringVar(value="1")
        self.preview_calibration_summary = tk.StringVar(
            value="Add picked points, fit a polynomial, then inspect residuals and apply the coefficients."
        )
        self.preview_efficiency_source = tk.StringVar(value="Cs-137")
        self.preview_efficiency_line = tk.StringVar(value="")
        self.preview_efficiency_activity_bq = tk.StringVar(value="1000.0")
        self.preview_efficiency_live_time_s = tk.StringVar(value="100.0")
        self.preview_efficiency_geometry_factor = tk.StringVar(value="1.0")
        self.preview_efficiency_fit_degree = tk.StringVar(value="2")
        self.preview_efficiency_output = tk.StringVar(
            value=str(self.project_dir / "efficiency_curve.json")
        )
        self.preview_efficiency_summary = tk.StringVar(
            value="Use counted calibration peaks to build and fit an efficiency curve."
        )

        source_frame = ttk.LabelFrame(controls, text="Spectrum sources")
        source_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        source_frame.columnconfigure(1, weight=1)
        self._path_row(
            source_frame, 0, "Primary spectrum:", self.preview_input, save=False
        )
        self._entry_row(
            source_frame,
            1,
            "Overlay spectra (; separated):",
            self.preview_overlay_inputs,
        )
        self._path_row(
            source_frame,
            2,
            "Peaks artifact (optional):",
            self.preview_peaks,
            save=False,
        )
        ttk.Label(source_frame, text="Y scale:").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            source_frame,
            textvariable=self.preview_y_scale,
            values=["linear", "log"],
            state="readonly",
            width=16,
        ).grid(row=3, column=1, sticky="w")
        self._entry_row(source_frame, 4, "X min keV (optional):", self.preview_x_min)
        self._entry_row(source_frame, 5, "X max keV (optional):", self.preview_x_max)
        self._path_row(
            source_frame, 6, "Save preview PNG:", self.preview_png_output, save=True
        )
        self._entry_row(
            source_frame, 7, "CLI plot title (optional):", self.preview_plot_title
        )
        self._path_row(
            source_frame,
            8,
            "CLI manual peak report:",
            self.preview_manual_peak_report,
            save=True,
        )
        ttk.Checkbutton(
            source_frame,
            text="Use background-subtracted spectrum for CLI plot export",
            variable=self.preview_plot_background_subtracted,
        ).grid(row=9, column=1, sticky="w", pady=(0, 4))

        button_row = ttk.Frame(source_frame)
        button_row.grid(row=10, column=1, sticky="w", pady=(6, 8))
        ttk.Button(
            button_row, text="Load Preview", command=self._load_spectrum_preview
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            button_row, text="Save PNG", command=self._save_spectrum_preview_png
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(
            button_row, text="Run CLI Plot Export", command=self._run_spectrum_plot
        ).grid(row=0, column=2)

        ttk.Label(
            source_frame,
            textvariable=self.preview_status,
            wraplength=340,
            justify="left",
        ).grid(row=11, column=0, columnspan=3, sticky="ew", pady=(2, 10))

        data_source_frame = ttk.LabelFrame(controls, text="Nuclear data sources")
        data_source_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        data_source_frame.columnconfigure(1, weight=1)
        ttk.Label(data_source_frame, text="ID source:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        preview_source_combo = ttk.Combobox(
            data_source_frame,
            textvariable=self.preview_database_source,
            values=get_gui_data_source_choices(self._custom_data_sources),
            state="readonly",
            width=24,
        )
        preview_source_combo.grid(row=0, column=1, sticky="w")
        preview_source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_preview_data_source_summary(),
        )
        self._path_row(
            data_source_frame,
            1,
            "Custom JSON/CSV:",
            self.preview_custom_database_path,
            save=False,
        )
        data_source_buttons = ttk.Frame(data_source_frame)
        data_source_buttons.grid(row=2, column=1, sticky="w", pady=(4, 4))
        ttk.Button(
            data_source_buttons,
            text="Refresh source",
            command=self._refresh_preview_data_source_summary,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            data_source_buttons,
            text="Register custom",
            command=self._register_custom_preview_data_source,
        ).grid(row=0, column=1)
        ttk.Label(
            data_source_frame,
            textvariable=self.preview_database_summary,
            wraplength=340,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        buffer_frame = ttk.LabelFrame(controls, text="Multi-buffer manager")
        buffer_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        buffer_frame.columnconfigure(0, weight=1)
        self.preview_buffer_list = tk.Listbox(
            buffer_frame, height=6, selectmode=tk.EXTENDED, exportselection=False
        )
        self.preview_buffer_list.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(
            buffer_frame, text="Add current", command=self._buffer_add_current
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(
            buffer_frame, text="Add file...", command=self._buffer_add_file
        ).grid(row=1, column=1, sticky="ew", pady=(6, 0), padx=4)
        ttk.Button(
            buffer_frame, text="Remove", command=self._buffer_remove_selected
        ).grid(row=1, column=2, sticky="ew", pady=(6, 0), padx=(4, 0))
        ttk.Button(
            buffer_frame, text="Use selection", command=self._buffer_use_selection
        ).grid(row=2, column=0, sticky="ew", pady=(4, 0), padx=(0, 4))
        ttk.Button(
            buffer_frame, text="Use as overlays", command=self._buffer_overlay_selection
        ).grid(row=2, column=1, sticky="ew", pady=(4, 0), padx=4)
        ttk.Button(buffer_frame, text="Clear", command=self._buffer_clear).grid(
            row=2, column=2, sticky="ew", pady=(4, 0), padx=(4, 0)
        )
        ttk.Label(buffer_frame, text="Arithmetic:").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            buffer_frame,
            textvariable=self.preview_buffer_operation,
            values=list(GUI_BUFFER_OPERATIONS),
            state="readonly",
            width=14,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Button(
            buffer_frame, text="Combine selected", command=self._buffer_apply_arithmetic
        ).grid(row=3, column=2, sticky="ew", pady=(8, 0))
        self._path_row(
            buffer_frame,
            4,
            "Write combined artifact:",
            self.preview_buffer_output,
            save=True,
        )

        roi_frame = ttk.LabelFrame(controls, text="Manual ROI draw / edit")
        roi_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        roi_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            roi_frame,
            text="Draw ROI from plot clicks",
            variable=self.preview_manual_roi_mode,
            command=self._toggle_roi_mode,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._entry_row(roi_frame, 1, "ROI label:", self.preview_roi_label)
        self._entry_row(roi_frame, 2, "Left keV:", self.preview_roi_left)
        self._entry_row(roi_frame, 3, "Right keV:", self.preview_roi_right)
        roi_button_row = ttk.Frame(roi_frame)
        roi_button_row.grid(row=4, column=1, sticky="w", pady=(4, 4))
        ttk.Button(
            roi_button_row,
            text="Add / Update ROI",
            command=self._add_or_update_manual_roi,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            roi_button_row, text="Delete ROI", command=self._delete_selected_manual_roi
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            roi_button_row, text="Clear ROIs", command=self._clear_manual_rois
        ).grid(row=0, column=2)
        self._path_row(roi_frame, 5, "ROI JSON:", self.preview_roi_file, save=True)
        roi_io_row = ttk.Frame(roi_frame)
        roi_io_row.grid(row=6, column=1, sticky="w")
        ttk.Button(
            roi_io_row, text="Load ROI file", command=self._load_manual_rois_file
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            roi_io_row, text="Save ROI file", command=self._save_manual_rois_file
        ).grid(row=0, column=1)

        self.preview_roi_table = ttk.Treeview(
            roi_frame,
            columns=("label", "left", "right", "notes"),
            show="headings",
            height=5,
        )
        for column, text, width in (
            ("label", "Label", 80),
            ("left", "Left keV", 70),
            ("right", "Right keV", 70),
            ("notes", "Notes", 100),
        ):
            self.preview_roi_table.heading(column, text=text)
            self.preview_roi_table.column(column, width=width, anchor="w")
        self.preview_roi_table.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        self.preview_roi_table.bind("<<TreeviewSelect>>", self._on_roi_selected)

        cal_frame = ttk.LabelFrame(controls, text="Calibration editor")
        cal_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        cal_frame.columnconfigure(1, weight=1)
        self._entry_row(cal_frame, 0, "C0:", self.preview_calibration_c0)
        self._entry_row(cal_frame, 1, "C1:", self.preview_calibration_c1)
        self._entry_row(cal_frame, 2, "C2:", self.preview_calibration_c2)
        self._entry_row(cal_frame, 3, "C3:", self.preview_calibration_c3)
        cal_buttons = ttk.Frame(cal_frame)
        cal_buttons.grid(row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Button(
            cal_buttons,
            text="Use loaded calibration",
            command=self._load_calibration_from_preview,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            cal_buttons,
            text="Apply calibration",
            command=self._apply_calibration_to_preview,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            cal_buttons, text="Reset", command=self._reset_calibration_editor
        ).grid(row=0, column=2)
        ttk.Separator(cal_frame, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=8
        )
        self._entry_row(
            cal_frame, 6, "Picked channel:", self.preview_calibration_point_channel
        )
        self._entry_row(
            cal_frame, 7, "Observed keV:", self.preview_calibration_point_observed
        )
        self._entry_row(
            cal_frame, 8, "Reference keV:", self.preview_calibration_point_reference
        )
        self._entry_row(
            cal_frame, 9, "Point label:", self.preview_calibration_point_label
        )
        ttk.Label(cal_frame, text="Polynomial order:").grid(
            row=10, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            cal_frame,
            textvariable=self.preview_calibration_order,
            values=["1", "2", "3"],
            state="readonly",
            width=8,
        ).grid(row=10, column=1, sticky="w")
        cal_point_buttons = ttk.Frame(cal_frame)
        cal_point_buttons.grid(row=11, column=1, sticky="w", pady=(4, 4))
        ttk.Button(
            cal_point_buttons,
            text="Pick selected peak",
            command=self._pick_calibration_point_from_selected_peak,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            cal_point_buttons, text="Add point", command=self._add_calibration_point
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            cal_point_buttons,
            text="Fit calibration",
            command=self._fit_preview_calibration_points,
        ).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(
            cal_point_buttons,
            text="Clear points",
            command=self._clear_calibration_points,
        ).grid(row=0, column=3)
        self.preview_calibration_table = ttk.Treeview(
            cal_frame,
            columns=("channel", "observed", "reference", "residual", "label"),
            show="headings",
            height=5,
        )
        for column, text, width in (
            ("channel", "Channel", 70),
            ("observed", "Observed", 75),
            ("reference", "Reference", 75),
            ("residual", "Residual", 70),
            ("label", "Label", 85),
        ):
            self.preview_calibration_table.heading(column, text=text)
            self.preview_calibration_table.column(
                column, width=width, anchor="e" if column != "label" else "w"
            )
        self.preview_calibration_table.grid(
            row=12, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        ttk.Label(
            cal_frame,
            textvariable=self.preview_calibration_summary,
            wraplength=340,
            justify="left",
        ).grid(row=13, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        eff_frame = ttk.LabelFrame(controls, text="Efficiency calibration")
        eff_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        eff_frame.columnconfigure(1, weight=1)
        ttk.Label(eff_frame, text="Calibration source:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        efficiency_source_combo = ttk.Combobox(
            eff_frame,
            textvariable=self.preview_efficiency_source,
            values=sorted(CALIBRATION_SOURCES),
            state="readonly",
            width=16,
        )
        efficiency_source_combo.grid(row=0, column=1, sticky="w")
        efficiency_source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_efficiency_line_choices(),
        )
        ttk.Label(eff_frame, text="Reference line:").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.preview_efficiency_line_combo = ttk.Combobox(
            eff_frame,
            textvariable=self.preview_efficiency_line,
            state="readonly",
            width=20,
        )
        self.preview_efficiency_line_combo.grid(row=1, column=1, sticky="w")
        self._entry_row(
            eff_frame, 2, "Source activity (Bq):", self.preview_efficiency_activity_bq
        )
        self._entry_row(
            eff_frame, 3, "Live time (s):", self.preview_efficiency_live_time_s
        )
        self._entry_row(
            eff_frame, 4, "Geometry factor:", self.preview_efficiency_geometry_factor
        )
        ttk.Label(eff_frame, text="Fit degree:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            eff_frame,
            textvariable=self.preview_efficiency_fit_degree,
            values=["1", "2", "3", "4"],
            state="readonly",
            width=8,
        ).grid(row=5, column=1, sticky="w")
        self._path_row(
            eff_frame, 6, "Curve JSON:", self.preview_efficiency_output, save=True
        )
        eff_buttons = ttk.Frame(eff_frame)
        eff_buttons.grid(row=7, column=1, sticky="w", pady=(4, 4))
        ttk.Button(
            eff_buttons,
            text="Use selected peak",
            command=self._add_efficiency_point_from_selected_peak,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            eff_buttons,
            text="Fit curve",
            command=self._fit_efficiency_curve_from_points,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            eff_buttons, text="Save curve", command=self._save_efficiency_curve
        ).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(
            eff_buttons, text="Clear", command=self._clear_efficiency_points
        ).grid(row=0, column=3)
        self.preview_efficiency_table = ttk.Treeview(
            eff_frame,
            columns=("source", "energy", "net", "eff", "unc"),
            show="headings",
            height=4,
        )
        for column, text, width in (
            ("source", "Source", 85),
            ("energy", "keV", 70),
            ("net", "Net", 70),
            ("eff", "Eff", 70),
            ("unc", "±", 70),
        ):
            self.preview_efficiency_table.heading(column, text=text)
            self.preview_efficiency_table.column(
                column, width=width, anchor="w" if column == "source" else "e"
            )
        self.preview_efficiency_table.grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        ttk.Label(
            eff_frame,
            textvariable=self.preview_efficiency_summary,
            wraplength=340,
            justify="left",
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        analysis_frame = ttk.LabelFrame(
            controls, text="Peak search / identification / counting"
        )
        analysis_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        analysis_frame.columnconfigure(1, weight=1)
        ttk.Label(analysis_frame, text="Finder:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            analysis_frame,
            textvariable=self.preview_peak_finder_method,
            values=sorted(list(PEAK_FINDER_METHODS) + ["consensus"]),
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w")
        self._entry_row(
            analysis_frame, 1, "Threshold σ:", self.preview_peak_threshold_sigma
        )
        self._entry_row(
            analysis_frame, 2, "Min distance:", self.preview_peak_min_distance
        )
        ttk.Label(analysis_frame, text="ID method:").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            analysis_frame,
            textvariable=self.preview_peak_identification_method,
            values=list(GUI_PEAK_IDENTIFICATION_METHODS),
            state="readonly",
            width=20,
        ).grid(row=3, column=1, sticky="w")
        self._entry_row(
            analysis_frame, 4, "Tolerance keV:", self.preview_peak_tolerance_keV
        )
        self._entry_row(
            analysis_frame, 5, "Consensus matches:", self.preview_peak_min_matches
        )
        self._entry_row(
            analysis_frame, 6, "Min intensity:", self.preview_peak_min_intensity
        )
        ttk.Button(
            analysis_frame,
            text="Auto-detect peaks",
            command=self._auto_detect_preview_peaks,
        ).grid(row=7, column=1, sticky="w", pady=(4, 4))
        ttk.Separator(analysis_frame, orient="horizontal").grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=8
        )
        ttk.Label(analysis_frame, text="Count method:").grid(
            row=9, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            analysis_frame,
            textvariable=self.preview_peak_counting_method,
            values=list(GUI_PEAK_COUNTING_METHODS),
            state="readonly",
            width=20,
        ).grid(row=9, column=1, sticky="w")
        ttk.Button(
            analysis_frame,
            text="Count selected peak",
            command=self._count_selected_peak,
        ).grid(row=10, column=1, sticky="w", pady=(4, 0))
        ttk.Label(
            analysis_frame,
            textvariable=self.preview_count_summary,
            wraplength=340,
            justify="left",
        ).grid(row=11, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        peak_frame = ttk.LabelFrame(controls, text="Detected / loaded peaks")
        peak_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        peak_frame.columnconfigure(0, weight=1)
        peak_frame.rowconfigure(0, weight=1)
        self.preview_peak_table = ttk.Treeview(
            peak_frame,
            columns=("energy", "area", "channel", "label"),
            show="headings",
            height=16,
        )
        self.preview_peak_table.heading("energy", text="Energy (keV)")
        self.preview_peak_table.heading("area", text="Area / Net")
        self.preview_peak_table.heading("channel", text="Channel")
        self.preview_peak_table.heading("label", text="Label")
        self.preview_peak_table.column("energy", width=95, anchor="e")
        self.preview_peak_table.column("area", width=95, anchor="e")
        self.preview_peak_table.column("channel", width=70, anchor="e")
        self.preview_peak_table.column("label", width=130, anchor="w")
        self.preview_peak_table.grid(row=0, column=0, sticky="nsew")
        self.preview_peak_table.bind("<<TreeviewSelect>>", self._on_peak_selected)
        peak_scroll = ttk.Scrollbar(
            peak_frame, orient="vertical", command=self.preview_peak_table.yview
        )
        peak_scroll.grid(row=0, column=1, sticky="ns")
        self.preview_peak_table.configure(yscrollcommand=peak_scroll.set)

        fit_frame = ttk.LabelFrame(
            controls, text="Constrained multiplet / deconvolution fit"
        )
        fit_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        fit_frame.columnconfigure(1, weight=1)
        ttk.Label(fit_frame, text="Fit mode:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            fit_frame,
            textvariable=self.preview_fit_mode,
            values=["multiplet", "hypermet"],
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w")
        self._entry_row(fit_frame, 1, "Fit half-width:", self.preview_fit_window)
        self._entry_row(
            fit_frame, 2, "Neighbor window keV:", self.preview_fit_neighbor_window
        )
        ttk.Checkbutton(
            fit_frame, text="Share sigma", variable=self.preview_fit_share_sigma
        ).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(
            fit_frame, text="Lock centroids", variable=self.preview_fit_lock_centroids
        ).grid(row=4, column=1, sticky="w")
        ttk.Label(fit_frame, text="Constraint template:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        constraint_combo = ttk.Combobox(
            fit_frame,
            textvariable=self.preview_fit_constraint_template,
            values=["identity", "shared-doublet", "shared-triplet"],
            state="readonly",
            width=18,
        )
        constraint_combo.grid(row=5, column=1, sticky="w")
        constraint_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_fit_constraint_template()
        )
        ttk.Label(fit_frame, text="Constraint matrix:").grid(
            row=6, column=0, sticky="nw", padx=(0, 8), pady=4
        )
        self.preview_fit_constraint_box = ScrolledText(
            fit_frame, wrap="none", height=4, width=28
        )
        self.preview_fit_constraint_box.grid(row=6, column=1, sticky="ew", pady=4)
        fit_buttons = ttk.Frame(fit_frame)
        fit_buttons.grid(row=7, column=1, sticky="w", pady=(4, 0))
        ttk.Button(
            fit_buttons,
            text="Apply template",
            command=self._apply_fit_constraint_template,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            fit_buttons,
            text="Fit selected / neighbors",
            command=self._fit_selected_peak_group,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            fit_buttons,
            text="Clear diagnostics",
            command=self._clear_preview_diagnostic_plot,
        ).grid(row=0, column=2)
        ttk.Label(
            fit_frame,
            textvariable=self.preview_fit_summary,
            wraplength=340,
            justify="left",
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        inspector_frame = ttk.LabelFrame(
            controls, text="Peak-fit / deconvolution inspector"
        )
        inspector_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        inspector_frame.columnconfigure(0, weight=1)
        ttk.Label(
            inspector_frame,
            textvariable=self.preview_inspector,
            wraplength=340,
            justify="left",
        ).grid(row=0, column=0, sticky="ew")
        inspector_buttons = ttk.Frame(inspector_frame)
        inspector_buttons.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(
            inspector_buttons,
            text="Zoom selected peak",
            command=self._zoom_selected_peak,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            inspector_buttons, text="Reset zoom", command=self._reset_preview_zoom
        ).grid(row=0, column=1)

        self._refresh_preview_data_source_summary()
        self._refresh_efficiency_line_choices()
        self._apply_fit_constraint_template()

        plot_frame = ttk.LabelFrame(frame, text="Spectrum display", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=1)

        ttk.Label(
            plot_frame,
            text=(
                "This viewer is modeled after the GUI plan and the reference patterns from "
                "PeakEasy, QuantumGold, SpecKit, Gamma-MCA, HDTV, and Physics-backed workflows."
            ),
            wraplength=820,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(
                plot_frame,
                text="matplotlib is not available, so the spectrum display cannot be rendered.",
                foreground="#8b0000",
            ).grid(row=1, column=0, sticky="nw")
            self.preview_figure = None
            self.preview_axes = None
            self.preview_canvas = None
            return

        self.preview_figure = Figure(figsize=(9.4, 5.6), dpi=100)
        self.preview_axes = self.preview_figure.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=plot_frame)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.preview_click_cid = self.preview_canvas.mpl_connect(
            "button_press_event", self._on_preview_click
        )
        self.preview_motion_cid = self.preview_canvas.mpl_connect(
            "motion_notify_event", self._on_preview_motion
        )
        self.preview_release_cid = self.preview_canvas.mpl_connect(
            "button_release_event", self._on_preview_release
        )
        if NavigationToolbar2Tk is not None:
            toolbar = NavigationToolbar2Tk(
                self.preview_canvas, plot_frame, pack_toolbar=False
            )
            toolbar.update()
            toolbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _build_peaks_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="3. Peaks")
        frame.columnconfigure(1, weight=1)

        self.peaks_input = tk.StringVar(value=str(self.project_dir / "spectrum.json"))
        self.peaks_output = tk.StringVar(value=str(self.project_dir / "peaks.json"))
        self.peaks_sensitivity = tk.StringVar(value="default")
        self.peaks_fit_window = tk.StringVar(value="6")
        self.peaks_manual_file = tk.StringVar(value="")
        self.peaks_profile = tk.StringVar(value="")
        self.peaks_background_file = tk.StringVar(value="")
        self.peaks_background_scale_mode = tk.StringVar(value="live")
        self.peaks_background_scale_factor = tk.StringVar(value="")
        self.peaks_energy_calibration = tk.StringVar(value="")
        self.peaks_efficiency_coefficients = tk.StringVar(value="")
        self.peaks_data_source = tk.StringVar(value="decay_2012")
        self.peaks_custom_source = tk.StringVar(value="")
        self.peaks_source_summary = tk.StringVar(
            value="Reference source used for downstream peak interpretation and provenance logging."
        )
        self.peaks_background_subtracted = tk.BooleanVar(value=False)
        self.peaks_validate = tk.BooleanVar(value=True)

        self._path_row(frame, 0, "Spectrum artifact:", self.peaks_input, save=False)
        self._path_row(frame, 1, "Output peaks artifact:", self.peaks_output, save=True)
        ttk.Label(frame, text="Sensitivity:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.peaks_sensitivity,
            values=["default", "sensitive", "conservative"],
            state="readonly",
            width=16,
        ).grid(row=2, column=1, sticky="w")
        self._entry_row(frame, 3, "Fit window:", self.peaks_fit_window)
        self._path_row(frame, 4, "Manual ROI file:", self.peaks_manual_file, save=False)
        ttk.Label(frame, text="Bundled profile:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.peaks_profile,
            values=get_gui_profile_choices(),
            state="readonly",
            width=28,
        ).grid(row=5, column=1, sticky="w")
        self._path_row(
            frame, 6, "Background spectrum:", self.peaks_background_file, save=False
        )
        ttk.Label(frame, text="Background scale mode:").grid(
            row=7, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.peaks_background_scale_mode,
            values=["live", "real", "manual"],
            state="readonly",
            width=16,
        ).grid(row=7, column=1, sticky="w")
        self._entry_row(
            frame, 8, "Background scale factor:", self.peaks_background_scale_factor
        )
        self._entry_row(
            frame, 9, "Energy calibration CSV:", self.peaks_energy_calibration
        )
        self._entry_row(
            frame,
            10,
            "Efficiency coefficients CSV:",
            self.peaks_efficiency_coefficients,
        )
        ttk.Label(frame, text="Reference source:").grid(
            row=11, column=0, sticky="w", padx=(0, 8), pady=4
        )
        peaks_source_combo = ttk.Combobox(
            frame,
            textvariable=self.peaks_data_source,
            values=get_gui_data_source_choices(self._custom_data_sources),
            state="readonly",
            width=28,
        )
        peaks_source_combo.grid(row=11, column=1, sticky="w")
        peaks_source_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_peaks_source_summary()
        )
        self._entry_row(
            frame, 12, "Custom source (optional):", self.peaks_custom_source
        )
        ttk.Label(
            frame,
            textvariable=self.peaks_source_summary,
            wraplength=420,
            justify="left",
        ).grid(row=13, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        ttk.Checkbutton(
            frame,
            text="Integrate manual ROI on background-subtracted spectrum",
            variable=self.peaks_background_subtracted,
        ).grid(row=14, column=1, sticky="w", pady=(0, 4))
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.peaks_validate
        ).grid(row=15, column=1, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Run Peak Detection", command=self._run_peaks).grid(
            row=16, column=1, sticky="w"
        )
        self._refresh_peaks_source_summary()

    def _build_activity_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="4. Activity")
        frame.columnconfigure(1, weight=1)

        self.activity_input = tk.StringVar(value=str(self.project_dir / "peaks.json"))
        self.activity_output = tk.StringVar(
            value=str(self.project_dir / "activities.json")
        )
        self.activity_live_time = tk.StringVar(value="")
        self.activity_eff = tk.StringVar(value="1.0")
        self.activity_emission = tk.StringVar(value="1.0")
        self.activity_half_life = tk.StringVar(value="1.0")
        self.activity_sample_mass_g = tk.StringVar(value="")
        self.activity_isotope = tk.StringVar(value="")
        self.activity_reaction = tk.StringVar(value="")
        self.activity_data_source = tk.StringVar(value="nndc_offline_activation")
        self.activity_custom_source = tk.StringVar(value="")
        self.activity_source_summary = tk.StringVar(
            value="Reference source for isotope half-life and gamma data lookups."
        )
        self.activity_result_summary = tk.StringVar(
            value="Activity results will summarize line-by-line uncertainty after a run."
        )
        self.activity_validate = tk.BooleanVar(value=True)

        self._path_row(frame, 0, "Peaks artifact:", self.activity_input, save=False)
        self._path_row(frame, 1, "Output activities:", self.activity_output, save=True)
        self._entry_row(frame, 2, "Live time (s, optional):", self.activity_live_time)
        self._entry_row(frame, 3, "Efficiency:", self.activity_eff)
        self._entry_row(frame, 4, "Emission probability:", self.activity_emission)
        self._entry_row(frame, 5, "Half-life (s):", self.activity_half_life)
        self._entry_row(
            frame, 6, "Sample mass (g, optional):", self.activity_sample_mass_g
        )
        self._entry_row(frame, 7, "Isotope override (optional):", self.activity_isotope)
        self._entry_row(
            frame, 8, "Reaction ID override (optional):", self.activity_reaction
        )
        ttk.Label(frame, text="Reference source:").grid(
            row=9, column=0, sticky="w", padx=(0, 8), pady=4
        )
        activity_source_combo = ttk.Combobox(
            frame,
            textvariable=self.activity_data_source,
            values=get_gui_data_source_choices(self._custom_data_sources),
            state="readonly",
            width=28,
        )
        activity_source_combo.grid(row=9, column=1, sticky="w")
        activity_source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_activity_source_summary(),
        )
        self._entry_row(
            frame, 10, "Custom source (optional):", self.activity_custom_source
        )
        activity_buttons = ttk.Frame(frame)
        activity_buttons.grid(row=11, column=1, sticky="w", pady=(2, 4))
        ttk.Button(
            activity_buttons,
            text="Lookup from source",
            command=self._lookup_activity_reference,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            activity_buttons,
            text="Load Summary",
            command=self._load_activity_result_summary,
        ).grid(row=0, column=1)
        ttk.Label(
            frame,
            textvariable=self.activity_source_summary,
            wraplength=420,
            justify="left",
        ).grid(row=12, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        ttk.Label(
            frame,
            textvariable=self.activity_result_summary,
            wraplength=420,
            justify="left",
        ).grid(row=13, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.activity_validate
        ).grid(row=14, column=1, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Run Activity", command=self._run_activity).grid(
            row=15, column=1, sticky="w"
        )
        self._refresh_activity_source_summary()

    def _build_rates_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="5. Rates")
        frame.columnconfigure(1, weight=1)

        self.rates_input = tk.StringVar(value=str(self.project_dir / "activities.json"))
        self.rates_output = tk.StringVar(value=str(self.project_dir / "rates.json"))
        self.rates_segments = tk.StringVar(value="")
        self.rates_duration = tk.StringVar(value="1.0")
        self.rates_half_life = tk.StringVar(value="1.0")
        self.rates_result_summary = tk.StringVar(
            value="Reaction-rate results will summarize propagated uncertainty after a run."
        )
        self.rates_validate = tk.BooleanVar(value=True)

        self._path_row(frame, 0, "Activities artifact:", self.rates_input, save=False)
        self._path_row(frame, 1, "Output rates artifact:", self.rates_output, save=True)
        self._path_row(
            frame, 2, "Segments JSON (optional):", self.rates_segments, save=False
        )
        self._entry_row(frame, 3, "Duration (s):", self.rates_duration)
        self._entry_row(frame, 4, "Half-life fallback (s):", self.rates_half_life)
        rates_buttons = ttk.Frame(frame)
        rates_buttons.grid(row=5, column=1, sticky="w", pady=(0, 4))
        ttk.Button(
            rates_buttons, text="Load Summary", command=self._load_rates_result_summary
        ).grid(row=0, column=0)
        ttk.Label(
            frame,
            textvariable=self.rates_result_summary,
            wraplength=420,
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.rates_validate
        ).grid(row=7, column=1, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Run Rates", command=self._run_rates).grid(
            row=8, column=1, sticky="w"
        )

    def _build_unfold_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="6. Unfold")
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls.columnconfigure(1, weight=1)

        self.unfold_rates = tk.StringVar(value=str(self.project_dir / "rates.json"))
        self.unfold_response = tk.StringVar(
            value=str(self.project_dir / "response.json")
        )
        self.response_cross_section_file = tk.StringVar(
            value=str(self.project_dir / "cross_sections.json")
        )
        self.response_number_densities_file = tk.StringVar(
            value=str(self.project_dir / "number_densities.json")
        )
        self.response_boundaries_file = tk.StringVar(
            value=str(self.project_dir / "boundaries.json")
        )
        self.response_output = self.unfold_response
        self.response_validate = tk.BooleanVar(value=True)
        self.response_status = tk.StringVar(
            value="Build a response bundle from cross sections, number densities, and group boundaries."
        )
        self.unfold_prior = tk.StringVar(value="")
        self.unfold_method = tk.StringVar(value="gls")
        self.unfold_unc = tk.StringVar(value="0.25")
        self.unfold_cov_model = tk.StringVar(value="diagonal")
        self.unfold_corr_len = tk.StringVar(value="1.0")
        self.unfold_max_iters = tk.StringVar(value="250")
        self.unfold_tolerance = tk.StringVar(value="1e-4")
        self.unfold_chi2_tolerance = tk.StringVar(value="0.01")
        self.unfold_relaxation = tk.StringVar(value="0.8")
        self.unfold_floor = tk.StringVar(value="1e-20")
        self.unfold_convergence_mode = tk.StringVar(value="relative")
        self.unfold_enforce_nonnegativity = tk.BooleanVar(value=True)
        self.unfold_verbose_solver = tk.BooleanVar(value=False)
        self.unfold_output = tk.StringVar(value=str(self.project_dir / "unfold.json"))
        self.unfold_validate = tk.BooleanVar(value=True)
        self.unfold_summary = tk.StringVar(
            value="Run an unfolding workflow to render the adjusted spectrum and solver diagnostics."
        )

        response_frame = ttk.LabelFrame(controls, text="Response builder")
        response_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        response_frame.columnconfigure(1, weight=1)
        self._path_row(
            response_frame,
            0,
            "Cross sections JSON:",
            self.response_cross_section_file,
            save=False,
        )
        self._path_row(
            response_frame,
            1,
            "Number densities JSON:",
            self.response_number_densities_file,
            save=False,
        )
        self._path_row(
            response_frame,
            2,
            "Group boundaries JSON:",
            self.response_boundaries_file,
            save=False,
        )
        self._path_row(
            response_frame, 3, "Response artifact:", self.response_output, save=True
        )
        response_buttons = ttk.Frame(response_frame)
        response_buttons.grid(row=4, column=1, sticky="w", pady=(4, 4))
        ttk.Button(
            response_buttons, text="Build Response", command=self._run_response
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            response_buttons,
            text="Load Response Summary",
            command=self._load_response_preview,
        ).grid(row=0, column=1)
        ttk.Checkbutton(
            response_frame,
            text="Validate artifact schema",
            variable=self.response_validate,
        ).grid(row=5, column=1, sticky="w")
        ttk.Label(
            response_frame,
            textvariable=self.response_status,
            wraplength=360,
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        self._path_row(controls, 7, "Rates artifact:", self.unfold_rates, save=False)
        self._path_row(
            controls, 8, "Response artifact:", self.unfold_response, save=False
        )
        self._path_row(
            controls, 9, "Prior flux JSON (optional):", self.unfold_prior, save=False
        )
        ttk.Label(controls, text="Method:").grid(
            row=10, column=0, sticky="w", padx=(0, 8), pady=4
        )
        method_combo = ttk.Combobox(
            controls,
            textvariable=self.unfold_method,
            values=list(GUI_UNFOLD_METHODS),
            state="readonly",
            width=20,
        )
        method_combo.grid(row=10, column=1, sticky="w")
        method_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_unfold_method_summary()
        )
        self._entry_row(controls, 11, "Prior uncertainty:", self.unfold_unc)
        ttk.Label(controls, text="Prior covariance model:").grid(
            row=12, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            controls,
            textvariable=self.unfold_cov_model,
            values=[m.value for m in cli_app.PriorCovarianceModel],
            state="readonly",
            width=20,
        ).grid(row=12, column=1, sticky="w")
        self._entry_row(controls, 13, "Prior correlation length:", self.unfold_corr_len)
        self._entry_row(controls, 14, "Max iterations:", self.unfold_max_iters)
        self._entry_row(controls, 15, "Tolerance:", self.unfold_tolerance)
        self._entry_row(controls, 16, "Chi² tolerance:", self.unfold_chi2_tolerance)
        self._entry_row(controls, 17, "Relaxation:", self.unfold_relaxation)
        self._entry_row(controls, 18, "Positive floor:", self.unfold_floor)
        ttk.Label(controls, text="MLEM convergence:").grid(
            row=19, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            controls,
            textvariable=self.unfold_convergence_mode,
            values=list(GUI_UNFOLD_MLEM_CONVERGENCE_MODES),
            state="readonly",
            width=20,
        ).grid(row=19, column=1, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Enforce non-negativity (GLS)",
            variable=self.unfold_enforce_nonnegativity,
        ).grid(row=20, column=1, sticky="w", pady=(0, 2))
        ttk.Checkbutton(
            controls,
            text="Verbose iterative solver log",
            variable=self.unfold_verbose_solver,
        ).grid(row=21, column=1, sticky="w", pady=(0, 2))
        self._path_row(
            controls, 22, "Output unfold artifact:", self.unfold_output, save=True
        )
        ttk.Label(
            controls, textvariable=self.unfold_summary, wraplength=360, justify="left"
        ).grid(row=23, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        ttk.Checkbutton(
            controls, text="Validate artifact schema", variable=self.unfold_validate
        ).grid(row=24, column=1, sticky="w", pady=(0, 8))
        unfold_buttons = ttk.Frame(controls)
        unfold_buttons.grid(row=25, column=1, sticky="w")
        ttk.Button(unfold_buttons, text="Run Unfold", command=self._run_unfold).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(
            unfold_buttons, text="Load Result", command=self._load_unfold_result_preview
        ).grid(row=0, column=1)

        plot_frame = ttk.LabelFrame(frame, text="Unfolded spectrum", padding=6)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=1)
        ttk.Label(
            plot_frame,
            text=(
                "View the solved spectrum here. GLS uses prior covariance controls, while GRAVEL and MLEM use the iterative parameters below. "
                "The plot will also show χ² history when the selected solver reports it."
            ),
            wraplength=820,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(
                plot_frame,
                text="matplotlib is not available, so the unfolded-spectrum display cannot be rendered.",
                foreground="#8b0000",
            ).grid(row=1, column=0, sticky="nw")
            self.unfold_figure = None
            self.unfold_canvas = None
            return

        self.unfold_figure = Figure(figsize=(8.8, 5.4), dpi=100)
        self.unfold_canvas = FigureCanvasTkAgg(self.unfold_figure, master=plot_frame)
        self.unfold_canvas.draw()
        self.unfold_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        if NavigationToolbar2Tk is not None:
            toolbar = NavigationToolbar2Tk(
                self.unfold_canvas, plot_frame, pack_toolbar=False
            )
            toolbar.update()
            toolbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._refresh_unfold_method_summary()

    def _build_compare_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="7. Compare")
        frame.columnconfigure(1, weight=1)

        self.compare_unfold = tk.StringVar(value=str(self.project_dir / "unfold.json"))
        self.compare_truth = tk.StringVar(value="")
        self.compare_output = tk.StringVar(
            value=str(self.project_dir / "validation.json")
        )
        self.compare_summary = tk.StringVar(
            value="Validation metrics will summarize end-to-end agreement after Compare runs."
        )
        self.compare_validate = tk.BooleanVar(value=True)

        self._path_row(frame, 0, "Unfold artifact:", self.compare_unfold, save=False)
        self._path_row(frame, 1, "Truth flux JSON:", self.compare_truth, save=False)
        self._path_row(
            frame, 2, "Output validation artifact:", self.compare_output, save=True
        )
        compare_buttons = ttk.Frame(frame)
        compare_buttons.grid(row=3, column=1, sticky="w", pady=(0, 4))
        ttk.Button(
            compare_buttons,
            text="Load Summary",
            command=self._load_compare_result_summary,
        ).grid(row=0, column=0)
        ttk.Label(
            frame, textvariable=self.compare_summary, wraplength=420, justify="left"
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.compare_validate
        ).grid(row=5, column=1, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Run Compare", command=self._run_compare).grid(
            row=6, column=1, sticky="w"
        )

    def _build_standards_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="8. Standards")
        frame.columnconfigure(1, weight=1)

        presets = get_standards_gui_presets()
        preset_labels = [preset.label for preset in presets.values()]
        self.standards_preset = tk.StringVar(value=next(iter(preset_labels), ""))
        self.standards_profile = tk.StringVar(value="")
        self.standards_reaction_category = tk.StringVar(value="all")
        self.standards_target = tk.StringVar(value="")
        self.standards_format = tk.StringVar(value="table")
        self.standards_notes = tk.StringVar(value="")
        self.standards_data_source = tk.StringVar(value="irdff_ii_dosimetry")
        self.standards_custom_source = tk.StringVar(value="")
        self.standards_source_summary = tk.StringVar(
            value="Select a non-gamma data source to browse dosimetry, k0, or flux-wire references."
        )

        ttk.Label(frame, text="Workflow preset:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        preset_box = ttk.Combobox(
            frame,
            textvariable=self.standards_preset,
            values=preset_labels,
            state="readonly",
            width=28,
        )
        preset_box.grid(row=0, column=1, sticky="w")
        preset_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._sync_standards_notes()
        )

        ttk.Label(frame, text="Override profile:").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.standards_profile,
            values=get_gui_profile_choices(),
            state="readonly",
            width=28,
        ).grid(row=1, column=1, sticky="w")

        ttk.Button(
            frame, text="Apply Preset to GUI", command=self._apply_standards_preset
        ).grid(row=2, column=1, sticky="w", pady=(4, 8))

        ttk.Label(frame, text="IRDFF category:").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.standards_reaction_category,
            values=ALLOWED_REACTION_CATEGORIES,
            state="readonly",
            width=20,
        ).grid(row=3, column=1, sticky="w")
        self._entry_row(frame, 4, "Target filter (optional):", self.standards_target)
        ttk.Label(frame, text="Output format:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            frame,
            textvariable=self.standards_format,
            values=["table", "json"],
            state="readonly",
            width=16,
        ).grid(row=5, column=1, sticky="w")
        ttk.Label(frame, text="Reference source:").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        standards_source_combo = ttk.Combobox(
            frame,
            textvariable=self.standards_data_source,
            values=get_gui_data_source_choices(self._custom_data_sources),
            state="readonly",
            width=28,
        )
        standards_source_combo.grid(row=6, column=1, sticky="w")
        standards_source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_standards_source_summary(),
        )
        self._entry_row(
            frame, 7, "Custom source (optional):", self.standards_custom_source
        )
        ttk.Label(
            frame,
            textvariable=self.standards_source_summary,
            wraplength=420,
            justify="left",
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        ttk.Button(
            frame, text="Browse Selected Source", command=self._run_reactions
        ).grid(row=9, column=1, sticky="w", pady=(4, 8))

        k0_frame = ttk.LabelFrame(frame, text="k0 workflow", padding=10)
        k0_frame.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(8, 8))
        k0_frame.columnconfigure(1, weight=1)
        k0_frame.rowconfigure(31, weight=1)

        self.k0_detector_points = tk.StringVar(
            value=str(self.project_dir / "k0_detector_points.json")
        )
        self.k0_detector_output = tk.StringVar(
            value=str(self.project_dir / "detector_characterization.json")
        )
        self.k0_facility_input = tk.StringVar(
            value=str(self.project_dir / "facility_characterization_input.json")
        )
        self.k0_facility_output = tk.StringVar(
            value=str(self.project_dir / "facility_characterization.json")
        )
        self.k0_peaks_input = tk.StringVar(value=str(self.project_dir / "peaks.json"))
        self.k0_spectrum_input = tk.StringVar(
            value=str(self.project_dir / "spectrum.json")
        )
        self.k0_observations_output = tk.StringVar(
            value=str(self.project_dir / "k0_observations.json")
        )
        self.k0_analysis_output = tk.StringVar(
            value=str(self.project_dir / "k0_analysis.json")
        )
        self.k0_library_file = tk.StringVar(value="")
        self.k0_import_input = tk.StringVar(
            value=str(self.project_dir / "kayzero_library")
        )
        self.k0_import_preferred_version = tk.StringVar(value="")
        self.k0_import_output = self.k0_library_file
        self.k0_import_report_output = tk.StringVar(
            value=str(self.project_dir / "kayzero_import_report.json")
        )
        self.k0_aux_library_file = tk.StringVar(value="")
        self.k0_project_id = tk.StringVar(value="")
        self.k0_sample_id = tk.StringVar(value="")
        self.k0_irradiation_id = tk.StringVar(value="")
        self.k0_measurement_id = tk.StringVar(value="")
        self.k0_aggregation_output = tk.StringVar(
            value=str(self.project_dir / "k0_aggregation.json")
        )
        self.k0_qaqc_plan = tk.StringVar(
            value=str(self.project_dir / "k0_qaqc_plan.json")
        )
        self.k0_qaqc_output = tk.StringVar(value=str(self.project_dir / "k0_qaqc.json"))
        self.k0_report_output = tk.StringVar(
            value=str(self.project_dir / "k0_report.json")
        )
        self.k0_sample_mass_g = tk.StringVar(value="0.1")
        self.k0_reference_mass_g = tk.StringVar(value="0.001")
        self.k0_reference_isotope = tk.StringVar(value="Au-198")
        self.k0_irradiation_time_s = tk.StringVar(value="600.0")
        self.k0_decay_time_s = tk.StringVar(value="3600.0")
        self.k0_validate = tk.BooleanVar(value=True)
        self.k0_status = tk.StringVar(
            value="Step through detector characterization, facility characterization, peak normalization, and k0 analysis."
        )

        self._path_row(
            k0_frame, 0, "Kayzero folder / zip:", self.k0_import_input, save=False
        )
        self._entry_row(
            k0_frame, 1, "Preferred version (opt):", self.k0_import_preferred_version
        )
        self._path_row(
            k0_frame, 2, "Governed k0 library:", self.k0_import_output, save=True
        )
        self._path_row(
            k0_frame,
            3,
            "Import report artifact:",
            self.k0_import_report_output,
            save=True,
        )
        ttk.Button(
            k0_frame, text="Import Kayzero", command=self._run_k0_import_kayzero
        ).grid(row=4, column=1, sticky="w", pady=(0, 8))
        self._path_row(
            k0_frame, 5, "Detector points:", self.k0_detector_points, save=False
        )
        self._path_row(
            k0_frame, 6, "Detector artifact:", self.k0_detector_output, save=True
        )
        self._path_row(
            k0_frame, 7, "Facility input:", self.k0_facility_input, save=False
        )
        self._path_row(
            k0_frame, 8, "Facility artifact:", self.k0_facility_output, save=True
        )
        self._path_row(k0_frame, 9, "Peaks artifact:", self.k0_peaks_input, save=False)
        self._path_row(
            k0_frame, 10, "Spectrum artifact:", self.k0_spectrum_input, save=False
        )
        self._path_row(
            k0_frame,
            11,
            "Observations artifact:",
            self.k0_observations_output,
            save=True,
        )
        self._path_row(
            k0_frame, 12, "Analysis artifact:", self.k0_analysis_output, save=True
        )
        self._path_row(
            k0_frame, 13, "k0 library (opt):", self.k0_library_file, save=False
        )
        self._path_row(
            k0_frame, 14, "Aux library (opt):", self.k0_aux_library_file, save=False
        )
        self._entry_row(k0_frame, 15, "Project ID:", self.k0_project_id)
        self._entry_row(k0_frame, 16, "Sample ID:", self.k0_sample_id)
        self._entry_row(k0_frame, 17, "Irradiation ID:", self.k0_irradiation_id)
        self._entry_row(k0_frame, 18, "Measurement ID:", self.k0_measurement_id)
        self._entry_row(k0_frame, 19, "Sample mass (g):", self.k0_sample_mass_g)
        self._entry_row(k0_frame, 20, "Reference mass (g):", self.k0_reference_mass_g)
        self._entry_row(k0_frame, 21, "Reference isotope:", self.k0_reference_isotope)
        self._entry_row(
            k0_frame, 22, "Irradiation time (s):", self.k0_irradiation_time_s
        )
        self._entry_row(k0_frame, 23, "Decay time (s):", self.k0_decay_time_s)
        self._path_row(
            k0_frame, 24, "Aggregation artifact:", self.k0_aggregation_output, save=True
        )
        self._path_row(k0_frame, 25, "QA/QC plan:", self.k0_qaqc_plan, save=False)
        self._path_row(k0_frame, 26, "QA/QC artifact:", self.k0_qaqc_output, save=True)
        self._path_row(
            k0_frame, 27, "Report artifact:", self.k0_report_output, save=True
        )

        k0_button_row = ttk.Frame(k0_frame)
        k0_button_row.grid(row=28, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(
            k0_button_row, text="Run Detector", command=self._run_k0_detector
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            k0_button_row, text="Run Facility", command=self._run_k0_facility
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            k0_button_row, text="Normalize Peaks", command=self._run_k0_normalize
        ).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(k0_button_row, text="Run k0", command=self._run_k0_analyze).grid(
            row=0, column=3, padx=(0, 6)
        )
        ttk.Button(
            k0_button_row, text="Aggregate", command=self._run_k0_aggregate
        ).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(k0_button_row, text="Run QA/QC", command=self._run_k0_qaqc).grid(
            row=0, column=5, padx=(0, 6)
        )
        ttk.Button(
            k0_button_row, text="Build Report", command=self._run_k0_report
        ).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(
            k0_button_row, text="Load Preview", command=self._load_k0_preview
        ).grid(row=0, column=7)
        ttk.Checkbutton(
            k0_frame, text="Validate artifacts", variable=self.k0_validate
        ).grid(row=29, column=1, sticky="w")
        ttk.Label(
            k0_frame, textvariable=self.k0_status, wraplength=760, justify="left"
        ).grid(row=30, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.k0_preview = ScrolledText(k0_frame, height=10, wrap="word")
        self.k0_preview.grid(row=31, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        self.k0_preview.configure(state="disabled")

        astm_frame = ttk.LabelFrame(frame, text="ASTM E261 workflow", padding=10)
        astm_frame.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        astm_frame.columnconfigure(1, weight=1)
        astm_frame.rowconfigure(5, weight=1)

        self.astm_e261_plan = tk.StringVar(
            value=str(self.project_dir / "astm_e261_plan.json")
        )
        self.astm_e261_output = tk.StringVar(
            value=str(self.project_dir / "astm_e261.json")
        )
        self.astm_e261_validate = tk.BooleanVar(value=True)
        self.astm_e261_status = tk.StringVar(
            value="Provide an ASTM E261 JSON plan, then run the workflow to produce a reactor dosimetry bundle."
        )

        self._path_row(astm_frame, 0, "Plan file:", self.astm_e261_plan, save=False)
        self._path_row(
            astm_frame, 1, "Output artifact:", self.astm_e261_output, save=True
        )
        astm_button_row = ttk.Frame(astm_frame)
        astm_button_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(
            astm_button_row, text="Run ASTM E261", command=self._run_astm_e261
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            astm_button_row, text="Load Preview", command=self._load_astm_e261_preview
        ).grid(row=0, column=1)
        ttk.Checkbutton(
            astm_frame, text="Validate artifacts", variable=self.astm_e261_validate
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            astm_frame,
            textvariable=self.astm_e261_status,
            wraplength=760,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.astm_e261_preview = ScrolledText(astm_frame, height=8, wrap="word")
        self.astm_e261_preview.grid(
            row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0)
        )
        self.astm_e261_preview.configure(state="disabled")

        astm_e262_frame = ttk.LabelFrame(frame, text="ASTM E262 workflow", padding=10)
        astm_e262_frame.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        astm_e262_frame.columnconfigure(1, weight=1)
        astm_e262_frame.rowconfigure(5, weight=1)

        self.astm_e262_plan = tk.StringVar(
            value=str(self.project_dir / "astm_e262_plan.json")
        )
        self.astm_e262_output = tk.StringVar(
            value=str(self.project_dir / "astm_e262.json")
        )
        self.astm_e262_validate = tk.BooleanVar(value=True)
        self.astm_e262_status = tk.StringVar(
            value="Provide an ASTM E262 JSON plan, then run the workflow to produce a thermal fluence bundle."
        )

        self._path_row(
            astm_e262_frame, 0, "Plan file:", self.astm_e262_plan, save=False
        )
        self._path_row(
            astm_e262_frame, 1, "Output artifact:", self.astm_e262_output, save=True
        )
        astm_e262_button_row = ttk.Frame(astm_e262_frame)
        astm_e262_button_row.grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            astm_e262_button_row, text="Run ASTM E262", command=self._run_astm_e262
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            astm_e262_button_row,
            text="Load Preview",
            command=self._load_astm_e262_preview,
        ).grid(row=0, column=1)
        ttk.Checkbutton(
            astm_e262_frame, text="Validate artifacts", variable=self.astm_e262_validate
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            astm_e262_frame,
            textvariable=self.astm_e262_status,
            wraplength=760,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.astm_e262_preview = ScrolledText(astm_e262_frame, height=8, wrap="word")
        self.astm_e262_preview.grid(
            row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0)
        )
        self.astm_e262_preview.configure(state="disabled")

        astm_e2005_frame = ttk.LabelFrame(frame, text="ASTM E2005 workflow", padding=10)
        astm_e2005_frame.grid(
            row=13, column=0, columnspan=3, sticky="nsew", pady=(0, 8)
        )
        astm_e2005_frame.columnconfigure(1, weight=1)
        astm_e2005_frame.rowconfigure(5, weight=1)

        self.astm_e2005_plan = tk.StringVar(
            value=str(self.project_dir / "astm_e2005_plan.json")
        )
        self.astm_e2005_output = tk.StringVar(
            value=str(self.project_dir / "astm_e2005.json")
        )
        self.astm_e2005_validate = tk.BooleanVar(value=True)
        self.astm_e2005_status = tk.StringVar(
            value="Provide an ASTM E2005 JSON plan, then run the workflow to produce a benchmark bundle."
        )

        self._path_row(
            astm_e2005_frame, 0, "Plan file:", self.astm_e2005_plan, save=False
        )
        self._path_row(
            astm_e2005_frame, 1, "Output artifact:", self.astm_e2005_output, save=True
        )
        astm_e2005_button_row = ttk.Frame(astm_e2005_frame)
        astm_e2005_button_row.grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            astm_e2005_button_row, text="Run ASTM E2005", command=self._run_astm_e2005
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            astm_e2005_button_row,
            text="Load Preview",
            command=self._load_astm_e2005_preview,
        ).grid(row=0, column=1)
        ttk.Checkbutton(
            astm_e2005_frame,
            text="Validate artifacts",
            variable=self.astm_e2005_validate,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            astm_e2005_frame,
            textvariable=self.astm_e2005_status,
            wraplength=760,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.astm_e2005_preview = ScrolledText(astm_e2005_frame, height=8, wrap="word")
        self.astm_e2005_preview.grid(
            row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0)
        )
        self.astm_e2005_preview.configure(state="disabled")

        astm_e3376_frame = ttk.LabelFrame(frame, text="ASTM E3376 workflow", padding=10)
        astm_e3376_frame.grid(
            row=14, column=0, columnspan=3, sticky="nsew", pady=(0, 8)
        )
        astm_e3376_frame.columnconfigure(1, weight=1)
        astm_e3376_frame.rowconfigure(5, weight=1)

        self.astm_e3376_plan = tk.StringVar(
            value=str(self.project_dir / "astm_e3376_plan.json")
        )
        self.astm_e3376_output = tk.StringVar(
            value=str(self.project_dir / "astm_e3376.json")
        )
        self.astm_e3376_validate = tk.BooleanVar(value=True)
        self.astm_e3376_status = tk.StringVar(
            value="Provide an ASTM E3376 JSON plan, then run the workflow to produce HPGe detection metrics."
        )

        self._path_row(
            astm_e3376_frame, 0, "Plan file:", self.astm_e3376_plan, save=False
        )
        self._path_row(
            astm_e3376_frame, 1, "Output artifact:", self.astm_e3376_output, save=True
        )
        astm_e3376_button_row = ttk.Frame(astm_e3376_frame)
        astm_e3376_button_row.grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            astm_e3376_button_row,
            text="Run ASTM E3376",
            command=self._run_astm_e3376,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            astm_e3376_button_row,
            text="Load Preview",
            command=self._load_astm_e3376_preview,
        ).grid(row=0, column=1)
        ttk.Checkbutton(
            astm_e3376_frame,
            text="Validate artifacts",
            variable=self.astm_e3376_validate,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            astm_e3376_frame,
            textvariable=self.astm_e3376_status,
            wraplength=760,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.astm_e3376_preview = ScrolledText(astm_e3376_frame, height=8, wrap="word")
        self.astm_e3376_preview.grid(
            row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0)
        )
        self.astm_e3376_preview.configure(state="disabled")

        rafm_frame = ttk.LabelFrame(
            frame, text="RAFM validation + benchmark workflows", padding=10
        )
        rafm_frame.grid(row=15, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        rafm_frame.columnconfigure(1, weight=1)

        self.rafm_example_root = tk.StringVar(
            value=str(self.project_dir / "examples" / "RAFM_irradiation")
        )
        self.rafm_raw_results_root = tk.StringVar(
            value=str(self.project_dir / "artifacts" / "rafm_raw")
        )
        self.rafm_qg_results_root = tk.StringVar(
            value=str(self.project_dir / "artifacts" / "rafm_qg")
        )
        self.rafm_comparison_root = tk.StringVar(
            value=str(self.project_dir / "artifacts" / "rafm_compare")
        )
        self.rafm_max_spectra = tk.StringVar(value="")
        self.rafm_flux_wire_counting_method = tk.StringVar(value="iec_tiered")
        self.rafm_generic_counting_method = tk.StringVar(value="gaussian_fit")
        self.rafm_enforce_thresholds = tk.BooleanVar(value=True)
        self.rafm_status = tk.StringVar(
            value="Run native FluxForge validation against bundled RAFM examples. This remains a desktop workflow with no browser dependency."
        )

        self._directory_row(rafm_frame, 0, "Example root:", self.rafm_example_root)
        self._directory_row(
            rafm_frame, 1, "Raw results root:", self.rafm_raw_results_root
        )
        self._directory_row(
            rafm_frame, 2, "QG results root:", self.rafm_qg_results_root
        )
        self._directory_row(
            rafm_frame, 3, "Comparison root:", self.rafm_comparison_root
        )
        self._entry_row(rafm_frame, 4, "Max spectra (opt):", self.rafm_max_spectra)
        ttk.Label(rafm_frame, text="Flux-wire counting:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            rafm_frame,
            textvariable=self.rafm_flux_wire_counting_method,
            values=list(GUI_RAFM_COUNTING_METHODS),
            state="readonly",
            width=18,
        ).grid(row=5, column=1, sticky="w")
        ttk.Label(rafm_frame, text="Generic counting:").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            rafm_frame,
            textvariable=self.rafm_generic_counting_method,
            values=list(GUI_RAFM_COUNTING_METHODS),
            state="readonly",
            width=18,
        ).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(
            rafm_frame,
            text="Enforce parity thresholds",
            variable=self.rafm_enforce_thresholds,
        ).grid(row=7, column=1, sticky="w")
        rafm_buttons = ttk.Frame(rafm_frame)
        rafm_buttons.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(
            rafm_buttons,
            text="Run RAFM Validation",
            command=self._run_rafm_validate,
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            rafm_buttons,
            text="Run QG Benchmark",
            command=self._run_rafm_qg_benchmark,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            rafm_buttons,
            text="Compare Branches",
            command=self._run_rafm_compare_branches,
        ).grid(row=0, column=2)
        ttk.Label(
            rafm_frame,
            textvariable=self.rafm_status,
            wraplength=760,
            justify="left",
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        notes = ScrolledText(frame, wrap="word", height=12)
        notes.grid(row=16, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        frame.rowconfigure(10, weight=1)
        frame.rowconfigure(11, weight=1)
        frame.rowconfigure(12, weight=1)
        frame.rowconfigure(13, weight=1)
        frame.rowconfigure(14, weight=1)
        frame.rowconfigure(15, weight=1)
        frame.rowconfigure(16, weight=1)
        self.standards_notes_box = notes
        self._sync_standards_notes()
        self._refresh_standards_source_summary()

    def _build_physics_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="9. Physics")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.physics_notebook = ttk.Notebook(frame)
        self.physics_notebook.grid(row=0, column=0, sticky="nsew")

        stacked = ttk.Frame(self.physics_notebook, padding=12)
        stacked.columnconfigure(1, weight=1)
        self.physics_notebook.add(stacked, text="Stacked target")
        self.stacked_projectile = tk.StringVar(value=Projectile.PROTON.name)
        self.stacked_beam_energy = tk.StringVar(value="30.0")
        self.stacked_foil_material = tk.StringVar(value="aluminum")
        self.stacked_foil_thickness = tk.StringVar(value="25.0")
        self.stacked_foil_reaction = tk.StringVar(value="")
        self.stacked_foil_target = tk.StringVar(value="")
        self.stacked_foil_product = tk.StringVar(value="")
        ttk.Label(stacked, text="Projectile:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            stacked,
            textvariable=self.stacked_projectile,
            values=[projectile.name for projectile in Projectile],
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w")
        self._entry_row(stacked, 1, "Beam energy (MeV):", self.stacked_beam_energy)
        ttk.Label(stacked, text="Foil material:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            stacked,
            textvariable=self.stacked_foil_material,
            values=sorted(STANDARD_MATERIALS),
            state="readonly",
            width=24,
        ).grid(row=2, column=1, sticky="w")
        self._entry_row(stacked, 3, "Thickness (μm):", self.stacked_foil_thickness)
        self._entry_row(stacked, 4, "Reaction (optional):", self.stacked_foil_reaction)
        self._entry_row(stacked, 5, "Target isotope:", self.stacked_foil_target)
        self._entry_row(stacked, 6, "Product isotope:", self.stacked_foil_product)
        stacked_buttons = ttk.Frame(stacked)
        stacked_buttons.grid(row=7, column=1, sticky="w", pady=(6, 6))
        ttk.Button(
            stacked_buttons, text="Add foil", command=self._physics_add_foil
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            stacked_buttons,
            text="Solve stack",
            command=self._physics_run_stacked_target,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            stacked_buttons, text="Clear foils", command=self._physics_clear_foils
        ).grid(row=0, column=2)
        self.stacked_foil_table = ttk.Treeview(
            stacked,
            columns=("material", "thickness", "reaction", "target", "product"),
            show="headings",
            height=7,
        )
        for column, text, width in (
            ("material", "Material", 110),
            ("thickness", "μm", 70),
            ("reaction", "Reaction", 110),
            ("target", "Target", 85),
            ("product", "Product", 85),
        ):
            self.stacked_foil_table.heading(column, text=text)
            self.stacked_foil_table.column(column, width=width, anchor="w")
        self.stacked_foil_table.grid(
            row=8, column=0, columnspan=3, sticky="nsew", pady=(4, 0)
        )
        stacked.rowconfigure(8, weight=1)
        self.stacked_summary = tk.StringVar(
            value="Add foils, then solve the Physics-style stacked-target energy loss profile."
        )
        ttk.Label(
            stacked, textvariable=self.stacked_summary, wraplength=760, justify="left"
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        decay = ttk.Frame(self.physics_notebook, padding=12)
        decay.columnconfigure(1, weight=1)
        self.physics_notebook.add(decay, text="Decay chain")
        self.decay_parent = tk.StringVar(value="Mn56")
        self.decay_parent_half_life = tk.StringVar(value="9285.6")
        self.decay_daughter = tk.StringVar(value="Fe56")
        self.decay_daughter_half_life = tk.StringVar(value="1e30")
        self.decay_initial_activity = tk.StringVar(value="1000.0")
        self.decay_end_time = tk.StringVar(value="8.0")
        self.decay_time_units = tk.StringVar(value="h")
        self.decay_production_rate = tk.StringVar(value="0.0")
        self.decay_summary = tk.StringVar(
            value="Solve simple Bateman chains with parent/daughter activities for Physics-style review."
        )
        self._entry_row(decay, 0, "Parent nuclide:", self.decay_parent)
        self._entry_row(decay, 1, "Parent half-life (s):", self.decay_parent_half_life)
        self._entry_row(decay, 2, "Daughter nuclide:", self.decay_daughter)
        self._entry_row(
            decay, 3, "Daughter half-life (s):", self.decay_daughter_half_life
        )
        self._entry_row(decay, 4, "Initial activity (Bq):", self.decay_initial_activity)
        self._entry_row(decay, 5, "End time:", self.decay_end_time)
        ttk.Label(decay, text="Units:").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            decay,
            textvariable=self.decay_time_units,
            values=["s", "min", "h", "d"],
            state="readonly",
            width=12,
        ).grid(row=6, column=1, sticky="w")
        self._entry_row(
            decay, 7, "Production rate (atoms/s):", self.decay_production_rate
        )
        ttk.Button(
            decay, text="Solve decay chain", command=self._physics_run_decay_chain
        ).grid(row=8, column=1, sticky="w", pady=(6, 6))
        ttk.Label(
            decay, textvariable=self.decay_summary, wraplength=760, justify="left"
        ).grid(row=9, column=0, columnspan=3, sticky="ew")

    def _build_report_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="10. Report")
        frame.columnconfigure(1, weight=1)

        self.report_spectrum = tk.StringVar(
            value=str(self.project_dir / "spectrum.json")
        )
        self.report_peaks = tk.StringVar(value=str(self.project_dir / "peaks.json"))
        self.report_lines = tk.StringVar(
            value=str(self.project_dir / "activities.json")
        )
        self.report_rates = tk.StringVar(value=str(self.project_dir / "rates.json"))
        self.report_unfold = tk.StringVar(value=str(self.project_dir / "unfold.json"))
        self.report_validation = tk.StringVar(
            value=str(self.project_dir / "validation.json")
        )
        self.report_validation_results_root = tk.StringVar(value="")
        self.report_output = tk.StringVar(value=str(self.project_dir / "report.json"))
        self.report_figure_dir = tk.StringVar(
            value=str(self.project_dir / "report_figures")
        )
        self.plots_unfold = self.report_unfold
        self.plots_rates = self.report_rates
        self.plots_response = tk.StringVar(
            value=str(self.project_dir / "response.json")
        )
        self.plots_prior_flux = tk.StringVar(value="")
        self.plots_output_dir = tk.StringVar(value=str(self.project_dir / "plots"))
        self.plots_format = tk.StringVar(value="png")
        self.plots_example = tk.BooleanVar(value=False)
        self.plots_include_response_plot = tk.BooleanVar(value=True)
        self.plots_validate = tk.BooleanVar(value=True)
        self.plots_status = tk.StringVar(
            value="Generate the headless master plot suite from bundled example inputs or current workflow artifacts."
        )
        self.report_validate = tk.BooleanVar(value=True)
        self.report_export_figures = tk.BooleanVar(value=True)
        self.report_status = tk.StringVar(
            value="Report exports can bundle a text report, summary tables, spectrum, activity, rate, unfold, and compare plots."
        )

        self._path_row(frame, 0, "Spectrum artifact:", self.report_spectrum, save=False)
        self._path_row(frame, 1, "Peaks artifact:", self.report_peaks, save=False)
        self._path_row(frame, 2, "Activities artifact:", self.report_lines, save=False)
        self._path_row(frame, 3, "Rates artifact:", self.report_rates, save=False)
        self._path_row(frame, 4, "Unfold artifact:", self.report_unfold, save=False)
        self._path_row(
            frame, 5, "Validation artifact:", self.report_validation, save=False
        )
        self._directory_row(
            frame, 6, "Validation results root:", self.report_validation_results_root
        )
        ttk.Button(
            frame, text="Auto-fill", command=self._auto_fill_report_from_validation_dir
        ).grid(row=6, column=3, padx=(8, 0))
        self._path_row(
            frame, 7, "Output report artifact:", self.report_output, save=True
        )
        self._directory_row(
            frame, 8, "Figure bundle directory:", self.report_figure_dir
        )
        plots_frame = ttk.LabelFrame(frame, text="Master plot suite", padding=10)
        plots_frame.grid(row=9, column=0, columnspan=4, sticky="nsew", pady=(10, 8))
        plots_frame.columnconfigure(1, weight=1)
        self._path_row(
            plots_frame, 0, "Unfold artifact:", self.plots_unfold, save=False
        )
        self._path_row(
            plots_frame, 1, "Response artifact:", self.plots_response, save=False
        )
        self._path_row(plots_frame, 2, "Rates artifact:", self.plots_rates, save=False)
        self._path_row(
            plots_frame, 3, "Prior flux JSON:", self.plots_prior_flux, save=False
        )
        self._directory_row(
            plots_frame, 4, "Plot output directory:", self.plots_output_dir
        )
        ttk.Label(plots_frame, text="Format:").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            plots_frame,
            textvariable=self.plots_format,
            values=["png", "pdf", "both"],
            state="readonly",
            width=12,
        ).grid(row=5, column=1, sticky="w")
        ttk.Checkbutton(
            plots_frame,
            text="Use bundled example inputs",
            variable=self.plots_example,
        ).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(
            plots_frame,
            text="Include response matrix plot",
            variable=self.plots_include_response_plot,
        ).grid(row=7, column=1, sticky="w")
        ttk.Checkbutton(
            plots_frame,
            text="Validate artifact schema",
            variable=self.plots_validate,
        ).grid(row=8, column=1, sticky="w")
        ttk.Button(plots_frame, text="Run Plot Suite", command=self._run_plots).grid(
            row=9, column=1, sticky="w", pady=(4, 4)
        )
        ttk.Label(
            plots_frame,
            textvariable=self.plots_status,
            wraplength=520,
            justify="left",
        ).grid(row=10, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(
            frame, text="Validate artifact schema", variable=self.report_validate
        ).grid(row=10, column=1, sticky="w", pady=(0, 4))
        ttk.Checkbutton(
            frame,
            text="Export figure bundle after report",
            variable=self.report_export_figures,
        ).grid(row=11, column=1, sticky="w", pady=(0, 8))
        button_row = ttk.Frame(frame)
        button_row.grid(row=12, column=1, sticky="w")
        ttk.Button(button_row, text="Run Report", command=self._run_report).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            button_row, text="Load Report", command=self._load_report_preview
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(
            button_row, text="Export Figures", command=self._export_report_figures
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(
            frame, textvariable=self.report_status, wraplength=520, justify="left"
        ).grid(row=13, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        frame.rowconfigure(14, weight=1)
        self.report_preview = ScrolledText(frame, height=18, wrap="word")
        self.report_preview.grid(
            row=14, column=0, columnspan=4, sticky="nsew", pady=(10, 0)
        )
        self.report_preview.configure(state="disabled")

    def _entry_row(
        self, frame: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(frame, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(frame, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=4
        )

    def _path_row(
        self, frame: ttk.Frame, row: int, label: str, variable: tk.StringVar, save: bool
    ) -> None:
        ttk.Label(frame, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = ttk.Entry(frame, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        browse_command = (
            (lambda: self._browse_save_path(variable))
            if save
            else (lambda: self._browse_open_path(variable))
        )
        ttk.Button(
            frame,
            text="Browse...",
            command=browse_command,
        ).grid(row=row, column=2, padx=(8, 0))

    def _directory_row(
        self, frame: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(frame, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(frame, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        ttk.Button(
            frame,
            text="Browse...",
            command=lambda: self._browse_directory_path(variable),
        ).grid(row=row, column=2, padx=(8, 0))
