"""Legacy Tk desktop GUI shell for FluxForge.

This package is kept as an archive/legacy fallback while the native Qt shell is
rebuilt under ``fluxforge.gui``.

The GUI is intentionally CLI-first:
- Every button maps to an existing ``fluxforge`` subcommand handler.
- The equivalent CLI command is shown in the run log.
- Users can copy the last generated CLI command for scripting/reproducibility.
"""

from __future__ import annotations

import contextlib
import io
import json
import shlex
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    Figure = None
from fluxforge.analysis.detector_calibration import (
    EfficiencyPoint,
    fit_efficiency_curve,
)
from fluxforge.analysis.peakfit import (
    fit_hypermet_peak,
    fit_multiple_peaks,
    fit_single_peak,
)
from fluxforge.core.runtime import offline_mode_enabled
from fluxforge.data.efficiency import CALIBRATION_SOURCES, EfficiencyCurve
from fluxforge.data.flux_wire_catalog import (
    get_flux_wire_catalog_entry,
    list_flux_wire_isotopes,
)
from fluxforge.data.irdff_access import get_default_library
from fluxforge.data.nndc import get_nuclear_data
from fluxforge.data.nuclear_data_sources import (
    load_gamma_identification_source,
    summarize_nuclear_data_source,
)
from fluxforge.io.artifacts import (
    read_k0_analysis_bundle,
    read_line_activities,
    read_report_bundle,
    read_reaction_rates,
    read_response_bundle,
    read_unfold_result,
    read_validation_bundle,
    write_report_bundle,
    write_spectrum_file,
)
from fluxforge.triga.cd_ratio import STANDARD_MONITORS
from fluxforge_gui import constants as gui_constants
from fluxforge_gui import presets as gui_presets
from fluxforge_gui.models import (
    GuiCalibrationFit,
    GuiCalibrationPoint,
    GuiDiagnosticPlot,
    GuiDiagnosticSeries,
    GuiEfficiencyCalibrationPoint,
    GuiManualRegion,
    GuiPeakCountingResult,
    GuiSpectrumPeak,
    GuiSpectrumPreview,
    GuiSpectrumSeries,
)
from fluxforge_gui.presets import (
    build_standards_preset_values,
    get_standards_gui_presets,
)
from fluxforge_gui.reporting import (
    build_gui_astm_e2005_preview,
    build_gui_astm_e261_preview,
    build_gui_astm_e262_preview,
    build_gui_astm_e3376_preview,
    build_gui_k0_preview,
    build_gui_report_preview,
    discover_gui_validation_report_inputs,
    render_gui_activity_result,
    render_gui_rate_result,
    render_gui_unfold_result,
    render_gui_validation_result,
    summarize_gui_activity_result,
    summarize_gui_rate_result,
    summarize_gui_unfold_result,
    summarize_gui_validation_result,
)
from fluxforge_gui.spectrum_ops import (
    _coerce_path_tokens,
    _series_to_spectrum,
    auto_detect_gui_peaks,
    build_calibration_residual_plot,
    build_efficiency_fit_diagnostic_plot,
    build_gui_spectrum_preview,
    build_peak_count_diagnostic_plot,
    combine_gui_spectrum_series,
    count_gui_peak,
    fit_gui_constrained_multiplet,
    fit_gui_energy_calibration,
    parse_gui_constraint_matrix,
    render_gui_spectrum_preview,
    save_gui_spectrum_preview_image,
)
from .commands import CommandsMixin
from .ui_builder import UiBuilderMixin


ALLOWED_REACTION_CATEGORIES = gui_constants.ALLOWED_REACTION_CATEGORIES
get_gui_profile_choices = gui_presets.get_gui_profile_choices


class FluxForgeGui(UiBuilderMixin, CommandsMixin):
    """Main Tk application window."""

    def __init__(self, root: tk.Tk, project_dir: Path | None = None) -> None:
        self.root = root
        self.root.title("FluxForge GUI (Spectrum + Workflow MVP)")
        self.root.geometry("1400x900")
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.offline_mode = offline_mode_enabled()
        self.last_cli_command = ""
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._preview_state: GuiSpectrumPreview | None = None
        self._preview_state_raw: GuiSpectrumPreview | None = None
        self._manual_regions: list[GuiManualRegion] = []
        self._pending_roi_start_keV: float | None = None
        self._active_roi_drag: tuple[int, str] | None = None
        self._buffer_paths: list[Path] = []
        self._selected_peak_energy_keV: float | None = None
        self._calibration_points: list[GuiCalibrationPoint] = []
        self._calibration_fit: GuiCalibrationFit | None = None
        self._peak_count_result: GuiPeakCountingResult | None = None
        self._stacked_foils: list[dict[str, str | float]] = []
        self._preview_diagnostic_plot: GuiDiagnosticPlot | None = None
        self._efficiency_points: list[GuiEfficiencyCalibrationPoint] = []
        self._efficiency_curve: EfficiencyCurve | None = None
        self._custom_data_sources: list[str | Path] = []

        self._configure_styles()

        self._build_layout()
        self._build_ingest_tab()
        self._build_spectrum_tab()
        self._build_peaks_tab()
        self._build_activity_tab()
        self._build_rates_tab()
        self._build_unfold_tab()
        self._build_compare_tab()
        self._build_standards_tab()
        self._build_physics_tab()
        self._build_report_tab()
        self._append_log(
            "FluxForge GUI is a native desktop application; no browser runtime is required."
        )
        if self.offline_mode:
            self._append_log(
                "Offline mode enabled: remote HTTP(S) data sources and runtime downloads are disabled."
            )
        self._append_log(f"Project directory: {self.project_dir}")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared UI helpers
    # ------------------------------------------------------------------

    def _browse_open_path(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(initialdir=str(self.project_dir))
        if path:
            variable.set(path)

    def _browse_save_path(self, variable: tk.StringVar) -> None:
        path = filedialog.asksaveasfilename(initialdir=str(self.project_dir))
        if path:
            variable.set(path)

    def _browse_directory_path(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=str(self.project_dir))
        if path:
            variable.set(path)

    def _choose_project_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self.project_dir))
        if not path:
            return
        self.project_dir = Path(path)
        self._append_log(f"Project directory changed to: {self.project_dir}")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _dispatch(
        self, func, args: Namespace, cli_tokens: list[str], on_success=None
    ) -> None:
        if self._busy:
            messagebox.showinfo("FluxForge GUI", "A command is already running.")
            return
        command_line = "fluxforge " + " ".join(shlex.quote(tok) for tok in cli_tokens)
        self.last_cli_command = command_line
        self.copy_btn.configure(state="normal")
        self._append_log(f"\n$ {command_line}")
        self._set_busy(True)
        future = self._executor.submit(self._execute_command, func, args)
        future.add_done_callback(
            lambda f, callback=on_success: self.root.after(
                0, self._on_command_done, f, callback
            )
        )

    @staticmethod
    def _execute_command(func, args: Namespace) -> str:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            func(args)
        return stream.getvalue().strip()

    def _on_command_done(self, future: Future, on_success=None) -> None:
        self._set_busy(False)
        try:
            output = future.result()
            if output:
                self._append_log(output)
            else:
                self._append_log("Command completed.")
            if on_success is not None:
                on_success()
        except Exception as exc:
            self._append_log(f"ERROR: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.status_var.set("Running..." if busy else "Idle")

    # ------------------------------------------------------------------
    # Logging + clipboard
    # ------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _copy_last_cli(self) -> None:
        if not self.last_cli_command:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_cli_command)
        self._append_log(f"Copied CLI: {self.last_cli_command}")

    @staticmethod
    def _optional_float(raw: str) -> float | None:
        text = raw.strip()
        return float(text) if text else None

    @staticmethod
    def _optional_path(raw: str) -> Path | None:
        text = raw.strip()
        return Path(text) if text else None

    def _refresh_unfold_method_summary(self) -> None:
        method = self.unfold_method.get().strip().lower()
        if method == "gls":
            summary = (
                "GLS runs the ASTM/STAYSL-style adjustment using the prior covariance controls. "
                "Use this for standards-driven spectrum adjustment with explicit prior information."
            )
        elif method == "gravel":
            summary = (
                "GRAVEL uses multiplicative SAND-II-style iterations. "
                "Tune iterations, tolerance, χ² tolerance, relaxation, and positive floor for convergence behavior."
            )
        else:
            summary = (
                "MLEM runs expectation-maximization updates from the prior/initial spectrum. "
                "Tune iterations, tolerance, χ² tolerance, relaxation, positive floor, and convergence mode."
            )
        self.unfold_summary.set(summary)

    def _render_unfold_result_preview(self, payload: dict[str, Any]) -> None:
        self.unfold_summary.set(summarize_gui_unfold_result(payload))
        if (
            getattr(self, "unfold_canvas", None) is None
            or getattr(self, "unfold_figure", None) is None
        ):
            return
        render_gui_unfold_result(payload, figure=self.unfold_figure)
        self.unfold_canvas.draw_idle()

    def _render_activity_result_preview(self, payload: dict[str, Any]) -> None:
        if (
            getattr(self, "activity_canvas", None) is None
            or getattr(self, "activity_figure", None) is None
        ):
            return
        render_gui_activity_result(
            payload,
            figure=self.activity_figure,
            y_scale=self.activity_plot_y_scale.get(),
        )
        self.activity_canvas.draw_idle()

    def _render_rates_result_preview(self, payload: dict[str, Any]) -> None:
        if (
            getattr(self, "rates_canvas", None) is None
            or getattr(self, "rates_figure", None) is None
        ):
            return
        render_gui_rate_result(
            payload,
            figure=self.rates_figure,
            y_scale=self.rates_plot_y_scale.get(),
        )
        self.rates_canvas.draw_idle()

    def _load_unfold_result_preview(self) -> None:
        try:
            payload = read_unfold_result(Path(self.unfold_output.get()))
        except Exception as exc:
            self._append_log(f"ERROR loading unfold result: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self._render_unfold_result_preview(payload)
        self._append_log(f"Loaded unfold preview from {self.unfold_output.get()}")

    def _load_activity_result_summary(self) -> None:
        try:
            payload = read_line_activities(Path(self.activity_output.get()))
        except Exception as exc:
            self._append_log(f"ERROR loading activity result: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self.activity_result_summary.set(summarize_gui_activity_result(payload))
        self._render_activity_result_preview(payload)
        self._append_log(f"Loaded activity summary from {self.activity_output.get()}")

    def _load_rates_result_summary(self) -> None:
        try:
            payload = read_reaction_rates(Path(self.rates_output.get()))
        except Exception as exc:
            self._append_log(f"ERROR loading rate result: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self.rates_result_summary.set(summarize_gui_rate_result(payload))
        self._render_rates_result_preview(payload)
        self._append_log(f"Loaded reaction-rate summary from {self.rates_output.get()}")

    def _refresh_activity_plot_if_ready(self) -> None:
        output = Path(self.activity_output.get().strip())
        if output.exists():
            self._load_activity_result_summary()

    def _refresh_rates_plot_if_ready(self) -> None:
        output = Path(self.rates_output.get().strip())
        if output.exists():
            self._load_rates_result_summary()

    def _load_compare_result_summary(self) -> None:
        try:
            payload = read_validation_bundle(Path(self.compare_output.get()))
        except Exception as exc:
            self._append_log(f"ERROR loading validation result: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self.compare_summary.set(summarize_gui_validation_result(payload))
        self._append_log(f"Loaded validation summary from {self.compare_output.get()}")

    @staticmethod
    def _save_report_figure(figure: Figure | None, output_path: Path) -> Path | None:
        if figure is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=140)
        return output_path

    def _export_report_figures(self) -> None:
        try:
            metadata = self._export_report_figures_impl()
        except Exception as exc:
            self._append_log(f"ERROR exporting report figures: {exc}")
            self.report_status.set(f"Figure export failed: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        exported = metadata.get("items", {})
        manifest = metadata.get("manifest")
        count = len(exported)
        summary = f"Exported {count} report figure(s)"
        if manifest:
            summary += f" with manifest {manifest}"
        self.report_status.set(summary)
        self._append_log(summary)

    def _export_report_figures_impl(self) -> dict[str, Any]:
        figure_dir_raw = self.report_figure_dir.get().strip()
        if not figure_dir_raw:
            raise ValueError("Choose a figure bundle directory first.")

        figure_dir = Path(figure_dir_raw)
        figure_dir.mkdir(parents=True, exist_ok=True)
        items: dict[str, dict[str, str]] = {}

        spectrum_path = self.report_spectrum.get().strip()
        peaks_path = self.report_peaks.get().strip()
        if spectrum_path:
            preview = build_gui_spectrum_preview(
                spectrum_path,
                peaks_path=peaks_path or None,
            )
            spectrum_output = figure_dir / "spectrum_preview.png"
            save_gui_spectrum_preview_image(preview, spectrum_output)
            items["spectrum_preview"] = {
                "title": "Spectrum preview",
                "kind": "spectrum",
                "path": spectrum_output.name,
                "source": spectrum_path,
            }

        activities_path = self.report_lines.get().strip()
        if activities_path:
            payload = read_line_activities(Path(activities_path))
            activity_output = figure_dir / "activity_summary.png"
            saved = self._save_report_figure(
                render_gui_activity_result(payload), activity_output
            )
            if saved is not None:
                items["activity_summary"] = {
                    "title": "Activity summary",
                    "kind": "activity",
                    "path": saved.name,
                    "source": activities_path,
                }

        rates_path = self.report_rates.get().strip()
        if rates_path:
            payload = read_reaction_rates(Path(rates_path))
            rates_output = figure_dir / "rate_summary.png"
            saved = self._save_report_figure(
                render_gui_rate_result(payload), rates_output
            )
            if saved is not None:
                items["rate_summary"] = {
                    "title": "Reaction rate summary",
                    "kind": "rates",
                    "path": saved.name,
                    "source": rates_path,
                }

        unfold_path = self.report_unfold.get().strip()
        if unfold_path:
            payload = read_unfold_result(Path(unfold_path))
            unfold_output = figure_dir / "unfold_summary.png"
            saved = self._save_report_figure(
                render_gui_unfold_result(payload), unfold_output
            )
            if saved is not None:
                items["unfold_summary"] = {
                    "title": "Unfolded spectrum diagnostics",
                    "kind": "unfold",
                    "path": saved.name,
                    "source": unfold_path,
                }

        validation_path = self.report_validation.get().strip()
        if validation_path:
            payload = read_validation_bundle(Path(validation_path))
            validation_output = figure_dir / "validation_summary.png"
            saved = self._save_report_figure(
                render_gui_validation_result(payload), validation_output
            )
            if saved is not None:
                items["validation_summary"] = {
                    "title": "Validation comparison",
                    "kind": "validation",
                    "path": saved.name,
                    "source": validation_path,
                }

        if not items:
            raise ValueError(
                "No report figures were exported. Provide at least one readable report input artifact."
            )

        manifest_path = figure_dir / "figure_manifest.json"
        manifest_payload = {
            "directory": str(figure_dir),
            "count": len(items),
            "items": items,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8"
        )

        report_path = Path(self.report_output.get())
        if report_path.exists():
            report_payload = read_report_bundle(report_path)
            write_report_bundle(
                report_path,
                summary=dict(report_payload.get("summary", {})),
                inputs=dict(report_payload.get("inputs", {})) or None,
                figures={
                    "directory": str(figure_dir),
                    "manifest": manifest_path.name,
                    "items": items,
                },
                tables=dict(report_payload.get("tables", {})) or None,
                text_report=dict(report_payload.get("text_report", {})) or None,
            )

        return {
            "directory": str(figure_dir),
            "manifest": manifest_path.name,
            "items": items,
        }

    def _set_report_preview_text(self, text: str) -> None:
        self.report_preview.configure(state="normal")
        self.report_preview.delete("1.0", "end")
        self.report_preview.insert("1.0", text)
        self.report_preview.configure(state="disabled")

    def _load_report_preview(self) -> None:
        try:
            payload = read_report_bundle(Path(self.report_output.get()))
            preview = build_gui_report_preview(payload, self.report_output.get())
        except Exception as exc:
            self._append_log(f"ERROR loading report preview: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self._set_report_preview_text(preview)
        text_path = (payload.get("text_report") or {}).get("path")
        status = f"Loaded report preview from {self.report_output.get()}"
        if text_path:
            status += f" ({text_path})"
        self.report_status.set(status)
        self._append_log(status)

    def _preview_calibration_coeffs(self) -> tuple[float, ...]:
        return (
            float(self.preview_calibration_c0.get() or 0.0),
            float(self.preview_calibration_c1.get() or 0.0),
            float(self.preview_calibration_c2.get() or 0.0),
            float(self.preview_calibration_c3.get() or 0.0),
        )

    def _series_with_coefficients(
        self, series: GuiSpectrumSeries, coefficients: tuple[float, ...]
    ) -> GuiSpectrumSeries:
        energies = np.zeros_like(series.channels, dtype=float)
        for power, coeff in enumerate(coefficients):
            energies += float(coeff) * (series.channels**power)
        return GuiSpectrumSeries(
            label=series.label,
            channels=np.asarray(series.channels, dtype=float),
            energies_keV=energies,
            counts=np.asarray(series.counts, dtype=float),
            calibration_coeffs=tuple(float(value) for value in coefficients),
        )

    def _preview_with_current_calibration(self) -> GuiSpectrumPreview | None:
        if self._preview_state_raw is None:
            return None
        coefficients = self._preview_calibration_coeffs()
        primary = self._series_with_coefficients(
            self._preview_state_raw.primary, coefficients
        )
        overlays = tuple(
            self._series_with_coefficients(item, coefficients)
            for item in self._preview_state_raw.overlays
        )
        peaks: list[GuiSpectrumPeak] = []
        source_peaks = self._preview_state_raw.peaks
        for peak in source_peaks:
            if peak.channel is None:
                peaks.append(peak)
                continue
            energy = sum(
                coeff * (float(peak.channel) ** power)
                for power, coeff in enumerate(coefficients)
            )
            peaks.append(
                GuiSpectrumPeak(
                    energy_keV=float(energy),
                    area=peak.area,
                    channel=peak.channel,
                    label=peak.label,
                )
            )
        return GuiSpectrumPreview(
            primary=primary, overlays=overlays, peaks=tuple(peaks)
        )

    def _selected_region_label(self) -> str | None:
        selection = self.preview_roi_table.selection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._manual_regions):
            return self._manual_regions[index].label
        return None

    def _render_current_preview(self) -> None:
        if self.preview_canvas is None or self.preview_figure is None:
            return
        preview = self._preview_with_current_calibration()
        if preview is None:
            return
        self._preview_state = preview
        render_gui_spectrum_preview(
            preview,
            manual_regions=tuple(self._manual_regions),
            selected_peak_energy_keV=self._selected_peak_energy_keV,
            selected_region_label=self._selected_region_label(),
            diagnostic_plot=self._preview_diagnostic_plot,
            y_log=self.preview_y_scale.get() == "log",
            x_log=self.preview_x_scale.get() == "log",
            x_min_keV=self._optional_float(self.preview_x_min.get()),
            x_max_keV=self._optional_float(self.preview_x_max.get()),
            group_peak_colors=self.preview_peak_color_mode.get() == "isotope",
            figure=self.preview_figure,
        )
        self.preview_canvas.draw_idle()

    def _refresh_buffer_list(self) -> None:
        self.preview_buffer_list.delete(0, tk.END)
        for path in self._buffer_paths:
            self.preview_buffer_list.insert(tk.END, str(path))

    def _selected_buffer_paths(self) -> list[Path]:
        return [
            self._buffer_paths[index]
            for index in self.preview_buffer_list.curselection()
            if 0 <= index < len(self._buffer_paths)
        ]

    def _buffer_add_current(self) -> None:
        current = self.preview_input.get().strip()
        if not current:
            return
        path = Path(current)
        if path not in self._buffer_paths:
            self._buffer_paths.append(path)
            self._refresh_buffer_list()

    def _buffer_add_file(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(self.project_dir))
        if not path:
            return
        resolved = Path(path)
        if resolved not in self._buffer_paths:
            self._buffer_paths.append(resolved)
            self._refresh_buffer_list()

    def _buffer_remove_selected(self) -> None:
        selected = set(self.preview_buffer_list.curselection())
        if not selected:
            return
        self._buffer_paths = [
            path
            for index, path in enumerate(self._buffer_paths)
            if index not in selected
        ]
        self._refresh_buffer_list()

    def _buffer_clear(self) -> None:
        self._buffer_paths.clear()
        self._refresh_buffer_list()

    def _buffer_use_selection(self) -> None:
        selected = self._selected_buffer_paths()
        if not selected:
            return
        self.preview_input.set(str(selected[0]))
        self.preview_overlay_inputs.set("; ".join(str(path) for path in selected[1:]))
        self._load_spectrum_preview()

    def _buffer_overlay_selection(self) -> None:
        selected = self._selected_buffer_paths()
        if not selected:
            return
        existing = list(_coerce_path_tokens(self.preview_overlay_inputs.get()))
        for path in selected:
            if path != Path(self.preview_input.get()) and path not in existing:
                existing.append(path)
        self.preview_overlay_inputs.set("; ".join(str(path) for path in existing))
        if self._preview_state_raw is not None:
            self._load_spectrum_preview()

    def _buffer_apply_arithmetic(self) -> None:
        selected = self._selected_buffer_paths()
        if len(selected) != 2:
            messagebox.showerror(
                "FluxForge GUI", "Select exactly two buffered spectra for arithmetic."
            )
            return
        try:
            left = build_gui_spectrum_preview(selected[0]).primary
            right = build_gui_spectrum_preview(selected[1]).primary
            combined = combine_gui_spectrum_series(
                left,
                right,
                operation=self.preview_buffer_operation.get(),
                label=f"{left.label} {self.preview_buffer_operation.get()} {right.label}",
            )
            output_path = Path(
                self.preview_buffer_output.get().strip()
                or self.project_dir / "combined_buffer.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_spectrum_file(output_path, _series_to_spectrum(combined))
        except Exception as exc:
            self._append_log(f"ERROR combining buffers: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        if output_path not in self._buffer_paths:
            self._buffer_paths.append(output_path)
            self._refresh_buffer_list()
        self.preview_input.set(str(output_path))
        self.preview_status.set(
            f"Created combined buffer {output_path.name} using {self.preview_buffer_operation.get()}."
        )
        self._append_log(
            f"Combined buffers into {output_path} with operation {self.preview_buffer_operation.get()}"
        )
        self._load_spectrum_preview()

    def _refresh_preview_data_source_summary(self) -> None:
        try:
            summary = summarize_nuclear_data_source(
                self.preview_database_source.get(),
                custom_path=self.preview_custom_database_path.get().strip() or None,
            )
        except Exception as exc:
            summary = f"Data source unavailable: {exc}"
        self.preview_database_summary.set(summary)

    def _register_custom_preview_data_source(self) -> None:
        path_text = self.preview_custom_database_path.get().strip()
        if not path_text:
            selected = filedialog.askopenfilename(initialdir=str(self.project_dir))
            if not selected:
                return
            path_text = selected
            self.preview_custom_database_path.set(path_text)
        locator: str | Path
        if "://" not in path_text:
            path = Path(path_text)
            if not path.exists():
                messagebox.showerror(
                    "FluxForge GUI", f"Custom data source not found: {path}"
                )
                return
            locator = path
        else:
            locator = path_text
        if locator not in self._custom_data_sources:
            self._custom_data_sources.append(locator)
        self.preview_database_source.set("custom_gamma_file")
        self._refresh_peaks_source_summary()
        self._refresh_activity_source_summary()
        self._refresh_standards_source_summary()
        if self.preview_database_source.get() == "custom_gamma_file":
            self._refresh_preview_data_source_summary()
        summary = "Registered custom gamma source. Supported locators: local JSON/CSV/YAML, sqlite:///... URIs, and python://module:loader."
        if not self.offline_mode:
            summary += " HTTP(S) JSON/CSV/YAML endpoints are also available."
        else:
            summary += (
                " HTTP(S) endpoints remain disabled because offline mode is active."
            )
        self.preview_database_summary.set(summary)

    def _refresh_efficiency_line_choices(self) -> None:
        source = self.preview_efficiency_source.get()
        lines = CALIBRATION_SOURCES.get(source, [])
        values = [
            f"{item['energy']:.2f} keV | I={item['intensity']:.4f}" for item in lines
        ]
        self.preview_efficiency_line_combo.configure(values=values)
        if values and self.preview_efficiency_line.get() not in values:
            self.preview_efficiency_line.set(values[0])

    def _selected_efficiency_reference(self) -> tuple[float, float]:
        source = self.preview_efficiency_source.get()
        selected = self.preview_efficiency_line.get().strip()
        lines = CALIBRATION_SOURCES.get(source, [])
        if not lines:
            raise ValueError(f"No calibration lines are defined for {source}.")
        if not selected:
            item = lines[0]
            return float(item["energy"]), float(item["intensity"])
        energy_text = selected.split("keV", 1)[0].strip()
        reference_energy = float(energy_text)
        for item in lines:
            if abs(float(item["energy"]) - reference_energy) < 1e-6:
                return float(item["energy"]), float(item["intensity"])
        raise ValueError("Selected calibration line is not valid.")

    def _apply_fit_constraint_template(self) -> None:
        template = self.preview_fit_constraint_template.get().strip().lower()
        text = ""
        if template == "shared-doublet":
            text = "1 0\n1 0"
        elif template == "shared-triplet":
            text = "1 0\n1 0\n0 1"
        self.preview_fit_constraint_box.delete("1.0", "end")
        self.preview_fit_constraint_box.insert("1.0", text)

    def _clear_preview_diagnostic_plot(self) -> None:
        self._preview_diagnostic_plot = None
        self._render_current_preview()

    def _log_selected_data_source(
        self, context_label: str, source_id: str, locator: str | None = None
    ) -> None:
        try:
            summary = summarize_nuclear_data_source(source_id, custom_path=locator)
        except Exception:
            summary = source_id
        self._append_log(f"{context_label} data source: {summary}")

    def _refresh_peaks_source_summary(self) -> None:
        try:
            self.peaks_source_summary.set(
                summarize_nuclear_data_source(
                    self.peaks_data_source.get(),
                    custom_path=self.peaks_custom_source.get().strip() or None,
                )
            )
        except Exception as exc:
            self.peaks_source_summary.set(f"Reference source unavailable: {exc}")

    def _refresh_activity_source_summary(self) -> None:
        try:
            self.activity_source_summary.set(
                summarize_nuclear_data_source(
                    self.activity_data_source.get(),
                    custom_path=self.activity_custom_source.get().strip() or None,
                )
            )
        except Exception as exc:
            self.activity_source_summary.set(f"Activity source unavailable: {exc}")

    def _refresh_standards_source_summary(self) -> None:
        try:
            self.standards_source_summary.set(
                summarize_nuclear_data_source(
                    self.standards_data_source.get(),
                    custom_path=self.standards_custom_source.get().strip() or None,
                )
            )
        except Exception as exc:
            self.standards_source_summary.set(f"Standards source unavailable: {exc}")

    def _lookup_activity_reference(self) -> None:
        isotope_text = self.activity_isotope.get().strip()
        if not isotope_text:
            messagebox.showerror(
                "FluxForge GUI",
                "Enter an isotope override before looking up reference data.",
            )
            return
        source_id = self.activity_data_source.get()
        try:
            if source_id in {
                "decay_2012",
                "fluxforge_bundled_gamma",
                "nndc_offline_activation",
                "custom_gamma_file",
            }:
                database = load_gamma_identification_source(
                    source_id,
                    custom_path=self.activity_custom_source.get().strip() or None,
                )
                normalized = isotope_text.replace("-", "")
                decay = database.get(normalized)
                if decay is None:
                    raise ValueError(f"{isotope_text} was not found in {source_id}.")
                strongest = decay.strongest_gamma_lines(1)
                if strongest:
                    self.activity_emission.set(
                        f"{strongest[0].intensity * strongest[0].norm:.6g}"
                    )
                if decay.halflife > 0:
                    self.activity_half_life.set(f"{decay.halflife:.6g}")
            else:
                data = get_nuclear_data(isotope_text)
                if data.get("main_gamma_keV") is not None and data.get("gamma_lines"):
                    self.activity_emission.set(
                        f"{float(data['gamma_lines'][0][1]):.6g}"
                    )
                if data.get("half_life_s"):
                    self.activity_half_life.set(f"{float(data['half_life_s']):.6g}")
        except Exception as exc:
            self._append_log(f"ERROR looking up activity reference: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self._log_selected_data_source(
            "Activity", source_id, self.activity_custom_source.get().strip() or None
        )

    def _browse_standards_source(self) -> None:
        source_id = self.standards_data_source.get()
        target_filter = self.standards_target.get().strip().lower()
        lines: list[str] = []
        if source_id == "irdff_ii_dosimetry":
            library = get_default_library()
            reactions = library.search(
                category=(
                    None
                    if self.standards_reaction_category.get() == "all"
                    else self.standards_reaction_category.get()
                )
            )
            for reaction in reactions:
                if (
                    target_filter
                    and target_filter not in reaction.target.lower()
                    and target_filter not in reaction.product.lower()
                ):
                    continue
                lines.append(
                    f"{reaction.full_name} | threshold={reaction.threshold_MeV:.3f} MeV | half-life={reaction.half_life_days:.3f} d"
                )
        elif source_id == "k0_naa_monitors":
            for element, data in sorted(STANDARD_MONITORS.items()):
                if (
                    target_filter
                    and target_filter not in element.lower()
                    and target_filter not in data["reaction"].lower()
                ):
                    continue
                lines.append(
                    f"{element}: {data['reaction']} | Q0={data['Q0']:.3f} | isotope={data['isotope']}"
                )
        elif source_id == "flux_wire_catalog":
            for isotope in list_flux_wire_isotopes():
                entry = get_flux_wire_catalog_entry(isotope)
                if entry is None:
                    continue
                haystack = (
                    f"{entry.isotope} {entry.parent_element} {entry.reaction}".lower()
                )
                if target_filter and target_filter not in haystack:
                    continue
                lines.append(
                    f"{entry.isotope}: parent={entry.parent_element} | reaction={entry.reaction} | lines={', '.join(f'{value:.1f}' for value in entry.target_lines_keV)}"
                )
        else:
            lines.append(
                summarize_nuclear_data_source(
                    source_id,
                    custom_path=self.standards_custom_source.get().strip() or None,
                )
            )
        if not lines:
            lines.append("No entries matched the current filter.")
        self.standards_notes_box.configure(state="normal")
        self.standards_notes_box.delete("1.0", "end")
        self.standards_notes_box.insert("1.0", "\n".join(lines[:200]))
        self.standards_notes_box.configure(state="disabled")
        self._log_selected_data_source(
            "Standards", source_id, self.standards_custom_source.get().strip() or None
        )

    def _sync_peak_table(self) -> None:
        self.preview_peak_table.delete(*self.preview_peak_table.get_children())
        peaks = self._preview_state.peaks if self._preview_state is not None else ()
        for peak in peaks[:250]:
            self.preview_peak_table.insert(
                "",
                "end",
                values=(
                    f"{peak.energy_keV:.3f}",
                    f"{peak.area:.3f}",
                    "" if peak.channel is None else str(peak.channel),
                    peak.label,
                ),
            )

    def _auto_detect_preview_peaks(self) -> None:
        if self._preview_state is None:
            self._load_spectrum_preview()
            if self._preview_state is None:
                return
        try:
            peaks = auto_detect_gui_peaks(
                self._preview_state.primary,
                finder_method=self.preview_peak_finder_method.get(),
                threshold_sigma=float(self.preview_peak_threshold_sigma.get() or 3.0),
                min_distance=int(self.preview_peak_min_distance.get() or 6),
                identification_method=self.preview_peak_identification_method.get(),
                tolerance_keV=float(self.preview_peak_tolerance_keV.get() or 1.5),
                min_matches=int(self.preview_peak_min_matches.get() or 2),
                identification_source_id=self.preview_database_source.get(),
                custom_source_path=self.preview_custom_database_path.get().strip()
                or None,
                min_intensity=float(self.preview_peak_min_intensity.get() or 0.0),
            )
        except Exception as exc:
            self._append_log(f"ERROR auto-detecting peaks: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        if self._preview_state_raw is not None:
            self._preview_state_raw = GuiSpectrumPreview(
                primary=self._preview_state_raw.primary,
                overlays=self._preview_state_raw.overlays,
                peaks=tuple(
                    GuiSpectrumPeak(
                        energy_keV=peak.energy_keV,
                        area=peak.area,
                        channel=peak.channel,
                        label=peak.label,
                    )
                    for peak in peaks
                ),
            )
        self._selected_peak_energy_keV = None
        self._preview_diagnostic_plot = None
        self._render_current_preview()
        self._sync_peak_table()
        self.preview_status.set(
            f"Detected {len(peaks)} peaks using {self.preview_peak_finder_method.get()} + {self.preview_peak_identification_method.get()} from {self.preview_database_source.get()}."
        )
        self._append_log(f"Auto-detected {len(peaks)} peaks for preview")

    def _count_selected_peak(self) -> None:
        peak = self._selected_peak()
        if peak is None or self._preview_state is None:
            messagebox.showerror(
                "FluxForge GUI", "Select a peak row before running peak counting."
            )
            return
        try:
            self._peak_count_result = count_gui_peak(
                self._preview_state.primary,
                peak,
                self.preview_peak_counting_method.get(),
            )
        except Exception as exc:
            self._append_log(f"ERROR counting peak: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        result = self._peak_count_result
        self._preview_diagnostic_plot = build_peak_count_diagnostic_plot(result)
        self.preview_count_summary.set(
            (
                f"{result.method}: net {result.net_counts:.3f} ± {result.net_uncertainty:.3f}, "
                f"gross {result.gross_counts:.3f} ± {result.gross_uncertainty:.3f}, "
                f"ROI channels {result.roi_bounds[0]}-{result.roi_bounds[1]}."
            )
        )
        self._render_current_preview()

    def _fit_selected_peak_group(self) -> None:
        peak = self._selected_peak()
        if peak is None or self._preview_state is None:
            messagebox.showerror(
                "FluxForge GUI", "Select a peak to run a multiplet or Hypermet fit."
            )
            return
        fit_width = int(self.preview_fit_window.get() or 8)
        neighbor_window_keV = float(self.preview_fit_neighbor_window.get() or 5.0)
        channels = np.asarray(self._preview_state.primary.channels, dtype=float)
        counts = np.asarray(self._preview_state.primary.counts, dtype=float)
        selected_channel = int(
            peak.channel
            if peak.channel is not None
            else round(
                np.interp(
                    peak.energy_keV, self._preview_state.primary.energies_keV, channels
                )
            )
        )
        constraint_text = self.preview_fit_constraint_box.get("1.0", "end").strip()
        try:
            if self.preview_fit_mode.get() == "hypermet":
                hypermet_peak, result = fit_hypermet_peak(
                    channels,
                    counts,
                    peak_channel=selected_channel,
                    fit_width=fit_width,
                    enable_tail=True,
                    enable_step=True,
                )
                roi_x = np.asarray(
                    channels[result.fit_region[0] : result.fit_region[1] + 1],
                    dtype=float,
                )
                roi_counts = np.asarray(
                    counts[result.fit_region[0] : result.fit_region[1] + 1], dtype=float
                )
                model = np.asarray(result.background, dtype=float) + np.asarray(
                    result.peak.evaluate(roi_x), dtype=float
                )
                self._preview_diagnostic_plot = GuiDiagnosticPlot(
                    title="Hypermet residuals",
                    x_label="Channel",
                    y_label="Counts / residual",
                    series=(
                        GuiDiagnosticSeries(
                            label="Counts",
                            x=roi_x,
                            y=roi_counts,
                            style="line",
                            color="#1f77b4",
                        ),
                        GuiDiagnosticSeries(
                            label="Model",
                            x=roi_x,
                            y=model,
                            style="line",
                            color="#d62728",
                        ),
                        GuiDiagnosticSeries(
                            label="Residuals",
                            x=roi_x,
                            y=np.asarray(result.residuals, dtype=float),
                            style="scatter",
                            color="#2ca02c",
                        ),
                    ),
                    reference_y=0.0,
                )
                self.preview_fit_summary.set(
                    (
                        f"Hypermet fit: centroid {hypermet_peak.centroid:.3f} ch, area {hypermet_peak.area:.3f}, "
                        f"FWHM {hypermet_peak.fwhm:.3f}, net uncertainty {result.net_counts_uncertainty:.3f}, "
                        f"χ²ᵣ {result.reduced_chi_squared:.3f}."
                    )
                )
                self._render_current_preview()
                return

            neighbor_channels = sorted(
                {
                    int(item.channel)
                    for item in self._preview_state.peaks
                    if item.channel is not None
                    and abs(item.energy_keV - peak.energy_keV) <= neighbor_window_keV
                }
            )
            if selected_channel not in neighbor_channels:
                neighbor_channels.append(selected_channel)
                neighbor_channels.sort()
            if constraint_text:
                constraint_matrix = parse_gui_constraint_matrix(
                    constraint_text, len(neighbor_channels)
                )
                constrained_results, roi_x, roi_counts, composite = (
                    fit_gui_constrained_multiplet(
                        channels,
                        counts,
                        neighbor_channels,
                        fit_width=fit_width,
                        amplitude_constraint_matrix=constraint_matrix,
                        share_sigma=self.preview_fit_share_sigma.get(),
                        fix_centroids=self.preview_fit_lock_centroids.get(),
                    )
                )
                self._preview_diagnostic_plot = GuiDiagnosticPlot(
                    title="Constrained multiplet diagnostic",
                    x_label="Channel",
                    y_label="Counts / residual",
                    series=(
                        GuiDiagnosticSeries(
                            label="Counts",
                            x=roi_x,
                            y=roi_counts,
                            style="line",
                            color="#1f77b4",
                        ),
                        GuiDiagnosticSeries(
                            label="Composite",
                            x=roi_x,
                            y=composite,
                            style="line",
                            color="#d62728",
                        ),
                        GuiDiagnosticSeries(
                            label="Residuals",
                            x=roi_x,
                            y=roi_counts - composite,
                            style="scatter",
                            color="#2ca02c",
                        ),
                    ),
                    reference_y=0.0,
                )
                summary_parts = [
                    f"ch {item['centroid']:.2f}: area {item['area']:.2f}, FWHM {item['fwhm']:.2f}"
                    for item in constrained_results[:4]
                ]
                self.preview_fit_summary.set(
                    f"Constrained multiplet fit ({len(constrained_results)} peaks, matrix {constraint_matrix.shape[0]}x{constraint_matrix.shape[1]}): "
                    + "; ".join(summary_parts)
                )
                self._render_current_preview()
                return
            if self.preview_fit_lock_centroids.get():
                results = [
                    fit_single_peak(
                        channels,
                        counts,
                        peak_channel=peak_channel,
                        fit_width=fit_width,
                        background_model="linear",
                        fix_centroid=True,
                    )
                    for peak_channel in neighbor_channels
                ]
            else:
                results = fit_multiple_peaks(
                    channels,
                    counts,
                    peak_channels=neighbor_channels,
                    fit_width=fit_width,
                    background_model="linear",
                    share_sigma=self.preview_fit_share_sigma.get(),
                )
        except Exception as exc:
            self._append_log(f"ERROR fitting selected peak group: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        if not results:
            self.preview_fit_summary.set("No multiplet fit results were returned.")
            return
        summary_parts = [
            f"ch {result.peak.centroid:.2f}: net {result.net_counts:.2f} ± {result.net_counts_uncertainty:.2f}, FWHM {result.peak.fwhm:.2f}"
            for result in results[:4]
        ]
        local_lo = max(0, min(result.fit_region[0] for result in results))
        local_hi = min(
            len(channels) - 1, max(result.fit_region[1] for result in results)
        )
        roi_x = np.asarray(channels[local_lo : local_hi + 1], dtype=float)
        roi_counts = np.asarray(counts[local_lo : local_hi + 1], dtype=float)
        composite = np.zeros_like(roi_x, dtype=float)
        residual_stack: list[np.ndarray] = []
        for result in results:
            local_x = np.asarray(
                channels[result.fit_region[0] : result.fit_region[1] + 1], dtype=float
            )
            local_model = np.asarray(result.background, dtype=float) + np.asarray(
                result.peak.evaluate(local_x), dtype=float
            )
            composite += np.interp(roi_x, local_x, local_model, left=0.0, right=0.0)
            residual_stack.append(
                np.interp(
                    roi_x,
                    local_x,
                    np.asarray(result.residuals, dtype=float),
                    left=0.0,
                    right=0.0,
                )
            )
        self._preview_diagnostic_plot = GuiDiagnosticPlot(
            title="Multiplet diagnostic",
            x_label="Channel",
            y_label="Counts / residual",
            series=(
                GuiDiagnosticSeries(
                    label="Counts", x=roi_x, y=roi_counts, style="line", color="#1f77b4"
                ),
                GuiDiagnosticSeries(
                    label="Composite",
                    x=roi_x,
                    y=composite,
                    style="line",
                    color="#d62728",
                ),
                GuiDiagnosticSeries(
                    label="Residuals",
                    x=roi_x,
                    y=(
                        np.sum(residual_stack, axis=0)
                        if residual_stack
                        else np.zeros_like(roi_x)
                    ),
                    style="scatter",
                    color="#2ca02c",
                ),
            ),
            reference_y=0.0,
        )
        self.preview_fit_summary.set(
            f"Multiplet fit ({len(results)} peaks, shared σ={self.preview_fit_share_sigma.get()}, lock centroids={self.preview_fit_lock_centroids.get()}): "
            + "; ".join(summary_parts)
        )
        self._render_current_preview()

    def _toggle_roi_mode(self) -> None:
        if not self.preview_manual_roi_mode.get():
            self._pending_roi_start_keV = None
            self._active_roi_drag = None
            return
        self.preview_status.set(
            "ROI draw mode enabled: click the plot twice to set left and right bounds."
        )

    def _roi_drag_threshold_keV(self) -> float:
        if self._preview_state is None:
            return 2.0
        energies = self._preview_state.primary.energies_keV
        if len(energies) < 2:
            return 2.0
        span = float(energies[-1] - energies[0])
        return max(0.5, span / 250.0)

    def _sync_manual_roi_table(self) -> None:
        self.preview_roi_table.delete(*self.preview_roi_table.get_children())
        for index, region in enumerate(self._manual_regions):
            self.preview_roi_table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    region.label,
                    f"{region.left_keV:.3f}",
                    f"{region.right_keV:.3f}",
                    region.notes,
                ),
            )

    def _on_preview_click(self, event) -> None:
        if not self.preview_manual_roi_mode.get() or event.xdata is None:
            return
        x_value = float(event.xdata)
        selection = self.preview_roi_table.selection()
        if selection:
            region_index = int(selection[0])
            region = self._manual_regions[region_index]
            threshold = self._roi_drag_threshold_keV()
            if abs(x_value - region.left_keV) <= threshold:
                self._active_roi_drag = (region_index, "left")
                self.preview_status.set(f"Dragging left ROI handle for {region.label}.")
                return
            if abs(x_value - region.right_keV) <= threshold:
                self._active_roi_drag = (region_index, "right")
                self.preview_status.set(
                    f"Dragging right ROI handle for {region.label}."
                )
                return
        for region_index, region in enumerate(self._manual_regions):
            if region.left_keV <= x_value <= region.right_keV:
                self.preview_roi_table.selection_set(str(region_index))
                self.preview_roi_table.focus(str(region_index))
                self._on_roi_selected()
                self.preview_status.set(f"Selected ROI {region.label} from plot.")
                self._render_current_preview()
                return
        if self._pending_roi_start_keV is None:
            self._pending_roi_start_keV = x_value
            self.preview_status.set(
                f"ROI start set at {x_value:.3f} keV. Click again to finish the ROI."
            )
            self.preview_roi_left.set(f"{x_value:.3f}")
            self.preview_roi_right.set("")
            return
        self.preview_roi_left.set(f"{min(self._pending_roi_start_keV, x_value):.3f}")
        self.preview_roi_right.set(f"{max(self._pending_roi_start_keV, x_value):.3f}")
        self._pending_roi_start_keV = None
        self._add_or_update_manual_roi()

    def _on_preview_motion(self, event) -> None:
        if self._active_roi_drag is None or event.xdata is None:
            return
        region_index, edge = self._active_roi_drag
        if not (0 <= region_index < len(self._manual_regions)):
            return
        region = self._manual_regions[region_index]
        x_value = float(event.xdata)
        if edge == "left":
            updated = GuiManualRegion(
                label=region.label,
                left_keV=min(x_value, region.right_keV),
                right_keV=max(x_value, region.right_keV),
                notes=region.notes,
            )
        else:
            updated = GuiManualRegion(
                label=region.label,
                left_keV=min(region.left_keV, x_value),
                right_keV=max(region.left_keV, x_value),
                notes=region.notes,
            )
        self._manual_regions[region_index] = updated
        self.preview_roi_left.set(f"{updated.left_keV:.3f}")
        self.preview_roi_right.set(f"{updated.right_keV:.3f}")
        self._sync_manual_roi_table()
        self.preview_roi_table.selection_set(str(region_index))
        self._render_current_preview()

    def _on_preview_release(self, _event) -> None:
        if self._active_roi_drag is not None:
            self.preview_status.set("ROI handle updated from on-canvas drag.")
        self._active_roi_drag = None

    def _add_or_update_manual_roi(self) -> None:
        label = (
            self.preview_roi_label.get().strip()
            or f"ROI {len(self._manual_regions) + 1}"
        )
        left = self._optional_float(self.preview_roi_left.get())
        right = self._optional_float(self.preview_roi_right.get())
        if left is None or right is None:
            messagebox.showerror(
                "FluxForge GUI", "Manual ROI requires both left and right bounds."
            )
            return
        region = GuiManualRegion(
            label=label, left_keV=min(left, right), right_keV=max(left, right)
        )
        selection = self.preview_roi_table.selection()
        if selection:
            self._manual_regions[int(selection[0])] = region
        else:
            self._manual_regions.append(region)
        self.preview_roi_label.set(f"ROI {len(self._manual_regions) + 1}")
        self.preview_roi_left.set("")
        self.preview_roi_right.set("")
        self._sync_manual_roi_table()
        self._render_current_preview()

    def _delete_selected_manual_roi(self) -> None:
        selection = self.preview_roi_table.selection()
        if not selection:
            return
        del self._manual_regions[int(selection[0])]
        self._sync_manual_roi_table()
        self._render_current_preview()

    def _clear_manual_rois(self) -> None:
        self._manual_regions.clear()
        self._sync_manual_roi_table()
        self._render_current_preview()

    def _on_roi_selected(self, _event=None) -> None:
        selection = self.preview_roi_table.selection()
        if not selection:
            return
        region = self._manual_regions[int(selection[0])]
        self.preview_roi_label.set(region.label)
        self.preview_roi_left.set(f"{region.left_keV:.3f}")
        self.preview_roi_right.set(f"{region.right_keV:.3f}")
        self._render_current_preview()

    def _load_manual_rois_file(self) -> None:
        path_text = self.preview_roi_file.get().strip()
        if not path_text:
            return
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._manual_regions = [
            GuiManualRegion(
                label=str(item.get("label") or item.get("name") or f"ROI {index + 1}"),
                left_keV=float(item["left_keV"]),
                right_keV=float(item["right_keV"]),
                notes=str(item.get("notes") or ""),
            )
            for index, item in enumerate(payload)
        ]
        self._sync_manual_roi_table()
        self._render_current_preview()

    def _save_manual_rois_file(self) -> None:
        path_text = self.preview_roi_file.get().strip()
        if not path_text:
            return
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "label": region.label,
                "left_keV": region.left_keV,
                "right_keV": region.right_keV,
                "notes": region.notes,
            }
            for region in self._manual_regions
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.preview_status.set(f"Saved {len(payload)} manual ROI(s) to {path}")

    def _load_calibration_from_preview(self) -> None:
        source = (
            self._preview_state_raw.primary
            if self._preview_state_raw is not None
            else None
        )
        coeffs = (
            source.calibration_coeffs
            if source and source.calibration_coeffs
            else (0.0, 1.0, 0.0, 0.0)
        )
        padded = tuple(list(coeffs[:4]) + [0.0] * (4 - len(coeffs[:4])))
        self.preview_calibration_c0.set(str(padded[0]))
        self.preview_calibration_c1.set(str(padded[1]))
        self.preview_calibration_c2.set(str(padded[2]))
        self.preview_calibration_c3.set(str(padded[3]))

    def _apply_calibration_to_preview(self) -> None:
        if self._preview_state_raw is None:
            return
        self._render_current_preview()
        self.preview_status.set(
            "Applied calibration editor coefficients to the loaded preview."
        )

    def _reset_calibration_editor(self) -> None:
        self.preview_calibration_c0.set("0.0")
        self.preview_calibration_c1.set("1.0")
        self.preview_calibration_c2.set("0.0")
        self.preview_calibration_c3.set("0.0")
        self._render_current_preview()

    def _sync_calibration_table(self) -> None:
        self.preview_calibration_table.delete(
            *self.preview_calibration_table.get_children()
        )
        residuals = (
            self._calibration_fit.residuals_keV
            if self._calibration_fit is not None
            else np.full(len(self._calibration_points), np.nan)
        )
        for index, point in enumerate(self._calibration_points):
            residual = residuals[index] if index < len(residuals) else np.nan
            self.preview_calibration_table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    f"{point.channel:.3f}",
                    f"{point.observed_energy_keV:.3f}",
                    f"{point.reference_energy_keV:.3f}",
                    "" if np.isnan(residual) else f"{residual:.4f}",
                    point.label,
                ),
            )

    def _pick_calibration_point_from_selected_peak(self) -> None:
        peak = self._selected_peak()
        if peak is None:
            messagebox.showerror(
                "FluxForge GUI", "Select a peak row to seed a calibration point."
            )
            return
        self.preview_calibration_point_channel.set(
            "" if peak.channel is None else f"{peak.channel:.3f}"
        )
        self.preview_calibration_point_observed.set(f"{peak.energy_keV:.3f}")
        if not self.preview_calibration_point_label.get().strip():
            self.preview_calibration_point_label.set(
                peak.label or f"Peak {peak.energy_keV:.1f} keV"
            )

    def _add_calibration_point(self) -> None:
        try:
            point = GuiCalibrationPoint(
                channel=float(self.preview_calibration_point_channel.get()),
                observed_energy_keV=float(
                    self.preview_calibration_point_observed.get()
                ),
                reference_energy_keV=float(
                    self.preview_calibration_point_reference.get()
                ),
                label=self.preview_calibration_point_label.get().strip(),
            )
        except ValueError as exc:
            messagebox.showerror(
                "FluxForge GUI", f"Calibration point is incomplete: {exc}"
            )
            return
        self._calibration_points.append(point)
        self.preview_calibration_point_channel.set("")
        self.preview_calibration_point_observed.set("")
        self.preview_calibration_point_reference.set("")
        self.preview_calibration_point_label.set("")
        self._calibration_fit = None
        self._sync_calibration_table()
        self.preview_calibration_summary.set(
            f"Added {len(self._calibration_points)} calibration point(s)."
        )

    def _fit_preview_calibration_points(self) -> None:
        try:
            self._calibration_fit = fit_gui_energy_calibration(
                self._calibration_points,
                order=int(self.preview_calibration_order.get() or 1),
            )
        except Exception as exc:
            self._append_log(f"ERROR fitting calibration points: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        coeffs = list(self._calibration_fit.coefficients[:4]) + [0.0] * max(
            0, 4 - len(self._calibration_fit.coefficients[:4])
        )
        self.preview_calibration_c0.set(f"{coeffs[0]:.12g}")
        self.preview_calibration_c1.set(f"{coeffs[1]:.12g}")
        self.preview_calibration_c2.set(f"{coeffs[2]:.12g}")
        self.preview_calibration_c3.set(f"{coeffs[3]:.12g}")
        self._sync_calibration_table()
        self._preview_diagnostic_plot = build_calibration_residual_plot(
            self._calibration_points, self._calibration_fit
        )
        rms = (
            float(np.sqrt(np.mean(self._calibration_fit.residuals_keV**2)))
            if len(self._calibration_fit.residuals_keV)
            else 0.0
        )
        self.preview_calibration_summary.set(
            f"Calibration fit complete: order {self.preview_calibration_order.get()}, R²={self._calibration_fit.r_squared:.6f}, residual RMS={rms:.4f} keV."
        )
        self._render_current_preview()

    def _clear_calibration_points(self) -> None:
        self._calibration_points.clear()
        self._calibration_fit = None
        self._preview_diagnostic_plot = None
        self._sync_calibration_table()
        self.preview_calibration_summary.set("Cleared manual calibration points.")

    def _sync_efficiency_table(self) -> None:
        self.preview_efficiency_table.delete(
            *self.preview_efficiency_table.get_children()
        )
        for index, point in enumerate(self._efficiency_points):
            self.preview_efficiency_table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    point.source_name,
                    f"{point.reference_energy_keV:.2f}",
                    f"{point.net_counts:.2f}",
                    f"{point.efficiency:.5f}",
                    f"{point.efficiency_uncertainty:.5f}",
                ),
            )

    def _add_efficiency_point_from_selected_peak(self) -> None:
        peak = self._selected_peak()
        if peak is None:
            messagebox.showerror(
                "FluxForge GUI",
                "Select a peak before adding an efficiency calibration point.",
            )
            return
        try:
            reference_energy, emission_probability = (
                self._selected_efficiency_reference()
            )
            net_counts = (
                float(self._peak_count_result.net_counts)
                if self._peak_count_result is not None
                else float(max(peak.area, 0.0))
            )
            point = EfficiencyPoint(
                energy_keV=reference_energy,
                net_counts=net_counts,
                live_time_s=float(self.preview_efficiency_live_time_s.get()),
                activity_bq=float(self.preview_efficiency_activity_bq.get()),
                emission_probability=emission_probability,
                geometry_factor=float(
                    self.preview_efficiency_geometry_factor.get() or 1.0
                ),
                count_uncertainty=np.sqrt(max(net_counts, 1.0)),
            )
            efficiency, uncertainty = point.efficiency()
            gui_point = GuiEfficiencyCalibrationPoint(
                source_name=self.preview_efficiency_source.get(),
                reference_energy_keV=reference_energy,
                measured_energy_keV=peak.energy_keV,
                net_counts=net_counts,
                live_time_s=point.live_time_s,
                activity_bq=point.activity_bq,
                emission_probability=emission_probability,
                geometry_factor=point.geometry_factor,
                efficiency=float(efficiency),
                efficiency_uncertainty=float(uncertainty),
                label=peak.label,
            )
        except Exception as exc:
            self._append_log(f"ERROR adding efficiency point: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self._efficiency_points.append(gui_point)
        self._sync_efficiency_table()
        self.preview_efficiency_summary.set(
            f"Added efficiency point at {gui_point.reference_energy_keV:.2f} keV using {gui_point.net_counts:.2f} net counts."
        )

    def _fit_efficiency_curve_from_points(self) -> None:
        if len(self._efficiency_points) < 2:
            messagebox.showerror(
                "FluxForge GUI",
                "Add at least two efficiency points before fitting a curve.",
            )
            return
        try:
            fit = fit_efficiency_curve(
                [point.to_efficiency_point() for point in self._efficiency_points],
                degree=int(self.preview_efficiency_fit_degree.get() or 2),
            )
        except Exception as exc:
            self._append_log(f"ERROR fitting efficiency curve: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self._efficiency_curve = fit.curve
        self._preview_diagnostic_plot = build_efficiency_fit_diagnostic_plot(
            self._efficiency_points, fit.curve
        )
        rms = (
            float(np.sqrt(np.mean(np.asarray(fit.residuals, dtype=float) ** 2)))
            if len(fit.residuals)
            else 0.0
        )
        self.preview_efficiency_summary.set(
            f"Efficiency fit complete: degree {self.preview_efficiency_fit_degree.get()}, log-space residual RMS={rms:.5f}."
        )
        self._render_current_preview()

    def _save_efficiency_curve(self) -> None:
        if self._efficiency_curve is None:
            self._fit_efficiency_curve_from_points()
            if self._efficiency_curve is None:
                return
        output = Path(self.preview_efficiency_output.get().strip())
        output.parent.mkdir(parents=True, exist_ok=True)
        self._efficiency_curve.save(output)
        self.preview_efficiency_summary.set(f"Saved efficiency curve to {output}.")

    def _clear_efficiency_points(self) -> None:
        self._efficiency_points.clear()
        self._efficiency_curve = None
        self._sync_efficiency_table()
        self.preview_efficiency_summary.set("Cleared efficiency calibration points.")

    def _sync_stacked_foil_table(self) -> None:
        self.stacked_foil_table.delete(*self.stacked_foil_table.get_children())
        for index, foil in enumerate(self._stacked_foils):
            self.stacked_foil_table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    foil["material"],
                    f"{float(foil['thickness_um']):.3f}",
                    foil["reaction"],
                    foil["target_isotope"],
                    foil["product_isotope"],
                ),
            )

    def _physics_add_foil(self) -> None:
        try:
            thickness_um = float(self.stacked_foil_thickness.get())
        except ValueError as exc:
            messagebox.showerror("FluxForge GUI", f"Invalid foil thickness: {exc}")
            return
        foil = {
            "material": self.stacked_foil_material.get(),
            "thickness_um": thickness_um,
            "reaction": self.stacked_foil_reaction.get().strip(),
            "target_isotope": self.stacked_foil_target.get().strip(),
            "product_isotope": self.stacked_foil_product.get().strip(),
        }
        self._stacked_foils.append(foil)
        self._sync_stacked_foil_table()
        self.stacked_summary.set(
            f"Added {len(self._stacked_foils)} foil(s) to the stacked-target workspace."
        )

    def _physics_clear_foils(self) -> None:
        self._stacked_foils.clear()
        self._sync_stacked_foil_table()
        self.stacked_summary.set("Cleared stacked-target foil list.")

    def _selected_peak(self) -> GuiSpectrumPeak | None:
        selection = self.preview_peak_table.selection()
        if not selection or self._preview_state is None:
            return None
        index = self.preview_peak_table.index(selection[0])
        if index >= len(self._preview_state.peaks):
            return None
        return self._preview_state.peaks[index]

    def _update_peak_inspector(self, peak: GuiSpectrumPeak | None) -> None:
        if peak is None or self._preview_state is None:
            self.preview_inspector.set(
                "Load peaks, then select a peak row to inspect fit/deconvolution context."
            )
            self._selected_peak_energy_keV = None
            return
        neighbors = [
            item
            for item in self._preview_state.peaks
            if item is not peak and abs(item.energy_keV - peak.energy_keV) <= 5.0
        ]
        left = peak.energy_keV - 5.0
        right = peak.energy_keV + 5.0
        local_mask = (self._preview_state.primary.energies_keV >= left) & (
            self._preview_state.primary.energies_keV <= right
        )
        local_max = (
            float(np.nanmax(self._preview_state.primary.counts[local_mask]))
            if np.any(local_mask)
            else 0.0
        )
        neighbor_text = (
            ", ".join(f"{item.energy_keV:.2f} keV" for item in neighbors[:5]) or "none"
        )
        self.preview_inspector.set(
            (
                f"Peak {peak.energy_keV:.3f} keV | area {peak.area:.3f} | channel {peak.channel}. "
                f"Local window: {left:.2f}-{right:.2f} keV, max counts {local_max:.2f}. "
                f"Possible multiplet/deconvolution neighbors within ±5 keV: {neighbor_text}."
            )
        )
        self._selected_peak_energy_keV = peak.energy_keV

    def _on_peak_selected(self, _event=None) -> None:
        peak = self._selected_peak()
        self._update_peak_inspector(peak)
        self._render_current_preview()

    def _zoom_selected_peak(self) -> None:
        peak = self._selected_peak()
        if peak is None:
            return
        self.preview_x_min.set(f"{peak.energy_keV - 8.0:.3f}")
        self.preview_x_max.set(f"{peak.energy_keV + 8.0:.3f}")
        self._render_current_preview()

    def _reset_preview_zoom(self) -> None:
        self.preview_x_min.set("")
        self.preview_x_max.set("")
        self._render_current_preview()

    def _load_spectrum_preview(self) -> None:
        if self.preview_canvas is None or self.preview_figure is None:
            messagebox.showerror(
                "FluxForge GUI", "matplotlib is required for the spectrum viewer."
            )
            return
        spectrum_path = self.preview_input.get().strip()
        if not spectrum_path:
            messagebox.showerror(
                "FluxForge GUI", "Choose a primary spectrum file or artifact first."
            )
            return

        overlay_paths = _coerce_path_tokens(self.preview_overlay_inputs.get())
        peaks_path_text = self.preview_peaks.get().strip()
        peaks_path = peaks_path_text if peaks_path_text else None

        try:
            preview = build_gui_spectrum_preview(
                spectrum_path,
                overlay_paths=overlay_paths,
                peaks_path=peaks_path,
            )
        except Exception as exc:
            self._append_log(f"ERROR loading spectrum preview: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        self._preview_state_raw = preview
        self._selected_peak_energy_keV = None
        self._preview_diagnostic_plot = None
        self._load_calibration_from_preview()
        self._render_current_preview()
        self._sync_peak_table()
        overlay_count = len(preview.overlays)
        self.preview_status.set(
            (
                f"Loaded {preview.primary.label} with {overlay_count} overlay(s) and "
                f"{len(preview.peaks)} peak marker(s)."
            )
        )
        self._update_peak_inspector(None)
        self._append_log(
            (
                f"Loaded spectrum preview: primary={spectrum_path}, overlays={overlay_count}, "
                f"peaks={len(preview.peaks)}"
            )
        )
        self._buffer_add_current()

    def _save_spectrum_preview_png(self) -> None:
        if self._preview_state is None:
            self._load_spectrum_preview()
            if self._preview_state is None:
                return

        output_path = self.preview_png_output.get().strip()
        if not output_path:
            messagebox.showerror("FluxForge GUI", "Choose an output PNG path first.")
            return

        try:
            saved = save_gui_spectrum_preview_image(
                self._preview_state,
                output_path,
                manual_regions=tuple(self._manual_regions),
                selected_region_label=self._selected_region_label(),
                diagnostic_plot=self._preview_diagnostic_plot,
                y_log=self.preview_y_scale.get() == "log",
                x_log=self.preview_x_scale.get() == "log",
                x_min_keV=self._optional_float(self.preview_x_min.get()),
                x_max_keV=self._optional_float(self.preview_x_max.get()),
                group_peak_colors=self.preview_peak_color_mode.get() == "isotope",
            )
        except Exception as exc:
            self._append_log(f"ERROR saving spectrum preview: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return

        self._append_log(f"Saved GUI spectrum preview: {saved}")
        self.preview_status.set(f"Saved preview image to {saved}")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _sync_standards_notes(self) -> None:
        preset = next(
            (
                item
                for item in get_standards_gui_presets().values()
                if item.label == self.standards_preset.get()
            ),
            None,
        )
        if preset is None:
            notes_text = ""
        else:
            values = build_standards_preset_values(
                preset.key, self.standards_profile.get()
            )
            notes_text = values["notes"]
            self.standards_reaction_category.set(str(values["reaction_category"]))
        self.standards_notes_box.configure(state="normal")
        self.standards_notes_box.delete("1.0", "end")
        self.standards_notes_box.insert("1.0", notes_text)
        self.standards_notes_box.configure(state="disabled")

    def _apply_standards_preset(self) -> None:
        preset = next(
            (
                item
                for item in get_standards_gui_presets().values()
                if item.label == self.standards_preset.get()
            ),
            None,
        )
        if preset is None:
            return
        values = build_standards_preset_values(preset.key, self.standards_profile.get())
        self.ingest_profile.set(str(values["profile"]))
        self.peaks_profile.set(str(values["profile"]))
        self.peaks_sensitivity.set(str(values["peaks_sensitivity"]))
        self.preview_peak_counting_method.set(str(values["peak_counting_method"]))
        self.peaks_background_subtracted.set(bool(values["background_subtracted"]))
        self.standards_reaction_category.set(str(values["reaction_category"]))
        if preset.key in {"astm_inl", "us_astm", "iaea_irdff_gma"}:
            self.standards_data_source.set("irdff_ii_dosimetry")
        elif preset.key == "k0_naa":
            self.standards_data_source.set("k0_naa_monitors")
        elif preset.key == "comparator_naa":
            self.standards_data_source.set("flux_wire_catalog")
        self._sync_standards_notes()
        self._refresh_standards_source_summary()
        self._append_log(f"Applied standards preset: {preset.label}")

    def _load_k0_preview(self) -> None:
        path = self._optional_path(self.k0_report_output.get())
        if path is None or not path.exists():
            path = self._optional_path(self.k0_analysis_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI",
                "Choose an existing k0 analysis or report artifact first.",
            )
            return
        if path.suffix.lower() == ".json":
            payload = (
                read_report_bundle(path)
                if path == Path(self.k0_report_output.get())
                else read_k0_analysis_bundle(path)
            )
        else:
            payload = read_k0_analysis_bundle(path)
        preview = build_gui_k0_preview(payload, path)
        self.k0_preview.configure(state="normal")
        self.k0_preview.delete("1.0", "end")
        self.k0_preview.insert("1.0", preview)
        self.k0_preview.configure(state="disabled")
        self.k0_status.set(
            "Loaded k0 preview from the latest analysis/report artifact."
        )

    def _after_k0_import_run(self) -> None:
        self.k0_status.set(
            "Imported Kayzero content into a governed FluxForge k0 library. The imported library path is now available for k0 analysis."
        )

    def _load_response_preview(self) -> None:
        path = self._optional_path(self.response_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI", "Choose an existing response bundle artifact first."
            )
            return
        payload = read_response_bundle(path)
        reactions = payload.get("reactions", [])
        boundaries = payload.get("boundaries_eV", [])
        group_count = max(len(boundaries) - 1, 0)
        self.response_status.set(
            f"Loaded response bundle with {len(reactions)} reactions and {group_count} energy groups from {path.name}."
        )
        self._append_log(self.response_status.get())

    def _load_astm_e3376_preview(self) -> None:
        path = self._optional_path(self.astm_e3376_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI", "Choose an existing ASTM E3376 output artifact first."
            )
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        preview = build_gui_astm_e3376_preview(payload, path)
        self.astm_e3376_preview.configure(state="normal")
        self.astm_e3376_preview.delete("1.0", "end")
        self.astm_e3376_preview.insert("1.0", preview)
        self.astm_e3376_preview.configure(state="disabled")
        self.astm_e3376_status.set(
            "Loaded ASTM E3376 preview from the latest workflow artifact."
        )

    def _after_rafm_validate_run(self) -> None:
        self.rafm_status.set(
            "Completed RAFM raw-spectrum validation. Review the run log and results root for generated artifacts."
        )

    def _after_rafm_qg_benchmark_run(self) -> None:
        self.rafm_status.set(
            "Completed RAFM QuantumGold benchmark processing. Review the run log and QG results root for generated artifacts."
        )

    def _after_rafm_compare_run(self) -> None:
        self.rafm_status.set(
            "Completed RAFM branch comparison. Review the run log and comparison root for generated artifacts."
        )

    def _load_astm_e2005_preview(self) -> None:
        if not hasattr(self, "astm_e2005_output"):
            return
        path = self._optional_path(self.astm_e2005_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI", "Choose an existing ASTM E2005 output artifact first."
            )
            return
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        preview = build_gui_astm_e2005_preview(payload, path)
        self.astm_e2005_preview.configure(state="normal")
        self.astm_e2005_preview.delete("1.0", "end")
        self.astm_e2005_preview.insert("1.0", preview)
        self.astm_e2005_preview.configure(state="disabled")
        self.astm_e2005_status.set(
            "Loaded ASTM E2005 preview from the latest workflow artifact"
        )

    def _load_astm_e261_preview(self) -> None:
        path = self._optional_path(self.astm_e261_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI", "Choose an existing ASTM E261 output artifact first."
            )
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        preview = build_gui_astm_e261_preview(payload, path)
        self.astm_e261_preview.configure(state="normal")
        self.astm_e261_preview.delete("1.0", "end")
        self.astm_e261_preview.insert("1.0", preview)
        self.astm_e261_preview.configure(state="disabled")
        self.astm_e261_status.set(
            "Loaded ASTM E261 preview from the latest workflow artifact."
        )

    def _load_astm_e262_preview(self) -> None:
        path = self._optional_path(self.astm_e262_output.get())
        if path is None or not path.exists():
            messagebox.showerror(
                "FluxForge GUI", "Choose an existing ASTM E262 output artifact first."
            )
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        preview = build_gui_astm_e262_preview(payload, path)
        self.astm_e262_preview.configure(state="normal")
        self.astm_e262_preview.delete("1.0", "end")
        self.astm_e262_preview.insert("1.0", preview)
        self.astm_e262_preview.configure(state="disabled")
        self.astm_e262_status.set(
            "Loaded ASTM E262 preview from the latest workflow artifact."
        )

    def _auto_fill_report_from_validation_dir(self) -> None:
        raw = self.report_validation_results_root.get().strip()
        if not raw:
            messagebox.showerror(
                "FluxForge GUI", "Choose a validation results directory first."
            )
            return
        try:
            defaults = discover_gui_validation_report_inputs(raw)
        except Exception as exc:
            self._append_log(f"ERROR discovering validation report inputs: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        self.report_validation_results_root.set(defaults["validation_results_root"])
        self.report_unfold.set(defaults["unfold_file"])
        self.report_validation.set(defaults["validation_file"])
        self.report_output.set(defaults["output"])
        self.report_figure_dir.set(defaults["figure_dir"])
        self.report_status.set(
            f"Auto-filled report inputs from {defaults['validation_results_root']}"
        )
        self._append_log(self.report_status.get())

    def _after_report_run(self) -> None:
        self._load_report_preview()
        if self.report_export_figures.get():
            self._export_report_figures()

    def _after_master_plots_run(self) -> None:
        self.plots_status.set(
            f"Generated master plot suite in {self.plots_output_dir.get().strip() or self.project_dir / 'plots'}."
        )
        self._append_log(self.plots_status.get())

    def _on_close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch_gui(project_dir: str | Path | None = None) -> None:
    """Launch the FluxForge Tk desktop GUI."""
    root = tk.Tk()
    FluxForgeGui(root, Path(project_dir) if project_dir else None)
    root.mainloop()


def main() -> None:
    """Module entrypoint for ``python -m fluxforge_gui.app``."""
    launch_gui()


if __name__ == "__main__":
    main()
