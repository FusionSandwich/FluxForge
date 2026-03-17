"""Desktop GUI shell for FluxForge.

SYNC_MARKER_TEMP

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
from typing import Any, Iterable, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from scipy import optimize

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    Figure = None

from fluxforge.analysis.flux_wire_analysis import (
    _covell_style_local_continuum_counts,
    _gilmore_moving_minimum_counts,
    _standards_tiered_counts,
)
from fluxforge.analysis.detector_calibration import EfficiencyPoint, fit_efficiency_curve
from fluxforge.analysis.peak_finders import PEAK_FINDER_METHODS, find_peaks_multi_method, get_peak_finder
from fluxforge.analysis.peakfit import fit_hypermet_peak, fit_multiple_peaks, fit_single_peak
from fluxforge.cli import app as cli_app
from fluxforge.data.efficiency import CALIBRATION_SOURCES, EfficiencyCurve
from fluxforge.data.flux_wire_catalog import get_flux_wire_catalog_entry, list_flux_wire_isotopes
from fluxforge.data.irdff_access import get_default_library
from fluxforge.data.nndc import get_nuclear_data
from fluxforge.data.gamma_database import get_database
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source, summarize_nuclear_data_source
from fluxforge.io.artifacts import (
    read_k0_analysis_bundle,
    read_line_activities,
    read_peak_report,
    read_report_bundle,
    read_reaction_rates,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_peak_report,
    write_report_bundle,
    write_spectrum_file,
)
from fluxforge.io.spe import GammaSpectrum
from fluxforge.physics.decay_chain import DecayChain
from fluxforge.physics.stacked_target import StackedTarget
from fluxforge.physics.stopping_power import Projectile, STANDARD_MATERIALS
from fluxforge.triga.cd_ratio import STANDARD_MONITORS
from fluxforge_gui.constants import (
    ALLOWED_REACTION_CATEGORIES,
    GUI_BUFFER_OPERATIONS,
    GUI_PEAK_COUNTING_METHODS,
    GUI_PEAK_IDENTIFICATION_METHODS,
    GUI_RAFM_COUNTING_METHODS,
    GUI_UNFOLD_METHODS,
    GUI_UNFOLD_MLEM_CONVERGENCE_MODES,
)
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
    get_gui_data_source_choices,
    get_gui_profile_choices,
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



class CommandsMixin:
    def _physics_run_stacked_target(self) -> None:
        if not self._stacked_foils:
            messagebox.showerror("FluxForge GUI", "Add at least one foil before solving the stacked target.")
            return
        try:
            stack = StackedTarget(
                projectile=Projectile[self.stacked_projectile.get()],
                beam_energy_MeV=float(self.stacked_beam_energy.get()),
            )
            for foil in self._stacked_foils:
                stack.add_foil(
                    foil["material"],
                    thickness_um=float(foil["thickness_um"]),
                    reaction=str(foil["reaction"] or None) if foil["reaction"] else None,
                    target_isotope=str(foil["target_isotope"] or None) if foil["target_isotope"] else None,
                    product_isotope=str(foil["product_isotope"] or None) if foil["product_isotope"] else None,
                )
            energies = stack.calculate_energies()
        except Exception as exc:
            self._append_log(f"ERROR solving stacked target: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        if not energies:
            self.stacked_summary.set("Stack calculation produced no energy points.")
            return
        summary_lines = [
            f"Foil {item.foil_index + 1}: Ein {item.energy_in_MeV:.3f} MeV, Emean {item.energy_mean_MeV:.3f} MeV, Eout {item.energy_out_MeV:.3f} MeV"
            for item in energies[:6]
        ]
        self.stacked_summary.set(
            f"Solved {len(energies)} foil position(s) for {self.stacked_projectile.get()} at {self.stacked_beam_energy.get()} MeV. "
            + " | ".join(summary_lines)
        )

    def _physics_run_decay_chain(self) -> None:
        parent = self.decay_parent.get().strip()
        daughter = self.decay_daughter.get().strip()
        try:
            parent_half_life = float(self.decay_parent_half_life.get())
            daughter_half_life = float(self.decay_daughter_half_life.get())
            initial_activity = float(self.decay_initial_activity.get())
            end_time = float(self.decay_end_time.get())
            production_rate = float(self.decay_production_rate.get() or 0.0)
            chain = DecayChain(
                parent,
                nuclide_data={
                    parent: {"half_life_s": parent_half_life, "decay_products": {daughter: 1.0}},
                    daughter: {"half_life_s": daughter_half_life, "decay_products": {}},
                },
            )
            result = chain.decay(
                initial_activity={parent: initial_activity},
                t_end=end_time,
                n_points=100,
                production_rates={parent: production_rate} if production_rate > 0.0 else None,
                units=self.decay_time_units.get(),
            )
        except Exception as exc:
            self._append_log(f"ERROR solving decay chain: {exc}")
            messagebox.showerror("FluxForge GUI", str(exc))
            return
        parent_final = float(result.activities[parent][-1]) if parent in result.activities else 0.0
        daughter_final = float(result.activities[daughter][-1]) if daughter in result.activities else 0.0
        peak_daughter = float(np.max(result.activities[daughter])) if daughter in result.activities else 0.0
        self.decay_summary.set(
            (
                f"Decay chain solved for {parent} → {daughter}. Final parent activity {parent_final:.3f} Bq, "
                f"final daughter activity {daughter_final:.3f} Bq, peak daughter activity {peak_daughter:.3f} Bq."
            )
        )

    def _run_ingest(self) -> None:
        background_file = self.ingest_background_file.get().strip()
        background_scale_factor = self.ingest_background_scale_factor.get().strip()
        profile = self.ingest_profile.get().strip() or None
        energy_calibration = self.ingest_energy_calibration.get().strip() or None
        efficiency_coefficients = self.ingest_efficiency_coefficients.get().strip() or None
        args = Namespace(
            input=Path(self.ingest_input.get()),
            output=Path(self.ingest_output.get()),
            profile=profile,
            background_file=Path(background_file) if background_file else None,
            background_scale_mode=self.ingest_background_scale_mode.get(),
            background_scale_factor=float(background_scale_factor) if background_scale_factor else None,
            energy_calibration=energy_calibration,
            efficiency_coefficients=efficiency_coefficients,
            validate=self.ingest_validate.get(),
        )
        tokens = ["ingest", "--input", str(args.input), "--output", str(args.output)]
        if args.profile:
            tokens.extend(["--profile", args.profile])
        if args.background_file:
            tokens.extend(["--background-file", str(args.background_file)])
        if args.background_scale_mode != "live":
            tokens.extend(["--background-scale-mode", args.background_scale_mode])
        if args.background_scale_factor is not None:
            tokens.extend(["--background-scale-factor", str(args.background_scale_factor)])
        if args.energy_calibration:
            tokens.extend(["--energy-calibration", args.energy_calibration])
        if args.efficiency_coefficients:
            tokens.extend(["--efficiency-coefficients", args.efficiency_coefficients])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_ingest, args, tokens)

    def _run_ingest_batch(self) -> None:
        background_file = self.ingest_batch_background_file.get().strip()
        background_scale_factor = self.ingest_batch_background_scale_factor.get().strip()
        profile = self.ingest_batch_profile.get().strip() or None
        energy_calibration = self.ingest_batch_energy_calibration.get().strip() or None
        efficiency_coefficients = self.ingest_batch_efficiency_coefficients.get().strip() or None
        background_adjusted_dir = self.ingest_batch_background_adjusted_dir.get().strip()
        final_corrected_dir = self.ingest_batch_final_corrected_dir.get().strip()
        args = Namespace(
            input_dir=Path(self.ingest_batch_input_dir.get()),
            output_dir=Path(self.ingest_batch_output_dir.get()),
            profile=profile,
            background_file=Path(background_file) if background_file else None,
            background_scale_mode=self.ingest_batch_background_scale_mode.get(),
            background_scale_factor=float(background_scale_factor) if background_scale_factor else None,
            energy_calibration=energy_calibration,
            efficiency_coefficients=efficiency_coefficients,
            background_adjusted_dir=Path(background_adjusted_dir) if background_adjusted_dir else None,
            final_corrected_dir=Path(final_corrected_dir) if final_corrected_dir else None,
            validate=self.ingest_batch_validate.get(),
        )
        tokens = [
            "ingest-batch",
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.profile:
            tokens.extend(["--profile", args.profile])
        if args.background_file:
            tokens.extend(["--background-file", str(args.background_file)])
        if args.background_scale_mode != "live":
            tokens.extend(["--background-scale-mode", args.background_scale_mode])
        if args.background_scale_factor is not None:
            tokens.extend(["--background-scale-factor", str(args.background_scale_factor)])
        if args.energy_calibration:
            tokens.extend(["--energy-calibration", args.energy_calibration])
        if args.efficiency_coefficients:
            tokens.extend(["--efficiency-coefficients", args.efficiency_coefficients])
        if args.background_adjusted_dir:
            tokens.extend(["--background-adjusted-dir", str(args.background_adjusted_dir)])
        if args.final_corrected_dir:
            tokens.extend(["--final-corrected-dir", str(args.final_corrected_dir)])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_ingest_batch, args, tokens)

    def _run_peaks(self) -> None:
        manual_peaks_file = self.peaks_manual_file.get().strip()
        background_file = self.peaks_background_file.get().strip()
        background_scale_factor = self.peaks_background_scale_factor.get().strip()
        profile = self.peaks_profile.get().strip() or None
        energy_calibration = self.peaks_energy_calibration.get().strip() or None
        efficiency_coefficients = self.peaks_efficiency_coefficients.get().strip() or None
        args = Namespace(
            spectrum_file=Path(self.peaks_input.get()),
            output=Path(self.peaks_output.get()),
            sensitivity=self.peaks_sensitivity.get(),
            fit_window=int(self.peaks_fit_window.get()),
            manual_peaks_file=Path(manual_peaks_file) if manual_peaks_file else None,
            profile=profile,
            background_file=Path(background_file) if background_file else None,
            background_scale_mode=self.peaks_background_scale_mode.get(),
            background_scale_factor=float(background_scale_factor) if background_scale_factor else None,
            energy_calibration=energy_calibration,
            efficiency_coefficients=efficiency_coefficients,
            background_subtracted=self.peaks_background_subtracted.get(),
            validate=self.peaks_validate.get(),
        )
        tokens = [
            "peaks",
            "--spectrum-file",
            str(args.spectrum_file),
            "--output",
            str(args.output),
            "--sensitivity",
            args.sensitivity,
            "--fit-window",
            str(args.fit_window),
        ]
        if args.manual_peaks_file:
            tokens.extend(["--manual-peaks-file", str(args.manual_peaks_file)])
        if args.profile:
            tokens.extend(["--profile", args.profile])
        if args.background_file:
            tokens.extend(["--background-file", str(args.background_file)])
        if args.background_scale_mode != "live":
            tokens.extend(["--background-scale-mode", args.background_scale_mode])
        if args.background_scale_factor is not None:
            tokens.extend(["--background-scale-factor", str(args.background_scale_factor)])
        if args.energy_calibration:
            tokens.extend(["--energy-calibration", args.energy_calibration])
        if args.efficiency_coefficients:
            tokens.extend(["--efficiency-coefficients", args.efficiency_coefficients])
        if args.background_subtracted:
            tokens.append("--background-subtracted")
        if not args.validate:
            tokens.append("--no-validate")
        self._log_selected_data_source("Peaks", self.peaks_data_source.get(), self.peaks_custom_source.get().strip() or None)
        self._dispatch(cli_app.cmd_peaks, args, tokens)

    def _run_activity(self) -> None:
        live_time = self.activity_live_time.get().strip()
        sample_mass = self.activity_sample_mass_g.get().strip()
        isotope = self.activity_isotope.get().strip() or None
        reaction = self.activity_reaction.get().strip() or None
        args = Namespace(
            peaks_file=Path(self.activity_input.get()),
            output=Path(self.activity_output.get()),
            live_time_s=float(live_time) if live_time else None,
            efficiency=float(self.activity_eff.get()),
            emission_probability=float(self.activity_emission.get()),
            half_life_s=float(self.activity_half_life.get()),
            sample_mass_g=float(sample_mass) if sample_mass else None,
            isotope=isotope,
            reaction_id=reaction,
            validate=self.activity_validate.get(),
        )
        tokens = [
            "activity",
            "--peaks-file",
            str(args.peaks_file),
            "--output",
            str(args.output),
            "--efficiency",
            str(args.efficiency),
            "--emission-probability",
            str(args.emission_probability),
            "--half-life-s",
            str(args.half_life_s),
        ]
        if args.live_time_s is not None:
            tokens.extend(["--live-time-s", str(args.live_time_s)])
        if args.sample_mass_g is not None:
            tokens.extend(["--sample-mass-g", str(args.sample_mass_g)])
        if args.isotope:
            tokens.extend(["--isotope", args.isotope])
        if args.reaction_id:
            tokens.extend(["--reaction-id", args.reaction_id])
        if not args.validate:
            tokens.append("--no-validate")
        self._log_selected_data_source("Activity", self.activity_data_source.get(), self.activity_custom_source.get().strip() or None)
        self._dispatch(cli_app.cmd_activity, args, tokens, on_success=self._load_activity_result_summary)

    def _run_rates(self) -> None:
        segments = self.rates_segments.get().strip()
        args = Namespace(
            lines_file=Path(self.rates_input.get()),
            segments_file=Path(segments) if segments else None,
            duration_s=float(self.rates_duration.get()),
            half_life_s=float(self.rates_half_life.get()),
            output=Path(self.rates_output.get()),
            validate=self.rates_validate.get(),
        )
        tokens = [
            "rates",
            "--lines-file",
            str(args.lines_file),
            "--duration-s",
            str(args.duration_s),
            "--half-life-s",
            str(args.half_life_s),
            "--output",
            str(args.output),
        ]
        if args.segments_file:
            tokens.extend(["--segments-file", str(args.segments_file)])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_rates, args, tokens, on_success=self._load_rates_result_summary)

    def _run_unfold(self) -> None:
        prior = self.unfold_prior.get().strip()
        args = Namespace(
            rates_file=Path(self.unfold_rates.get()),
            response_file=Path(self.unfold_response.get()),
            prior_flux_file=Path(prior) if prior else None,
            method=self.unfold_method.get(),
            prior_uncertainty=float(self.unfold_unc.get()),
            prior_cov_model=self.unfold_cov_model.get(),
            prior_correlation_length=float(self.unfold_corr_len.get()),
            max_iters=int(self.unfold_max_iters.get()),
            tolerance=float(self.unfold_tolerance.get()),
            chi2_tolerance=float(self.unfold_chi2_tolerance.get()),
            relaxation=float(self.unfold_relaxation.get()),
            floor=float(self.unfold_floor.get()),
            convergence_mode=self.unfold_convergence_mode.get(),
            enforce_nonnegativity=self.unfold_enforce_nonnegativity.get(),
            verbose_solver=self.unfold_verbose_solver.get(),
            output=Path(self.unfold_output.get()),
            validate=self.unfold_validate.get(),
        )
        tokens = [
            "unfold",
            "--rates-file",
            str(args.rates_file),
            "--response-file",
            str(args.response_file),
            "--method",
            args.method,
            "--prior-uncertainty",
            str(args.prior_uncertainty),
            "--prior-cov-model",
            args.prior_cov_model,
            "--prior-correlation-length",
            str(args.prior_correlation_length),
            "--max-iters",
            str(args.max_iters),
            "--tolerance",
            str(args.tolerance),
            "--chi2-tolerance",
            str(args.chi2_tolerance),
            "--relaxation",
            str(args.relaxation),
            "--floor",
            str(args.floor),
            "--output",
            str(args.output),
        ]
        if args.prior_flux_file:
            tokens.extend(["--prior-flux-file", str(args.prior_flux_file)])
        if args.method == "mlem" and args.convergence_mode != "relative":
            tokens.extend(["--convergence-mode", args.convergence_mode])
        if not args.enforce_nonnegativity:
            tokens.append("--no-enforce-nonnegativity")
        if args.verbose_solver:
            tokens.append("--verbose-solver")
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_unfold, args, tokens, on_success=self._load_unfold_result_preview)

    def _run_compare(self) -> None:
        args = Namespace(
            unfold_file=Path(self.compare_unfold.get()),
            truth_flux_file=Path(self.compare_truth.get()),
            output=Path(self.compare_output.get()),
            validate=self.compare_validate.get(),
        )
        tokens = [
            "compare",
            "--unfold-file",
            str(args.unfold_file),
            "--truth-flux-file",
            str(args.truth_flux_file),
            "--output",
            str(args.output),
        ]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_compare, args, tokens, on_success=self._load_compare_result_summary)

    def _run_reactions(self) -> None:
        source_id = self.standards_data_source.get()
        if source_id != "irdff_ii_dosimetry":
            self._browse_standards_source()
            return
        target = self.standards_target.get().strip() or None
        args = Namespace(
            category=self.standards_reaction_category.get(),
            target=target,
            format=self.standards_format.get(),
        )
        tokens = ["reactions", "--category", args.category, "--format", args.format]
        if args.target:
            tokens.extend(["--target", args.target])
        self._log_selected_data_source("Standards", source_id, self.standards_custom_source.get().strip() or None)
        self._dispatch(cli_app.cmd_reactions, args, tokens)

    def _run_k0_detector(self) -> None:
        spectrum_hint = self.k0_spectrum_input.get().strip()
        detector_id = Path(spectrum_hint).stem if spectrum_hint else "hpge-detector"
        args = Namespace(
            points_file=Path(self.k0_detector_points.get()),
            detector_id=detector_id,
            reference_position_mm=200.0,
            degree=2,
            peak_to_total_ratio=None,
            coincidence_mode="not_applied",
            output=Path(self.k0_detector_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = [
            "k0-detector",
            "--points-file",
            str(args.points_file),
            "--detector-id",
            args.detector_id,
            "--reference-position-mm",
            str(args.reference_position_mm),
            "--degree",
            str(args.degree),
            "--coincidence-mode",
            args.coincidence_mode,
            "--output",
            str(args.output),
        ]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_detector, args, tokens)

    def _run_k0_facility(self) -> None:
        args = Namespace(
            input=Path(self.k0_facility_input.get()),
            output=Path(self.k0_facility_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = ["k0-facility", "--input", str(args.input), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_facility, args, tokens)

    def _run_k0_normalize(self) -> None:
        args = Namespace(
            peaks_file=Path(self.k0_peaks_input.get()),
            spectrum_file=self._optional_path(self.k0_spectrum_input.get()),
            detector_characterization_file=self._optional_path(self.k0_detector_output.get()),
            detector_id=None,
            geometry_id=None,
            irradiation_time_s=float(self.k0_irradiation_time_s.get()),
            decay_time_s=float(self.k0_decay_time_s.get()),
            counting_time_s=None,
            import_format="peak_report",
            project_id=self.k0_project_id.get().strip() or None,
            sample_id=self.k0_sample_id.get().strip() or None,
            irradiation_id=self.k0_irradiation_id.get().strip() or None,
            measurement_id=self.k0_measurement_id.get().strip() or None,
            expert_override=False,
            allow_advanced_lines=False,
            output=Path(self.k0_observations_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = [
            "k0-normalize",
            "--peaks-file",
            str(args.peaks_file),
            "--irradiation-time-s",
            str(args.irradiation_time_s),
            "--decay-time-s",
            str(args.decay_time_s),
            "--output",
            str(args.output),
        ]
        if args.spectrum_file:
            tokens.extend(["--spectrum-file", str(args.spectrum_file)])
        if args.detector_characterization_file:
            tokens.extend(["--detector-characterization-file", str(args.detector_characterization_file)])
        if args.project_id:
            tokens.extend(["--project-id", args.project_id])
        if args.sample_id:
            tokens.extend(["--sample-id", args.sample_id])
        if args.irradiation_id:
            tokens.extend(["--irradiation-id", args.irradiation_id])
        if args.measurement_id:
            tokens.extend(["--measurement-id", args.measurement_id])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_normalize, args, tokens)

    def _run_k0_analyze(self) -> None:
        args = Namespace(
            observations_file=Path(self.k0_observations_output.get()),
            facility_file=Path(self.k0_facility_output.get()),
            sample_mass_g=float(self.k0_sample_mass_g.get()),
            reference_isotope=self.k0_reference_isotope.get().strip() or "Au-198",
            reference_mass_g=float(self.k0_reference_mass_g.get()),
            k0_library_file=self._optional_path(self.k0_library_file.get()),
            auxiliary_library_file=self._optional_path(self.k0_aux_library_file.get()),
            output=Path(self.k0_analysis_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = [
            "k0-analyze",
            "--observations-file",
            str(args.observations_file),
            "--facility-file",
            str(args.facility_file),
            "--sample-mass-g",
            str(args.sample_mass_g),
            "--reference-isotope",
            args.reference_isotope,
            "--reference-mass-g",
            str(args.reference_mass_g),
            "--output",
            str(args.output),
        ]
        if args.k0_library_file:
            tokens.extend(["--k0-library-file", str(args.k0_library_file)])
        if args.auxiliary_library_file:
            tokens.extend(["--auxiliary-library-file", str(args.auxiliary_library_file)])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_analyze, args, tokens, on_success=self._load_k0_preview)

    def _run_k0_aggregate(self) -> None:
        analysis_file = Path(self.k0_analysis_output.get())
        args = Namespace(
            analysis_files=[analysis_file],
            output=Path(self.k0_aggregation_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = ["k0-aggregate", "--analysis-files", str(analysis_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_aggregate, args, tokens)

    def _run_k0_qaqc(self) -> None:
        args = Namespace(
            plan_file=Path(self.k0_qaqc_plan.get()),
            output=Path(self.k0_qaqc_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = ["k0-qaqc", "--plan-file", str(args.plan_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_qaqc, args, tokens)

    def _run_k0_report(self) -> None:
        args = Namespace(
            analysis_file=Path(self.k0_analysis_output.get()),
            aggregation_file=self._optional_path(self.k0_aggregation_output.get()),
            qaqc_file=self._optional_path(self.k0_qaqc_output.get()),
            output=Path(self.k0_report_output.get()),
            validate=self.k0_validate.get(),
        )
        tokens = ["k0-report", "--analysis-file", str(args.analysis_file), "--output", str(args.output)]
        if args.aggregation_file:
            tokens.extend(["--aggregation-file", str(args.aggregation_file)])
        if args.qaqc_file:
            tokens.extend(["--qaqc-file", str(args.qaqc_file)])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_k0_report, args, tokens, on_success=self._load_k0_preview)

    def _run_astm_e2005(self) -> None:
        args = Namespace(
            plan_file=Path(self.astm_e2005_plan.get()) if hasattr(self, 'astm_e2005_plan') else Path('astm_e2005_plan.json'),
            output=Path(self.astm_e2005_output.get()) if hasattr(self, 'astm_e2005_output') else Path('astm_e2005.json'),
            validate=self.astm_e2005_validate.get() if hasattr(self, 'astm_e2005_validate') else True,
        )
        tokens = ["astm-e2005", "--plan-file", str(args.plan_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_astm_e2005, args, tokens, on_success=self._load_astm_e2005_preview)

    def _run_astm_e261(self) -> None:
        args = Namespace(
            plan_file=Path(self.astm_e261_plan.get()),
            output=Path(self.astm_e261_output.get()),
            validate=self.astm_e261_validate.get(),
        )
        tokens = ["astm-e261", "--plan-file", str(args.plan_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_astm_e261, args, tokens, on_success=self._load_astm_e261_preview)

    def _run_astm_e262(self) -> None:
        args = Namespace(
            plan_file=Path(self.astm_e262_plan.get()),
            output=Path(self.astm_e262_output.get()),
            validate=self.astm_e262_validate.get(),
        )
        tokens = ["astm-e262", "--plan-file", str(args.plan_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_astm_e262, args, tokens, on_success=self._load_astm_e262_preview)

    def _run_astm_e3376(self) -> None:
        args = Namespace(
            plan_file=Path(self.astm_e3376_plan.get()),
            output=Path(self.astm_e3376_output.get()),
            validate=self.astm_e3376_validate.get(),
        )
        tokens = ["astm-e3376", "--plan-file", str(args.plan_file), "--output", str(args.output)]
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_astm_e3376, args, tokens, on_success=self._load_astm_e3376_preview)

    def _run_rafm_validate(self) -> None:
        max_spectra = self.rafm_max_spectra.get().strip()
        flux_wire_method = self.rafm_flux_wire_counting_method.get().strip() or None
        generic_method = self.rafm_generic_counting_method.get().strip() or None
        if flux_wire_method and flux_wire_method not in GUI_RAFM_COUNTING_METHODS:
            raise ValueError(f"Unsupported RAFM flux-wire counting method: {flux_wire_method}")
        if generic_method and generic_method not in GUI_RAFM_COUNTING_METHODS:
            raise ValueError(f"Unsupported RAFM generic counting method: {generic_method}")
        args = Namespace(
            example_root=Path(self.rafm_example_root.get()),
            results_root=Path(self.rafm_raw_results_root.get()),
            max_spectra=int(max_spectra) if max_spectra else None,
            flux_wire_counting_method=flux_wire_method,
            generic_counting_method=generic_method,
            no_fail=not self.rafm_enforce_thresholds.get(),
        )
        tokens = ["rafm-validate", "--example-root", str(args.example_root), "--results-root", str(args.results_root)]
        if args.max_spectra is not None:
            tokens.extend(["--max-spectra", str(args.max_spectra)])
        if args.flux_wire_counting_method:
            tokens.extend(["--flux-wire-counting-method", args.flux_wire_counting_method])
        if args.generic_counting_method:
            tokens.extend(["--generic-counting-method", args.generic_counting_method])
        if args.no_fail:
            tokens.append("--no-fail")
        self._dispatch(cli_app.cmd_rafm_validate, args, tokens, on_success=self._after_rafm_validate_run)

    def _run_rafm_qg_benchmark(self) -> None:
        max_spectra = self.rafm_max_spectra.get().strip()
        args = Namespace(
            example_root=Path(self.rafm_example_root.get()),
            results_root=Path(self.rafm_qg_results_root.get()),
            max_spectra=int(max_spectra) if max_spectra else None,
        )
        tokens = ["rafm-qg-benchmark", "--example-root", str(args.example_root), "--results-root", str(args.results_root)]
        if args.max_spectra is not None:
            tokens.extend(["--max-spectra", str(args.max_spectra)])
        self._dispatch(cli_app.cmd_rafm_qg_benchmark, args, tokens, on_success=self._after_rafm_qg_benchmark_run)

    def _run_rafm_compare_branches(self) -> None:
        args = Namespace(
            raw_results_root=Path(self.rafm_raw_results_root.get()),
            qg_results_root=Path(self.rafm_qg_results_root.get()),
            output_root=Path(self.rafm_comparison_root.get()),
        )
        tokens = [
            "rafm-compare-branches",
            "--raw-results-root",
            str(args.raw_results_root),
            "--qg-results-root",
            str(args.qg_results_root),
            "--output-root",
            str(args.output_root),
        ]
        self._dispatch(cli_app.cmd_rafm_compare_branches, args, tokens, on_success=self._after_rafm_compare_run)

    def _run_report(self) -> None:
        args = Namespace(
            spectrum_file=self._optional_path(self.report_spectrum.get()),
            peaks_file=self._optional_path(self.report_peaks.get()),
            lines_file=self._optional_path(self.report_lines.get()),
            rates_file=self._optional_path(self.report_rates.get()),
            unfold_file=self._optional_path(self.report_unfold.get()),
            validation_file=self._optional_path(self.report_validation.get()),
            validation_results_root=self._optional_path(self.report_validation_results_root.get()),
            output=Path(self.report_output.get()),
            validate=self.report_validate.get(),
        )
        tokens = ["report"]
        if args.spectrum_file:
            tokens.extend(["--spectrum-file", str(args.spectrum_file)])
        if args.peaks_file:
            tokens.extend(["--peaks-file", str(args.peaks_file)])
        if args.lines_file:
            tokens.extend(["--lines-file", str(args.lines_file)])
        if args.rates_file:
            tokens.extend(["--rates-file", str(args.rates_file)])
        if args.unfold_file:
            tokens.extend(["--unfold-file", str(args.unfold_file)])
        if args.validation_file:
            tokens.extend(["--validation-file", str(args.validation_file)])
        if args.validation_results_root:
            tokens.extend(["--validation-results-root", str(args.validation_results_root)])
        tokens.extend(["--output", str(args.output)])
        if not args.validate:
            tokens.append("--no-validate")
        self._dispatch(cli_app.cmd_report, args, tokens, on_success=self._after_report_run)
