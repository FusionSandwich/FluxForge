import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge_gui.app import (
    GuiManualRegion,
    GuiCalibrationPoint,
    GuiDiagnosticPlot,
    GuiDiagnosticSeries,
    GuiSpectrumSeries,
    ALLOWED_REACTION_CATEGORIES,
    auto_detect_gui_peaks,
    build_gui_k0_preview,
    build_gui_astm_e2005_preview,
    build_gui_report_preview,
    discover_gui_validation_report_inputs,
    build_gui_spectrum_preview,
    build_standards_preset_values,
    combine_gui_spectrum_series,
    count_gui_peak,
    fit_gui_energy_calibration,
    get_gui_profile_choices,
    get_standards_gui_presets,
    parse_gui_constraint_matrix,
    render_gui_activity_result,
    render_gui_rate_result,
    render_gui_unfold_result,
    render_gui_validation_result,
    render_gui_spectrum_preview,
    save_gui_spectrum_preview_image,
    summarize_gui_activity_result,
    summarize_gui_rate_result,
    summarize_gui_unfold_result,
    summarize_gui_validation_result,
)
from fluxforge.io.artifacts import write_peak_report, write_spectrum_file
from fluxforge.io.spe import GammaSpectrum


def test_gui_profile_choices_include_standards_profiles():
    choices = get_gui_profile_choices()

    assert choices[0] == ""
    assert "astm_inl_dosimetry" in choices
    assert "us_astm_reactor_dosimetry" in choices


def test_standards_presets_use_known_profiles_and_categories():
    presets = get_standards_gui_presets()
    choices = set(get_gui_profile_choices())

    assert {
        "astm_inl",
        "us_astm",
        "iaea_irdff_gma",
        "k0_naa",
        "comparator_naa",
        "curie_like",
    }.issubset(presets)
    for preset in presets.values():
        if preset.default_profile is not None:
            assert preset.default_profile in choices
        assert preset.reaction_category in ALLOWED_REACTION_CATEGORIES
        assert preset.peaks_sensitivity in {"default", "sensitive", "conservative"}
        assert preset.peak_counting_method in {
            "gaussian_fit",
            "covell_local",
            "iec_tiered",
        }
        assert preset.notes


def test_build_standards_preset_values_uses_default_profile():
    values = build_standards_preset_values("astm_inl")

    assert values["profile"] == "astm_inl_dosimetry"
    assert values["peaks_sensitivity"] == "conservative"
    assert values["peak_counting_method"] == "iec_tiered"
    assert values["background_subtracted"] is True
    assert values["reaction_category"] == "fast"
    assert "ASTM/INL detector profile" in values["notes"]


def test_build_standards_preset_values_honors_override_profile():
    values = build_standards_preset_values("astm_inl", "us_astm_reactor_dosimetry")

    assert values["profile"] == "us_astm_reactor_dosimetry"


def test_k0_naa_preset_keeps_profile_optional():
    values = build_standards_preset_values("k0_naa")

    assert values["profile"] == ""
    assert values["peaks_sensitivity"] == "sensitive"
    assert values["reaction_category"] == "thermal"


def test_iaea_preset_reuses_dosimetry_profile_with_full_category_view():
    values = build_standards_preset_values("iaea_irdff_gma")

    assert values["profile"] == "astm_inl_dosimetry"
    assert values["reaction_category"] == "all"
    assert "IRDFF-II reactions" in values["notes"]


def test_curie_like_preset_highlights_api_backed_workflow():
    values = build_standards_preset_values("curie_like")

    assert values["profile"] == ""
    assert values["background_subtracted"] is True
    assert values["reaction_category"] == "all"
    assert "stacked-target" in values["notes"]


def test_build_gui_spectrum_preview_reads_primary_overlay_and_peaks(tmp_path):
    primary = GammaSpectrum(
        counts=np.array([0.0, 2.0, 5.0, 3.0, 1.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="primary",
        calibration={"energy": [0.0, 100.0]},
    )
    overlay = GammaSpectrum(
        counts=np.array([1.0, 1.5, 2.0, 1.5, 1.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="overlay",
        calibration={"energy": [0.0, 100.0]},
    )
    primary_path = tmp_path / "spectrum.json"
    overlay_path = tmp_path / "overlay.json"
    peaks_path = tmp_path / "peaks.json"
    write_spectrum_file(primary_path, primary)
    write_spectrum_file(overlay_path, overlay)
    write_peak_report(
        peaks_path,
        spectrum_id="primary",
        live_time_s=10.0,
        peaks=[
            {"channel": 2, "energy_keV": 200.0, "area": 42.0, "label": "Cs-like"},
        ],
    )

    preview = build_gui_spectrum_preview(
        primary_path, overlay_paths=[overlay_path], peaks_path=peaks_path
    )

    assert preview.primary.label == "primary"
    assert preview.primary.channels.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert preview.primary.calibration_coeffs == (0.0, 100.0)
    assert len(preview.overlays) == 1
    assert preview.overlays[0].label == "overlay"
    assert preview.peaks[0].energy_keV == 200.0
    assert preview.peaks[0].area == 42.0
    assert preview.primary.energies_keV.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0]


def test_build_gui_k0_preview_accepts_report_bundle(tmp_path):
    report_path = tmp_path / "k0_report.json"
    text_path = tmp_path / "k0_report.txt"
    text_path.write_text("demo k0 report\n", encoding="utf-8")

    preview = build_gui_k0_preview(
        {
            "schema": "fluxforge.report_bundle.v1",
            "summary": {"element_count": 1},
            "text_report": {"path": text_path.name, "format": "text/plain"},
        },
        report_path,
    )

    assert "demo k0 report" in preview


def test_build_gui_astm_e2005_preview_formats_transfers_and_indices(tmp_path):
    bundle_path = tmp_path / "astm_e2005.json"
    payload = {
        "fluence_transfers": [
            {
                "transfer_id": "ni_transfer",
                "fluence_rate_cm2_s": 1.23e8,
                "fluence_rate_unc_cm2_s": 4.56e6,
            }
        ],
        "spectral_indices": [
            {
                "index_id": "co_pair",
                "measured_index": 0.98,
                "measured_index_unc": 0.02,
                "calculated_index": 1.03,
                "calculated_index_unc": 0.03,
                "c_e_ratio": 1.051,
                "c_e_ratio_unc": 0.04,
            }
        ],
    }

    preview = build_gui_astm_e2005_preview(payload, bundle_path)

    assert "FluxForge ASTM E2005 Preview" in preview
    assert "Transfer: ni_transfer" in preview
    assert "Index: co_pair" in preview
    assert "C/E Ratio: 1.051" in preview


def test_render_gui_spectrum_preview_plots_series_and_peak_markers(tmp_path):
    spectrum = GammaSpectrum(
        counts=np.array([0.0, 4.0, 8.0, 4.0, 0.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="demo",
        calibration={"energy": [0.0, 50.0]},
    )
    spectrum_path = tmp_path / "demo.json"
    peaks_path = tmp_path / "demo_peaks.json"
    write_spectrum_file(spectrum_path, spectrum)
    write_peak_report(
        peaks_path,
        spectrum_id="demo",
        live_time_s=10.0,
        peaks=[{"channel": 2, "energy_keV": 100.0, "area": 8.0, "label": "peak"}],
    )

    preview = build_gui_spectrum_preview(spectrum_path, peaks_path=peaks_path)
    figure, axes = render_gui_spectrum_preview(
        preview,
        manual_regions=[GuiManualRegion(label="ROI 1", left_keV=75.0, right_keV=125.0)],
        selected_peak_energy_keV=100.0,
        y_log=False,
    )

    assert figure is not None
    assert axes.get_xlabel() == "Energy (keV)"
    assert len(axes.lines) >= 3  # main spectrum + peak guide line + selected peak line


def test_render_gui_spectrum_preview_supports_diagnostic_subplot(tmp_path):
    spectrum = GammaSpectrum(
        counts=np.array([1.0, 3.0, 6.0, 3.0, 1.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="diag",
        calibration={"energy": [0.0, 50.0]},
    )
    spectrum_path = tmp_path / "diag.json"
    write_spectrum_file(spectrum_path, spectrum)
    preview = build_gui_spectrum_preview(spectrum_path)

    figure, axes = render_gui_spectrum_preview(
        preview,
        diagnostic_plot=GuiDiagnosticPlot(
            title="Residuals",
            x_label="Channel",
            y_label="Residual",
            series=(
                GuiDiagnosticSeries(
                    label="res",
                    x=np.arange(5, dtype=float),
                    y=np.array([0.0, 1.0, 0.0, -1.0, 0.0]),
                    style="scatter",
                ),
            ),
            reference_y=0.0,
        ),
    )

    assert figure is not None
    assert axes.get_title() == "FluxForge Spectrum Viewer"
    assert len(figure.axes) == 2


def test_render_gui_unfold_result_supports_convergence_history():
    payload = {
        "method": "mlem",
        "boundaries_eV": [1e-5, 1e-3, 1.0, 1e3],
        "flux": [2.0, 5.0, 1.5],
        "covariance": [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.01]],
        "chi2": 0.42,
        "diagnostics": {
            "iterations": 12,
            "converged": True,
            "chi2_history": [4.0, 1.5, 0.42],
            "measured_rates": [1.0, 2.0],
            "measured_rate_uncertainties": [0.1, 0.2],
            "predicted_rates": [1.1, 1.8],
            "predicted_rate_uncertainties": [0.05, 0.08],
            "rate_residuals": [0.1, -0.2],
            "rate_pulls": [1.0, -1.0],
            "reactions": ["Au", "Co"],
        },
    }

    figure = render_gui_unfold_result(payload)

    assert figure is not None
    assert len(figure.axes) >= 4
    assert "MLEM" in figure.axes[0].get_title()


def test_render_gui_activity_and_rate_results_support_uncertainties():
    activity_figure = render_gui_activity_result(
        {
            "lines": [
                {
                    "isotope": "Sc46",
                    "energy_keV": 889.3,
                    "activity_Bq": 120.0,
                    "activity_unc_Bq": 6.0,
                },
                {
                    "isotope": "Sc46",
                    "energy_keV": 1120.5,
                    "activity_Bq": 100.0,
                    "activity_unc_Bq": 7.0,
                },
            ]
        }
    )
    rate_figure = render_gui_rate_result(
        {
            "rates": [
                {"reaction_id": "Ti-46(n,p)Sc-46", "rate": 5.0, "uncertainty": 0.4},
                {"reaction_id": "Fe-54(n,p)Mn-54", "rate": 2.0, "uncertainty": 0.3},
            ]
        }
    )

    assert activity_figure is not None
    assert rate_figure is not None
    assert activity_figure.axes[0].get_ylabel() == "Activity (Bq)"
    assert rate_figure.axes[0].get_ylabel() == "Rate (reactions/s)"


def test_render_gui_validation_result_plots_flux_and_residuals():
    figure = render_gui_validation_result(
        {
            "metrics": {"mae": 0.02},
            "truth_flux": [1.0, 2.0, 3.0],
            "predicted_flux": [1.1, 1.9, 2.8],
            "residuals": [0.1, -0.1, -0.2],
        }
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Validation flux comparison"
    assert figure.axes[1].get_title() == "Validation residuals"


def test_gui_render_helpers_avoid_matplotlib_deprecation(tmp_path):
    spectrum = GammaSpectrum(
        counts=np.array([0.0, 4.0, 8.0, 4.0, 0.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="demo",
        calibration={"energy": [0.0, 50.0]},
    )
    spectrum_path = tmp_path / "demo.json"
    peaks_path = tmp_path / "demo_peaks.json"
    write_spectrum_file(spectrum_path, spectrum)
    write_peak_report(
        peaks_path,
        spectrum_id="demo",
        live_time_s=10.0,
        peaks=[{"channel": 2, "energy_keV": 100.0, "area": 8.0, "label": "peak"}],
    )
    preview = build_gui_spectrum_preview(spectrum_path, peaks_path=peaks_path)
    preview_png = tmp_path / "preview.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error", matplotlib.MatplotlibDeprecationWarning)
        save_gui_spectrum_preview_image(preview, preview_png)
        activity_figure = render_gui_activity_result(
            {
                "lines": [
                    {
                        "isotope": "Sc46",
                        "energy_keV": 889.3,
                        "activity_Bq": 120.0,
                        "activity_unc_Bq": 6.0,
                    },
                    {
                        "isotope": "Sc46",
                        "energy_keV": 1120.5,
                        "activity_Bq": 100.0,
                        "activity_unc_Bq": 7.0,
                    },
                ]
            }
        )
        rate_figure = render_gui_rate_result(
            {
                "rates": [
                    {
                        "reaction_id": "Ti-46(n,p)Sc-46",
                        "rate": 5.0,
                        "uncertainty": 0.4,
                    },
                    {
                        "reaction_id": "Fe-54(n,p)Mn-54",
                        "rate": 2.0,
                        "uncertainty": 0.3,
                    },
                ]
            }
        )
        validation_figure = render_gui_validation_result(
            {
                "metrics": {"mae": 0.02},
                "truth_flux": [1.0, 2.0, 3.0],
                "predicted_flux": [1.1, 1.9, 2.8],
                "residuals": [0.1, -0.1, -0.2],
            }
        )
        unfold_figure = render_gui_unfold_result(
            {
                "method": "mlem",
                "boundaries_eV": [1e-5, 1e-3, 1.0, 1e3],
                "flux": [2.0, 5.0, 1.5],
                "covariance": [
                    [0.04, 0.0, 0.0],
                    [0.0, 0.09, 0.0],
                    [0.0, 0.0, 0.01],
                ],
                "chi2": 0.42,
                "diagnostics": {
                    "iterations": 12,
                    "converged": True,
                    "chi2_history": [4.0, 1.5, 0.42],
                    "measured_rates": [1.0, 2.0],
                    "measured_rate_uncertainties": [0.1, 0.2],
                    "predicted_rates": [1.1, 1.8],
                    "predicted_rate_uncertainties": [0.05, 0.08],
                    "rate_residuals": [0.1, -0.2],
                    "rate_pulls": [1.0, -1.0],
                    "reactions": ["Au", "Co"],
                },
            }
        )

    assert preview_png.exists()
    assert activity_figure is not None
    assert rate_figure is not None
    assert validation_figure is not None
    assert unfold_figure is not None


def test_summarize_gui_unfold_result_reports_method_and_iterations():
    summary = summarize_gui_unfold_result(
        {
            "method": "gravel",
            "boundaries_eV": [1e-5, 1e-3, 1.0],
            "flux": [3.0, 1.0],
            "chi2": 0.35,
            "diagnostics": {"iterations": 18, "converged": True},
        }
    )

    assert "GRAVEL" in summary
    assert "Iterations: 18" in summary
    assert "Converged" in summary


def test_activity_and_rate_summaries_include_uncertainty_context():
    activity_summary = summarize_gui_activity_result(
        {
            "lines": [
                {
                    "isotope": "Sc46",
                    "energy_keV": 889.3,
                    "activity_Bq": 120.0,
                    "activity_unc_Bq": 6.0,
                    "radioactive_mass_g": 1e-12,
                    "specific_activity_Bq_g": 240.0,
                },
                {
                    "isotope": "Sc46",
                    "energy_keV": 1120.5,
                    "activity_Bq": 100.0,
                    "activity_unc_Bq": 7.0,
                    "radioactive_mass_g": 2e-12,
                    "specific_activity_Bq_g": 200.0,
                },
            ]
        }
    )
    rate_summary = summarize_gui_rate_result(
        {
            "rates": [
                {"reaction_id": "Ti-46(n,p)Sc-46", "rate": 5.0, "uncertainty": 0.4},
                {"reaction_id": "Fe-54(n,p)Mn-54", "rate": 2.0, "uncertainty": 0.3},
            ]
        }
    )

    assert "Median relative σ" in activity_summary
    assert "Sc46" in activity_summary
    assert "Total radioactive mass" in activity_summary
    assert "Median relative σ" in rate_summary
    assert "Ti-46(n,p)Sc-46" in rate_summary


def test_validation_summary_reports_final_metrics():
    summary = summarize_gui_validation_result(
        {
            "metrics": {"mae": 0.02, "rmse": 0.03, "chi2": 0.8},
            "truth_flux": [1.0, 2.0, 3.0],
            "predicted_flux": [1.1, 1.9, 2.8],
            "residuals": [0.1, -0.1, -0.2],
        }
    )

    assert "MAE" in summary
    assert "CHI2" in summary
    assert "Max |residual|" in summary


def test_combine_gui_spectrum_series_supports_buffer_arithmetic():
    left = GuiSpectrumSeries(
        label="left",
        channels=np.arange(4, dtype=float),
        energies_keV=np.arange(4, dtype=float) * 10.0,
        counts=np.array([4.0, 8.0, 12.0, 16.0]),
        calibration_coeffs=(0.0, 10.0),
    )
    right = GuiSpectrumSeries(
        label="right",
        channels=np.arange(4, dtype=float),
        energies_keV=np.arange(4, dtype=float) * 10.0,
        counts=np.array([2.0, 4.0, 6.0, 8.0]),
        calibration_coeffs=(0.0, 10.0),
    )

    combined = combine_gui_spectrum_series(
        left, right, operation="ratio", label="ratio"
    )

    assert combined.label == "ratio"
    assert combined.counts.tolist() == [2.0, 2.0, 2.0, 2.0]
    assert combined.calibration_coeffs == (0.0, 10.0)


def test_fit_gui_energy_calibration_returns_coefficients_and_residuals():
    fit = fit_gui_energy_calibration(
        [
            GuiCalibrationPoint(
                channel=0.0,
                observed_energy_keV=0.0,
                reference_energy_keV=5.0,
                label="p0",
            ),
            GuiCalibrationPoint(
                channel=10.0,
                observed_energy_keV=100.0,
                reference_energy_keV=25.0,
                label="p1",
            ),
            GuiCalibrationPoint(
                channel=20.0,
                observed_energy_keV=200.0,
                reference_energy_keV=45.0,
                label="p2",
            ),
        ],
        order=1,
    )

    assert np.isclose(fit.coefficients[0], 5.0)
    assert np.isclose(fit.coefficients[1], 2.0)
    assert np.allclose(fit.residuals_keV, 0.0)
    assert np.isclose(fit.r_squared, 1.0)


def test_auto_detect_gui_peaks_and_count_selected_peak():
    channels = np.arange(0, 1024, dtype=float)
    energies = channels.copy()
    counts = 5.0 + 400.0 * np.exp(-0.5 * ((channels - 662.0) / 4.0) ** 2)
    series = GuiSpectrumSeries(
        label="demo",
        channels=channels,
        energies_keV=energies,
        counts=counts,
        calibration_coeffs=(0.0, 1.0),
    )

    peaks = auto_detect_gui_peaks(
        series,
        finder_method="scipy",
        threshold_sigma=3.0,
        min_distance=10,
        identification_method="line_match",
        tolerance_keV=2.0,
        min_matches=1,
    )

    assert peaks
    selected = min(peaks, key=lambda item: abs(item.energy_keV - 662.0))
    assert abs(selected.energy_keV - 662.0) < 2.0

    result = count_gui_peak(series, selected, "covell_local")

    assert result.net_counts > 0.0
    assert result.net_uncertainty > 0.0
    assert result.roi_bounds[0] < result.roi_bounds[1]


def test_parse_gui_constraint_matrix_parses_free_form_text():
    matrix = parse_gui_constraint_matrix("1 0\n0.5 1\n0 1", 3)

    assert matrix.shape == (3, 2)
    assert np.allclose(matrix[1], [0.5, 1.0])


def test_build_gui_report_preview_prefers_text_report(tmp_path):
    report_path = tmp_path / "report.json"
    text_path = tmp_path / "report.txt"
    text_path.write_text(
        "FluxForge Standard Activation Report\nActivity summary\n", encoding="utf-8"
    )

    preview = build_gui_report_preview(
        {"text_report": {"path": "report.txt"}, "summary": {"peak_count": 3}},
        report_path,
    )

    assert "FluxForge Standard Activation Report" in preview
    assert "Activity summary" in preview


def test_build_gui_report_preview_falls_back_to_summary(tmp_path):
    preview = build_gui_report_preview(
        {"summary": {"peak_count": 3, "chi2": 0.1}}, tmp_path / "report.json"
    )

    assert "FluxForge Report Preview" in preview
    assert "peak_count: 3" in preview
    assert "chi2: 0.1" in preview


def test_build_gui_k0_preview_formats_element_results(tmp_path):
    preview = build_gui_k0_preview(
        {
            "summary": {"element_count": 1, "reference_isotope": "Au-198"},
            "element_results": [
                {
                    "element": "Co",
                    "concentration_ug_g": 12.5,
                    "concentration_unc_ug_g": 0.8,
                }
            ],
            "recognized_but_not_applied": ["westcott_gT_not_applied:Lu-176"],
            "capability_flags": {"supports_thermal_inaa": "validated"},
            "libraries": {
                "standard_k0_library": {
                    "library_id": "fluxforge.k0.starter",
                    "version": "2026.03-starter-v1",
                }
            },
        },
        tmp_path / "k0_analysis.json",
    )

    assert "FluxForge k0-NAA Preview" in preview
    assert "Co: 12.5 ± 0.8 ug/g" in preview
    assert "westcott_gT_not_applied:Lu-176" in preview
    assert "fluxforge.k0.starter" in preview


def test_discover_gui_validation_report_inputs_prefers_gls(tmp_path):
    results_root = tmp_path / "rafm_validation"
    (results_root / "unfolding").mkdir(parents=True)
    (results_root / "unfolding" / "gls.json").write_text("{}", encoding="utf-8")

    defaults = discover_gui_validation_report_inputs(results_root)

    assert defaults["validation_results_root"] == str(results_root)
    assert defaults["unfold_file"] == str(results_root / "unfolding" / "gls.json")
    assert defaults["output"] == str(results_root / "validation_report.json")
    assert defaults["figure_dir"] == str(results_root / "validation_report_figures")
