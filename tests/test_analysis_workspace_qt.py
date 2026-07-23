import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.analysis.detector_calibration import EfficiencyPoint  # noqa: E402
from fluxforge.core.analysis_workspace import (  # noqa: E402
    ActivityCalculationResult,
    bayesian_match_peak_candidates,
    calculate_peak_activity,
    compute_cascade_sum_lines,
    detect_peak_candidates,
    estimate_spectral_phenomena,
    extract_survey_points,
    fit_efficiency_model,
    subtract_background_counts,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.dialogs.auto_peak_review_dialog import (
    AutoPeakReviewDialog,
)  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import GUIMode, ModeManager, ModeState  # noqa: E402
from fluxforge.gui.nuclide_search import (
    GammaLineMatchResult,
    NuclideSearchController,
)  # noqa: E402
from fluxforge.gui.panels.modern_shell import (  # noqa: E402
    build_demo_background_spectrum,
    build_demo_overlay_spectrum,
    build_demo_spectrum,
)
from fluxforge.gui.analysis_workspace import (  # noqa: E402
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    SpectrumSlot,
)
from fluxforge.io.flux_wire import EfficiencyCalibration  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from tests._phase6_real_data import (  # noqa: E402
    DEFAULT_PHASE6_SAMPLE_ID,
    load_phase6_real_activity_results,
    load_phase6_real_activity_review,
    load_phase6_real_optimization_grids,
    load_phase6_real_target_weights_text,
)

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402
    from PySide6.QtWidgets import QDialog, QInputDialog, QPushButton  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


def _make_efficiency_points():
    return (
        EfficiencyPoint(
            energy_keV=121.78,
            net_counts=82000.0,
            live_time_s=100.0,
            activity_bq=1e5,
            emission_probability=1.0,
            count_uncertainty=286.0,
        ),
        EfficiencyPoint(
            energy_keV=356.01,
            net_counts=41000.0,
            live_time_s=100.0,
            activity_bq=1e5,
            emission_probability=1.0,
            count_uncertainty=203.0,
        ),
        EfficiencyPoint(
            energy_keV=661.657,
            net_counts=19000.0,
            live_time_s=100.0,
            activity_bq=1e5,
            emission_probability=1.0,
            count_uncertainty=138.0,
        ),
        EfficiencyPoint(
            energy_keV=1173.228,
            net_counts=12500.0,
            live_time_s=100.0,
            activity_bq=1e5,
            emission_probability=1.0,
            count_uncertainty=112.0,
        ),
        EfficiencyPoint(
            energy_keV=1332.492,
            net_counts=10300.0,
            live_time_s=100.0,
            activity_bq=1e5,
            emission_probability=1.0,
            count_uncertainty=101.0,
        ),
    )


def _seed_phase6_real_workspace(window: FluxForgeMainWindow) -> dict[str, str]:
    review = load_phase6_real_activity_review(DEFAULT_PHASE6_SAMPLE_ID)
    results = load_phase6_real_activity_results(DEFAULT_PHASE6_SAMPLE_ID)
    window.analysis_workspace.set_activity_results(results)
    window.bottom_dock.widget().activity_results_panel._last_activity_review = (
        review  # noqa: SLF001
    )
    return load_phase6_real_optimization_grids(DEFAULT_PHASE6_SAMPLE_ID)


def test_analysis_core_helpers_cover_workspace_surfaces():
    foreground = build_demo_spectrum()
    background = build_demo_background_spectrum()
    overlay = build_demo_overlay_spectrum()

    peaks = detect_peak_candidates(foreground)
    assert len(peaks) >= 3

    matched = bayesian_match_peak_candidates(peaks)
    assert all(peak.nuclide for peak in matched[:3])

    survey_points = extract_survey_points(
        (
            ("foreground", foreground),
            ("background", background),
            ("overlay", overlay),
        )
    )
    assert len(survey_points) == 3

    subtracted = subtract_background_counts(
        foreground,
        background,
        mode="statistical",
        scale=1.0,
    )
    assert subtracted.shape == foreground.counts.shape
    assert float(np.sum(subtracted)) < float(np.sum(foreground.counts))

    cascade = compute_cascade_sum_lines(("Co60",))
    assert any(value == pytest.approx(2505.72, abs=1.0) for value in cascade)


def test_efficiency_models_and_activity_results_are_available():
    fits = {
        model_key: fit_efficiency_model(_make_efficiency_points(), model_key=model_key)
        for model_key in (
            "log_poly_2",
            "log_poly_3",
            "gray_functional",
            "semi_empirical_hpge",
        )
    }

    for fit in fits.values():
        assert fit.points_used == 5
        efficiency = np.asarray(fit.curve.efficiency([661.657]), dtype=float).reshape(
            -1
        )[0]
        assert 0.0 < efficiency < 1.0

    matched = bayesian_match_peak_candidates(
        detect_peak_candidates(build_demo_spectrum())
    )
    result = calculate_peak_activity(
        matched[1],
        build_demo_spectrum(),
        efficiency_curve=fits["log_poly_2"].curve,
        gamma_intensity=1.0,
        half_life_s=5.27 * 365.25 * 24.0 * 3600.0,
        source_age_s=12.0 * 3600.0,
    )
    assert result.activity_bq > 0.0
    assert result.mda_bq > 0.0
    assert "Bateman correction" in result.chain_summary

    fits["log_poly_2"].curve.uncertainty_model = {
        "type": "constant",
        "value": 0.25,
    }
    high_efficiency_uncertainty = calculate_peak_activity(
        matched[1],
        build_demo_spectrum(),
        efficiency_curve=fits["log_poly_2"].curve,
        gamma_intensity=1.0,
        half_life_s=5.27 * 365.25 * 24.0 * 3600.0,
        source_age_s=12.0 * 3600.0,
    )
    assert high_efficiency_uncertainty.uncertainty_bq > result.uncertainty_bq


def test_line_match_browser_and_gamma_phenomena_estimates_are_available():
    controller = NuclideSearchController(SelectionBus())
    hits = controller.line_matches_for_energy(661.657, tolerance_keV=2.0, query="cs")

    assert hits
    assert any("cs" in hit.nuclide.lower() for hit in hits)
    assert all(abs(hit.delta_keV) <= 2.0 for hit in hits)

    phenomena = estimate_spectral_phenomena(1332.492)
    kinds = {item.kind for item in phenomena}
    assert {
        "compton_edge",
        "backscatter",
        "single_escape",
        "double_escape",
        "annihilation",
    } <= kinds
    assert all(item.energy_keV > 0.0 for item in phenomena)


def test_analysis_workspace_tracks_loaded_spectra_and_role_assignments():
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(
                SpectrumSlot(
                    key="foreground",
                    label="Foreground",
                    spectrum=build_demo_spectrum(),
                ),
                SpectrumSlot(
                    key="background",
                    label="Background",
                    spectrum=build_demo_background_spectrum(),
                ),
            ),
            active_spectrum_key="foreground",
        )
    )
    controller.set_detector_efficiency(
        EfficiencyCalibration(detector_id="South", source_distance_cm=25.0)
    )
    assert controller.state.detector_efficiency.detector_id == "South"
    assert controller.describe()["detector_id"] == "South"

    sample_key = controller.register_loaded_spectrum(
        build_demo_spectrum(),
        label="sample.spe",
        source_path="/tmp/sample.spe",
    )
    background_key = controller.register_loaded_spectrum(
        build_demo_background_spectrum(),
        label="sample.spe",
        source_path="/tmp/background.spe",
    )
    controller.assign_loaded_spectrum_to_slot(sample_key, "foreground")
    controller.assign_loaded_spectrum_to_slot(background_key, "background")

    assert sample_key == "sample-spe"
    assert background_key == "sample-spe-2"
    assert controller.slot("foreground").source_label == "sample.spe"
    assert controller.slot("background").source_path == "/tmp/background.spe"
    assert controller.describe()["loaded_spectrum_count"] == 2
    assert controller.describe()["slot_sources"]["background"] == "sample.spe"
    assert controller.describe()["bayesian_source_id"] == "fluxforge_bundled_gamma"
    assert controller.describe()["ml_source_id"] == "fluxforge_bundled_gamma"


def _write_spectrum_csv(path: Path, spectrum) -> None:
    lines = ["channel,counts"]
    for channel, count in zip(
        np.asarray(spectrum.channels, dtype=float),
        np.asarray(spectrum.counts, dtype=float),
    ):
        lines.append(f"{int(round(float(channel)))},{float(count):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_peak_workflow_supports_undo_pin_tag_and_selection_sync(
    monkeypatch,
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    peak_panel = window.bottom_dock.widget().peak_table_panel
    QTest.mouseClick(peak_panel.auto_find_button, Qt.LeftButton)
    _qapp().processEvents()
    assert peak_panel.table.rowCount() == len(peaks)
    assert window.analysis_workspace.describe()["peak_count"] == len(peaks)

    QTest.mouseClick(peak_panel.match_button, Qt.LeftButton)
    _qapp().processEvents()

    co60_row = 0
    for row in range(peak_panel.table.rowCount()):
        if "co60" in peak_panel.table.item(row, 4).text().lower():
            co60_row = row
            break
    peak_panel.table.setCurrentCell(co60_row, 0)
    peak_panel.table.selectRow(co60_row)
    peak_panel._publish_selected_peak()
    _qapp().processEvents()

    selected_peak = window.analysis_workspace.selected_peak()
    assert selected_peak is not None
    assert window.selection_bus.state.peak_energy_keV == pytest.approx(
        selected_peak.energy_keV,
        abs=1.0,
    )

    QTest.mouseClick(peak_panel.pin_button, Qt.LeftButton)
    _qapp().processEvents()
    assert "Co60" in window.analysis_workspace.state.pinned_nuclides
    assert len(window.analysis_workspace.state.cascade_sum_lines_keV) >= 1

    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("qa-check", True)
    )
    QTest.mouseClick(peak_panel.tag_button, Qt.LeftButton)
    _qapp().processEvents()
    assert "qa-check" in peak_panel.table.item(co60_row, 5).text()

    window.undo_stack.undo()
    _qapp().processEvents()
    assert "qa-check" not in peak_panel.table.item(co60_row, 5).text()

    window.undo_stack.redo()
    _qapp().processEvents()
    assert "qa-check" in peak_panel.table.item(co60_row, 5).text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_table_follows_selection_bus_peak_energy_updates():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    window.analysis_workspace.replace_peaks(peaks)
    _qapp().processEvents()

    target_row = min(
        range(len(peaks)),
        key=lambda idx: abs(float(peaks[idx].energy_keV) - 661.657),
    )
    window.selection_bus.publish_peak(float(peaks[target_row].energy_keV))
    _qapp().processEvents()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    assert peak_panel.table.currentRow() == target_row
    selected = window.analysis_workspace.selected_peak()
    assert selected is not None
    assert selected.peak_id == peaks[target_row].peak_id
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_panel_exposes_full_search_surface_and_method_specific_databases(
    monkeypatch,
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    search_keys = {
        peak_panel.peak_search_selector.combo.itemData(index)
        for index in range(peak_panel.peak_search_selector.combo.count())
    }
    assert {
        "mariscotti",
        "simple",
        "window",
        "chunked",
        "scipy",
        "segmented",
        "derivative",
        "second_difference",
        "direct_scipy",
        "wavelet",
        "relative_extrema",
        "consensus",
        "nasa_peaksearch",
    }.issubset(search_keys)

    assert peak_panel.bayesian_source_combo.count() >= 4
    assert peak_panel.ml_source_combo.count() >= 4

    peak_panel.bayesian_source_combo.setCurrentIndex(
        max(peak_panel.bayesian_source_combo.findData("nndc_offline_activation"), 0)
    )
    peak_panel.ml_source_combo.setCurrentIndex(
        max(peak_panel.ml_source_combo.findData("fluxforge_bundled_gamma"), 0)
    )
    peak_panel.peak_search_selector.set_current_key("window")
    _qapp().processEvents()

    assert window.analysis_workspace.state.bayesian_source_id == str(
        peak_panel.bayesian_source_combo.currentData()
    )
    assert window.analysis_workspace.state.ml_source_id == str(
        peak_panel.ml_source_combo.currentData()
    )

    peaks = detect_peak_candidates(build_demo_spectrum(), method="window")
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)
    QTest.mouseClick(peak_panel.auto_find_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window.analysis_workspace.state.peak_search_method == "window"
    assert peak_panel.table.rowCount() == len(peaks)
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_builds_and_exports_timeseries(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    panel.observable_combo.setCurrentIndex(
        max(panel.observable_combo.findData("activity"), 0)
    )
    panel.nuclide_focus_combo.setCurrentIndex(
        max(panel.nuclide_focus_combo.findData("Mo-99"), 0)
    )
    result = panel.build_inventory_timeline()
    assert result is not None
    assert "Mo-99" in result.activity_series
    assert "Tc-99m" in result.activity_series

    csv_path = tmp_path / "inventory_timeseries.csv"
    plot_path = tmp_path / "inventory_activity.png"
    panel.export_time_series_csv(csv_path)
    panel.export_plot(plot_path)

    assert csv_path.exists()
    assert plot_path.exists()
    assert "dose_rate_uSv_h" in csv_path.read_text(encoding="utf-8")
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_difom_preview_reports_score():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    score = panel.preview_difom_score()

    assert score is not None
    assert score > 0.0
    assert "DI-FOM preview score" in panel.difom_summary.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_fim_preview_reports_score_and_diagnostics():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    panel.fim_objective_combo.setCurrentIndex(
        max(panel.fim_objective_combo.findData("fim-d"), 0)
    )
    score = panel.preview_fim_score()

    assert score is not None
    assert score == pytest.approx(score)
    assert "FIM preview (fim-d) score" in panel.fim_summary.text()
    assert "condition number" in panel.fim_summary.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_mwdcs_preview_reports_score_and_window_count():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    panel.mwdcs_window_count_spin.setValue(3)
    panel.mwdcs_full_spectrum_checkbox.setChecked(True)
    score = panel.preview_mwdcs_score()

    assert score is not None
    assert score > 0.0
    assert "MWDCS preview score" in panel.mwdcs_summary.text()
    assert "3 window(s)" in panel.mwdcs_summary.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_bassd_preview_requires_advanced_guard_and_reports_score():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    disabled_score = panel.preview_bassd_score()
    assert disabled_score is None
    assert "Enable advanced objectives" in panel.bassd_summary.text()

    panel.advanced_objective_checkbox.setChecked(True)
    score = panel.preview_bassd_score()
    assert score is not None
    assert score == pytest.approx(score)
    assert "BASS-D preview utility" in panel.bassd_summary.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_inventory_timeline_panel_stbdmr_preview_requires_guard_and_reports_diagnostics():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    disabled_score = panel.preview_stbdmr_score()
    assert disabled_score is None
    assert "Enable advanced objectives" in panel.stbdmr_summary.text()

    panel.advanced_objective_checkbox.setChecked(True)
    panel.stbdmr_differentiable_checkbox.setChecked(True)
    score = panel.preview_stbdmr_score()

    assert score is not None
    assert score == pytest.approx(score)
    assert "STBD-MR preview score" in panel.stbdmr_summary.text()
    assert "graph density" in panel.stbdmr_summary.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_masking_review_panel_runs_and_exports_tables(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    _seed_phase6_real_workspace(window)
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().masking_review_panel
    panel.energy_window_spin.setValue(10.0)
    QTest.mouseClick(panel.refresh_button, Qt.LeftButton)
    _qapp().processEvents()
    result = panel._last_rows

    assert result is not None
    assert panel.line_table.rowCount() > 0
    assert panel.isotope_table.rowCount() > 0
    assert (
        "alternate-line" in panel.summary.text().lower()
        or "recommendation" in panel.summary.text().lower()
    )

    lines_csv = tmp_path / "masking_lines.csv"
    isotopes_csv = tmp_path / "masking_isotopes.csv"
    panel.export_lines_csv(lines_csv)
    panel.export_isotopes_csv(isotopes_csv)
    assert lines_csv.exists()
    assert isotopes_csv.exists()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_optimization_workspace_panel_runs_and_exports_phase6_bundle(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    grids = _seed_phase6_real_workspace(window)
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().optimization_workspace_panel
    panel.irradiation_grid_edit.setText(grids["irradiation_grid_s"])
    panel.cooldown_grid_edit.setText(grids["cooldown_grid_s"])
    panel.count_grid_edit.setText(grids["count_grid_s"])
    panel.objective_combo.setCurrentIndex(
        max(panel.objective_combo.findData("di-fom"), 0)
    )
    QTest.mouseClick(panel.run_button, Qt.LeftButton)
    _qapp().processEvents()
    payload = panel._last_output_payload

    assert payload is not None
    assert panel.heatmap_table.rowCount() > 0
    assert "Objective: di-fom" in panel.recommendation_browser.toPlainText()

    grid_csv = tmp_path / "optimization_grid.csv"
    ffexp_path = tmp_path / "phase6_bundle.ffexp"
    panel.export_grid_csv(grid_csv)
    panel.export_ffexp(ffexp_path)

    assert grid_csv.exists()
    assert ffexp_path.exists()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_optimization_workspace_panel_advanced_guard_and_second_irradiation_panel(
    tmp_path,
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    grids = _seed_phase6_real_workspace(window)
    window.show()
    _qapp().processEvents()

    optimizer = window.bottom_dock.widget().optimization_workspace_panel
    optimizer.irradiation_grid_edit.setText(grids["irradiation_grid_s"])
    optimizer.cooldown_grid_edit.setText(grids["cooldown_grid_s"])
    optimizer.count_grid_edit.setText(grids["count_grid_s"])
    optimizer.objective_combo.setCurrentIndex(
        max(optimizer.objective_combo.findData("bass-d"), 0)
    )
    QTest.mouseClick(optimizer.run_button, Qt.LeftButton)
    _qapp().processEvents()
    disabled = optimizer._last_output_payload
    assert disabled is None
    assert "advanced objectives" in optimizer.summary.text().lower()

    optimizer.advanced_checkbox.setChecked(True)
    QTest.mouseClick(optimizer.run_button, Qt.LeftButton)
    _qapp().processEvents()
    enabled = optimizer._last_output_payload
    assert enabled is not None
    assert optimizer.heatmap_table.rowCount() > 0

    panel = window.bottom_dock.widget().second_irradiation_panel
    panel.flux_scales_edit.setText("1.0")
    panel.duration_factors_edit.setText("1.0")
    panel.cooling_grid_edit.setText(grids["cooldown_grid_s"])
    panel.target_weights_edit.setText(
        load_phase6_real_target_weights_text(DEFAULT_PHASE6_SAMPLE_ID)
    )
    QTest.mouseClick(panel.run_button, Qt.LeftButton)
    _qapp().processEvents()
    payload = panel._last_payload
    assert payload is not None
    assert panel.table.rowCount() > 0
    assert "Selected label" in panel.browser.toPlainText()

    selected_csv = tmp_path / "second_irradiation_selected.csv"
    panel.export_selected_csv(selected_csv)
    assert selected_csv.exists()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_optimization_workspace_panel_runs_ldrd_worked_example_action(
    monkeypatch, tmp_path
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().optimization_workspace_panel
    output_root = tmp_path / "phase6_gui_worked_example"

    def fake_run_phase6_ldrd_worked_example(*, sample_id, output_root):
        assert sample_id == DEFAULT_PHASE6_SAMPLE_ID
        target = Path(output_root)
        target.mkdir(parents=True, exist_ok=True)
        summary = target / "WORKED_EXAMPLE_SUMMARY.md"
        summary.write_text("# probe\n", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "fluxforge.gui.panels.phase6.run_phase6_ldrd_worked_example",
        fake_run_phase6_ldrd_worked_example,
    )

    panel.ldrd_sample_id_edit.setText(DEFAULT_PHASE6_SAMPLE_ID)
    panel.ldrd_output_root_edit.setText(str(output_root))
    QTest.mouseClick(panel.ldrd_worked_example_button, Qt.LeftButton)
    _qapp().processEvents()

    summary = output_root / "WORKED_EXAMPLE_SUMMARY.md"
    assert summary.exists()
    assert panel._last_worked_example_summary_path == summary
    assert "worked example completed" in panel.summary.text().lower()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_astm_mode_locks_peak_identification_databases_to_standard_sources():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(
            initial_state=ModeState(
                mode=GUIMode.STANDARDS,
                standard="ASTM E181",
                theme="dark",
            )
        ),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    sidebar = window.left_dock.widget()

    assert peak_panel.bayesian_source_combo.count() == 1
    assert peak_panel.bayesian_source_combo.currentData() == "decay_2012"
    assert peak_panel.bayesian_source_combo.isEnabled() is False
    assert peak_panel.ml_source_combo.count() == 1
    assert peak_panel.ml_source_combo.currentData() == "decay_2012"
    assert peak_panel.ml_source_combo.isEnabled() is False

    assert sidebar.gamma_source_combo.count() == 1
    assert sidebar.gamma_source_combo.currentData() == "decay_2012"
    assert sidebar.gamma_source_combo.isEnabled() is False
    assert sidebar.calibration_source_combo.count() == 1
    assert (
        sidebar.calibration_source_combo.currentData() == "calibration_standard_sources"
    )
    assert sidebar.calibration_source_combo.isEnabled() is False
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_astm_mode_peak_matching_workflows_use_locked_gamma_source(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(
            initial_state=ModeState(
                mode=GUIMode.STANDARDS,
                standard="ASTM E181",
                theme="dark",
            )
        ),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    window.library_manager.set_gamma_identification_source("nndc_offline_activation")
    peaks = detect_peak_candidates(build_demo_spectrum())
    window.analysis_workspace.replace_peaks(peaks)
    window.analysis_workspace.select_peak(peaks[0].peak_id)
    _qapp().processEvents()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    captured: dict[str, str] = {}

    def _fake_bayesian_match(
        peaks_arg, *, source_id="fluxforge_bundled_gamma", custom_path=None
    ):
        del custom_path
        captured["bayesian_source_id"] = str(source_id)
        return tuple(peaks_arg)

    monkeypatch.setattr(
        "fluxforge.gui.panels.modern_shell.bayesian_match_peak_candidates",
        _fake_bayesian_match,
    )

    engine = window.registries.nuclide_id_engines.get("ml_peak_onnx")

    def _fake_analyze_peaks(
        peaks_arg,
        *,
        source_id="fluxforge_bundled_gamma",
        custom_path=None,
        prefer_gpu=False,
    ):
        del peaks_arg, custom_path, prefer_gpu
        captured["ml_source_id"] = str(source_id)
        return ()

    monkeypatch.setattr(engine, "analyze_peaks", _fake_analyze_peaks)

    peak_panel.run_bayesian_match()
    peak_panel.run_ml_peak_analysis()
    _qapp().processEvents()

    assert captured["bayesian_source_id"] == "decay_2012"
    assert captured["ml_source_id"] == "decay_2012"
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_astm_mode_reference_overlays_use_locked_gamma_source(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(
            initial_state=ModeState(
                mode=GUIMode.STANDARDS,
                standard="ASTM E181",
                theme="dark",
            )
        ),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    window.library_manager.set_gamma_identification_source("nndc_offline_activation")
    peaks = detect_peak_candidates(build_demo_spectrum())
    assigned = bayesian_match_peak_candidates(peaks)
    window.analysis_workspace.replace_peaks(assigned)
    window.analysis_workspace.select_peak(assigned[0].peak_id)
    _qapp().processEvents()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    captured: dict[str, str] = {}

    def _fake_cascade_lines(
        pinned, *, source_id="fluxforge_bundled_gamma", custom_path=None
    ):
        del pinned, custom_path
        captured["source_id"] = str(source_id)
        return (2505.72,)

    monkeypatch.setattr(
        "fluxforge.gui.panels.modern_shell.compute_cascade_sum_lines",
        _fake_cascade_lines,
    )

    peak_panel._pin_selected_nuclide()
    _qapp().processEvents()

    assert captured["source_id"] == "decay_2012"
    window.close()


def _prepare_reassignable_peak_assignment(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    peak_panel = window.bottom_dock.widget().peak_table_panel
    peak_panel.run_auto_peak_search()
    peak_panel.run_bayesian_match()
    peak_panel.peak_id_filter.setText("")
    _qapp().processEvents()

    for row in range(peak_panel.table.rowCount()):
        peak_panel.table.setCurrentCell(row, 0)
        peak_panel.table.selectRow(row)
        peak_panel._publish_selected_peak()
        _qapp().processEvents()

        original_peak = window.analysis_workspace.selected_peak()
        if original_peak is None:
            continue

        primary_nuclide = original_peak.nuclide or "Cs-137"
        alternate_nuclide = "Co-60" if primary_nuclide.lower() != "co-60" else "Ba-133"
        alternate_energy = float(original_peak.energy_keV) + 0.5

        def _fake_line_matches_for_energy(
            energy_keV: float,
            *,
            tolerance_keV: float = 2.0,
            query: str = "",
            limit: int = 48,
            min_intensity: float = 0.0,
        ):
            del tolerance_keV, min_intensity
            token = "".join(
                character for character in query.lower() if character.isalnum()
            )
            matches = [
                GammaLineMatchResult(
                    nuclide=primary_nuclide,
                    display_name=primary_nuclide,
                    line_energy_keV=float(original_peak.energy_keV),
                    delta_keV=round(
                        float(original_peak.energy_keV) - float(energy_keV), 3
                    ),
                    intensity=0.9,
                    half_life_s=1.0,
                ),
                GammaLineMatchResult(
                    nuclide=alternate_nuclide,
                    display_name=alternate_nuclide,
                    line_energy_keV=alternate_energy,
                    delta_keV=round(alternate_energy - float(energy_keV), 3),
                    intensity=0.7,
                    half_life_s=1.0,
                ),
            ]
            if token:
                matches = [
                    match
                    for match in matches
                    if token
                    in "".join(
                        character
                        for character in f"{match.nuclide} {match.display_name}".lower()
                        if character.isalnum()
                    )
                ]
            return matches[:limit]

        def _fake_reference_lines_for_nuclide(nuclide: str, *, limit=None):
            anchor = (
                float(original_peak.energy_keV)
                if nuclide == primary_nuclide
                else alternate_energy
            )
            lines = (round(anchor, 3), round(anchor + 31.0, 3), round(anchor + 63.0, 3))
            if limit is None:
                return lines
            return lines[:limit]

        monkeypatch.setattr(
            peak_panel.nuclide_controller,
            "line_matches_for_energy",
            _fake_line_matches_for_energy,
        )
        monkeypatch.setattr(
            peak_panel.nuclide_controller,
            "reference_lines_for_nuclide",
            _fake_reference_lines_for_nuclide,
        )
        peak_panel.peak_id_tolerance.setValue(2.0)
        peak_panel._refresh_peak_id_matches()
        _qapp().processEvents()

        alternate_row = next(
            (
                index
                for index, match in enumerate(peak_panel._current_match_results)
                if match.nuclide != primary_nuclide
            ),
            None,
        )
        if alternate_row is not None:
            peak_panel.peak_id_matches.setCurrentRow(alternate_row)
            peak_panel._match_selection_changed()
            _qapp().processEvents()
            return (
                window,
                peak_panel,
                original_peak,
                peak_panel._current_match_results[alternate_row],
            )

    window.close()
    pytest.fail(
        "Expected at least one reassignable peak in the demo analysis workspace."
    )


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_id_browser_supports_manual_isotope_reassignment(monkeypatch):
    window, peak_panel, original_peak, reassigned = (
        _prepare_reassignable_peak_assignment(monkeypatch)
    )

    assert peak_panel.peak_id_energy.value() == pytest.approx(
        original_peak.energy_keV,
        abs=1.0,
    )
    assert peak_panel.peak_id_matches.count() >= 1

    peak_panel._assign_selected_isotope()
    _qapp().processEvents()

    updated_peak = window.analysis_workspace.selected_peak()
    assert updated_peak is not None
    assert updated_peak.status == "manual"
    assert updated_peak.nuclide == reassigned.nuclide
    assert (
        updated_peak.nuclide != original_peak.nuclide
        or peak_panel.peak_id_tolerance.value() > 2.0
    )
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_id_browser_supports_clearing_manual_assignment(monkeypatch):
    window, peak_panel, _original_peak, _reassigned = (
        _prepare_reassignable_peak_assignment(monkeypatch)
    )

    peak_panel._assign_selected_isotope()
    _qapp().processEvents()
    assert window.analysis_workspace.selected_peak() is not None
    assert window.analysis_workspace.selected_peak().nuclide is not None

    peak_panel._clear_selected_peak_assignment()
    _qapp().processEvents()
    cleared_peak = window.analysis_workspace.selected_peak()
    assert cleared_peak is not None
    assert cleared_peak.nuclide is None
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_id_browser_supports_typed_centroid_filtering_and_phenomena_guides():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    assert sidebar.gamma_source_combo.count() >= 4
    assert sidebar.gamma_source_combo.currentData() == (
        window.library_manager.state.gamma_identification_source_id
    )
    assert "Identification" in sidebar.library_summary.toPlainText()

    peak_panel = window.bottom_dock.widget().peak_table_panel
    peak_panel.peak_id_energy.setValue(661.657)
    peak_panel.peak_id_tolerance.setValue(2.0)
    peak_panel.peak_id_filter.setText("cs")
    _qapp().processEvents()

    assert peak_panel.peak_id_matches.count() >= 1
    assert peak_panel._current_match_results
    assert all(
        "cs" in f"{match.nuclide} {match.display_name}".lower()
        for match in peak_panel._current_match_results
    )
    first_match = peak_panel._current_match_results[0]
    assert abs(first_match.line_energy_keV - 661.657) <= 2.0

    peak_panel.peak_id_matches.setCurrentRow(0)
    _qapp().processEvents()

    phenomena = [
        peak_panel.peak_id_phenomena.item(index).text().lower()
        for index in range(peak_panel.peak_id_phenomena.count())
    ]
    assert any("compton edge" in text for text in phenomena)
    assert any("backscatter" in text for text in phenomena)
    assert len(window.selection_bus.state.annotation_lines) >= 3
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_peak_id_browser_use_selected_peak_button_restores_peak_centroid(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    peak_panel = window.bottom_dock.widget().peak_table_panel
    QTest.mouseClick(peak_panel.auto_find_button, Qt.LeftButton)
    _qapp().processEvents()

    peak_panel.table.selectRow(0)
    _qapp().processEvents()

    selected_peak = window.analysis_workspace.selected_peak()
    assert selected_peak is not None
    peak_panel.peak_id_energy.setValue(400.0)
    _qapp().processEvents()

    QTest.mouseClick(peak_panel.use_selected_peak_button, Qt.LeftButton)
    _qapp().processEvents()

    assert peak_panel.peak_id_energy.value() == pytest.approx(
        selected_peak.energy_keV,
        abs=1.0,
    )
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_sidebar_nuclide_workbench_supports_saved_lists_mixtures_and_details():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    offline_index = sidebar.gamma_source_combo.findData("nndc_offline_activation")
    if offline_index >= 0:
        sidebar.gamma_source_combo.setCurrentIndex(offline_index)
        _qapp().processEvents()

    sidebar.nuclide_query.setText("co")
    _qapp().processEvents()
    assert sidebar.nuclides.count() >= 1
    sidebar.nuclides.setCurrentRow(0)
    _qapp().processEvents()

    assert "Specific activity" in sidebar.nuclide_details_browser.toPlainText()
    assert sidebar.nuclide_line_table.rowCount() >= 1

    QTest.mouseClick(sidebar.save_selected_nuclide_button, Qt.LeftButton)
    _qapp().processEvents()
    assert sidebar.saved_nuclides.count() == 1
    assert "User Define List" in sidebar.saved_nuclide_summary.toPlainText()

    QTest.mouseClick(sidebar.apply_saved_overlay_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window.selection_bus.state.nuclide is not None
    assert window.selection_bus.state.nuclide.startswith("Saved:")
    assert len(window.selection_bus.state.annotation_lines) >= 1

    QTest.mouseClick(sidebar.add_selected_mixture_button, Qt.LeftButton)
    _qapp().processEvents()

    sidebar.nuclide_query.setText("cs")
    _qapp().processEvents()
    assert sidebar.nuclides.count() >= 1
    sidebar.nuclides.setCurrentRow(0)
    _qapp().processEvents()
    QTest.mouseClick(sidebar.add_selected_mixture_button, Qt.LeftButton)
    _qapp().processEvents()

    assert sidebar.mixture_table.rowCount() >= 2
    sidebar.mixture_table.item(0, 1).setText("2.0")
    sidebar.mixture_table.item(1, 1).setText("1.0")
    _qapp().processEvents()

    QTest.mouseClick(sidebar.normalize_mixture_button, Qt.LeftButton)
    _qapp().processEvents()
    weights = [
        float(sidebar.mixture_table.item(row, 1).text())
        for row in range(sidebar.mixture_table.rowCount())
    ]
    assert sum(weights) == pytest.approx(1.0, abs=1.0e-6)
    assert "Mixture" in sidebar.mixture_summary.toPlainText()

    QTest.mouseClick(sidebar.apply_mixture_overlay_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window.selection_bus.state.nuclide is not None
    assert window.selection_bus.state.nuclide.startswith("Mixture:")
    assert len(window.selection_bus.state.annotation_lines) >= 2
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_exposes_log_scale_and_peak_label_toggles():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    sidebar.nuclides.setCurrentRow(0)
    _qapp().processEvents()
    QTest.mouseClick(sidebar.save_selected_nuclide_button, Qt.LeftButton)
    _qapp().processEvents()
    QTest.mouseClick(sidebar.apply_saved_overlay_button, Qt.LeftButton)
    _qapp().processEvents()

    canvas = window.central_tabs.canvas
    assert canvas._log_scale is False
    assert canvas._peak_labels_visible is True
    assert len(canvas._annotation_label_items) >= 1

    window._log_scale_action.trigger()
    _qapp().processEvents()
    assert canvas._log_scale is True

    window._peak_labels_action.trigger()
    _qapp().processEvents()
    assert canvas._peak_labels_visible is False
    assert len(canvas._annotation_label_items) == 0

    window._peak_labels_action.trigger()
    _qapp().processEvents()
    assert canvas._peak_labels_visible is True
    assert len(canvas._annotation_label_items) >= 1
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_activity_background_and_survey_map_workflows(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    bottom = window.bottom_dock.widget()
    bottom.peak_table_panel.run_auto_peak_search()
    bottom.peak_table_panel.run_bayesian_match()
    _qapp().processEvents()

    co60_row = 0
    for row in range(bottom.peak_table_panel.table.rowCount()):
        if "co60" in bottom.peak_table_panel.table.item(row, 4).text().lower():
            co60_row = row
            break
    bottom.peak_table_panel.table.selectRow(co60_row)
    _qapp().processEvents()

    fit = fit_efficiency_model(_make_efficiency_points(), model_key="log_poly_2")
    window.analysis_workspace.set_efficiency_fit(fit)
    activity_panel = bottom.activity_results_panel
    activity_panel.background_mode_combo.setCurrentIndex(
        activity_panel.background_mode_combo.findData("statistical")
    )
    activity_panel.source_age_hours.setValue(24.0)
    QTest.mouseClick(activity_panel.compute_activity_button, Qt.LeftButton)
    _qapp().processEvents()

    assert len(window.analysis_workspace.state.activity_results) == 1
    assert "Bateman correction" in activity_panel.results.toPlainText()
    assert window.analysis_workspace.state.background_mode == "statistical"

    survey_text = bottom.survey_map_panel.browser.toPlainText()
    assert "demo_hpge_workspace" in survey_text
    assert "43.0731" in survey_text

    spectrum_tabs = window.central_tabs.spectrum_slot_tabs
    assert spectrum_tabs.count() == 3
    spectrum_tabs.setCurrentIndex(1)
    _qapp().processEvents()
    assert window.analysis_workspace.state.active_spectrum_key == "background"
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_activity_review_exports_csv_and_plots(monkeypatch, tmp_path):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    bottom = window.bottom_dock.widget()
    bottom.peak_table_panel.run_auto_peak_search()
    bottom.peak_table_panel.run_bayesian_match()
    _qapp().processEvents()

    fit = fit_efficiency_model(_make_efficiency_points(), model_key="log_poly_2")
    window.analysis_workspace.set_efficiency_fit(fit)

    activity_panel = bottom.activity_results_panel
    activity_panel.source_age_hours.setValue(24.0)
    review = activity_panel.analyze_spectrum_activities()
    assert review is not None
    assert len(review.isotope_summaries) >= 1

    csv_path = tmp_path / "activity_review.csv"
    decay_path = tmp_path / "activity_decay.png"
    bateman_path = tmp_path / "activity_bateman.png"
    activity_panel.export_activity_csv(csv_path)
    activity_panel.export_decay_plot(decay_path)
    activity_panel.export_bateman_plot(bateman_path)

    assert csv_path.exists()
    assert decay_path.exists()
    assert bateman_path.exists()
    assert "irradiation_time_activity_Bq" in csv_path.read_text(encoding="utf-8")
    assert "Irradiation-time activity" in activity_panel.results.toPlainText()
    assert len(window.analysis_workspace.state.activity_results) >= 1
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_activity_unit_selector_rescales_activity_review_plot(
    monkeypatch, tmp_path
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    bottom = window.bottom_dock.widget()
    bottom.peak_table_panel.run_auto_peak_search()
    bottom.peak_table_panel.run_bayesian_match()
    _qapp().processEvents()

    fit = fit_efficiency_model(_make_efficiency_points(), model_key="log_poly_2")
    window.analysis_workspace.set_efficiency_fit(fit)

    activity_panel = bottom.activity_results_panel
    review = activity_panel.analyze_spectrum_activities()
    assert review is not None

    activity_panel.activity_unit_combo.setCurrentIndex(
        max(activity_panel.activity_unit_combo.findData("kBq"), 0)
    )
    _qapp().processEvents()

    captured = {}

    def fake_plot_decay_curves(data, *, ylabel=None, save_path=None, **kwargs):
        captured["data"] = data
        captured["ylabel"] = ylabel
        Path(save_path).write_text("plot", encoding="utf-8")
        return object(), object()

    monkeypatch.setattr(
        "fluxforge.gui.panels.modern_shell.plot_decay_curves",
        fake_plot_decay_curves,
    )

    plot_path = tmp_path / "activity_decay_kbq.png"
    activity_panel.export_decay_plot(plot_path)

    assert plot_path.exists()
    assert captured["ylabel"] == "Activity (kBq)"
    raw_first_value = next(iter(review.decay_plot_data.values()))[0][1]
    scaled_first_value = next(iter(captured["data"].values()))[0][1]
    assert scaled_first_value == pytest.approx(raw_first_value / 1000.0)
    assert "kBq" in activity_panel.results.toPlainText()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_inventory_panel_activity_unit_selector_updates_headers_and_values():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.set_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="Mo-99 feed",
                age_corrected_uncertainty_bq=45.0,
            ),
        )
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().inventory_timeline_panel
    panel.observable_combo.setCurrentIndex(
        max(panel.observable_combo.findData("activity"), 0)
    )
    panel.nuclide_focus_combo.setCurrentIndex(
        max(panel.nuclide_focus_combo.findData("Mo-99"), 0)
    )
    result = panel.build_inventory_timeline()
    assert result is not None

    panel.activity_unit_combo.setCurrentIndex(
        max(panel.activity_unit_combo.findData("kBq"), 0)
    )
    panel._render_result(result)
    _qapp().processEvents()

    assert panel.table.horizontalHeaderItem(2).text() == "Activity (kBq)"
    assert panel.table.horizontalHeaderItem(6).text() == "Act. Sigma (kBq)"
    assert float(panel.table.item(0, 2).text()) < 1.0
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_sidebar_registers_and_removes_user_library(monkeypatch, tmp_path):
    registry_path = tmp_path / "library_registry.json"
    gamma_path = tmp_path / "gamma.csv"
    gamma_path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\nCo60,1332.5,1.0,166344192.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry_path))

    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    sidebar.gamma_source_combo.setCurrentIndex(
        max(sidebar.gamma_source_combo.findData("custom_gamma_file"), 0)
    )
    sidebar.custom_gamma_path.setText(str(gamma_path))
    sidebar.custom_gamma_alias.setText("Lab Ref")
    QTest.mouseClick(sidebar.register_custom_gamma_button, Qt.LeftButton)
    _qapp().processEvents()

    assert sidebar.gamma_source_combo.findData("user_gamma_lab_ref") >= 0
    assert (
        window.library_manager.state.gamma_identification_source_id
        == "user_gamma_lab_ref"
    )

    QTest.mouseClick(sidebar.remove_registered_gamma_button, Qt.LeftButton)
    _qapp().processEvents()

    assert (
        window.library_manager.state.gamma_identification_source_id
        == "fluxforge_bundled_gamma"
    )
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_background_selector_updates_subtracted_foreground_and_overlay(
    tmp_path,
):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    foreground_path = tmp_path / "sample.csv"
    background_path = tmp_path / "background.csv"
    overlay_path = tmp_path / "overlay.csv"
    _write_spectrum_csv(foreground_path, build_demo_spectrum())
    _write_spectrum_csv(background_path, build_demo_background_spectrum())
    _write_spectrum_csv(overlay_path, build_demo_overlay_spectrum())

    window.open_path(foreground_path)
    window.open_path(background_path)
    window.open_path(overlay_path)
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    foreground_index = sidebar.foreground_spectrum_combo.findData("sample-csv")
    background_index = sidebar.background_spectrum_combo.findData("background-csv")
    overlay_index = sidebar.overlay_spectrum_combo.findData("overlay-csv")

    assert foreground_index >= 0
    assert background_index >= 0
    assert overlay_index >= 0

    sidebar.foreground_spectrum_combo.setCurrentIndex(foreground_index)
    sidebar.background_spectrum_combo.setCurrentIndex(background_index)
    sidebar.overlay_spectrum_combo.setCurrentIndex(overlay_index)
    _qapp().processEvents()

    state = window.analysis_workspace.state
    assert state.loaded_spectra[-3].label == "sample.csv"
    assert state.loaded_spectra[-2].label == "background.csv"
    assert state.loaded_spectra[-1].label == "overlay.csv"
    assert window.analysis_workspace.slot("foreground").source_label == "sample.csv"
    assert window.analysis_workspace.slot("background").source_label == "background.csv"
    assert window.analysis_workspace.slot("overlay").source_label == "overlay.csv"

    canvas = window.central_tabs.canvas
    expected = subtract_background_counts(
        window.analysis_workspace.spectrum("foreground"),
        window.analysis_workspace.spectrum("background"),
        mode=state.background_mode,
        scale=state.background_scale,
    )
    assert np.allclose(np.asarray(canvas.buffer.full_resolution, dtype=float), expected)
    assert len(canvas._overlay_traces) == 2
    status = canvas.status_label.text().lower()
    assert "sample.csv" in status
    assert "background.csv" in status
    assert "overlay.csv" in status

    assigned_roles = {
        sidebar.files.topLevelItem(index)
        .text(0): sidebar.files.topLevelItem(index)
        .text(1)
        for index in range(sidebar.files.topLevelItemCount())
    }
    assert "Foreground" in assigned_roles["sample.csv"]
    assert "Background" in assigned_roles["background.csv"]
    assert "Secondary Overlay" in assigned_roles["overlay.csv"]
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_roi_tools_panel_supports_mouse_driven_roi_analysis_and_statistics(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    bottom = window.bottom_dock.widget()
    peak_panel = bottom.peak_table_panel
    QTest.mouseClick(peak_panel.auto_find_button, Qt.LeftButton)
    _qapp().processEvents()

    peak_panel.table.selectRow(0)
    peak_panel._publish_selected_peak()
    _qapp().processEvents()

    roi_panel = bottom.roi_tools_panel
    roi_panel.peak_search_selector.set_current_key("mariscotti")
    roi_panel.background_selector.set_current_key("roi_sideband")
    _qapp().processEvents()

    QTest.mouseClick(roi_panel.use_selected_peak_button, Qt.LeftButton)
    _qapp().processEvents()
    assert roi_panel.roi_right.value() > roi_panel.roi_left.value()

    QTest.mouseClick(roi_panel.analyze_button, Qt.LeftButton)
    _qapp().processEvents()

    roi_result = window.analysis_workspace.state.roi_analysis
    assert roi_result is not None
    assert roi_result.net_counts > 0.0
    assert window.analysis_workspace.state.peak_search_method == "mariscotti"
    assert window.analysis_workspace.state.roi_background_method == "roi_sideband"
    assert "Gross:" in roi_panel.summary.toPlainText()
    assert roi_panel.component_table.rowCount() >= 1

    QTest.mouseClick(roi_panel.statistics_button, Qt.LeftButton)
    _qapp().processEvents()

    stats = window.analysis_workspace.state.roi_statistics
    assert stats is not None
    assert stats.sample_count >= 2
    assert roi_panel.statistics_table.rowCount() == stats.sample_count
    window.close()
