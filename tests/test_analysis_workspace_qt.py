import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.analysis.detector_calibration import EfficiencyPoint  # noqa: E402
from fluxforge.core.analysis_workspace import (  # noqa: E402
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
from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.nuclide_search import NuclideSearchController  # noqa: E402
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
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402

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
        for model_key in ("log_poly_2", "log_poly_3", "gray_functional", "semi_empirical_hpge")
    }

    for fit in fits.values():
        assert fit.points_used == 5
        efficiency = np.asarray(fit.curve.efficiency([661.657]), dtype=float).reshape(-1)[0]
        assert 0.0 < efficiency < 1.0

    matched = bayesian_match_peak_candidates(detect_peak_candidates(build_demo_spectrum()))
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


def test_line_match_browser_and_gamma_phenomena_estimates_are_available():
    controller = NuclideSearchController(SelectionBus())
    hits = controller.line_matches_for_energy(661.657, tolerance_keV=2.0, query="cs")

    assert hits
    assert any("cs" in hit.nuclide.lower() for hit in hits)
    assert all(abs(hit.delta_keV) <= 2.0 for hit in hits)

    phenomena = estimate_spectral_phenomena(1332.492)
    kinds = {item.kind for item in phenomena}
    assert {"compton_edge", "backscatter", "single_escape", "double_escape", "annihilation"} <= kinds
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


def _write_spectrum_csv(path: Path, spectrum) -> None:
    lines = ["channel,counts"]
    for channel, count in zip(np.asarray(spectrum.channels, dtype=float), np.asarray(spectrum.counts, dtype=float)):
        lines.append(f"{int(round(float(channel)))},{float(count):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt analysis workspace dependencies are unavailable.",
)
def test_main_window_peak_workflow_supports_undo_pin_tag_and_selection_sync(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
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

    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("qa-check", True))
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
def test_peak_id_browser_supports_manual_assignment_reassignment_and_guides(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    peaks = detect_peak_candidates(build_demo_spectrum())
    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AutoPeakReviewDialog, "accepted_peaks", lambda self: peaks)

    peak_panel = window.bottom_dock.widget().peak_table_panel
    peak_panel.run_auto_peak_search()
    peak_panel.run_bayesian_match()
    _qapp().processEvents()

    original_peak = None
    alternate_row = None
    peak_panel.peak_id_filter.setText("")
    for row in range(peak_panel.table.rowCount()):
        peak_panel.table.setCurrentCell(row, 0)
        peak_panel.table.selectRow(row)
        peak_panel._publish_selected_peak()
        peak_panel.peak_id_tolerance.setValue(2.0)
        _qapp().processEvents()

        candidate_peak = window.analysis_workspace.selected_peak()
        if candidate_peak is None:
            continue

        candidate_row = next(
            (
                index
                for index, match in enumerate(peak_panel._current_match_results)
                if match.nuclide != candidate_peak.nuclide
            ),
            None,
        )
        if candidate_row is None:
            peak_panel.peak_id_tolerance.setValue(25.0)
            _qapp().processEvents()
            candidate_row = next(
                (
                    index
                    for index, match in enumerate(peak_panel._current_match_results)
                    if match.nuclide != candidate_peak.nuclide
                ),
                None,
            )
        if candidate_row is None:
            peak_panel.peak_id_tolerance.setValue(250.0)
            _qapp().processEvents()
            candidate_row = next(
                (
                    index
                    for index, match in enumerate(peak_panel._current_match_results)
                    if match.nuclide != candidate_peak.nuclide
                ),
                None,
            )
        if candidate_row is not None:
            original_peak = candidate_peak
            alternate_row = candidate_row
            break

    assert original_peak is not None
    assert alternate_row is not None
    assert peak_panel.peak_id_energy.value() == pytest.approx(
        original_peak.energy_keV,
        abs=1.0,
    )
    assert peak_panel.peak_id_matches.count() >= 1

    peak_panel.peak_id_matches.setCurrentRow(alternate_row)
    peak_panel._match_selection_changed()
    _qapp().processEvents()
    reassigned = peak_panel._current_match_results[alternate_row]

    peak_panel._assign_selected_isotope()
    _qapp().processEvents()

    updated_peak = window.analysis_workspace.selected_peak()
    assert updated_peak is not None
    assert updated_peak.status == "manual"
    assert updated_peak.nuclide == reassigned.nuclide
    assert updated_peak.nuclide != original_peak.nuclide or peak_panel.peak_id_tolerance.value() > 2.0

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
def test_main_window_activity_background_and_survey_map_workflows(monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
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
def test_main_window_background_selector_updates_subtracted_foreground_and_overlay(tmp_path):
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
        sidebar.files.topLevelItem(index).text(0): sidebar.files.topLevelItem(index).text(1)
        for index in range(sidebar.files.topLevelItemCount())
    }
    assert "Foreground" in assigned_roles["sample.csv"]
    assert "Background" in assigned_roles["background.csv"]
    assert "Secondary Overlay" in assigned_roles["overlay.csv"]
    window.close()
