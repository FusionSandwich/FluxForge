"""Canonical efficiency-profile, migration, and undo regressions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.analysis_workspace import (
    ActivityCalculationResult,
    EfficiencyCalibrationFitResult,
    PeakCandidate,
    calculate_peak_activity,
    fit_efficiency_model,
)
from fluxforge.core.workspace_document import (
    WorkspaceDocument,
    WorkspaceValidationError,
)
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.gui.analysis_workspace import (
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    SpectrumSlot,
)
from fluxforge.gui.workspace_undo import ApplyDetectorProfileCommand
from fluxforge.io.flux_wire import EfficiencyCalibration
from fluxforge.io.spe import GammaSpectrum


def _controller() -> AnalysisWorkspaceController:
    spectrum = GammaSpectrum(
        counts=[10, 20, 30, 20, 10],
        live_time=100.0,
        spectrum_id="calibration-sample",
        detector_id="South",
        calibration={"energy": [0.0, 0.5]},
    )
    return AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(SpectrumSlot("foreground", "Foreground", spectrum),)
        )
    )


def _fit() -> EfficiencyCalibrationFitResult:
    return EfficiencyCalibrationFitResult(
        model_key="log_poly_2",
        model_label="Log Polynomial (2)",
        curve=EfficiencyCurve.from_polynomial(
            [-4.0, -0.7, 0.02], energy_range=(100.0, 1500.0)
        ),
        residuals=(0.0, 0.01, -0.01),
        rmse=0.008,
        points_used=3,
    )


def _detector(*, uncertainty: float = 0.026) -> EfficiencyCalibration:
    return EfficiencyCalibration(
        detector_id="South-HPGe",
        C1=-20.26,
        C2=10.29,
        C3=-1.655,
        C4=0.0867,
        geometry_factor_A=0.01,
        al_window_T1_um=450.0,
        detector_thickness_DI_cm=6.45,
        dead_layer_DL_um=0.7,
        incident_angle_AI_deg=12.0,
        detector_diameter_cm=6.0,
        source_distance_cm=25.0,
        relative_uncertainty=uncertainty,
    )


def test_efficiency_profile_is_canonical_and_survives_session_round_trip() -> None:
    controller = _controller()
    spectrum_id = controller.document.active_spectrum_id
    assert spectrum_id is not None
    points = (
        EfficiencyPoint(100.0, 1000.0, 100.0, 1000.0, 0.5, activity_source_id="A"),
        EfficiencyPoint(500.0, 500.0, 100.0, 1000.0, 0.5, activity_source_id="A"),
        EfficiencyPoint(1000.0, 250.0, 100.0, 1000.0, 0.5, activity_source_id="A"),
    )
    controller.apply_efficiency_calibration(_fit(), _detector(), points=points)

    record = controller.document.spectrum_by_id(spectrum_id)
    assert record is not None and record.detector_profile_id is not None
    profile = controller.active_detector_profile()
    assert profile is not None and profile.efficiency_model is not None
    assert profile.efficiency_model.model_key == "log_poly_2"
    assert len(profile.efficiency_model.points) == 3
    assert all(point["activity_source_id"] == "A" for point in profile.efficiency_model.points)
    assert profile.geometry.crystal_length_cm == pytest.approx(6.45)
    assert profile.geometry.dead_layer_um == pytest.approx(0.7)
    assert profile.geometry.window_thickness_um == pytest.approx(450.0)
    assert profile.geometry.distance_cm == pytest.approx(25.0)
    assert profile.geometry.angle_deg == pytest.approx(12.0)
    assert controller.state.detector_efficiency.C1 == pytest.approx(-20.26)
    assert controller.state.efficiency_fit is not None
    assert controller.state.efficiency_fit.curve.efficiency_uncertainty(
        661.657
    ) == pytest.approx(0.026)

    restored = AnalysisWorkspaceController(
        WorkspaceDocument.from_dict(controller.document.to_dict())
    )
    assert restored.active_detector_profile() == profile
    assert restored.state.detector_efficiency.source_distance_cm == pytest.approx(25.0)
    assert restored.state.efficiency_fit.model_key == "log_poly_2"
    assert restored.state.efficiency_fit.curve.efficiency(661.657) == pytest.approx(
        controller.state.efficiency_fit.curve.efficiency(661.657)
    )


def test_detector_uncertainty_edit_updates_canonical_activity_input() -> None:
    controller = _controller()
    controller.apply_efficiency_calibration(_fit(), _detector())
    before = controller.state.efficiency_fit.curve.efficiency_uncertainty(661.657)

    controller.set_detector_efficiency(_detector(uncertainty=0.075))

    profile = controller.active_detector_profile()
    assert profile is not None
    assert before == pytest.approx(0.026)
    assert controller.state.efficiency_fit.curve.efficiency_uncertainty(
        661.657
    ) == pytest.approx(0.075)
    assert profile.efficiency_model.uncertainty_model["value"] == pytest.approx(
        0.075
    )


def test_fitted_covariance_contributes_to_activity_uncertainty_after_round_trip() -> None:
    points = tuple(
        EfficiencyPoint(
            energy_keV=energy, net_counts=eff * 1e7,
            live_time_s=100.0, activity_bq=1e5,
            emission_probability=1.0, count_uncertainty=eff * 1e5,
        )
        for energy, eff in ((100, 0.025), (200, 0.018), (400, 0.012), (800, 0.008))
    )
    fit = fit_efficiency_model(points, model_key="log_poly_2")
    controller = _controller()
    controller.apply_efficiency_calibration(fit, _detector(uncertainty=0.0), points=points)
    restored = AnalysisWorkspaceController(
        WorkspaceDocument.from_dict(controller.document.to_dict())
    )
    curve = restored.state.efficiency_fit.curve
    relative = float(curve.efficiency_uncertainty(400.0))
    assert relative > 0
    peak = PeakCandidate("p", 2.0, 400.0, 5.0, (399.0, 401.0), 10000.0, 1.0)
    result = calculate_peak_activity(
        peak, restored.spectrum(), efficiency_curve=curve,
        gamma_intensity=0.8, half_life_s=1e8,
    )
    assert result.uncertainty_bq / result.activity_bq == pytest.approx(
        (1 / 10000 + relative**2) ** 0.5
    )


def test_efficiency_profile_undo_restores_prior_detector_state() -> None:
    controller = _controller()
    spectrum_id = controller.document.active_spectrum_id
    assert spectrum_id is not None
    before = controller.active_detector_profile()
    after = controller.proposed_efficiency_profile(_fit(), _detector())
    command = ApplyDetectorProfileCommand(
        controller, spectrum_id=spectrum_id, before=before, after=after
    )

    command.redo()
    assert controller.active_detector_profile() == after
    assert controller.state.efficiency_fit is not None

    command.undo()
    assert controller.active_detector_profile() is None
    assert controller.state.efficiency_fit is None

    command.redo()
    assert controller.active_detector_profile() == after


def test_legacy_efficiency_fit_migrates_without_changing_measurement() -> None:
    controller = AnalysisWorkspaceController(
        replace(_controller().state, efficiency_fit=_fit(), detector_efficiency=_detector())
    )
    migrated = controller.document
    assert controller.active_detector_profile() is not None
    legacy = replace(
        migrated,
        detector_profiles=(),
        spectra=tuple(replace(item, detector_profile_id=None) for item in migrated.spectra),
        workflow_state={
            **migrated.workflow_state,
            "analysis_workspace_v1": {
                **migrated.workflow_state["analysis_workspace_v1"],
                "efficiency_fit": migrated.detector_profiles[0].efficiency_model.parameters["fit_result"],
                "detector_efficiency": migrated.detector_profiles[0].efficiency_model.parameters["detector_calibration"],
            },
        },
    )
    assert legacy.workflow_state["analysis_workspace_v1"]["efficiency_fit"] is not None

    restored = AnalysisWorkspaceController(legacy)

    assert restored.active_detector_profile() is not None
    assert restored.state.efficiency_fit.model_key == "log_poly_2"
    assert restored.state.detector_efficiency.detector_id == "South-HPGe"
    assert restored.spectrum().counts is legacy.spectra[0].spectrum.counts
    assert restored.document.workflow_state["analysis_workspace_v1"]["efficiency_fit"] is None


def test_migrated_fit_does_not_leak_to_unprofiled_spectrum() -> None:
    foreground = _controller().state.spectra[0]
    background = SpectrumSlot(
        "background", "Background",
        GammaSpectrum(counts=[2, 3, 4], live_time=100.0, spectrum_id="background"),
    )
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(foreground, background),
            efficiency_fit=_fit(),
            detector_efficiency=_detector(),
        )
    )
    assert controller.active_detector_profile() is not None
    controller.set_activity_results((ActivityCalculationResult(
        "Cs-137", 661.657, 10.0, 1.0, 10.0, 0.0, 1.0, 0.0, ""
    ),))
    assert controller.state.activity_results
    controller.select_spectrum("background")
    assert controller.active_detector_profile() is None
    assert controller.state.efficiency_fit is None
    assert controller.state.detector_efficiency is None
    assert controller.state.activity_results == ()


def test_shared_profile_edit_clones_profile_and_undo_restores_reference() -> None:
    foreground = _controller().state.spectra[0]
    background = SpectrumSlot(
        "background", "Background",
        GammaSpectrum(counts=[2, 3, 4], live_time=100.0, spectrum_id="background"),
    )
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(spectra=(foreground, background))
    )
    controller.apply_efficiency_calibration(_fit(), _detector())
    first_id = controller.document.active_spectrum_id
    first_profile = controller.active_detector_profile()
    assert first_id is not None and first_profile is not None
    background_id = next(
        item.spectrum_id for item in controller.document.spectra
        if item.spectrum_id != first_id
    )
    controller.apply_detector_profile(background_id, first_profile)
    edited = replace(_detector(), detector_id="Background-HPGe")
    after = controller.proposed_efficiency_profile(
        _fit(), edited, spectrum_id=background_id
    )
    assert after.detector_profile_id != first_profile.detector_profile_id
    command = ApplyDetectorProfileCommand(
        controller, spectrum_id=background_id, before=first_profile, after=after
    )
    command.redo()
    assert controller.document.spectrum_by_id(first_id).detector_profile_id == first_profile.detector_profile_id
    assert controller.document.detector_profile_by_id(first_profile.detector_profile_id) == first_profile
    assert controller.document.spectrum_by_id(background_id).detector_profile_id == after.detector_profile_id
    command.undo()
    assert controller.document.spectrum_by_id(background_id).detector_profile_id == first_profile.detector_profile_id
    assert controller.document.detector_profile_by_id(after.detector_profile_id) is None


def test_malformed_persisted_points_and_covariance_are_rejected() -> None:
    controller = _controller()
    controller.apply_efficiency_calibration(_fit(), _detector())
    profile = controller.active_detector_profile()
    assert profile is not None
    malformed_points = replace(
        profile, efficiency_model=replace(
            profile.efficiency_model, points=({"unexpected": 1},)
        )
    )
    with pytest.raises(WorkspaceValidationError, match="points\\[0\\]"):
        controller.apply_detector_profile(
            controller.document.active_spectrum_id, malformed_points
        )
    bad_fit = dict(profile.efficiency_model.parameters["fit_result"])
    bad_curve = dict(bad_fit["curve"])
    bad_curve["uncertainty_model"] = {
        "type": "fit_covariance", "parameter_names": ["a0"],
        "covariance": [[-1.0]],
    }
    bad_fit["curve"] = bad_curve
    malformed_covariance = replace(
        profile, efficiency_model=replace(
            profile.efficiency_model,
            parameters={**profile.efficiency_model.parameters, "fit_result": bad_fit},
        ),
    )
    with pytest.raises(WorkspaceValidationError, match="positive semidefinite"):
        controller.apply_detector_profile(
            controller.document.active_spectrum_id, malformed_covariance
        )
