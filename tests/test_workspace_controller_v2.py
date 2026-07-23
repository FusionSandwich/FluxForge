from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fluxforge.core.analysis_workspace import (
    ActivityCalculationResult,
    PeakCandidate,
    ROIAnalysisResult,
    SurveyPoint,
)
from fluxforge.core.workspace_document import (
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    DetectorGeometry,
    DetectorProfile,
    PeakModel,
    SpectrumRoleAssignment,
    WorkspaceDocument,
    WorkspaceSpectrum,
    WorkspaceValidationError,
)
from fluxforge.gui.analysis_workspace import (
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    LoadedSpectrumRecord,
    SpectrumSlot,
)
from fluxforge.io.flux_wire import EfficiencyCalibration
from fluxforge.io.spe import GammaSpectrum


def _spectrum(spectrum_id: str = "sample") -> GammaSpectrum:
    return GammaSpectrum(
        counts=np.asarray([2.0, 5.0, 9.0, 4.0]),
        channels=np.arange(4),
        energies=np.asarray([100.0, 101.0, 102.0, 103.0]),
        live_time=10.0,
        real_time=11.0,
        spectrum_id=spectrum_id,
    )


def _candidate(
    peak_id: str = "peak-1", *, nuclide: str | None = "Cs-137"
) -> PeakCandidate:
    return PeakCandidate(
        peak_id=peak_id,
        channel=2.0,
        energy_keV=102.0,
        significance=6.5,
        roi_bounds_keV=(100.5, 103.5),
        net_counts=13.0,
        fit_quality=1.2,
        nuclide=nuclide,
        candidate_nuclides=("Cs-137", "Ba-137m"),
        reference_lines_keV=(661.657,),
        tags=("reviewed",),
    )


def _peak(spectrum_id: str = "sample", peak_id: str = "peak-1") -> PeakModel:
    return PeakModel(
        peak_id=peak_id,
        spectrum_id=spectrum_id,
        roi_id=None,
        centroid_channel=2.0,
        centroid_energy_keV=102.0,
        net_counts=13.0,
        significance=6.5,
        fit_quality=1.2,
        manual_overrides={"roi_bounds_keV": [100.5, 103.5]},
    )


def _document() -> WorkspaceDocument:
    spectrum = _spectrum()
    peak = _peak()
    roi = AnalysisROI(
        roi_id="roi-1",
        spectrum_id="sample",
        signal_range=(100.5, 103.5),
        left_background_range=(98.0, 100.0),
        right_background_range=(104.0, 106.0),
        associated_peak_ids=("peak-1",),
    )
    return WorkspaceDocument(
        document_id="controller-test",
        spectra=(
            WorkspaceSpectrum(
                spectrum_id="sample",
                spectrum=spectrum,
                label="UWNR sample",
                source_path="C:/data/sample.asc",
                extensions={"slot_label": "Foreground"},
            ),
        ),
        spectrum_roles=(
            SpectrumRoleAssignment(role="foreground", spectrum_ids=("sample",)),
        ),
        active_spectrum_id="sample",
        rois=(roi,),
        peaks=(peak,),
        detector_profiles=(
            DetectorProfile(
                detector_profile_id="hpge-1",
                detector_id="South",
                geometry=DetectorGeometry(distance_cm=25.0),
            ),
        ),
        viewports=(
            CanvasViewport(
                viewport_id="spectrum",
                spectrum_id="sample",
                x_range=(100.0, 104.0),
                selected_roi_id="roi-1",
            ),
        ),
        pinned_nuclides=("Cs-137",),
        provenance={"campaign": "UWNR"},
        extensions={"future_contract": {"retained": True}},
    )


def test_legacy_state_is_adapted_to_canonical_document_without_copying_arrays():
    spectrum = _spectrum()
    state = AnalysisWorkspaceState(
        spectra=(SpectrumSlot("foreground", "Foreground", spectrum),),
        loaded_spectra=(
            LoadedSpectrumRecord(
                "sample", "UWNR sample", spectrum, "C:/data/sample.asc"
            ),
        ),
        active_spectrum_key="foreground",
        peaks=(_candidate(),),
        pinned_nuclides=("Cs-137",),
    )

    controller = AnalysisWorkspaceController(state)

    assert controller.document.active_spectrum_id == "foreground"
    assert controller.document.spectrum_roles == (
        SpectrumRoleAssignment(role="foreground", spectrum_ids=("foreground",)),
    )
    canonical = controller.document.spectrum_by_id("foreground")
    assert canonical is not None
    assert canonical.spectrum is spectrum
    assert canonical.spectrum.counts is spectrum.counts
    model = controller.document.peak_by_id("foreground", "peak-1")
    assert model is not None
    assert model.assignments[0].nuclide == "Cs-137"
    assert model.manual_overrides["roi_bounds_keV"] == (100.5, 103.5)


def test_source_key_maps_slot_role_to_registered_spectrum_identity():
    spectrum = _spectrum()
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(
                SpectrumSlot(
                    "foreground",
                    "Foreground",
                    spectrum,
                    source_key="sample",
                    source_label="UWNR sample",
                ),
            ),
            loaded_spectra=(LoadedSpectrumRecord("sample", "UWNR sample", spectrum),),
        )
    )

    assert controller.document.spectrum_roles[0].spectrum_ids == ("sample",)
    assert controller.document.active_spectrum_id == "sample"
    assert (
        controller.document.spectrum_by_id("sample").spectrum.counts is spectrum.counts
    )


def test_document_constructor_and_set_document_rebuild_compatibility_projection():
    document = _document()
    controller = AnalysisWorkspaceController(document)
    state_events = []
    document_events = []
    controller.subscribe(state_events.append)
    controller.subscribe_document(document_events.append)

    replacement = replace(
        document, title="Restored session", pinned_nuclides=("Co-60",)
    )
    returned = controller.set_document(replacement)

    assert returned is replacement
    assert controller.document is replacement
    assert controller.state.active_spectrum_key == "foreground"
    assert controller.state.loaded_spectra[0].source_path == "C:/data/sample.asc"
    assert (
        controller.state.spectra[0].spectrum.counts
        is document.spectra[0].spectrum.counts
    )
    assert controller.state.peaks[0].roi_bounds_keV == (100.5, 103.5)
    assert controller.state.pinned_nuclides == ("Co-60",)
    assert state_events == [controller.state]
    assert document_events == [replacement]


def test_legacy_mutation_preserves_unrelated_document_leaves_and_array_identity():
    document = _document()
    secondary = WorkspaceSpectrum(
        spectrum_id="secondary",
        spectrum=_spectrum("secondary"),
        label="Repeat count",
    )
    document = replace(
        document,
        spectra=document.spectra + (secondary,),
        spectrum_roles=document.spectrum_roles
        + (
            SpectrumRoleAssignment(
                role="comparison", spectrum_ids=("sample", "secondary")
            ),
        ),
    )
    counts = document.spectra[0].spectrum.counts
    controller = AnalysisWorkspaceController(document)

    controller.set_peak_search_method("second_derivative")

    assert controller.document.rois is document.rois
    assert controller.document.detector_profiles is document.detector_profiles
    assert controller.document.viewports is document.viewports
    assert controller.document.provenance == {"campaign": "UWNR"}
    assert controller.document.extensions["future_contract"]["retained"] is True
    assert controller.document.spectrum_roles[-1].spectrum_ids == (
        "sample",
        "secondary",
    )
    assert controller.document.spectra[0].spectrum.counts is counts
    assert (
        controller.document.workflow_state["analysis_workspace_v1"][
            "peak_search_method"
        ]
        == "second_derivative"
    )


def test_inventory_edit_preserves_multi_spectrum_and_unprojected_roles():
    document = _document()
    secondary = WorkspaceSpectrum(
        spectrum_id="secondary",
        spectrum=_spectrum("secondary"),
        label="Repeat count",
    )
    document = replace(
        document,
        spectra=document.spectra + (secondary,),
        spectrum_roles=document.spectrum_roles
        + (
            SpectrumRoleAssignment(
                role="comparison", spectrum_ids=("sample", "secondary")
            ),
            SpectrumRoleAssignment(role="archive", spectrum_ids=("secondary",)),
        ),
    )
    controller = AnalysisWorkspaceController(document)

    controller.register_loaded_spectrum(_spectrum("new"), label="New count", key="new")

    roles = {
        item.role: item.spectrum_ids for item in controller.document.spectrum_roles
    }
    assert roles["comparison"] == ("sample", "secondary")
    assert roles["archive"] == ("secondary",)


def test_legacy_workflow_dtos_and_detector_configuration_round_trip_in_document():
    spectrum = _spectrum()
    activity = ActivityCalculationResult(
        nuclide="Cs-137",
        line_energy_keV=661.657,
        activity_bq=120.0,
        uncertainty_bq=4.0,
        age_corrected_activity_bq=121.0,
        mda_bq=2.0,
        half_life_s=1.0e9,
        source_age_s=60.0,
        chain_summary="direct",
    )
    roi = ROIAnalysisResult(
        label="Cs-137",
        roi_bounds_keV=(650.0, 670.0),
        gross_counts=100.0,
        gross_counts_uncertainty=10.0,
        background_counts=20.0,
        background_counts_uncertainty=4.0,
        net_counts=80.0,
        net_counts_uncertainty=10.77,
        centroid_keV=661.6,
        centroid_uncertainty_keV=0.1,
        significance=7.4,
        background_method="sideband",
        peak_search_method="mariscotti",
        sideband_bounds_keV=((640.0, 648.0), (672.0, 680.0)),
        notes=("accepted",),
    )
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(SpectrumSlot("foreground", "Foreground", spectrum),),
            activity_results=(activity,),
            roi_analysis=roi,
            survey_points=(SurveyPoint("A", 46.7, -92.1, "foreground"),),
            detector_efficiency=EfficiencyCalibration(
                detector_id="South", source_distance_cm=25.0
            ),
        )
    )

    restored = AnalysisWorkspaceController(
        WorkspaceDocument.from_dict(controller.document.to_dict())
    ).state

    assert restored.activity_results == (activity,)
    assert restored.roi_analysis == roi
    assert restored.survey_points[0].latitude == pytest.approx(46.7)
    assert restored.detector_efficiency.detector_id == "South"
    assert restored.detector_efficiency.source_distance_cm == pytest.approx(25.0)


def test_peak_and_roi_leaf_edits_cleanup_only_dependent_references():
    controller = AnalysisWorkspaceController(_document())
    added = replace(_peak(peak_id="peak-2"), centroid_channel=2.5)
    controller.upsert_peak_model(added)
    assert controller.document.peak_by_id("sample", "peak-2") is added

    controller.delete_peak_model("sample", "peak-1")
    assert controller.document.peak_by_id("sample", "peak-1") is None
    assert controller.document.roi_by_id("roi-1").associated_peak_ids == ()
    assert controller.document.detector_profiles[0].detector_id == "South"

    controller.delete_roi("roi-1")
    assert controller.document.roi_by_id("roi-1") is None
    assert controller.document.viewport_by_id("spectrum").selected_roi_id is None


def test_role_detector_viewport_and_pinned_leaf_edits_are_canonical():
    controller = AnalysisWorkspaceController(_document())
    profile = replace(
        controller.document.detector_profiles[0],
        geometry=DetectorGeometry(distance_cm=12.5),
    )
    viewport = replace(controller.document.viewports[0], x_range=(101.0, 103.0))

    controller.assign_spectrum_role("background", "sample")
    controller.upsert_detector_profile(profile)
    controller.upsert_viewport(viewport)
    controller.set_pinned_nuclides(("Co-60", "Co-60", "Cs-137"))

    assert controller.document.spectrum_roles[-1].role == "background"
    assert (
        controller.document.detector_profile_by_id("hpge-1").geometry.distance_cm
        == 12.5
    )
    assert controller.document.viewport_by_id("spectrum").x_range == (101.0, 103.0)
    assert controller.document.pinned_nuclides == ("Co-60", "Cs-137")
    assert controller.state.pinned_nuclides == ("Co-60", "Cs-137")


def test_apply_calibration_copies_container_reuses_count_arrays_and_updates_profile_leaf():
    controller = AnalysisWorkspaceController(_document())
    original = controller.document.spectrum_by_id("sample").spectrum
    calibration = CalibrationModel(
        model_key="linear",
        coefficients=(0.5, 2.0),
        deviation_pairs=((102.5, 0.25),),
    )

    controller.apply_calibration("sample", calibration)

    calibrated_record = controller.document.spectrum_by_id("sample")
    calibrated = calibrated_record.spectrum
    assert calibrated is not original
    assert calibrated.counts is original.counts
    assert calibrated.counts_uncertainty is original.counts_uncertainty
    assert calibrated.channels is original.channels
    assert original.calibration == {}
    assert np.array_equal(original.energies, np.asarray([100.0, 101.0, 102.0, 103.0]))
    assert calibrated.calibration["energy"] == [0.5, 2.0]
    assert calibrated_record.detector_profile_id == "sample-detector"
    assert (
        controller.document.detector_profile_by_id("sample-detector").energy_calibration
        is calibration
    )
    assert calibrated.energies.shape == original.channels.shape

    controller.apply_calibration("sample", None)
    cleared = controller.document.spectrum_by_id("sample").spectrum
    assert cleared is not calibrated
    assert cleared.counts is original.counts
    assert cleared.energies is None
    assert calibrated.energies is not None
    assert (
        controller.document.detector_profile_by_id("sample-detector").energy_calibration
        is None
    )


def test_set_document_rejects_invalid_reference_without_changing_controller():
    original = _document()
    controller = AnalysisWorkspaceController(original)
    invalid = replace(original, active_spectrum_id="missing")

    with pytest.raises(WorkspaceValidationError, match="active_spectrum_id"):
        controller.set_document(invalid)

    assert controller.document is original
