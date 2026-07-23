from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest

from fluxforge.core import WorkspaceDocument as LazyWorkspaceDocument
from fluxforge.core.workspace_document import (
    WORKSPACE_DOCUMENT_SCHEMA,
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    CorrectionSettings,
    DetectorGeometry,
    DetectorProfile,
    EfficiencyModelState,
    FitDiagnostics,
    NuclideAssignment,
    PeakComponent,
    PeakModel,
    SpectrumRoleAssignment,
    WorkspaceDocument,
    WorkspaceSpectrum,
    WorkspaceValidationError,
)
from fluxforge.io.spe import GammaSpectrum


def _spectrum() -> GammaSpectrum:
    return GammaSpectrum(
        counts=np.asarray([0.0, 4.0, 9.0, 1.0]),
        channels=np.arange(4),
        live_time=10.0,
        real_time=12.0,
        spectrum_id="detector-native-id",
        detector_id="HPGe-1",
        calibration={"energy": [0.0, 1.5]},
        metadata={"operator": "UWNR"},
    )


def _rich_document() -> WorkspaceDocument:
    calibration = CalibrationModel(
        model_key="polynomial",
        coefficients=(0.0, 1.5),
        # A channel/energy pair is data, not an increasing mathematical range.
        calibration_points=((2.0, 1.0), (3.0, 4.5)),
        deviation_pairs=((100.0, -0.04),),
        covariance=((0.04, 0.0), (0.0, 0.0001)),
        valid_range=(0.0, 3000.0),
    )
    profile = DetectorProfile(
        detector_profile_id="profile-1",
        detector_id="HPGe-1",
        energy_calibration=calibration,
        fwhm_calibration=CalibrationModel(
            model_key="sqrt-polynomial", coefficients=(0.8, 0.001)
        ),
        efficiency_model=EfficiencyModelState(
            model_key="log-polynomial",
            parameters={"coefficients": [-3.1, -0.7]},
            points=({"energy_keV": 661.657, "efficiency": 0.012},),
            covariance=((0.01, 0.0), (0.0, 0.02)),
            uncertainty_model={"relative_percent": 2.5},
        ),
        geometry=DetectorGeometry(
            crystal_diameter_cm=6.0,
            crystal_length_cm=6.5,
            dead_layer_um=0.7,
            window_material="aluminum",
            window_thickness_um=500.0,
            distance_cm=12.0,
        ),
        corrections=CorrectionSettings(
            attenuation_enabled=True,
            self_shielding_enabled=True,
            summing_enabled=False,
        ),
        uncertainty={"geometry_relative": 0.01},
        covariance=((0.04,),),
    )
    component = PeakComponent(
        component_id="component-1",
        shape="gaussian-tail",
        centroid=2.0,
        area=12.0,
        amplitude=7.0,
        fwhm=1.2,
        uncertainty=1.0,
        parameters={"tail": 0.03},
    )
    assignment = NuclideAssignment(
        nuclide="Cs-137",
        line_energy_keV=661.657,
        library_id="fixture-lib@1",
        confidence=0.95,
        manual=True,
    )
    peak = PeakModel(
        peak_id="peak-1",
        spectrum_id="spectrum-1",
        roi_id="roi-1",
        centroid_channel=2.0,
        centroid_energy_keV=3.0,
        components=(component,),
        assignments=(assignment,),
        tags=("reviewed",),
        status="accepted",
        net_counts=10.0,
        significance=3.2,
        fit_quality=1.1,
        normalized_residuals=(-0.2, 0.1),
        residual_channels=(1.0, 2.0),
    )
    roi = AnalysisROI(
        roi_id="roi-1",
        spectrum_id="spectrum-1",
        signal_range=(1.0, 2.5),
        left_background_range=(0.0, 1.0),
        right_background_range=(2.5, 3.5),
        associated_peak_ids=("peak-1",),
        fit_revision=4,
        label="661.7 keV review",
    )
    diagnostics = FitDiagnostics(
        diagnostic_id="diagnostic-1",
        spectrum_id="spectrum-1",
        roi_id="roi-1",
        peak_id="peak-1",
        fit_revision=4,
        status="valid",
        x=(1.0, 2.0),
        observed=(4.0, 9.0),
        model=(4.2, 8.7),
        uncertainty=(2.0, 3.0),
        normalized_residuals=(-0.1, 0.1),
        goodness_of_fit={"chi_squared": 0.02},
        method="poisson-likelihood",
    )
    return WorkspaceDocument(
        document_id="uwrn-analysis",
        title="UWNR HPGe analysis",
        created_at="2026-07-22T20:00:00+00:00",
        updated_at="2026-07-22T20:01:00+00:00",
        spectra=(
            WorkspaceSpectrum(
                spectrum_id="spectrum-1",
                spectrum=_spectrum(),
                label="UWNR foreground",
                source_path="fixture.spe",
                detector_profile_id="profile-1",
                provenance={"sha256": "abc"},
            ),
        ),
        spectrum_roles=(
            SpectrumRoleAssignment(role="foreground", spectrum_ids=("spectrum-1",)),
        ),
        active_spectrum_id="spectrum-1",
        rois=(roi,),
        peaks=(peak,),
        detector_profiles=(profile,),
        fit_diagnostics=(diagnostics,),
        viewports=(
            CanvasViewport(
                viewport_id="primary-spectrum",
                spectrum_id="spectrum-1",
                x_range=(0.5, 3.5),
                y_range=(0.1, 12.0),
                log_y=True,
                overlays=("fits", "roi-bounds"),
                residual_mode="compact",
                selected_roi_id="roi-1",
                crosshair_enabled=True,
            ),
        ),
        pinned_nuclides=("Cs-137",),
        nuclide_tags={"Cs-137": ("benchmark",)},
        plot_settings={"theme": "dark"},
        workflow_state={"analysis_mode": "expert"},
        provenance={"application": "FluxForge"},
        extensions={"plugin.example": {"value": 2}},
    )


def test_workspace_document_v2_rich_json_round_trip() -> None:
    document = _rich_document()
    payload = document.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    restored = WorkspaceDocument.from_dict(json.loads(encoded))

    assert restored.to_dict() == payload
    assert restored.schema == WORKSPACE_DOCUMENT_SCHEMA
    assert restored.spectra[0].spectrum.counts.tolist() == [0.0, 4.0, 9.0, 1.0]
    assert restored.detector_profiles[0].efficiency_model is not None
    assert restored.viewports[0].selected_roi_id == "roi-1"


def test_empty_workspace_round_trip_is_valid() -> None:
    document = WorkspaceDocument(
        document_id="empty",
        created_at="2026-07-22T20:00:00+00:00",
        updated_at="2026-07-22T20:00:00+00:00",
    )
    assert (
        WorkspaceDocument.from_dict(document.to_dict()).to_dict() == document.to_dict()
    )


def test_core_lazy_export_does_not_create_io_import_cycle() -> None:
    assert LazyWorkspaceDocument is WorkspaceDocument
    assert _spectrum().counts.shape == (4,)


def test_schema_resource_is_packaged_and_matches_domain_version() -> None:
    resource = files("fluxforge").joinpath(
        "resources/schemas/workspace_document_v2.schema.json"
    )
    schema = json.loads(resource.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == WORKSPACE_DOCUMENT_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == 2
    assert schema["additionalProperties"] is False


def _schema_payload() -> tuple[dict, dict]:
    resource = files("fluxforge").joinpath(
        "resources/schemas/workspace_document_v2.schema.json"
    )
    return (
        json.loads(resource.read_text(encoding="utf-8")),
        _rich_document().to_dict(),
    )


def _set_path(payload: dict, path: tuple[object, ...], value: object) -> dict:
    malformed = deepcopy(payload)
    target: object = malformed
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return malformed


@pytest.mark.parametrize(
    "path,value,expected_path",
    [
        (("title",), 12, "document.title"),
        (("pinned_nuclides", 0), 137, "document.pinned_nuclides"),
        (("nuclide_tags", "Cs-137", 0), False, "document.nuclide_tags.Cs-137"),
        (("spectra", 0, "label"), ["foreground"], "spectrum.label"),
        (
            ("spectra", 0, "spectrum", "detector_id"),
            7,
            "spectrum.spectrum.detector_id",
        ),
        (("rois", 0, "associated_peak_ids", 0), 1, "roi.associated_peak_ids"),
        (("peaks", 0, "shape"), 2, "peak.shape"),
        (("peaks", 0, "tags", 0), {"tag": "reviewed"}, "peak.tags"),
        (("peaks", 0, "status"), True, "peak.status"),
        (
            ("peaks", 0, "assignments", 0, "library_id"),
            3,
            "assignment.library_id",
        ),
        (
            ("detector_profiles", 0, "detector_id"),
            ["HPGe-1"],
            "detector_profile.detector_id",
        ),
        (
            ("fit_diagnostics", 0, "warning_flags"),
            ["review", 3],
            "fit_diagnostics.warning_flags",
        ),
        (("fit_diagnostics", 0, "method"), 4, "fit_diagnostics.method"),
        (("viewports", 0, "x_unit"), 1, "viewport.x_unit"),
        (("viewports", 0, "overlays", 0), False, "viewport.overlays"),
    ],
)
def test_from_dict_rejects_non_strings_in_typed_string_fields(
    path, value, expected_path
) -> None:
    payload = _set_path(_rich_document().to_dict(), path, value)
    with pytest.raises(WorkspaceValidationError, match=expected_path):
        WorkspaceDocument.from_dict(payload)


def test_external_json_schema_accepts_complete_canonical_document() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema, payload = _schema_payload()
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("pinned_nuclides", 0), 137),
        (("nuclide_tags", "Cs-137", 0), False),
        (("spectra", 0, "spectrum", "detector_id"), 7),
        (("peaks", 0, "tags", 0), {"tag": "reviewed"}),
        (("detector_profiles", 0, "detector_id"), ["HPGe-1"]),
        (("fit_diagnostics", 0, "warning_flags"), ["review", 3]),
        (("viewports", 0, "overlays", 0), False),
        (("detector_profiles", 0, "unexpected"), "not allowed"),
    ],
)
def test_external_json_schema_rejects_malformed_domain_objects(path, value) -> None:
    schema, payload = _schema_payload()
    malformed = _set_path(payload, path, value)
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(malformed)


def test_every_domain_object_definition_rejects_unknown_properties() -> None:
    schema, _ = _schema_payload()
    domain_definitions = {
        "gammaSpectrum",
        "workspaceSpectrum",
        "spectrumRole",
        "analysisROI",
        "peakComponent",
        "nuclideAssignment",
        "peakModel",
        "calibrationModel",
        "efficiencyModelState",
        "detectorGeometry",
        "correctionSettings",
        "detectorProfile",
        "fitDiagnostics",
        "canvasViewport",
    }
    assert all(
        schema["$defs"][name]["additionalProperties"] is False
        for name in domain_definitions
    )


@pytest.mark.parametrize(
    "mutator, expected_path",
    [
        (
            lambda doc: replace(doc, active_spectrum_id="missing"),
            "active_spectrum_id",
        ),
        (
            lambda doc: replace(
                doc,
                spectra=(doc.spectra[0], doc.spectra[0]),
            ),
            "spectra\\[1\\].spectrum_id",
        ),
        (
            lambda doc: replace(
                doc,
                detector_profiles=(
                    replace(
                        doc.detector_profiles[0], covariance=((1.0, 2.0), (2.0, 1.0))
                    ),
                ),
            ),
            "positive semidefinite",
        ),
        (
            lambda doc: replace(
                doc,
                rois=(replace(doc.rois[0], right_background_range=(2.0, 3.0)),),
            ),
            "right_background_range",
        ),
        (
            lambda doc: replace(
                doc,
                fit_diagnostics=(
                    replace(doc.fit_diagnostics[0], normalized_residuals=(0.1,)),
                ),
            ),
            "equal non-empty",
        ),
    ],
)
def test_nonphysical_or_dangling_state_fails_clearly(mutator, expected_path) -> None:
    with pytest.raises(WorkspaceValidationError, match=expected_path):
        mutator(_rich_document()).validate()


def test_negative_net_peak_area_is_preserved_for_review() -> None:
    document = _rich_document()
    negative_peak = replace(document.peaks[0], net_counts=-2.0, status="invalid")
    reviewed = replace(document, peaks=(negative_peak,))

    assert WorkspaceDocument.from_dict(reviewed.to_dict()).peaks[0].net_counts == -2.0


def test_nonfinite_extension_fails_before_json_write() -> None:
    with pytest.raises(WorkspaceValidationError, match="extensions.result"):
        replace(_rich_document(), extensions={"result": float("nan")}).to_dict()


@pytest.mark.parametrize(
    "value, type_name",
    [
        (np.asarray([1.0]), "ndarray"),
        (object(), "object"),
    ],
)
def test_non_json_extension_values_fail_with_field_path(value, type_name) -> None:
    with pytest.raises(
        WorkspaceValidationError,
        match=rf"extensions.result.*not {type_name}",
    ):
        replace(_rich_document(), extensions={"result": value}).to_dict()


def test_invalid_schema_and_lossy_integer_are_rejected() -> None:
    payload = _rich_document().to_dict()
    payload["schema_version"] = 2.0
    with pytest.raises(WorkspaceValidationError, match="schema_version.*integer"):
        WorkspaceDocument.from_dict(payload)

    payload = _rich_document().to_dict()
    payload["rois"][0]["fit_revision"] = "4"
    with pytest.raises(WorkspaceValidationError, match="roi.fit_revision.*integer"):
        WorkspaceDocument.from_dict(payload)


def test_unknown_document_fields_are_rejected() -> None:
    payload = _rich_document().to_dict()
    payload["temporary_gui_note"] = "do not persist"
    with pytest.raises(WorkspaceValidationError, match="unknown fields"):
        WorkspaceDocument.from_dict(payload)
