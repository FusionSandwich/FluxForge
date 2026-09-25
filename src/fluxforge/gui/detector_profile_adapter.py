"""Bridge legacy efficiency controls to the canonical detector profile.

The persisted fit, measurement points, and detector geometry live together in
``DetectorProfile``.  This adapter does not apply geometric corrections: a fitted
absolute efficiency already includes the calibration geometry.  Corrections for
another geometry require an explicit transfer model and belong to quantification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.analysis_workspace import EfficiencyCalibrationFitResult
from fluxforge.core.workspace_document import (
    DetectorGeometry,
    DetectorProfile,
    EfficiencyModelState,
)
from fluxforge.io.flux_wire import EfficiencyCalibration


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    raise TypeError(f"Unsupported detector-profile value: {type(value).__name__}")


def detector_profile_with_efficiency(
    *,
    spectrum_id: str,
    prior: DetectorProfile | None,
    fit: EfficiencyCalibrationFitResult,
    detector: EfficiencyCalibration,
    points: Sequence[EfficiencyPoint] = (),
) -> DetectorProfile:
    """Create a validated profile leaf for one measured efficiency fit."""

    curve = fit.curve
    fit_payload = {
        field.name: _json_value(getattr(fit, field.name))
        for field in dataclasses.fields(fit)
        if field.name != "curve"
    }
    fit_payload["curve"] = {
        "model_type": curve.model_type,
        "parameters": _json_value(curve.parameters),
        "energy_range": _json_value(curve.energy_range),
        "detector_id": detector.detector_id,
        "calibration_date": curve.calibration_date,
        "calibration_sources": _json_value(curve.calibration_sources),
        "geometry": {
            **_json_value(curve.geometry),
            "geometry_factor_A": float(detector.geometry_factor_A),
            "al_window_T1_um": float(detector.al_window_T1_um),
            "detector_thickness_DI_cm": float(detector.detector_thickness_DI_cm),
            "dead_layer_DL_um": float(detector.dead_layer_DL_um),
            "incident_angle_AI_deg": float(detector.incident_angle_AI_deg),
            "detector_diameter_cm": float(detector.detector_diameter_cm),
            "source_distance_cm": float(detector.source_distance_cm),
        },
        "uncertainty_model": {},
    }
    covariance = getattr(fit, "covariance", ())
    if covariance is None:
        covariance = ()
    covariance_rows = tuple(
        tuple(float(value) for value in row) for row in covariance
    )
    if covariance_rows:
        uncertainty_model = {
            "type": "fit_covariance",
            "covariance": [list(row) for row in covariance_rows],
            "parameter_names": list(getattr(fit, "covariance_parameters", ())),
            "relative_systematic": float(detector.relative_uncertainty),
        }
    else:
        uncertainty_model = {
            "type": "constant",
            "value": float(detector.relative_uncertainty),
        }
    fit_payload["curve"]["uncertainty_model"] = uncertainty_model
    model = EfficiencyModelState(
        model_key=fit.model_key,
        parameters={
            "fit_result": fit_payload,
            "detector_calibration": _json_value(detector),
        },
        points=tuple(_json_value(point) for point in points),
        covariance=covariance_rows,
        uncertainty_model=fit_payload["curve"]["uncertainty_model"],
    )
    old_geometry = prior.geometry if prior is not None else DetectorGeometry()
    geometry = DetectorGeometry(
        crystal_diameter_cm=float(detector.detector_diameter_cm),
        crystal_length_cm=float(detector.detector_thickness_DI_cm),
        dead_layer_um=float(detector.dead_layer_DL_um),
        window_material=old_geometry.window_material or "Aluminum",
        window_thickness_um=float(detector.al_window_T1_um),
        distance_cm=float(detector.source_distance_cm),
        angle_deg=float(detector.incident_angle_AI_deg),
        metadata={
            **dict(old_geometry.metadata),
            "geometry_factor_A": float(detector.geometry_factor_A),
            "legacy_coefficients": [
                float(detector.C1),
                float(detector.C2),
                float(detector.C3),
                float(detector.C4),
            ],
        },
    )
    profile = dataclasses.replace(
        prior
        or DetectorProfile(detector_profile_id=f"{spectrum_id}-detector-profile"),
        detector_id=str(detector.detector_id),
        efficiency_model=model,
        geometry=geometry,
        uncertainty={
            **dict(prior.uncertainty if prior is not None else {}),
            "efficiency_relative_fraction": float(detector.relative_uncertainty),
        },
        provenance={
            **dict(prior.provenance if prior is not None else {}),
            "efficiency_model_key": fit.model_key,
            "efficiency_points_used": int(fit.points_used),
        },
    )
    profile.validate("detector_profile")
    return profile


def detector_calibration_from_profile(
    profile: DetectorProfile,
) -> EfficiencyCalibration:
    """Restore editable detector fields from their canonical profile state."""

    model = profile.efficiency_model
    raw = model.parameters.get("detector_calibration") if model else None
    if isinstance(raw, Mapping):
        allowed = {field.name for field in dataclasses.fields(EfficiencyCalibration)}
        return EfficiencyCalibration(**{key: raw[key] for key in allowed if key in raw})
    geometry = profile.geometry
    coefficients = geometry.metadata.get("legacy_coefficients", ())
    values = list(coefficients) if isinstance(coefficients, (tuple, list)) else []
    values += [0.0] * (4 - len(values))
    return EfficiencyCalibration(
        detector_id=profile.detector_id,
        C1=float(values[0]),
        C2=float(values[1]),
        C3=float(values[2]),
        C4=float(values[3]),
        geometry_factor_A=float(geometry.metadata.get("geometry_factor_A", 1.0)),
        al_window_T1_um=float(geometry.window_thickness_um or 0.0),
        detector_thickness_DI_cm=float(geometry.crystal_length_cm or 0.0),
        dead_layer_DL_um=float(geometry.dead_layer_um or 0.0),
        incident_angle_AI_deg=float(geometry.angle_deg or 0.0),
        detector_diameter_cm=float(geometry.crystal_diameter_cm or 0.0),
        source_distance_cm=float(geometry.distance_cm or 0.0),
        relative_uncertainty=float(
            profile.uncertainty.get("efficiency_relative_fraction", 0.0)
        ),
    )
