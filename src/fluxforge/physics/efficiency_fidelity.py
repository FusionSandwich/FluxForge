"""Source-fidelity controls for explicitly selected efficiency comparisons.

This additive audit never installs a physical calibration or adjusts an activity.
PGT's residual is a polynomial in log(E), NOT exp(polynomial(log(E))).
The proprietary McMaster routines and their edge corrections are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import struct
from typing import Callable, Sequence

import numpy as np

from fluxforge.analysis.qg_calibration import QGEfficiencyTable
from fluxforge.data.efficiency import EfficiencyCurve


def efficiency_fraction(value: float, unit: str) -> float:
    """Convert declared absolute efficiency; plausible unit errors need controls."""
    if unit not in {"percent", "fraction"}:
        raise ValueError("Efficiency unit must be explicitly percent or fraction")
    result = float(value) / (100.0 if unit == "percent" else 1.0)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError("Absolute efficiency must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class Thickness:
    value: float
    unit: str

    def cm(self) -> float:
        factors = {"um": 1e-4, "mm": 0.1, "cm": 1.0}
        if self.unit not in factors:
            raise ValueError("Thickness requires um, mm or cm; areal density differs")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("Thickness must be finite and nonnegative")
        return self.value * factors[self.unit]

    def areal_density(self, density_g_cm3: float) -> float:
        if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0:
            raise ValueError("Density must be finite and positive in g/cm^3")
        return self.cm() * density_g_cm3


def logarithmic_residual(
    energy_keV: float | Sequence[float], coefficients: Sequence[float], *, log_base: str
) -> np.ndarray:
    """PGT multiplicative residual; log base is explicit, not inferred from C1."""
    energy = np.asarray(energy_keV, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.shape != (4,) or np.any(~np.isfinite(coeffs)):
        raise ValueError("Four finite C1-C4 coefficients required")
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Energy must be finite and positive in keV")
    if log_base not in {"ln", "log10"}:
        raise ValueError("Log base must be explicitly ln or log10")
    x = np.log(energy) if log_base == "ln" else np.log10(energy)
    return np.polynomial.polynomial.polyval(x, coeffs)


def coefficient_round_trip(value: float, source_token: str) -> dict:
    """Compare at the source's decimal precision, retaining signed discrepancies."""
    source = Decimal(source_token)
    if not source.is_finite() or not math.isfinite(value):
        raise ValueError("Finite coefficient and decimal source token required")
    tolerance = abs(Decimal(1).scaleb(source.as_tuple().exponent)) / 2
    delta = Decimal(str(value)) - source
    return {
        "value": float(value),
        "source_token": source_token,
        "delta": float(delta),
        "half_last_printed_digit": float(tolerance),
        "within_printed_rounding": abs(delta) <= tolerance,
    }


def known_value_control(
    actual_fraction: float, expected_fraction: float, *, relative_tolerance: float
) -> dict:
    """Independent specified control, including errors that still lie below one."""
    expected = efficiency_fraction(expected_fraction, "fraction")
    if not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("Relative tolerance must be finite in [0, 1)")
    try:
        actual = efficiency_fraction(actual_fraction, "fraction")
    except ValueError:
        return {"status": "invalid_value", "passed": False, "ratio": None}
    ratio = actual / expected
    return {
        "status": "known_value_control",
        "passed": abs(ratio - 1) <= relative_tolerance,
        "ratio": ratio,
        "suspected_percent_fraction_error": math.isclose(ratio, 100, rel_tol=0.01)
        or math.isclose(ratio, 0.01, rel_tol=0.01),
    }


@dataclass(frozen=True)
class Geometry:
    detector_id: str
    distance_cm: float | None
    sample_geometry: str | None
    near_contact: bool = False

    def __post_init__(self):
        if self.distance_cm is not None and (
            not math.isfinite(self.distance_cm) or self.distance_cm < 0
        ):
            raise ValueError("Distance must be finite and nonnegative in cm")


@dataclass(frozen=True)
class CurveIdentity:
    method_id: str
    kind: str
    source_refs: tuple[str, ...]
    geometry: Geometry
    energy_range_keV: tuple[float, float]
    validation_status: str
    attenuation_data: str
    formula: str
    interpolation: str
    count_basis: str = "not_applicable"

    def __post_init__(self):
        # Type hints/frozen dataclasses alone do not freeze caller-owned lists.
        if isinstance(self.source_refs, str):
            raise ValueError(
                "Source references require a sequence of complete identities"
            )
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(
            self, "energy_range_keV", tuple(float(e) for e in self.energy_range_keV)
        )
        statuses = {
            "source_calibration_export": "conditional_source_reproduction",
            "report_effective": "conditional_report_reconstruction",
            "model": "unvalidated_model",
        }
        if statuses.get(self.kind) != self.validation_status:
            raise ValueError("Curve kind and validation status must remain distinct")
        if (
            not self.method_id
            or not self.source_refs
            or any(not isinstance(s, str) or not s.strip() for s in self.source_refs)
        ):
            raise ValueError("Method identity and source references required")
        lo, hi = self.energy_range_keV
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo < hi:
            raise ValueError("Finite positive increasing energy range required")
        if not all((self.attenuation_data, self.formula, self.interpolation)):
            raise ValueError(
                "Attenuation, formula and interpolation identities required"
            )
        if (
            self.kind == "report_effective"
            and self.count_basis != "historical_report_net"
        ):
            raise ValueError(
                "Report-effective curves must retain historical report net basis"
            )

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return dict(
            asdict(self),
            energy_unit="keV",
            output_efficiency_unit="fraction",
            extrapolation="forbidden",
            scientific_admission=False,
            physical_qualified=False,
        )


@dataclass(frozen=True)
class AuditCurve:
    identity: CurveIdentity
    evaluator: Callable[[float], float | dict] | None
    unsupported_reason: str | None = None

    def at(self, energy_keV: float, geometry: Geometry) -> dict:
        """Evaluate a comparison without upgrading it to a physical calibration."""
        row = {
            "energy_keV": float(energy_keV) if math.isfinite(energy_keV) else None,
            "method_id": self.identity.method_id,
            "kind": self.identity.kind,
            "validation_status": self.identity.validation_status,
            "efficiency_fraction": None,
            "admissible_for_comparison": False,
            "scientific_admission": False,
            "physical_qualified": False,
        }
        if not math.isfinite(energy_keV) or energy_keV <= 0:
            return dict(
                row, status="invalid_energy", requested_energy_text=str(energy_keV)
            )
        source = self.identity.geometry
        if geometry.near_contact and source.distance_cm == 25:
            return dict(row, status="excluded_near_contact_25cm_transfer")
        if (
            not all(
                (
                    source.detector_id,
                    geometry.detector_id,
                    source.sample_geometry,
                    geometry.sample_geometry,
                )
            )
            or source.distance_cm is None
            or geometry.distance_cm is None
        ):
            return dict(row, status="unknown_geometry")
        if source != geometry:
            return dict(row, status="excluded_geometry_mismatch")
        if (
            not self.identity.energy_range_keV[0]
            <= energy_keV
            <= self.identity.energy_range_keV[1]
        ):
            return dict(row, status="excluded_outside_range")
        if self.evaluator is None:
            return dict(row, status="unsupported_model", reason=self.unsupported_reason)
        try:
            result = self.evaluator(energy_keV)
            if isinstance(result, dict):
                # Existing diagnostics retain brackets, lines and raw values.
                row.update({k: v for k, v in result.items() if k not in row})
                raw = result.get("efficiency_fraction")
                if raw is None:
                    return dict(row, status=result["status"])
            else:
                raw = result
            value = efficiency_fraction(float(raw), "fraction")
        except (ValueError, OverflowError, FloatingPointError) as error:
            return dict(row, status="excluded_invalid_model_value", reason=str(error))
        return dict(
            row,
            status="conditional_comparison",
            efficiency_fraction=value,
            admissible_for_comparison=True,
        )


def source_table_curve(table: QGEfficiencyTable, *, unit_assumption: str) -> AuditCurve:
    """Reuse byte-bound original adjacent interpolation, including invalid rows."""
    if unit_assumption not in {"percent", "fraction"}:
        raise ValueError("Explicit source export unit assumption required")
    if (
        table.provenance.get("detector") != "South HPGe"
        or table.provenance.get("nominal_geometry") != "Small vial 0.5 mL, 25 cm"
    ):
        raise ValueError(
            "This source adapter requires the declared South small-vial 25 cm profile"
        )
    identity = CurveIdentity(
        method_id=f"south_source_export_{unit_assumption}",
        kind="source_calibration_export",
        source_refs=(f"{table.path.name}:sha256:{table.sha256}",),
        geometry=Geometry(table.provenance["detector"], 25.0, "small_vial_0.5mL"),
        energy_range_keV=(float(table.energies[0]), float(table.energies[-1])),
        validation_status="conditional_source_reproduction",
        attenuation_data="vendor McMaster-based routines; unavailable; raw export only",
        formula=(
            f"original exported values / "
            f"{'100' if unit_assumption == 'percent' else '1'}; unit conditional"
        ),
        interpolation=(
            "linear in keV and original efficiency; " "adjacent positive rows only"
        ),
    )
    return AuditCurve(
        identity, lambda e: table.diagnostic_at(e, unit_assumption=unit_assumption)
    )


def existing_efficiency_curve(
    curve: EfficiencyCurve, identity: CurveIdentity
) -> AuditCurve:
    """Explicit alternative using existing EfficiencyCurve, with audit range gates.

    Snapshot the caller's curve; do not change a shared curve or allow its later
    mutation to alter this comparison. Existing implicit endpoint behavior is
    blocked outside the declared range.
    """
    if (
        identity.kind != "model"
        or tuple(curve.energy_range) != identity.energy_range_keV
    ):
        raise ValueError(
            "Existing model adapter requires matching declared energy range"
        )
    snapshot = deepcopy(curve)
    return AuditCurve(identity, lambda e: float(snapshot.efficiency(e)))


def compare_efficiencies(
    reference: AuditCurve,
    alternatives: Sequence[AuditCurve],
    energies_keV: Sequence[float],
    *,
    geometry: Geometry,
    selected_method: str,
) -> dict:
    """Export pointwise deltas/admissibility; selection never changes any defaults."""
    curves = (reference, *alternatives)
    ids = [c.identity.method_id for c in curves]
    if len(ids) != len(set(ids)) or selected_method not in ids:
        raise ValueError(
            "Unique alternatives and an explicit listed method choice required"
        )
    rows = []
    for energy in energies_keV:
        ref = reference.at(energy, geometry)
        for curve in curves:
            row = curve.at(energy, geometry)
            both = ref["admissible_for_comparison"] and row["admissible_for_comparison"]
            delta = (
                row["efficiency_fraction"] - ref["efficiency_fraction"]
                if both
                else None
            )
            rows.append(
                dict(
                    row,
                    reference_method=reference.identity.method_id,
                    delta_fraction=delta,
                    delta_relative_to_reference=(
                        delta / ref["efficiency_fraction"] if both else None
                    ),
                    selected=curve.identity.method_id == selected_method,
                )
            )
    return {
        "selected_method": selected_method,
        "curves": [c.identity.to_dict() for c in curves],
        "rows": rows,
        "interpretation": (
            "conditional pointwise efficiency comparison; " "no activity correction"
        ),
    }


def read_study_efficiency_header(path: Path, *, expected_sha256: str) -> dict:
    """Read corroborated offsets ONLY from an explicitly byte-pinned study file.

    The caller must bind expected_sha256 to a separately reviewed study manifest.
    Appendix C's later fields disagree with these 32 files; no general decoder
    or claim that saved settings establish final-report processing is made.
    """
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Original ANS hash differs from the reviewed study identity")
    if len(payload) < 1548 or struct.unpack_from("<h", payload, 0)[0] != 4:
        raise ValueError("Unsupported study header revision/length")
    first, last = struct.unpack_from("<2h", payload, 1020)
    nrois = struct.unpack_from("<h", payload, 1026)[0]
    if (
        (first, last) != (0, 8191)
        or nrois < 0
        or len(payload) != 1548 + 8192 * 4 + 50 * nrois
    ):
        raise ValueError("Study channels/ROI length do not close")
    offsets = {
        "A": 504,
        "export_DI_slot": 520,
        "C1": 524,
        "C2": 528,
        "C3": 532,
        "C4": 536,
        "Error_slot": 540,
        "detector_thickness_cm": 640,
        "diameter_cm": 644,
        "source_distance_cm": 656,
        "angle_deg": 660,
        "window_um": 664,
        "dead_layer_observed_um": 824,
    }
    values = {
        name: struct.unpack_from("<f", payload, offset)[0]
        for name, offset in offsets.items()
    }
    if any(not math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite saved efficiency fields")
    return {
        "source_sha256": digest,
        "offsets": offsets,
        "values": values,
        "detector_id": payload[628:640].decode("ascii").strip(" \0"),
        "scope": "byte-pinned rev4 study layout; saved header only",
        "unsupported_details": [
            "export DI slot is not established as physical detector thickness",
            "Error scaling/semantics are unspecified; not standard uncertainty",
            "dead layer at observed 824 matches report um; manual offset/unit differs",
            "saved settings do not establish final report settings",
        ],
    }


def audit_header_export(
    header: dict, table: QGEfficiencyTable, report_text: str
) -> dict:
    """Cross-check binary, export and printed report without reassigning fields."""
    values = header["values"]
    tokens = dict(
        re.findall(r"^\s*(C[1-4]):\s*([+\-\d.Ee]+)", report_text, flags=re.MULTILINE)
    )
    a = re.search(r"Geometry Factor \(A\):\s*([+\-\d.Ee]+)", report_text)
    if set(tokens) != {"C1", "C2", "C3", "C4"} or a is None:
        raise ValueError("Report coefficient anchors incomplete")
    tokens["A"] = a[1]
    vendor = table.vendor_coefficients
    # The CSV's decimal tokens are more precise than the printed report.
    import csv
    import io

    payload = table.path.read_bytes()
    records = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if hashlib.sha256(payload).hexdigest() != table.sha256:
        raise ValueError("Source export changed since byte-bound snapshot")
    export_tokens = dict(zip(records[0][4:], records[1][4:]))
    comparisons = {
        key: {
            "header_to_export": coefficient_round_trip(values[key], export_tokens[key]),
            "header_to_report": coefficient_round_trip(values[key], tokens[key]),
            "export_to_report": coefficient_round_trip(vendor[key], tokens[key]),
        }
        for key in tokens
    }
    anchors = {}
    patterns = {
        "window_um": r"Al Window \(T1\):\s*([\d.]+)\s*(um|mm|cm)\b",
        "dead_layer_observed_um": (
            r"Detector Dead Layer \(DL\):\s*" r"([\d.]+)\s*(um|mm|cm)\b"
        ),
        "detector_thickness_cm": r"Detector Thickness \(DI\)\s*([\d.]+)\s*(um|mm|cm)\b",
        "diameter_cm": r"Detector Diameter:\s*([\d.]+)\s*(um|mm|cm)\b",
        "source_distance_cm": r"Source Dist\.:\s*([\d.]+)\s*(um|mm|cm)\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match is None:
            anchors[key] = {"status": "unknown_missing_report_field", "matches": None}
            continue
        saved_unit = "um" if key.endswith("um") else "cm"
        normalized = Thickness(values[key], saved_unit).cm()
        reported = Thickness(float(match[1]), match[2]).cm()
        half_digit_cm = (
            float(abs(Decimal(1).scaleb(Decimal(match[1]).as_tuple().exponent)) / 2)
            * Thickness(1, match[2]).cm()
        )
        anchors[key] = {
            "status": "checked_declared_units",
            "saved_raw": values[key],
            "saved_unit": saved_unit,
            "report_token": match[1],
            "report_unit": match[2],
            "saved_cm": normalized,
            "report_cm": reported,
            "matches": abs(normalized - reported) <= half_digit_cm,
        }
    angle = re.search(r"Det\. Incident Angle \(AI\):\s*([\d.]+)\s*deg", report_text)
    anchors["angle_deg"] = (
        {"status": "unknown_missing_report_field", "matches": None}
        if angle is None
        else coefficient_round_trip(values["angle_deg"], angle[1])
    )
    if angle is not None:
        anchors["angle_deg"]["matches"] = anchors["angle_deg"][
            "within_printed_rounding"
        ]
    matches = [a["matches"] for a in anchors.values()]
    anchor_status = (
        "contradiction"
        if False in matches
        else "unknown" if None in matches else "corroborated_at_printed_precision"
    )
    return {
        "coefficients": comparisons,
        "coefficient_round_trips_pass": all(
            c[side]["within_printed_rounding"]
            for c in comparisons.values()
            for side in c
        ),
        "export_DI_raw": vendor["DI"],
        "saved_DI_slot": values["export_DI_slot"],
        "physical_detector_thickness_cm": values["detector_thickness_cm"],
        "thickness_geometry_anchors": anchors,
        "report_anchor_status": anchor_status,
        "header_window_matches_export": values["window_um"] == vendor["T1"],
        "header_dead_layer_matches_export": values["dead_layer_observed_um"]
        == vendor["DL"],
        "DI_mapping_status": "unsupported: exported DI differs from physical thickness",
        "Error_export_raw": vendor["Error"],
        "Error_saved_raw": values["Error_slot"],
        "Error_interpretation": "unspecified; never treated as uncertainty",
    }


@dataclass(frozen=True)
class GammaYield:
    value: float
    unit: str
    source_ref: str
    role: str

    def fraction(self) -> float:
        if not self.source_ref or self.role not in {
            "historical_report",
            "modern_evaluated",
        }:
            raise ValueError("Historical or modern evaluated source identity required")
        if self.unit not in {"percent", "fraction"}:
            raise ValueError("Explicit gamma yield unit required")
        value = self.value / (100 if self.unit == "percent" else 1)
        # Gamma yields are photons/decay, and can exceed 1 (e.g. annihilation).
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Gamma yield must be finite and positive photons/decay")
        return value


def compare_gamma_yields(historical: GammaYield, modern: GammaYield | None) -> dict:
    """Retain both yield records; no preferred yield, activity or silent replacement."""
    from dataclasses import asdict

    if historical.role != "historical_report" or (
        modern is not None and modern.role != "modern_evaluated"
    ):
        raise ValueError("Yield roles must be explicitly historical and modern")
    h = historical.fraction()
    m = modern.fraction() if modern is not None else None
    return {
        "historical": dict(asdict(historical), fraction=h),
        "modern": dict(asdict(modern), fraction=m) if modern is not None else None,
        "modern_status": (
            "explicit_evaluated_input" if modern is not None else "unavailable"
        ),
        "delta_fraction": m - h if m is not None else None,
        "interpretation": (
            "yield-only comparison; " "neither efficiency nor activity is changed"
        ),
    }


def report_effective_point(
    *,
    net_counts: float,
    activity_bq: float,
    live_time_s: float,
    historical_yield: GammaYield,
    report_ref: str,
) -> dict:
    """Back-calculated response, never an independent calibration or physical count.

    Using the printed line activity can embed QG's reference-time, attenuation,
    processing and historical-yield conventions. Do not apply corrections again.
    """
    if historical_yield.role != "historical_report" or not report_ref:
        raise ValueError("Historical report/yield identities required")
    probability = historical_yield.fraction()
    inputs = (net_counts, activity_bq, live_time_s)
    if any(not math.isfinite(v) or v <= 0 for v in inputs):
        return {
            "status": "invalid_report_effective_inputs",
            "efficiency_fraction": None,
            "scientific_admission": False,
            "independent_absolute_validation": False,
        }
    value = net_counts / (activity_bq * live_time_s * probability)
    try:
        value = efficiency_fraction(value, "fraction")
    except ValueError:
        value = None
    return {
        "status": (
            "conditional_report_reconstruction"
            if value is not None
            else "invalid_report_effective_value"
        ),
        "kind": "report_effective",
        "efficiency_fraction": value,
        "count_basis": "historical_report_net",
        "report_ref": report_ref,
        "net_counts": net_counts,
        "activity_bq": activity_bq,
        "live_time_s": live_time_s,
        "historical_yield_fraction": probability,
        "historical_yield_source_ref": historical_yield.source_ref,
        "scientific_admission": False,
        "independent_absolute_validation": False,
        "interpretation": (
            "effective response includes unresolved vendor conventions; "
            "not intrinsic efficiency"
        ),
    }


def report_effective_curve(
    energies_keV: Sequence[float],
    points: Sequence[dict],
    *,
    geometry: Geometry,
    source_refs: tuple[str, ...],
) -> AuditCurve:
    """Linear descriptive reconstruction with no fit to target activities."""
    energy = np.array(energies_keV, dtype=float, copy=True)
    energy.setflags(write=False)
    if (
        energy.ndim != 1
        or len(energy) != len(points)
        or len(energy) < 2
        or np.any(~np.isfinite(energy))
        or np.any(energy <= 0)
        or np.any(np.diff(energy) <= 0)
    ):
        raise ValueError("At least two strictly increasing finite energies required")
    values = []
    for point in points:
        if (
            point.get("kind") != "report_effective"
            or point.get("status") != "conditional_report_reconstruction"
            or point.get("count_basis") != "historical_report_net"
        ):
            raise ValueError(
                "Every descriptive knot must be a valid "
                "historical report reconstruction"
            )
        values.append(efficiency_fraction(point["efficiency_fraction"], "fraction"))
    refs = (
        *source_refs,
        *(p["report_ref"] for p in points),
        *(p["historical_yield_source_ref"] for p in points),
    )
    identity = CurveIdentity(
        method_id="historical_report_effective",
        kind="report_effective",
        source_refs=refs,
        geometry=geometry,
        energy_range_keV=(float(energy[0]), float(energy[-1])),
        validation_status="conditional_report_reconstruction",
        attenuation_data="embedded in report effective response; unspecified",
        formula="N_QG/(A_line_Bq*live_seconds*historical_yield_fraction)",
        interpolation="linear descriptive knots; not an independent calibration",
        count_basis="historical_report_net",
    )
    return AuditCurve(identity, lambda e: float(np.interp(e, energy, values)))


def xcom_pgt_alternative(
    *,
    geometry: Geometry,
    coefficients: Sequence[float],
    geometry_factor: float,
    window: Thickness,
    dead_layer: Thickness,
    detector: Thickness,
    angle_deg: float,
    log_base: str,
    energy_range_keV: tuple[float, float],
    source_refs: tuple[str, ...],
) -> AuditCurve:
    """Conditional density-aware PGT-shaped model using existing embedded XCOM.

    This is an explicit alternative, never vendor reproduction. Al/Ge materials
    are explicit here; other window compositions need another named alternative.
    µ/ρ [cm²/g] × ρ [g/cm³] × thickness [cm] is dimensionless.
    """
    from fluxforge.data import xcom
    from fluxforge.physics.attenuation import get_material

    if not math.isfinite(geometry_factor) or not 0 < geometry_factor <= 1:
        raise ValueError("Geometry factor must lie in (0, 1]")
    if not math.isfinite(angle_deg) or not -90 < angle_deg < 90:
        raise ValueError("Incidence angle must lie strictly between -90 and 90 degrees")
    coeffs = tuple(float(c) for c in coefficients)
    logarithmic_residual(100, coeffs, log_base=log_base)
    thicknesses = (window.cm(), dead_layer.cm(), detector.cm())
    if thicknesses[2] <= 0:
        raise ValueError("Active detector thickness must be positive")
    al, ge = get_material("Al"), get_material("Ge")
    lo = max(float(al.data.energies_keV[0]), float(ge.data.energies_keV[0]))
    hi = min(float(al.data.energies_keV[-1]), float(ge.data.energies_keV[-1]))
    if not lo <= energy_range_keV[0] < energy_range_keV[1] <= hi:
        raise ValueError("Range must lie within existing attenuation table support")
    cos_angle = math.cos(math.radians(angle_deg))

    def evaluate(e):
        mu_al, mu_ge = float(al.mu(e)[0]), float(ge.mu(e)[0])
        attenuation = math.exp(
            -(mu_al * thicknesses[0] + mu_ge * thicknesses[1]) / cos_angle
        )
        absorption = -math.expm1(-mu_ge * thicknesses[2] / cos_angle)
        return (
            geometry_factor
            * attenuation
            * absorption
            * float(logarithmic_residual(e, coeffs, log_base=log_base))
        )

    data_hash = hashlib.sha256(Path(xcom.__file__).read_bytes()).hexdigest()
    identity = CurveIdentity(
        method_id=f"pgt_shaped_xcom_density_aware_{log_base}",
        kind="model",
        source_refs=(
            *source_refs,
            f"xcom.py:sha256:{data_hash}",
            (
                f"inputs:C1-C4={coeffs};A={geometry_factor};window_Al={window};"
                f"dead_Ge={dead_layer};active_Ge={detector};AI_deg={angle_deg}"
            ),
        ),
        geometry=geometry,
        energy_range_keV=energy_range_keV,
        validation_status="unvalidated_model",
        attenuation_data=(
            f"FluxForge embedded XCOM-labeled Al/Ge; sha256:{data_hash}; "
            f"densities {al.density},{ge.density} g/cm3; "
            "not vendor McMaster"
        ),
        formula=(
            "A*exp(-(mu_Al*t_window+mu_Ge*t_dead)/cos(AI))"
            f"*(-expm1(-mu_Ge*t_active/cos(AI)))*P({log_base}(E_keV)); "
            "mu=mu/rho*rho"
        ),
        interpolation="existing XCOM log-log linear; audit forbids extrapolation",
    )
    return AuditCurve(identity, evaluate)


@dataclass(frozen=True)
class ReferencePoint:
    reference_id: str
    energy_keV: float
    efficiency_fraction: float
    source_id: str
    origin: str
    geometry: Geometry
    relative_tolerance: float
    certificate_ref: str | None = None


def _source_identity_keys(refs: Sequence[str]) -> set[str]:
    """Normalize exact labels and SHA aliases; path/label changes cannot hide reuse."""
    keys = set()
    for ref in refs:
        label = ref.strip().casefold()
        keys.add(label)
        keys.update(
            "sha256:" + digest
            for digest in re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", label)
        )
    return keys


def check_references(
    curve: AuditCurve,
    references: Sequence[ReferencePoint],
    *,
    fitted_source_ids: Sequence[str],
) -> list[dict]:
    """Reference checks explicitly separate algebraic controls from independence.

    QG target activities/effective responses can never serve as independent
    absolute-efficiency validation, even when held out of a curve's fit.
    """
    allowed = {
        "implementation_control",
        "source_export_holdout",
        "independent_calibration",
        "quantumgold_target_activity",
        "report_effective",
    }
    rows = []
    for point in references:
        if point.origin not in allowed or not point.source_id or not point.reference_id:
            raise ValueError("Reference origin and source identity required")
        evaluated = curve.at(point.energy_keV, point.geometry)
        control = known_value_control(
            (
                evaluated["efficiency_fraction"]
                if evaluated["efficiency_fraction"] is not None
                else math.nan
            ),
            point.efficiency_fraction,
            relative_tolerance=point.relative_tolerance,
        )
        source_keys = _source_identity_keys((point.source_id,))
        curve_reused = bool(
            source_keys & _source_identity_keys(curve.identity.source_refs)
        )
        reused = bool(source_keys & _source_identity_keys(fitted_source_ids))
        certificate_keys = (
            _source_identity_keys((point.certificate_ref,))
            if point.certificate_ref
            else set()
        )
        certificate_bound = any(
            k.startswith("sha256:") for k in source_keys & certificate_keys
        )
        # This establishes declared independence of the point, not certificate
        # authenticity, calibration uncertainty or whole-curve qualification.
        independent = (
            point.origin == "independent_calibration"
            and not curve_reused
            and not reused
            and certificate_bound
        )
        independence_status = (
            "same_curve_source"
            if curve_reused
            else (
                "fitted_source_reused"
                if reused
                else (
                    "control_only_origin"
                    if point.origin != "independent_calibration"
                    else (
                        "missing_or_unbound_certificate_identity"
                        if not certificate_bound
                        else "separately_identified_certificate_declared"
                    )
                )
            )
        )
        rows.append(
            {
                "reference_id": point.reference_id,
                "source_id": point.source_id,
                "origin": point.origin,
                "fitted_source_reused": reused,
                "curve_source_reused": curve_reused,
                "certificate_ref": point.certificate_ref,
                "independence_status": independence_status,
                "independent_absolute_validation": independent,
                "independent_validation_pass": independent and control["passed"],
                "control": control,
                "evaluation": evaluated,
                "qualification": (
                    "pointwise control only; " "does not qualify the entire curve"
                ),
            }
        )
    return rows
