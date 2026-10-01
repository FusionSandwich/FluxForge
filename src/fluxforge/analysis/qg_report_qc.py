"""Report-only gamma-yield QC; hypotheses never correct activities or rates."""

import hashlib
import json
import math

from fluxforge.data.rafm_decay import DATA_PATH, get_rafm_decay_entry


def qg_yield_diagnostic(
    isotope, energy_keV, reported_value, *, match_tolerance_keV=1.0
):
    """Compare explicit percent/fraction hypotheses to a nearby bundled line.

    RAD INT alone does not declare its unit. Agreement with a bundled line
    cannot establish the vendor library convention or validate calibration.
    """
    if not math.isfinite(match_tolerance_keV) or match_tolerance_keV <= 0:
        raise ValueError("Line match tolerance must be finite and positive")
    raw = float(reported_value)
    out = {
        "reference_rad_int_reported_value": raw,
        "reference_rad_int_reported_unit": "unspecified",
        "reference_rad_int_percent_assumption_fraction": raw / 100.0,
        "reference_rad_int_fraction_assumption": raw,
        "yield_qc_status": "reference_unavailable",
        "yield_qc_report_only": True,
        "scientific_admission": False,
        "source_qualification": "vendor gamma library, calibration and history unqualified",
        "bundled_line_match_tolerance_keV": match_tolerance_keV,
    }
    if not math.isfinite(raw) or raw <= 0 or not math.isfinite(energy_keV):
        out["yield_qc_status"] = "invalid_reported_yield_or_energy"
        return out
    entry = get_rafm_decay_entry(isotope)
    lines = [] if entry is None else entry.get("gamma_lines", [])
    matches = sorted(lines, key=lambda line: abs(line["energy_keV"] - energy_keV))
    if not matches or abs(matches[0]["energy_keV"] - energy_keV) > match_tolerance_keV:
        return out
    if len(matches) > 1 and math.isclose(
        abs(matches[0]["energy_keV"] - energy_keV),
        abs(matches[1]["energy_keV"] - energy_keV),
        abs_tol=1e-9,
    ):
        out["yield_qc_status"] = "ambiguous_reference_line"
        return out
    line = matches[0]
    expected = float(line["intensity"])
    uncertainty = float(line.get("intensity_uncertainty", 0.0))
    # A reporting threshold, not a statistical significance test: report-library
    # uncertainty and calibration are unknown. Retain its explicit definition.
    tolerance = max(0.05 * expected, 3 * uncertainty)
    percent_matches = abs(raw / 100.0 - expected) <= tolerance
    fraction_matches = abs(raw - expected) <= tolerance
    source_bytes = DATA_PATH.read_bytes()
    metadata = json.loads(source_bytes).get("_metadata", {})
    out.update(
        {
            "bundled_line_energy_keV": float(line["energy_keV"]),
            "bundled_line_energy_delta_keV": float(energy_keV - line["energy_keV"]),
            "bundled_emission_probability": expected,
            "bundled_intensity_uncertainty": uncertainty,
            "bundled_intensity_unit": "photons/parent_decay",
            "bundled_decay_data_path": str(DATA_PATH.resolve()),
            "bundled_decay_data_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "bundled_decay_data_origin": metadata.get("origin"),
            "bundled_uncertainty_source": metadata.get("line_uncertainty_source"),
            "yield_qc_tolerance_fraction": tolerance,
            "yield_qc_tolerance_definition": "max(5% of bundled intensity, 3 * bundled uncertainty); reporting heuristic",
            "percent_assumption_matches_bundled": percent_matches,
            "fraction_assumption_matches_bundled": fraction_matches,
            "yield_qc_status": (
                "yield_convention_discrepancy"
                if fraction_matches and not percent_matches
                else (
                    "consistent_with_percent_assumption"
                    if percent_matches and not fraction_matches
                    else (
                        "ambiguous_convention"
                        if percent_matches
                        else "yield_value_discrepancy"
                    )
                )
            ),
        }
    )
    return out
