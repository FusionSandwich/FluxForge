"""Explicit acquisition settings for instructional Qt report bundles."""

from __future__ import annotations

from copy import deepcopy
import math


SETTING_FIELDS = (
    "amplifier_gain",
    "shaping_time_us",
    "high_voltage_v",
    "count_geometry",
)


def instrument_provenance(workspace, overrides=None, *, require_complete=False):
    """Record settings for every loaded acquisition, without inventing defaults.

    Imported settings use the documented `metadata.instrument_settings` keys.
    Overrides are explicit user entries keyed by canonical workspace spectrum ID.
    Timing and MCA coefficients always come from the recorded spectrum itself.
    """
    overrides = overrides or {}
    records = []
    missing = []
    for record in workspace.get("spectra", []):
        spectrum_id = record["spectrum_id"]
        spectrum = record["spectrum"]
        imported = spectrum.get("metadata", {}).get("instrument_settings", {})
        if not isinstance(imported, dict):
            raise ValueError("metadata.instrument_settings must be a mapping.")
        user = overrides.get(spectrum_id, {})
        fields = {}
        absent = []
        for name in SETTING_FIELDS:
            entered = user.get(name)
            explicit_user = entered is not None and str(entered).strip() != ""
            value = entered if explicit_user else imported.get(name)
            source = "user_entered" if explicit_user else "recorded_metadata"
            if value is None or str(value).strip() == "":
                value = None
                source = "unavailable"
                absent.append(name)
            elif name == "count_geometry":
                if not isinstance(value, str):
                    raise ValueError("Count geometry must be a recorded description.")
                value = value.strip()
            else:
                if isinstance(value, bool):
                    raise ValueError(f"{name} must be a finite numeric value.")
                try:
                    value = float(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{name} must be a finite numeric value.") from exc
                if not math.isfinite(value) or (
                    name != "high_voltage_v" and value <= 0
                ):
                    raise ValueError(f"Invalid {name} for spectrum {spectrum_id}.")
            fields[name] = {"value": value, "source": source}
        live = spectrum.get("live_time", 0)
        real = spectrum.get("real_time", 0)
        for name, value in (("live_time_s", live), ("real_time_s", real)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Invalid recorded {name}.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid recorded {name}.")
            if value == 0:
                absent.append(name)
            fields[name] = {
                "value": value if value > 0 else None,
                "source": "recorded_spectrum" if value > 0 else "unavailable",
            }
        if live > real:
            absent.append("counting_time_order")
        calibration = deepcopy(spectrum.get("calibration", {}))
        coefficients = calibration.get("energy")
        if coefficients is None or coefficients == []:
            absent.append("mca_calibration")
        elif not isinstance(coefficients, list) or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in coefficients
        ):
            raise ValueError("MCA energy coefficients must be a finite numeric list.")
        fields["mca_calibration"] = {
            "value": calibration or None,
            "source": "recorded_spectrum" if coefficients else "unavailable",
            "energy_unit": "keV",
            "coefficient_order": "ascending_powers_of_channel",
        }
        records.append(
            {
                "spectrum_id": spectrum_id,
                "label": record.get("label", ""),
                "source_path": record.get("source_path"),
                "source_hash": record.get("source_hash"),
                "detector_id": spectrum.get("detector_id", ""),
                "settings": fields,
                "missing_required": absent,
            }
        )
        missing.extend(f"{spectrum_id}:{name}" for name in absent)
    if not records:
        missing.append("loaded_spectrum")
    if require_complete and missing:
        raise ValueError(
            "Instructional report needs recorded settings: " + ", ".join(missing)
        )
    return {
        "schema": "fluxforge.instrument_provenance.v1",
        "scientific_admission": False,
        "units": {
            "amplifier_gain": "dimensionless",
            "shaping_time_us": "microseconds",
            "high_voltage_v": "volts",
            "live_time_s": "seconds",
            "real_time_s": "seconds",
        },
        "complete_required_settings": not missing,
        "missing_required": missing,
        "records": records,
    }


def validate_instructional_snapshot(snapshot):
    """Require a complete, reproducible settings record for instructional exports."""
    if not snapshot.get("instructional_report", False):
        return
    recorded = snapshot.get("instrument_provenance", {})
    overrides = {}
    for record in recorded.get("records", []):
        overrides[record["spectrum_id"]] = {
            name: record["settings"][name]["value"]
            for name in SETTING_FIELDS
            if record["settings"].get(name, {}).get("source") == "user_entered"
        }
    expected = instrument_provenance(
        snapshot["workspace"], overrides, require_complete=True
    )
    if recorded != expected:
        raise ValueError("Instrument provenance does not match the recorded workspace.")
