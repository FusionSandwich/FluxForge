"""Source-bound operating-log gate and activation-history Jacobian.

Correspondence chronology or an assumed constant-power interval is not a
complete operating log. Passing this input gate never grants scientific admission.
"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from fluxforge.analysis.flux_unfold import irradiation_history_factor


def _bound_bytes(binding):
    if (
        not isinstance(binding, dict)
        or not binding.get("path")
        or not binding.get("sha256")
    ):
        raise ValueError("Operating log evidence needs path and SHA256")
    payload = Path(binding["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError("Operating log evidence SHA256 mismatch")
    return payload


def load_operating_history(
    binding,
    *,
    expected_end=None,
    expected_segments=None,
    expected_sample=None,
    require_separability=False,
):
    """Verify a complete, time-zone-bound log index and power/rod evidence bytes."""
    data = json.loads(_bound_bytes(binding))
    if (
        data.get("schema") != "fluxforge-operating-history-v1"
        or data.get("coverage_complete") is not True
        or not data.get("source_id")
        or not data.get("power_basis")
    ):
        raise ValueError(
            "A complete irradiation operating log with source/power basis is required"
        )
    start, end = (datetime.fromisoformat(data[k]) for k in ("start", "end"))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Operating log needs ordered timezone-aware start/end")
    if expected_end is not None and (
        expected_end.tzinfo is None or expected_end != end
    ):
        raise ValueError(
            "Operating log EOI disagrees with the qualified schedule/timezone"
        )
    if expected_sample is not None and expected_sample not in data.get(
        "monitor_ids", []
    ):
        raise ValueError("Operating log does not bind the current monitor identity")
    segments = [
        (float(s["duration_s"]), float(s["relative_power"])) for s in data["segments"]
    ]
    if (
        not segments
        or any(
            not math.isfinite(d) or not math.isfinite(p) or d <= 0 or p < 0
            for d, p in segments
        )
        or not any(p > 0 for _, p in segments)
        or not math.isclose(
            sum(d for d, _ in segments),
            (end - start).total_seconds(),
            abs_tol=1e-6,
            rel_tol=0,
        )
    ):
        raise ValueError("Operating log segments must cover the full elapsed interval")
    if (
        expected_segments is not None
        and list(map(tuple, expected_segments)) != segments
    ):
        raise ValueError("Operating log segments disagree with rate history")
    evidence = data.get("evidence", {})
    for name in ("reactor_power", "control_rods"):
        ref = evidence.get(name)
        if not isinstance(ref, dict) or not ref.get("units"):
            raise ValueError(
                "Complete reactor-power and control-rod operating evidence is required"
            )
        _bound_bytes(ref)
    separability = data.get(
        "history_model"
    ) == "separable_local_spectrum" and isinstance(
        data.get("spectrum_shape_evidence"), dict
    )
    if separability:
        _bound_bytes(data["spectrum_shape_evidence"])
    if require_separability and not separability:
        raise ValueError(
            "Physical scalar history requires bound evidence for local spectrum separability"
        )
    reference_power = data.get("reference_power")
    if require_separability and (
        not isinstance(reference_power, dict)
        or not isinstance(reference_power.get("units"), str)
        or not reference_power["units"].strip()
        or type(reference_power.get("value")) not in (int, float)
        or not math.isfinite(reference_power["value"])
        or reference_power["value"] <= 0
    ):
        raise ValueError(
            "Physical relative power requires an explicit positive reference power and units"
        )
    return segments, {
        "source_id": data["source_id"],
        "path": binding["path"],
        "sha256": binding["sha256"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "power_basis": data["power_basis"],
        "reference_power": reference_power,
        "evidence": evidence,
        "monitor_ids": data.get("monitor_ids", []),
        "coverage_complete": True,
        "history_model": data.get("history_model", "separable_local_spectrum_assumed"),
        "local_spectrum_separability_qualified": separability,
        "spectrum_shape_evidence": data.get("spectrum_shape_evidence"),
        "qualification": "byte/coverage input gate; scientific admission separate",
    }


def history_rate_jacobian(half_life_s, segments):
    """d(log R)/d(duration_s[j], relative_power[j]) at fixed EOI activity.

    R = A_EOI / (N S). Half-life/EOI conversion uncertainty is a separate
    component; correlations may be included in a jointly supplied source block.
    """
    segments = [(float(d), float(p)) for d, p in segments]
    saturation = irradiation_history_factor(half_life_s, irradiation_history=segments)
    if not math.isfinite(saturation) or saturation <= 0:
        raise ValueError("Finite positive irradiation history factor required")
    lam = math.log(2) / half_life_s
    later = np.cumsum([d for d, _ in segments][::-1])[::-1] - [d for d, _ in segments]
    basis = [
        -math.expm1(-lam * d) * math.exp(-lam * t) for (d, _), t in zip(segments, later)
    ]
    terms = [p * b for (_, p), b in zip(segments, basis)]
    gradient, names, units = [], [], []
    for j, ((d, p), t, b) in enumerate(zip(segments, later, basis)):
        gradient.extend(
            [
                -lam * (p * math.exp(-lam * (d + t)) - sum(terms[:j])) / saturation,
                -b / saturation,
            ]
        )
        names.extend([f"duration_s[{j}]", f"relative_power[{j}]"])
        units.extend(["s", "relative power"])
    return gradient, names, units
