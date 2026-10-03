"""Source-bound QG identity exceptions; never alter native detections."""

from __future__ import annotations

import math
from typing import Any, Sequence


def corrected_reference_ids(
    report: str,
    rows: Sequence[dict[str, Any]],
    corrections: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy reference rows and apply only exact, source-bound corrections.

    A changed report, line number, original ID or energy invalidates the
    exception instead of silently suppressing a new mismatch.
    """
    result = [dict(row) for row in rows]
    applicable = [item for item in corrections if item["report"] == report]
    seen = set()
    for correction in applicable:
        key = (correction["source_sha256"], correction["source_line_number"])
        if key in seen:
            raise ValueError("Duplicate QG identity correction")
        seen.add(key)
        candidates = [
            row
            for row in result
            if row.get("report_source_sha256") == key[0]
            and row.get("report_source_line_number") == key[1]
            and row["isotope"] == correction["reported_isotope"]
            and math.isclose(
                float(row["energy_keV"]),
                float(correction["reported_energy_keV"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"QG identity correction does not bind exactly one original row: {report}"
            )
        isotope = correction["corrected_isotope"]
        if not isinstance(isotope, str) or not isotope.strip():
            raise ValueError("Corrected isotope must be a nonempty string")
        row = candidates[0]
        row["reported_isotope"] = row["isotope"]
        row["isotope"] = isotope
        row["identity_correction"] = dict(correction)
    return result
