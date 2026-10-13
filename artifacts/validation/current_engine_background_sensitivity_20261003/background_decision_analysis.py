"""Read-only, source-bound summary of background leverage in the RAFM replay."""

import hashlib
import json
import statistics
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT / "three_scenario_comparison.json"
FORENSICS = ROOT / "background_forensics.json"
ORIGINAL_ANS = (
    ROOT.parents[2]
    / "examples/RAFM_irradiation/quantumgold_reference/originals/ANS/Co-Cd-RAFM-1.ANS"
)
OUTPUT = ROOT / "background_decision_analysis.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
forensics = json.loads(FORENSICS.read_text(encoding="utf-8"))
assert sha256(COMPARISON) == (
    "e35d11061dda3257a8874de0cdebfae59261276f00b30e49ea49d2347771034f"
)
assert sha256(FORENSICS) == (
    "ad52409728f7ab47ae5a26c34099ebf8e239aaf173c8f69946c7a38127905dc2"
)
assert comparison["engine_commit"] == forensics["engine_commit"]
assert comparison["input_tree_sha256"] == forensics["input_tree_sha256"]
assert forensics["sources"]["sample"]["sha256"] == (
    "7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba"
)
assert sha256(ORIGINAL_ANS) == (
    "d04a65ed6052a08b831788a63e89f2eb291b5de4905357de2227e7a845f90ce7"
)

rows = []
for row in comparison["activity_rows"]:
    modes = row["modes"]
    if not all(
        modes[mode]["matched"] and modes[mode]["activity_Bq"] is not None
        for mode in ("north", "south", "zero_ambient")
    ):
        continue
    north = modes["north"]["activity_Bq"]
    south = modes["south"]["activity_Bq"]
    qg = row["QG_report_Bq"]
    assert north > 0 and south > 0 and qg > 0
    rows.append(
        {
            "sample_id": row["sample_id"],
            "isotope": row["isotope"],
            "sample_group": row["sample_group"],
            "south_over_north_minus_one": south / north - 1,
            "south_over_qg_minus_one": south / qg - 1,
        }
    )
assert len(rows) == comparison["triple_common_finite_rows"] == 81
low_leverage = [r for r in rows if abs(r["south_over_north_minus_one"]) < 0.05]
large_leverage = sorted(
    rows, key=lambda r: abs(r["south_over_north_minus_one"]), reverse=True
)

blob = ORIGINAL_ANS.read_bytes()
ans_calibration = struct.unpack_from("<3f", blob, 424)
profile_calibration = forensics["applied_profile_energy_calibration"]
asc_header_calibration = forensics["native_ASC_calibrations"]["CoCd_sample"]
assert max(abs(a - b) for a, b in zip(ans_calibration, profile_calibration)) < 0.0005

result = {
    "status": "READ_ONLY_CONDITIONAL_DIAGNOSTIC",
    "provenance": {
        "comparison_sha256": sha256(COMPARISON),
        "forensics_sha256": sha256(FORENSICS),
        "original_CoCd_ANS_sha256": sha256(ORIGINAL_ANS),
        "engine_commit": comparison["engine_commit"],
        "input_tree_sha256": comparison["input_tree_sha256"],
    },
    "matched_activity_rows": len(rows),
    "median_abs_south_over_north_minus_one": statistics.median(
        abs(r["south_over_north_minus_one"]) for r in rows
    ),
    "rows_with_abs_south_north_shift_below_5pct": len(low_leverage),
    "rows_with_abs_south_north_shift_above_10pct": sum(
        abs(r["south_over_north_minus_one"]) > 0.10 for r in rows
    ),
    "rows_with_abs_south_north_shift_above_20pct": sum(
        abs(r["south_over_north_minus_one"]) > 0.20 for r in rows
    ),
    "low_background_leverage_rows": {
        "count": len(low_leverage),
        "median_abs_south_QG_relative_difference": statistics.median(
            abs(r["south_over_qg_minus_one"]) for r in low_leverage
        ),
        "count_with_abs_south_QG_difference_above_20pct": sum(
            abs(r["south_over_qg_minus_one"]) > 0.20 for r in low_leverage
        ),
    },
    "largest_south_north_shifts": large_leverage[:10],
    "CoCd_source_calibration": {
        "original_ANS_energy_coefficients": ans_calibration,
        "workflow_profile_energy_coefficients": profile_calibration,
        "converted_ASC_header_energy_coefficients": asc_header_calibration,
        "interpretation": "Original ANS supports the workflow profile at printed precision; the converted ASC header differs and is not an independent calibration correction.",
    },
    "qualification": "All differences are descriptive comparisons of the existing conditional engine replay, not evidence that either measured background is physically applicable or that QG activity is a calibration truth.",
}
OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)
