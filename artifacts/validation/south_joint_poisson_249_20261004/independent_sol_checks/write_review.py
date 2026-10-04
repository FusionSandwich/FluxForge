"""Package the independent read-only software review evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent.parent
p = json.loads((ART / "accepted/pilot.json").read_text())
oracles = json.loads((ART / "independent_sol_checks/oracles_accepted.json").read_text())
def sha(path, lf=False):
    payload = path.read_bytes()
    return hashlib.sha256(payload.replace(b"\r\n", b"\n") if lf else payload).hexdigest()
pins = {name: sha(ROOT / name, True) for name in p["engine"]["files_sha256"]}
assert pins == p["engine"]["files_sha256"]
# Recompute the length-prefixed whole engine receipt without calling the driver.
paths = [ROOT / "pyproject.toml", ROOT / "examples/RAFM_irradiation/run_portable_qg_example.py", ROOT / "examples/RAFM_irradiation/run_integrated_qg_methods.py"]
paths += [q for q in (ROOT / "src/fluxforge").rglob("*") if q.is_file() and "__pycache__" not in q.parts and q.suffix not in (".pyc", ".pyo")]
digest = hashlib.sha256()
for path in sorted(paths, key=lambda q: q.relative_to(ROOT).as_posix()):
    name, data = path.relative_to(ROOT).as_posix().encode(), path.read_bytes()
    digest.update(len(name).to_bytes(4, "big") + name + len(data).to_bytes(8, "big") + data)
assert digest.hexdigest() == p["complete_engine"]["source_sha256"]
assert len(paths) == p["complete_engine"]["file_count"]
assert all(oracles[k] for k in ("replay_pilot_json_byte_identical", "replay_output_hashes_identical", "declared_output_hashes_valid", "current_input_hashes_match_receipts", "xcom_in_declared_engine_receipt"))
tests = ET.parse(ART / "independent_sol_final_tests.xml").getroot().find("testsuite").attrib
assert tests["tests"] == "91" and tests["failures"] == tests["errors"] == "0"
attrs = subprocess.run(["git", "check-attr", "text", "--", *[(ART / q).relative_to(ROOT).as_posix() for q in ("accepted/pilot.json", "accepted/comparison.csv", "accepted/sample_residuals.png", "accepted_tests.xml", "independent_sol_review.json")]], cwd=ROOT, text=True, capture_output=True, check=True).stdout
assert attrs.count(": text: unset") == 5
nominal = [r for r in p["rows"] if r["response"] == "nominal" and r["scenario"] in ("south_native", "ambient_off")]
fixed = [r for r in p["rows"] if r["scenario"] in ("south_native", "ambient_off")]
free = [r for r in p["rows"] if r["scenario"] == "south_free_normalization"]
assert len(fixed) == 24 and all(r["joint"]["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit" for r in fixed)
assert len(free) == 2 and all(r["joint"]["status"] == "unidentifiable" and r["joint"]["count_average_activity_bq"] is None and r["joint"]["profile"] is None for r in free)
with (ART / "accepted/comparison.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len(p["rows"])
for j, r in zip(p["rows"], rows):
    assert j["joint"]["status"] == r["status"]
    if j["joint"]["count_average_activity_bq"] is None:
        assert r["conditional_activity_bq"] == ""
    else:
        assert j["joint"]["count_average_activity_bq"] == float(r["conditional_activity_bq"])
synthetics = [dict(name=r["name"], truth_area=r["truth_area"], candidate_area=r["joint"]["candidate_area"], status=r["joint"]["status"], adequacy=r["joint"]["model_diagnostics"]["adequacy_flag"], generation=r["generation"]) for r in p["synthetic_challenges"]]
report = dict(
    schema="fluxforge-independent-sol-software-review-v1",
    issue=249,
    reviewer="independent Sol subagent; implementation read-only",
    reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    verdict="PASS_BOUNDED_SOFTWARE_REVIEW",
    implementation_revision=p["engine"]["revision"],
    required_integration_ancestor=p["engine"]["required_ancestor"],
    script_canonical_lf_sha256=pins["examples/RAFM_irradiation/south_joint_poisson_pilot.py"],
    tests_canonical_lf_sha256=sha(ROOT / "tests/test_south_joint_poisson_pilot.py", True),
    accepted_pilot_byte_sha256=sha(ART / "accepted/pilot.json"),
    accepted_output_hashes=json.loads((ART / "accepted/OUTPUT_HASHES.json").read_text()),
    complete_engine_receipt=p["complete_engine"],
    complete_engine_hash_independently_recomputed=True,
    all_declared_engine_pins_match_current_canonical_source=True,
    runtime=p["runtime"],
    open_software_blockers=[],
    findings=[dict(id="SOL-249-01", priority="P2", status="CORRECTED_AND_INDEPENDENTLY_VERIFIED", original_finding="Early pilot canonical source list omitted src/fluxforge/data/xcom.py, although activity efficiency conversion invokes its attenuation data and interpolation.", resolution="Implementation commit 55f610320758b6f7c3076d3682c1d974c18ce1c8 adds xcom.py, core/calibration.py, pyproject.toml and complete-engine receipt. Final accepted pins and length-prefixed engine hash match independently recomputed hashes; final artifacts replay byte-identically.", evidence=dict(xcom_canonical_sha256=pins["src/fluxforge/data/xcom.py"], accepted_replay_byte_identical=True))],
    independently_verified=oracles,
    tests=dict(command="& C:/Users/Josh/projects/FluxForge/.venv/Scripts/python.exe -m pytest tests/test_south_joint_poisson_pilot.py tests/test_joint_poisson.py tests/test_histogram_background.py tests/test_background_covariance.py -q --junitxml=artifacts/validation/south_joint_poisson_249_20261004/independent_sol_final_tests.xml", result="91 passed in 11.18s", xml_summary=tests, independent_xml_sha256=sha(ART / "independent_sol_final_tests.xml")),
    commands=[
        "$env:PYTHONPATH='src'; & C:/Users/Josh/projects/FluxForge/.venv/Scripts/python.exe examples/RAFM_irradiation/south_joint_poisson_pilot.py --output artifacts/validation/south_joint_poisson_249_20261004/independent_sol_replay",
        "$env:PYTHONPATH='src'; & C:/Users/Josh/projects/FluxForge/.venv/Scripts/python.exe examples/RAFM_irradiation/south_joint_poisson_pilot.py --output artifacts/validation/south_joint_poisson_249_20261004/independent_sol_final_replay",
        "& C:/Users/Josh/projects/FluxForge/.venv/Scripts/python.exe artifacts/validation/south_joint_poisson_249_20261004/independent_sol_checks/review_oracles.py accepted independent_sol_final_replay",
        "& C:/Users/Josh/projects/FluxForge/.venv/Scripts/python.exe artifacts/validation/south_joint_poisson_249_20261004/independent_sol_checks/write_review.py",
    ],
    checks=[
        "Direct regex parsing of all 8192 ASC channel/count pairs matches direct uint32 ANS decoding; every candidate preserves the same integer sample ROI counts.",
        "Direct South ANS float32 calibration and double live/real/timestamp decoding matches receipts. South original native counts total 543427; sample/background live ratio is exactly 9. Real durations remain separate and are not substituted into the Poisson exposure ratio.",
        "Sample grid uses declared current integration nominal profile; original ASC and ANS calibration disagreement is preserved. Native South observation bins and native response integrations remain distinct and unrebinned. Independent midpoint/support selection matches recorded ROI edges and background counts.",
        "Independent pairwise overlap and linear-functional variance oracle matches IEC net and full propagated covariance including ROI/sideband cross terms. South variance differs from diagonal-only approximation at both lines, demonstrating that correlations are actually retained.",
        "IEC control explicitly identifies fixed Covell singlet scope and does not represent full tiered campaign selection. Ambient-off retains local continuum and has no background likelihood or fabricated measured zero counts.",
        "Independent xlogy likelihood oracle and Pearson formula account for every sample and ambient deviance contribution, including failed free-normalization candidates. Zero-mean residual handling is covered by focused tests.",
        "All 24 fixed-response candidates remain converged with strong lack of fit; both unidentifiable normalization candidates retain predictions/parameters while activity and profile are null. CSV preserves failed statuses and blank unavailable conversions.",
        "The 10 baseline nominal/free candidates have exactly unchanged areas and deviances after additional declared sensitivities. No background or response winner is promoted. QG report information is only source/identity context, never a numeric inference target.",
        "Rounded deterministic CDF Asimov challenges cover zero/weak/strong recovery, deliberate shifted/broad response inadequacy, free normalization unidentifiability and optimizer-budget failure. They do not establish calibrated coverage.",
        "Final sample and ambient residual plots visually inspected: visible centroid/core-wing mismatch, separate native ambient bins, full candidate curves and diagnostic labels. Final legend sits outside sample data panels.",
        "Input byte hashes are identical before and after both replays; all accepted output hashes validate. Final accepted JSON, CSV and both PNGs replay byte-for-byte on this runtime and checkout.",
        "Nested artifact .gitattributes verified with git check-attr: accepted JSON/CSV/PNG and top-level JSON/XML have text unset.",
    ],
    scientific_diagnostics=dict(fixed_candidates=len(fixed), strong_lack_of_fit_candidates=len(fixed), unidentifiable_controls=len(free), nominal_candidates=[dict(energy_keV=r["energy_keV"], scenario=r["scenario"], continuum=r["sample_continuum"], deviance=r["joint"]["deviance"], diagnostics=r["joint"]["model_diagnostics"], conditional_activity_bq=r["joint"]["count_average_activity_bq"]) for r in nominal], synthetic_challenges=synthetics),
    qualification=dict(scientific_admission=False, exact_vendor_parity=False, physical_inversion_qualified=False, software_scope="Source-bound opt-in software pilot and diagnostics only; full campaign replay and integration acceptance remain owned by parent.", unknown_calibration_covariance=p["calibration_covariance"], ambient_temporal_applicability=p["ambient_identity"]["temporal_applicability"], unresolved=["All declared physical-data models retain strong lack of fit.", "Later South background temporal applicability remains unresolved.", "Original ASC versus ANS/profile calibration disagreement and unzoned acquisition/report clock conflict remain unresolved.", "Efficiency/calibration covariance is UNKNOWN, not known zero; conditional intervals omit these uncertainties.", "Nominal efficiency/yield conversions exclude summing, attenuation and decay; no QG-target tuning or qualified line combination.", "Rounded deterministic challenges and asymptotic chi-square/profile screens do not calibrate sparse-count coverage or nuisance-boundary tests.", "Ambient-off is a declared sensitivity; no claim of recovered final vendor setting.", "Raw layout/timing decoding is source-bound and corroborated but is not vendor format certification.", "Identical local replay is verified; no separate Git-free relocated pilot was run by this reviewer."]),
    publication_attributes_evidence=attrs,
)
(ART / "independent_sol_review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("verdict", "implementation_revision", "script_canonical_lf_sha256", "accepted_pilot_byte_sha256", "open_software_blockers")}, indent=2))
