"""Independent source and linear-functional oracle; does not invoke pilot helpers."""
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ART = OUT.parent
source_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
replay_dir = sys.argv[2] if len(sys.argv) > 2 else "independent_sol_replay"
p = json.loads((ART / source_dir / "pilot.json").read_text())
replay = json.loads((ART / replay_dir / "pilot.json").read_text())
manifest_path = ROOT / "examples/RAFM_irradiation/quantumgold_reference/manifest.json"
if not manifest_path.exists():
    candidates = list((ROOT / "examples/RAFM_irradiation/quantumgold_reference").glob("*.json"))
    manifest_path = next(q for q in candidates if "measurements" in json.loads(q.read_text()))
manifest = json.loads(manifest_path.read_text())
row = next(r for r in manifest["measurements"] if r["measurement_id"] == "Co-Cd-RAFM-1")
asc = (ROOT / row["files"]["ASC"]).read_text()
pairs = [(int(a), int(b)) for a, b in re.findall(r"^\s*(\d+)\s+(\d+)\s*$", asc, re.M)]
assert [a for a, _ in pairs] == list(range(8192))
sc = np.asarray([b for _, b in pairs], dtype=float)
ans = (ROOT / row["files"]["ANS"]).read_bytes()
assert np.array_equal(sc, struct.unpack_from("<8192I", ans, 1548))
bgblob = (ROOT / p["ambient_identity"]["source_path"]).read_bytes()
bc = np.asarray(struct.unpack_from("<8192I", bgblob, 1548), dtype=float)
polynomial = np.asarray(struct.unpack_from("<3f", bgblob, 424))
bg_real, bg_live = [struct.unpack_from("<d", bgblob, n)[0] for n in (96, 104)]
bg_time = datetime(1899, 12, 30) + timedelta(days=struct.unpack_from("<d", bgblob, 80)[0])
sample_live = float(re.search(r"Elapsed Live Time:\s*([\d.]+)", asc).group(1))
sample_real = float(re.search(r"Elapsed Real Time:\s*([\d.]+)", asc).group(1))
assert polynomial.tolist() == p["ambient_identity"]["used_energy_polynomial_keV"]
assert [bg_real, bg_live] == [p["ambient_identity"]["real_time_s"], p["ambient_identity"]["live_time_s"]]
assert bg_time.isoformat() == p["ambient_identity"]["start_time_unzoned"]
assert [sample_real, sample_live] == [p["sample_identity"]["real_time_s"], p["sample_identity"]["live_time_s"]]
channels = np.arange(8192, dtype=float)
sa = np.asarray(p["sample_identity"]["used_energy_polynomial_keV"])
se = sa[0] + sa[1] * channels + sa[2] * channels ** 2
be = polynomial[0] + polynomial[1] * channels + polynomial[2] * channels ** 2
def edges(centers):
    return np.concatenate(([centers[0] - (centers[1] - centers[0]) / 2], (centers[1:] + centers[:-1]) / 2, [centers[-1] + (centers[-1] - centers[-2]) / 2]))
es, eb = edges(se), edges(be)
config = json.loads((ROOT / "examples/RAFM_irradiation/metadata/workflow_config.json").read_text())
scale = sample_live / bg_live
checks = []
for candidate in p["rows"]:
    lo, hi = candidate["roi_sample_channels_inclusive"]
    assert np.array_equal(sc[lo:hi + 1], candidate["sample_original_counts"])
    if "sample_native_edges_keV" in candidate:
        np.testing.assert_array_equal(es[lo:hi + 2], candidate["sample_native_edges_keV"])
    if candidate["scenario"] != "ambient_off":
        blo, bhi = candidate["roi_ambient_channels_inclusive"]
        native_selection = np.flatnonzero((eb[:-1] < es[hi + 1]) & (eb[1:] > es[lo]))
        assert [int(native_selection[0]), int(native_selection[-1])] == [blo, bhi]
        np.testing.assert_array_equal(bc[blo:bhi + 1], candidate["ambient_original_counts"])
        if "ambient_native_edges_keV" in candidate:
            np.testing.assert_array_equal(eb[blo:bhi + 2], candidate["ambient_native_edges_keV"])
        assert candidate["joint"]["exposure_scale"] == scale * candidate["joint"]["normalization"]
    fit = candidate["joint"]
    y, mu = np.asarray(candidate["sample_original_counts"]), np.asarray(fit["sample_expected"])
    ds = 2 * np.sum(xlogy(y, y / mu) - y + mu)
    np.testing.assert_allclose(ds, fit["model_diagnostics"]["sample_poisson_deviance"], rtol=1e-10)
    np.testing.assert_allclose((y - mu) / np.sqrt(mu), fit["sample_poisson_residuals"]["pearson"])
    if candidate["scenario"] != "ambient_off":
        yb, mub = np.asarray(candidate["ambient_original_counts"]), np.asarray(fit["ambient_expected"])
        db = 2 * np.sum(xlogy(yb, yb / mub) - yb + mub)
        np.testing.assert_allclose(db, fit["model_diagnostics"]["background_poisson_deviance"], rtol=1e-10)
    if candidate.get("iec") and candidate["response"] == "nominal" and candidate["sample_continuum"] == "linear":
        center = (lo + hi) // 2
        slope = sa[1] + 2 * sa[2] * center
        gap = int(round(config["flux_wire_background_gap_fwhm"] * candidate["fixed_fwhm_keV"] / slope))
        width = config["flux_wire_background_width_channels"]
        target = np.r_[np.arange(lo - gap - width, lo - gap), np.arange(lo, hi + 1), np.arange(hi + gap + 1, hi + gap + width + 1)]
        weights = np.r_[np.full(width, -(hi - lo + 1) / (2 * width)), np.ones(hi - lo + 1), np.full(width, -(hi - lo + 1) / (2 * width))]
        # Independent vectorized pairwise overlap, avoiding engine W implementation.
        overlap = np.maximum(0, np.minimum(es[target + 1, None], eb[None, 1:]) - np.maximum(es[target, None], eb[None, :-1])) / np.diff(eb)
        transported = weights @ overlap
        net = weights @ sc[target]
        variance = weights ** 2 @ sc[target]
        diagonal_only_variance = variance
        if candidate["scenario"] == "south_native":
            net -= scale * (transported @ bc)
            variance += scale ** 2 * (transported ** 2 @ bc)
            diagonal_only_variance += scale ** 2 * (weights ** 2 @ ((overlap ** 2) @ bc))
        np.testing.assert_allclose(net, candidate["iec"]["net"], rtol=1e-12)
        np.testing.assert_allclose(variance, candidate["iec"]["std_full_covariance"] ** 2, rtol=1e-12)
        checks.append(dict(energy_keV=candidate["energy_keV"], scenario=candidate["scenario"], roi=[lo, hi], roi_original_counts=int(sc[lo:hi + 1].sum()), independent_net=float(net), independent_full_covariance_variance=float(variance), diagonal_only_variance=float(diagonal_only_variance)))
baseline = json.loads((ART / "baseline/pilot.json").read_text())
baseline_matches = []
for row0 in baseline["rows"]:
    current = next(r for r in p["rows"] if all(r[k] == row0[k] for k in ("energy_keV", "scenario", "sample_continuum")) and r["response"] == "nominal")
    baseline_matches.append(dict(energy_keV=row0["energy_keV"], scenario=row0["scenario"], continuum=row0["sample_continuum"], area_unchanged=current["joint"]["candidate_area"] == row0["joint"]["candidate_area"], deviance_unchanged=current["joint"]["deviance"] == row0["joint"]["deviance"]))
input_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in p["inputs_sha256"]}
assert input_hashes == p["inputs_sha256"] == replay["inputs_sha256"]
output_hashes_valid = all(hashlib.sha256((ART / source_dir / name).read_bytes()).hexdigest() == expected for name, expected in json.loads((ART / source_dir / "OUTPUT_HASHES.json").read_text()).items())
assert output_hashes_valid
result = dict(status="INDEPENDENT_ORACLES_PASSED", source_dir=source_dir, replay_dir=replay_dir, original_ASC_channels=8192, ASC_ANS_full_count_array_equal=True, background_sha256=hashlib.sha256(bgblob).hexdigest(), raw_background_total=int(bc.sum()), raw_background_polynomial=polynomial.tolist(), raw_background_live_s=bg_live, raw_background_real_s=bg_real, raw_background_start_unzoned=bg_time.isoformat(), raw_sample_live_s=sample_live, raw_sample_real_s=sample_real, live_exposure_scale=scale, controls=checks, baseline_nominal_matches=baseline_matches, all_current_rows_checked=len(p["rows"]), current_input_hashes_match_receipts=True, declared_output_hashes_valid=True, replay_pilot_json_byte_identical=(ART / source_dir / "pilot.json").read_bytes() == (ART / replay_dir / "pilot.json").read_bytes(), replay_output_hashes_identical=json.loads((ART / source_dir / "OUTPUT_HASHES.json").read_text()) == json.loads((ART / replay_dir / "OUTPUT_HASHES.json").read_text()), xcom_hash_current=hashlib.sha256((ROOT / "src/fluxforge/data/xcom.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(), xcom_in_declared_engine_receipt="src/fluxforge/data/xcom.py" in p["engine"]["files_sha256"])
(OUT / ("oracles_" + source_dir + ".json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
