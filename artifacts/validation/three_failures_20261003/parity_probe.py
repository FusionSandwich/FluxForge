"""Independent JSON-level perturbations; never changes fixture tolerances."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from fluxforge.validation import reference_parity as parity

repo = Path(__file__).resolve().parents[3]
case = repo / "tests/spectra/reference_parity/cases/minimal_peak_search_algorithm_case"
manifest = json.loads((case / "manifest.json").read_text())
expected = json.loads((case / "detected_peaks_expected.json").read_text())
actual = parity._run_peak_search_case(case, manifest)["detected_peaks_expected.json"]
checks = []

def compare(name, observed, expected_pass, tolerances=None, reference=None):
    mismatches = parity._compare_json_values(
        expected if reference is None else reference,
        observed,
        path="detected_peaks_expected.json",
        tolerances=manifest["tolerances"] if tolerances is None else tolerances,
    )
    passed = not mismatches
    checks.append({"name":name,"expected_match":expected_pass,"observed_match":passed,"mismatches":mismatches})
    assert passed == expected_pass, name

compare("actual fixture output", actual, True)
compare("actual output without declared tolerance", actual, False, tolerances={})
for field in ("energies_keV", "first_peak_keV"):
    for delta in (-8.001, -8.0, 0.000006, 8.0, 8.001):
        reference = {field: [200.0] if field == "energies_keV" else 200.0}
        observed = {field: [200.0 + delta] if field == "energies_keV" else 200.0 + delta}
        compare(f"{field} delta {delta}", observed, abs(delta) <= 8, reference=reference)
    for suffix, broad, explicit in (("abs",8.0,0.01),("rel",0.1,0.001)):
        for keys in ((f"energy_keV_{suffix}",f"{field}_{suffix}"),(f"{field}_{suffix}",f"energy_keV_{suffix}")):
            values={f"energy_keV_{suffix}":broad,f"{field}_{suffix}":explicit}
            limits={key:values[key] for key in keys}
            compare(f"{field} explicit {suffix} precedence {list(limits)}", {field:[201.0] if field == "energies_keV" else 201.0},False, tolerances=limits,reference={field:[200.0] if field == "energies_keV" else 200.0})
    compare(f"{field} relative alias", {field:[201.0] if field == "energies_keV" else 201.0},True,tolerances={"energy_keV_rel":0.01},reference={field:[200.0] if field == "energies_keV" else 200.0})
observed=copy.deepcopy(expected)
observed["channels"][0]+=0.001
compare("channel perturbation cannot borrow energy tolerance",observed,False)
observed=copy.deepcopy(expected)
observed["energies_keV"].pop()
compare("energy array length is strict",observed,False)
observed=copy.deepcopy(expected)
observed["peak_count"]+=3
compare("peak count beyond declared tolerance",observed,False)
payload={"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),"python":sys.executable,"prefix":sys.prefix,"fixture_tolerances":manifest["tolerances"],"expected":expected,"observed":actual,"source_sha256":hashlib.sha256(Path(parity.__file__).read_bytes()).hexdigest(),"probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"checks":checks,"passed":True}
Path(__file__).with_name("parity_probe.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
print(f"{len(checks)} independent checks passed")
