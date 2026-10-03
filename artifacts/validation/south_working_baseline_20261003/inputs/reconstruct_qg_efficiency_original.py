"""Recover effective line efficiencies from reported QG net areas/activities.

This is an algebraic reconstruction of software output, not independent detector
calibration. Original values and source bytes are unchanged. The decay correction
is an explicitly hypothetical uniform live-fraction, count-start reference model.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
study = root.parents[2] / "inl_rafm_2025"
analysis = study / "analysis"
reconciliation = study / "data/measurement_reconciliation_2026-09-16"
paths = {"roi": reconciliation/"report_roi_rows.csv", "summary": reconciliation/"reported_summary_rows.json",
         "provenance": reconciliation/"provenance.json", "matrix": analysis/"count_metadata_matrix_32_2026-09-27.json",
         "efficiency": Path(r"C:\Users\joshu\Downloads\eff.csv")}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = {str(p):sha(p) for p in paths.values()}
matrix = json.loads(paths["matrix"].read_text())
assert hashes[str(paths["efficiency"])] == matrix["inputs_sha256"]["eff.csv"]
counts = {r["measurement_id"]:r for r in matrix["counts"]}
summaries = {(r["measurement_id"],r["nuclide"]):r for r in json.loads(paths["summary"].read_text())}
source_by_sha = {r["sha256"]:r for r in json.loads(paths["provenance"].read_text())["sources"]}
roi = list(csv.DictReader(paths["roi"].open(encoding="utf-8",newline="")))
for digest in set(r["report_sha256"] for r in roi):
    p = Path(source_by_sha[digest]["path"])
    assert sha(p) == digest
    hashes[str(p)] = digest
eff = []
for r in csv.reader(paths["efficiency"].open(newline="")):
    try:
        energy, value = float(r[0]),float(r[1])
        eff.append((energy,value))
    except (ValueError,IndexError):
        pass
eff = np.array(eff)
assert np.all(np.diff(eff[:,0])>0)
unit_s = {"s":1., "m":60., "h":3600., "d":86400., "a":365.25*86400.}
rows = []
for r in roi:
    count = counts[r["measurement_id"]]
    h = count["QG_report_header"]
    record = {"measurement_id":r["measurement_id"], "nuclide":r["nuclide"],
              "report_path":source_by_sha[r["report_sha256"]]["path"], "report_sha256":r["report_sha256"],
              "line_number":int(r["line_number"]), "original_line":r["original_line"],
              "detector":h["detector_id"], "distance_cm_printed":h["source_distance_cm_printed"],
              "live_s":h["live_s"], "real_s":h["real_s"], "activity_reference_printed":h["activity_reference_printed"],
              "net_counts":float(r["net_counts"]) if r["net_counts"] else None,
              "line_activity_uCi":float(r["roi_activity_uCi"]) if r["roi_activity_uCi"] else None,
              "rad_int_printed":float(r["rad_int_printed"]) if r["rad_int_printed"] else None,
              "energy_keV":float(r["assignment_energy_keV"]) if r["assignment_energy_keV"] else None,
              "scientific_admission":False}
    vals = [record["net_counts"], record["line_activity_uCi"], record["rad_int_printed"], h["live_s"], record["energy_keV"]]
    if any(v is None or not math.isfinite(v) or v<=0 for v in vals):
        record["status"] = "not_reconstructed_nonpositive_or_missing_input"
        rows.append(record)
        continue
    net, activity, intensity, live, energy = vals
    if intensity > 100:
        record["status"] = "not_reconstructed_probability_outside_percent_range"
        rows.append(record)
        continue
    probability = intensity/100
    implied = net/(activity*37000*live*probability)
    record.update(status="conditional_effective_efficiency", probability_convention="printed RAD INT/100, unverified",
                  effective_fraction=implied,
                  alternative_if_RAD_INT_is_fraction=(implied/100 if intensity<=1 else None))
    summary = summaries.get((r["measurement_id"],r["nuclide"]))
    if summary and summary["report_half_life_unit"] in unit_s:
        half_s = summary["report_half_life_value"]*unit_s[summary["report_half_life_unit"]]
        x = math.log(2)*h["real_s"]/half_s
        factor = -math.expm1(-x)/x
        record.update(half_life_s_printed=half_s, hypothetical_start_activity_average_decay=factor,
                      hypothetical_start_activity_intrinsic_fraction=implied/factor)
    same_coeff = all(count.get("candidate_C1_C4_A_within_QG_printed_rounding",{}).values())
    nominal = (h["detector_id"]=="South" and h["source_distance_cm_printed"]==25 and same_coeff)
    record["candidate_curve_nominal_comparison"] = nominal
    if eff[0,0] <= energy <= eff[-1,0]:
        candidate_percent = float(np.interp(energy, eff[:,0],eff[:,1]))
        record["candidate_csv_efficiency_printed"] = candidate_percent
        if candidate_percent > 0:
            record["ratio_to_candidate_if_csv_percent"] = implied/(candidate_percent/100)
            if "hypothetical_start_activity_intrinsic_fraction" in record:
                record["hypothetical_decay_corrected_ratio_to_candidate"] = record["hypothetical_start_activity_intrinsic_fraction"]/(candidate_percent/100)
    rows.append(record)
eligible = [r for r in rows if r["status"]=="conditional_effective_efficiency"]
long_lived = [r for r in eligible if r["candidate_curve_nominal_comparison"] and r.get("hypothetical_start_activity_average_decay",0)>.99 and "ratio_to_candidate_if_csv_percent" in r and r["rad_int_printed"]>1]
out = {"schema":"qg-effective-efficiency-reconstruction-v1", "script_sha256":sha(__file__), "input_sha256":hashes,
       "rows":rows, "summary":{"roi_rows":len(rows), "reconstructed_rows":len(eligible),
       "nominal_long_lived_comparison_rows":len(long_lived),
       "nominal_long_lived_median_ratio":float(np.median([r["ratio_to_candidate_if_csv_percent"] for r in long_lived]))},
       "formula":"epsilon_effective=N_net/[37000*A_line_uCi*t_live*(RAD_INT_printed/100)]",
       "scientific_admission":False, "limitations":[
           "Reconstructed from QG-derived activity and the same reported net counts; agreement is algebraic, not independent calibration.",
           "Effective efficiency combines intrinsic efficiency, activity reference and count-time/processing corrections.",
           "Alternative intrinsic efficiency assumes count-start activity, simple decay, uniform live fraction and printed half-life; no feeding.",
           "RAD INT=1 may use an unknown display/library convention; both interpretations retained, no activity correction.",
           "Unknown calibration source/certificate/covariance/geometry and processing semantics remain unknown.",
           "No efficiency uncertainty inferred from correlated net/activity fields or from cross-line scatter.",
           "Candidate percent units and applicability are hypotheses; Fe-Cd geometry kept separate."]}
target = root/"QG_EFFECTIVE_EFFICIENCY.json"
if target.exists():
    raise FileExistsError("Preserve receipt; choose new output label for a changed rerun")
target.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out["summary"]))
for reaction_sample in ["Co-Cd-RAFM-1", "Sc-RAFM-1", "Fe-Cd-RAFM-1"]:
    print(json.dumps({"sample":reaction_sample,"lines":[{k:r.get(k) for k in ["energy_keV","effective_fraction","ratio_to_candidate_if_csv_percent","hypothetical_decay_corrected_ratio_to_candidate","candidate_curve_nominal_comparison"]} for r in eligible if r["measurement_id"]==reaction_sample]}))
