"""Focused failure-path and independent arithmetic checks of the study outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

from build_baseline import interpolated_response, refit_qg_activity

root = Path(__file__).resolve().parent


def csv_rows(name):
    with (root / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    raise AssertionError('Invalid request accepted')


provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
for pin in provenance['inputs']:
    assert hashlib.sha256((root / pin['package_path']).read_bytes()).hexdigest() == pin['sha256']
rows = json.loads((root / 'line_response_lookup.json').read_text(encoding='utf-8'))['rows']
timeline = csv_rows('measurement_identity_and_timing.csv')
assert len(timeline) == len({r['ans_sha256'] for r in timeline}) == 32
assert len({r['measurement_id'] for r in timeline}) == 32
ids = {r['measurement_id'] for r in timeline}
assert all(r['measurement_id'] in ids for r in rows)
assert sum(bool(r['qg_report_sha256']) for r in timeline) == 31
assert sum(bool(r['asc_sha256']) for r in timeline) == 30
assert all(r['acquisition_start_QG_unzoned'] for r in timeline)
assert not any(r['pooled_curve_anchor'] for r in rows if r['measurement_id']=='Fe-Cd-RAFM-1')
assert not any(r['pooled_curve_anchor'] for r in rows if r['rad_int_printed'] is not None and r['rad_int_printed']<=1)
assert not any(r['measurement_id']=='RAFM-A-2hr' for r in rows)
for r in rows:
    if r['fallback_status'] == 'same_count_qg_anchor':
        # Counterexample: a doubled new area must double activity, not silently
        # replay the original or apply count-window decay a second time.
        target = r['qg_line_activity_uCi'] * 37000
        assert math.isclose(refit_qg_activity(r, 2*r['net_counts']), 2*target, rel_tol=1e-12)
        assert math.isclose(refit_qg_activity(r, -r['net_counts']), -target, rel_tol=1e-12)
        rejects(refit_qg_activity, r, float('nan'))
    else:
        rejects(refit_qg_activity, r, 10)
fixture = [dict(energy_keV=50, efficiency_fraction=-.1),
           dict(energy_keV=60, efficiency_fraction=.2),
           dict(energy_keV=70, efficiency_fraction=.4)]
rejects(interpolated_response, fixture, 55)
rejects(interpolated_response, fixture, 49)
rejects(interpolated_response, fixture, 71)
rejects(interpolated_response, fixture, float('nan'))
assert math.isclose(interpolated_response(fixture, 65), .3)
sc48 = csv_rows('sc48_corrected_yield_scenario.csv')
assert len(sc48) == 12
for r in sc48:
    assert math.isclose(float(r['simple_decay_count_start_activity_Bq']),
                        float(r['count_average_activity_Bq'])*float(r['simple_decay_start_to_average_factor']), rel_tol=1e-12)
    if r['measurement_id']=='Ti-RAFM-1' and r['energy_keV']=='983.5':
        assert math.isclose(float(r['count_average_activity_Bq']), 242.63256611529454, rel_tol=1e-10)
        assert float(r['preserved_QG_line_activity_Bq'])/float(r['count_average_activity_Bq']) > 90
scenarios = csv_rows('irradiation_timing_scenarios.csv')
for mid in ('RAFM-A-15d','RAFM-B-15d','RAFM-C-15d','RAFM-N-15d'):
    matches = [r for r in scenarios if r['measurement_id']==mid]
    assert len(matches)==2
    cooling = sorted(float(r['cooling_to_count_start_s']) for r in matches)
    assert cooling[1]-cooling[0] == 77280  # 21 h 28 min date conflict retained.
receipt = dict(status='PASS', scope='working_comparison_failure_paths_and_source_identity',
               input_files_hash_checked=len(provenance['inputs']), physical_counts=32,
               no_geometry_or_intensity_exclusion_bypass=True, no_silent_timing_shift=True,
               no_claim_of_independent_calibration=True)
(root/'VERIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
