"""Reproduce the user-authorized QG-conditioned South comparison data, stdlib only.

Run: python build_baseline.py. Inputs are byte-pinned in provenance.json.
This study does not change FluxForge's independent-calibration qualification gates.
"""
import bisect
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / 'inputs' / name).read_text(encoding='utf-8'))


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (ROOT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def interpolated_response(points, energy_keV):
    """Linear interpolation only within supplied positive adjacent knots; no extrapolation."""
    energies = [p['energy_keV'] for p in points]
    if not math.isfinite(energy_keV) or not energies or not energies[0] <= energy_keV <= energies[-1]:
        raise ValueError('Energy outside working response range')
    i = bisect.bisect_left(energies, energy_keV)
    if energies[i] == energy_keV:
        indices = [i]
    else:
        indices = [i - 1, i]
    if any(points[j]['efficiency_fraction'] <= 0 for j in indices):
        raise ValueError('Invalid efficiency bracket; do not bridge invalid values')
    if len(indices) == 1:
        return points[i]['efficiency_fraction']
    lo, hi = points[i - 1], points[i]
    t = (energy_keV - lo['energy_keV']) / (hi['energy_keV'] - lo['energy_keV'])
    return lo['efficiency_fraction'] * (1 - t) + hi['efficiency_fraction'] * t


def refit_qg_activity(row, new_net_counts):
    """Same physical count/line only: use its original QG activity-per-net-count anchor.

    Signed new net counts are retained. This returns the QG-reference line activity
    in Bq, not a nuclide-combined or EOI activity, and invents no uncertainty.
    """
    factor = row.get('qg_activity_Bq_per_net_count')
    if row['fallback_status'] != 'same_count_qg_anchor' or factor is None:
        raise ValueError('Line has no accepted working QG anchor')
    if not math.isfinite(new_net_counts):
        raise ValueError('Nonfinite new net counts')
    return factor * new_net_counts


def build():
    provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8'))
    for source in provenance['inputs']:
        p = ROOT / source['package_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == source['sha256'], p
    qg = load('QG_EFFECTIVE_EFFICIENCY.json')
    chronology = load('chronology_join_receipt.json')
    with (ROOT / 'inputs' / 'report_roi_rows.csv').open(encoding='utf-8', newline='') as f:
        original_rois = {(r['measurement_id'], int(r['line_number'])): r for r in csv.DictReader(f)}
    curve = []
    with (ROOT / 'inputs' / 'South Small Vial 25cm.csv').open(encoding='utf-8-sig', newline='') as f:
        for line, r in enumerate(csv.reader(f), 1):
            try:
                e, v = float(r[0]), float(r[1])
            except (ValueError, IndexError):
                continue
            curve.append(dict(energy_keV=e, efficiency_fraction=v / 100,
                              original_efficiency_value=v, source_line=line,
                              status='working_percent_assumption' if v > 0 else 'excluded_nonpositive'))
    assert all(a['energy_keV'] < b['energy_keV'] for a, b in zip(curve, curve[1:]))
    write_csv('south_25cm_recovered_curve.csv', curve)
    rows, groups, ratios = [], defaultdict(list), []
    for r in qg['rows']:
        raw = original_rois[(r['measurement_id'], r['line_number'])]
        valid = r['status'] == 'conditional_effective_efficiency'
        ambiguous = valid and r['rad_int_printed'] <= 1
        # Known Sc48 1.00 display issue and ALL <=1 display ambiguity remain visible.
        accepted = valid and not ambiguous
        status = 'same_count_qg_anchor' if accepted else ('excluded_ambiguous_RAD_INT' if ambiguous else r['status'])
        nominal = bool(r.get('candidate_curve_nominal_comparison'))
        row = dict(measurement_id=r['measurement_id'], nuclide=r['nuclide'],
                   energy_keV=r['energy_keV'], detector=r['detector'],
                   distance_cm=r['distance_cm_printed'], report_sha256=r['report_sha256'],
                   report_line=r['line_number'], net_counts=r['net_counts'],
                   net_error_printed=raw['net_error_printed'],
                   qg_line_activity_uCi=r['line_activity_uCi'], rad_int_printed=r['rad_int_printed'],
                   live_s=r['live_s'], real_s=r['real_s'], fallback_status=status,
                   known_issue=('Sc48_1.00_RAD_INT_display' if ambiguous and r['nuclide']=='Sc48' else ''),
                   effective_response_fraction=r.get('effective_fraction'),
                   count_start_simple_decay_response_fraction=r.get('hypothetical_start_activity_intrinsic_fraction'),
                   qg_activity_Bq_per_net_count=(r['line_activity_uCi'] * 37000 / r['net_counts'] if accepted else None),
                   recovered_curve_fraction=None, ratio_to_recovered_curve=None,
                   pooled_curve_anchor=False, independently_calibrated=False)
        if valid:
            computed = r['net_counts'] / (r['line_activity_uCi'] * 37000 * r['live_s'] * (r['rad_int_printed'] / 100))
            assert math.isclose(computed, r['effective_fraction'], rel_tol=1e-12)
            if nominal:
                try:
                    row['recovered_curve_fraction'] = interpolated_response(curve, r['energy_keV'])
                    row['ratio_to_recovered_curve'] = computed / row['recovered_curve_fraction']
                except ValueError:
                    pass
        # Negligible count-window decay reduces reference-time mixing; arbitrary
        # unknown attenuation/processing corrections are still embedded in the response.
        anchor = accepted and nominal and r.get('hypothetical_start_activity_average_decay', 0) > .99
        row['pooled_curve_anchor'] = anchor
        if anchor:
            groups[r['energy_keV']].append(row)
            if row['ratio_to_recovered_curve'] is not None:
                ratios.append(row['ratio_to_recovered_curve'])
        rows.append(row)
    knots = []
    for energy, values in sorted(groups.items()):
        efficiencies = [r['effective_response_fraction'] for r in values]
        knots.append(dict(energy_keV=energy, efficiency_fraction=statistics.median(efficiencies),
                          n_rows=len(values), minimum=min(efficiencies), maximum=max(efficiencies),
                          measurement_ids=';'.join(sorted({r['measurement_id'] for r in values})),
                          basis='QG_conditioned_long_lived_South_25cm', spread_is_uncertainty=False))
    write_csv('qg_conditioned_response_knots.csv', knots)
    write_json('line_response_lookup.json', {'basis': 'user_authorized_QG_working_reference', 'rows': rows})
    write_csv('line_response_lookup.csv', rows)
    nuclear = load('SC48_QUANTITATIVE_DIAGNOSTIC.json')['nuclear_data']
    sc48 = []
    for r in rows:
        if r['nuclide'] != 'Sc48' or not r['recovered_curve_fraction']:
            continue
        probability = nuclear['central_probabilities'].get(str(r['energy_keV']))
        if probability is None:
            continue
        response = r['recovered_curve_fraction']
        average = r['net_counts'] / (response * probability * r['live_s'])
        x = math.log(2) * r['real_s'] / (nuclear['half_life_h'] * 3600)
        decay = -math.expm1(-x) / x
        sc48.append(dict(measurement_id=r['measurement_id'], energy_keV=r['energy_keV'],
                         report_sha256=r['report_sha256'], report_line=r['report_line'],
                         preserved_QG_line_activity_Bq=r['qg_line_activity_uCi'] * 37000,
                         ENSDF_photons_per_decay=probability, recovered_curve_fraction=response,
                         count_average_activity_Bq=average, simple_decay_count_start_activity_Bq=average / decay,
                         simple_decay_start_to_average_factor=1 / decay,
                         basis='recovered_curve_percent_assumption_ENSDF_central_yield_QG_net',
                         nuclear_data_url=nuclear['url'], independently_calibrated=False))
    write_csv('sc48_corrected_yield_scenario.csv', sc48)
    timeline, scenarios = [], []
    for r in chronology['rows']:
        a, j, identity = r['acquisition'], r['campaign_irradiation_join'], r['identity_join']
        timeline.append(dict(measurement_id=r['measurement_id'], sample_id=r['sample_id'],
                             sample_kind=r['sample_kind'], ans_sha256=identity['ans_sha256'],
                             qg_report_sha256=identity.get('qg_report_sha256'), asc_sha256=identity.get('asc_sha256'),
                             acquisition_start_QG_unzoned=a.get('candidate_start_local_unzoned'),
                             acquisition_end_derived_unzoned=a.get('candidate_end_local_unzoned_derived_from_real_time'),
                             live_s=a.get('live_time_s'), real_s=a.get('real_time_s'),
                             irradiation_start_schedule_unzoned=j.get('candidate_start'),
                             irradiation_end_schedule_unzoned=j.get('candidate_end'),
                             irradiation_join_status=j.get('candidate_status'),
                             acquisition_basis='QG/ANS_header_working_clock_no_automatic_shift',
                             absolute_timezone_assigned=False))
        choices = [('maintained_schedule', j.get('candidate_start'), j.get('candidate_end'))]
        alternative = j.get('correspondence_candidate')
        if alternative:
            choices.append(('RAFM4_advisor_correspondence_preferred_working_scenario',
                            alternative['start_local_unzoned'], alternative['end_local_unzoned']))
        for label, start, end in choices:
            if start and end and a.get('candidate_start_local_unzoned'):
                parse = datetime.fromisoformat
                scenarios.append(dict(measurement_id=r['measurement_id'], scenario=label,
                                      irradiation_start_unzoned=start, EOI_unzoned=end,
                                      irradiation_duration_s=(parse(end)-parse(start)).total_seconds(),
                                      cooling_to_count_start_s=(parse(a['candidate_start_local_unzoned'])-parse(end)).total_seconds(),
                                      clock_assumption='shared_naive_local_clock', inferred_physical_event=False))
    write_csv('measurement_identity_and_timing.csv', timeline)
    write_csv('irradiation_timing_scenarios.csv', scenarios)
    accepted = [r for r in rows if r['fallback_status'] == 'same_count_qg_anchor']
    errors = [abs(refit_qg_activity(r, r['net_counts']) / (r['qg_line_activity_uCi'] * 37000) - 1) for r in accepted]
    summary = dict(schema='south-working-baseline-v1', physical_counts=len(timeline),
                   roi_rows=len(rows), reconstructible_effective_response_rows=sum(r['effective_response_fraction'] is not None for r in rows),
                   same_count_usable_QG_anchors=len(accepted), excluded_ambiguous_RAD_INT=sum(r['fallback_status']=='excluded_ambiguous_RAD_INT' for r in rows),
                   recovered_curve_rows=len(curve), recovered_curve_nonpositive_rows=sum(r['efficiency_fraction']<=0 for r in curve),
                   empirical_anchor_rows=sum(len(v) for v in groups.values()), empirical_knots=len(knots),
                   sc48_corrected_yield_scenario_rows=len(sc48),
                   empirical_energy_range_keV=[knots[0]['energy_keV'], knots[-1]['energy_keV']],
                   median_QG_to_recovered_curve_ratio=statistics.median(ratios),
                   maximum_QG_algebraic_replay_relative_error=max(errors),
                   replay_is_independent_validation=False, added_arbitrary_uncertainty=False,
                   input_hash_checks_passed=True, working_comparison_ready=True, independently_calibrated=False)
    write_json('BUILD_RECEIPT.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    build()
