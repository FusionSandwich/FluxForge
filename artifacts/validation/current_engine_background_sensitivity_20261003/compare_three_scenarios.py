"""Compare pinned measured-background runs with a zero-ambient control."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = {name: Path(r'D:\FluxForge_current_engine_' + name + '_20261003')
        for name in ('north', 'south', 'zero_ambient')}
OUT = Path(r'D:\FluxForge_current_engine_ambient_control_comparison_20261003.json')
ENGINE = 'a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
INPUT_TREE = '45803d05cfe2c49aa541c9bfc2079d441feb9c7978e81a6a68a732fedc0fdfe1'
NORTH_HASH = '505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374'
SOUTH_HASH = '96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344'
AUDIT_HASH = 'c63f9f6e095da6dcc0a86d4e4d8a36e150a45299122ac2304258d82a29942c4d'
SCHEDULE = Path(r'D:\FluxForge_validation_20261003\latest_replay\workflow_inputs\metadata\sample_schedule.json')
schedule = json.loads(SCHEDULE.read_text(encoding='utf-8'))
assert schedule['flux_wires']['CU-RAFM-1']['base_name'] == 'CU-RAFM-1'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row['sample_id'], row['isotope']): row for row in rows}
    assert len(rows) == len(keyed)
    return keyed

def read_rates(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row['sample_id'], row['reaction_id']): row for row in rows}
    assert len(rows) == len(keyed)
    return keyed

def finite_float(value):
    if value in (None, '', 'None'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None

receipts = {name: read_json(root/'SCENARIO_RECEIPT.json') for name, root in ROOT.items()}
summaries = {name: read_json(root/'validation_summary.json') for name, root in ROOT.items()}
tables = {name: read_rows(root/'tables/isotope_comparison.csv') for name, root in ROOT.items()}
rate_tables = {name:read_rates(root/'tables/flux_wire_reaction_rates.csv') for name,root in ROOT.items()}
for name in ROOT:
    receipt = receipts[name]
    assert receipt['status'] == 'COMPLETED' and receipt['engine_commit'] == ENGINE
    assert receipt['input_identity']['input_tree_sha256'] == INPUT_TREE
    assert summaries[name]['n_raw_analyzed'] == 30 and summaries[name]['n_matched_pairs'] == 29
    assert receipt['counting_methods'] == ['iec_tiered', 'iec_tiered']
    assert tables[name].keys() == tables['north'].keys()
    assert rate_tables[name].keys() == rate_tables['north'].keys()

assert receipts['north']['mode'] == 'north'
assert receipts['north']['original_source_sha256'] == NORTH_HASH
assert receipts['north']['South_background_injection_count'] == 0
assert receipts['south']['mode'] == 'south'
assert receipts['south']['original_source_sha256'] == SOUTH_HASH
assert receipts['south']['South_background_injection_count'] == 1
zero = receipts['zero_ambient']
assert zero['mode'] == 'zero_ambient_control'
assert zero['original_background_slot_sha256'] == NORTH_HASH
assert zero['injection_count'] == 1 and zero['synthetic_background_total_counts'] == 0
assert zero['QG_saved_header_evidence']['sha256'] == AUDIT_HASH
assert zero['QG_final_report_ambient_state'] == 'UNKNOWN'

rows = []
for key in sorted(tables['north']):
    by_mode = {name: tables[name][key] for name in ROOT}
    references = {row['reference_activity_bq'] for row in by_mode.values()}
    assert len(references) == 1
    groups = {row['sample_group'] for row in by_mode.values()}
    assert len(groups) == 1
    reported_group = next(iter(groups))
    canonical_group = 'flux_wires' if key[0] == 'Cu-RAFM-1_25cm' else reported_group
    assert key[0] != 'Cu-RAFM-1_25cm' or reported_group == 'RAFM1'
    rows.append(dict(sample_id=key[0], reported_sample_group=reported_group,
                     sample_group=canonical_group, isotope=key[1],
                     QG_report_Bq=finite_float(next(iter(references))),
                     modes={name:dict(matched=row['matched'] == 'True',
                                      activity_Bq=finite_float(row['raw_activity_bq']),
                                      QG_relative_error=finite_float(row['relative_activity_error']))
                            for name,row in by_mode.items()}))

def common(*modes):
    return [row for row in rows if all(row['modes'][mode]['matched'] and
            row['modes'][mode]['QG_relative_error'] is not None for mode in modes)]

triple = common(*ROOT)
group_summary = {}
for group in sorted({row['sample_group'] for row in triple}):
    group_rows = [row for row in triple if row['sample_group'] == group]
    group_summary[group] = dict(common_finite_rows=len(group_rows),
                                median_abs_QG_relative_error={name:statistics.median(
                                    abs(row['modes'][name]['QG_relative_error']) for row in group_rows)
                                    for name in ROOT})
pairwise = {}
for first, second in (('north','south'),('north','zero_ambient'),('south','zero_ambient')):
    both = common(first, second)
    pairwise[f'{first}_vs_{second}'] = dict(
        common_finite_rows=len(both),
        median_abs_QG_relative_error={mode:statistics.median(abs(row['modes'][mode]['QG_relative_error'])
                                                        for row in both) for mode in (first,second)},
        second_better=sum(abs(row['modes'][second]['QG_relative_error']) <
                          abs(row['modes'][first]['QG_relative_error']) for row in both),
        first_better=sum(abs(row['modes'][second]['QG_relative_error']) >
                         abs(row['modes'][first]['QG_relative_error']) for row in both))

co_key = ('Co-Cd-RAFM-1_25cm','Co60')
assert co_key in tables['north']
rate_rows=[]
for key in sorted(rate_tables['north']):
    notes={name:rate_tables[name][key]['rate_note'] or None for name in ROOT}
    values={name:(None if notes[name] else finite_float(rate_tables[name][key]['reaction_rate']))
            for name in ROOT}
    rate_rows.append(dict(sample_id=key[0],reaction_id=key[1],reaction_rate_per_atom_s=values,
                          exclusion_reason=notes,
                          south_over_north=(values['south']/values['north']
                                            if values['north'] is not None and values['south'] is not None and values['north'] else None),
                          zero_over_north=(values['zero_ambient']/values['north']
                                           if values['north'] is not None and values['zero_ambient'] is not None and values['north'] else None)))
result = dict(status='COMPLETED',engine_commit=ENGINE,input_tree_sha256=INPUT_TREE,
              audit_sha256=AUDIT_HASH,
              interpretation='zero ambient is a counterfactual protocol control, not a measured physical background',
              QG_final_report_ambient_state='UNKNOWN',
              South_temporal_applicability='UNRESOLVED',
              sample_group_correction='Cu-RAFM-1_25cm is source-manifest flux wire; workflow table calls it RAFM1',
              pairwise=pairwise,
              triple_common_finite_rows=len(triple),
              group_summary=group_summary,
              triple_median_abs_QG_relative_error={name:statistics.median(
                  abs(row['modes'][name]['QG_relative_error']) for row in triple) for name in ROOT},
              CoCd_isotope_activity_Bq={name:finite_float(tables[name][co_key]['raw_activity_bq']) for name in ROOT},
              CoCd_QG_report_Bq=finite_float(tables['north'][co_key]['reference_activity_bq']),
              overall_passed={name:summaries[name]['overall_passed'] for name in ROOT},
              failing_sample_count={name:len(summaries[name]['failing_samples']) for name in ROOT},
              qg_internal_consistency_flags={name:summaries[name]['qg_internal_consistency_flags'] for name in ROOT},
              fluxforge_line_consistency_flags={name:summaries[name]['fluxforge_line_consistency_flags'] for name in ROOT},
              reaction_rate_rows=rate_rows,
              activity_rows=rows)
OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key not in ('activity_rows','reaction_rate_rows')},indent=2))
