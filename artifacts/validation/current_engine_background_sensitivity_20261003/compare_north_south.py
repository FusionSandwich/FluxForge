"""Pair current-engine RAFM background scenarios and keep count bases distinct."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = {name:Path(r'D:\FluxForge_current_engine_' + name + '_20261003')
        for name in ('north','south')}
OUT = Path(r'D:\FluxForge_current_engine_background_comparison_20261003.json')

def load_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def number(value):
    return float(value) if value not in (None,'','None') else None

receipts = {name:json.loads((root/'SCENARIO_RECEIPT.json').read_text())
            for name,root in ROOT.items()}
summaries = {name:json.loads((root/'validation_summary.json').read_text())
             for name,root in ROOT.items()}
assert all(r['status']=='COMPLETED' and r['engine_commit']=='a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
           for r in receipts.values())
assert receipts['north']['mode'] == 'north'
assert receipts['north']['original_source_sha256'] == '505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374'
assert receipts['north']['South_background_injection_count'] == 0
assert receipts['south']['mode'] == 'south'
assert receipts['south']['original_source_sha256'] == '96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344'
assert receipts['south']['South_background_injection_count'] == 1
assert all(s['n_raw_analyzed']==30 and s['n_matched_pairs']==29 for s in summaries.values())
assert receipts['north']['input_identity']['input_tree_sha256'] == receipts['south']['input_identity']['input_tree_sha256']
assert receipts['north']['counting_methods'] == receipts['south']['counting_methods'] == ['iec_tiered','iec_tiered']

isotopes = {}
for name,root in ROOT.items():
    raw = load_csv(root/'tables/isotope_comparison.csv')
    keyed = {(r['sample_id'],r['isotope']):r for r in raw}
    assert len(raw)==len(keyed)
    isotopes[name]=keyed
assert isotopes['north'].keys()==isotopes['south'].keys()

rows=[]
for key in sorted(isotopes['north']):
    n,s=isotopes['north'][key],isotopes['south'][key]
    assert n['reference_activity_bq']==s['reference_activity_bq']
    a,b=number(n['raw_activity_bq']),number(s['raw_activity_bq'])
    rows.append(dict(sample_id=key[0],isotope=key[1],QG_report_Bq=number(s['reference_activity_bq']),
                     matched_north=n['matched']=='True',matched_south=s['matched']=='True',
                     north_Bq=a,south_Bq=b,south_over_north=(b/a if a and b is not None else None),
                     north_QG_relative_error=number(n['relative_activity_error']),
                     south_QG_relative_error=number(s['relative_activity_error'])))
common=[r for r in rows if r['matched_north'] and r['matched_south']
        and r['north_QG_relative_error'] is not None and r['south_QG_relative_error'] is not None
        and math.isfinite(r['north_QG_relative_error']) and math.isfinite(r['south_QG_relative_error'])]
status_changes=[r for r in rows if r['matched_north'] != r['matched_south']]

def co_cd_peak_bases(name):
    root=ROOT[name]
    artifact=json.loads((root/'analysis_json/Co-Cd-RAFM-1_25cm.json').read_text())
    return [dict(energy_keV=p['energy_keV'],physical_net_counts=p['net_counts'],
                 comparison_net_counts=p.get('comparison_net_counts'),
                 line_activity_Bq=p.get('activity_bq'))
            for p in artifact['peaks'] if p['isotope']=='Co60']

top=sorted([r for r in rows if r['south_over_north'] is not None],
           key=lambda r:abs(r['south_over_north']-1),reverse=True)
result=dict(status='COMPLETED', engine_commit=receipts['south']['engine_commit'],
            input_identity=receipts['south']['input_identity'],
            counting_methods=receipts['south']['counting_methods'],
            background_hashes={name:receipt['original_source_sha256']
                               for name,receipt in receipts.items()},
            QG_background_match='UNKNOWN',South_temporal_applicability='UNRESOLVED',
            scope='30 original ASC spectra; 29 paired QG reports; current validation engine',
            summaries={name:dict(overall_passed=s['overall_passed'],
                                 failing_samples=s['failing_samples'],
                                 qg_internal_consistency_flags=s['qg_internal_consistency_flags'],
                                 fluxforge_line_consistency_flags=s['fluxforge_line_consistency_flags'])
                       for name,s in summaries.items()},
            common_finite_activity_rows=len(common),
            median_abs_QG_relative_error={name:statistics.median(abs(r[name+'_QG_relative_error'])
                                                          for r in common) for name in ROOT},
            QG_error_improved_by_south=sum(abs(r['south_QG_relative_error']) < abs(r['north_QG_relative_error']) for r in common),
            QG_error_worsened_by_south=sum(abs(r['south_QG_relative_error']) > abs(r['north_QG_relative_error']) for r in common),
            activity_match_status_changes=status_changes,
            CoCd_peak_count_bases={name:co_cd_peak_bases(name) for name in ROOT},
            largest_relative_activity_changes=top[:20],activity_rows=rows)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('activity_rows','largest_relative_activity_changes','summaries')},indent=2))
