"""Read-only triage of source-bound RAFM current-engine line comparisons."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(r'D:\FluxForge_current_engine_south_20261003')
receipt=json.loads((ROOT/'SCENARIO_RECEIPT.json').read_text())
assert receipt['status']=='COMPLETED' and receipt['mode']=='south'
assert receipt['engine_commit']=='a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
assert receipt['input_identity']['input_tree_sha256']=='45803d05cfe2c49aa541c9bfc2079d441feb9c7978e81a6a68a732fedc0fdfe1'
path=ROOT/'tables/line_diagnostics.csv'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='366edf1c0607c8609ad89261a4d84cd56586df4bffe2392bb1c00c06426e8144'
rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))

def number(v):
    try: return float(v)
    except (ValueError,TypeError): return None

matched=[]
analysis_cache={}
analysis_hashes={}
for r in rows:
    refc=number(r['reference_net_counts']); rawc=number(r['raw_net_counts'])
    refa=number(r['reference_line_activity_bq']); rawa=number(r['raw_line_activity_bq'])
    if (r['matched'].lower()=='true' and r['isotope_match'].lower()=='true'
        and all(x is not None and x>0 for x in (refc,rawc,refa,rawa))):
        sample=r['sample_id']
        if sample not in analysis_cache:
            analysis_path=ROOT/'analysis_json'/f'{sample}.json'
            payload=analysis_path.read_bytes()
            analysis_cache[sample]=json.loads(payload)
            analysis_hashes[sample]=hashlib.sha256(payload).hexdigest()
        candidates=[p for p in analysis_cache[sample]['peaks']
                    if p['isotope']==r['raw_isotope'] and abs(p['energy_keV']-float(r['raw_energy_keV']))<1e-6]
        assert len(candidates)==1,(sample,r['reference_isotope'],r['raw_energy_keV'],len(candidates))
        peak=candidates[0]
        comparison_count=peak['comparison_net_counts']
        csv_count_basis='comparison_net_counts' if comparison_count is not None else 'physical_net_counts_fallback'
        assert abs((comparison_count if comparison_count is not None else peak['net_counts'])-rawc)<1e-5
        assert abs(peak['activity_bq']-rawa)<1e-5
        assert peak['net_counts']>0
        matched.append(dict(sample=r['sample_id'],isotope=r['reference_isotope'],energy=number(r['reference_energy_keV']),
                            csv_count_ratio=rawc/refc,csv_count_basis=csv_count_basis,
                            physical_count_ratio=peak['net_counts']/refc,
                            conversion_ratio=(refa/refc)/(rawa/peak['net_counts']),
                            qg_rad_int=r['reported_rad_int_text'],source_qc=r['source_qc_bucket'],
                            diagnostic=r['diagnostic_bucket'],qg_refc=refc,ff_csvc=rawc,ff_physicalc=peak['net_counts'],
                            qg_refa=refa,ff_rawa=rawa,
                            qg_report_sha256=r['report_source_sha256'],
                            qg_report_line_number=r['report_source_line_number']))

by_isotope=defaultdict(list)
for r in matched: by_isotope[r['isotope']].append(r)
summary=[]
for iso,group in by_isotope.items():
    close=[r for r in group if .8<=r['physical_count_ratio']<=1.2]
    summary.append(dict(isotope=iso,matched_lines=len(group),physical_count_close_lines=len(close),
                        median_csv_count_ratio=statistics.median(r['csv_count_ratio'] for r in group),
                        median_physical_count_ratio=statistics.median(r['physical_count_ratio'] for r in group),
                        median_conversion_ratio=statistics.median(r['conversion_ratio'] for r in group),
                        physical_close_count_median_conversion_ratio=(statistics.median(r['conversion_ratio'] for r in close) if close else None)))
summary.sort(key=lambda r:(-r['physical_count_close_lines'],-r['matched_lines'],r['isotope']))
out=dict(receipt_sha256=hashlib.sha256((ROOT/'SCENARIO_RECEIPT.json').read_bytes()).hexdigest(),
         line_diagnostics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         analysis_json_sha256=analysis_hashes,
         source='current-engine South conditional campaign',
         matched_positive_lines=len(matched),
         csv_count_close_lines=sum(.8<=r['csv_count_ratio']<=1.2 for r in matched),
         csv_count_explicit_comparison_lines=sum(r['csv_count_basis']=='comparison_net_counts' for r in matched),
         physical_count_close_lines=sum(.8<=r['physical_count_ratio']<=1.2 for r in matched),
         isotope_summary=summary,
         rad_int_1p00_physical_close_count_large_conversion=[r for r in matched if r['qg_rad_int']=='1.00' and .8<=r['physical_count_ratio']<=1.2 and r['conversion_ratio']>2],
         sc48_same_sample_internal_control=[r for r in matched if r['sample']=='Ti-RAFM-1_25cm' and r['isotope']=='Sc48'],
         physical_close_count_large_conversion=[r for r in matched if .8<=r['physical_count_ratio']<=1.2 and (r['conversion_ratio']>2 or r['conversion_ratio']<.5)])
Path(r'D:\FluxForge_campaign_pattern_scan_20261004.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(matched_positive_lines=out['matched_positive_lines'],csv_count_close_lines=out['csv_count_close_lines'],
                      physical_count_close_lines=out['physical_count_close_lines'],
                      rad_int_1p00_physical_close_count_large_conversion=out['rad_int_1p00_physical_close_count_large_conversion'],
                      sc48_same_sample_internal_control=out['sc48_same_sample_internal_control']),indent=2))
