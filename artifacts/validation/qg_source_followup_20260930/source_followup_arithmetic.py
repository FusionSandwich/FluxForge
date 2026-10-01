import hashlib, json, math, subprocess
from pathlib import Path
from fluxforge.io.flux_wire import read_processed_txt
from fluxforge.examples.rafm_workflow import decay_correction_factor, report_count_real_time_s
old = json.loads(Path('artifacts/validation/qg_report_qc_20260930/receipt.json').read_text())
rows = []
for rec in old['reports']:
    if not rec['measurement'].startswith('Ti'): continue
    p = Path(rec['report_path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rec['report_sha256']
    data = read_processed_txt(p)
    n = next(n for n in data.nuclides if n.isotope == 'Sc48')
    lines = n.peaks
    def weighted(items):
        weights = [x['net_counts']/x['net_unc'] for x in items]
        return sum(w*x['activity'] for w,x in zip(weights,items))/sum(weights)
    strong = [x for x in lines if x['center_keV'] > 900]
    all_value, strong_value = weighted(lines), weighted(strong)
    selected = strong_value if rec['measurement'] == 'Ti-RAFM-1' else all_value
    assert abs(selected - n.activity) <= 0.0005
    try: report_count_real_time_s({},data)
    except ValueError: pass
    else: raise AssertionError('Unknown count-decay processing guessed')
    assert report_count_real_time_s({'qg_report_activity_includes_count_decay':True},data) == 0
    assert report_count_real_time_s({'qg_report_activity_includes_count_decay':False},data) == data.real_time
    rows.append(dict(measurement=rec['measurement'],report_sha256=rec['report_sha256'],printed_summary_activity=n.activity,activity_unit=n.activity_unit,conditional_weight='N/u(N), not reconstructed vendor activity uncertainty',all_four_summary=all_value,three_stronger_summary=strong_value,matching_hypothesis='three stronger' if rec['measurement']=='Ti-RAFM-1' else 'all four',live_time_s=data.live_time,real_time_s=data.real_time,printed_half_life_s=n.half_life_seconds,uniform_live_fraction_start_average_factor=decay_correction_factor(n.half_life_seconds,data.real_time,0.0)))
paths=['src/fluxforge/examples/rafm_workflow.py','src/fluxforge/analysis/flux_unfold.py','tests/test_count_decay_time.py']
out=dict(schema='qg-source-followup-v1',inspected_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},reports=rows,report_values_unchanged=True,rate_or_summary_correction=False,scientific_admission=False,no_new_runtime_defect_demonstrated=True,manual=dict(url='https://ludlums.com/images/product_manuals/QTMmanual.pdf',version='4.04.00',pdf_pages=[46,118,119,126],deployed_version_applicability='unverified'),nuclear_reference=dict(url='https://nds.iaea.org/sgnucdat/safeg2008.pdf',pdf_page=116,qualification='historical percent yields; not independent covariance or calibration'),limitations=['Conditional subset/weight matches do not establish vendor selection or summary uncertainty','Count factor requires uniform live fraction; historical decay processing unverified','No full power/rod history, detector covariance or active calibration identity qualified','Private mail provenance is held separately and is excluded from this public receipt'])
p=Path('D:/FluxForgeQA/receipts/qg_report_qc_20260930/source_followup_arithmetic.json')
p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))