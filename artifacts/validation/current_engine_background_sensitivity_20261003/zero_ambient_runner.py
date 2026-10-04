"""Counterfactual QG ambient-off control on pinned current FluxForge engine."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ENGINE = Path(r'D:\FluxForge_figure_ownership_fix')
INPUT = Path(r'D:\FluxForge_validation_20261003\latest_replay\workflow_inputs')
NORTH_SOURCE = INPUT/'background.ASC'
OUT = Path(r'D:\FluxForge_current_engine_zero_ambient_20261003')
IDENTITY = json.loads(Path(r'D:\FluxForge_current_input_identity_20261003.json').read_text())
DIAGNOSTIC_PATH = Path(r'D:\FluxForge_QuantumGold_documentation_20261003\saved_settings_audit.json')
diagnostic_blob = DIAGNOSTIC_PATH.read_bytes()
diagnostic = json.loads(diagnostic_blob)
assert diagnostic['status'] == 'BOUNDED_SAVED_SETTINGS_DIAGNOSTIC'
assert diagnostic['files_checked'] == 32 and len(diagnostic['rows']) == 32
assert diagnostic['layout_offsets']['analysis_ctrl']['offset'] == 860
assert all(row['values']['analysis_ctrl'] == 1 and
           row['saved_setting_inference']['ambient_background_correction_enabled'] is False
           for row in diagnostic['rows'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ENGINE,text=True).strip() == 'a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ENGINE,text=True).strip()
digest = hashlib.sha256()
for path in sorted((p for p in INPUT.rglob('*') if p.is_file()),
                   key=lambda p:p.relative_to(INPUT).as_posix()):
    name = path.relative_to(INPUT).as_posix().encode('utf-8')
    blob = path.read_bytes()
    digest.update(len(name).to_bytes(4,'big'))
    digest.update(name)
    digest.update(len(blob).to_bytes(8,'big'))
    digest.update(blob)
assert digest.hexdigest() == IDENTITY['input_tree_sha256']
assert hashlib.sha256(NORTH_SOURCE.read_bytes()).hexdigest() == '505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374'
if OUT.exists():
    raise FileExistsError(OUT)

sys.path.insert(0,str(ENGINE/'src'))
from fluxforge.examples import rafm_workflow as workflow
from fluxforge.io.spe import GammaSpectrum

original = workflow.read_raw_asc
injections = 0
def read_with_zero_ambient(path,*args,**kwargs):
    global injections
    parsed = original(path,*args,**kwargs)
    if Path(path).resolve() == NORTH_SOURCE.resolve():
        injections += 1
        native = parsed.spectrum
        assert native is not None and native.counts.shape == (8192,)
        parsed.spectrum = GammaSpectrum(
            counts=np.zeros_like(native.counts,dtype=float),
            counts_uncertainty=np.zeros_like(native.counts,dtype=float),
            channels=np.asarray(native.channels).copy(),
            energies=np.asarray(native.energies).copy() if native.energies is not None else None,
            live_time=float(native.live_time),real_time=float(native.real_time),
            start_time=native.start_time,spectrum_id='synthetic_zero_ambient_control',
            detector_id='counterfactual_none',calibration=dict(native.calibration),
            metadata={'control':'zero-valued background route; no measured ambient subtraction',
                      'physical_source':'NONE'})
    return parsed

with patch.object(workflow,'read_raw_asc',side_effect=read_with_zero_ambient):
    summary=workflow.run_rafm_validation(INPUT,results_root=OUT,
                                         enforce_thresholds=False,generate_plots=False,
                                         flux_wire_counting_method='iec_tiered',
                                         generic_targeted_counting_method='iec_tiered')
assert injections == 1,injections
receipt=dict(status='COMPLETED',mode='zero_ambient_control',
             engine_commit='a7bcc680d1f5e06b1d9dae405241fc380087ca2b',
             input_identity=IDENTITY,original_background_slot_sha256=hashlib.sha256(NORTH_SOURCE.read_bytes()).hexdigest(),
             injection_count=injections,synthetic_background_total_counts=0,
             QG_saved_header_ambient_off='SUPPORTED_BY_PINNED_DIAGNOSTIC',
             QG_saved_header_evidence=dict(path=str(DIAGNOSTIC_PATH),
                 sha256=hashlib.sha256(diagnostic_blob).hexdigest(),files_checked=32,
                 analysis_ctrl_offset=860,observed_analysis_ctrl_values=[1],
                 manual_url=diagnostic['manual_url'],manual_version=diagnostic['manual_version'],
                 qualification='saved ANS settings only; not proof of final report state'),
             QG_final_report_ambient_state='UNKNOWN',physical_background_choice='NOT_ESTABLISHED',
             counting_methods=['iec_tiered','iec_tiered'],
             interpretation='counterfactual QG-protocol sensitivity only; not a measured physical background',
             summary=summary)
(OUT/'SCENARIO_RECEIPT.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='summary'},indent=2),flush=True)
