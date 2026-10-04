"""Read-only, source-bound comparison of measured RAFM ambient spectra."""
import hashlib
import json
import csv
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ENGINE = Path(r'D:\FluxForge_figure_ownership_fix')
INPUT = Path(r'D:\FluxForge_validation_20261003\latest_replay\workflow_inputs')
NORTH = INPUT/'background.ASC'
SOUTH = Path(r'D:\FluxForge_south_background_scenario\examples\RAFM_irradiation\quantumgold_reference\supplemental_inputs\South 4hr Background Terminal.ANS')
SAMPLE = INPUT/'raw_gamma_spec/flux_wires/Co-Cd-RAFM-1_25cm.ASC'
EXPECT = {
    'north':'505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374',
    'south':'96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344',
}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ENGINE,text=True).strip()=='a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ENGINE,text=True).strip()
input_digest=hashlib.sha256()
for path in sorted((p for p in INPUT.rglob('*') if p.is_file()),key=lambda p:p.relative_to(INPUT).as_posix()):
    name=path.relative_to(INPUT).as_posix().encode('utf-8')
    payload=path.read_bytes()
    input_digest.update(len(name).to_bytes(4,'big'))
    input_digest.update(name)
    input_digest.update(len(payload).to_bytes(8,'big'))
    input_digest.update(payload)
assert input_digest.hexdigest()=='45803d05cfe2c49aa541c9bfc2079d441feb9c7978e81a6a68a732fedc0fdfe1'
assert hashlib.sha256(NORTH.read_bytes()).hexdigest() == EXPECT['north']
assert hashlib.sha256(SOUTH.read_bytes()).hexdigest() == EXPECT['south']
assert NORTH.read_text(errors='replace').splitlines()[0].strip().startswith('ID: North 4hr background terminal')
sys.path.insert(0,str(ENGINE/'src'))
from fluxforge.io.flux_wire import read_raw_asc, read_processed_txt
from fluxforge.examples.rafm_workflow import workflow_profile_energy_calibration

config=json.loads((INPUT/'metadata/workflow_config.json').read_text())
profile_coeff=workflow_profile_energy_calibration(config)
assert profile_coeff == [-1.694,0.4996,6.71e-08]
north_native=read_raw_asc(NORTH).spectrum
sample_native=read_raw_asc(SAMPLE).spectrum
north=read_raw_asc(NORTH,energy_calibration_override=profile_coeff).spectrum
sample=read_raw_asc(SAMPLE,energy_calibration_override=profile_coeff).spectrum
assert north is not None and sample is not None
blob=SOUTH.read_bytes()
south_counts=np.array(struct.unpack_from('<8192I',blob,1548),dtype=float)
south_coeff=struct.unpack_from('<3f',blob,424)
south_energies=np.polynomial.polynomial.polyval(np.arange(8192),south_coeff)
south_live=struct.unpack_from('<d',blob,104)[0]
spectra={
    'north':(np.asarray(north.counts,dtype=float),np.asarray(north.energies,dtype=float),float(north.live_time)),
    'south':(south_counts,south_energies,south_live),
    'sample':(np.asarray(sample.counts,dtype=float),np.asarray(sample.energies,dtype=float),float(sample.live_time)),
}
assert all(len(counts)==len(energies)==8192 for counts,energies,_ in spectra.values())
ans_root=Path(r'D:\FluxForge_south_background_scenario\examples\RAFM_irradiation\quantumgold_reference\originals\ANS')
ans_files=sorted(ans_root.glob('*.ANS'))
assert len(ans_files)==32
detector_ids={p.name:p.read_bytes()[628:640].decode('ascii').strip() for p in ans_files}
assert set(detector_ids.values())=={'South'}
assert blob[628:640].decode('ascii').strip()=='South'
qg_root=INPUT/'QG_processed_gamma_data'
qg_files=sorted(qg_root.rglob('*.txt'))
assert len(qg_files)==31
for qg_file in qg_files:
    detector_lines=[line for line in qg_file.read_text(errors='replace').splitlines() if 'Detector ID:' in line]
    assert len(detector_lines)==1 and detector_lines[0].split('Detector ID:',1)[1].split()[0]=='South'

def band(spec,low,high):
    counts,energy,live=spectra[spec]
    observed=float(counts[(energy>=low)&(energy<high)].sum())
    return dict(observed_counts=observed,counts_per_s=observed/live,
                scaled_to_CoCd_live_counts=observed*sample.live_time/live)

def peak_window(spec,center):
    # Source-comparison diagnostic only. These are not the workflow's IEC peak fits.
    counts,energy,live=spectra[spec]
    core=(energy>=center-5)&(energy<center+5)
    left=(energy>=center-25)&(energy<center-10)
    right=(energy>=center+10)&(energy<center+25)
    # Sideband density per keV, allowing slightly different energy grids.
    density=(float(counts[left].sum())+float(counts[right].sum()))/30.0
    gross=float(counts[core].sum())
    local_continuum=density*10.0
    return dict(gross_counts=gross,local_continuum_counts=local_continuum,
                sideband_excess_counts=gross-local_continuum,
                sideband_excess_scaled_to_CoCd_live=(gross-local_continuum)*sample.live_time/live,
                max_energy_keV=float(energy[core][np.argmax(counts[core])]),
                max_bin_counts=float(np.max(counts[core])))

bands=[(0,100),(100,500),(500,1000),(1000,1500),(1500,2500),(2500,4000)]
result=dict(status='READ_ONLY_DIAGNOSTIC',sources={
    'north':dict(path=str(NORTH),sha256=EXPECT['north'],start_time=str(north.start_time)),
    'south':dict(path=str(SOUTH),sha256=EXPECT['south'],start_time='2025-10-03 15:49:14 (saved ANS; local/unspecified timezone)'),
    'sample':dict(path=str(SAMPLE),sha256=hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),start_time=str(sample.start_time))},
    detector_identity=dict(ANS_original_files=len(ans_files),ANS_detector_ids=sorted(set(detector_ids.values())),
                           QG_report_files=len(qg_files),QG_report_detector_ids=['South'],
                           South_background_native_detector_id='South',
                           North_ASC_header_id='North 4hr background terminal'),
    engine_commit='a7bcc680d1f5e06b1d9dae405241fc380087ca2b',
    input_tree_sha256=input_digest.hexdigest(),
    applied_profile_energy_calibration=profile_coeff,
    native_ASC_calibrations={'north':list(north_native.calibration['energy']),
                             'CoCd_sample':list(sample_native.calibration['energy'])},
    spectra={name:dict(live_time_s=live,total_counts=float(counts.sum()),
                       total_cps=float(counts.sum()/live),
                       energy_coefficients=(list(south_coeff) if name=='south' else
                                            list((north if name=='north' else sample).calibration['energy'])))
             for name,(counts,energy,live) in spectra.items()},
    bands=[dict(energy_low_keV=lo,energy_high_keV=hi,
                spectra={name:band(name,lo,hi) for name in spectra}) for lo,hi in bands],
    Co60_windows={str(center):{name:peak_window(name,center) for name in spectra}
                  for center in (1173.228,1332.492)},
    qualification='Fixed energy bands and sidebands compare source rates only; not a physically qualified peak fit or background choice.')

run_roots={mode:Path(r'D:\FluxForge_current_engine_'+mode+'_20261003')
           for mode in ('north','south','zero_ambient')}
run_modes={'north':'north','south':'south','zero_ambient':'zero_ambient_control'}
result['campaign_artifact_provenance']={}
for mode,run in run_roots.items():
    receipt_path=run/'SCENARIO_RECEIPT.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='COMPLETED' and receipt['mode']==run_modes[mode]
    assert receipt['engine_commit']==result['engine_commit']
    assert receipt['input_identity']['input_tree_sha256']==result['input_tree_sha256']
    if mode=='zero_ambient':
        assert receipt['original_background_slot_sha256']==EXPECT['north']
        assert receipt['injection_count']==1 and receipt['synthetic_background_total_counts']==0
    else:
        assert receipt['original_source_sha256']==EXPECT[mode]
        assert receipt['South_background_injection_count']==int(mode=='south')
    result['campaign_artifact_provenance'][mode]=dict(
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        receipt_mode=receipt['mode'],
        consumed_artifact_sha256={})

def campaign_artifact(mode,relative):
    path=run_roots[mode]/relative
    payload=path.read_bytes()
    result['campaign_artifact_provenance'][mode]['consumed_artifact_sha256'][relative]=hashlib.sha256(payload).hexdigest()
    return payload

for center in (1173.228,1332.492):
    result.setdefault('CoCd_fixed_window_controls',{})[str(center)]={}
    for mode in ('north','south','zero_ambient'):
        rows=list(csv.DictReader(campaign_artifact(mode,'counts/Co-Cd-RAFM-1_25cm_counts.csv').decode('utf-8').splitlines()))
        energy=np.array([float(row['energy_keV']) for row in rows])
        counts=np.array([float(row['background_adjusted_counts']) for row in rows])
        core=(energy>=center-5)&(energy<center+5)
        sides=((energy>=center-25)&(energy<center-10))|((energy>=center+10)&(energy<center+25))
        assert int(core.sum())==20 and int(sides.sum())==60
        gross=float(counts[core].sum())
        local=float(counts[sides].sum()/3)
        artifact=json.loads(campaign_artifact(mode,'analysis_json/Co-Cd-RAFM-1_25cm.json'))
        peak=min((p for p in artifact['peaks'] if p['isotope']=='Co60'),
                 key=lambda p:abs(p['energy_keV']-center))
        result['CoCd_fixed_window_controls'][str(center)][mode]=dict(
            fixed_window_gross_counts=gross,fixed_window_sideband_counts=local,
            fixed_window_net_counts=gross-local,
            workflow_IEC_net_counts=peak['net_counts'],
            workflow_IEC_gross_counts=peak['gross_counts'],
            workflow_IEC_energy_keV=peak['energy_keV'])

cocd_report_path=qg_root/'flux_wires/Co-Cd-RAFM-1_25cm.txt'
cocd_report=read_processed_txt(cocd_report_path)
cocd=next(n for n in cocd_report.nuclides if n.isotope=='Co60')
assert [p['net_counts'] for p in cocd.peaks]==[11202.0,11439.0]
result['CoCd_QG_control']=dict(
    QG_report_sha256=hashlib.sha256(cocd_report_path.read_bytes()).hexdigest(),
    QG_net_counts=[p['net_counts'] for p in cocd.peaks],
    FluxForge_zero_ambient_IEC_net_counts=[result['CoCd_fixed_window_controls'][str(center)]['zero_ambient']['workflow_IEC_net_counts']
                                         for center in (1173.228,1332.492)],
    qualification='Saved ANS ambient-off flag and line-count proximity support an ambient-off QG control; final report settings are unverified.')

v52_report_path=qg_root/'RAFM3/RAFM3-A_300sEOI.txt'
v52_report=read_processed_txt(v52_report_path)
v52=next(n for n in v52_report.nuclides if n.isotope=='V52')
assert len(v52.peaks)==1 and v52.peaks[0]['net_counts']==98446.0
v52_library=json.loads((INPUT/'metadata/sample_gamma_library.json').read_text())['V52']
v52_line=next(line for line in v52_library['gamma_lines'] if abs(line['energy_keV']-1434.09)<0.01)
v52_south=json.loads(campaign_artifact('south','analysis_json/RAFM3-A_300sEOI.json'))
v52_peak=next(p for p in v52_south['peaks'] if p['isotope']=='V52')
v52_isotope=v52_south['isotopes']['V52']
result['V52_example']=dict(QG_report_sha256=hashlib.sha256(v52_report_path.read_bytes()).hexdigest(),
                            QG_report_line_number=v52.peaks[0]['source_line_number'],
                            QG_report_rad_int_text=v52.peaks[0]['reported_rad_int_text'],
                            QG_report_rad_int_unit=v52.peaks[0]['reported_rad_int_unit'],
                            QG_report_net_counts=v52.peaks[0]['net_counts'],
                            QG_report_activity_Bq=v52.activity*37000.0,
                            FluxForge_South_detected_merged_line_net_counts=v52_peak['net_counts'],
                            FluxForge_South_detected_merged_line_activity_Bq=v52_peak['activity_bq'],
                            FluxForge_South_IEC_isotope_activity_Bq=v52_isotope['activity_bq'],
                            local_gamma_library_intensity_fraction=v52_line['intensity'],
                            QG_over_FluxForge_detected_line_activity_ratio=v52.activity*37000.0/v52_peak['activity_bq'],
                            QG_over_FluxForge_IEC_isotope_activity_ratio=v52.activity*37000.0/v52_isotope['activity_bq'],
                            QG_over_FluxForge_detected_line_net_count_ratio=v52.peaks[0]['net_counts']/v52_peak['net_counts'],
                            QG_over_FluxForge_per_count_conversion_ratio=(v52.activity*37000.0/v52.peaks[0]['net_counts'])/(v52_peak['activity_bq']/v52_peak['net_counts']),
                            FluxForge_conversion_with_QG_counts_and_yield_0p01_Bq=(v52_peak['activity_bq']/v52_peak['net_counts'])*v52.peaks[0]['net_counts']*100.0,
                            interpretation='A 1%-versus-100% intensity convention is a leading, testable explanation; exact QG source library and remaining efficiency/timing differences are unverified. Detected-line and targeted IEC paths differ.')
al28=next(n for n in v52_report.nuclides if n.isotope=='Al28')
assert len(al28.peaks)==1 and al28.peaks[0]['reported_rad_int_text']=='1.00'
al28_peak=next(p for p in v52_south['peaks'] if p['isotope']=='Al28')
al28_line=next(line for line in json.loads((INPUT/'metadata/sample_gamma_library.json').read_text())['Al28']['gamma_lines']
               if abs(line['energy_keV']-1778.7)<0.01)
result['Al28_example']=dict(QG_report_sha256=hashlib.sha256(v52_report_path.read_bytes()).hexdigest(),
                             QG_report_line_number=al28.peaks[0]['source_line_number'],
                             QG_report_rad_int_text=al28.peaks[0]['reported_rad_int_text'],
                             QG_report_rad_int_unit=al28.peaks[0]['reported_rad_int_unit'],
                             QG_report_net_counts=al28.peaks[0]['net_counts'],
                             QG_report_activity_Bq=al28.activity*37000.0,
                             FluxForge_South_detected_line_net_counts=al28_peak['net_counts'],
                             FluxForge_South_detected_line_activity_Bq=al28_peak['activity_bq'],
                             local_gamma_library_intensity_fraction=al28_line['intensity'],
                             QG_over_FluxForge_per_count_conversion_ratio=(al28.activity*37000.0/al28.peaks[0]['net_counts'])/(al28_peak['activity_bq']/al28_peak['net_counts']),
                             FluxForge_conversion_with_QG_counts_and_yield_0p01_Bq=(al28_peak['activity_bq']/al28_peak['net_counts'])*al28.peaks[0]['net_counts']*100.0,
                             interpretation='Independent second line supports a 1%-versus-100% QG intensity hypothesis; exact source library is not verified.')
Path(r'D:\FluxForge_background_forensics_20261003.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
