"""Bounded Co-Cd South-background amplitude stress test, not background calibration."""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ENGINE=Path(r'D:\FluxForge_figure_ownership_fix')
INPUT=Path(r'D:\FluxForge_validation_20261003\latest_replay\workflow_inputs')
SOUTH=Path(r'D:\FluxForge_south_background_scenario\examples\RAFM_irradiation\quantumgold_reference\supplemental_inputs\South 4hr Background Terminal.ANS')
ROOT=Path(r'D:\FluxForge_south_scale_probe_v3_20261003')
assert not ROOT.exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ENGINE,text=True).strip()=='a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ENGINE,text=True).strip()
blob=SOUTH.read_bytes()
assert hashlib.sha256(blob).hexdigest()=='96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344'
assert blob[628:640].decode('ascii').strip()=='South'
input_digest=hashlib.sha256()
for path in sorted((p for p in INPUT.rglob('*') if p.is_file()),key=lambda p:p.relative_to(INPUT).as_posix()):
    name=path.relative_to(INPUT).as_posix().encode('utf-8')
    payload=path.read_bytes()
    input_digest.update(len(name).to_bytes(4,'big'))
    input_digest.update(name)
    input_digest.update(len(payload).to_bytes(8,'big'))
    input_digest.update(payload)
assert input_digest.hexdigest()=='45803d05cfe2c49aa541c9bfc2079d441feb9c7978e81a6a68a732fedc0fdfe1'
counts=np.array(struct.unpack_from('<8192I',blob,1548),dtype=float)
coeff=np.array(struct.unpack_from('<3f',blob,424))
channels=np.arange(8192)
energies=np.polynomial.polynomial.polyval(channels,coeff)
live=struct.unpack_from('<d',blob,104)[0]
real=struct.unpack_from('<d',blob,96)[0]
sys.path.insert(0,str(ENGINE/'src'))
from fluxforge.examples import rafm_workflow as w
from fluxforge.io.spe import GammaSpectrum

raw=INPUT/'raw_gamma_spec/flux_wires/Co-Cd-RAFM-1_25cm.ASC'
qg=INPUT/'QG_processed_gamma_data/flux_wires/Co-Cd-RAFM-1_25cm.txt'
assert hashlib.sha256(raw.read_bytes()).hexdigest()=='7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba'
metadata=None
results=[]
for factor in (0.0,0.25,0.5,0.75,1.0,1.25):
    out=ROOT/f'factor_{factor:.2f}'
    paths=w.default_paths(INPUT,results_root=out)
    if metadata is None:
        metadata=w.load_rafm_example_metadata(paths.example_root)
        metadata.config['flux_wire_counting_method']='iec_tiered'
    tree=w.ensure_results_tree(out)
    bg=GammaSpectrum(counts=counts*factor,
        counts_uncertainty=np.sqrt(counts)*factor,
        channels=channels.copy(),energies=energies.copy(),
        live_time=live,real_time=real,spectrum_id=f'South amplitude stress factor {factor}',
        detector_id='South',calibration={'energy':coeff.tolist()},
        metadata={'source_sha256':hashlib.sha256(blob).hexdigest(),
                  'control_factor':factor,'status':'STRESS_TEST_NOT_PHYSICAL_SCALE'})
    artifact=w.analyze_flux_wire_sample(raw,metadata,paths,tree,bg,qg,'co-cd-rafm-1')
    co=[p for p in artifact['peaks'] if p['isotope']=='Co60']
    assert len(co)==2
    isotope=artifact['isotopes']['Co60']
    results.append(dict(factor=factor,activity_Bq=isotope['activity_bq'],
                        peak_net_counts=[p['net_counts'] for p in co],
                        peak_gross_counts=[p['gross_counts'] for p in co],
                        peak_background_adjusted_gross_counts=[p['background_adjusted_gross_counts'] for p in co],
                        peak_centers_keV=[p['energy_keV'] for p in co]))
    print(json.dumps(results[-1]),flush=True)
receipt=dict(status='COMPLETED',scope='one Co-Cd spectrum; synthetic South-amplitude stress test',
             engine_commit='a7bcc680d1f5e06b1d9dae405241fc380087ca2b',
             South_source_sha256=hashlib.sha256(blob).hexdigest(),
             input_tree_sha256=input_digest.hexdigest(),sample_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
             QG_report_sha256=hashlib.sha256(qg.read_bytes()).hexdigest(),
             method='iec_tiered',qualification='factor is a synthetic stress parameter, not inferred background correction',
             results=results)
(ROOT/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
