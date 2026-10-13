"""Bounded 30-ASC scenario on the current 4615-lineage engine with a pinned background."""
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

ENGINE = Path(r'D:\FluxForge_figure_ownership_fix')
INPUT = Path(r'D:\FluxForge_validation_20261003\latest_replay\workflow_inputs')
SOUTH_SOURCE = Path(r'D:\FluxForge_south_background_scenario\examples\RAFM_irradiation\quantumgold_reference\supplemental_inputs\South 4hr Background Terminal.ANS')
NORTH_SOURCE = INPUT/'background.ASC'
INPUT_IDENTITY = json.loads(Path(r'D:\FluxForge_current_input_identity_20261003.json').read_text())
input_digest = hashlib.sha256()
for input_path in sorted((p for p in INPUT.rglob('*') if p.is_file()),
                         key=lambda p:p.relative_to(INPUT).as_posix()):
    name = input_path.relative_to(INPUT).as_posix().encode('utf-8')
    payload = input_path.read_bytes()
    input_digest.update(len(name).to_bytes(4,'big'))
    input_digest.update(name)
    input_digest.update(len(payload).to_bytes(8,'big'))
    input_digest.update(payload)
assert input_digest.hexdigest() == INPUT_IDENTITY['input_tree_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ENGINE, text=True).strip() == 'a7bcc680d1f5e06b1d9dae405241fc380087ca2b'
assert not subprocess.check_output(['git','status','--porcelain'], cwd=ENGINE, text=True).strip()
sys.path.insert(0, str(ENGINE/'src'))
from fluxforge.examples import rafm_workflow as workflow
from fluxforge.io.spe import GammaSpectrum

mode = sys.argv[1] if len(sys.argv) == 2 else None
if mode not in ('north','south'):
    raise SystemExit('Usage: script north|south')
out = Path(r'D:\FluxForge_current_engine_' + mode + '_20261003')
if out.exists():
    raise FileExistsError(out)

if mode == 'north':
    sha = hashlib.sha256(NORTH_SOURCE.read_bytes()).hexdigest()
    assert sha == '505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374'
    summary = workflow.run_rafm_validation(INPUT, results_root=out,
                                           enforce_thresholds=False, generate_plots=False,
                                           flux_wire_counting_method='iec_tiered',
                                           generic_targeted_counting_method='iec_tiered')
    injection_count = 0
    background_source = str(NORTH_SOURCE)
else:
    blob = SOUTH_SOURCE.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    assert sha == '96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344'
    assert len(blob) == 36616 and b'South 4 hr background terminal' in blob[:1548]
    coeff = list(struct.unpack_from('<3f', blob, 424))
    live, real = struct.unpack_from('<d', blob, 104)[0], struct.unpack_from('<d', blob, 96)[0]
    assert live == 14400.0 and live <= real < 14500
    start = datetime(1899, 12, 30) + timedelta(days=struct.unpack_from('<d', blob, 80)[0])
    counts = np.asarray(struct.unpack_from('<8192I', blob, 1548), dtype=float)
    assert int(counts.sum()) == 543427
    channels = np.arange(8192)
    energy = coeff[0] + coeff[1]*channels + coeff[2]*channels**2
    assert np.all(np.diff(energy) > 0)
    south = GammaSpectrum(counts=counts, channels=channels, energies=energy,
                          live_time=live, real_time=real, start_time=start,
                          spectrum_id='South 4 hr background terminal', detector_id='South',
                          calibration={'energy': coeff},
                          metadata={'source_file': str(SOUTH_SOURCE), 'source_sha256': sha})
    original_read = workflow.read_raw_asc
    injection_count = 0
    def selected_read(path, *args, **kwargs):
        global injection_count
        if Path(path).resolve() == NORTH_SOURCE.resolve():
            injection_count += 1
            parsed = original_read(path, *args, **kwargs)
            parsed.spectrum = south
            return parsed
        return original_read(path, *args, **kwargs)
    with patch.object(workflow, 'read_raw_asc', side_effect=selected_read):
        summary = workflow.run_rafm_validation(INPUT, results_root=out,
                                               enforce_thresholds=False, generate_plots=False,
                                               flux_wire_counting_method='iec_tiered',
                                               generic_targeted_counting_method='iec_tiered')
    assert injection_count == 1, injection_count
    background_source = str(SOUTH_SOURCE)

receipt = dict(status='COMPLETED', mode=mode, engine_commit='a7bcc680d1f5e06b1d9dae405241fc380087ca2b',
               engine_base='4615e61bbb262d974e0326a44107bc952e4cb903',
               original_source_sha256=sha, source_path=background_source,
               South_background_injection_count=injection_count,
               counting_methods=['iec_tiered','iec_tiered'],
               QG_reference_used_for_activity=False,
               QG_reference_line_library_independence='NOT_ESTABLISHED',
               South_temporal_applicability='UNRESOLVED', QG_background_match='UNKNOWN',
               numerical_status='conditional; integrated-bin-overlap rebinning with covariance; background applicability and profile efficiency unqualified',
               input_identity=INPUT_IDENTITY,
               summary=summary)
(out/'SCENARIO_RECEIPT.json').write_text(json.dumps(receipt, indent=2, default=str)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'summary'}, indent=2), flush=True)
