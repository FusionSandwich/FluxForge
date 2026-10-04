"""Offline campaign and method comparison, using only repository-bound inputs.

Run with existing FluxForge dependencies:
  python examples/RAFM_irradiation/run_integrated_qg_methods.py --output NEW_FOLDER
Use --skip-campaigns for the bounded component examples only. No installation,
QuantumGold binary, Git metadata or machine-specific data path is required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def load_driver():
    path = ROOT / 'examples/RAFM_irradiation/run_portable_qg_example.py'
    spec = importlib.util.spec_from_file_location('integrated_portable_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def component_environment():
    """Execute every example against this checkout, including direct imports."""
    return dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUTF8='1',
                MPLBACKEND='Agg', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def native_only_controls(driver, checked, output):
    """Dataset-bound ANS counts without fabricating ASC or a geometry match."""
    import numpy as np
    from fluxforge.analysis.flux_wire_analysis import analyze_flux_wire_targeted
    from fluxforge.analysis.qg_protocol import parse_saved_state, LAYOUT_ID
    from fluxforge.data.rafm_profile import load_rafm_profile
    from fluxforge.io.flux_wire import FluxWireData, _efficiency_from_override
    from fluxforge.io.spe import GammaSpectrum

    manifest = checked['manifest']
    pins = {p['path']: p for p in manifest['resources']}
    profile = load_rafm_profile('rafm_25cm')
    background, details = driver.background_scenario(ROOT, checked, 'south_native')
    receipts = []
    for row in manifest['measurements']:
        if row['files']['ASC']:
            continue
        source = row['files']['ANS']
        blob = driver.bound_path(ROOT, source).read_bytes()
        report_path = row['files']['QG_report']
        report = driver.bound_path(ROOT, report_path).read_bytes() if report_path else None
        state = parse_saved_state(blob, expected_sha256=pins[source]['sha256'], layout_id=LAYOUT_ID,
                                 report=report, expected_report_sha256=pins[report_path]['sha256'] if report else None)
        distance = row['QG_header'].get('source_distance_cm_printed')
        receipt = dict(measurement_id=row['measurement_id'], native_sha256=state.source_sha256,
                       count_basis='original_dataset_bound_ANS_integer_counts',
                       geometry_cm_printed=distance, independent_absolute_qualification=False,
                       creates_ASC=False, background=details, reaction_rate=None,
                       reaction_rate_status='UNAVAILABLE: irradiation/timing qualification pending')
        if distance != 25.0:
            receipt.update(status='EXCLUDED_GEOMETRY', reason='25 cm efficiency is inapplicable; native counts remain exported by campaign replay')
        else:
            counts = np.asarray(checked['arrays'][row['measurement_id']], dtype=float)
            coefficients = struct.unpack_from('<3f', blob, 424)
            channels = np.arange(len(counts))
            energies = coefficients[0] + coefficients[1]*channels + coefficients[2]*channels**2
            if not np.all(np.diff(energies) > 0):
                raise ValueError('Nonmonotonic native sample energy calibration')
            live, real = struct.unpack_from('<d', blob, 104)[0], struct.unpack_from('<d', blob, 96)[0]
            start = datetime(1899, 12, 30) + timedelta(days=struct.unpack_from('<d', blob, 80)[0])
            if live <= 0 or real < live:
                raise ValueError('Invalid native count durations')
            spectrum = GammaSpectrum(counts=counts, channels=channels, energies=energies,
                                     live_time=live, real_time=real, start_time=start,
                                     spectrum_id=row['workflow_stem'], detector_id='South')
            data = FluxWireData(sample_id=row['workflow_stem'], file_type='dataset_bound_ans',
                                source_file=source, spectrum=spectrum, start_time=start,
                                live_time=live, real_time=real, energy_calibration=list(coefficients),
                                efficiency=_efficiency_from_override(profile.efficiency), resolution=list(profile.resolution))
            # The nominal detector profile remains a conditional response, not
            # a recovered independent calibration or report activity target.
            result = analyze_flux_wire_targeted(data=data, reference_data=None,
                         background_spectrum=background, profile_name='rafm_25cm',
                         counting_method='iec_tiered')
            receipt.update(status='CONDITIONAL_NATIVE_RAW_DIAGNOSTIC', analysis=result.to_dict(),
                           calibration='native energy; conditional nominal 25 cm efficiency/resolution')
        dump(output / (row['measurement_id'] + '.json'), receipt)
        receipts.append({k: v for k, v in receipt.items() if k != 'analysis'})
    return receipts


def current_line_combinations(campaign, engine_sha):
    """Freeze current-engine selected lines; do not admit new isotopes or targets."""
    import numpy as np
    from fluxforge.analysis.activity_combination import (
        ActivityLine, CovarianceComponent, METHODS, combine_activity_lines)

    rows = []
    for path in sorted((campaign / 'raw_replay').rglob('*.json')):
        artifact = json.loads(path.read_text())
        if not isinstance(artifact, dict) or artifact.get('sample_group') != 'flux_wires' or 'isotopes' not in artifact:
            continue
        source = 'SHA256:' + hashlib.sha256(path.read_bytes()).hexdigest()
        for isotope, payload in artifact['isotopes'].items():
            if isotope in ('Sc48', 'Ni57'):
                rows.append(dict(sample_id=artifact['sample_id'], isotope=isotope,
                                 status='EXCLUDED_REFERENCE_OR_LINE_ADMISSION', methods=[]))
                continue
            # These diagnostics are constructed from exactly the engine-selected
            # physical activity lines, not comparison_net_counts or QG summaries.
            chosen = payload.get('single_peak_activity_diagnostics', [])
            if not chosen:
                continue
            reference = payload.get('activity_reference')
            if reference != 'count_average_live_normalized':
                rows.append(dict(sample_id=artifact['sample_id'], isotope=isotope,
                                 status='EXCLUDED_UNKNOWN_OR_INCOMPATIBLE_ACTIVITY_REFERENCE', methods=[]))
                continue
            lines = [ActivityLine(str(p['energy_keV']), isotope,
                         p['line_activity_bq'], p['line_activity_unc_bq'], True,
                         'inherited current-engine numerical line selection; physical qualification pending',
                         source, 'physical_background_adjusted_net_counts',
                         reference) for p in chosen]
            diagonal = np.diag([line.sigma_bq**2 for line in lines])
            covariance = [CovarianceComponent('declared_line_variance', diagonal, source,
                          'independence assumed; reported line errors not decomposed'),
                          CovarianceComponent('shared_calibration', None, 'UNKNOWN',
                          'source calibration covariance unavailable; never treated as zero')]
            methods = [combine_activity_lines(lines, method=method,
                       uncertainty_definition='conditional reported line errors; missing shared calibration',
                       engine_identity=engine_sha, analysis_role='method_control',
                       covariance_components=covariance, allow_incomplete_uncertainty=True)
                       for method in METHODS]
            rows.append(dict(sample_id=artifact['sample_id'], isotope=isotope,
                             line_set_source_sha256=source, methods=methods))
    return rows


def run(output, skip_campaigns=False):
    driver = load_driver()
    checked = driver.verify_inputs(ROOT, driver.MANIFEST_PATH)
    runtime = driver.runtime_compatibility(ROOT)
    if runtime['status'] != 'COMPATIBLE':
        raise RuntimeError('Full integrated example requires compatible declared core dependencies: ' + str(runtime))
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; originals/results are never overwritten')
    output.mkdir(parents=True)
    engine = driver.engine_identity(ROOT)
    components = output / 'components'
    components.mkdir()
    jobs = [
        ('qg_protocol', 'examples/qg_protocol/run_example.py', ['--ambient','off','--continuum','on','--aggregation','historical_activity_over_sigma']),
        ('efficiency', 'examples/efficiency_audit/run_audit.py', ['--selected-method','south_source_export_percent']),
        ('combination', 'examples/activity_combination/compare_same_lines.py', []),
        ('joint_poisson', 'examples/RAFM_irradiation/joint_poisson_pilot.py', []),
        ('multiplet', 'examples/validation/multiplet_240/run_example.py', []),
        ('repeated_counts', 'examples/RAFM_irradiation/repeated_count_validation/run_example.py', []),
    ]
    executions = []
    for name, script, args in jobs:
        target = components / (name if name == 'efficiency' else name + '.json')
        result = subprocess.run([sys.executable, str(ROOT/script), '--output', str(target), *args],
                                cwd=ROOT, env=component_environment(),
                                capture_output=True, text=True, encoding='utf-8', timeout=600)
        (components / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        executions.append(dict(component=name, script=script, returncode=result.returncode,
                               script_sha256=hashlib.sha256((ROOT/script).read_bytes()).hexdigest(),
                               output=target.relative_to(output).as_posix()))
        if result.returncode:
            dump(output/'COMPONENT_FAILURE.json', dict(executions=executions, engine=engine))
            raise RuntimeError(name + ' failed; see component log')
    native = output / 'native_only'
    native.mkdir()
    native_receipts = native_only_controls(driver, checked, native)
    campaigns = {}
    if not skip_campaigns:
        for mode in ('south_native', 'ambient_off'):
            destination = output / mode
            campaigns[mode] = driver.run(ROOT, destination, background_mode=mode,
                                         raw_independent=True, generate_plots=False)
            combos = current_line_combinations(destination, engine['source_sha256'])
            dump(destination/'CURRENT_LINE_COMBINATIONS.json', combos)
    rows = []
    for mode in campaigns:
        path = output/mode/'raw_replay/tables/isotope_comparison.csv'
        if not path.exists():
            raise FileNotFoundError('Completed campaign lacks activity comparison table: ' + str(path))
        with path.open(newline='', encoding='utf-8') as stream:
            scenario_rows = [{'scenario':mode, **row} for row in csv.DictReader(stream)]
        if not scenario_rows:
            raise ValueError('Completed campaign has an empty activity comparison table: ' + str(path))
        rows.extend(scenario_rows)
    if rows:
        with (output/'CAMPAIGN_ACTIVITY_COMPARISONS.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    after = driver.verify_inputs(ROOT, driver.MANIFEST_PATH)
    if after['manifest_sha256'] != checked['manifest_sha256']:
        raise ValueError('Dataset changed during execution')
    if driver.engine_identity(ROOT) != engine:
        raise ValueError('Engine source changed during execution')
    receipt = dict(status='SOFTWARE_COMPONENTS_AND_CAMPAIGNS_COMPLETED' if campaigns else 'BOUNDED_COMPONENTS_COMPLETED',
                   source_counts=checked['observed'], dataset_manifest_sha256=checked['manifest_sha256'],
                   engine=engine, runtime=runtime, components=executions, native_only=native_receipts,
                   campaigns={k:v['raw_workflow_summary'] for k,v in campaigns.items()},
                   scientific_admission=False, exact_vendor_parity=False,
                   limitations=['South temporal applicability unresolved; ambient-off is a saved-state reproduction control',
                       'QG error/library/model details unknown; raw count-average vs report time references remain distinct',
                       'Advanced component results retain failure, conditional and non-identifiability states',
                       'Calibration covariance unavailable; method comparisons do not qualify neutron-flux inversions'])
    dump(output/'INTEGRATED_METHODS_RECEIPT.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-campaigns', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    receipt = run(args.output.resolve(), args.skip_campaigns)
    print(json.dumps(dict(status=receipt['status'], source_counts=receipt['source_counts'],
                          scientific_admission=receipt['scientific_admission'])))
