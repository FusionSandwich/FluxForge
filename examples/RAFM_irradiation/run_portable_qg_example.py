"""Offline, repository-relative QuantumGold input audit and full software replay.

No download, native Quantum installation, drive letter or historical source path
is required. Run --verify-only with the Python standard library; the full replay
uses FluxForge's normal declared runtime dependencies.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
from importlib import metadata as distribution_metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tomllib

REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / 'examples/RAFM_irradiation/quantumgold_reference/manifest.json'


def engine_identity(repo: Path) -> dict:
    """Hash executable FluxForge sources independently of the input dataset.

    Git metadata is deliberately unnecessary: a relocated source archive has
    the same identity, while an edited analysis file receives a new identity.
    """
    package = repo/'src/fluxforge'
    required = [repo/'pyproject.toml', repo/'examples/RAFM_irradiation/run_portable_qg_example.py']
    integrated_driver = repo/'examples/RAFM_irradiation/run_integrated_qg_methods.py'
    if integrated_driver.is_file():
        required.append(integrated_driver)
    if not package.is_dir() or any(not path.is_file() for path in required):
        return dict(status='UNKNOWN', source_sha256=None, revision=None,
                    reason='Complete engine source tree or pyproject.toml is unavailable')
    files = required + [path for path in package.rglob('*') if path.is_file()
                        and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(repo).as_posix()):
        name = path.relative_to(repo).as_posix().encode('utf-8')
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(4, 'big'))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)
    return dict(status='IDENTIFIED_BY_CONTENT', source_sha256=digest.hexdigest(),
                revision=None, file_count=len(files),
                scope='pyproject.toml, replay drivers present in this source, complete src/fluxforge tree',
                algorithm='SHA-256 over sorted relative path and file bytes, length-prefixed')


def _numeric_version(value: str):
    return tuple(int(piece) for piece in value.split('.')) if re.fullmatch(r'\d+(?:\.\d+)*', value) else None


def _declared_range_status(installed: str, specifier: str) -> str:
    actual = _numeric_version(installed)
    if actual is None:
        return 'UNKNOWN'
    for constraint in specifier.split(','):
        match = re.fullmatch(r'\s*(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)\s*', constraint)
        if match is None:
            return 'UNKNOWN'
        operator, expected_text = match.groups()
        expected = _numeric_version(expected_text)
        width = max(len(actual), len(expected))
        left, right = actual + (0,)*(width-len(actual)), expected + (0,)*(width-len(expected))
        accepted = {'>=':left >= right, '<=':left <= right, '==':left == right,
                    '!=':left != right, '>':left > right, '<':left < right}[operator]
        if not accepted:
            return 'INCOMPATIBLE'
    return 'COMPATIBLE'


def runtime_compatibility(repo: Path) -> dict:
    """Report installed core distributions against this source's declarations.

    This uses only the standard library, so --verify-only keeps its light gate.
    Optional GUI, ML, reporting and development extras are outside the core run.
    """
    project_file = repo/'pyproject.toml'
    if not project_file.is_file():
        return dict(status='UNKNOWN', reason='pyproject.toml is unavailable', packages={})
    project = tomllib.loads(project_file.read_text(encoding='utf-8'))['project']
    python_spec = project['requires-python']
    python_version = '.'.join(str(value) for value in sys.version_info[:3])
    python_status = _declared_range_status(python_version, python_spec)
    packages = {}
    for requirement in project['dependencies']:
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(.*)', requirement)
        name, specifier = match.groups()
        try:
            installed = distribution_metadata.version(name)
            status = _declared_range_status(installed, specifier)
        except distribution_metadata.PackageNotFoundError:
            installed, status = None, 'MISSING'
        packages[name.lower()] = dict(installed_version=installed, declared=specifier,
                                      status=status)
    statuses = [python_status] + [value['status'] for value in packages.values()]
    overall = ('INCOMPATIBLE' if 'INCOMPATIBLE' in statuses else
               'UNKNOWN' if any(value != 'COMPATIBLE' for value in statuses) else 'COMPATIBLE')
    return dict(status=overall, python=dict(version=python_version, declared=python_spec,
                                           status=python_status, executable=sys.executable),
                packages=packages, version_source='importlib.metadata in executing interpreter',
                scope='declared core dependencies only')


def bound_path(repo: Path, relative: str) -> Path:
    if not relative or '\\' in relative or ':' in relative or Path(relative).is_absolute():
        raise ValueError('Resource must use a repository-relative POSIX path')
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError('Resource escapes repository')
    return path


def read_asc_counts(path: Path) -> tuple[int, ...]:
    counts = []
    in_data = False
    for line in path.read_text(encoding='cp1252').splitlines():
        if 'Channel' in line and 'Contents' in line:
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError('Unexpected ASC channel record')
        channel, count = map(int, parts)
        if channel != len(counts) or count < 0:
            raise ValueError('Noncontiguous channel axis or negative counts')
        counts.append(count)
    if len(counts) != 8192:
        raise ValueError('ASC must contain exactly 8192 channels')
    return tuple(counts)


def verify_inputs(repo: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'fluxforge-portable-quantumgold-reference-v1':
        raise ValueError('Unsupported reference manifest')
    resources = {}
    for resource in manifest['resources']:
        name = resource['path']
        if name in resources:
            raise ValueError('Duplicate resource path')
        path = bound_path(repo, name)
        payload = path.read_bytes()
        if len(payload) != resource['bytes'] or hashlib.sha256(payload).hexdigest() != resource['sha256']:
            raise ValueError('Source hash/size mismatch: ' + name)
        resources[name] = resource
    rows = manifest['measurements']
    baseline = bound_path(repo, manifest['baseline_root'])
    label_path = manifest['identity_labels_path']
    if label_path not in resources:
        raise ValueError('Identity label table must be source-pinned')
    labels = {r['measurement_id']:r for r in json.loads(bound_path(repo,label_path).read_text(encoding='utf-8'))}
    metadata = {r['measurement_id']:r for r in json.loads((baseline/'inputs/count_metadata_matrix_32_2026-09-27.json').read_text(encoding='utf-8'))['counts']}
    chronology = {r['measurement_id']:r for r in json.loads((baseline/'inputs/chronology_join_receipt.json').read_text(encoding='utf-8'))['rows']}
    if len(rows) != 32 or len({r['measurement_id'] for r in rows}) != 32:
        raise ValueError('Wrong or duplicate measurement identities')
    arrays, report_sources = {}, {}
    for r in rows:
        mid = r['measurement_id']
        if mid not in labels or mid not in metadata or mid not in chronology:
            raise ValueError('Measurement absent from source-bound identity tables')
        if any(r.get(k) != v for k,v in labels[mid].items()):
            raise ValueError('Manifest labels differ from source identity: '+mid)
        if r['timeline'] != chronology[mid]:
            raise ValueError('Manifest timing differs from source chronology: '+mid)
        if r['QG_header'] != metadata[mid].get('QG_report_header') or r['ASC_header'] != metadata[mid].get('ASC_header'):
            raise ValueError('Manifest headers differ from source metadata: '+mid)
        for name in r['files'].values():
            if name is not None and name not in resources:
                raise ValueError('Measurement file is not source-pinned')
        for kind,name in r['files'].items():
            expected = metadata[mid]['source_sha256'].get(kind)
            if (resources[name]['sha256'] if name else None) != expected:
                raise ValueError('Manifest source file misbound: '+mid+' '+kind)
        ans_path = bound_path(repo, r['files']['ANS'])
        ans = ans_path.read_bytes()
        if r['native_payload_offset'] != 1548 or r['channel_count'] != 8192 or len(ans) < 34316:
            raise ValueError('Unexpected empirical native spectrum layout')
        counts = struct.unpack_from('<8192I', ans, 1548)
        if list(struct.unpack_from('<3f',ans,424)) != r['native_energy_polynomial_keV']:
            raise ValueError('Native energy coefficients misbound')
        if hashlib.sha256(struct.pack('<8192I', *counts)).hexdigest() != r['channel_array_sha256']:
            raise ValueError('Native channel array identity changed')
        if r['files']['ASC'] and read_asc_counts(bound_path(repo, r['files']['ASC'])) != counts:
            raise ValueError('ASC/ANS array mismatch: ' + r['measurement_id'])
        arrays[r['measurement_id']] = counts
        if r['files']['QG_report']:
            digest = resources[r['files']['QG_report']]['sha256']
            if digest in report_sources:
                raise ValueError('One QG report bound to multiple measurements')
            report_sources[digest] = (r['measurement_id'], bound_path(repo, r['files']['QG_report']))
    roi_path = baseline/'inputs/report_roi_rows.csv'
    with roi_path.open(encoding='utf-8', newline='') as f:
        roi_rows = list(csv.DictReader(f))
    summaries = json.loads((baseline/'inputs/reported_summary_rows.json').read_text(encoding='utf-8'))
    seen = set()
    for r in roi_rows + summaries:
        mid, path = report_sources[r['report_sha256']]
        key = (r['measurement_id'], int(r['line_number']))
        if mid != r['measurement_id'] or key in seen:
            raise ValueError('Misbound or duplicate extracted report row')
        seen.add(key)
        original = path.read_bytes().decode('cp1252').splitlines()[int(r['line_number'])-1]
        if original != r['original_line']:
            raise ValueError('Extracted text does not match source report line')
    observed = dict(physical_counts=len(rows), original_ANS=len(arrays),
                    original_ASC=sum(bool(r['files']['ASC']) for r in rows),
                    original_QG_reports=len(report_sources), extracted_ROI_rows=len(roi_rows),
                    extracted_nuclide_summaries=len(summaries))
    if observed != manifest['completeness']:
        raise ValueError('Manifest completeness disagrees with its actual sources')
    supplement_path = repo/'examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/manifest.json'
    supplement = json.loads(supplement_path.read_text(encoding='utf-8'))
    supplemental_blobs = {}
    for pin in supplement['resources']:
        blob = bound_path(repo,pin['path']).read_bytes()
        if len(blob)!=pin['bytes'] or hashlib.sha256(blob).hexdigest()!=pin['sha256']:
            raise ValueError('Supplemental source hash/size mismatch')
        supplemental_blobs[pin['role']] = blob
    older = json.loads(supplemental_blobs['source_bound_older_QG_extraction'].decode('utf-8'))
    original = supplemental_blobs['older_RAFM1_unpaired_QG_report']
    if hashlib.sha256(original).hexdigest()!=older['source_sha256']:
        raise ValueError('Older report/extraction binding mismatch')
    lines = original.decode('cp1252').splitlines()
    if len(lines)!=len(older['rows']) or any(lines[r['line_number']-1]!=r['original_line'] for r in older['rows']):
        raise ValueError('Older report extraction incomplete or misbound')
    supplemental_audit = dict(resources_hash_checked=len(supplement['resources']),
                              older_report_ROI_rows=sum(r['classification']=='ROI' for r in older['rows']),
                              older_report_summaries=sum(r['classification']=='nuclide_summary' for r in older['rows']),
                              South_native_background_preserved=True, primary_workflow_background_changed=False,
                              manifest_sha256=hashlib.sha256(supplement_path.read_bytes()).hexdigest())
    if supplemental_audit['older_report_ROI_rows']!=12 or supplemental_audit['older_report_summaries']!=6:
        raise ValueError('Older report extraction completeness changed')
    return dict(manifest=manifest, arrays=arrays, observed=observed,
                supplemental_audit=supplemental_audit,
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                resource_count=len(resources), source_lines_verified=len(seen))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def label_replay_outputs(output: Path, manifest: dict) -> dict:
    """Bind exported group labels to the source roster, preserving legacy labels.

    The maintained workflow may infer RAFM1 from a monitor's RAFM-1 filename.
    Only group-label fields are corrected here; all numerical results and timing
    assumptions are unchanged. Source inputs and older RAFM1 results are untouched.
    """
    roster = {r['workflow_stem']:r for r in manifest['measurements']}
    corrections = []

    def walk(value, inherited_id=None, source=''):
        if isinstance(value, list):
            for item in value:
                walk(item, inherited_id, source)
        elif isinstance(value, dict):
            sample = value.get('sample_id', inherited_id)
            r = roster.get(sample)
            if r is not None and 'sample_group' in value:
                expected = 'flux_wires' if r['sample_kind']=='monitor' else r['cohort']
                if value['sample_group'] != expected:
                    corrections.append(dict(file=source, measurement_id=r['measurement_id'],
                                            legacy_group=value['sample_group'], group=expected))
                    value['legacy_sample_group'] = value['sample_group']
                    value['sample_group'] = expected
                    value['sample_group_label_basis'] = 'source_bound_campaign_manifest'
            for item in list(value.values()):
                if isinstance(item, (dict,list)):
                    walk(item, sample, source)

    for folder in ('raw_replay','qg_report_replay'):
        for path in (output/folder).rglob('*_comparison.txt'):
            r = roster.get(path.stem.removesuffix('_comparison'))
            if r is None:
                continue
            expected = 'flux_wires' if r['sample_kind']=='monitor' else r['cohort']
            lines = path.read_text(encoding='utf-8').splitlines()
            for i,line in enumerate(lines):
                if line.startswith('Sample group: ') and line != 'Sample group: '+expected:
                    old = line.removeprefix('Sample group: ')
                    lines[i:i+1] = ['Sample group: '+expected,'Legacy sample group: '+old]
                    corrections.append(dict(file=path.relative_to(output).as_posix(),
                                            measurement_id=r['measurement_id'],legacy_group=old,group=expected))
                    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
                    break
        for path in (output/folder).rglob('*.json'):
            value = json.loads(path.read_text(encoding='utf-8'))
            before = len(corrections)
            walk(value, source=path.relative_to(output).as_posix())
            if len(corrections) != before:
                dump(path,value)
        for path in (output/folder).rglob('*.csv'):
            with path.open(encoding='utf-8',newline='') as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                if not fields or 'sample_group' not in fields or 'sample_id' not in fields:
                    continue
                rows = list(reader)
            before = len(corrections)
            for r in rows:
                walk(r,source=path.relative_to(output).as_posix())
            if len(corrections) != before:
                fields += [name for name in ('legacy_sample_group','sample_group_label_basis') if name not in fields]
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer = csv.DictWriter(f,fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
    result = dict(basis='source_bound_campaign_manifest', numerical_results_changed=False,
                  corrections=corrections)
    dump(output/'OUTPUT_LABEL_RECONCILIATION.json',result)
    return result





def background_scenario(repo: Path, checked: dict, mode: str):
    """Explicit current-engine controls; later South remains conditional."""
    if mode == 'north_historical':
        return None, dict(mode=mode, detector='North', applicability='cross-detector sensitivity only')
    if mode not in ('south_native', 'ambient_off'):
        raise ValueError('Unsupported background scenario')
    import numpy as np
    from fluxforge.io.spe import GammaSpectrum
    supplement = json.loads((repo/'examples/RAFM_irradiation/quantumgold_reference/'
                             'supplemental_inputs/manifest.json').read_text())
    pins = [p for p in supplement['resources']
            if p['role'] == 'recovered_South_native_background_not_ASC']
    if len(pins) != 1:
        raise ValueError('South source identity is ambiguous')
    pin = pins[0]
    blob = bound_path(repo, pin['path']).read_bytes()
    if len(blob) != pin['bytes'] or hashlib.sha256(blob).hexdigest() != pin['sha256']:
        raise ValueError('South source identity mismatch')
    if len(blob) != 36616 or b'South 4 hr background terminal' not in blob[:1548]:
        raise ValueError('Unsupported South background layout')
    coefficients = struct.unpack_from('<3f', blob, 424)
    live, real = struct.unpack_from('<d', blob, 104)[0], struct.unpack_from('<d', blob, 96)[0]
    serial = struct.unpack_from('<d', blob, 80)[0]
    counts = np.asarray(struct.unpack_from('<8192I', blob, 1548), dtype=float)
    channels = np.arange(8192)
    energies = coefficients[0] + coefficients[1]*channels + coefficients[2]*channels**2
    if live != 14400 or not live <= real < 14500 or not 40000 < serial < 50000:
        raise ValueError('Unsupported South background timing')
    if int(counts.sum()) != 543427 or not np.all(np.diff(energies) > 0):
        raise ValueError('Unsupported South count/calibration anchors')
    start = datetime(1899, 12, 30) + timedelta(days=serial)
    details = dict(mode=mode, detector='South', source_sha256=pin['sha256'],
                   start_time_unzoned=start.isoformat(), live_time_s=live,
                   temporal_applicability='UNRESOLVED',
                   calibration='native polynomial; count-conserving bin overlap and covariance')
    if mode == 'ambient_off':
        counts = np.zeros_like(counts)
        details = dict(mode=mode, source_sha256=None, synthetic=True,
                       interpretation='no separate measured ambient; local continuum retained',
                       evidence='saved header inference; final report setting unknown')
        start, real = None, live
    spectrum = GammaSpectrum(counts=counts, channels=channels, energies=energies,
                             live_time=live, real_time=real, start_time=start,
                             spectrum_id='South measured' if mode == 'south_native' else 'synthetic zero ambient control',
                             detector_id='South' if mode == 'south_native' else 'synthetic',
                             calibration={'energy': list(coefficients)}, metadata=details)
    return spectrum, details



def run(repo: Path, output: Path | None, verify_only: bool = False, *,
        background_mode: str = 'north_historical', raw_independent: bool = False,
        generate_plots: bool = True) -> dict:
    checked = verify_inputs(repo, repo/'examples/RAFM_irradiation/quantumgold_reference/manifest.json')
    manifest = checked['manifest']
    receipt = dict(status='INPUT_AUDIT_PASS', mode='reference_reproduction',
                   comparison_basis='QG report reproduction; not independent raw activity agreement',
                   counts=checked['observed'],
                   resources_hash_checked=checked['resource_count'],
                   original_report_lines_verified=checked['source_lines_verified'],
                   supplemental_audit=checked['supplemental_audit'],
                   required_external_data_paths=[], requires_quantumgold_installation=False,
                   independent_absolute_qualification=False,
                   dataset_identity=dict(status='SOURCE_BOUND',
                                         source_manifest_sha256=checked['manifest_sha256'],
                                         revision=None),
                   engine_identity=engine_identity(repo),
                   runtime_compatibility=runtime_compatibility(repo))
    if verify_only:
        return receipt
    if output is None:
        raise ValueError('Full replay requires an explicit --output directory')
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing data are not overwritten')
    # Native spectra are parsed empirically for this pinned dataset only. Export
    # all arrays, including the two native-only counts, without calling them ASC.
    output.mkdir(parents=True)
    dump(output/'INPUT_AUDIT.json', receipt)
    native_output = output/'native_channel_arrays'
    native_output.mkdir()
    for mid, counts in checked['arrays'].items():
        with (native_output/(mid+'.csv')).open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('channel', 'counts'))
            writer.writerows(enumerate(counts))
    baseline_src = bound_path(repo, manifest['baseline_root'])
    baseline_out = output/'working_baseline'
    shutil.copytree(baseline_src, baseline_out, ignore=shutil.ignore_patterns('__pycache__'))
    spec = importlib.util.spec_from_file_location('portable_qg_baseline', baseline_out/'build_baseline.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.build()
    # Stage exact authoritative files under their maintained workflow stems.
    staged = output/'workflow_inputs'
    runtime = bound_path(repo, manifest['runtime_root'])
    shutil.copytree(runtime/'metadata', staged/'metadata')
    shutil.copyfile(runtime/'background.ASC', staged/'background.ASC')
    wire_root = staged/'raw_gamma_spec/flux_wires'
    wire_root.mkdir(parents=True)
    shutil.copyfile(runtime/'spectrum_vit_j.csv', wire_root/'spectrum_vit_j.csv')
    for r in manifest['measurements']:
        group = 'flux_wires' if r['sample_kind']=='monitor' else r['cohort']
        for kind, folder, extension in [('ASC','raw_gamma_spec','.ASC'), ('QG_report','QG_processed_gamma_data','.txt')]:
            if not r['files'][kind]:
                continue
            target = staged/folder/group/(r['workflow_stem']+extension)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bound_path(repo, r['files'][kind]), target)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(repo/'src'))
    from fluxforge.examples.rafm_workflow import run_rafm_validation, run_qg_benchmark
    # Software execution and physical agreement are separate. Preserve all
    # failures and admission exclusions; do not turn --no-fail into validation.
    kwargs = {}
    if background_mode != 'north_historical':
        background, background_details = background_scenario(repo, checked, background_mode)
        kwargs['background_spectrum_override'] = background
    else:
        background_details = dict(mode=background_mode, applicability='cross-detector sensitivity only')
    if raw_independent:
        kwargs.update(flux_wire_counting_method='iec_tiered',
                      generic_targeted_counting_method='iec_tiered')
    if not generate_plots:
        kwargs['generate_plots'] = False
    raw = run_rafm_validation(staged, output/'raw_replay', enforce_thresholds=False, **kwargs)
    report = run_qg_benchmark(staged, output/'qg_report_replay')
    label_receipt = label_replay_outputs(output,manifest)
    report = json.loads((output/'qg_report_replay/qg_benchmark_summary.json').read_text(encoding='utf-8'))
    receipt.update(status='SOFTWARE_REPLAY_COMPLETED',
                   comparison_modes=dict(reference_reproduction='executed',
                                         raw_vs_report='not_run',
                                         same_count_QG_derived='baseline_artifact_only'),
                   source_input_completeness=checked['observed'],
                   raw_workflow_summary=raw, qg_report_workflow_summary=report,
                   native_only_counts=[r['measurement_id'] for r in manifest['measurements'] if not r['files']['ASC']],
                   missing_QG_reports=[r['measurement_id'] for r in manifest['measurements'] if r['missing_report']],
                   native_only_policy='arrays exported; no fictitious ASC export or wrong-geometry raw activity',
                   scientific_status='QG-conditioned example/diagnostics; preserve threshold failures and admission exclusions',
                   software_python=sys.version, source_manifest_sha256=checked['manifest_sha256'])
    receipt['output_label_reconciliation'] = label_receipt
    receipt['background_scenario'] = background_details
    receipt['raw_reference_used_for_analysis'] = not raw_independent
    if raw_independent:
        receipt['mode'] = 'independent_raw_comparison_and_separate_report_reproduction'
        receipt['comparison_basis'] = 'raw IEC counts and separate QG report replay; source-library/profile inputs remain conditional'
        receipt['comparison_modes']['raw_vs_report'] = 'executed'
    dump(output/'REPLAY_RECEIPT.json', receipt)
    return receipt


class SouthWorkingCurve:
    """Positive, adjacent brackets of the recovered South 25 cm percent export."""

    relative_uncertainty = 0.0  # Not a calibration uncertainty estimate.

    def __init__(self, path: Path):
        import numpy as np

        self.path = path
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        segments, segment = [], []
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                energy = float(row['energy_keV'])
                fraction = float(row['efficiency_fraction'])
                original = float(row['original_efficiency_value'])
                valid = (row['status'] == 'working_percent_assumption' and
                         np.isfinite(fraction) and 0 < fraction <= 1 and
                         np.isclose(fraction, original/100, rtol=1e-12, atol=1e-15))
                if row['status'] == 'working_percent_assumption' and not valid:
                    raise ValueError('Recovered curve violates its percent interpretation')
                if valid:
                    if segment and energy <= segment[-1][0]:
                        raise ValueError('Recovered curve energies must increase')
                    segment.append((energy, fraction))
                elif segment:
                    segments.append(segment)
                    segment = []
        if segment:
            segments.append(segment)
        if not segments:
            raise ValueError('Recovered curve has no positive adjacent brackets')
        self.segments = segments
        self.energy_min_keV = min(part[0][0] for part in segments)
        self.energy_max_keV = max(part[-1][0] for part in segments)

    def verify_source_export(self, original_path: Path) -> None:
        """Reject a derived table that differs from the hash-bound raw export."""
        expected = []
        with original_path.open(encoding='utf-8-sig', newline='') as handle:
            for line, cells in enumerate(csv.reader(handle), 1):
                try:
                    energy, value = float(cells[0]), float(cells[1])
                except (ValueError, IndexError):
                    continue
                expected.append((energy, value/100, value, line,
                                 'working_percent_assumption' if value > 0 else 'excluded_nonpositive'))
        with self.path.open(encoding='utf-8', newline='') as handle:
            actual = [(float(row['energy_keV']), float(row['efficiency_fraction']),
                       float(row['original_efficiency_value']), int(row['source_line']), row['status'])
                      for row in csv.DictReader(handle)]
        if actual != expected:
            raise ValueError('Recovered curve differs from source-bound original percent export')

    def efficiency(self, energy_keV):
        import numpy as np

        scalar = np.isscalar(energy_keV)
        energies = np.atleast_1d(np.asarray(energy_keV, dtype=float))
        values = np.full(energies.shape, np.nan)
        for segment in self.segments:
            x, y = np.asarray(segment, dtype=float).T
            inside = np.isfinite(energies) & (energies >= x[0]) & (energies <= x[-1])
            values[inside] = np.interp(energies[inside], x, y)
        if scalar:
            if not np.isfinite(values[0]):
                raise ValueError('Energy is outside positive adjacent South 25 cm curve brackets')
            return float(values[0])
        return values

    def to_dict(self):
        return dict(model='recovered_South_small_vial_25cm',
                    source_sha256=self.source_sha256,
                    source_path=self.path.name,
                    export_unit_assumption='percent', applied_unit='fraction',
                    energy_range_keV=[self.energy_min_keV, self.energy_max_keV],
                    positive_bracket_segments=len(self.segments),
                    interpolation='linear within positive adjacent brackets only',
                    calibration_uncertainty='unknown; not treated as zero')


def run_raw_comparison(repo: Path, output: Path, measurement_id: str,
                       efficiency_mode: str = 'south_recovered',
                       counting_method: str = 'iec_tiered',
                       background_mode: str = 'north_historical') -> dict:
    """One source-bound South 25 cm monitor reduction, independent of QG activity."""
    if counting_method not in ('iec_tiered', 'covell', 'gilmore'):
        raise ValueError('Raw comparison requires iec_tiered, covell, or gilmore counting')
    if efficiency_mode not in ('south_recovered', 'legacy_profile'):
        raise ValueError('Unknown raw efficiency mode')
    if background_mode not in ('north_historical', 'south_native', 'ambient_off'):
        raise ValueError('Unknown raw background mode')
    checked = verify_inputs(repo, repo/'examples/RAFM_irradiation/quantumgold_reference/manifest.json')
    manifest = checked['manifest']
    rows = [row for row in manifest['measurements'] if row['measurement_id'] == measurement_id]
    if len(rows) != 1:
        raise ValueError('Unknown measurement identity')
    row = rows[0]
    if row['sample_kind'] != 'monitor' or not row['workflow_stem'].endswith('_25cm'):
        raise ValueError('Only source-labeled South 25 cm monitors are eligible; near-contact is excluded')
    if row['measurement_id'] in {'Ti-RAFM-1', 'Ti-RAFM-1a', 'Ti-RAFM-1b'}:
        raise ValueError('Ti-produced Sc-48 printed intensity ambiguity needs a separate declared scenario')
    if not row['files']['ASC']:
        raise ValueError('Raw comparison requires an original source-bound ASC spectrum')
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing data are not overwritten')

    import copy
    sys.path.insert(0, str(repo/'src'))
    from fluxforge.analysis.flux_wire_analysis import (
        analyze_flux_wire_targeted, build_gamma_library, get_expected_isotopes)
    from fluxforge.examples import rafm_workflow as workflow
    from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc
    from fluxforge.io.spe import GammaSpectrum
    import numpy as np

    runtime = bound_path(repo, manifest['runtime_root'])
    metadata = workflow.load_rafm_example_metadata(runtime)
    energy_override = workflow.workflow_profile_energy_calibration(metadata.config)
    raw_path = bound_path(repo, row['files']['ASC'])
    raw_data = read_raw_asc(raw_path, energy_calibration_override=energy_override,
                            profile_name=metadata.config['profile_name'])
    if raw_data.spectrum is None:
        raise ValueError('ASC spectrum could not be parsed')
    raw_data.sample_id = row['workflow_stem']
    raw_data.spectrum.spectrum_id = row['workflow_stem']
    if background_mode == 'ambient_off':
        background = None
        background_details = dict(mode=background_mode, detector=None,
                                  source_sha256=None, QG_background_match='UNKNOWN',
                                  interpretation='explicit no measured ambient subtraction; local continuum retained',
                                  qualification='saved ambient-off state is not proof of final report processing')
    elif background_mode == 'north_historical':
        background_path = runtime/'background.ASC'
        background_blob = background_path.read_bytes()
        background = read_raw_asc(background_path,
                                  energy_calibration_override=energy_override,
                                  profile_name=metadata.config['profile_name']).spectrum
        if background is None:
            raise ValueError('Historical North background could not be parsed')
        background_details = dict(mode=background_mode, detector='North',
                                  source_sha256=hashlib.sha256(background_blob).hexdigest(),
                                  calibration='historical workflow override',
                                  QG_background_match='UNKNOWN')
    else:
        supplement = json.loads((repo/'examples/RAFM_irradiation/quantumgold_reference/'
                                 'supplemental_inputs/manifest.json').read_text(encoding='utf-8'))
        pins = [item for item in supplement['resources']
                if item['role'] == 'recovered_South_native_background_not_ASC']
        if len(pins) != 1:
            raise ValueError('South native background source identity is ambiguous')
        pin = pins[0]
        background_path = bound_path(repo, pin['path'])
        background_blob = background_path.read_bytes()
        if len(background_blob) != pin['bytes'] or hashlib.sha256(background_blob).hexdigest() != pin['sha256']:
            raise ValueError('South native background source hash/size mismatch')
        if len(background_blob) != 36616 or b'South 4 hr background terminal' not in background_blob[:1548]:
            raise ValueError('Unexpected South native background layout or identity')
        coefficients = struct.unpack_from('<3f', background_blob, 424)
        live_s = struct.unpack_from('<d', background_blob, 104)[0]
        real_s = struct.unpack_from('<d', background_blob, 96)[0]
        serial_day = struct.unpack_from('<d', background_blob, 80)[0]
        if live_s != 14400.0 or not live_s <= real_s < 14500 or not 40000 < serial_day < 50000:
            raise ValueError('Unexpected South native background timing')
        counts = np.asarray(struct.unpack_from('<8192I', background_blob, 1548), dtype=float)
        channels = np.arange(8192)
        energies = coefficients[0] + coefficients[1]*channels + coefficients[2]*channels**2
        if not np.all(np.diff(energies) > 0) or int(counts.sum()) != 543427:
            raise ValueError('Unexpected South native background channels or calibration')
        start = datetime(1899, 12, 30) + timedelta(days=serial_day)
        background = GammaSpectrum(counts=counts, channels=channels, energies=energies,
                                   live_time=live_s, real_time=real_s, start_time=start,
                                   spectrum_id='South 4 hr background terminal', detector_id='South',
                                   calibration={'energy': list(coefficients)},
                                   metadata={'source_file': pin['path'], 'source_sha256': pin['sha256']})
        background_details = dict(mode=background_mode, detector='South',
                                  source_sha256=pin['sha256'], source_path=pin['path'],
                                  native_payload_offset=1548, channel_count=8192,
                                  energy_polynomial_keV=list(coefficients),
                                  live_time_s=live_s, real_time_s=real_s,
                                  start_time_unzoned=start.isoformat(),
                                  total_counts=int(counts.sum()),
                                  calibration='native hash-bound polynomial; current core histogram-overlap alignment',
                                  QG_background_match='UNKNOWN',
                                  temporal_applicability='UNRESOLVED')
    curve_path = bound_path(repo, manifest['baseline_root'])/'south_25cm_recovered_curve.csv'
    curve = SouthWorkingCurve(curve_path)
    curve.verify_source_export(bound_path(repo, manifest['baseline_root'] +
                                         '/inputs/South Small Vial 25cm.csv'))
    config = metadata.config
    energy_low = max(float(config.get('min_peak_energy_keV', 80)), curve.energy_min_keV)
    energy_high = min(float(config.get('max_peak_energy_keV', 3000)), curve.energy_max_keV)
    expected_lines = build_gamma_library(
        isotope_filter=get_expected_isotopes(row['workflow_stem']))
    excluded_expected_lines = [dict(isotope=line.isotope, energy_keV=line.energy_keV,
                                    reason='outside positive South 25 cm curve energy range')
                               for line in expected_lines
                               if not energy_low <= line.energy_keV <= energy_high]

    def analyze(calibration):
        data = copy.deepcopy(raw_data)
        data.efficiency = calibration
        return analyze_flux_wire_targeted(
            data=data, reference_data=None, background_spectrum=background,
            background_scale_mode='live', background_subtract=background is not None,
            profile_name=config['profile_name'], counting_method=counting_method,
            peak_threshold=0.0,
            min_energy_keV=energy_low,
            max_energy_keV=energy_high,
            roi_width_fwhm=float(config.get('flux_wire_roi_width_fwhm', 4.0)),
            background_width_channels=int(config.get('flux_wire_background_width_channels', 1)),
            background_gap_fwhm=float(config.get('flux_wire_background_gap_fwhm', 0.0)),
            comparison_capture_range_channels=int(config.get('flux_wire_comparison_capture_range_channels', 32)),
            broad_peak_ratio_threshold=float(config.get('flux_wire_broad_peak_ratio_threshold', 1.2)),
            broad_peak_net_threshold=float(config.get('flux_wire_broad_peak_net_threshold', 5000)),
            compact_window_edge_penalty=float(config.get('flux_wire_compact_window_edge_penalty', 80)),
            compact_window_asymmetry_penalty=float(config.get('flux_wire_compact_window_asymmetry_penalty', 10)),
            compact_window_width_penalty=float(config.get('flux_wire_compact_window_width_penalty', 120)),
            broad_window_net_agreement_tolerance=float(config.get('flux_wire_broad_window_net_agreement_tolerance', .12)),
            broad_window_max_raw_gross_ratio=float(config.get('flux_wire_broad_window_max_raw_gross_ratio', 1.35)),
            comparison_background_model=str(config.get('flux_wire_comparison_background_model', 'constant')),
        )

    legacy = analyze(raw_data.efficiency)
    selected = analyze(curve if efficiency_mode == 'south_recovered' else raw_data.efficiency)
    def peak_counts(result):
        return {(p.isotope, round(p.energy_keV, 2)): (p.net_counts, p.net_counts_unc)
                for p in result.peaks}
    if peak_counts(legacy) != peak_counts(selected):
        raise ValueError('Efficiency selection changed raw peak counts')
    qg = read_processed_txt(bound_path(repo, row['files']['QG_report']),
                            profile_name=config['profile_name']) if row['files']['QG_report'] else None
    reference = ({n.isotope: n.activity_bq for n in qg.nuclides} if qg else {})
    selected_activities = selected.nuclide_activities
    legacy_activities = legacy.nuclide_activities
    activity_differences = {}
    for isotope in sorted(set(selected_activities) | set(legacy_activities)):
        selected_bq = selected_activities.get(isotope, {}).get('activity_bq')
        legacy_bq = legacy_activities.get(isotope, {}).get('activity_bq')
        activity_differences[isotope] = dict(
            selected_Bq=selected_bq, legacy_Bq=legacy_bq,
            selected_minus_legacy_Bq=(selected_bq-legacy_bq if selected_bq is not None
                                      and legacy_bq is not None else None),
            selected_over_legacy=(selected_bq/legacy_bq if selected_bq is not None
                                  and legacy_bq else None),
            QG_reference_Bq=reference.get(isotope),
            selected_activity_reference=selected_activities.get(isotope, {}).get('activity_reference'),
            QG_activity_reference='reported_measurement_date_count_start',
            QG_comparison_status='UNHARMONIZED_REFERENCE_TIME')
    line_witnesses = []
    for peak in selected.peaks:
        if peak.isotope and curve.energy_min_keV <= peak.energy_keV <= curve.energy_max_keV:
            line_witnesses.append(dict(isotope=peak.isotope, energy_keV=peak.energy_keV,
                                       net_counts=peak.net_counts, efficiency_used=peak.efficiency,
                                       curve_efficiency=curve.efficiency(peak.energy_keV)))
    receipt = dict(status='RAW_COMPARISON_COMPLETED', mode='raw_comparison',
                   comparison_basis='raw counts vs separate QG report; reference activity not used for analysis',
                   comparison_modes=dict(reference_reproduction='not_run',
                                         raw_vs_report='executed',
                                         same_count_QG_derived='not_run'),
                   reference_used_for_analysis=False, independent_absolute_qualification=False,
                   measurement_id=measurement_id, sample_id=row['workflow_stem'],
                   counting_method=counting_method, efficiency_mode=efficiency_mode,
                   available_recovered_curve=curve.to_dict(),
                   selected_curve=(curve.to_dict() if efficiency_mode == 'south_recovered' else None),
                   original_export_sha256=next(item['sha256'] for item in manifest['resources']
                                               if item['path'] == manifest['baseline_root'] +
                                               '/inputs/South Small Vial 25cm.csv'),
                   baseline_builder_sha256=next(item['sha256'] for item in manifest['resources']
                                                if item['path'] == manifest['baseline_root'] +
                                                '/build_baseline.py'),
                   selected_efficiency_source=(curve.to_dict() if efficiency_mode == 'south_recovered'
                                               else dict(model='legacy_profile', profile=config['profile_name'])),
                   raw_source_sha256=next(item['sha256'] for item in manifest['resources']
                                          if item['path'] == row['files']['ASC']),
                   channel_array_sha256=row['channel_array_sha256'],
                   background_source_sha256=background_details['source_sha256'],
                   background_basis=('explicit ambient-off comparison scenario; final QG state unresolved'
                                     if background_mode == 'ambient_off' else
                                     'historical bundled North background' if background_mode == 'north_historical'
                                     else 'recovered native South detector scenario; applicability unresolved'),
                   background_details=background_details,
                   background_processing=dict(
                       measured_background_subtracted=background is not None,
                       scale_mode='live' if background is not None else None,
                       scale_factor=float(raw_data.live_time/background.live_time) if background is not None else None,
                       negative_policy='hybrid',
                       energy_alignment='integrated_counts_bin_overlap_if_needed',
                       covariance='C_sample + scale_factor**2 * W C_background W.T' if background is not None else 'C_sample',
                       local_continuum_model=str(config.get('flux_wire_comparison_background_model','constant'))),
                   energy_scope_keV=[curve.energy_min_keV, curve.energy_max_keV],
                   unsupported_energy_policy='outside positive adjacent brackets excluded from line analysis',
                   excluded_expected_lines=excluded_expected_lines,
                   QG_reference_activities_Bq=reference,
                   QG_reference_time_basis='reported Measurement Date; count-start convention not harmonized with raw count-average',
                   selected_raw_activities=selected_activities,
                   legacy_raw_activities=legacy_activities,
                   selected_vs_legacy_activity=activity_differences,
                   selected_efficiency_line_witnesses=line_witnesses,
                   selected_curve_line_witnesses=(line_witnesses if efficiency_mode == 'south_recovered' else []),
                   counts_same_under_efficiency_selection=True,
                   dataset_identity=dict(status='SOURCE_BOUND',
                                         source_manifest_sha256=checked['manifest_sha256'],revision=None),
                   engine_identity=engine_identity(repo),
                   runtime_compatibility=runtime_compatibility(repo),
                   conditional_calibration=True,
                   uncertainty_basis='activity_unc_bq is conditional on the selected response; calibration uncertainty is unquantified',
                   calibration_limits=['export percent unit is a documented assumption',
                                       'same-count QG agreement is not independent absolute calibration',
                                       'calibration covariance and 2025 active applicability unknown',
                                       'no Sc-48 intensity or timing scenario substitution'])
    output.mkdir(parents=True)
    dump(output/'RAW_SELECTED_ANALYSIS.json', selected.to_dict())
    dump(output/'RAW_LEGACY_ANALYSIS.json', legacy.to_dict())
    dump(output/'REPLAY_RECEIPT.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Stdlib-only source completeness/identity audit')
    parser.add_argument('--output', type=Path, help='Fresh output directory for the complete software example')
    parser.add_argument('--raw-sample', help='Source-bound monitor measurement ID for one independent raw comparison')
    parser.add_argument('--efficiency-mode', choices=['south_recovered','legacy_profile'],
                        default='south_recovered')
    parser.add_argument('--counting-method', choices=['iec_tiered','covell','gilmore'],
                        default='iec_tiered')
    parser.add_argument('--background-mode', choices=['north_historical','south_native','ambient_off'],
                        default='north_historical')
    args = parser.parse_args()
    if args.raw_sample and args.verify_only:
        parser.error('--raw-sample and --verify-only are separate modes')
    if args.raw_sample and not args.output:
        parser.error('--raw-sample requires --output')
    if not args.raw_sample and args.background_mode != 'north_historical':
        parser.error('--background-mode ' + args.background_mode + ' requires --raw-sample')
    result = (run_raw_comparison(REPO, args.output, args.raw_sample,
                                 args.efficiency_mode, args.counting_method,
                                 args.background_mode)
              if args.raw_sample else run(REPO, args.output, args.verify_only))
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('workflow_summary')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
