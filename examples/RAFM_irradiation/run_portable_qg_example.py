"""Offline, repository-relative QuantumGold input audit and full software replay.

No download, native Quantum installation, drive letter or historical source path
is required. Run --verify-only with the Python standard library; the full replay
uses FluxForge's normal declared runtime dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / 'examples/RAFM_irradiation/quantumgold_reference/manifest.json'


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
    return dict(manifest=manifest, arrays=arrays, observed=observed,
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


def run(repo: Path, output: Path | None, verify_only: bool = False) -> dict:
    checked = verify_inputs(repo, repo/'examples/RAFM_irradiation/quantumgold_reference/manifest.json')
    manifest = checked['manifest']
    receipt = dict(status='INPUT_AUDIT_PASS', counts=checked['observed'],
                   resources_hash_checked=checked['resource_count'],
                   original_report_lines_verified=checked['source_lines_verified'],
                   required_external_data_paths=[], requires_quantumgold_installation=False,
                   independent_absolute_qualification=False)
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
    raw = run_rafm_validation(staged, output/'raw_replay', enforce_thresholds=False)
    report = run_qg_benchmark(staged, output/'qg_report_replay')
    label_receipt = label_replay_outputs(output,manifest)
    report = json.loads((output/'qg_report_replay/qg_benchmark_summary.json').read_text(encoding='utf-8'))
    receipt.update(status='SOFTWARE_REPLAY_COMPLETED',
                   source_input_completeness=checked['observed'],
                   raw_workflow_summary=raw, qg_report_workflow_summary=report,
                   native_only_counts=[r['measurement_id'] for r in manifest['measurements'] if not r['files']['ASC']],
                   missing_QG_reports=[r['measurement_id'] for r in manifest['measurements'] if r['missing_report']],
                   native_only_policy='arrays exported; no fictitious ASC export or wrong-geometry raw activity',
                   scientific_status='QG-conditioned example/diagnostics; preserve threshold failures and admission exclusions',
                   software_python=sys.version, source_manifest_sha256=checked['manifest_sha256'])
    receipt['output_label_reconciliation'] = label_receipt
    dump(output/'REPLAY_RECEIPT.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Stdlib-only source completeness/identity audit')
    parser.add_argument('--output', type=Path, help='Fresh output directory for the complete software example')
    args = parser.parse_args()
    result = run(REPO, args.output, args.verify_only)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('workflow_summary')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
