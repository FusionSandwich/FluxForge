"""Index distinct passing cases and verify the built wheel and report bundle."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
groups = ['science', 'shell', 'exports', 'production', 'calibration_dialog', 'sessions', 'targets']
cases = set()
counts = {}
for group in groups:
    tests = ET.parse(OUT / (group + '.xml')).findall('.//testcase')
    assert tests and not any(test.find('failure') is not None or test.find('error') is not None
                             or test.find('skipped') is not None for test in tests)
    counts[group] = len(tests)
    cases.update((test.get('classname'), test.get('name')) for test in tests)
wheel = next((OUT / 'wheel').glob('*.whl'))
with ZipFile(wheel) as archive:
    names = archive.namelist()
    assert any(name.startswith('fluxforge/gui/') for name in names)
    assert not any(name.startswith('fluxforge_gui/') or name.startswith('archive/') for name in names)
    entrypoints = next(name for name in names if name.endswith('.dist-info/entry_points.txt'))
    scripts = archive.read(entrypoints).decode()
    assert 'fluxforge-gui = fluxforge.gui.app:main' in scripts
    assert 'fluxforge-gui-legacy' not in scripts
bundle = OUT / 'native-final/review-report.zip'
with ZipFile(bundle) as archive:
    manifest = json.loads(archive.read('manifest.json'))
    for name, receipt in manifest['files'].items():
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == receipt['sha256']
        assert len(data) == receipt['bytes']
    assert manifest['scientific_admission'] is False
    snapshot = json.loads(archive.read('snapshot.json'))
    assert snapshot['tables'] and snapshot['views']
result = {
    'review_source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'integrated_science_base': '701288a30f418ed7dc331e6deda0044eb220e72e',
    'published_gui_reports_head': 'a85f3bf716155e569a5cadef249e5387ed84d2a3',
    'passing_groups': counts, 'distinct_passing_cases': len(cases),
    'wheel': {'name': wheel.name, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
              'entry_points': scripts, 'legacy_gui_absent': True},
    'bundle': {'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
               'manifest_verified': True, 'scientific_admission': False},
    'limits': ['Targeted checks across integration and repair revisions; not a full-suite receipt.',
               'Two combined native runs timed out at 360 seconds; split groups passed.',
               'Wheel inspected and compiled; Windows installer/frozen EXE not rebuilt.'],
}
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
