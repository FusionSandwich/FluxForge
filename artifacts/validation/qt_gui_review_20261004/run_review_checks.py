"""Bounded checks for the current science/Qt integration; run from repo root."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), FLUXFORGE_OFFLINE="1",
           MPLBACKEND="Agg", QT_QPA_PLATFORM="windows", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
groups = {
    "calibration": ["tests/test_gui_calibration_target.py", "tests/test_calibration_workspace_qt.py",
                    "tests/test_workspace_session_qt.py"],
    "gui": ["tests/test_gui_consolidation.py", "tests/test_modern_gui_shell.py",
            "tests/test_gui_report_bundle.py", "tests/test_instrument_provenance.py",
            "tests/test_production_gui_mode.py", "tests/test_reaction_rate_workspace_qt.py",
            "tests/test_irradiation_history_workspace_qt.py",
            "tests/test_spectrum_file_queue_workspace_qt.py"],
    "science": ["tests/test_reference_parity_runner.py", "tests/test_rafm_workflow.py",
                "tests/test_rafm_validation_independence.py", "tests/test_flux_wire_sample_sources.py",
                "tests/test_joint_peak_covariance.py", "tests/test_astm_e261.py",
                "tests/test_astm_e262.py", "tests/test_reporting.py"],
}
groups['shell'] = groups['gui'][:2]
groups['exports'] = groups['gui'][2:4] + groups['gui'][5:]
groups['production'] = groups['gui'][4:5]
groups['targets'] = ['tests/test_gui_calibration_target.py']
groups['calibration_dialog'] = ['tests/test_calibration_workspace_qt.py']
groups['sessions'] = ['tests/test_workspace_session_qt.py']
receipt_path = OUT / (('checks-' + '-'.join(sys.argv[1:])) if sys.argv[1:] else 'checks')
receipts = []
for name, tests in groups.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    start = time.monotonic()
    source_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args = tests + ["-vv", f"--junitxml={OUT / (name + '.xml')}"]
    # Keep real user settings separate from all Qt test windows.
    code = (
        "from PySide6.QtCore import QSettings; import pytest, sys; "
        "QSettings.setDefaultFormat(QSettings.IniFormat); "
        f"QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, {str(OUT / 'settings')!r}); "
        "sys.exit(pytest.main(sys.argv[1:]))"
    )
    command = [sys.executable, "-u", "-c", code, *args]
    try:
        with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=360)
        receipt = {"group": name, "exit_code": result.returncode, "tests": tests,
                   "elapsed_seconds": round(time.monotonic() - start, 2)}
    except subprocess.TimeoutExpired:
        receipt = {"group": name, "timeout_seconds": 360, "tests": tests}
    receipt['source_revision'] = source_revision
    receipts.append(receipt)
    receipt_path.with_suffix('.json').write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)
sys.exit(0 if all(item.get('exit_code') == 0 for item in receipts) else 1)
