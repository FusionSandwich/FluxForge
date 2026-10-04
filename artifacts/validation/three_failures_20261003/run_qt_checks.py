"""Bounded per-group checks against the Qt consolidation revision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

out=Path(__file__).resolve().parent
repo=out.parents[2]
revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
groups={
    "qt_consolidation":["tests/test_gui_consolidation.py"],
    "qt_production":["tests/test_production_gui_mode.py"],
    "qt_shell":["tests/test_modern_gui_shell.py"],
    "qt_new_workspaces":["tests/test_irradiation_history_workspace_qt.py","tests/test_reaction_rate_workspace_qt.py","tests/test_spectrum_file_queue_workspace_qt.py"],
    "qt_calibration":["tests/test_calibration_workspace_qt.py"],
    "qt_sessions":["tests/test_workspace_session_qt.py"],
    "qt_parity":["tests/test_reference_parity_runner.py","tests/test_reference_parity_energy_tolerance.py","tests/test_parity_fixture_manifests.py"],
}
results=[]
label=os.environ.get("QT_CHECK_RUN_LABEL", "")
progress=out/f"{label}qt_progress.jsonl"
def record(row):
    with progress.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)

for name,files in groups.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    name=label+name
    command=[sys.executable,"-u","-m","pytest","-q","-o","faulthandler_timeout=30",*files,f"--junitxml={out/name}.xml"]
    record({"group":name,"status":"started","git_head":revision,"command":command})
    started=time.monotonic()
    try:
        run=subprocess.run(command,cwd=repo,capture_output=True,timeout=int(os.environ.get("QT_CHECK_TIMEOUT", "90")))
        stdout,stderr,code=run.stdout,run.stderr,run.returncode
        timed_out=False
    except subprocess.TimeoutExpired as exc:
        stdout,stderr,code=exc.stdout or b"",exc.stderr or b"",None
        timed_out=True
    (out/f"{name}.log").write_bytes(stdout+b"\nSTDERR:\n"+stderr)
    row={"group":name,"git_head":revision,"exit_code":code,"timed_out":timed_out,"elapsed_s":round(time.monotonic()-started,2),"files":files}
    if (out/f"{name}.xml").exists():
        cases=list(ET.parse(out/f"{name}.xml").getroot().iter("testcase"))
        row.update(cases=len(cases),failed=sum(c.find("failure") is not None or c.find("error") is not None for c in cases),skipped=sum(c.find("skipped") is not None for c in cases))
    results.append(row)
    record(row)
summary={"git_head":revision,"python":sys.executable,"prefix":sys.prefix,"qt_qpa_platform":os.environ.get("QT_QPA_PLATFORM"),"groups":results,"passed":all(r["exit_code"]==0 for r in results),"single_revision_full_suite":False}
(out/f"{label}qt_checks_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
sys.exit(0 if summary["passed"] else 1)
