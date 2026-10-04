"""Package compact original/follow-up evidence without modifying frozen artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

out = Path(__file__).resolve().parent
repo = out.parents[2]
frozen = Path("C:/Users/Josh/.codex/worktrees/issue21-frozen-validation/FluxForge/artifacts/validation")
original_files = [
    frozen / "issue21_full_20261003/core.cases.jsonl",
    frozen / "issue21_full_20261003/environment.json",
    frozen / "issue21_resumed_20261003/gui_39.cases.jsonl",
    frozen / "issue21_resumed_20261003/gui_39.log",
    frozen / "issue21_resumed_20261003/source_environment.json",
    frozen / "issue21_resumed_20261003/summary.json",
]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original_hashes = [{"path":str(p),"sha256":sha(p)} for p in original_files]
failures=[]
for p in original_files:
    if p.suffix == ".jsonl":
        failures.extend(r for r in map(json.loads,p.read_text(encoding="utf-8").splitlines()) if r["outcome"] == "failed")
assert len(failures) == 3
(out / "original_failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
for prefix,target in (("coordinate_guarded_tmp","coordinate_failure.json"),("event_mode_tmp","event_mode_run.json")):
    paths=list((out/prefix).rglob("run.json"))
    assert len(paths)==1
    shutil.copyfile(paths[0],out/target)
    if prefix=="coordinate_guarded_tmp":
        shutil.copyfile(paths[0].with_name("failure.png"),out/"coordinate_failure.png")
shutil.copyfile(out/"coordinate_diagnostic/before-fit.png",out/"coordinate_before_fit.png")
shutil.copyfile(out/"coordinate_diagnostic/clicks.json",out/"coordinate_clicks.json")
def junit(file):
    root=ET.parse(out/file).getroot()
    cases=list(root.iter("testcase"))
    return {"cases":len(cases),"failed":sum(c.find("failure") is not None or c.find("error") is not None for c in cases),"skipped":sum(c.find("skipped") is not None for c in cases),"seconds":round(sum(float(c.get("time","0")) for c in cases),3)}

source = "51654ce6814c0547a52705369018cf495501beeb"
runs=[
    {"name":"frozen parity reproduction","git_head":"36ed262f5fba11c6f9c92d1d182de977b11b565e","junit":"parity_frozen.xml"},
    {"name":"published parity verification","git_head":"3ed85ee0ca9df39f0dbfe50499c485bca45d9d55","junit":"parity_3ed85ee.xml"},
    {"name":"unmodified coordinate replay","git_head":"3ed85ee0ca9df39f0dbfe50499c485bca45d9d55","github_actions":"false","junit":"coordinate_original.xml"},
    {"name":"guarded coordinate replay","git_head_at_launch":"3ed85ee0ca9df39f0dbfe50499c485bca45d9d55","source_equivalent_commit":source,"driver_modified_at_launch":True,"github_actions":"false","junit":"coordinate_guarded.xml","run_json":"coordinate_failure.json"},
    {"name":"event mode replay","git_head":source,"github_actions":"true","junit":"event_mode.xml","run_json":"event_mode_run.json"},
    {"name":"final parity and driver contracts","git_head":source,"junit":"focused_final.xml"},
]
for r in runs:
    r["result"]=junit(r["junit"])
    r["junit_sha256"]=sha(out/r["junit"])
probe=json.loads((out/"parity_probe.json").read_text())
assert probe["passed"] and len(probe["checks"])==25
assert json.loads((out/"coordinate_failure.json").read_text())["status"]=="failed"
assert json.loads((out/"event_mode_run.json").read_text())["status"]=="ok"
assert runs[-1]["result"]["cases"] == 17 and runs[-1]["result"]["failed"] == 0
receipt={"schema":"fluxforge.recorded_validation_failures.followup.v1","date":"2026-10-03","base_commit":"3ed85ee0ca9df39f0dbfe50499c485bca45d9d55","repair_commit":source,"environment_prefix":"C:/Users/Josh/.codex/environments/fluxforge-issue21-complete","pythonpath":"worktree/src","original_suite":{"git_head":"36ed262f5fba11c6f9c92d1d182de977b11b565e","collected":2081,"passed":2052,"skipped":26,"failed":3,"unreported":0},"original_files_sha256":original_hashes,"runs":runs,"independent_parity_probe":{"git_head":probe["git_head"],"checks":len(probe["checks"]),"passed":True,"file":"parity_probe.json","sha256":sha(out/"parity_probe.json")},"resolved_original_failures":2,"unresolved_original_failures":1,"coordinate_workflow_passed":False,"single_revision_full_suite_passed":False,"integration":{"parity_fix":"PR #217 a88a2c4 already present; no duplicate comparator implementation","legacy_gui":"Archive old GUI and improve new one owns Tk archival. Relocate diagnostics with archived driver or retire active legacy test; a passing physical coordinate replay remains unestablished.","new_issue_implementation":"out of scope"}}
(repo/"docs/reviews/ISSUE_21_THREE_FAILURES_FOLLOWUP_2026-10-03.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
assert all(sha(Path(r["path"]))==r["sha256"] for r in original_hashes)
print(json.dumps({"runs":[{"name":r["name"],"result":r["result"]} for r in runs],"original_files_unchanged":True}))
