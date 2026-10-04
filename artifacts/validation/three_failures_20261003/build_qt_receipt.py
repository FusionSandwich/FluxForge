"""Create the portable current-GUI receipt from the preserved group records."""
import hashlib
import json
from pathlib import Path

out=Path(__file__).resolve().parent
repo=out.parents[2]
offscreen=json.loads((out/"qt_checks_summary.json").read_text())
native=json.loads((out/"native_qt_checks_summary.json").read_text())
snapshot=json.loads((out/"qt_snapshot.json").read_text())
precedence=json.loads((out/"published_parity_precedence.json").read_text())
passed=[g for g in offscreen["groups"] if g["exit_code"] == 0]+native["groups"]
assert len(passed)==7 and sum(g["cases"] for g in passed)==82
assert all(g["git_head"] == snapshot["git_head"] for g in passed)
assert all(g["failed"]==0 and g["skipped"]==0 for g in passed)
assert snapshot["status"]=="ok" and not snapshot["legacy_gui_imported"]
names=["qt_checks_summary.json","native_qt_checks_summary.json","qt_snapshot.json","qt_example_workspace.png","qt_empty_workspace.png","qt_offscreen_missing_fonts.png","published_parity_precedence.json"]
names += [f"{g['group']}.{suffix}" for g in passed for suffix in ("xml","log")]
receipt={"schema":"fluxforge.qt_integration.validation.v1","date":"2026-10-03","gui":"PySide6/Qt","source_commit":snapshot["git_head"],"consolidation_base_commit":"b618f5e3bf41323d613b8193b43a8ec26f94ec90","python":snapshot["python"],"prefix":snapshot["prefix"],"qt_version":snapshot["qt_version"],"groups":passed,"selected_counts":{"passed":82,"failed":0,"skipped":0},"profiles":{"offscreen":offscreen["qt_qpa_platform"],"native":native["qt_qpa_platform"]},"incomplete_attempts":[g for g in offscreen["groups"] if g["exit_code"]!=0],"native_gui_snapshot":{"status":snapshot["status"],"window_size":snapshot["window_size"],"default_font":snapshot["default_font"],"legacy_gui_imported":False,"source_sha256":snapshot["source_sha256"]},"published_acceptance_precedence":{"git_head":precedence[1]["git_head"],"correct_checks":precedence[1]["correct"],"total_checks":8,"integration_requirement":"Retain #217 explicit-field precedence; latest acceptance comparator is insertion-order dependent."},"legacy_coordinate_workflow":{"historical_result":"reproducible failure","current_gui_status":"Tk test archived outside active discovery; not a claimed passing replay"},"single_revision_full_suite_passed":False,"artifact_sha256":{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in names}}
(repo/"docs/reviews/ISSUE_21_QT_INTEGRATION_2026-10-03.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps({"source_commit":snapshot["git_head"],"passed":82,"latest_acceptance":precedence[1]["git_head"],"precedence_correct":precedence[1]["correct"]}))
