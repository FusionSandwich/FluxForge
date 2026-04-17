from __future__ import annotations

import json
import os
from pathlib import Path
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from gui_module3_workflows_probe import main as module3_probe_main  # noqa: E402
from fluxforge.gui import describe_gui_scaffold  # noqa: E402


def _checklist_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "checked": 0, "unchecked": 0}
    checked = 0
    unchecked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token.startswith("- [x]") or token.startswith("- [X]"):
            checked += 1
        elif token.startswith("- [ ]"):
            unchecked += 1
    return {
        "exists": True,
        "checked": checked,
        "unchecked": unchecked,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: gui_phase327_release_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = module3_probe_main([str(output_dir)])
    if result != 0:
        return result

    checklist_path = REPO_ROOT / "docs" / "PHASE3_27_RELEASE_CHECKLIST.md"
    payload = {
        "schema": "fluxforge.gui_phase327_probe.v1",
        "output_dir": str(output_dir),
        "gallery": str(output_dir / "index.html"),
        "qt_available": bool(describe_gui_scaffold().get("qt_available", False)),
        "checklist": _checklist_status(checklist_path),
    }
    report = output_dir / "phase327_release_report.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
