"""Capture a fitted HPGe efficiency dialog with the selected Qt platform."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gui_hpge_efficiency_probe.py <output-dir>", file=sys.stderr)
        return 2
    output_dir = Path(sys.argv[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    dialog = EfficiencyCalibrationDialog(mode_manager=ModeManager())
    dialog.show()
    app.processEvents()
    dialog._seed_demo_points()
    dialog._fit_model()
    fit = dialog.accepted_fit()
    if fit is None or not dialog.accept_button.isEnabled():
        raise RuntimeError("HPGe dialog did not produce an applicable fit")
    app.processEvents()
    screenshot = output_dir / "hpge-efficiency-fit.png"
    if not dialog.grab().save(str(screenshot)):
        raise RuntimeError("Could not capture HPGe dialog")
    evidence = {
        "qt_platform": app.platformName(),
        "model": fit.model_key,
        "points": fit.points_used,
        "diagnostic_rows": dialog.point_diagnostics.rowCount(),
        "review_status": fit.fit_quality["review_status"],
        "covariance_parameters": list(fit.covariance_parameters),
        "screenshot": screenshot.name,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence))
    dialog.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
