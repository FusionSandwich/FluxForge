"""Observe the unmodified driver's clicks without changing their dispatch."""
import importlib.util
import json
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("desktop_driver", root_path / "tests/gui_desktop_driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
events = []
original_click = driver._click_widget
original_fit = driver.FluxForgeGui._fit_preview_calibration_points

def fit(self):
    events.append({"fit_callback": True, "points": [vars(p) for p in self._calibration_points]})
    return original_fit(self)

def click(root, backend, widget):
    driver._scroll_widget_into_view(root, widget)
    x, y = driver._center_of(widget)
    hit = root.winfo_containing(x, y)
    record = {"text": widget.cget("text") if "text" in widget.keys() else "entry", "target": str(widget), "center": [x,y], "hit": str(hit), "screen": [root.winfo_screenwidth(),root.winfo_screenheight()], "canvas_yview": None}
    if record["text"] == "Fit calibration":
        driver._take_screenshot(root, backend, output, "before-fit.png")
    events.append(record)
    original_click(root, backend, widget)
    record["center_after"] = list(driver._center_of(widget))
    (output / "clicks.json").write_text(json.dumps(events,indent=2),encoding="utf-8")

driver.FluxForgeGui._fit_preview_calibration_points = fit
driver._click_widget = click
try:
    payload = driver.run_acceptance(output)
except Exception as exc:
    payload = {"status":"failed", "exception": repr(exc)}
    raise
finally:
    (output / "clicks.json").write_text(json.dumps(events,indent=2),encoding="utf-8")
    (output / "diagnostic.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
