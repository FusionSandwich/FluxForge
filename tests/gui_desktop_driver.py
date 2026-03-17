"""Real desktop interaction driver for FluxForge GUI acceptance runs."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
import tkinter as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fluxforge_gui.app import FluxForgeGui
from fluxforge.io.artifacts import (
    write_report_bundle,
    write_unfold_result,
    write_validation_bundle,
)


def _walk_widgets(widget: tk.Misc):
    yield widget
    for child in widget.winfo_children():
        yield from _walk_widgets(child)


def _find_widget_by_text(root: tk.Misc, text: str, classes: set[str]):
    for widget in _walk_widgets(root):
        if widget.winfo_class() not in classes:
            continue
        with contextlib.suppress(tk.TclError):
            if not widget.winfo_ismapped():
                continue
        with contextlib.suppress(tk.TclError):
            if widget.cget("text") == text:
                return widget
    raise LookupError(f"Widget not found: {text}")


def _find_labelframe(root: tk.Misc, text: str):
    return _find_widget_by_text(root, text, {"TLabelframe", "Labelframe"})


def _grid_widget(frame: tk.Misc, *, row: int, column: int, classes: set[str]):
    for widget in frame.grid_slaves(row=row, column=column):
        if widget.winfo_class() in classes:
            return widget
    raise LookupError(
        f"Grid widget not found in {frame} at row={row}, column={column}, classes={classes}"
    )


def _center_of(widget: tk.Misc) -> tuple[int, int]:
    widget.update_idletasks()
    return (
        int(widget.winfo_rootx() + (widget.winfo_width() / 2)),
        int(widget.winfo_rooty() + (widget.winfo_height() / 2)),
    )


def _tree_item_center(
    tree: tk.Misc, item_id: str, column: str = "#1"
) -> tuple[int, int]:
    tree.update_idletasks()
    bbox = tree.bbox(item_id, column)
    if not bbox:
        raise LookupError(f"Tree item {item_id} is not visible.")
    x, y, width, height = bbox
    return (
        int(tree.winfo_rootx() + x + width / 2),
        int(tree.winfo_rooty() + y + height / 2),
    )


def _notebook_tab_center(notebook: tk.Misc, label: str) -> tuple[int, int] | None:
    notebook.update_idletasks()
    for tab_id in notebook.tabs():
        if notebook.tab(tab_id, "text") != label:
            continue
        x, y, width, height = notebook.bbox(tab_id)
        if width <= 0 or height <= 0:
            return None
        return (
            int(notebook.winfo_rootx() + x + width / 2),
            int(notebook.winfo_rooty() + y + height / 2),
        )
    raise LookupError(f"Notebook tab not found: {label}")


class _LinuxBackend:
    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.05
        self._pyautogui = pyautogui

    def click(self, x: int, y: int) -> None:
        self._pyautogui.click(x=x, y=y)

    def write(self, text: str) -> None:
        self._pyautogui.write(text, interval=0.02)

    def clear_entry(self) -> None:
        self._pyautogui.hotkey("ctrl", "a")
        self._pyautogui.press("backspace")

    def screenshot(self, output_path: Path) -> None:
        subprocess.run(
            ["import", "-window", "root", str(output_path)],
            check=True,
            cwd=REPO_ROOT,
        )


class _WindowsBackend:
    def __init__(self) -> None:
        from PIL import ImageGrab
        from pywinauto import keyboard, mouse

        self._ImageGrab = ImageGrab
        self._keyboard = keyboard
        self._mouse = mouse

    def click(self, x: int, y: int) -> None:
        self._mouse.click(coords=(x, y))

    def write(self, text: str) -> None:
        self._keyboard.send_keys(text, with_spaces=True, pause=0.02)

    def clear_entry(self) -> None:
        self._keyboard.send_keys("^a{BACKSPACE}", pause=0.02)

    def screenshot(self, output_path: Path) -> None:
        self._ImageGrab.grab().save(output_path)


def _build_backend():
    if sys.platform == "win32":
        return _WindowsBackend()
    return _LinuxBackend()


def _prefer_ci_safe_tk_events() -> bool:
    return (
        sys.platform == "win32"
        and os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    )


def _pump(root: tk.Tk, duration: float = 0.2) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)


def _wait_for(
    root: tk.Tk,
    predicate,
    *,
    timeout: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {description}.")


def _click_widget(root: tk.Tk, backend, widget: tk.Misc) -> None:
    root.update_idletasks()
    local_x = max(int(widget.winfo_width() / 2), 1)
    local_y = max(int(widget.winfo_height() / 2), 1)
    if sys.platform == "win32" and not _prefer_ci_safe_tk_events():
        x, y = _center_of(widget)
        backend.click(x, y)
    else:
        widget.focus_force()
        with contextlib.suppress(tk.TclError, AttributeError):
            widget.invoke()
            _pump(root, 0.15)
            return
        widget.event_generate("<Enter>")
        widget.event_generate("<Motion>", x=local_x, y=local_y)
        widget.event_generate("<ButtonPress-1>", x=local_x, y=local_y)
        widget.event_generate("<ButtonRelease-1>", x=local_x, y=local_y)
    _pump(root, 0.15)


def _set_entry_text(root: tk.Tk, backend, entry: tk.Misc, value: str) -> None:
    _click_widget(root, backend, entry)
    entry.focus_force()
    backend.clear_entry()
    _pump(root, 0.05)
    if value:
        backend.write(value)
        _pump(root, 0.1)
    if entry.get() != value:
        entry.delete(0, "end")
        if value:
            entry.insert(0, value)
        _pump(root, 0.05)


def _click_notebook_tab(root: tk.Tk, backend, notebook: tk.Misc, label: str) -> None:
    center = _notebook_tab_center(notebook, label)
    if center is not None:
        x, y = center
        backend.click(x, y)
        _pump(root, 0.2)
    if notebook.tab(notebook.select(), "text") == label:
        return
    for tab_id in notebook.tabs():
        if notebook.tab(tab_id, "text") == label:
            notebook.select(tab_id)
            _pump(root, 0.2)
            return
    raise LookupError(f"Notebook tab not found: {label}")


def _click_tree_item(root: tk.Tk, backend, tree: tk.Misc, item_id: str) -> None:
    x, y = _tree_item_center(tree, item_id)
    if sys.platform == "win32" and not _prefer_ci_safe_tk_events():
        backend.click(x, y)
    else:
        rel_x = max(int(x - tree.winfo_rootx()), 1)
        rel_y = max(int(y - tree.winfo_rooty()), 1)
        tree.selection_set(item_id)
        tree.focus(item_id)
        tree.see(item_id)
        tree.event_generate("<Motion>", x=rel_x, y=rel_y)
        tree.event_generate("<ButtonPress-1>", x=rel_x, y=rel_y)
        tree.event_generate("<ButtonRelease-1>", x=rel_x, y=rel_y)
    _pump(root, 0.2)


def _take_screenshot(root: tk.Tk, backend, output_dir: Path, name: str) -> Path:
    root.update()
    output_path = output_dir / name
    backend.screenshot(output_path)
    return output_path


def run_acceptance(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _build_backend()
    root = tk.Tk()
    root.geometry("1400x900+0+0")
    app = FluxForgeGui(root, project_dir=output_dir)
    root.deiconify()
    root.lift()
    with contextlib.suppress(tk.TclError):
        root.attributes("-topmost", True)
    root.focus_force()

    evidence: dict[str, object] = {
        "platform": sys.platform,
        "offline_mode": bool(app.offline_mode),
        "screenshots": [],
        "artifacts": [],
        "cli_commands": {},
        "steps": [],
    }

    def log_step(step: str, **details) -> None:
        evidence["steps"].append({"step": step, **details})

    def sync_dispatch(func, args, cli_tokens, on_success=None) -> None:
        command_line = "fluxforge " + " ".join(
            shlex.quote(token) for token in cli_tokens
        )
        app.last_cli_command = command_line
        app.copy_btn.configure(state="normal")
        app._append_log(f"\n$ {command_line}")
        app._set_busy(True)
        try:
            output = app._execute_command(func, args)
            if output:
                app._append_log(output)
            else:
                app._append_log("Command completed.")
            if on_success is not None:
                on_success()
        except Exception as exc:  # pragma: no cover - surfaced in test stderr
            app._append_log(f"ERROR: {exc}")
            raise
        finally:
            app._set_busy(False)

    app._dispatch = sync_dispatch  # type: ignore[method-assign]

    try:
        raw_spectrum_input = (
            REPO_ROOT
            / "examples"
            / "RAFM_irradiation"
            / "raw_gamma_spec"
            / "flux_wires"
            / "Ti-RAFM-1a_25cm.ASC"
        )
        background_input = (
            REPO_ROOT / "examples" / "RAFM_irradiation" / "background.ASC"
        )
        spectrum_input = output_dir / "ti_rafm_1a_25cm_ingested.json"
        gui_preview_output = output_dir / "gui_preview_native.png"
        cli_plot_output = output_dir / "cli_spectrum_plot.png"
        roi_output = output_dir / "manual_rois.json"
        peak_report_output = output_dir / "manual_peak_report.json"
        response_output = output_dir / "response.json"
        unfold_output = output_dir / "unfold.json"
        validation_output = output_dir / "validation.json"
        k0_report_output = output_dir / "k0_report.json"
        k0_text_output = output_dir / "k0_report.txt"
        plots_output_dir = output_dir / "plot_suite"
        response_output.write_text(
            json.dumps(
                {
                    "schema": "fluxforge.response_bundle.v1",
                    "matrix": [[1.0, 0.1], [0.2, 0.9]],
                    "reactions": ["Au-197(n,g)", "Ti-46(n,p)"],
                    "boundaries_eV": [1e-5, 1e-3, 1.0],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_unfold_result(
            unfold_output,
            boundaries_eV=[1e-5, 1e-3, 1.0],
            reactions=["Au-197(n,g)", "Ti-46(n,p)"],
            flux=[3.0, 1.0],
            covariance=[[0.05, 0.0], [0.0, 0.02]],
            chi2=0.35,
            method="gls",
            diagnostics={"iterations": 12, "converged": True},
        )
        write_validation_bundle(
            validation_output,
            metrics={"mae": 0.02, "rmse": 0.03, "chi2": 0.8},
            truth_flux=[1.0, 2.0, 3.0],
            predicted_flux=[1.1, 1.9, 2.8],
            residuals=[0.1, -0.1, -0.2],
        )
        k0_text_output.write_text("desktop k0 review report\n", encoding="utf-8")
        write_report_bundle(
            k0_report_output,
            summary={"element_count": 1, "mode": "desktop-review"},
            text_report={"path": k0_text_output.name, "format": "text/plain"},
        )

        app.ingest_input.set(str(raw_spectrum_input))
        app.ingest_output.set(str(spectrum_input))
        app.ingest_background_file.set(str(background_input))
        app.preview_input.set(str(spectrum_input))
        app.preview_peaks.set("")
        app.preview_png_output.set(str(gui_preview_output))
        app.preview_roi_file.set(str(roi_output))
        app.preview_manual_peak_report.set(str(peak_report_output))
        app.preview_plot_title.set("FluxForge RAFM Native Desktop Acceptance")
        app.response_output.set(str(response_output))
        app.unfold_response.set(str(response_output))
        app.unfold_output.set(str(unfold_output))
        app.compare_output.set(str(validation_output))
        app.k0_report_output.set(str(k0_report_output))
        app.plots_output_dir.set(str(plots_output_dir))

        _pump(root, 0.4)
        launch_shot = _take_screenshot(root, backend, output_dir, "01-launch.png")
        evidence["screenshots"].append(str(launch_shot))
        log_step("launch", title=root.title())

        _click_notebook_tab(root, backend, app.notebook, "1. Ingest")
        run_ingest_btn = _find_widget_by_text(root, "Run Ingest", {"TButton", "Button"})
        _click_widget(root, backend, run_ingest_btn)
        _wait_for(
            root,
            lambda: spectrum_input.exists(),
            timeout=20.0,
            description="raw RAFM ingest",
        )
        evidence["artifacts"].append(str(spectrum_input))
        log_step(
            "ingest_raw",
            input=str(raw_spectrum_input),
            background=str(background_input),
            output=str(spectrum_input),
        )

        _click_notebook_tab(root, backend, app.notebook, "2. Spectrum")
        load_preview_btn = _find_widget_by_text(
            root, "Load Preview", {"TButton", "Button"}
        )
        _click_widget(root, backend, load_preview_btn)
        _wait_for(
            root,
            lambda: app._preview_state is not None
            and "Loaded" in app.preview_status.get(),
            timeout=15.0,
            description="spectrum preview load",
        )
        auto_detect_btn = _find_widget_by_text(
            root, "Auto-detect peaks", {"TButton", "Button"}
        )
        _click_widget(root, backend, auto_detect_btn)
        _wait_for(
            root,
            lambda: app._preview_state is not None
            and len(app._preview_state.peaks) > 0,
            timeout=15.0,
            description="preview peak detection",
        )
        peak_items = app.preview_peak_table.get_children()
        if not peak_items:
            raise LookupError("Peak table is empty after auto-detection.")
        _click_tree_item(root, backend, app.preview_peak_table, peak_items[0])
        _wait_for(
            root,
            lambda: app._selected_peak() is not None,
            timeout=5.0,
            description="peak row selection",
        )
        count_peak_btn = _find_widget_by_text(
            root, "Count selected peak", {"TButton", "Button"}
        )
        _click_widget(root, backend, count_peak_btn)
        _wait_for(
            root,
            lambda: "ROI channels" in app.preview_count_summary.get(),
            timeout=10.0,
            description="selected peak count",
        )
        loaded_shot = _take_screenshot(
            root, backend, output_dir, "02-spectrum-loaded.png"
        )
        evidence["screenshots"].append(str(loaded_shot))
        log_step(
            "load_preview",
            preview_status=app.preview_status.get(),
            spectrum_label=(
                app._preview_state.primary.label if app._preview_state else ""
            ),
        )
        selected_peak = app._selected_peak()
        log_step(
            "peak_selection",
            peak_count=len(app._preview_state.peaks) if app._preview_state else 0,
            selected_peak_energy_keV=(
                selected_peak.energy_keV if selected_peak is not None else None
            ),
            count_summary=app.preview_count_summary.get(),
        )

        app.preview_controls_canvas.yview_moveto(0.32)
        _pump(root, 0.2)
        roi_frame = _find_labelframe(root, "Manual ROI draw / edit")
        roi_label_entry = _grid_widget(
            roi_frame, row=1, column=1, classes={"TEntry", "Entry"}
        )
        roi_left_entry = _grid_widget(
            roi_frame, row=2, column=1, classes={"TEntry", "Entry"}
        )
        roi_right_entry = _grid_widget(
            roi_frame, row=3, column=1, classes={"TEntry", "Entry"}
        )
        add_roi_btn = _find_widget_by_text(
            root, "Add / Update ROI", {"TButton", "Button"}
        )

        _set_entry_text(root, backend, roi_label_entry, "Native ROI")
        _set_entry_text(root, backend, roi_left_entry, "350")
        _set_entry_text(root, backend, roi_right_entry, "520")
        _click_widget(root, backend, add_roi_btn)
        _wait_for(
            root,
            lambda: len(app._manual_regions) == 1,
            timeout=5.0,
            description="ROI create",
        )

        _click_tree_item(root, backend, app.preview_roi_table, "0")
        _wait_for(
            root,
            lambda: bool(app.preview_roi_table.selection()),
            timeout=3.0,
            description="ROI row selection",
        )
        _set_entry_text(root, backend, roi_left_entry, "360")
        _set_entry_text(root, backend, roi_right_entry, "540")
        _click_widget(root, backend, add_roi_btn)
        _wait_for(
            root,
            lambda: len(app._manual_regions) == 1
            and abs(app._manual_regions[0].left_keV - 360.0) < 1e-9
            and abs(app._manual_regions[0].right_keV - 540.0) < 1e-9,
            timeout=5.0,
            description="ROI edit",
        )
        log_step(
            "roi_edit",
            roi_count=len(app._manual_regions),
            roi_bounds=[
                app._manual_regions[0].left_keV,
                app._manual_regions[0].right_keV,
            ],
        )

        app.preview_controls_canvas.yview_moveto(0.48)
        _pump(root, 0.2)
        calibration_frame = _find_labelframe(root, "Calibration editor")
        picked_channel_entry = _grid_widget(
            calibration_frame, row=6, column=1, classes={"TEntry", "Entry"}
        )
        observed_entry = _grid_widget(
            calibration_frame, row=7, column=1, classes={"TEntry", "Entry"}
        )
        reference_entry = _grid_widget(
            calibration_frame, row=8, column=1, classes={"TEntry", "Entry"}
        )
        point_label_entry = _grid_widget(
            calibration_frame, row=9, column=1, classes={"TEntry", "Entry"}
        )
        add_point_btn = _find_widget_by_text(root, "Add point", {"TButton", "Button"})
        pick_selected_peak_btn = _find_widget_by_text(
            root, "Pick selected peak", {"TButton", "Button"}
        )
        fit_calibration_btn = _find_widget_by_text(
            root, "Fit calibration", {"TButton", "Button"}
        )
        apply_calibration_btn = _find_widget_by_text(
            root, "Apply calibration", {"TButton", "Button"}
        )

        _set_entry_text(root, backend, picked_channel_entry, "0")
        _set_entry_text(root, backend, observed_entry, "0")
        _set_entry_text(root, backend, reference_entry, "0")
        _set_entry_text(root, backend, point_label_entry, "origin")
        _click_widget(root, backend, add_point_btn)

        _click_widget(root, backend, pick_selected_peak_btn)
        _wait_for(
            root,
            lambda: bool(picked_channel_entry.get()) and bool(observed_entry.get()),
            timeout=5.0,
            description="selected peak calibration pick",
        )
        _set_entry_text(root, backend, reference_entry, observed_entry.get())
        _set_entry_text(root, backend, point_label_entry, "selected_peak")
        _click_widget(root, backend, add_point_btn)

        _wait_for(
            root,
            lambda: len(app._calibration_points) == 2,
            timeout=5.0,
            description="calibration points",
        )
        _click_widget(root, backend, fit_calibration_btn)
        _wait_for(
            root,
            lambda: app._calibration_fit is not None
            and "Calibration fit complete" in app.preview_calibration_summary.get(),
            timeout=10.0,
            description="calibration fit",
        )
        _click_widget(root, backend, apply_calibration_btn)
        _wait_for(
            root,
            lambda: "Applied calibration editor coefficients"
            in app.preview_status.get(),
            timeout=5.0,
            description="calibration apply",
        )
        calibration_shot = _take_screenshot(
            root, backend, output_dir, "03-roi-calibration.png"
        )
        evidence["screenshots"].append(str(calibration_shot))
        log_step(
            "calibration",
            point_count=len(app._calibration_points),
            coefficients=[
                app.preview_calibration_c0.get(),
                app.preview_calibration_c1.get(),
                app.preview_calibration_c2.get(),
                app.preview_calibration_c3.get(),
            ],
        )

        app.preview_controls_canvas.yview_moveto(0.0)
        _pump(root, 0.2)
        save_png_btn = _find_widget_by_text(root, "Save PNG", {"TButton", "Button"})
        _click_widget(root, backend, save_png_btn)
        _wait_for(
            root,
            lambda: gui_preview_output.exists(),
            timeout=10.0,
            description="GUI preview PNG export",
        )
        evidence["artifacts"].append(str(gui_preview_output))
        log_step("save_png", output=str(gui_preview_output))

        app.preview_png_output.set(str(cli_plot_output))
        _pump(root, 0.1)
        run_cli_plot_btn = _find_widget_by_text(
            root, "Run CLI Plot Export", {"TButton", "Button"}
        )
        _click_widget(root, backend, run_cli_plot_btn)
        _wait_for(
            root,
            lambda: (not app._busy) and cli_plot_output.exists(),
            timeout=20.0,
            description="CLI spectrum plot export",
        )
        _click_widget(root, backend, app.copy_btn)
        _wait_for(
            root,
            lambda: bool(root.clipboard_get()),
            timeout=5.0,
            description="clipboard capture for spectrum plot",
        )
        evidence["artifacts"].extend(
            [
                str(cli_plot_output),
                str(roi_output),
                str(peak_report_output),
                str(response_output),
                str(unfold_output),
                str(validation_output),
                str(k0_report_output),
            ]
        )
        evidence["cli_commands"]["spectrum_plot"] = root.clipboard_get()
        log_step(
            "run_spectrum_plot",
            command=root.clipboard_get(),
            last_cli=app.last_cli_command,
        )

        _click_notebook_tab(root, backend, app.notebook, "8. Standards")
        app.standards_preset.set("k0-NAA")
        _pump(root, 0.1)
        apply_preset_btn = _find_widget_by_text(
            root, "Apply Preset to GUI", {"TButton", "Button"}
        )
        _click_widget(root, backend, apply_preset_btn)
        _wait_for(
            root,
            lambda: app.standards_data_source.get() == "k0_naa_monitors",
            timeout=5.0,
            description="k0 standards preset",
        )
        standards_shot = _take_screenshot(
            root, backend, output_dir, "04-standards-preset.png"
        )
        evidence["screenshots"].append(str(standards_shot))
        log_step(
            "standards_preset",
            preset=app.standards_preset.get(),
            data_source=app.standards_data_source.get(),
            status=app.standards_source_summary.get(),
        )

        _click_notebook_tab(root, backend, app.notebook, "6. Unfold")
        load_response_btn = _find_widget_by_text(
            root, "Load Response Summary", {"TButton", "Button"}
        )
        load_unfold_btn = _find_widget_by_text(
            root, "Load Result", {"TButton", "Button"}
        )
        _click_widget(root, backend, load_response_btn)
        _wait_for(
            root,
            lambda: "Loaded response bundle" in app.response_status.get(),
            timeout=5.0,
            description="response summary load",
        )
        _click_widget(root, backend, load_unfold_btn)
        _wait_for(
            root,
            lambda: "Method: GLS" in app.unfold_summary.get(),
            timeout=5.0,
            description="unfold preview load",
        )
        log_step(
            "unfold_preview",
            response_status=app.response_status.get(),
            unfold_summary=app.unfold_summary.get(),
        )

        _click_notebook_tab(root, backend, app.notebook, "7. Compare")
        load_compare_btn = _find_widget_by_text(
            root, "Load Summary", {"TButton", "Button"}
        )
        _click_widget(root, backend, load_compare_btn)
        _wait_for(
            root,
            lambda: "MAE" in app.compare_summary.get(),
            timeout=5.0,
            description="compare summary load",
        )
        compare_shot = _take_screenshot(
            root, backend, output_dir, "05-unfold-compare.png"
        )
        evidence["screenshots"].append(str(compare_shot))
        log_step("compare_summary", summary=app.compare_summary.get())

        _click_notebook_tab(root, backend, app.notebook, "10. Report")
        example_checkbox = _find_widget_by_text(
            root, "Use bundled example inputs", {"TCheckbutton", "Checkbutton"}
        )
        if not app.plots_example.get():
            _click_widget(root, backend, example_checkbox)
            _wait_for(
                root,
                lambda: bool(app.plots_example.get()),
                timeout=3.0,
                description="bundled example toggle",
            )
        run_plot_suite_btn = _find_widget_by_text(
            root, "Run Plot Suite", {"TButton", "Button"}
        )
        _click_widget(root, backend, run_plot_suite_btn)
        _wait_for(
            root,
            lambda: (not app._busy)
            and plots_output_dir.exists()
            and any(plots_output_dir.iterdir()),
            timeout=30.0,
            description="master plot suite run",
        )
        _click_widget(root, backend, app.copy_btn)
        _wait_for(
            root,
            lambda: "plots" in root.clipboard_get(),
            timeout=5.0,
            description="clipboard capture for plot suite",
        )
        report_shot = _take_screenshot(root, backend, output_dir, "06-report-plots.png")
        evidence["screenshots"].append(str(report_shot))
        evidence["artifacts"].append(str(plots_output_dir))
        evidence["cli_commands"]["plot_suite"] = root.clipboard_get()
        log_step(
            "run_plot_suite",
            command=root.clipboard_get(),
            plot_count=len(list(plots_output_dir.iterdir())),
        )

        evidence["status"] = "ok"
        return evidence
    finally:
        with contextlib.suppress(Exception):
            root.destroy()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: gui_desktop_driver.py <output-dir>")
    output_dir = Path(argv[1]).resolve()
    payload = run_acceptance(output_dir)
    (output_dir / "run.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
