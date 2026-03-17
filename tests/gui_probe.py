"""Native Tk probe helpers for FluxForge GUI regression tests."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
import tkinter as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fluxforge_gui.app import FluxForgeGui


def _walk_widgets(widget: tk.Misc):
    yield widget
    for child in widget.winfo_children():
        yield from _walk_widgets(child)


def _find_button(root: tk.Misc, text: str):
    for widget in _walk_widgets(root):
        if (
            widget.winfo_class() in {"TButton", "Button"}
            and widget.cget("text") == text
        ):
            return widget
    raise LookupError(f"Button not found: {text}")


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _build_app() -> tuple[tk.Tk, FluxForgeGui]:
    root = tk.Tk()
    app = FluxForgeGui(root, project_dir=REPO_ROOT)
    root.update_idletasks()
    return root, app


def snapshot() -> dict[str, object]:
    root, app = _build_app()
    try:
        buttons = sorted(
            {
                widget.cget("text")
                for widget in _walk_widgets(root)
                if widget.winfo_class() in {"TButton", "Button"}
            }
        )
        return {
            "title": root.title(),
            "tabs": [
                app.notebook.tab(tab_id, "text") for tab_id in app.notebook.tabs()
            ],
            "buttons": buttons,
            "status": app.status_var.get(),
            "offline_mode": bool(app.offline_mode),
        }
    finally:
        root.destroy()


def capture_button_cli(button_text: str) -> dict[str, object]:
    root, app = _build_app()
    try:
        app.preview_input.set(
            str(REPO_ROOT / "examples" / "spectroscopy_data" / "sample.spe")
        )
        app.preview_png_output.set(
            str(REPO_ROOT / "artifacts" / "manual_review" / "gui_probe_plot.png")
        )
        app.preview_roi_file.set(
            str(REPO_ROOT / "artifacts" / "manual_review" / "gui_probe_rois.json")
        )
        app.preview_manual_peak_report.set(
            str(
                REPO_ROOT / "artifacts" / "manual_review" / "gui_probe_peak_report.json"
            )
        )
        app.response_cross_section_file.set(
            str(REPO_ROOT / "examples" / "unfolding_benchmark" / "cross_sections.json")
        )
        app.response_number_densities_file.set(
            str(
                REPO_ROOT / "examples" / "unfolding_benchmark" / "number_densities.json"
            )
        )
        app.response_boundaries_file.set(
            str(
                REPO_ROOT / "examples" / "unfolding_benchmark" / "group_boundaries.json"
            )
        )
        app.k0_import_input.set(str(REPO_ROOT / "artifacts" / "k0_import"))
        app.rafm_example_root.set(str(REPO_ROOT / "examples" / "RAFM_irradiation"))

        captured: dict[str, object] = {}

        def fake_dispatch(func, args, cli_tokens, on_success=None):
            command_line = "fluxforge " + " ".join(
                shlex.quote(token) for token in cli_tokens
            )
            app.last_cli_command = command_line
            app.copy_btn.configure(state="normal")
            captured["func_name"] = getattr(func, "__name__", str(func))
            captured["cli_tokens"] = list(cli_tokens)
            captured["command_line"] = command_line
            captured["args"] = _to_jsonable(vars(args))
            if on_success is not None:
                captured["on_success"] = getattr(
                    on_success, "__name__", str(on_success)
                )

        app._dispatch = fake_dispatch  # type: ignore[method-assign]
        button = _find_button(root, button_text)
        button.invoke()
        root.update_idletasks()
        if app.last_cli_command:
            app.copy_btn.invoke()
            root.update_idletasks()
            captured["clipboard"] = root.clipboard_get()
        captured["copy_state"] = str(app.copy_btn.cget("state"))
        return captured
    finally:
        root.destroy()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit("usage: gui_probe.py snapshot|button-cli [button text]")
    command = argv[1]
    if command == "snapshot":
        payload = snapshot()
    elif command == "button-cli":
        if len(argv) < 3:
            raise SystemExit("button-cli requires the button text")
        payload = capture_button_cli(argv[2])
    else:
        raise SystemExit(f"unknown probe command: {command}")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
