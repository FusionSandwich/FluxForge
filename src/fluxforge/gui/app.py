"""FluxForge desktop GUI launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fluxforge.gui.backends import available_renderer_status
from fluxforge.gui.main_window import (
    FluxForgeMainWindow,
    MainWindowScaffold,
    modern_gui_unavailable_message,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import QApplication


def describe_gui_scaffold() -> dict[str, object]:
    """Return a diagnostic summary of the desktop GUI."""

    scaffold = MainWindowScaffold()
    return {
        "dock_zones": [zone.code for zone in scaffold.zones],
        "zone_titles": [zone.title for zone in scaffold.zones],
        "themes": ["dark", "light", "system"],
        "status": "qt-shell-ready" if QT_AVAILABLE else "qt-shell-optional",
        "qt_available": QT_AVAILABLE,
        "renderer_backends": list(available_renderer_status()),
        "modern_entrypoint": "fluxforge-gui",
        "legacy_entrypoint": "fluxforge-gui-legacy",
    }


def launch_modern_gui(
    project_dir: str | Path | None = None,
    *,
    developer_tools: bool = False,
    open_example: bool = False,
) -> int:
    """Launch the production desktop GUI if its native dependencies are available."""

    if not QT_AVAILABLE:
        raise RuntimeError(modern_gui_unavailable_message())

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("FluxForge")
    app.setApplicationName("FluxForgeNext")

    window = FluxForgeMainWindow(
        selection_bus=SelectionBus.shared(),
        developer_tools=developer_tools,
        load_example=open_example,
    )
    if project_dir is not None:
        window.file_label.setText(f"Project: {Path(project_dir)}")
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint for the FluxForge desktop GUI."""

    parser = argparse.ArgumentParser(prog="fluxforge-gui")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Optional project directory to show in the status bar.",
    )
    parser.add_argument(
        "--developer-tools",
        action="store_true",
        help="Show parity, diagnostics, and other developer-only workspaces.",
    )
    parser.add_argument(
        "--open-example",
        action="store_true",
        help="Open the bundled HPGe example instead of an empty workspace.",
    )
    args = parser.parse_args(argv)
    if not QT_AVAILABLE:
        print(modern_gui_unavailable_message(), file=sys.stderr)
        return 1
    return launch_modern_gui(
        project_dir=args.project_dir,
        developer_tools=args.developer_tools,
        open_example=args.open_example,
    )
