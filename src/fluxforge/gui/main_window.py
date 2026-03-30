"""Main window shell for the next-generation FluxForge GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.backends import available_renderer_status
    from fluxforge.gui.panels import (
        BottomWorkspaceTabs,
        CentralWorkspaceTabs,
        SidebarPanel,
        ToolContextPanel,
    )
    from fluxforge.gui.qt_compat import (
        QAction,
        QApplication,
        QByteArray,
        QDockWidget,
        QKeySequence,
        QLabel,
        QMainWindow,
        QProgressBar,
        QSettings,
        QStatusBar,
        QToolBar,
        Qt,
    )
    from fluxforge.gui.theme_manager import load_stylesheet, resolve_theme
    from fluxforge.gui.widgets import HardwareLedWidget, ModeSwitcherWidget


@dataclass(frozen=True)
class DockZone:
    """Logical dock zone in the next-generation main window."""

    code: str
    title: str
    description: str


DEFAULT_DOCK_ZONES = (
    DockZone("A", "Menu / Toolbar", "Global actions, workflow mode, and shortcuts."),
    DockZone("B", "Left Sidebar", "Files, nuclides, results, and QA surfaces."),
    DockZone("C", "Central Canvas", "Primary spectrum and shared analytical workspace."),
    DockZone("D", "Bottom Panels", "Peak table, calibration, activity, batch, and log."),
    DockZone("E", "Right Sidebar", "Context-sensitive tool parameters."),
    DockZone("F", "Status Bar", "Cursor readout, hardware LED, and task progress."),
)


@dataclass
class MainWindowScaffold:
    """Lightweight representation of the planned main-window layout."""

    zones: tuple[DockZone, ...] = field(default_factory=lambda: DEFAULT_DOCK_ZONES)

    def zone_map(self) -> dict[str, DockZone]:
        """Return the dock zones keyed by their single-letter code."""

        return {zone.code: zone for zone in self.zones}


def modern_gui_unavailable_message() -> str:
    """Return the additive-launch guidance when Qt extras are absent."""

    reason = (
        f"{type(QT_IMPORT_ERROR).__name__}: {QT_IMPORT_ERROR}"
        if QT_IMPORT_ERROR
        else "Qt extras are unavailable."
    )
    return (
        "The next-generation FluxForge GUI requires the optional native GUI extras "
        "(`PySide6` and `pyqtgraph`). "
        f"Current import status: {reason} "
        "Use the legacy Tk shell via `fluxforge-gui-legacy` if you need the archived interface."
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class FluxForgeMainWindow(QMainWindow):
        """Modern dockable shell for the roadmap GUI."""

        ORGANIZATION = "FluxForge"
        APPLICATION = "FluxForgeNext"

        def __init__(
            self,
            mode_manager: ModeManager | None = None,
            selection_bus: SelectionBus | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("FluxForgeMainWindow")
            self.setWindowTitle("FluxForge Next")
            self.resize(1560, 980)

            self.settings = QSettings(self.ORGANIZATION, self.APPLICATION)
            self.mode_manager = mode_manager or ModeManager(settings=self.settings)
            self.selection_bus = selection_bus or SelectionBus.shared()
            self.setDockOptions(
                QMainWindow.AllowNestedDocks
                | QMainWindow.AllowTabbedDocks
                | QMainWindow.AnimatedDocks
            )

            self._build_menu_bar()
            self._build_toolbar()
            self._build_central_workspace()
            self._build_docks()
            self._build_status_bar()
            self._restore_layout()

            self.mode_manager.subscribe(self._on_mode_state_changed)
            self.selection_bus.subscribe(self._on_selection_changed)
            self._on_mode_state_changed(self.mode_manager.state)

        def _build_menu_bar(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self._action("Open Spectrum...", "Ctrl+O"))
            file_menu.addAction(self._action("Open Session...", "Ctrl+Shift+O"))
            file_menu.addSeparator()
            file_menu.addAction(self._action("Export Report...", "Ctrl+E"))
            file_menu.addAction(self._action("Export ANSI N42.42...", "Ctrl+Shift+E"))

            edit_menu = self.menuBar().addMenu("&Edit")
            edit_menu.addAction(self._action("Undo", "Ctrl+Z"))
            edit_menu.addAction(self._action("Redo", "Ctrl+Shift+Z"))

            view_menu = self.menuBar().addMenu("&View")
            view_menu.addAction(self._action("Toggle Full Canvas", "F11"))
            view_menu.addAction(self._action("Restore Default Layout"))
            renderer_menu = view_menu.addMenu("Renderer")
            for backend in available_renderer_status():
                label = backend["display_name"]
                if backend["recommended"]:
                    label += " ★"
                renderer_menu.addAction(self._action(label))

            analysis_menu = self.menuBar().addMenu("&Analysis")
            analysis_menu.addAction(self._action("Auto Find Peaks", "Ctrl+A"))
            analysis_menu.addAction(self._action("Nuclide Search"))

            calibration_menu = self.menuBar().addMenu("&Calibration")
            calibration_menu.addAction(self._action("Energy + FWHM Workspace"))
            calibration_menu.addAction(self._action("Quick Slider Mode"))

            tools_menu = self.menuBar().addMenu("&Tools")
            tools_menu.addAction(self._action("QA History"))
            tools_menu.addAction(self._action("Hardware Dashboard"))

            help_menu = self.menuBar().addMenu("&Help")
            help_menu.addAction(self._action("Shortcut Reference", "F1"))
            help_menu.addAction(self._action("About FluxForge Next"))

        def _build_toolbar(self) -> None:
            toolbar = QToolBar("Primary", self)
            toolbar.setObjectName("PrimaryToolbar")
            toolbar.setMovable(False)
            toolbar.addWidget(ModeSwitcherWidget(self.mode_manager, toolbar))
            self.addToolBar(Qt.TopToolBarArea, toolbar)

        def _build_central_workspace(self) -> None:
            self.central_tabs = CentralWorkspaceTabs(
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                parent=self,
            )
            self.setCentralWidget(self.central_tabs)

        def _wrap_dock(self, title: str, widget, area) -> QDockWidget:
            dock = QDockWidget(title, self)
            dock.setObjectName(f"{title.replace(' ', '')}Dock")
            dock.setWidget(widget)
            dock.setAllowedAreas(
                Qt.LeftDockWidgetArea
                | Qt.RightDockWidgetArea
                | Qt.BottomDockWidgetArea
            )
            self.addDockWidget(area, dock)
            return dock

        def _build_docks(self) -> None:
            self.left_dock = self._wrap_dock(
                "Workspace",
                SidebarPanel(selection_bus=self.selection_bus, parent=self),
                Qt.LeftDockWidgetArea,
            )
            self.bottom_dock = self._wrap_dock(
                "Analysis Surface",
                BottomWorkspaceTabs(
                    mode_manager=self.mode_manager,
                    selection_bus=self.selection_bus,
                    parent=self,
                ),
                Qt.BottomDockWidgetArea,
            )
            self.right_dock = self._wrap_dock(
                "Inspector",
                ToolContextPanel(
                    mode_manager=self.mode_manager,
                    selection_bus=self.selection_bus,
                    parent=self,
                ),
                Qt.RightDockWidgetArea,
            )

        def _build_status_bar(self) -> None:
            status = QStatusBar(self)
            status.setObjectName("FluxForgeStatusBar")
            self.setStatusBar(status)

            self.cursor_label = QLabel("Cursor: --", self)
            self.file_label = QLabel("File: none", self)
            self.mode_label = QLabel("Mode: Expert", self)
            self.renderer_label = QLabel("Renderer: PyQtGraph", self)
            self.progress = QProgressBar(self)
            self.progress.setObjectName("StatusProgress")
            self.progress.setMaximumWidth(180)
            self.progress.setRange(0, 100)
            self.progress.setValue(12)
            self.progress.setFormat("Shell 12%")
            self.hardware_led = HardwareLedWidget(self)

            status.addWidget(self.cursor_label, 1)
            status.addWidget(self.file_label, 1)
            status.addPermanentWidget(self.mode_label)
            status.addPermanentWidget(self.renderer_label)
            status.addPermanentWidget(self.progress)
            status.addPermanentWidget(self.hardware_led)
            self.hardware_led.set_status("offline", "NO DEVICE")

        def _action(self, text: str, shortcut: str | None = None) -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.setEnabled(False)
            return action

        def _restore_layout(self) -> None:
            geometry = self.settings.value("main_window/geometry")
            if isinstance(geometry, QByteArray):
                self.restoreGeometry(geometry)
            state = self.settings.value("main_window/state")
            if isinstance(state, QByteArray):
                self.restoreState(state)

        def _save_layout(self) -> None:
            self.settings.setValue("main_window/geometry", self.saveGeometry())
            self.settings.setValue("main_window/state", self.saveState())
            self.settings.sync()

        def _on_mode_state_changed(self, state) -> None:
            resolved_theme = resolve_theme(state.theme)
            QApplication.instance().setStyleSheet(load_stylesheet(resolved_theme))
            label = f"Mode: {state.mode.value.title()}"
            if state.standard:
                label += f" · {state.standard}"
            self.mode_label.setText(label)
            self.progress.setFormat(f"{state.mode.value.title()} shell ready")
            self.renderer_label.setText("Renderer: PyQtGraph-first")

        def _on_selection_changed(self, state: SelectionState) -> None:
            cursor_parts = []
            if state.peak_energy_keV is not None:
                cursor_parts.append(f"Peak {state.peak_energy_keV:.3f} keV")
            if state.nuclide:
                cursor_parts.append(state.nuclide)
            if state.roi_bounds_keV:
                cursor_parts.append(
                    f"ROI {state.roi_bounds_keV[0]:.1f}-{state.roi_bounds_keV[1]:.1f} keV"
                )
            self.cursor_label.setText(
                "Cursor: " + (" | ".join(cursor_parts) if cursor_parts else "--")
            )

        def closeEvent(self, event) -> None:
            self._save_layout()
            super().closeEvent(event)


else:

    class FluxForgeMainWindow:  # pragma: no cover - placeholder without Qt
        """Import-safe placeholder when the optional Qt dependencies are missing."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(modern_gui_unavailable_message())
