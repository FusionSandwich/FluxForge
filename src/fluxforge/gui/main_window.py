"""Main window shell for the next-generation FluxForge GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fluxforge.gui.file_workflow import RecentFilesManager, normalize_dropped_paths
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.analysis_workspace import (
    LoadedSpectrumRecord,
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    SpectrumSlot,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.io import read_ffs_session, read_spectrum_any
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.reporting.engine import ReportingEngine
from fluxforge.standards import QAMonitor, StandardsEvaluationContext, register_builtin_standards_modules
from fluxforge.plugins import bootstrap_builtin_registries

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.backends import available_renderer_status
    from fluxforge.gui.dialogs import (
        CalibrationWorkspaceDialog,
        PuIsotopicsDialog,
        QAHistoryDialog,
        ReportExportDialog,
        StandardsReviewDialog,
        UnfoldingWorkspaceDialog,
    )
    from fluxforge.gui.panels import (
        BottomWorkspaceTabs,
        CentralWorkspaceTabs,
        SidebarPanel,
        ToolContextPanel,
    )
    from fluxforge.gui.panels.modern_shell import (
        build_demo_background_spectrum,
        build_demo_overlay_spectrum,
        build_demo_spectrum,
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
        QUndoStack,
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
            self.library_manager = DataLibraryManager(settings=self.settings)
            self.recent_files = RecentFilesManager(self.settings)
            self.qa_monitor = QAMonitor()
            self.qa_monitor.seed_demo_history()
            self.reporting_engine = ReportingEngine()
            self.registries = bootstrap_builtin_registries()
            register_builtin_standards_modules(self.registries)
            self.undo_stack = QUndoStack(self)
            self.analysis_workspace = AnalysisWorkspaceController(
                self._build_initial_workspace_state()
            )
            self._calibration_dialog = None
            self._unfolding_dialog = None
            self._qa_history_dialog = None
            self._report_dialog = None
            self._pu_isotopics_dialog = None
            self._standards_review_dialog = None
            self.setDockOptions(
                QMainWindow.AllowNestedDocks
                | QMainWindow.AllowTabbedDocks
                | QMainWindow.AnimatedDocks
            )
            self.setAcceptDrops(True)

            self._build_menu_bar()
            self._build_toolbar()
            self._build_central_workspace()
            self._build_docks()
            self._build_status_bar()
            self._restore_layout()
            self._refresh_analysis_workspace_derivatives()

            self.mode_manager.subscribe(self._on_mode_state_changed)
            self.selection_bus.subscribe(self._on_selection_changed)
            self.analysis_workspace.subscribe(self._on_workspace_state_changed)
            self._on_mode_state_changed(self.mode_manager.state)
            self._on_workspace_state_changed(self.analysis_workspace.state)

        def _build_initial_workspace_state(self) -> AnalysisWorkspaceState:
            foreground = build_demo_spectrum()
            background = build_demo_background_spectrum()
            overlay = build_demo_overlay_spectrum()
            loaded_spectra = (
                LoadedSpectrumRecord(
                    key="demo-foreground",
                    label="Demo Foreground",
                    spectrum=foreground,
                ),
                LoadedSpectrumRecord(
                    key="demo-background",
                    label="Demo Background",
                    spectrum=background,
                ),
                LoadedSpectrumRecord(
                    key="demo-overlay",
                    label="Demo Overlay",
                    spectrum=overlay,
                ),
            )
            spectra = (
                SpectrumSlot(
                    key="foreground",
                    label="Foreground",
                    spectrum=foreground,
                    source_key="demo-foreground",
                    source_label="Demo Foreground",
                ),
                SpectrumSlot(
                    key="background",
                    label="Background",
                    spectrum=background,
                    source_key="demo-background",
                    source_label="Demo Background",
                ),
                SpectrumSlot(
                    key="overlay",
                    label="Secondary Overlay",
                    spectrum=overlay,
                    source_key="demo-overlay",
                    source_label="Demo Overlay",
                ),
            )
            return AnalysisWorkspaceState(
                spectra=spectra,
                loaded_spectra=loaded_spectra,
                active_spectrum_key="foreground",
            )

        def _build_menu_bar(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self._action("Open Spectrum...", "Ctrl+O"))
            file_menu.addAction(self._action("Open Session...", "Ctrl+Shift+O"))
            file_menu.addSeparator()
            self._report_export_action = self._action(
                "Export Report...",
                "Ctrl+E",
                enabled=self.reporting_engine.template_backend_available(),
                handler=self._open_report_export,
            )
            file_menu.addAction(self._report_export_action)
            file_menu.addAction(self._action("Export ANSI N42.42...", "Ctrl+Shift+E"))

            edit_menu = self.menuBar().addMenu("&Edit")
            undo_action = self.undo_stack.createUndoAction(self, "Undo")
            undo_action.setShortcut(QKeySequence("Ctrl+Z"))
            edit_menu.addAction(undo_action)
            redo_action = self.undo_stack.createRedoAction(self, "Redo")
            redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
            edit_menu.addAction(redo_action)

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
            analysis_menu.addAction(
                self._action(
                    "Auto Find Peaks",
                    "Ctrl+A",
                    enabled=True,
                    handler=self._run_auto_peak_search,
                )
            )
            analysis_menu.addAction(self._action("Nuclide Search"))
            analysis_menu.addAction(
                self._action(
                    "Run ASTM Check",
                    enabled=True,
                    handler=self._open_standards_review,
                )
            )
            analysis_menu.addAction(
                self._action(
                    "Spectrum Unfolding Workspace",
                    enabled=True,
                    handler=self._open_unfolding_workspace,
                )
            )
            self._pu_isotopics_action = self._action(
                "Pu Isotopics Wizard...",
                enabled=True,
                handler=self._open_pu_isotopics_wizard,
            )
            analysis_menu.addAction(self._pu_isotopics_action)

            calibration_menu = self.menuBar().addMenu("&Calibration")
            calibration_menu.addAction(
                self._action(
                    "Manual Workflow",
                    enabled=True,
                    handler=self._open_manual_calibration_workflow,
                )
            )
            calibration_menu.addAction(
                self._action(
                    "Standards Workflow",
                    enabled=True,
                    handler=self._open_standards_calibration_workflow,
                )
            )
            calibration_menu.addAction(
                self._action(
                    "Energy + FWHM Workspace",
                    enabled=True,
                    handler=self._open_energy_fwhm_workspace,
                )
            )
            calibration_menu.addAction(
                self._action(
                    "Quick Slider Mode",
                    enabled=True,
                    handler=self._open_quick_slider_calibration_mode,
                )
            )

            tools_menu = self.menuBar().addMenu("&Tools")
            self._qa_history_action = self._action(
                "QA History",
                enabled=True,
                handler=self._open_qa_history,
            )
            tools_menu.addAction(self._qa_history_action)
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
                workspace_controller=self.analysis_workspace,
                qa_monitor=self.qa_monitor,
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
                SidebarPanel(
                    mode_manager=self.mode_manager,
                    selection_bus=self.selection_bus,
                    workspace_controller=self.analysis_workspace,
                    library_manager=self.library_manager,
                    qa_monitor=self.qa_monitor,
                    open_qa_history=self._open_qa_history,
                    open_standards_review=self._open_standards_review,
                    parent=self,
                ),
                Qt.LeftDockWidgetArea,
            )
            self.bottom_dock = self._wrap_dock(
                "Analysis Surface",
                BottomWorkspaceTabs(
                    mode_manager=self.mode_manager,
                    selection_bus=self.selection_bus,
                    workspace_controller=self.analysis_workspace,
                    library_manager=self.library_manager,
                    undo_stack=self.undo_stack,
                    open_calibration_workspace=self._open_energy_fwhm_workspace,
                    open_quick_slider_calibration_workspace=self._open_quick_slider_calibration_mode,
                    open_manual_calibration_workspace=self._open_manual_calibration_workflow,
                    open_standards_calibration_workspace=self._open_standards_calibration_workflow,
                    parent=self,
                ),
                Qt.BottomDockWidgetArea,
            )
            self.right_dock = self._wrap_dock(
                "Inspector",
                ToolContextPanel(
                    mode_manager=self.mode_manager,
                    selection_bus=self.selection_bus,
                    workspace_controller=self.analysis_workspace,
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
            self.library_label = QLabel("Library: bundled gamma", self)
            self.renderer_label = QLabel("Renderer: PyQtGraph", self)
            self.predictive_label = QLabel("Predictive: --", self)
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
            status.addPermanentWidget(self.library_label)
            status.addPermanentWidget(self.renderer_label)
            status.addPermanentWidget(self.predictive_label)
            status.addPermanentWidget(self.progress)
            status.addPermanentWidget(self.hardware_led)
            self.hardware_led.set_status("offline", "NO DEVICE")
            self.hardware_led.set_click_handler(self._open_dashboard_tab)
            self.library_manager.subscribe(self._on_library_state_changed)
            self._on_library_state_changed(self.library_manager.state)
            self._update_predictive_status()

        def _action(
            self,
            text: str,
            shortcut: str | None = None,
            *,
            enabled: bool = False,
            handler=None,
        ) -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.setEnabled(enabled)
            if handler is not None:
                action.triggered.connect(handler)
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
            if hasattr(self, "_pu_isotopics_action"):
                self._pu_isotopics_action.setEnabled(state.mode is not GUIMode.SIMPLE)

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
            self._update_predictive_status()

        def _on_library_state_changed(self, _state) -> None:
            label = self.library_manager.record_for_category("gamma_identification").label
            self.library_label.setText(f"Library: {label}")

        def _on_workspace_state_changed(self, state) -> None:
            active_slot = next(
                (slot for slot in state.spectra if slot.key == state.active_spectrum_key),
                None,
            )
            if active_slot is not None:
                self.file_label.setText(
                    f"File: {active_slot.source_label or active_slot.spectrum.spectrum_id or active_slot.label}"
                )
            self._update_predictive_status()

        def _update_predictive_status(self) -> None:
            active = self.analysis_workspace.spectrum()
            if active is None:
                self.predictive_label.setText("Predictive: --")
                return
            records = list(self.analysis_workspace.loaded_spectrum_records())
            records.sort(
                key=lambda record: (
                    record.spectrum.start_time or datetime.max,
                    record.label,
                )
            )
            history = tuple(record.spectrum for record in records) or (active,)
            forecast = estimate_count_target_forecast(
                active,
                roi_bounds_keV=self.selection_bus.state.roi_bounds_keV,
                target_counts=10000.0,
                history_spectra=history,
            )
            recalibration = estimate_recalibration_forecast(self.qa_monitor.history())
            eta_text = (
                f"{int(round(forecast.eta_seconds / 60.0))}m"
                if forecast.eta_seconds and forecast.eta_seconds > 60.0
                else f"{int(round(forecast.eta_seconds or 0.0))}s"
            )
            qa_text = (
                f"QA {int(round(recalibration.days_until_recalibration))}d"
                if recalibration is not None
                and recalibration.days_until_recalibration is not None
                else "QA stable"
            )
            self.predictive_label.setText(f"Predictive: ETA {eta_text} | {qa_text}")

        def open_path(self, path: str | Path) -> None:
            """Open a spectrum or session file without a modal file dialog."""

            source = Path(path)
            if source.suffix.lower() == ".ffs":
                session = read_ffs_session(source)
                loaded_keys: list[str] = []
                for index, spectrum in enumerate(session.spectra):
                    session_source = (
                        session.source_files[index]
                        if index < len(session.source_files)
                        else None
                    )
                    label = (
                        Path(session_source).name
                        if session_source
                        else (spectrum.spectrum_id or f"{source.stem} spectrum {index + 1}")
                    )
                    loaded_keys.append(
                        self.analysis_workspace.register_loaded_spectrum(
                            spectrum,
                            label=label,
                            source_path=session_source,
                        )
                    )
                if loaded_keys:
                    active_index = min(
                        max(session.active_spectrum_index, 0),
                        len(loaded_keys) - 1,
                    )
                    active_key = loaded_keys[active_index]
                    slot_key = self.analysis_workspace.state.active_spectrum_key
                    self.analysis_workspace.assign_loaded_spectrum_to_slot(
                        active_key,
                        slot_key,
                    )
                self.recent_files.record_many(session.recent_files or [source])
            else:
                spectrum = read_spectrum_any(source)
                loaded_key = self.analysis_workspace.register_loaded_spectrum(
                    spectrum,
                    label=source.name,
                    source_path=str(source),
                )
                if hasattr(self.central_tabs, "load_spectrum"):
                    self.central_tabs.load_spectrum(
                        spectrum,
                        source_key=loaded_key,
                        source_label=source.name,
                        source_path=str(source),
                    )
                self.recent_files.record(source)
            self.file_label.setText(f"File: {source.name}")
            self._refresh_analysis_workspace_derivatives()

        def dragEnterEvent(self, event) -> None:
            mime_data = event.mimeData()
            paths = []
            if mime_data.hasUrls():
                paths = [url.toLocalFile() for url in mime_data.urls()]
            elif mime_data.hasText():
                paths = mime_data.text().splitlines()
            if normalize_dropped_paths(paths):
                event.acceptProposedAction()
                return
            super().dragEnterEvent(event)

        def dropEvent(self, event) -> None:
            mime_data = event.mimeData()
            paths = []
            if mime_data.hasUrls():
                paths = [url.toLocalFile() for url in mime_data.urls()]
            elif mime_data.hasText():
                paths = mime_data.text().splitlines()

            normalized = normalize_dropped_paths(paths)
            if not normalized:
                super().dropEvent(event)
                return

            for path in normalized:
                self.open_path(path)
            event.acceptProposedAction()

        def _open_energy_fwhm_workspace(self, advanced_tab: str | None = None) -> None:
            if self._calibration_dialog is not None and self._calibration_dialog.isVisible():
                if advanced_tab is not None:
                    self._calibration_dialog.set_active_advanced_tab(advanced_tab)
                self._calibration_dialog.raise_()
                self._calibration_dialog.activateWindow()
                return

            current_spectrum = None
            if hasattr(self.central_tabs, "current_spectrum"):
                current_spectrum = self.central_tabs.current_spectrum()

            self._calibration_dialog = CalibrationWorkspaceDialog(
                spectrum=current_spectrum,
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                library_manager=self.library_manager,
                on_apply=self._apply_calibration_workspace_result,
                parent=self,
            )
            if advanced_tab is not None:
                self._calibration_dialog.set_active_advanced_tab(advanced_tab)
            self._calibration_dialog.show()

        def _open_manual_calibration_workflow(self) -> None:
            if self.mode_manager.state.mode is not GUIMode.EXPERT:
                self.mode_manager.set_mode(GUIMode.EXPERT)
            self._open_energy_fwhm_workspace()

        def _open_standards_calibration_workflow(self) -> None:
            state = self.mode_manager.state
            if state.mode is not GUIMode.STANDARDS:
                self.mode_manager.set_standard(state.standard or "ASTM E181")
            self._open_energy_fwhm_workspace()

        def _open_quick_slider_calibration_mode(self) -> None:
            self._open_energy_fwhm_workspace(advanced_tab="quick_slider")

        def _run_auto_peak_search(self) -> None:
            bottom_widget = self.bottom_dock.widget()
            if hasattr(bottom_widget, "run_auto_peak_search"):
                bottom_widget.run_auto_peak_search()

        def _open_unfolding_workspace(self) -> None:
            if self._unfolding_dialog is not None and self._unfolding_dialog.isVisible():
                self._unfolding_dialog.raise_()
                self._unfolding_dialog.activateWindow()
                return

            self._unfolding_dialog = UnfoldingWorkspaceDialog(
                mode_manager=self.mode_manager,
                parent=self,
            )
            self._unfolding_dialog.show()

        def _build_standards_context(self) -> StandardsEvaluationContext:
            latest = next(iter(self.qa_monitor.status_snapshot()), None)
            return StandardsEvaluationContext(
                calibration_order=2,
                max_residual_keV=0.18,
                efficiency_uncertainty_pct=2.6,
                fwhm_at_413_keV=1.08,
                qa_centroid_drift_keV=(latest.centroid_drift_keV if latest else 0.0),
                qa_fwhm_degradation_pct=(
                    latest.fwhm_degradation_pct if latest else 0.0
                ),
                before_calibration=(latest.last_check if latest else None),
                measured_at=(latest.last_check if latest else None),
                after_calibration=(latest.last_check if latest else None),
                net_counts={"primary": 1200.0, "Pu-240 160.3": 1205.0},
                line_observations={"c1030": ()},
            )

        def _build_report_context(self, template_name: str) -> dict[str, object]:
            state = self.analysis_workspace.state
            peak_rows = "".join(
                f"<tr><td>{peak.energy_keV:.3f}</td><td>{peak.nuclide or 'Unassigned'}</td><td>{peak.net_counts:.1f}</td></tr>"
                for peak in state.peaks
            ) or "<tr><td colspan='3'>No peaks</td></tr>"
            peak_table = (
                "<table><tr><th>Energy</th><th>Nuclide</th><th>Net Counts</th></tr>"
                + peak_rows
                + "</table>"
            )
            activity_rows = "".join(
                f"<tr><td>{result.nuclide}</td><td>{result.line_energy_keV:.3f}</td><td>{result.activity_bq:.3f}</td><td>{result.uncertainty_bq:.3f}</td></tr>"
                for result in state.activity_results
            ) or "<tr><td colspan='4'>No activity results</td></tr>"
            activity_table = (
                "<table><tr><th>Nuclide</th><th>Line</th><th>Activity</th><th>σ</th></tr>"
                + activity_rows
                + "</table>"
            )
            standards_rows = []
            context = self._build_standards_context()
            for key in ("ASTM E181", "ASTM E1297", "ASTM E1218", "ASTM C1232", "ASTM C1030"):
                module = self.registries.standards_modules.get(key)
                evaluation = module.evaluate(context)
                standards_rows.append(
                    f"<tr><td>{module.display_name}</td><td>{evaluation.overall_status}</td><td>{evaluation.summary}</td></tr>"
                )
            astm_status_table = (
                "<table><tr><th>Standard</th><th>Status</th><th>Summary</th></tr>"
                + "".join(standards_rows)
                + "</table>"
            )
            qa_status_snapshot = "<br/>".join(
                f"{item.nuclide} {item.energy_keV:.2f} keV | drift {item.centroid_drift_keV:+.3f} keV | FWHM {item.fwhm_degradation_pct:+.2f}%"
                for item in self.qa_monitor.status_snapshot()
            ) or "No QA snapshot available."
            provenance = (
                f"mode={self.mode_manager.state.mode.value}\n"
                f"standard={self.mode_manager.state.standard}\n"
                f"gamma_library={self.library_manager.state.gamma_identification_source_id}\n"
                f"peaks={len(state.peaks)}\n"
                f"activity_results={len(state.activity_results)}"
            )
            batch_rows = "No batch results yet."
            aggregate_csv = ""
            bottom_widget = self.bottom_dock.widget()
            if hasattr(bottom_widget, "batch_queue_panel"):
                panel = bottom_widget.batch_queue_panel
                if getattr(panel, "results", ()):
                    batch_rows = "<br/>".join(
                        f"{result.label}: {result.peak_count} peaks, {result.backend}"
                        for result in panel.results
                    )
                    aggregate_csv = (panel.last_output_dir / "aggregate.csv").read_text(encoding="utf-8") if panel.last_output_dir and (panel.last_output_dir / "aggregate.csv").exists() else ""
            payload = {
                "title": "FluxForge Module 3 Report",
                "spectrum_image": "Modern Qt spectrum canvas snapshot",
                "calibration_curve": "Embedded calibration curve placeholder",
                "calibration_residuals": "Embedded calibration residuals placeholder",
                "efficiency_curve": "Embedded efficiency curve placeholder",
                "efficiency_residuals": "Embedded efficiency residuals placeholder",
                "residuals_grid": "Embedded residual thumbnails placeholder",
                "peak_table": peak_table,
                "activity_table": activity_table,
                "astm_status_table": astm_status_table,
                "qa_status_snapshot": qa_status_snapshot,
                "provenance": provenance,
                "batch_rows": batch_rows,
                "aggregate_csv": aggregate_csv or "No batch CSV available.",
            }
            return payload

        def _build_standards_evaluations(self):
            context = self._build_standards_context()
            evaluations = []
            for key in ("ASTM E181", "ASTM E1297", "ASTM E1218", "ASTM C1232", "ASTM C1030"):
                module = self.registries.standards_modules.get(key)
                evaluations.append(module.evaluate(context))
            return tuple(evaluations)

        def _open_report_export(self) -> None:
            if not self.reporting_engine.template_backend_available():
                self.statusBar().showMessage(
                    "Report export requires the optional reporting extra (`Jinja2`).",
                    6000,
                )
                return
            if self._report_dialog is not None and self._report_dialog.isVisible():
                self._report_dialog.raise_()
                self._report_dialog.activateWindow()
                return
            self._report_dialog = ReportExportDialog(
                engine=self.reporting_engine,
                context_factory=self._build_report_context,
                parent=self,
            )
            self._report_dialog.show()

        def _open_standards_review(self) -> None:
            if (
                self._standards_review_dialog is not None
                and self._standards_review_dialog.isVisible()
            ):
                self._standards_review_dialog.refresh()
                self._standards_review_dialog.raise_()
                self._standards_review_dialog.activateWindow()
                return
            self._standards_review_dialog = StandardsReviewDialog(
                evaluation_factory=self._build_standards_evaluations,
                parent=self,
            )
            self._standards_review_dialog.show()

        def _open_qa_history(self) -> None:
            if self._qa_history_dialog is not None and self._qa_history_dialog.isVisible():
                self._qa_history_dialog.raise_()
                self._qa_history_dialog.activateWindow()
                return
            self._qa_history_dialog = QAHistoryDialog(self.qa_monitor, parent=self)
            self._qa_history_dialog.show()

        def _open_dashboard_tab(self) -> None:
            if hasattr(self, "central_tabs"):
                self.central_tabs.setCurrentIndex(1)

        def _open_pu_isotopics_wizard(self) -> None:
            if self.mode_manager.state.mode is GUIMode.SIMPLE:
                return
            if self._pu_isotopics_dialog is not None and self._pu_isotopics_dialog.isVisible():
                self._pu_isotopics_dialog.raise_()
                self._pu_isotopics_dialog.activateWindow()
                return
            self._pu_isotopics_dialog = PuIsotopicsDialog(
                peaks=self.analysis_workspace.state.peaks,
                parent=self,
            )
            self._pu_isotopics_dialog.show()

        def _apply_calibration_workspace_result(
            self,
            spectrum,
            energy_fit,
            fwhm_fit,
        ) -> None:
            if hasattr(self.central_tabs, "load_spectrum"):
                self.central_tabs.load_spectrum(spectrum)
            self._refresh_analysis_workspace_derivatives()
            self.file_label.setText(
                f"File: {spectrum.spectrum_id or 'workspace spectrum'}"
            )
            message = (
                f"Applied calibration order {energy_fit.order} "
                f"(RMS {energy_fit.rms_keV:.4f} keV)"
            )
            if fwhm_fit is not None:
                message += f" | FWHM RMS {fwhm_fit.rms_keV:.4f} keV"
            self.statusBar().showMessage(message, 6000)
            self.progress.setValue(72)
            self.progress.setFormat("Calibration applied")

        def _refresh_analysis_workspace_derivatives(self) -> None:
            from fluxforge.core.analysis_workspace import compute_cascade_sum_lines, extract_survey_points

            spectra = [
                (slot.key, slot.spectrum)
                for slot in self.analysis_workspace.state.spectra
            ]
            self.analysis_workspace.set_survey_points(extract_survey_points(spectra))
            self.analysis_workspace.set_cascade_sum_lines(
                compute_cascade_sum_lines(
                    self.analysis_workspace.state.pinned_nuclides,
                    source_id=self.library_manager.state.gamma_identification_source_id,
                    custom_path=self.library_manager.state.custom_gamma_path,
                )
            )

        def closeEvent(self, event) -> None:
            self._save_layout()
            super().closeEvent(event)


else:

    class FluxForgeMainWindow:  # pragma: no cover - placeholder without Qt
        """Import-safe placeholder when the optional Qt dependencies are missing."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(modern_gui_unavailable_message())
