"""Main window for the FluxForge desktop GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from fluxforge.gui.file_workflow import RecentFilesManager, normalize_dropped_paths
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.workflow_presets import WorkflowPresetManager
from fluxforge.gui.analysis_workspace import (
    LoadedSpectrumRecord,
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    SpectrumSlot,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.workspace_undo import ApplyCalibrationCommand
from fluxforge.io import (
    FluxForgeSession,
    read_ffs_session,
    read_spectrum_any,
    write_ffs_session,
)
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.core.workspace_document import CalibrationModel, DetectorProfile
from fluxforge.reporting.engine import ReportingEngine
from fluxforge.standards import (
    QAMonitor,
    StandardsEvaluationContext,
    register_builtin_standards_modules,
)
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
        QComboBox,
        QDockWidget,
        QFileDialog,
        QInputDialog,
        QKeySequence,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSettings,
        QStatusBar,
        QToolBar,
        QUndoStack,
        Qt,
    )
    from PySide6.QtWidgets import QMessageBox
    from fluxforge.gui.theme_manager import load_stylesheet, resolve_theme
    from fluxforge.gui.widgets import HardwareLedWidget, ModeSwitcherWidget


@dataclass(frozen=True)
class DockZone:
    """Logical dock zone in the desktop workspace."""

    code: str
    title: str
    description: str


DEFAULT_DOCK_ZONES = (
    DockZone("A", "Menu / Toolbar", "Global actions, workflow mode, and shortcuts."),
    DockZone("B", "Left Sidebar", "Files, nuclides, results, and QA surfaces."),
    DockZone(
        "C", "Central Canvas", "Primary spectrum and shared analytical workspace."
    ),
    DockZone(
        "D", "Bottom Panels", "Peak table, calibration, activity, batch, and log."
    ),
    DockZone("E", "Right Sidebar", "Context-sensitive tool parameters."),
    DockZone("F", "Status Bar", "Cursor readout, hardware LED, and task progress."),
)


@dataclass
class MainWindowScaffold:
    """Lightweight representation of the main-window layout."""

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
        "The FluxForge desktop GUI requires the optional native GUI extras "
        "(`PySide6` and `pyqtgraph`). "
        f"Current import status: {reason}"
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class FluxForgeMainWindow(QMainWindow):
        """Dockable analyst workspace for HPGe and neutron-spectrum workflows."""

        ORGANIZATION = "FluxForge"
        APPLICATION = "FluxForgeNext"

        def __init__(
            self,
            mode_manager: ModeManager | None = None,
            selection_bus: SelectionBus | None = None,
            settings=None,
            qa_monitor: QAMonitor | None = None,
            developer_tools: bool = False,
            load_example: bool = False,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("FluxForgeMainWindow")
            self.setWindowTitle("FluxForge — HPGe Analysis")
            self.resize(1560, 980)

            self.settings = settings or QSettings(self.ORGANIZATION, self.APPLICATION)
            self.developer_tools = bool(developer_tools)
            self.mode_manager = mode_manager or ModeManager(settings=self.settings)
            self.selection_bus = selection_bus or SelectionBus.shared()
            self.library_manager = DataLibraryManager(settings=self.settings)
            self.workflow_presets = WorkflowPresetManager(settings=self.settings)
            self.recent_files = RecentFilesManager(self.settings)
            self.qa_monitor = qa_monitor or QAMonitor()
            if load_example:
                self.qa_monitor.seed_demo_history()
            self.reporting_engine = ReportingEngine()
            self.registries = bootstrap_builtin_registries()
            register_builtin_standards_modules(self.registries)
            self.undo_stack = QUndoStack(self)
            self.analysis_workspace = AnalysisWorkspaceController(
                self._build_initial_workspace_state(include_example=load_example)
            )
            self._session_path: Path | None = None
            self._session_device_snapshot: list[dict] = []
            self._session_metadata: dict = {}
            self._session_created_at: str | None = None
            self._session_recent_files: tuple[str, ...] = ()
            self._document_dirty = False
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
            self.workflow_presets.subscribe(self._on_workflow_presets_changed)
            self._refresh_workflow_controls()

            self.mode_manager.subscribe(self._on_mode_state_changed)
            self.selection_bus.subscribe(self._on_selection_changed)
            self.analysis_workspace.subscribe(self._on_workspace_state_changed)
            self.analysis_workspace.subscribe_document(
                self._on_workspace_document_changed
            )
            self._on_mode_state_changed(self.mode_manager.state)
            self._on_workspace_state_changed(self.analysis_workspace.state)

        def _build_initial_workspace_state(
            self,
            *,
            include_example: bool = False,
        ) -> AnalysisWorkspaceState:
            if not include_example:
                return AnalysisWorkspaceState()
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

        def _workspace_contains_bundled_example(self) -> bool:
            return any(
                str(slot.source_key or "").startswith("demo-")
                for slot in self.analysis_workspace.state.spectra
            )

        def _build_menu_bar(self) -> None:
            self._requires_spectrum_actions: list[QAction] = []
            self._requires_example_actions: list[QAction] = []
            file_menu = self.menuBar().addMenu("&File")
            file_menu.setObjectName("FileMenu")
            file_menu.menuAction().setObjectName("OpenFileMenuAction")
            file_menu.addAction(
                self._action(
                    "Open Spectrum...",
                    "Ctrl+O",
                    enabled=True,
                    handler=self._open_spectrum_dialog,
                    object_name="OpenSpectrumAction",
                )
            )
            file_menu.addAction(
                self._action(
                    "Open Session...",
                    "Ctrl+Shift+O",
                    enabled=True,
                    handler=self._open_session_dialog,
                    object_name="OpenSessionAction",
                )
            )
            self._save_session_action = self._action(
                "Save Session",
                "Ctrl+S",
                enabled=True,
                handler=self.save_session,
                object_name="SaveSessionAction",
            )
            file_menu.addAction(self._save_session_action)
            file_menu.addAction(
                self._action(
                    "Save Session As...",
                    "Ctrl+Shift+S",
                    enabled=True,
                    handler=self._save_session_as_dialog,
                    object_name="SaveSessionAsAction",
                )
            )
            file_menu.addAction(
                self._action(
                    "Open Example",
                    enabled=True,
                    handler=self._load_example_workspace,
                    object_name="OpenExampleAction",
                )
            )
            file_menu.addSeparator()
            self._report_export_action = self._action(
                "Export Report...",
                "Ctrl+E",
                enabled=self.reporting_engine.template_backend_available(),
                handler=self._open_report_export,
                object_name="ExportReportAction",
            )
            file_menu.addAction(self._report_export_action)
            if self.developer_tools:
                file_menu.addAction(
                    self._unavailable_action(
                        "Export ANSI N42.42...",
                        "ANSI N42.42 export is not implemented in this build.",
                        shortcut="Ctrl+Shift+E",
                    )
                )

            edit_menu = self.menuBar().addMenu("&Edit")
            edit_menu.setObjectName("EditMenu")
            edit_menu.menuAction().setObjectName("OpenEditMenuAction")
            undo_action = self.undo_stack.createUndoAction(self, "Undo")
            undo_action.setObjectName("UndoAction")
            undo_action.setShortcut(QKeySequence("Ctrl+Z"))
            edit_menu.addAction(undo_action)
            redo_action = self.undo_stack.createRedoAction(self, "Redo")
            redo_action.setObjectName("RedoAction")
            redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
            edit_menu.addAction(redo_action)

            view_menu = self.menuBar().addMenu("&View")
            view_menu.setObjectName("ViewMenu")
            view_menu.menuAction().setObjectName("OpenViewMenuAction")
            view_menu.addAction(
                self._action(
                    "Toggle Full Canvas",
                    "F11",
                    enabled=True,
                    handler=self._toggle_full_canvas,
                    object_name="ToggleFullCanvasAction",
                )
            )
            view_menu.addAction(
                self._action(
                    "Restore Default Layout",
                    enabled=True,
                    handler=self._restore_default_layout,
                    object_name="RestoreDefaultLayoutAction",
                )
            )
            self._log_scale_action = self._action(
                "Log Scale",
                "Ctrl+L",
                enabled=True,
                handler=self._toggle_log_scale,
                checkable=True,
                checked=False,
                object_name="ToggleLogScaleAction",
            )
            view_menu.addAction(self._log_scale_action)
            self._peak_labels_action = self._action(
                "Peak Labels",
                enabled=True,
                handler=self._toggle_peak_labels,
                checkable=True,
                checked=True,
                object_name="TogglePeakLabelsAction",
            )
            view_menu.addAction(self._peak_labels_action)
            if self.developer_tools:
                renderer_menu = view_menu.addMenu("Renderer Diagnostics")
                renderer_menu.setObjectName("RendererDiagnosticsMenu")
                renderer_menu.menuAction().setObjectName(
                    "OpenRendererDiagnosticsMenuAction"
                )
                for backend in available_renderer_status():
                    label = backend["display_name"]
                    if backend["recommended"]:
                        label += " ★"
                    availability = (
                        "available" if backend["available"] else "unavailable"
                    )
                    renderer_menu.addAction(
                        self._unavailable_action(
                            f"{label} ({availability})",
                            "Runtime renderer switching is unavailable.",
                        )
                    )

            analysis_menu = self.menuBar().addMenu("&Analysis")
            analysis_menu.setObjectName("AnalysisMenu")
            analysis_menu.menuAction().setObjectName("OpenAnalysisMenuAction")
            self._auto_peak_action = self._action(
                "Auto Find Peaks",
                "Ctrl+A",
                enabled=self.analysis_workspace.spectrum() is not None,
                handler=self._run_auto_peak_search,
                object_name="AutoFindPeaksAction",
            )
            analysis_menu.addAction(self._auto_peak_action)
            self._requires_spectrum_actions.append(self._auto_peak_action)
            analysis_menu.addAction(
                self._action(
                    "Nuclide Search",
                    enabled=True,
                    handler=self._focus_nuclide_search,
                    object_name="FocusNuclideSearchAction",
                )
            )
            self._run_astm_check_action = self._action(
                "Run ASTM Check",
                enabled=self.analysis_workspace.spectrum() is not None,
                handler=self._open_standards_review,
                object_name="RunAstmCheckAction",
            )
            analysis_menu.addAction(self._run_astm_check_action)
            self._requires_spectrum_actions.append(self._run_astm_check_action)
            self._unfolding_action = self._action(
                "Spectrum Unfolding Workspace",
                enabled=self._workspace_contains_bundled_example(),
                handler=self._open_unfolding_workspace,
                object_name="OpenSpectrumUnfoldingAction",
            )
            analysis_menu.addAction(self._unfolding_action)
            self._requires_example_actions.append(self._unfolding_action)
            self._pu_isotopics_action = self._action(
                "Pu Isotopics Wizard...",
                enabled=self.analysis_workspace.spectrum() is not None,
                handler=self._open_pu_isotopics_wizard,
                object_name="OpenPuIsotopicsAction",
            )
            analysis_menu.addAction(self._pu_isotopics_action)
            self._requires_spectrum_actions.append(self._pu_isotopics_action)

            calibration_menu = self.menuBar().addMenu("&Calibration")
            calibration_menu.setObjectName("CalibrationMenu")
            calibration_menu.menuAction().setObjectName("OpenCalibrationMenuAction")
            for calibration_action in (
                self._action(
                    "Manual Workflow",
                    enabled=self.analysis_workspace.spectrum() is not None,
                    handler=self._open_manual_calibration_workflow,
                    object_name="OpenManualCalibrationAction",
                ),
                self._action(
                    "Standards Workflow",
                    enabled=self.analysis_workspace.spectrum() is not None,
                    handler=self._open_standards_calibration_workflow,
                    object_name="OpenStandardsCalibrationAction",
                ),
                self._action(
                    "Energy + FWHM Workspace",
                    enabled=self.analysis_workspace.spectrum() is not None,
                    handler=self._open_energy_fwhm_workspace,
                    object_name="OpenEnergyFwhmCalibrationAction",
                ),
                self._action(
                    "Quick Slider Mode",
                    enabled=self.analysis_workspace.spectrum() is not None,
                    handler=self._open_quick_slider_calibration_mode,
                    object_name="OpenQuickSliderCalibrationAction",
                ),
            ):
                calibration_menu.addAction(calibration_action)
                self._requires_spectrum_actions.append(calibration_action)

            tools_menu = self.menuBar().addMenu("&Tools")
            tools_menu.setObjectName("ToolsMenu")
            tools_menu.menuAction().setObjectName("OpenToolsMenuAction")
            self._qa_history_action = self._action(
                "QA History",
                enabled=True,
                handler=self._open_qa_history,
                object_name="OpenQaHistoryAction",
            )
            tools_menu.addAction(self._qa_history_action)
            if self.developer_tools:
                tools_menu.addAction(
                    self._unavailable_action(
                        "Hardware Diagnostics",
                        "Hardware acquisition is unavailable in this build.",
                    )
                )

            workspace_menu = self.menuBar().addMenu("&Workspaces")
            workspace_menu.setObjectName("WorkspacesMenu")
            workspace_menu.menuAction().setObjectName("OpenWorkspacesMenuAction")
            workspace_menu.addAction(
                self._action(
                    "Analysis Surface",
                    enabled=True,
                    handler=self._focus_analysis_surface_dock,
                    object_name="FocusAnalysisSurfaceAction",
                )
            )
            workspace_menu.addAction(
                self._action(
                    "Workspace Sidebar",
                    enabled=True,
                    handler=self._focus_workspace_dock,
                    object_name="FocusWorkspaceSidebarAction",
                )
            )
            workspace_menu.addAction(
                self._action(
                    "Tool Inspector",
                    enabled=True,
                    handler=self._focus_inspector_dock,
                    object_name="FocusToolInspectorAction",
                )
            )
            workspace_menu.addSeparator()
            workspace_menu.addAction(
                self._action(
                    "Open QA History",
                    enabled=True,
                    handler=self._open_qa_history,
                    object_name="WorkspaceOpenQaHistoryAction",
                )
            )
            self._workspace_standards_review_action = self._action(
                "Open Standards Review",
                enabled=self.analysis_workspace.spectrum() is not None,
                handler=self._open_standards_review,
                object_name="WorkspaceOpenStandardsReviewAction",
            )
            workspace_menu.addAction(self._workspace_standards_review_action)
            self._requires_spectrum_actions.append(
                self._workspace_standards_review_action
            )
            workspace_menu.addAction(
                self._action(
                    "Open Report Export",
                    enabled=True,
                    handler=self._open_report_export,
                    object_name="WorkspaceOpenReportExportAction",
                )
            )
            self._workspace_unfolding_action = self._action(
                "Open Unfolding Workspace",
                enabled=self._workspace_contains_bundled_example(),
                handler=self._open_unfolding_workspace,
                object_name="WorkspaceOpenUnfoldingAction",
            )
            workspace_menu.addAction(self._workspace_unfolding_action)
            self._requires_example_actions.append(self._workspace_unfolding_action)
            workspace_menu.addSeparator()
            workspace_menu.addAction(
                self._action(
                    "Load Saved Workflow",
                    enabled=True,
                    handler=self._load_selected_workflow,
                    object_name="LoadSavedWorkflowAction",
                )
            )
            workspace_menu.addAction(
                self._action(
                    "Save Current Workflow...",
                    enabled=True,
                    handler=self._save_current_workflow_dialog,
                    object_name="SaveCurrentWorkflowAction",
                )
            )
            workspace_menu.addAction(
                self._action(
                    "Delete Saved Workflow",
                    enabled=True,
                    handler=self._delete_selected_workflow,
                    object_name="DeleteSavedWorkflowAction",
                )
            )

            help_menu = self.menuBar().addMenu("&Help")
            help_menu.setObjectName("HelpMenu")
            help_menu.menuAction().setObjectName("OpenHelpMenuAction")
            help_menu.addAction(
                self._action(
                    "Shortcut Reference",
                    "F1",
                    enabled=True,
                    handler=self._show_shortcut_reference,
                    object_name="ShowShortcutReferenceAction",
                )
            )
            help_menu.addAction(
                self._action(
                    "About FluxForge",
                    enabled=True,
                    handler=self._show_about,
                    object_name="ShowAboutFluxForgeAction",
                )
            )

        def _build_toolbar(self) -> None:
            toolbar = QToolBar("Primary", self)
            toolbar.setObjectName("PrimaryToolbar")
            toolbar.toggleViewAction().setObjectName("TogglePrimaryToolbarAction")
            self.primary_toolbar = toolbar
            toolbar.setMovable(False)
            toolbar.addWidget(ModeSwitcherWidget(self.mode_manager, toolbar))
            toolbar.addSeparator()
            workflow_label = QLabel("Workflow", toolbar)
            workflow_label.setObjectName("WorkflowPresetLabel")
            toolbar.addWidget(workflow_label)
            self.workflow_combo = QComboBox(toolbar)
            self.workflow_combo.setObjectName("WorkflowPresetCombo")
            self.workflow_combo.setMinimumContentsLength(24)
            toolbar.addWidget(self.workflow_combo)
            self.load_workflow_button = QPushButton("Load", toolbar)
            self.load_workflow_button.setObjectName("LoadWorkflowPresetButton")
            self.load_workflow_button.clicked.connect(self._load_selected_workflow)
            toolbar.addWidget(self.load_workflow_button)
            self.save_workflow_button = QPushButton("Save", toolbar)
            self.save_workflow_button.setObjectName("SaveWorkflowPresetButton")
            self.save_workflow_button.clicked.connect(
                self._save_current_workflow_dialog
            )
            toolbar.addWidget(self.save_workflow_button)
            self.delete_workflow_button = QPushButton("Delete", toolbar)
            self.delete_workflow_button.setObjectName("DeleteWorkflowPresetButton")
            self.delete_workflow_button.clicked.connect(self._delete_selected_workflow)
            toolbar.addWidget(self.delete_workflow_button)
            toolbar.addSeparator()
            toolbar.addAction(self._log_scale_action)
            toolbar.addAction(self._peak_labels_action)
            self.addToolBar(Qt.TopToolBarArea, toolbar)

        def _build_central_workspace(self) -> None:
            self.central_tabs = CentralWorkspaceTabs(
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                workspace_controller=self.analysis_workspace,
                qa_monitor=self.qa_monitor,
                undo_stack=self.undo_stack,
                parent=self,
            )
            self.setCentralWidget(self.central_tabs)
            if hasattr(self.central_tabs, "canvas"):
                self.central_tabs.canvas.set_log_scale(
                    self._log_scale_action.isChecked()
                )
                self.central_tabs.canvas.set_peak_labels_visible(
                    self._peak_labels_action.isChecked()
                )

        def _wrap_dock(self, title: str, widget, area) -> QDockWidget:
            dock = QDockWidget(title, self)
            dock.setObjectName(f"{title.replace(' ', '')}Dock")
            dock.toggleViewAction().setObjectName(
                f"Toggle{title.replace(' ', '')}DockAction"
            )
            dock.setWidget(widget)
            dock.setAllowedAreas(
                Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
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
                    standards_context_factory=self._build_standards_context,
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
                    developer_tools=self.developer_tools,
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
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.progress.setVisible(False)
            self.hardware_led = HardwareLedWidget(self)
            if not self.developer_tools:
                self.renderer_label.hide()
                self.hardware_led.hide()

            status.addWidget(self.cursor_label, 1)
            status.addWidget(self.file_label, 1)
            status.addPermanentWidget(self.mode_label)
            status.addPermanentWidget(self.library_label)
            if self.developer_tools:
                status.addPermanentWidget(self.renderer_label)
            status.addPermanentWidget(self.predictive_label)
            status.addPermanentWidget(self.progress)
            if self.developer_tools:
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
            checkable: bool = False,
            checked: bool = False,
            object_name: str | None = None,
        ) -> QAction:
            action = QAction(text, self)
            resolved_object_name = object_name
            if not resolved_object_name:
                stem = "".join(
                    character for character in text.title() if character.isalnum()
                )
                resolved_object_name = f"{stem}Action"
            action.setObjectName(resolved_object_name)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.setEnabled(enabled)
            action.setCheckable(checkable)
            if checkable:
                action.setChecked(checked)
            if handler is not None:
                action.triggered.connect(handler)
            return action

        def _unavailable_action(
            self,
            text: str,
            reason: str,
            *,
            shortcut: str | None = None,
        ) -> QAction:
            """Create an honest disabled menu item with a discoverable explanation."""

            action = self._action(f"{text} (not available)", shortcut)
            action.setToolTip(reason)
            action.setStatusTip(reason)
            return action

        def _open_spectrum_dialog(self) -> None:
            filename, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Open HPGe spectrum",
                self.recent_files.files()[0] if self.recent_files.files() else "",
                (
                    "Spectrum files (*.asc *.ASC *.cnf *.CNF *.chn *.CHN *.spc *.SPC "
                    "*.spe *.SPE *.n42 *.N42 *.xml *.XML *.csv *.CSV);;All files (*)"
                ),
            )
            if filename:
                self._open_dialog_path(filename)

        def _open_session_dialog(self) -> None:
            filename, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Open FluxForge session",
                self.recent_files.files()[0] if self.recent_files.files() else "",
                "FluxForge sessions (*.ffs);;All files (*)",
            )
            if filename:
                self._open_dialog_path(filename)

        def _save_session_as_dialog(self) -> bool:
            initial = str(
                self._session_path
                or (
                    Path(self.recent_files.files()[0]).with_suffix(".ffs")
                    if self.recent_files.files()
                    else Path("fluxforge-analysis.ffs")
                )
            )
            filename, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save FluxForge session",
                initial,
                "FluxForge sessions (*.ffs);;All files (*)",
            )
            if not filename:
                return False
            return self.save_session(filename)

        def save_session(self, path: str | Path | None = None) -> bool:
            """Atomically save the complete canonical analysis document."""

            if isinstance(path, bool):
                # QAction.triggered supplies its checked state.
                path = None
            target = Path(path) if path is not None else self._session_path
            if target is None:
                return self._save_session_as_dialog()
            if target.suffix.lower() != ".ffs":
                target = target.with_suffix(".ffs")

            viewport = (
                self.central_tabs.viewport_state()
                if hasattr(self.central_tabs, "viewport_state")
                else None
            )
            if viewport is not None:
                self.analysis_workspace.upsert_viewport(viewport)
            recent_files = tuple(
                dict.fromkeys(
                    (
                        str(target),
                        *self._session_recent_files,
                        *self.recent_files.files(),
                    )
                )
            )
            session = FluxForgeSession(
                document=self.analysis_workspace.document,
                recent_files=recent_files,
                device_snapshot=deepcopy(self._session_device_snapshot),
                metadata=deepcopy(self._session_metadata),
                created_at=self._session_created_at,
            )
            try:
                write_ffs_session(target, session)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Could not save session",
                    f"FluxForge could not save {target}.\n\n{exc}",
                )
                return False
            self._session_path = target.resolve()
            self._session_created_at = session.created_at
            self._session_recent_files = tuple(session.recent_files)
            self.recent_files.record(target)
            self._document_dirty = False
            self.undo_stack.setClean()
            self.setWindowModified(False)
            self.file_label.setText(f"File: {target.name}")
            self.statusBar().showMessage(f"Session saved: {target.name}", 4000)
            return True

        def _load_example_workspace(self) -> None:
            """Load the bundled deterministic HPGe example on explicit request."""

            self.qa_monitor.seed_demo_history()
            self.analysis_workspace.set_document(
                AnalysisWorkspaceController(
                    self._build_initial_workspace_state(include_example=True)
                ).document
            )
            self._session_path = None
            self._session_device_snapshot = []
            self._session_metadata = {}
            self._session_created_at = None
            self._session_recent_files = ()
            self.undo_stack.clear()
            self._refresh_analysis_workspace_derivatives()
            self.file_label.setText("File: bundled HPGe example")
            self.statusBar().showMessage("Bundled HPGe example loaded", 4000)

        def _open_dialog_path(self, filename: str) -> None:
            try:
                self.open_path(filename)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Could not open file",
                    f"FluxForge could not open {filename}.\n\n{exc}",
                )

        def _toggle_full_canvas(self) -> None:
            if getattr(self, "_full_canvas_active", False):
                self.showNormal()
                for widget, was_visible in self._full_canvas_visibility:
                    widget.setVisible(was_visible)
                self._full_canvas_active = False
                self.statusBar().showMessage("Full-canvas view disabled", 3000)
            else:
                self._full_canvas_visibility = tuple(
                    # ``isVisible()`` includes transient native-window mapping
                    # state and can report false for a shown toolbar under the
                    # Linux offscreen/Wayland backends.  ``isHidden()`` records
                    # the explicit analyst layout choice that must be restored.
                    (widget, not widget.isHidden())
                    for widget in (
                        self.left_dock,
                        self.bottom_dock,
                        self.right_dock,
                        self.primary_toolbar,
                        self.statusBar(),
                    )
                )
                for widget, _was_visible in self._full_canvas_visibility:
                    widget.hide()
                self._full_canvas_active = True
                self.showFullScreen()
                self.statusBar().showMessage(
                    "Full-canvas view enabled; press F11 to exit",
                    3000,
                )

        def _restore_default_layout(self) -> None:
            self.left_dock.show()
            self.bottom_dock.show()
            self.right_dock.show()
            self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)
            self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
            self.resize(1560, 980)
            self.statusBar().showMessage("Default workspace layout restored", 3000)

        def _focus_nuclide_search(self) -> None:
            self._focus_workspace_dock()
            sidebar = self.left_dock.widget()
            query = getattr(sidebar, "nuclide_query", None)
            if query is None:
                return
            if hasattr(sidebar, "ensureWidgetVisible"):
                sidebar.ensureWidgetVisible(query)
            query.setFocus()
            query.selectAll()
            self.statusBar().showMessage("Nuclide search ready", 3000)

        def _show_shortcut_reference(self) -> None:
            QMessageBox.information(
                self,
                "FluxForge shortcuts",
                "\n".join(
                    (
                        "Ctrl+O — Open spectrum",
                        "Ctrl+Shift+O — Open FluxForge session",
                        "Ctrl+S — Save FluxForge session",
                        "Ctrl+Shift+S — Save FluxForge session as",
                        "Ctrl+E — Export report",
                        "Ctrl+Z / Ctrl+Shift+Z — Undo / redo",
                        "Ctrl+L — Toggle logarithmic spectrum scale",
                        "F11 — Toggle full-canvas view",
                        "F1 — Show this shortcut reference",
                    )
                ),
            )

        def _show_about(self) -> None:
            QMessageBox.information(
                self,
                "About FluxForge",
                (
                    "FluxForge HPGe Analysis\n\n"
                    "Cross-platform gamma spectroscopy, calibration, activity "
                    "analysis, "
                    "response processing, and spectrum unfolding."
                ),
            )

        def _toggle_log_scale(self, enabled: bool) -> None:
            if hasattr(self, "central_tabs") and hasattr(
                self.central_tabs, "set_log_scale"
            ):
                self.central_tabs.set_log_scale(bool(enabled))
            self.statusBar().showMessage(
                f"Canvas scale: {'log' if enabled else 'linear'}",
                3000,
            )

        def _toggle_peak_labels(self, visible: bool) -> None:
            if hasattr(self, "central_tabs") and hasattr(
                self.central_tabs, "set_peak_labels_visible"
            ):
                self.central_tabs.set_peak_labels_visible(bool(visible))
            self.statusBar().showMessage(
                f"Peak labels {'enabled' if visible else 'hidden'}",
                3000,
            )

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
            profile = state.theme_profile or "custom"
            self.progress.setFormat(f"{state.mode.value.title()} shell ready")
            self.renderer_label.setText(
                f"Renderer: PyQtGraph-first | Theme {state.theme} ({profile})"
            )
            if hasattr(self, "_pu_isotopics_action"):
                self._pu_isotopics_action.setEnabled(
                    state.mode is not GUIMode.SIMPLE
                    and self.analysis_workspace.spectrum() is not None
                )

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
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode is GUIMode.STANDARDS
                else None
            )
            label = self.library_manager.record_for_category(
                "gamma_identification",
                standard=standard,
            ).label
            self.library_label.setText(f"Library: {label}")

        def _on_workspace_state_changed(self, state) -> None:
            active_slot = next(
                (
                    slot
                    for slot in state.spectra
                    if slot.key == state.active_spectrum_key
                ),
                None,
            )
            if active_slot is not None:
                self.file_label.setText(
                    f"File: {active_slot.source_label or active_slot.spectrum.spectrum_id or active_slot.label}"
                )
            else:
                self.file_label.setText("File: none")
            has_spectrum = active_slot is not None
            for action in getattr(self, "_requires_spectrum_actions", ()):
                action.setEnabled(has_spectrum)
            if hasattr(self, "_pu_isotopics_action"):
                self._pu_isotopics_action.setEnabled(
                    has_spectrum and self.mode_manager.state.mode is not GUIMode.SIMPLE
                )
            has_example = self._workspace_contains_bundled_example()
            for action in getattr(self, "_requires_example_actions", ()):
                action.setEnabled(has_example)
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
                self.analysis_workspace.set_document(session.document)
                self._session_path = source.resolve()
                self._session_device_snapshot = deepcopy(session.device_snapshot)
                self._session_metadata = deepcopy(session.metadata)
                self._session_created_at = session.created_at
                self._session_recent_files = tuple(session.recent_files)
                self.undo_stack.clear()
                self.undo_stack.setClean()
                viewport = session.document.viewport_by_id("primary-spectrum")
                if viewport is not None and hasattr(
                    self.central_tabs, "apply_viewport_state"
                ):
                    self.central_tabs.apply_viewport_state(viewport)
                    self._log_scale_action.setChecked(viewport.log_y)
                    self._peak_labels_action.setChecked(viewport.labels_visible)
                if viewport is not None and viewport.selected_roi_id:
                    roi = session.document.roi_by_id(viewport.selected_roi_id)
                    if roi is not None:
                        self.selection_bus.publish(
                            SelectionState(
                                spectrum_id=roi.spectrum_id,
                                roi_id=roi.roi_id,
                                roi_bounds_keV=roi.signal_range,
                            )
                        )
                self.recent_files.record_many((source, *session.recent_files))
                self._document_dirty = False
                self.setWindowModified(False)
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
            if source.suffix.lower() == ".ffs":
                self._document_dirty = False
                self.setWindowModified(False)

        def _on_workspace_document_changed(self, document) -> None:
            viewport = document.viewport_by_id("primary-spectrum")
            log_y = viewport.log_y if viewport is not None else False
            labels_visible = viewport.labels_visible if viewport is not None else True
            for action, checked in (
                (self._log_scale_action, log_y),
                (self._peak_labels_action, labels_visible),
            ):
                action.blockSignals(True)
                action.setChecked(bool(checked))
                action.blockSignals(False)
            self._document_dirty = True
            self.setWindowModified(True)

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
            if (
                self._calibration_dialog is not None
                and self._calibration_dialog.isVisible()
            ):
                if advanced_tab is not None:
                    self._calibration_dialog.set_active_advanced_tab(advanced_tab)
                self._calibration_dialog.raise_()
                self._calibration_dialog.activateWindow()
                return

            current_spectrum = None
            if hasattr(self.central_tabs, "current_spectrum"):
                current_spectrum = self.central_tabs.current_spectrum()
            if current_spectrum is None:
                self.statusBar().showMessage(
                    "Load a spectrum before opening calibration.",
                    5000,
                )
                return

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
            if (
                self._unfolding_dialog is not None
                and self._unfolding_dialog.isVisible()
            ):
                self._unfolding_dialog.raise_()
                self._unfolding_dialog.activateWindow()
                return

            if not self._workspace_contains_bundled_example():
                self.statusBar().showMessage(
                    "Open the bundled example to inspect the current unfolding "
                    "workspace.",
                    5000,
                )
                return

            self._unfolding_dialog = UnfoldingWorkspaceDialog(
                mode_manager=self.mode_manager,
                parent=self,
            )
            self._unfolding_dialog.show()

        def _build_standards_context(self) -> StandardsEvaluationContext:
            spectrum = self.analysis_workspace.spectrum()
            if spectrum is None:
                raise RuntimeError("Load a spectrum before evaluating ASTM standards.")

            state = self.analysis_workspace.state
            latest = next(iter(self.qa_monitor.status_snapshot()), None)
            calibration = dict(spectrum.calibration or {})
            metadata = dict(spectrum.metadata or {})

            energy_coefficients = calibration.get("energy")
            calibration_order = None
            if isinstance(energy_coefficients, (list, tuple)):
                calibration_order = max(len(energy_coefficients) - 1, 0)

            def optional_float(*keys: str) -> float | None:
                for source in (calibration, metadata):
                    for key in keys:
                        value = source.get(key)
                        if value is None:
                            continue
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            continue
                return None

            net_counts: dict[str, float] = {}
            if state.peaks:
                primary = max(state.peaks, key=lambda peak: peak.net_counts)
                net_counts["primary"] = float(primary.net_counts)
            for peak in state.peaks:
                net_counts[peak.peak_id] = float(peak.net_counts)
                if peak.nuclide:
                    key = f"{peak.nuclide} {peak.energy_keV:.1f}"
                    net_counts[key] = float(peak.net_counts)

            efficiency_uncertainty = optional_float(
                "efficiency_uncertainty_pct",
                "efficiency_uncertainty_percent",
            )
            return StandardsEvaluationContext(
                calibration_order=calibration_order,
                calibration_rms_keV=optional_float(
                    "calibration_rms_keV",
                    "energy_rms_keV",
                ),
                max_residual_keV=optional_float(
                    "max_residual_keV",
                    "energy_max_residual_keV",
                ),
                efficiency_uncertainty_pct=efficiency_uncertainty,
                fwhm_at_413_keV=optional_float(
                    "fwhm_at_413_keV",
                    "fwhm_413_keV",
                ),
                qa_centroid_drift_keV=(latest.centroid_drift_keV if latest else None),
                qa_fwhm_degradation_pct=(
                    latest.fwhm_degradation_pct if latest else None
                ),
                before_calibration=(latest.last_check if latest else None),
                measured_at=spectrum.start_time,
                after_calibration=(latest.last_check if latest else None),
                net_counts=net_counts,
                line_observations={},
                extra={
                    "spectrum_id": spectrum.spectrum_id,
                    "detector_id": spectrum.detector_id,
                    "live_time_s": float(spectrum.live_time),
                },
            )

        def _build_report_context(self, template_name: str) -> dict[str, object]:
            state = self.analysis_workspace.state
            peak_rows = (
                "".join(
                    f"<tr><td>{peak.energy_keV:.3f}</td><td>{peak.nuclide or 'Unassigned'}</td><td>{peak.net_counts:.1f}</td></tr>"
                    for peak in state.peaks
                )
                or "<tr><td colspan='3'>No peaks</td></tr>"
            )
            peak_table = (
                "<table><tr><th>Energy</th><th>Nuclide</th><th>Net Counts</th></tr>"
                + peak_rows
                + "</table>"
            )
            activity_rows = (
                "".join(
                    f"<tr><td>{result.nuclide}</td><td>{result.line_energy_keV:.3f}</td><td>{result.activity_bq:.3f}</td><td>{result.uncertainty_bq:.3f}</td></tr>"
                    for result in state.activity_results
                )
                or "<tr><td colspan='4'>No activity results</td></tr>"
            )
            activity_table = (
                "<table><tr><th>Nuclide</th><th>Line</th><th>Activity</th><th>σ</th></tr>"
                + activity_rows
                + "</table>"
            )
            standards_rows = []
            if self.analysis_workspace.spectrum() is not None:
                context = self._build_standards_context()
                for key in (
                    "ASTM E181",
                    "ASTM E1297",
                    "ASTM E1218",
                    "ASTM C1232",
                    "ASTM C1030",
                ):
                    module = self.registries.standards_modules.get(key)
                    evaluation = module.evaluate(context)
                    standards_rows.append(
                        f"<tr><td>{module.display_name}</td>"
                        f"<td>{evaluation.overall_status}</td>"
                        f"<td>{evaluation.summary}</td></tr>"
                    )
            else:
                standards_rows.append(
                    "<tr><td colspan='3'>Load a spectrum to run standards "
                    "checks.</td></tr>"
                )
            astm_status_table = (
                "<table><tr><th>Standard</th><th>Status</th><th>Summary</th></tr>"
                + "".join(standards_rows)
                + "</table>"
            )
            qa_status_snapshot = (
                "<br/>".join(
                    f"{item.nuclide} {item.energy_keV:.2f} keV | drift {item.centroid_drift_keV:+.3f} keV | FWHM {item.fwhm_degradation_pct:+.2f}%"
                    for item in self.qa_monitor.status_snapshot()
                )
                or "No QA snapshot available."
            )
            provenance = (
                f"mode={self.mode_manager.state.mode.value}\n"
                f"standard={self.mode_manager.state.standard}\n"
                f"gamma_library={self.library_manager.resolved_state(standard=self.mode_manager.state.standard if self.mode_manager.state.mode is GUIMode.STANDARDS else None).gamma_identification_source_id}\n"
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
                    aggregate_csv = (
                        (panel.last_output_dir / "aggregate.csv").read_text(
                            encoding="utf-8"
                        )
                        if panel.last_output_dir
                        and (panel.last_output_dir / "aggregate.csv").exists()
                        else ""
                    )
            payload = {
                "title": "FluxForge Module 3 Report",
                "spectrum_image": "",
                "calibration_curve": "",
                "calibration_residuals": "",
                "efficiency_curve": "",
                "efficiency_residuals": "",
                "residuals_grid": "",
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
            for key in (
                "ASTM E181",
                "ASTM E1297",
                "ASTM E1218",
                "ASTM C1232",
                "ASTM C1030",
            ):
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
            if self.analysis_workspace.spectrum() is None:
                self.statusBar().showMessage(
                    "Load a spectrum before running an ASTM review.",
                    5000,
                )
                return
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
            if (
                self._qa_history_dialog is not None
                and self._qa_history_dialog.isVisible()
            ):
                self._qa_history_dialog.raise_()
                self._qa_history_dialog.activateWindow()
                return
            self._qa_history_dialog = QAHistoryDialog(self.qa_monitor, parent=self)
            self._qa_history_dialog.show()

        def _open_dashboard_tab(self) -> None:
            if hasattr(self, "central_tabs"):
                self.central_tabs.setCurrentIndex(1)

        def _focus_workspace_dock(self) -> None:
            self.left_dock.show()
            self.left_dock.raise_()

        def _focus_analysis_surface_dock(self) -> None:
            self.bottom_dock.show()
            self.bottom_dock.raise_()

        def _focus_inspector_dock(self) -> None:
            self.right_dock.show()
            self.right_dock.raise_()

        def _on_workflow_presets_changed(self, _presets, _active_name) -> None:
            self._refresh_workflow_controls()

        def _refresh_workflow_controls(self) -> None:
            if not hasattr(self, "workflow_combo"):
                return
            active_name = self.workflow_presets.active_workflow_name()
            self.workflow_combo.blockSignals(True)
            self.workflow_combo.clear()
            for preset in self.workflow_presets.available_workflows():
                suffix = " [built-in]" if preset.built_in else ""
                label = f"{preset.name}{suffix}"
                self.workflow_combo.addItem(label, preset.name)
                index = self.workflow_combo.count() - 1
                self.workflow_combo.setItemData(
                    index, preset.description, Qt.ToolTipRole
                )
            if active_name:
                active_index = self.workflow_combo.findData(active_name)
                if active_index >= 0:
                    self.workflow_combo.setCurrentIndex(active_index)
            elif self.workflow_combo.count() > 0:
                self.workflow_combo.setCurrentIndex(0)
            self.workflow_combo.blockSignals(False)
            selected_name = self._selected_workflow_name()
            selected = (
                self.workflow_presets.get_workflow(selected_name)
                if selected_name is not None
                else None
            )
            can_delete = bool(selected is not None and not selected.built_in)
            self.delete_workflow_button.setEnabled(can_delete)
            self.load_workflow_button.setEnabled(self.workflow_combo.count() > 0)

        def _selected_workflow_name(self) -> str | None:
            if not hasattr(self, "workflow_combo"):
                return None
            value = self.workflow_combo.currentData()
            text = str(value or "").strip()
            return text or None

        def _snapshot_current_workflow(self) -> dict[str, object]:
            state = self.analysis_workspace.state
            workspace_payload = {
                "peak_search_method": state.peak_search_method,
                "bayesian_source_id": state.bayesian_source_id,
                "ml_source_id": state.ml_source_id,
                "roi_background_method": state.roi_background_method,
                "background_mode": state.background_mode,
                "background_scale": float(state.background_scale),
                "background_visible": bool(state.background_visible),
            }
            payload = {
                "version": 2,
                "mode_state": self.mode_manager.describe(),
                "library_state": self.library_manager.describe(),
                "view_state": {
                    "log_scale": bool(self._log_scale_action.isChecked()),
                    "peak_labels": bool(self._peak_labels_action.isChecked()),
                },
                "central_state": (
                    self.central_tabs.workflow_state()
                    if hasattr(self.central_tabs, "workflow_state")
                    else {}
                ),
                "sidebar_state": (
                    self.left_dock.widget().workflow_state()
                    if hasattr(self.left_dock.widget(), "workflow_state")
                    else {}
                ),
                "bottom_state": (
                    self.bottom_dock.widget().workflow_state()
                    if hasattr(self.bottom_dock.widget(), "workflow_state")
                    else {}
                ),
                "workspace_state": workspace_payload,
            }
            return payload

        def _save_current_workflow_dialog(self) -> None:
            current_name = (
                self.workflow_presets.active_workflow_name() or "custom-workflow"
            )
            name, accepted = QInputDialog.getText(
                self,
                "Save Workflow",
                "Workflow name:",
                text=current_name,
            )
            if not accepted:
                return
            workflow_name = name.strip()
            if not workflow_name:
                self.statusBar().showMessage("Workflow name cannot be empty.", 4000)
                return
            try:
                self.workflow_presets.save_workflow(
                    workflow_name,
                    self._snapshot_current_workflow(),
                    description=(
                        "Saved from FluxForge at "
                        f"{datetime.now().isoformat(timespec='minutes')}"
                    ),
                )
            except Exception as exc:
                self.statusBar().showMessage(str(exc), 6000)
                return
            self._refresh_workflow_controls()
            self.statusBar().showMessage(
                f"Saved workflow preset: {workflow_name}",
                5000,
            )

        def _load_selected_workflow(self) -> None:
            workflow_name = self._selected_workflow_name()
            if not workflow_name:
                self.statusBar().showMessage("Select a workflow preset first.", 4000)
                return
            preset = self.workflow_presets.get_workflow(workflow_name)
            if preset is None:
                self.statusBar().showMessage(
                    f"Unknown workflow preset: {workflow_name}",
                    5000,
                )
                return
            self._apply_workflow_payload(preset.payload)
            self.workflow_presets.set_active_workflow(preset.name)
            self._refresh_workflow_controls()
            self.statusBar().showMessage(
                f"Loaded workflow preset: {preset.name}",
                5000,
            )

        def _delete_selected_workflow(self) -> None:
            workflow_name = self._selected_workflow_name()
            if not workflow_name:
                self.statusBar().showMessage("Select a workflow preset first.", 4000)
                return
            preset = self.workflow_presets.get_workflow(workflow_name)
            if preset is None:
                self.statusBar().showMessage(
                    f"Unknown workflow preset: {workflow_name}",
                    5000,
                )
                return
            if preset.built_in:
                self.statusBar().showMessage(
                    "Built-in workflow presets cannot be deleted.",
                    5000,
                )
                return
            self.workflow_presets.delete_workflow(workflow_name)
            self._refresh_workflow_controls()
            self.statusBar().showMessage(
                f"Deleted workflow preset: {workflow_name}",
                5000,
            )

        def _reset_analysis_workspace(self) -> None:
            self.qa_monitor.clear_demo_history()
            self.analysis_workspace.set_state(
                self._build_initial_workspace_state(include_example=False)
            )
            self.selection_bus.publish(SelectionState())

        def _apply_workflow_payload(self, payload: dict[str, object]) -> None:
            mode_payload, library_payload = (
                self.workflow_presets.extract_mode_and_library_state(payload)
            )
            if mode_payload:
                self.mode_manager.apply_state(mode_payload)
            if library_payload:
                self.library_manager.apply_state(library_payload)

            workspace_payload = payload.get("workspace_state")
            if isinstance(workspace_payload, dict):
                config_payload = {
                    key: workspace_payload.get(key)
                    for key in (
                        "peak_search_method",
                        "bayesian_source_id",
                        "ml_source_id",
                        "roi_background_method",
                        "background_mode",
                        "background_scale",
                        "background_visible",
                    )
                }
                if config_payload.get("peak_search_method") is not None:
                    self.analysis_workspace.set_peak_search_method(
                        str(config_payload["peak_search_method"])
                    )
                if config_payload.get("bayesian_source_id") is not None:
                    self.analysis_workspace.set_bayesian_source_id(
                        str(config_payload["bayesian_source_id"])
                    )
                if config_payload.get("ml_source_id") is not None:
                    self.analysis_workspace.set_ml_source_id(
                        str(config_payload["ml_source_id"])
                    )
                if config_payload.get("roi_background_method") is not None:
                    self.analysis_workspace.set_roi_background_method(
                        str(config_payload["roi_background_method"])
                    )
                self.analysis_workspace.set_background_config(
                    mode=(
                        str(config_payload["background_mode"])
                        if config_payload.get("background_mode") is not None
                        else None
                    ),
                    scale=(
                        float(config_payload["background_scale"])
                        if config_payload.get("background_scale") is not None
                        else None
                    ),
                    visible=(
                        bool(config_payload["background_visible"])
                        if config_payload.get("background_visible") is not None
                        else None
                    ),
                )
            view_payload = payload.get("view_state")
            if isinstance(view_payload, dict):
                if "log_scale" in view_payload:
                    self._log_scale_action.setChecked(bool(view_payload["log_scale"]))
                    self._toggle_log_scale(bool(view_payload["log_scale"]))
                if "peak_labels" in view_payload:
                    self._peak_labels_action.setChecked(
                        bool(view_payload["peak_labels"])
                    )
                    self._toggle_peak_labels(bool(view_payload["peak_labels"]))

            central_payload = payload.get("central_state")
            if isinstance(central_payload, dict) and hasattr(
                self.central_tabs,
                "apply_workflow_state",
            ):
                self.central_tabs.apply_workflow_state(central_payload)

            sidebar_payload = payload.get("sidebar_state")
            sidebar_widget = self.left_dock.widget()
            if isinstance(sidebar_payload, dict) and hasattr(
                sidebar_widget,
                "apply_workflow_state",
            ):
                sidebar_widget.apply_workflow_state(sidebar_payload)

            bottom_payload = payload.get("bottom_state")
            bottom_widget = self.bottom_dock.widget()
            if isinstance(bottom_payload, dict) and hasattr(
                bottom_widget,
                "apply_workflow_state",
            ):
                bottom_widget.apply_workflow_state(bottom_payload)

            self._refresh_analysis_workspace_derivatives()

        def _open_pu_isotopics_wizard(self) -> None:
            if self.mode_manager.state.mode is GUIMode.SIMPLE:
                return
            if (
                self._pu_isotopics_dialog is not None
                and self._pu_isotopics_dialog.isVisible()
            ):
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
            spectrum_id = self.analysis_workspace.document.active_spectrum_id
            workspace_spectrum = (
                self.analysis_workspace.document.spectrum_by_id(spectrum_id)
                if spectrum_id is not None
                else None
            )
            if spectrum_id is None or workspace_spectrum is None:
                return
            existing_profile = (
                self.analysis_workspace.document.detector_profile_by_id(
                    workspace_spectrum.detector_profile_id
                )
                if workspace_spectrum.detector_profile_id
                else None
            )
            profile_base = existing_profile or DetectorProfile(
                detector_profile_id=f"{spectrum_id}-detector-profile",
                detector_id=str(workspace_spectrum.spectrum.detector_id or ""),
            )
            prior_leaf = existing_profile
            if prior_leaf is None:
                prior_coefficients = tuple(
                    float(value)
                    for value in workspace_spectrum.spectrum.calibration.get(
                        "energy", ()
                    )
                )
                raw_deviation_pairs = workspace_spectrum.spectrum.calibration.get(
                    "deviation_pairs", ()
                )
                prior_deviation_pairs = tuple(
                    (
                        float(item.get("energy_keV", 0.0)),
                        float(item.get("correction_keV", 0.0)),
                    )
                    for item in raw_deviation_pairs
                    if isinstance(item, dict)
                )
                prior_leaf = replace(
                    profile_base,
                    energy_calibration=(
                        CalibrationModel(
                            model_key="legacy-polynomial",
                            coefficients=prior_coefficients,
                            deviation_pairs=prior_deviation_pairs,
                        )
                        if prior_coefficients
                        else None
                    ),
                )
            energy_model = CalibrationModel(
                model_key=f"polynomial-{energy_fit.order}",
                coefficients=tuple(float(value) for value in energy_fit.coefficients),
                deviation_pairs=tuple(
                    (float(pair.energy_keV), float(pair.correction_keV))
                    for pair in energy_fit.deviation_pairs
                ),
                provenance={
                    "chi_squared": float(energy_fit.chi_squared),
                    "reduced_chi_squared": float(energy_fit.reduced_chi_squared),
                    "rms_keV": float(energy_fit.rms_keV),
                },
            )
            fwhm_model = (
                CalibrationModel(
                    model_key=str(fwhm_fit.model),
                    coefficients=tuple(float(value) for value in fwhm_fit.coefficients),
                    provenance={
                        "chi_squared": float(fwhm_fit.chi_squared),
                        "reduced_chi_squared": float(fwhm_fit.reduced_chi_squared),
                        "rms_keV": float(fwhm_fit.rms_keV),
                    },
                )
                if fwhm_fit is not None
                else (
                    existing_profile.fwhm_calibration
                    if existing_profile is not None
                    else None
                )
            )
            profile = replace(
                profile_base,
                energy_calibration=energy_model,
                fwhm_calibration=fwhm_model,
            )
            command = ApplyCalibrationCommand(
                self.analysis_workspace,
                spectrum_id=spectrum_id,
                before=prior_leaf,
                after=profile,
            )
            if self.undo_stack is not None:
                self.undo_stack.push(command)
            else:
                command.redo()
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
            from fluxforge.core.analysis_workspace import (
                compute_cascade_sum_lines,
                extract_survey_points,
            )

            spectra = [
                (slot.key, slot.spectrum)
                for slot in self.analysis_workspace.state.spectra
            ]
            self.analysis_workspace.set_survey_points(extract_survey_points(spectra))
            self.analysis_workspace.set_cascade_sum_lines(
                compute_cascade_sum_lines(
                    self.analysis_workspace.state.pinned_nuclides,
                    source_id=self.library_manager.resolved_state(
                        standard=(
                            self.mode_manager.state.standard
                            if self.mode_manager.state.mode is GUIMode.STANDARDS
                            else None
                        )
                    ).gamma_identification_source_id,
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
