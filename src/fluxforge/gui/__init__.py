"""Next-generation GUI scaffolding exports."""

from fluxforge.gui.app import describe_gui_scaffold, launch_modern_gui, main
from fluxforge.gui.backends import (
    PYQTGRAPH_AVAILABLE,
    VISPY_AVAILABLE,
    PyQtGraphSpectrumCanvas,
    VispySpectrumCanvas,
    available_renderer_status,
    register_builtin_render_backends,
)
from fluxforge.gui.main_window import (
    DEFAULT_DOCK_ZONES,
    DockZone,
    FluxForgeMainWindow,
    MainWindowScaffold,
    modern_gui_unavailable_message,
)
from fluxforge.gui.dialogs import (
    CalibrationWorkspaceDialog,
    PuIsotopicsDialog,
    QAHistoryDialog,
    ReportExportDialog,
    StandardsReviewDialog,
    UnfoldingWorkspaceDialog,
)
from fluxforge.gui.library_manager import DataLibraryManager, DataLibraryState
from fluxforge.gui.mode_manager import GUIMode, ModeManager, ModeState
from fluxforge.gui.file_workflow import RecentFilesManager, normalize_dropped_paths
from fluxforge.gui.nuclide_search import NuclideSearchController
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind
from fluxforge.gui.workflow_presets import (
    DEFAULT_WORKFLOW_PRESETS,
    WorkflowPreset,
    WorkflowPresetManager,
)
from fluxforge.gui.spectrum_canvas import (
    HierarchicalSpectrumBuffer,
    HierarchicalSpectrumLevel,
    ReferenceLine,
    RendererCapabilities,
    SpectrumCanvas,
    SpectrumTrace,
)
from fluxforge.gui.widgets import MethodSelectorWidget

__all__ = [
    "DEFAULT_DOCK_ZONES",
    "CalibrationWorkspaceDialog",
    "CanvasIntent",
    "CanvasIntentKind",
    "DockZone",
    "FluxForgeMainWindow",
    "HierarchicalSpectrumBuffer",
    "HierarchicalSpectrumLevel",
    "GUIMode",
    "MainWindowScaffold",
    "MethodSelectorWidget",
    "ModeManager",
    "ModeState",
    "PuIsotopicsDialog",
    "QAHistoryDialog",
    "ReportExportDialog",
    "StandardsReviewDialog",
    "DataLibraryManager",
    "DataLibraryState",
    "DEFAULT_WORKFLOW_PRESETS",
    "PYQTGRAPH_AVAILABLE",
    "QT_AVAILABLE",
    "PyQtGraphSpectrumCanvas",
    "RecentFilesManager",
    "ReferenceLine",
    "RendererCapabilities",
    "SelectionBus",
    "SelectionState",
    "SpectrumCanvas",
    "SpectrumTrace",
    "WorkflowPreset",
    "WorkflowPresetManager",
    "UnfoldingWorkspaceDialog",
    "VISPY_AVAILABLE",
    "VispySpectrumCanvas",
    "available_renderer_status",
    "describe_gui_scaffold",
    "launch_modern_gui",
    "main",
    "modern_gui_unavailable_message",
    "normalize_dropped_paths",
    "NuclideSearchController",
    "register_builtin_render_backends",
]
