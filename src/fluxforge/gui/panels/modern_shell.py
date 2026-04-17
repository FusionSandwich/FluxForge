"""Modern panels for the next-generation GUI shell."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from fluxforge.analysis.optimization_difom import (
    build_difom_terms_from_activity_results,
    evaluate_difom,
)
from fluxforge.analysis.optimization_fim import evaluate_fim
from fluxforge.analysis.optimization_mwdcs import (
    build_mwdcs_candidate_from_activity_results,
    evaluate_mwdcs,
)
from fluxforge.analysis.optimization_bassd import (
    build_bassd_candidate_from_activity_results,
    evaluate_bassd,
)
from fluxforge.analysis.optimization_stbdmr import (
    build_stbdmr_candidate_from_activity_results,
    evaluate_stbdmr,
)
from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.batch_analysis import (
    BatchAnalysisJob,
    run_batch_analysis_queue,
    write_batch_outputs,
)
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_dead_time_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.core.analysis_workspace import (
    ActivityCalculationResult,
    PeakCandidate,
    analyze_roi_region,
    apply_ml_peak_predictions,
    bayesian_match_peak_candidates,
    compute_roi_statistics,
    compute_cascade_sum_lines,
    detect_peak_candidates,
    estimate_spectral_phenomena,
    extract_survey_points,
    register_builtin_peak_search_methods,
    register_builtin_roi_background_methods,
    register_builtin_nuclide_id_engines,
    background_adjusted_spectrum,
    subtract_background_counts,
)
from fluxforge.core.activity_review import (
    ActivityReviewResult,
    review_spectrum_activation,
)
from fluxforge.core.inventory_timeline import (
    DEFAULT_DECAY_SOURCE_ID,
    build_inventory_state_from_activity_results,
    build_time_grid,
    compute_inventory_time_evolution,
)
from fluxforge.data.nuclear_data_sources import list_nuclear_data_sources_by_capability
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, pyqtgraph_backend_status
from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.nuclide_search import GammaLineMatchResult, NuclideSearchController
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController, SpectrumSlot, WorkspaceStateCommand
from fluxforge.gui.panels.modern_shell_shared import (
    current_tab_label,
    set_combo_data,
    set_tab_label,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import ReferenceLine, SpectrumTrace
from fluxforge.gui.widgets.method_selector import MethodSelectorWidget
from fluxforge.io.spe import GammaSpectrum
from fluxforge.plots.activation import plot_decay_curves
from fluxforge.plugins import bootstrap_builtin_registries
from fluxforge.standards import (
    QAMonitor,
    StandardsEvaluationContext,
    register_builtin_standards_modules,
)

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    if PYQTGRAPH_AVAILABLE:
        import pyqtgraph as pg

    from fluxforge.gui.backends import PyQtGraphSpectrumCanvas
    from fluxforge.gui.panels.phase6 import (
        MaskingReviewPanel,
        OptimizationWorkspacePanel,
        SecondIrradiationPlannerPanel,
    )
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QUndoStack,
        QVBoxLayout,
        QWidget,
        Qt,
    )


MODERN_LOG_LINES = (
    "Qt shell initialized",
    "Renderer strategy: PyQtGraph first, Vispy additive",
    "Mode-aware workflow locking ready",
    "Tk GUI demoted to explicit legacy fallback",
)

_ACTIVITY_UNIT_FACTORS = {
    "Bq": 1.0,
    "kBq": 1.0e3,
    "MBq": 1.0e6,
    "GBq": 1.0e9,
    "uCi": 3.7e4,
    "mCi": 3.7e7,
    "Ci": 3.7e10,
}


def _activity_unit_factor(unit: str) -> float:
    return float(_ACTIVITY_UNIT_FACTORS.get(str(unit), 1.0))


def _scale_activity_points(
    points: tuple[tuple[float, float, float], ...],
    unit: str,
) -> tuple[tuple[float, float, float], ...]:
    factor = _activity_unit_factor(unit)
    return tuple(
        (float(time_s), float(value) / factor, float(uncertainty) / factor)
        for time_s, value, uncertainty in points
    )


def _format_activity_value(
    value_bq: float,
    unit: str,
    *,
    precision: str = ".6g",
) -> str:
    scaled_value = float(value_bq) / _activity_unit_factor(unit)
    return f"{scaled_value:{precision}} {unit}"


def _selection_summary(state: SelectionState) -> str:
    """Format the current cross-panel selection state."""

    fragments = []
    if state.peak_energy_keV is not None:
        fragments.append(f"Peak {state.peak_energy_keV:.3f} keV")
    if state.nuclide:
        fragments.append(state.nuclide)
    if state.roi_bounds_keV:
        fragments.append(
            f"ROI {state.roi_bounds_keV[0]:.1f}-{state.roi_bounds_keV[1]:.1f} keV"
        )
    if state.reference_lines_keV:
        fragments.append(f"{len(state.reference_lines_keV)} ref lines")
    if state.annotation_lines:
        fragments.append(f"{len(state.annotation_lines)} guides")
    return " | ".join(fragments) if fragments else "No active selection"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _demo_counts() -> tuple[float, ...]:
    """Synthetic spectrum used to keep the shell visually alive before I/O lands."""

    counts = []
    for channel in range(2048):
        background = 18.0 + (channel / 96.0)
        peak_a = 1180.0 / (1.0 + ((channel - 662.0) / 10.5) ** 2)
        peak_b = 780.0 / (1.0 + ((channel - 1173.0) / 14.0) ** 2)
        peak_c = 640.0 / (1.0 + ((channel - 1332.0) / 16.0) ** 2)
        counts.append(background + peak_a + peak_b + peak_c)
    return tuple(counts)


def build_demo_spectrum() -> GammaSpectrum:
    """Return the native demo spectrum used by the redesigned shell."""

    counts = _demo_counts()
    channels = tuple(float(index) for index in range(len(counts)))
    start_time = datetime(2026, 3, 30, 11, 0)
    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        channels=np.asarray(channels, dtype=float),
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=321.0,
        start_time=start_time,
        spectrum_id="demo_hpge_workspace",
        detector_id="demo-hpge",
        gps={"latitude": 43.0731, "longitude": -89.4012},
        metadata={
            "source": "analysis-demo",
            "gps": {"latitude": 43.0731, "longitude": -89.4012},
            "input_count_rate_cps": 47500.0,
        },
    )


def build_demo_background_spectrum() -> GammaSpectrum:
    """Return the background companion spectrum for the analysis demo shell."""

    counts = np.asarray(_demo_counts(), dtype=float) * 0.16
    channels = np.arange(len(counts), dtype=float)
    start_time = datetime(2026, 3, 30, 10, 0)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=309.0,
        start_time=start_time,
        spectrum_id="demo_hpge_background",
        detector_id="demo-hpge",
        gps={"latitude": 43.0736, "longitude": -89.4019},
        metadata={
            "source": "analysis-demo-background",
            "gps": {"latitude": 43.0736, "longitude": -89.4019},
            "input_count_rate_cps": 13800.0,
        },
    )


def build_demo_overlay_spectrum() -> GammaSpectrum:
    """Return the secondary overlay companion spectrum for the analysis demo shell."""

    counts = np.asarray(_demo_counts(), dtype=float) * 0.62
    counts[540:620] *= 1.18
    channels = np.arange(len(counts), dtype=float)
    start_time = datetime(2026, 3, 30, 10, 30)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=316.0,
        start_time=start_time,
        spectrum_id="demo_hpge_overlay",
        detector_id="demo-hpge",
        gps={"latitude": 43.0742, "longitude": -89.4024},
        metadata={
            "source": "analysis-demo-overlay",
            "gps": {"latitude": 43.0742, "longitude": -89.4024},
            "input_count_rate_cps": 29800.0,
        },
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    def _card(title: str, body: str, accent: str | None = None) -> QFrame:
        frame = QFrame()
        frame.setObjectName("HeroCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title_label = QLabel(title, frame)
        title_label.setObjectName("HeroCardTitle")
        layout.addWidget(title_label)

        body_label = QLabel(body, frame)
        body_label.setWordWrap(True)
        body_label.setObjectName("HeroCardBody")
        layout.addWidget(body_label)

        if accent:
            accent_label = QLabel(accent, frame)
            accent_label.setObjectName("HeroCardAccent")
            accent_label.setWordWrap(True)
            layout.addWidget(accent_label)

        layout.addStretch(1)
        return frame


    def _peak_status_dot(status: str) -> str:
        if status == "matched":
            return "●"
        if status == "manual":
            return "◆"
        if status == "review":
            return "◐"
        return "○"


    class PeakTablePanel(QWidget):
        """Peak table, review workflow, and Bayesian matching surface."""

        HEADERS = ("Status", "Peak keV", "ROI keV", "Candidates", "Nuclide", "Tags")

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager,
            undo_stack: QUndoStack | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self.undo_stack = undo_stack
            self.nuclide_controller = NuclideSearchController(
                self.selection_bus,
                library_manager=self.library_manager,
            )
            self.registries = bootstrap_builtin_registries()
            register_builtin_peak_search_methods(self.registries)
            register_builtin_nuclide_id_engines(self.registries)
            self._current_match_results: tuple[GammaLineMatchResult, ...] = ()
            self._selection_sync_guard = False

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Detected peaks, manual tags, and Bayesian library matches stay in one "
                    "undoable table synchronized through the modern workspace state."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            action_row = QHBoxLayout()
            action_row.setSpacing(8)
            self.auto_find_button = QPushButton("Auto Find Peaks", self)
            self.auto_find_button.setObjectName("AutoFindPeaksButton")
            self.auto_find_button.clicked.connect(self.run_auto_peak_search)
            action_row.addWidget(self.auto_find_button)

            self.peak_search_selector = MethodSelectorWidget(
                self.registries.peak_search_methods,
                self.mode_manager,
                title="Peak Search",
                parent=self,
            )
            self.peak_search_selector.setObjectName("PeakSearchMethodSelector")
            current_search_method = self.workspace_controller.state.peak_search_method
            if current_search_method in self.registries.peak_search_methods.keys():
                self.peak_search_selector.set_current_key(current_search_method)
            self.peak_search_selector.combo.currentIndexChanged.connect(
                self._peak_search_method_changed
            )
            action_row.addWidget(self.peak_search_selector, 1)

            self.match_button = QPushButton("Bayesian Match", self)
            self.match_button.setObjectName("BayesianMatchPeaksButton")
            self.match_button.clicked.connect(self.run_bayesian_match)
            action_row.addWidget(self.match_button)

            self.ml_button = QPushButton("ML Peak Analysis", self)
            self.ml_button.setObjectName("MlPeakAnalysisButton")
            self.ml_button.clicked.connect(self.run_ml_peak_analysis)
            action_row.addWidget(self.ml_button)

            self.pin_button = QPushButton("Pin Selected Nuclide", self)
            self.pin_button.clicked.connect(self._pin_selected_nuclide)
            action_row.addWidget(self.pin_button)

            self.tag_button = QPushButton("Tag Selected Peak", self)
            self.tag_button.clicked.connect(self._tag_selected_peak)
            action_row.addWidget(self.tag_button)

            self.clear_button = QPushButton("Clear Peaks", self)
            self.clear_button.clicked.connect(self._clear_peaks)
            action_row.addWidget(self.clear_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self.table = QTableWidget(0, len(self.HEADERS), self)
            self.table.setObjectName("PeakTableWidget")
            self.table.setHorizontalHeaderLabels(self.HEADERS)
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.itemSelectionChanged.connect(self._publish_selected_peak)
            peak_header = self.table.horizontalHeader()
            peak_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            peak_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            peak_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            peak_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            layout.addWidget(self.table, 1)

            self.summary = QLabel("No peaks have been added yet.", self)
            self.summary.setObjectName("PanelBody")
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)

            self.ml_summary = QLabel("ML proposals have not been generated yet.", self)
            self.ml_summary.setObjectName("PanelBody")
            self.ml_summary.setWordWrap(True)
            layout.addWidget(self.ml_summary)

            layout.addWidget(self._build_identification_sources_panel())
            layout.addWidget(self._build_peak_id_panel())

            self.workspace_controller.subscribe(self._sync_state)
            self.selection_bus.subscribe(self._sync_from_selection_bus)
            self.library_manager.subscribe(lambda _state: self._sync_identification_sources())
            self.mode_manager.subscribe(lambda _state: self._sync_identification_sources())
            self._sync_identification_sources()
            self._sync_state(self.workspace_controller.state)

        def _analysis_standard(self) -> str | None:
            state = self.mode_manager.state
            if state.mode is GUIMode.STANDARDS:
                return state.standard
            return None

        def _build_identification_sources_panel(self) -> QWidget:
            group = QGroupBox("Identification Databases", self)
            group.setObjectName("PeakIdentificationSourcesPanel")
            layout = QGridLayout(group)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setHorizontalSpacing(8)
            layout.setVerticalSpacing(8)

            layout.addWidget(QLabel("Bayesian matcher", group), 0, 0)
            self.bayesian_source_combo = QComboBox(group)
            self.bayesian_source_combo.setObjectName("BayesianLibraryCombo")
            self.bayesian_source_combo.currentIndexChanged.connect(
                self._bayesian_source_changed
            )
            layout.addWidget(self.bayesian_source_combo, 0, 1)

            layout.addWidget(QLabel("ML proposals", group), 1, 0)
            self.ml_source_combo = QComboBox(group)
            self.ml_source_combo.setObjectName("MlPeakLibraryCombo")
            self.ml_source_combo.currentIndexChanged.connect(self._ml_source_changed)
            layout.addWidget(self.ml_source_combo, 1, 1)

            self.id_source_summary = QLabel("", group)
            self.id_source_summary.setObjectName("PanelBody")
            self.id_source_summary.setWordWrap(True)
            layout.addWidget(self.id_source_summary, 2, 0, 1, 2)
            return group

        def _sync_identification_sources(self) -> None:
            standard = self._analysis_standard()
            resolved_state = self.library_manager.resolved_state(standard=standard)
            self.nuclide_controller.set_source(
                resolved_state.gamma_identification_source_id,
                custom_path=resolved_state.custom_gamma_path,
            )
            records = self.library_manager.available_sources(
                "gamma_identification",
                standard=standard,
            )
            locked = self.library_manager.locked_source_for_category(
                "gamma_identification",
                standard=standard,
            )
            self._populate_identification_source_combo(
                self.bayesian_source_combo,
                records,
                self.workspace_controller.state.bayesian_source_id,
            )
            self._populate_identification_source_combo(
                self.ml_source_combo,
                records,
                self.workspace_controller.state.ml_source_id,
            )
            combo_enabled = locked is None and len(records) > 1
            self.bayesian_source_combo.setEnabled(combo_enabled)
            self.ml_source_combo.setEnabled(combo_enabled)
            if locked is not None:
                locked_record = self.library_manager.record_for_category(
                    "gamma_identification",
                    standard=standard,
                )
                self.id_source_summary.setText(
                    f"Standards mode locks Bayesian and ML identification to {locked_record.label}."
                )
            else:
                bayesian_label = self.bayesian_source_combo.currentText() or "Unselected"
                ml_label = self.ml_source_combo.currentText() or "Unselected"
                self.id_source_summary.setText(
                    f"Bayesian DB: {bayesian_label} | ML DB: {ml_label}"
                )

        def _populate_identification_source_combo(
            self,
            combo: QComboBox,
            records,
            preferred_source_id: str,
        ) -> None:
            combo.blockSignals(True)
            current = combo.currentData()
            combo.clear()
            for record in records:
                combo.addItem(record.label, record.source_id)
            target = preferred_source_id or current
            if target is not None:
                index = combo.findData(target)
                if index >= 0:
                    combo.setCurrentIndex(index)
            if combo.currentIndex() < 0 and combo.count() > 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        def _bayesian_source_changed(self) -> None:
            source_id = self.bayesian_source_combo.currentData()
            if source_id:
                self.workspace_controller.set_bayesian_source_id(str(source_id))
                self._sync_identification_sources()

        def _ml_source_changed(self) -> None:
            source_id = self.ml_source_combo.currentData()
            if source_id:
                self.workspace_controller.set_ml_source_id(str(source_id))
                self._sync_identification_sources()

        def _build_peak_id_panel(self) -> QWidget:
            group = QGroupBox("Peak ID Browser", self)
            group.setObjectName("PeakIdBrowserPanel")
            layout = QVBoxLayout(group)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Start from the selected peak centroid or type an energy, browse isotope "
                    "lines from the active library within a default ±2 keV window, and keep "
                    "manual peak assignments editable after Bayesian matching."
                ),
                group,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            controls = QGridLayout()
            controls.setHorizontalSpacing(8)
            controls.setVerticalSpacing(8)

            controls.addWidget(QLabel("Centroid (keV)", group), 0, 0)
            self.peak_id_energy = QDoubleSpinBox(group)
            self.peak_id_energy.setObjectName("PeakIdCentroidSpin")
            self.peak_id_energy.setDecimals(3)
            self.peak_id_energy.setRange(0.0, 10000.0)
            self.peak_id_energy.setSingleStep(0.25)
            controls.addWidget(self.peak_id_energy, 0, 1)

            controls.addWidget(QLabel("Window (±keV)", group), 0, 2)
            self.peak_id_tolerance = QDoubleSpinBox(group)
            self.peak_id_tolerance.setObjectName("PeakIdToleranceSpin")
            self.peak_id_tolerance.setDecimals(2)
            self.peak_id_tolerance.setRange(0.1, 250.0)
            self.peak_id_tolerance.setSingleStep(0.25)
            self.peak_id_tolerance.setValue(2.0)
            controls.addWidget(self.peak_id_tolerance, 0, 3)

            controls.addWidget(QLabel("Element / isotope filter", group), 1, 0)
            self.peak_id_filter = QLineEdit(group)
            self.peak_id_filter.setObjectName("PeakIdFilterInput")
            self.peak_id_filter.setPlaceholderText("Optional filter, e.g. Cs or Co-60")
            controls.addWidget(self.peak_id_filter, 1, 1, 1, 3)
            layout.addLayout(controls)

            action_row = QHBoxLayout()
            self.use_selected_peak_button = QPushButton("Use Selected Peak", group)
            self.use_selected_peak_button.setObjectName("PeakIdUseSelectedPeakButton")
            self.use_selected_peak_button.clicked.connect(self._use_selected_peak_centroid)
            action_row.addWidget(self.use_selected_peak_button)

            self.assign_isotope_button = QPushButton("Assign Selected Isotope", group)
            self.assign_isotope_button.setObjectName("PeakIdAssignSelectedIsotopeButton")
            self.assign_isotope_button.clicked.connect(self._assign_selected_isotope)
            action_row.addWidget(self.assign_isotope_button)

            self.clear_assignment_button = QPushButton("Clear Peak ID", group)
            self.clear_assignment_button.setObjectName("PeakIdClearAssignmentButton")
            self.clear_assignment_button.clicked.connect(self._clear_selected_peak_assignment)
            action_row.addWidget(self.clear_assignment_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            layout.addWidget(QLabel("Library matches", group))
            self.peak_id_matches = QListWidget(group)
            self.peak_id_matches.setObjectName("PeakIdMatchesList")
            layout.addWidget(self.peak_id_matches, 1)

            layout.addWidget(QLabel("Estimated gamma-spectroscopy phenomena", group))
            self.peak_id_phenomena = QListWidget(group)
            self.peak_id_phenomena.setObjectName("PeakIdPhenomenaList")
            layout.addWidget(self.peak_id_phenomena, 1)

            self.peak_id_summary = QLabel("Select a peak or type a centroid to browse isotope lines.", group)
            self.peak_id_summary.setObjectName("PanelBody")
            self.peak_id_summary.setWordWrap(True)
            layout.addWidget(self.peak_id_summary)

            self.peak_id_energy.valueChanged.connect(self._refresh_peak_id_matches)
            self.peak_id_tolerance.valueChanged.connect(self._refresh_peak_id_matches)
            self.peak_id_filter.textChanged.connect(self._refresh_peak_id_matches)
            self.peak_id_matches.itemSelectionChanged.connect(self._match_selection_changed)
            return group

        def run_auto_peak_search(self) -> None:
            spectrum = self.workspace_controller.spectrum()
            if spectrum is None:
                self.summary.setText("No active spectrum is available for peak search.")
                return
            method = self.peak_search_selector.current_key() or self.workspace_controller.state.peak_search_method
            peaks = detect_peak_candidates(spectrum, method=method)
            dialog = AutoPeakReviewDialog(peaks, parent=self)
            if dialog.exec() != QDialog.Accepted:
                return
            accepted = dialog.accepted_peaks()
            self._commit_state_change(
                "Auto find peaks",
                self.workspace_controller.state,
                self.workspace_controller.state.__class__(
                    **{
                        **self.workspace_controller.state.__dict__,
                        "peaks": tuple(accepted),
                        "selected_peak_id": accepted[0].peak_id if accepted else None,
                        "peak_search_method": method,
                    }
                ),
            )

        def _peak_search_method_changed(self) -> None:
            method = self.peak_search_selector.current_key()
            if method:
                self.workspace_controller.set_peak_search_method(str(method))

        def run_bayesian_match(self) -> None:
            state = self.workspace_controller.state
            if not state.peaks:
                self.summary.setText("Run peak search before Bayesian matching.")
                return
            source_id = str(
                self.bayesian_source_combo.currentData() or state.bayesian_source_id
            )
            matched = bayesian_match_peak_candidates(
                state.peaks,
                source_id=source_id,
                custom_path=self.library_manager.state.custom_gamma_path,
            )
            self._commit_state_change(
                "Bayesian match peaks",
                state,
                state.__class__(
                    **{
                        **state.__dict__,
                        "peaks": tuple(matched),
                    }
                ),
            )

        def run_ml_peak_analysis(self) -> None:
            state = self.workspace_controller.state
            peaks = state.peaks
            if not peaks:
                self.run_auto_peak_search()
                peaks = self.workspace_controller.state.peaks
            if not peaks:
                self.ml_summary.setText(
                    "ML peak proposals require at least one detected peak."
                )
                return
            engine = self.registries.nuclide_id_engines.get("ml_peak_onnx")
            source_id = str(self.ml_source_combo.currentData() or state.ml_source_id)
            predictions = engine.analyze_peaks(
                peaks,
                source_id=source_id,
                custom_path=self.library_manager.state.custom_gamma_path,
            )
            updated_peaks = apply_ml_peak_predictions(peaks, predictions)
            self._commit_state_change(
                "ML peak proposals",
                self.workspace_controller.state,
                self.workspace_controller.state.__class__(
                    **{
                        **self.workspace_controller.state.__dict__,
                        "peaks": tuple(updated_peaks),
                    }
                ),
            )
            if predictions:
                lead = predictions[0]
                self.ml_summary.setText(
                    f"ML lead: {lead.predicted_nuclide} at {lead.predicted_line_keV:.3f} keV "
                    f"(confidence {lead.confidence:.2f}, uncertainty {lead.uncertainty_keV:.2f} keV, backend {lead.backend})."
                )
            else:
                self.ml_summary.setText(
                    "ML peak analysis finished without a confident library proposal."
                )
            self._sync_identification_sources()

        def _pin_selected_nuclide(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None or not peak.nuclide:
                return
            state = self.workspace_controller.state
            pinned = list(state.pinned_nuclides)
            if peak.nuclide not in pinned:
                pinned.append(peak.nuclide)
            standard = self._analysis_standard()
            resolved_state = self.library_manager.resolved_state(standard=standard)
            cascade_sum_lines_keV = compute_cascade_sum_lines(
                pinned,
                source_id=resolved_state.gamma_identification_source_id,
                custom_path=resolved_state.custom_gamma_path,
            )
            self._commit_state_change(
                "Pin nuclide",
                state,
                state.__class__(
                    **{
                        **state.__dict__,
                        "pinned_nuclides": tuple(pinned),
                        "cascade_sum_lines_keV": cascade_sum_lines_keV,
                    }
                ),
            )

        def _tag_selected_peak(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None:
                return
            tag, accepted = QInputDialog.getText(
                self,
                "Tag Peak",
                "Enter a tag for the selected peak:",
            )
            if not accepted:
                return
            text = tag.strip()
            if not text:
                return
            tags = tuple(dict.fromkeys((*peak.tags, text)))
            updated_peak = PeakCandidate(
                peak_id=peak.peak_id,
                channel=peak.channel,
                energy_keV=peak.energy_keV,
                significance=peak.significance,
                roi_bounds_keV=peak.roi_bounds_keV,
                net_counts=peak.net_counts,
                fit_quality=peak.fit_quality,
                status=peak.status,
                nuclide=peak.nuclide,
                candidate_nuclides=peak.candidate_nuclides,
                reference_lines_keV=peak.reference_lines_keV,
                tags=tags,
                normalized_residuals=peak.normalized_residuals,
                residual_channels=peak.residual_channels,
            )
            state = self.workspace_controller.state
            peaks = [
                updated_peak if item.peak_id == updated_peak.peak_id else item
                for item in state.peaks
            ]
            self._commit_state_change(
                "Tag peak",
                state,
                state.__class__(**{**state.__dict__, "peaks": tuple(peaks)}),
            )

        def _use_selected_peak_centroid(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None:
                return
            self.peak_id_energy.blockSignals(True)
            self.peak_id_energy.setValue(float(peak.energy_keV))
            self.peak_id_energy.blockSignals(False)
            self._refresh_peak_id_matches()

        def _selected_match(self) -> GammaLineMatchResult | None:
            row = self.peak_id_matches.currentRow()
            if row < 0 or row >= len(self._current_match_results):
                return None
            return self._current_match_results[row]

        def _peak_annotations_for_match(
            self,
            match: GammaLineMatchResult,
        ) -> tuple[ReferenceLine, ...]:
            annotations = [
                ReferenceLine(
                    energy_keV=float(match.line_energy_keV),
                    label=f"{match.display_name} photopeak",
                    color="#72d6ff",
                )
            ]
            for phenomenon in estimate_spectral_phenomena(match.line_energy_keV):
                annotations.append(
                    ReferenceLine(
                        energy_keV=float(phenomenon.energy_keV),
                        label=phenomenon.label,
                        color=phenomenon.color,
                    )
                )
            return tuple(annotations)

        def _refresh_peak_id_matches(self) -> None:
            energy_keV = float(self.peak_id_energy.value())
            tolerance_keV = float(self.peak_id_tolerance.value())
            query = self.peak_id_filter.text().strip()
            if energy_keV <= 0.0:
                self._current_match_results = ()
                self.peak_id_matches.clear()
                self.peak_id_phenomena.clear()
                self.peak_id_summary.setText(
                    "Select a peak or type a centroid to browse isotope lines."
                )
                return

            matches = tuple(
                self.nuclide_controller.line_matches_for_energy(
                    energy_keV,
                    tolerance_keV=tolerance_keV,
                    query=query,
                )
            )
            self._current_match_results = matches
            self.peak_id_matches.blockSignals(True)
            self.peak_id_matches.clear()
            for match in matches:
                item = QListWidgetItem(
                    (
                        f"{match.display_name} · {match.line_energy_keV:.3f} keV "
                        f"(Δ {match.delta_keV:+.3f} keV, I={match.intensity:.3f})"
                    ),
                    self.peak_id_matches,
                )
                item.setData(Qt.UserRole, match.nuclide)
            self.peak_id_matches.blockSignals(False)
            if matches:
                self.peak_id_matches.setCurrentRow(0)
                self.peak_id_summary.setText(
                    (
                        f"{len(matches)} isotope lines found within ±{tolerance_keV:.2f} keV "
                        f"using {self.nuclide_controller.source_label()}."
                    )
                )
                self._match_selection_changed()
            else:
                self.peak_id_phenomena.clear()
                self.peak_id_summary.setText(
                    (
                        f"No isotope lines found within ±{tolerance_keV:.2f} keV using "
                        f"{self.nuclide_controller.source_label()}."
                    )
                )
                peak = self.workspace_controller.selected_peak()
                self.selection_bus.publish(
                    SelectionState(
                        peak_energy_keV=peak.energy_keV if peak is not None else energy_keV,
                        roi_bounds_keV=peak.roi_bounds_keV if peak is not None else None,
                        nuclide=peak.nuclide if peak is not None else None,
                        reference_lines_keV=peak.reference_lines_keV if peak is not None else (),
                    )
                )

        def _match_selection_changed(self) -> None:
            match = self._selected_match()
            peak = self.workspace_controller.selected_peak()
            self.peak_id_phenomena.clear()
            if match is None:
                return
            for phenomenon in estimate_spectral_phenomena(match.line_energy_keV):
                QListWidgetItem(
                    f"{phenomenon.label} · {phenomenon.energy_keV:.3f} keV",
                    self.peak_id_phenomena,
                )
            reference_lines = self.nuclide_controller.reference_lines_for_nuclide(match.nuclide)
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=peak.energy_keV if peak is not None else float(self.peak_id_energy.value()),
                    roi_bounds_keV=peak.roi_bounds_keV if peak is not None else None,
                    nuclide=match.nuclide,
                    reference_lines_keV=reference_lines,
                    annotation_lines=self._peak_annotations_for_match(match),
                )
            )

        def _assign_selected_isotope(self) -> None:
            peak = self.workspace_controller.selected_peak()
            match = self._selected_match()
            if peak is None or match is None:
                return
            reference_lines = self.nuclide_controller.reference_lines_for_nuclide(match.nuclide)
            updated_peak = replace(
                peak,
                status="manual",
                nuclide=match.nuclide,
                candidate_nuclides=tuple(
                    dict.fromkeys((match.nuclide, *peak.candidate_nuclides))
                ),
                reference_lines_keV=reference_lines,
            )
            state = self.workspace_controller.state
            peaks = [
                updated_peak if item.peak_id == updated_peak.peak_id else item
                for item in state.peaks
            ]
            self._commit_state_change(
                "Assign peak isotope",
                state,
                state.__class__(**{**state.__dict__, "peaks": tuple(peaks)}),
            )
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=updated_peak.energy_keV,
                    roi_bounds_keV=updated_peak.roi_bounds_keV,
                    nuclide=updated_peak.nuclide,
                    reference_lines_keV=updated_peak.reference_lines_keV,
                    annotation_lines=self._peak_annotations_for_match(match),
                )
            )

        def _clear_selected_peak_assignment(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None:
                return
            updated_peak = replace(
                peak,
                status="candidate" if peak.candidate_nuclides else "review",
                nuclide=None,
                reference_lines_keV=(),
            )
            state = self.workspace_controller.state
            peaks = [
                updated_peak if item.peak_id == updated_peak.peak_id else item
                for item in state.peaks
            ]
            self._commit_state_change(
                "Clear peak isotope",
                state,
                state.__class__(**{**state.__dict__, "peaks": tuple(peaks)}),
            )
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=updated_peak.energy_keV,
                    roi_bounds_keV=updated_peak.roi_bounds_keV,
                    nuclide=None,
                    reference_lines_keV=(),
                )
            )

        def _clear_peaks(self) -> None:
            state = self.workspace_controller.state
            self._commit_state_change(
                "Clear peaks",
                state,
                state.__class__(**{**state.__dict__, "peaks": (), "selected_peak_id": None}),
            )

        def _commit_state_change(self, description: str, before, after) -> None:
            if self.undo_stack is not None:
                self.undo_stack.push(
                    WorkspaceStateCommand(
                        self.workspace_controller,
                        description=description,
                        before=before,
                        after=after,
                    )
                )
                return
            self.workspace_controller.set_state(after)

        def _sync_state(self, state) -> None:
            if self.peak_search_selector.current_key() != state.peak_search_method:
                self.peak_search_selector.combo.blockSignals(True)
                self.peak_search_selector.set_current_key(state.peak_search_method)
                self.peak_search_selector.combo.blockSignals(False)
                self.peak_search_selector._sync_badge()
            if self.bayesian_source_combo.currentData() != state.bayesian_source_id:
                index = self.bayesian_source_combo.findData(state.bayesian_source_id)
                if index >= 0:
                    self.bayesian_source_combo.blockSignals(True)
                    self.bayesian_source_combo.setCurrentIndex(index)
                    self.bayesian_source_combo.blockSignals(False)
            if self.ml_source_combo.currentData() != state.ml_source_id:
                index = self.ml_source_combo.findData(state.ml_source_id)
                if index >= 0:
                    self.ml_source_combo.blockSignals(True)
                    self.ml_source_combo.setCurrentIndex(index)
                    self.ml_source_combo.blockSignals(False)
            self._sync_identification_sources()

            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for row, peak in enumerate(state.peaks):
                self.table.insertRow(row)
                for column, text in enumerate(
                    (
                        _peak_status_dot(peak.status),
                        f"{peak.energy_keV:.3f}",
                        f"{peak.roi_bounds_keV[0]:.1f}-{peak.roi_bounds_keV[1]:.1f}",
                        ", ".join(peak.candidate_nuclides) or "pending",
                        peak.nuclide or "unassigned",
                        ", ".join(peak.tags),
                    )
                ):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setData(Qt.UserRole, peak.peak_id)
                    self.table.setItem(row, column, item)
                if peak.peak_id == state.selected_peak_id:
                    self.table.selectRow(row)
            self.table.blockSignals(False)

            if state.peaks:
                matched = sum(1 for peak in state.peaks if peak.status == "matched")
                manual = sum(1 for peak in state.peaks if peak.status == "manual")
                self.summary.setText(
                    f"{len(state.peaks)} peaks tracked. {matched} Bayesian matched, {manual} manually assigned. "
                    f"Pinned nuclides: {', '.join(state.pinned_nuclides) or 'none'}."
                )
            else:
                self.summary.setText("No peaks have been added yet.")

            selected_peak = self.workspace_controller.selected_peak()
            if selected_peak is not None:
                self.peak_id_energy.blockSignals(True)
                self.peak_id_energy.setValue(float(selected_peak.energy_keV))
                self.peak_id_energy.blockSignals(False)
            self.assign_isotope_button.setEnabled(selected_peak is not None)
            self.clear_assignment_button.setEnabled(
                selected_peak is not None and selected_peak.nuclide is not None
            )
            self.use_selected_peak_button.setEnabled(selected_peak is not None)
            self._refresh_peak_id_matches()

        def _publish_selected_peak(self) -> None:
            if self._selection_sync_guard:
                return
            row = self.table.currentRow()
            if row < 0 or row >= len(self.workspace_controller.state.peaks):
                return
            peak = self.workspace_controller.state.peaks[row]
            self.workspace_controller.select_peak(peak.peak_id)
            annotation_lines = ()
            if peak.nuclide and peak.reference_lines_keV:
                annotation_lines = tuple(
                    ReferenceLine(
                        energy_keV=float(energy),
                        label=f"{peak.nuclide} ref",
                    )
                    for energy in peak.reference_lines_keV[:4]
                )
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=peak.energy_keV,
                    roi_bounds_keV=peak.roi_bounds_keV,
                    nuclide=peak.nuclide,
                    reference_lines_keV=peak.reference_lines_keV,
                    annotation_lines=annotation_lines,
                )
            )

        def _sync_from_selection_bus(self, state: SelectionState) -> None:
            if self._selection_sync_guard:
                return
            if state.peak_energy_keV is None:
                return
            peaks = self.workspace_controller.state.peaks
            if not peaks:
                return
            target_row = min(
                range(len(peaks)),
                key=lambda index: abs(float(peaks[index].energy_keV) - float(state.peak_energy_keV)),
            )
            target_peak = peaks[target_row]
            if abs(float(target_peak.energy_keV) - float(state.peak_energy_keV)) > 3.0:
                return
            current_row = self.table.currentRow()
            if current_row == target_row and self.workspace_controller.state.selected_peak_id == target_peak.peak_id:
                return
            self._selection_sync_guard = True
            try:
                self.table.selectRow(target_row)
                self.workspace_controller.select_peak(target_peak.peak_id)
            finally:
                self._selection_sync_guard = False


    class ActivityResultsPanel(QWidget):
        """Efficiency, activity, source age, and background workflow surface."""

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self._last_activity_review: ActivityReviewResult | None = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Activity review consumes the active efficiency fit, matched peaks, the "
                    "chosen gamma library, and the time since end of irradiation (EOI) to "
                    "produce count-time and irradiation-time isotope activities."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            control_row = QHBoxLayout()
            self.fit_efficiency_button = QPushButton("Fit Efficiency", self)
            self.fit_efficiency_button.clicked.connect(self._fit_efficiency)
            control_row.addWidget(self.fit_efficiency_button)

            self.compute_activity_button = QPushButton("Compute Activity", self)
            self.compute_activity_button.clicked.connect(self._compute_activity)
            control_row.addWidget(self.compute_activity_button)

            self.analyze_spectrum_button = QPushButton("Analyze Spectrum", self)
            self.analyze_spectrum_button.clicked.connect(self._analyze_spectrum)
            control_row.addWidget(self.analyze_spectrum_button)

            self.export_csv_button = QPushButton("Export CSV", self)
            self.export_csv_button.clicked.connect(self._export_activity_csv_dialog)
            control_row.addWidget(self.export_csv_button)

            self.export_decay_button = QPushButton("Save Decay Plot", self)
            self.export_decay_button.clicked.connect(self._export_decay_plot_dialog)
            control_row.addWidget(self.export_decay_button)

            self.export_bateman_button = QPushButton("Save Bateman Plot", self)
            self.export_bateman_button.clicked.connect(self._export_bateman_plot_dialog)
            control_row.addWidget(self.export_bateman_button)

            self.background_mode_combo = QComboBox(self)
            self.background_mode_combo.addItem("Simple", "simple")
            self.background_mode_combo.addItem("Scaled", "scaled")
            self.background_mode_combo.addItem("Statistical", "statistical")
            self.background_mode_combo.currentIndexChanged.connect(self._background_mode_changed)
            control_row.addWidget(self.background_mode_combo)

            self.background_scale = QDoubleSpinBox(self)
            self.background_scale.setObjectName("BackgroundScaleSpin")
            self.background_scale.setDecimals(3)
            self.background_scale.setRange(0.1, 10.0)
            self.background_scale.setSingleStep(0.1)
            self.background_scale.setValue(1.0)
            self.background_scale.valueChanged.connect(self._background_scale_changed)
            control_row.addWidget(self.background_scale)

            self.background_visible = QCheckBox("Show background overlay", self)
            self.background_visible.setChecked(True)
            self.background_visible.toggled.connect(self._background_visible_toggled)
            control_row.addWidget(self.background_visible)

            self.source_age_hours = QDoubleSpinBox(self)
            self.source_age_hours.setObjectName("SourceAgeHoursSpin")
            self.source_age_hours.setRange(0.0, 87600.0)
            self.source_age_hours.setDecimals(2)
            self.source_age_hours.setSuffix(" h since EOI")
            control_row.addWidget(self.source_age_hours)

            control_row.addWidget(QLabel("Activity units", self))
            self.activity_unit_combo = QComboBox(self)
            self.activity_unit_combo.setObjectName("ActivityResultsUnitCombo")
            for unit in _ACTIVITY_UNIT_FACTORS:
                self.activity_unit_combo.addItem(unit, unit)
            control_row.addWidget(self.activity_unit_combo)
            control_row.addStretch(1)
            layout.addLayout(control_row)

            self.summary = QLabel("No efficiency fit has been applied yet.", self)
            self.summary.setObjectName("PanelBody")
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)

            self.results = QTextBrowser(self)
            self.results.setObjectName("ActivityResultsBrowser")
            layout.addWidget(self.results, 1)

            self.workspace_controller.subscribe(self._sync_state)
            self.activity_unit_combo.currentIndexChanged.connect(
                self._refresh_activity_display
            )
            self._sync_state(self.workspace_controller.state)

        def _fit_efficiency(self) -> None:
            dialog = EfficiencyCalibrationDialog(
                mode_manager=self.mode_manager,
                points=self._seed_efficiency_points(),
                parent=self,
            )
            if dialog.exec() != dialog.Accepted:
                return
            fit = dialog.accepted_fit()
            if fit is None:
                return
            self._last_activity_review = None
            self.workspace_controller.set_efficiency_fit(fit)

        def _seed_efficiency_points(self) -> tuple[EfficiencyPoint, ...]:
            peaks = self.workspace_controller.state.peaks
            if not peaks:
                return ()
            points: list[EfficiencyPoint] = []
            for peak in peaks[:5]:
                points.append(
                    EfficiencyPoint(
                        energy_keV=peak.energy_keV,
                        net_counts=max(peak.net_counts, 1.0),
                        live_time_s=max(float(self.workspace_controller.spectrum().live_time or 100.0), 1.0),
                        activity_bq=1e5,
                        emission_probability=1.0,
                        count_uncertainty=max(np.sqrt(max(peak.net_counts, 1.0)), 1.0),
                    )
                )
            return tuple(points)

        def _current_activity_source(self) -> tuple[str, str | None]:
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode is GUIMode.STANDARDS
                else None
            )
            resolved = self.library_manager.resolved_state(standard=standard)
            return (
                resolved.gamma_identification_source_id,
                resolved.custom_gamma_path,
            )

        def _current_activity_unit(self) -> str:
            return str(self.activity_unit_combo.currentData() or "Bq")

        def _refresh_activity_display(self) -> None:
            self._sync_state(self.workspace_controller.state)

        def _scaled_activity_plot_data(
            self,
            plot_data: dict[str, tuple[tuple[float, float, float], ...]],
        ) -> dict[str, tuple[tuple[float, float, float], ...]]:
            unit = self._current_activity_unit()
            return {
                label: _scale_activity_points(points, unit)
                for label, points in plot_data.items()
            }

        def _review_peaks(self, *, selected_only: bool) -> tuple[PeakCandidate, ...]:
            if selected_only:
                peak = self.workspace_controller.selected_peak()
                return (peak,) if peak is not None else ()
            return tuple(
                peak for peak in self.workspace_controller.state.peaks if peak.nuclide
            )

        def _review_to_activity_results(
            self,
            review: ActivityReviewResult,
        ) -> tuple[ActivityCalculationResult, ...]:
            return tuple(
                ActivityCalculationResult(
                    nuclide=item.nuclide,
                    line_energy_keV=(
                        item.matched_line_energies_keV[0]
                        if item.matched_line_energies_keV
                        else (item.peak_energies_keV[0] if item.peak_energies_keV else 0.0)
                    ),
                    activity_bq=item.count_time_activity_bq,
                    uncertainty_bq=item.count_time_uncertainty_bq,
                    age_corrected_activity_bq=item.irradiation_time_activity_bq,
                    mda_bq=0.0,
                    half_life_s=item.half_life_s,
                    source_age_s=item.cooling_time_s,
                    chain_summary=item.chain_summary,
                    age_corrected_uncertainty_bq=item.irradiation_time_uncertainty_bq,
                )
                for item in review.isotope_summaries
            )

        def _run_activity_review(
            self,
            *,
            selected_only: bool,
        ) -> ActivityReviewResult | None:
            peaks = self._review_peaks(selected_only=selected_only)
            fit = self.workspace_controller.state.efficiency_fit
            spectrum = self.workspace_controller.spectrum()
            if fit is None or spectrum is None:
                self.summary.setText(
                    "Fit an efficiency curve before running activity review."
                )
                return None
            if not peaks:
                self.summary.setText(
                    "No matched peaks are available for activity review."
                )
                return None
            if any(peak.nuclide is None for peak in peaks):
                self.summary.setText(
                    "Assign nuclides to peaks before running activity review."
                )
                return None

            source_id, custom_gamma_path = self._current_activity_source()
            try:
                review = review_spectrum_activation(
                    peaks,
                    live_time_s=max(float(spectrum.live_time or 1.0), 1.0),
                    efficiency_curve=fit.curve,
                    cooling_time_s=float(self.source_age_hours.value()) * 3600.0,
                    source_id=source_id,
                    custom_gamma_path=custom_gamma_path,
                    dead_time_fraction=float(
                        getattr(spectrum, "dead_time_fraction", 0.0) or 0.0
                    ),
                )
            except Exception as exc:
                self.summary.setText(str(exc))
                return None

            self._last_activity_review = review
            self.workspace_controller.set_activity_results(
                self._review_to_activity_results(review)
            )
            return review

        def _compute_activity(self) -> None:
            review = self._run_activity_review(selected_only=True)
            if review is None:
                return
            self.summary.setText(
                f"{self.workspace_controller.state.efficiency_fit.model_label} active. "
                f"Selected-peak review resolved {len(review.line_results)} line(s) to "
                f"{len(review.isotope_summaries)} isotope(s)."
            )

        def _analyze_spectrum(self) -> None:
            review = self._run_activity_review(selected_only=False)
            if review is None:
                return
            self.summary.setText(
                f"{self.workspace_controller.state.efficiency_fit.model_label} active. "
                f"Spectrum review resolved {len(review.line_results)} line(s) across "
                f"{len(review.isotope_summaries)} isotope(s)."
            )

        def analyze_spectrum_activities(self) -> ActivityReviewResult | None:
            """Test-friendly wrapper for full-spectrum activity review."""

            return self._run_activity_review(selected_only=False)

        def current_activity_review(self) -> ActivityReviewResult | None:
            return self._last_activity_review

        def workflow_state(self) -> dict[str, object]:
            return {
                "background_mode": str(self.background_mode_combo.currentData() or "simple"),
                "background_scale": float(self.background_scale.value()),
                "background_visible": bool(self.background_visible.isChecked()),
                "source_age_hours": float(self.source_age_hours.value()),
                "activity_unit": str(self.activity_unit_combo.currentData() or "Bq"),
            }

        def apply_workflow_state(self, payload: dict[str, object] | None) -> None:
            if not payload:
                return
            if "background_mode" in payload:
                set_combo_data(self.background_mode_combo, payload["background_mode"])
            if "background_scale" in payload:
                self.background_scale.setValue(float(payload["background_scale"]))
            if "background_visible" in payload:
                self.background_visible.setChecked(bool(payload["background_visible"]))
            if "source_age_hours" in payload:
                self.source_age_hours.setValue(float(payload["source_age_hours"]))
            if "activity_unit" in payload:
                set_combo_data(self.activity_unit_combo, payload["activity_unit"])

        def _activity_review_for_export(self) -> ActivityReviewResult | None:
            if self._last_activity_review is not None:
                return self._last_activity_review
            return self._run_activity_review(selected_only=False)

        def _prompt_save_path(self, default_name: str, file_filter: str) -> Path | None:
            filename, _selected = QFileDialog.getSaveFileName(
                self,
                "Save Activity Review Output",
                str(Path.cwd() / default_name),
                file_filter,
            )
            return Path(filename) if filename else None

        def _write_csv_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
            if not rows:
                raise ValueError("No activity rows are available to export.")
            fieldnames: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(str(key))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        def export_activity_csv(self, path: str | Path) -> Path:
            review = self._activity_review_for_export()
            if review is None:
                raise ValueError("Activity review is not available.")
            resolved = Path(path)
            self._write_csv_rows(resolved, review.isotope_rows())
            return resolved

        def export_decay_plot(self, path: str | Path) -> Path:
            review = self._activity_review_for_export()
            if review is None:
                raise ValueError("Activity review is not available.")
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            unit = self._current_activity_unit()
            plot_decay_curves(
                self._scaled_activity_plot_data(review.decay_plot_data),
                title="Spectrum Half-Life Decay Review",
                xlabel="Time Since EOI (s)",
                ylabel=f"Activity ({unit})",
                log_y=True,
                log_x=False,
                half_lives=review.half_lives_s,
                save_path=resolved,
            )
            return resolved

        def export_bateman_plot(self, path: str | Path) -> Path:
            review = self._activity_review_for_export()
            if review is None:
                raise ValueError("Activity review is not available.")
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            unit = self._current_activity_unit()
            plot_decay_curves(
                self._scaled_activity_plot_data(review.bateman_plot_data),
                title="Spectrum Bateman Parent/Daughter Review",
                xlabel="Time Since EOI (s)",
                ylabel=f"EOI-Equivalent Inventory ({unit})",
                log_y=False,
                log_x=False,
                half_lives=review.bateman_half_lives_s,
                save_path=resolved,
            )
            return resolved

        def _export_activity_csv_dialog(self) -> None:
            path = self._prompt_save_path(
                "activity_review_isotopes.csv",
                "CSV Files (*.csv)",
            )
            if path is None:
                return
            self.export_activity_csv(path)

        def _export_decay_plot_dialog(self) -> None:
            path = self._prompt_save_path(
                "activity_review_decay.png",
                "PNG Files (*.png)",
            )
            if path is None:
                return
            self.export_decay_plot(path)

        def _export_bateman_plot_dialog(self) -> None:
            path = self._prompt_save_path(
                "activity_review_bateman.png",
                "PNG Files (*.png)",
            )
            if path is None:
                return
            self.export_bateman_plot(path)

        def _background_mode_changed(self) -> None:
            self.workspace_controller.set_background_config(
                mode=str(self.background_mode_combo.currentData())
            )

        def _background_scale_changed(self) -> None:
            self.workspace_controller.set_background_config(
                scale=float(self.background_scale.value())
            )

        def _background_visible_toggled(self, checked: bool) -> None:
            self.workspace_controller.set_background_config(visible=checked)

        def _sync_state(self, state) -> None:
            index = self.background_mode_combo.findData(state.background_mode)
            if index >= 0:
                self.background_mode_combo.blockSignals(True)
                self.background_mode_combo.setCurrentIndex(index)
                self.background_mode_combo.blockSignals(False)
            self.background_scale.blockSignals(True)
            self.background_scale.setValue(float(state.background_scale))
            self.background_scale.blockSignals(False)
            self.background_visible.blockSignals(True)
            self.background_visible.setChecked(bool(state.background_visible))
            self.background_visible.blockSignals(False)

            if state.efficiency_fit is None:
                self.summary.setText("No efficiency fit has been applied yet.")
            else:
                summary = (
                    f"{state.efficiency_fit.model_label} active. "
                    f"RMSE {state.efficiency_fit.rmse:.6f}."
                )
                if self._last_activity_review is not None:
                    summary += (
                        f" Reviewed {len(self._last_activity_review.line_results)} line(s) "
                        f"across {len(self._last_activity_review.isotope_summaries)} isotope(s)."
                    )
                self.summary.setText(summary)
            exports_enabled = bool(state.activity_results)
            self.export_csv_button.setEnabled(exports_enabled)
            self.export_decay_button.setEnabled(exports_enabled)
            self.export_bateman_button.setEnabled(exports_enabled)
            if state.activity_results:
                unit = self._current_activity_unit()
                blocks = []
                for result in state.activity_results:
                    irradiation_uncertainty_bq = float(
                        result.age_corrected_uncertainty_bq or 0.0
                    )
                    lines = [
                        f"Nuclide: {result.nuclide}",
                        f"Representative line: {result.line_energy_keV:.3f} keV",
                        (
                            "Count-time activity: "
                            f"{_format_activity_value(result.activity_bq, unit)} "
                            f"± {_format_activity_value(result.uncertainty_bq, unit, precision='.3g')}"
                        ),
                        (
                            "Irradiation-time activity: "
                            f"{_format_activity_value(result.age_corrected_activity_bq, unit)}"
                        ),
                    ]
                    if irradiation_uncertainty_bq > 0.0:
                        lines.append(
                            "Irradiation-time sigma: "
                            + _format_activity_value(
                                irradiation_uncertainty_bq,
                                unit,
                                precision=".3g",
                            )
                        )
                    if result.mda_bq > 0.0:
                        lines.append(
                            "MDA: " + _format_activity_value(result.mda_bq, unit)
                        )
                    lines.append(result.chain_summary)
                    blocks.append("\n".join(lines))
                self.results.setPlainText(
                    "\n\n".join(blocks)
                )
            else:
                self._last_activity_review = None
                self.results.setPlainText("No activity result has been calculated yet.")


    class InventoryTimelinePanel(QWidget):
        """Inventory/time-evolution surface fed by the current activity review."""

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self._last_result = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Inventory / Time Evolution reconstructs the irradiation-time inventory "
                    "from the current spectrum activity review, then propagates activity, "
                    "atoms, mass, and dose to arbitrary times."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            controls = QGridLayout()
            controls.setHorizontalSpacing(10)
            controls.setVerticalSpacing(8)

            controls.addWidget(QLabel("Decay library", self), 0, 0)
            self.decay_source_combo = QComboBox(self)
            self.decay_source_combo.setObjectName("InventoryDecaySourceCombo")
            for record in list_nuclear_data_sources_by_capability("inventory-decay"):
                self.decay_source_combo.addItem(record.label, record.source_id)
            default_index = self.decay_source_combo.findData(DEFAULT_DECAY_SOURCE_ID)
            if default_index >= 0:
                self.decay_source_combo.setCurrentIndex(default_index)
            controls.addWidget(self.decay_source_combo, 0, 1)

            controls.addWidget(QLabel("Reference nuclide", self), 0, 2)
            self.nuclide_focus_combo = QComboBox(self)
            self.nuclide_focus_combo.setObjectName("InventoryNuclideFocusCombo")
            self.nuclide_focus_combo.addItem("Top contributors", "__top__")
            self.nuclide_focus_combo.addItem("All nuclides", "__all__")
            controls.addWidget(self.nuclide_focus_combo, 0, 3)

            controls.addWidget(QLabel("Time origin", self), 1, 0)
            self.time_origin_combo = QComboBox(self)
            self.time_origin_combo.setObjectName("InventoryTimeOriginCombo")
            self.time_origin_combo.addItem("EOI", "eoi")
            self.time_origin_combo.addItem("Count Start", "count_start")
            self.time_origin_combo.addItem("Count End", "count_end")
            controls.addWidget(self.time_origin_combo, 1, 1)

            controls.addWidget(QLabel("Observable", self), 1, 2)
            self.observable_combo = QComboBox(self)
            self.observable_combo.setObjectName("InventoryObservableCombo")
            self.observable_combo.addItem("Activity", "activity")
            self.observable_combo.addItem("Atoms", "atoms")
            self.observable_combo.addItem("Mass", "mass")
            self.observable_combo.addItem("Dose", "dose")
            controls.addWidget(self.observable_combo, 1, 3)

            controls.addWidget(QLabel("Start (h)", self), 2, 0)
            self.time_start_hours = QDoubleSpinBox(self)
            self.time_start_hours.setObjectName("InventoryTimeStartHoursSpin")
            self.time_start_hours.setRange(-1.0e5, 1.0e5)
            self.time_start_hours.setDecimals(3)
            self.time_start_hours.setValue(0.0)
            controls.addWidget(self.time_start_hours, 2, 1)

            controls.addWidget(QLabel("Stop (h)", self), 2, 2)
            self.time_stop_hours = QDoubleSpinBox(self)
            self.time_stop_hours.setObjectName("InventoryTimeStopHoursSpin")
            self.time_stop_hours.setRange(-1.0e5, 1.0e5)
            self.time_stop_hours.setDecimals(3)
            self.time_stop_hours.setValue(48.0)
            controls.addWidget(self.time_stop_hours, 2, 3)

            controls.addWidget(QLabel("Points", self), 3, 0)
            self.time_point_count = QSpinBox(self)
            self.time_point_count.setObjectName("InventoryTimePointCountSpin")
            self.time_point_count.setRange(2, 200)
            self.time_point_count.setValue(25)
            controls.addWidget(self.time_point_count, 3, 1)

            controls.addWidget(QLabel("Distance (cm)", self), 3, 2)
            self.distance_cm = QDoubleSpinBox(self)
            self.distance_cm.setObjectName("InventoryDoseDistanceSpin")
            self.distance_cm.setRange(0.1, 1.0e5)
            self.distance_cm.setDecimals(2)
            self.distance_cm.setValue(30.0)
            controls.addWidget(self.distance_cm, 3, 3)

            controls.addWidget(QLabel("Top N", self), 4, 0)
            self.top_n_spin = QSpinBox(self)
            self.top_n_spin.setObjectName("InventoryTopContributorCountSpin")
            self.top_n_spin.setRange(1, 20)
            self.top_n_spin.setValue(8)
            controls.addWidget(self.top_n_spin, 4, 1)

            controls.addWidget(QLabel("Activity units", self), 4, 2)
            self.activity_unit_combo = QComboBox(self)
            self.activity_unit_combo.setObjectName("InventoryActivityUnitCombo")
            for unit in _ACTIVITY_UNIT_FACTORS:
                self.activity_unit_combo.addItem(unit, unit)
            controls.addWidget(self.activity_unit_combo, 4, 3)

            controls.addWidget(QLabel("FIM objective", self), 5, 0)
            self.fim_objective_combo = QComboBox(self)
            self.fim_objective_combo.setObjectName("InventoryFIMObjectiveCombo")
            self.fim_objective_combo.addItem("FIM D-opt", "fim-d")
            self.fim_objective_combo.addItem("FIM A-opt", "fim-a")
            self.fim_objective_combo.addItem("FIM C-opt", "fim-c")
            controls.addWidget(self.fim_objective_combo, 5, 1)

            controls.addWidget(QLabel("MWDCS windows", self), 5, 2)
            self.mwdcs_window_count_spin = QSpinBox(self)
            self.mwdcs_window_count_spin.setObjectName("InventoryMWDCSWindowCountSpin")
            self.mwdcs_window_count_spin.setRange(1, 8)
            self.mwdcs_window_count_spin.setValue(3)
            controls.addWidget(self.mwdcs_window_count_spin, 5, 3)

            self.mwdcs_full_spectrum_checkbox = QCheckBox("MWDCS full-spectrum mode", self)
            self.mwdcs_full_spectrum_checkbox.setObjectName("InventoryMWDCSFullSpectrumCheck")
            controls.addWidget(self.mwdcs_full_spectrum_checkbox, 6, 0, 1, 2)

            self.advanced_objective_checkbox = QCheckBox("Enable advanced objectives", self)
            self.advanced_objective_checkbox.setObjectName("InventoryAdvancedObjectiveCheck")
            controls.addWidget(self.advanced_objective_checkbox, 6, 2, 1, 2)

            self.stbdmr_differentiable_checkbox = QCheckBox(
                "STBD-MR differentiable graph mode",
                self,
            )
            self.stbdmr_differentiable_checkbox.setObjectName("InventorySTBDMRDifferentiableCheck")
            controls.addWidget(self.stbdmr_differentiable_checkbox, 7, 0, 1, 2)

            button_row = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh Timeline", self)
            self.refresh_button.clicked.connect(self._refresh_inventory)
            button_row.addWidget(self.refresh_button)

            self.export_csv_button = QPushButton("Export CSV", self)
            self.export_csv_button.clicked.connect(self._export_csv_dialog)
            button_row.addWidget(self.export_csv_button)

            self.export_plot_button = QPushButton("Save Plot", self)
            self.export_plot_button.clicked.connect(self._export_plot_dialog)
            button_row.addWidget(self.export_plot_button)

            self.preview_difom_button = QPushButton("Preview DI-FOM", self)
            self.preview_difom_button.clicked.connect(self._preview_difom_score)
            button_row.addWidget(self.preview_difom_button)

            self.preview_fim_button = QPushButton("Preview FIM", self)
            self.preview_fim_button.clicked.connect(self._preview_fim_score)
            button_row.addWidget(self.preview_fim_button)

            self.preview_mwdcs_button = QPushButton("Preview MWDCS", self)
            self.preview_mwdcs_button.clicked.connect(self._preview_mwdcs_score)
            button_row.addWidget(self.preview_mwdcs_button)

            self.preview_bassd_button = QPushButton("Preview BASS-D", self)
            self.preview_bassd_button.clicked.connect(self._preview_bassd_score)
            button_row.addWidget(self.preview_bassd_button)

            self.preview_stbdmr_button = QPushButton("Preview STBD-MR", self)
            self.preview_stbdmr_button.clicked.connect(self._preview_stbdmr_score)
            button_row.addWidget(self.preview_stbdmr_button)
            button_row.addStretch(1)

            layout.addLayout(controls)
            layout.addLayout(button_row)

            self.summary = QLabel(
                "Run an activity review first to seed the irradiation-time inventory.",
                self,
            )
            self.summary.setObjectName("PanelBody")
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)

            self.difom_summary = QLabel(
                "DI-FOM preview is unavailable until activity-review results are loaded.",
                self,
            )
            self.difom_summary.setObjectName("PanelBody")
            self.difom_summary.setWordWrap(True)
            layout.addWidget(self.difom_summary)

            self.fim_summary = QLabel(
                "FIM preview is unavailable until activity-review results are loaded.",
                self,
            )
            self.fim_summary.setObjectName("PanelBody")
            self.fim_summary.setWordWrap(True)
            layout.addWidget(self.fim_summary)

            self.mwdcs_summary = QLabel(
                "MWDCS preview is unavailable until activity-review results are loaded.",
                self,
            )
            self.mwdcs_summary.setObjectName("PanelBody")
            self.mwdcs_summary.setWordWrap(True)
            layout.addWidget(self.mwdcs_summary)

            self.bassd_summary = QLabel(
                "BASS-D preview is unavailable until advanced objectives are enabled.",
                self,
            )
            self.bassd_summary.setObjectName("PanelBody")
            self.bassd_summary.setWordWrap(True)
            layout.addWidget(self.bassd_summary)

            self.stbdmr_summary = QLabel(
                "STBD-MR preview is unavailable until advanced objectives are enabled.",
                self,
            )
            self.stbdmr_summary.setObjectName("PanelBody")
            self.stbdmr_summary.setWordWrap(True)
            layout.addWidget(self.stbdmr_summary)

            self.family_browser = QTextBrowser(self)
            self.family_browser.setObjectName("InventoryFamilyBrowser")
            layout.addWidget(self.family_browser, 1)

            self.table = QTableWidget(0, 8, self)
            self.table.setObjectName("InventoryTimelineTable")
            self.table.setHorizontalHeaderLabels(
                (
                    "Nuclide",
                    "Rel. Time (h)",
                    "Activity (Bq)",
                    "Atoms",
                    "Mass (g)",
                    "Dose (uSv/h)",
                    "Act. Sigma",
                    "Dose Sigma",
                )
            )
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table, 2)

            self.workspace_controller.subscribe(self._sync_workspace_state)
            self.nuclide_focus_combo.currentIndexChanged.connect(self._render_last_result)
            self.observable_combo.currentIndexChanged.connect(self._render_last_result)
            self.activity_unit_combo.currentIndexChanged.connect(self._render_last_result)
            self.fim_objective_combo.currentIndexChanged.connect(self._preview_fim_score)
            self.mwdcs_window_count_spin.valueChanged.connect(self._preview_mwdcs_score)
            self.mwdcs_full_spectrum_checkbox.toggled.connect(self._preview_mwdcs_score)
            self.advanced_objective_checkbox.toggled.connect(self._preview_bassd_score)
            self.advanced_objective_checkbox.toggled.connect(self._preview_stbdmr_score)
            self.stbdmr_differentiable_checkbox.toggled.connect(self._preview_stbdmr_score)
            self._sync_workspace_state(self.workspace_controller.state)

        def build_inventory_timeline(self):
            return self._compute_inventory_timeline()

        def _current_focus(self) -> str:
            data = self.nuclide_focus_combo.currentData()
            return str(data) if data else "__top__"

        def _current_observable(self) -> str:
            data = self.observable_combo.currentData()
            return str(data) if data else "activity"

        def _current_activity_unit(self) -> str:
            return str(self.activity_unit_combo.currentData() or "Bq")

        def _scale_activity_value(self, value_bq: float) -> float:
            return float(value_bq) / _activity_unit_factor(self._current_activity_unit())

        def _scale_activity_plot_data(
            self,
            plot_data: dict[str, tuple[tuple[float, float, float], ...]],
        ) -> dict[str, tuple[tuple[float, float, float], ...]]:
            unit = self._current_activity_unit()
            return {
                label: _scale_activity_points(points, unit)
                for label, points in plot_data.items()
            }

        def _current_time_grid_s(self) -> tuple[float, ...]:
            return build_time_grid(
                start_s=float(self.time_start_hours.value()) * 3600.0,
                stop_s=float(self.time_stop_hours.value()) * 3600.0,
                count=int(self.time_point_count.value()),
            )

        def _inventory_state(self):
            spectrum = self.workspace_controller.spectrum()
            if spectrum is None:
                self.summary.setText("No active spectrum is available for inventory evolution.")
                return None
            activity_results = self.workspace_controller.state.activity_results
            if not activity_results:
                self.summary.setText(
                    "Run an activity review first to seed the irradiation-time inventory."
                )
                return None
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode is GUIMode.STANDARDS
                else None
            )
            resolved = self.library_manager.resolved_state(standard=standard)
            return build_inventory_state_from_activity_results(
                activity_results,
                live_time_s=max(float(spectrum.live_time or 0.0), 0.0),
                gamma_source_id=resolved.gamma_identification_source_id,
                custom_gamma_path=resolved.custom_gamma_path,
                sample_id=str(spectrum.spectrum_id or ""),
                decay_source_id=str(
                    self.decay_source_combo.currentData() or DEFAULT_DECAY_SOURCE_ID
                ),
            )

        def _compute_inventory_timeline(self):
            inventory_state = self._inventory_state()
            if inventory_state is None:
                return None
            try:
                result = compute_inventory_time_evolution(
                    inventory_state,
                    relative_times_s=self._current_time_grid_s(),
                    time_origin=str(self.time_origin_combo.currentData() or "eoi"),
                    decay_source_id=str(
                        self.decay_source_combo.currentData() or DEFAULT_DECAY_SOURCE_ID
                    ),
                    distance_cm=float(self.distance_cm.value()),
                )
            except Exception as exc:
                self.summary.setText(str(exc))
                return None
            self._last_result = result
            return result

        def _refresh_inventory(self) -> None:
            result = self._compute_inventory_timeline()
            if result is None:
                return
            total_activity_points = result.activity_series.get("Total")
            total_dose_points = result.dose_series.get("Total")
            final_activity = (
                total_activity_points.points[-1][1]
                if total_activity_points is not None and total_activity_points.points
                else 0.0
            )
            final_dose = (
                total_dose_points.points[-1][1]
                if total_dose_points is not None and total_dose_points.points
                else 0.0
            )
            self.summary.setText(
                (
                    f"Evolved {len(result.inventory_state.seeds)} seed isotope(s) across "
                    f"{len(result.relative_times_s)} time points. "
                    f"Final total activity {_format_activity_value(final_activity, self._current_activity_unit())}; "
                    f"final total dose {final_dose:.6g} uSv/h."
                )
            )
            self._render_result(result)
            self._preview_difom_score()
            self._preview_fim_score()
            self._preview_mwdcs_score()
            self._preview_bassd_score()
            self._preview_stbdmr_score()

        def preview_difom_score(self) -> float | None:
            """Return a proxy DI-FOM score from current activity-review line estimates."""

            activity_results = tuple(self.workspace_controller.state.activity_results)
            if not activity_results:
                self.difom_summary.setText(
                    "DI-FOM preview is unavailable until activity-review results are loaded."
                )
                return None

            line_terms = build_difom_terms_from_activity_results(activity_results)
            if not line_terms:
                self.difom_summary.setText(
                    "DI-FOM preview is unavailable because no positive line terms were found."
                )
                return None

            evaluation = evaluate_difom(line_terms)
            self.difom_summary.setText(
                (
                    f"DI-FOM preview score: {evaluation.total_score:.6g} from "
                    f"{len(evaluation.line_scores)} line(s). "
                    "Proxy model uses age-corrected activity as signal and activity "
                    "uncertainty as background until full masking terms are enabled."
                )
            )
            return float(evaluation.total_score)

        def _preview_difom_score(self) -> None:
            self.preview_difom_score()

        def preview_fim_score(self) -> float | None:
            """Return a proxy FIM score from current activity-review line estimates."""

            activity_results = tuple(self.workspace_controller.state.activity_results)
            if not activity_results:
                self.fim_summary.setText(
                    "FIM preview is unavailable until activity-review results are loaded."
                )
                return None

            line_terms = build_difom_terms_from_activity_results(activity_results)
            if not line_terms:
                self.fim_summary.setText(
                    "FIM preview is unavailable because no positive line terms were found."
                )
                return None

            objective = str(self.fim_objective_combo.currentData() or "fim-d")
            focus = self._current_focus()
            target_nuclide = focus if focus not in {"__top__", "__all__"} else None
            try:
                evaluation = evaluate_fim(
                    line_terms,
                    objective=objective,
                    target_nuclide=target_nuclide,
                    nuisance_variance_fraction=0.05,
                    regularization=1.0e-6,
                )
            except ValueError as exc:
                self.fim_summary.setText(f"FIM preview unavailable: {exc}")
                return None

            diagnostics = evaluation.diagnostics
            self.fim_summary.setText(
                (
                    f"FIM preview ({objective}) score: {evaluation.objective_score:.6g} across "
                    f"{len(diagnostics.nuclides)} nuclide parameter(s). "
                    f"condition number {diagnostics.condition_number:.6g}, "
                    f"effective rank {diagnostics.effective_rank}."
                )
            )
            return float(evaluation.objective_score)

        def _preview_fim_score(self) -> None:
            self.preview_fim_score()

        def preview_mwdcs_score(self) -> float | None:
            """Return a proxy MWDCS score from activity-review results."""

            activity_results = tuple(self.workspace_controller.state.activity_results)
            if not activity_results:
                self.mwdcs_summary.setText(
                    "MWDCS preview is unavailable until activity-review results are loaded."
                )
                return None

            window_count = max(int(self.mwdcs_window_count_spin.value()), 1)
            stop_time_s = max(float(self.time_stop_hours.value()) * 3600.0, 0.0)
            if window_count == 1:
                offsets = (0.0,)
            elif stop_time_s <= 0.0:
                offsets = tuple(np.linspace(0.0, 86400.0, window_count))
            else:
                offsets = tuple(np.linspace(0.0, stop_time_s, window_count))

            candidate = build_mwdcs_candidate_from_activity_results(
                activity_results,
                label="gui_preview",
                window_offsets_s=offsets,
                window_count_time_s=900.0,
            )
            if len(candidate.windows) == 0:
                self.mwdcs_summary.setText(
                    "MWDCS preview is unavailable because no positive line terms were found."
                )
                return None

            full_spectrum_mode = bool(self.mwdcs_full_spectrum_checkbox.isChecked())
            evaluation = evaluate_mwdcs(
                candidate.windows,
                full_spectrum_mode=full_spectrum_mode,
                overlap_penalty=0.1,
            )
            self.mwdcs_summary.setText(
                (
                    f"MWDCS preview score: {evaluation.total_score:.6g} across "
                    f"{len(evaluation.window_scores)} window(s). "
                    f"Full-spectrum mode: {'on' if full_spectrum_mode else 'off'}."
                )
            )
            return float(evaluation.total_score)

        def _preview_mwdcs_score(self) -> None:
            self.preview_mwdcs_score()

        def preview_bassd_score(self) -> float | None:
            """Return a proxy BASS-D utility score from activity-review results."""

            if not bool(self.advanced_objective_checkbox.isChecked()):
                self.bassd_summary.setText(
                    "BASS-D preview is disabled. Enable advanced objectives to continue."
                )
                return None

            activity_results = tuple(self.workspace_controller.state.activity_results)
            if not activity_results:
                self.bassd_summary.setText(
                    "BASS-D preview is unavailable until activity-review results are loaded."
                )
                return None

            candidate = build_bassd_candidate_from_activity_results(
                activity_results,
                label="gui_preview",
                cooldown_time_s=max(float(self.time_start_hours.value()) * 3600.0, 0.0),
                count_time_s=900.0,
            )
            if len(candidate.actions) == 0 or len(candidate.actions[0].lines) == 0:
                self.bassd_summary.setText(
                    "BASS-D preview is unavailable because no positive line terms were found."
                )
                return None

            evaluation = evaluate_bassd(
                candidate.actions,
                dose_weight=0.02,
                exploration_temperature=0.0,
                seed=17,
            )
            self.bassd_summary.setText(
                (
                    f"BASS-D preview utility: {evaluation.total_utility:.6g} across "
                    f"{len(evaluation.action_scores)} action(s). "
                    "Advanced guard: enabled."
                )
            )
            return float(evaluation.total_utility)

        def _preview_bassd_score(self) -> None:
            self.preview_bassd_score()

        def preview_stbdmr_score(self) -> float | None:
            """Return a proxy STBD-MR score from activity-review results."""

            if not bool(self.advanced_objective_checkbox.isChecked()):
                self.stbdmr_summary.setText(
                    "STBD-MR preview is disabled. Enable advanced objectives to continue."
                )
                return None

            activity_results = tuple(self.workspace_controller.state.activity_results)
            if not activity_results:
                self.stbdmr_summary.setText(
                    "STBD-MR preview is unavailable until activity-review results are loaded."
                )
                return None

            window_count = max(int(self.mwdcs_window_count_spin.value()), 1)
            stop_time_s = max(float(self.time_stop_hours.value()) * 3600.0, 0.0)
            if window_count == 1:
                offsets = (0.0,)
            elif stop_time_s <= 0.0:
                offsets = tuple(np.linspace(0.0, 86400.0, window_count))
            else:
                offsets = tuple(np.linspace(0.0, stop_time_s, window_count))

            candidate = build_stbdmr_candidate_from_activity_results(
                activity_results,
                label="gui_preview",
                window_offsets_s=offsets,
                window_count_time_s=900.0,
            )
            if len(candidate.windows) == 0:
                self.stbdmr_summary.setText(
                    "STBD-MR preview is unavailable because no positive line terms were found."
                )
                return None

            differentiable_mode = bool(self.stbdmr_differentiable_checkbox.isChecked())
            evaluation = evaluate_stbdmr(
                candidate.windows,
                masking_regularization=0.1,
                differentiable_graph_mode=differentiable_mode,
                graph_temperature=2.0,
            )
            diagnostics = evaluation.diagnostics
            self.stbdmr_summary.setText(
                (
                    f"STBD-MR preview score: {evaluation.total_score:.6g} across "
                    f"{len(evaluation.window_scores)} window(s). "
                    f"graph density {diagnostics.graph_density:.4g}, "
                    f"masking penalty {diagnostics.masking_penalty:.6g}."
                )
            )
            return float(evaluation.total_score)

        def _preview_stbdmr_score(self) -> None:
            self.preview_stbdmr_score()

        def _selected_plot_data(self, result) -> dict[str, tuple[tuple[float, float, float], ...]]:
            observable = self._current_observable()
            focus = self._current_focus()
            if focus == "__all__":
                return {
                    label: series.points
                    for label, series in result.series_for(observable).items()
                }
            if focus not in {"__top__", "__all__"}:
                series = result.series_for(observable)
                selected: dict[str, tuple[tuple[float, float, float], ...]] = {}
                if "Total" in series:
                    selected["Total"] = series["Total"].points
                if focus in series:
                    selected[focus] = series[focus].points
                return selected
            return result.plot_data(
                observable,
                top_n=int(self.top_n_spin.value()),
                include_total=True,
            )

        def _render_last_result(self) -> None:
            if self._last_result is not None:
                self._render_result(self._last_result)

        def _render_result(self, result) -> None:
            focus = self._current_focus()
            if focus in {"__top__", "__all__"}:
                candidate_labels = [
                    label
                    for label in result.parents_by_nuclide.keys()
                    if label in result.activity_series and label != "Total"
                ]
                focus = candidate_labels[0] if candidate_labels else "Total"
            self.family_browser.setPlainText(self._family_summary(result, focus))
            rows = [
                row
                for row in result.time_series_rows()
                if self._row_visible(row)
            ]
            activity_unit = self._current_activity_unit()
            self.table.setHorizontalHeaderLabels(
                (
                    "Nuclide",
                    "Rel. Time (h)",
                    f"Activity ({activity_unit})",
                    "Atoms",
                    "Mass (g)",
                    "Dose (uSv/h)",
                    f"Act. Sigma ({activity_unit})",
                    "Dose Sigma (uSv/h)",
                )
            )
            self.table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = (
                    row["nuclide"],
                    f"{float(row['relative_time_s']) / 3600.0:.3f}",
                    f"{self._scale_activity_value(float(row['activity_bq'])):.6g}",
                    f"{float(row['atoms']):.6g}",
                    f"{float(row['mass_g']):.6g}",
                    f"{float(row['dose_rate_uSv_h']):.6g}",
                    f"{self._scale_activity_value(float(row['activity_uncertainty_bq'])):.3g}",
                    f"{float(row['dose_rate_uncertainty_uSv_h']):.3g}",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    self.table.setItem(row_index, column, item)

        def _row_visible(self, row: dict[str, object]) -> bool:
            focus = self._current_focus()
            nuclide = str(row.get("nuclide", ""))
            if focus == "__all__":
                return True
            if focus == "__top__":
                if self._last_result is None:
                    return nuclide == "Total"
                selected = set(self._selected_plot_data(self._last_result).keys())
                return nuclide in selected
            return nuclide in {"Total", focus}

        def _family_summary(self, result, nuclide: str) -> str:
            reference_rows = result.reference_rows("eoi")
            half_life_s = None
            for row in reference_rows:
                if str(row.get("nuclide")) == nuclide:
                    half_life_s = row.get("half_life_s")
                    break
            parents = ", ".join(result.parents_by_nuclide.get(nuclide, ())) or "none"
            daughters = ", ".join(result.daughters_by_nuclide.get(nuclide, ())) or "none"
            lines = [
                f"Reference nuclide: {nuclide}",
                f"Immediate parents: {parents}",
                f"Immediate daughters: {daughters}",
                f"EOI to count start: {_format_duration(result.inventory_state.schedule.count_start_time_s)}",
                f"Count live time: {_format_duration(result.inventory_state.schedule.count_live_time_s)}",
            ]
            if half_life_s:
                lines.append(f"Half-life: {_format_duration(float(half_life_s))}")
            lines.append("")
            lines.extend(result.notes)
            return "\n".join(lines)

        def _prompt_save_path(self, default_name: str, file_filter: str) -> Path | None:
            filename, _selected = QFileDialog.getSaveFileName(
                self,
                "Save Inventory Output",
                str(Path.cwd() / default_name),
                file_filter,
            )
            return Path(filename) if filename else None

        def _write_csv_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
            if not rows:
                raise ValueError("No inventory rows are available to export.")
            fieldnames: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(str(key))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        def export_time_series_csv(self, path: str | Path) -> Path:
            result = self._last_result or self._compute_inventory_timeline()
            if result is None:
                raise ValueError("Inventory timeline is not available.")
            resolved = Path(path)
            self._write_csv_rows(resolved, result.time_series_rows())
            return resolved

        def export_plot(self, path: str | Path) -> Path:
            result = self._last_result or self._compute_inventory_timeline()
            if result is None:
                raise ValueError("Inventory timeline is not available.")
            observable = self._current_observable()
            plot_data = self._selected_plot_data(result)
            ylabel = {
                "activity": f"Activity ({self._current_activity_unit()})",
                "atoms": "Atoms",
                "mass": "Mass (g)",
                "dose": "Dose Rate (uSv/h)",
            }[observable]
            if observable == "activity":
                plot_data = self._scale_activity_plot_data(plot_data)
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            half_lives = {
                str(row["nuclide"]): float(row["half_life_s"])
                for row in result.reference_rows("eoi")
                if row.get("half_life_s") is not None
            }
            plot_decay_curves(
                plot_data,
                title=f"Inventory Time Evolution ({observable.title()})",
                xlabel=(
                    f"Time Since "
                    f"{str(self.time_origin_combo.currentData() or 'eoi').replace('_', ' ').title()} (s)"
                ),
                ylabel=ylabel,
                log_y=observable in {"activity", "atoms", "dose"},
                log_x=False,
                half_lives=half_lives,
                save_path=resolved,
            )
            return resolved

        def _export_csv_dialog(self) -> None:
            path = self._prompt_save_path(
                "inventory_timeseries.csv",
                "CSV Files (*.csv)",
            )
            if path is None:
                return
            self.export_time_series_csv(path)

        def _export_plot_dialog(self) -> None:
            path = self._prompt_save_path(
                f"inventory_{self._current_observable()}.png",
                "PNG Files (*.png)",
            )
            if path is None:
                return
            self.export_plot(path)

        def _sync_workspace_state(self, state) -> None:
            current_focus = self._current_focus()
            self.nuclide_focus_combo.blockSignals(True)
            self.nuclide_focus_combo.clear()
            self.nuclide_focus_combo.addItem("Top contributors", "__top__")
            self.nuclide_focus_combo.addItem("All nuclides", "__all__")
            for result in state.activity_results:
                self.nuclide_focus_combo.addItem(result.nuclide, result.nuclide)
            index = self.nuclide_focus_combo.findData(current_focus)
            if index < 0:
                index = 0
            self.nuclide_focus_combo.setCurrentIndex(index)
            self.nuclide_focus_combo.blockSignals(False)

            enabled = bool(state.activity_results)
            self.refresh_button.setEnabled(enabled)
            self.export_csv_button.setEnabled(enabled)
            self.export_plot_button.setEnabled(enabled)
            self.preview_difom_button.setEnabled(enabled)
            self.preview_fim_button.setEnabled(enabled)
            self.preview_mwdcs_button.setEnabled(enabled)
            self.preview_bassd_button.setEnabled(enabled)
            self.preview_stbdmr_button.setEnabled(enabled)
            self.fim_objective_combo.setEnabled(enabled)
            self.mwdcs_window_count_spin.setEnabled(enabled)
            self.mwdcs_full_spectrum_checkbox.setEnabled(enabled)
            self.advanced_objective_checkbox.setEnabled(enabled)
            self.stbdmr_differentiable_checkbox.setEnabled(enabled)
            self.nuclide_focus_combo.setEnabled(enabled)
            if not enabled:
                self._last_result = None
                self.family_browser.setPlainText(
                    "Run an activity review first to seed the irradiation-time inventory."
                )
                self.table.setRowCount(0)
                self.difom_summary.setText(
                    "DI-FOM preview is unavailable until activity-review results are loaded."
                )
                self.fim_summary.setText(
                    "FIM preview is unavailable until activity-review results are loaded."
                )
                self.mwdcs_summary.setText(
                    "MWDCS preview is unavailable until activity-review results are loaded."
                )
                self.bassd_summary.setText(
                    "BASS-D preview is unavailable until advanced objectives are enabled."
                )
                self.stbdmr_summary.setText(
                    "STBD-MR preview is unavailable until advanced objectives are enabled."
                )

        def workflow_state(self) -> dict[str, object]:
            return {
                "decay_source_id": str(
                    self.decay_source_combo.currentData() or DEFAULT_DECAY_SOURCE_ID
                ),
                "nuclide_focus": self._current_focus(),
                "time_origin": str(self.time_origin_combo.currentData() or "eoi"),
                "observable": self._current_observable(),
                "time_start_hours": float(self.time_start_hours.value()),
                "time_stop_hours": float(self.time_stop_hours.value()),
                "time_point_count": int(self.time_point_count.value()),
                "distance_cm": float(self.distance_cm.value()),
                "top_n": int(self.top_n_spin.value()),
                "activity_unit": self._current_activity_unit(),
                "fim_objective": str(self.fim_objective_combo.currentData() or "fim-d"),
                "mwdcs_window_count": int(self.mwdcs_window_count_spin.value()),
                "mwdcs_full_spectrum": bool(
                    self.mwdcs_full_spectrum_checkbox.isChecked()
                ),
                "advanced_objectives": bool(
                    self.advanced_objective_checkbox.isChecked()
                ),
                "stbdmr_differentiable": bool(
                    self.stbdmr_differentiable_checkbox.isChecked()
                ),
            }

        def apply_workflow_state(self, payload: dict[str, object] | None) -> None:
            if not payload:
                return
            if "decay_source_id" in payload:
                set_combo_data(self.decay_source_combo, payload["decay_source_id"])
            if "time_origin" in payload:
                set_combo_data(self.time_origin_combo, payload["time_origin"])
            if "observable" in payload:
                set_combo_data(self.observable_combo, payload["observable"])
            if "time_start_hours" in payload:
                self.time_start_hours.setValue(float(payload["time_start_hours"]))
            if "time_stop_hours" in payload:
                self.time_stop_hours.setValue(float(payload["time_stop_hours"]))
            if "time_point_count" in payload:
                self.time_point_count.setValue(max(int(payload["time_point_count"]), 2))
            if "distance_cm" in payload:
                self.distance_cm.setValue(float(payload["distance_cm"]))
            if "top_n" in payload:
                self.top_n_spin.setValue(max(int(payload["top_n"]), 1))
            if "activity_unit" in payload:
                set_combo_data(self.activity_unit_combo, payload["activity_unit"])
            if "fim_objective" in payload:
                set_combo_data(self.fim_objective_combo, payload["fim_objective"])
            if "mwdcs_window_count" in payload:
                self.mwdcs_window_count_spin.setValue(
                    max(int(payload["mwdcs_window_count"]), 1)
                )
            if "mwdcs_full_spectrum" in payload:
                self.mwdcs_full_spectrum_checkbox.setChecked(
                    bool(payload["mwdcs_full_spectrum"])
                )
            if "advanced_objectives" in payload:
                self.advanced_objective_checkbox.setChecked(
                    bool(payload["advanced_objectives"])
                )
            if "stbdmr_differentiable" in payload:
                self.stbdmr_differentiable_checkbox.setChecked(
                    bool(payload["stbdmr_differentiable"])
                )
            if "nuclide_focus" in payload:
                set_combo_data(self.nuclide_focus_combo, payload["nuclide_focus"])


    class RoiToolsPanel(QWidget):
        """Explicit ROI/background workflow surface for offline parity work."""

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.registries = bootstrap_builtin_registries()
            if len(self.registries.peak_search_methods) == 0:
                register_builtin_peak_search_methods(self.registries)
            if len(self.registries.roi_background_models) == 0:
                register_builtin_roi_background_methods(self.registries)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Explicit ROI/background workspace for offline parity: choose a peak-search "
                    "method, estimate continuum with sidebands or SNIP, decompose overlaps, and "
                    "review the same ROI across many loaded spectra."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            bounds_row = QHBoxLayout()
            bounds_row.addWidget(QLabel("ROI left (keV)", self))
            self.roi_left = QDoubleSpinBox(self)
            self.roi_left.setObjectName("RoiLeftBoundSpin")
            self.roi_left.setRange(0.0, 10000.0)
            self.roi_left.setDecimals(3)
            self.roi_left.setSingleStep(0.25)
            bounds_row.addWidget(self.roi_left)
            bounds_row.addWidget(QLabel("ROI right (keV)", self))
            self.roi_right = QDoubleSpinBox(self)
            self.roi_right.setObjectName("RoiRightBoundSpin")
            self.roi_right.setRange(0.0, 10000.0)
            self.roi_right.setDecimals(3)
            self.roi_right.setSingleStep(0.25)
            bounds_row.addWidget(self.roi_right)
            self.use_selected_peak_button = QPushButton("Use Selected Peak ROI", self)
            self.use_selected_peak_button.setObjectName("RoiUseSelectedPeakButton")
            self.use_selected_peak_button.clicked.connect(self._use_selected_peak_roi)
            bounds_row.addWidget(self.use_selected_peak_button)
            bounds_row.addStretch(1)
            layout.addLayout(bounds_row)

            selector_row = QHBoxLayout()
            self.peak_search_selector = MethodSelectorWidget(
                self.registries.peak_search_methods,
                self.mode_manager,
                title="Peak Search",
                parent=self,
            )
            self.peak_search_selector.setObjectName("RoiPeakSearchMethodSelector")
            selector_row.addWidget(self.peak_search_selector, 1)
            self.background_selector = MethodSelectorWidget(
                self.registries.roi_background_models,
                self.mode_manager,
                title="ROI Background",
                parent=self,
            )
            self.background_selector.setObjectName("RoiBackgroundMethodSelector")
            selector_row.addWidget(self.background_selector, 1)
            layout.addLayout(selector_row)

            options_row = QHBoxLayout()
            options_row.addWidget(QLabel("Sideband width (keV)", self))
            self.sideband_width = QDoubleSpinBox(self)
            self.sideband_width.setObjectName("RoiSidebandWidthSpin")
            self.sideband_width.setRange(0.5, 250.0)
            self.sideband_width.setDecimals(2)
            self.sideband_width.setSingleStep(0.5)
            self.sideband_width.setValue(4.0)
            options_row.addWidget(self.sideband_width)
            self.overlap_checkbox = QCheckBox("Decompose overlaps", self)
            self.overlap_checkbox.setObjectName("RoiDecomposeOverlapsCheck")
            self.overlap_checkbox.setChecked(True)
            options_row.addWidget(self.overlap_checkbox)
            options_row.addWidget(QLabel("Max comps", self))
            self.max_components = QSpinBox(self)
            self.max_components.setObjectName("RoiMaxComponentsSpin")
            self.max_components.setRange(1, 6)
            self.max_components.setValue(3)
            options_row.addWidget(self.max_components)
            options_row.addStretch(1)
            layout.addLayout(options_row)

            action_row = QHBoxLayout()
            self.analyze_button = QPushButton("Analyze ROI", self)
            self.analyze_button.setObjectName("AnalyzeRoiButton")
            self.analyze_button.clicked.connect(self._analyze_roi)
            action_row.addWidget(self.analyze_button)
            self.statistics_button = QPushButton("ROI Statistics", self)
            self.statistics_button.setObjectName("AnalyzeRoiStatisticsButton")
            self.statistics_button.clicked.connect(self._analyze_roi_statistics)
            action_row.addWidget(self.statistics_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self.summary = QTextBrowser(self)
            self.summary.setObjectName("RoiAnalysisSummaryBrowser")
            layout.addWidget(self.summary)

            layout.addWidget(QLabel("Overlap components", self))
            self.component_table = QTableWidget(0, 5, self)
            self.component_table.setObjectName("RoiOverlapTable")
            self.component_table.setHorizontalHeaderLabels(
                ("Centroid (keV)", "Net Counts", "σ Net", "FWHM (ch)", "χ²/dof")
            )
            self.component_table.verticalHeader().setVisible(False)
            self.component_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.component_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.component_table, 1)

            layout.addWidget(QLabel("ROI statistics across loaded spectra", self))
            self.statistics_table = QTableWidget(0, 4, self)
            self.statistics_table.setObjectName("RoiStatisticsTable")
            self.statistics_table.setHorizontalHeaderLabels(
                ("Spectrum", "Net Counts", "σ Net", "Centroid (keV)")
            )
            self.statistics_table.verticalHeader().setVisible(False)
            self.statistics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.statistics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.statistics_table, 1)

            self.peak_search_selector.combo.currentIndexChanged.connect(
                self._peak_search_method_changed
            )
            self.background_selector.combo.currentIndexChanged.connect(
                self._background_method_changed
            )
            self.workspace_controller.subscribe(self._sync_state)
            self._sync_state(self.workspace_controller.state)

        def _use_selected_peak_roi(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None:
                return
            self.roi_left.setValue(float(peak.roi_bounds_keV[0]))
            self.roi_right.setValue(float(peak.roi_bounds_keV[1]))
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=peak.energy_keV,
                    roi_bounds_keV=peak.roi_bounds_keV,
                    nuclide=peak.nuclide,
                    reference_lines_keV=peak.reference_lines_keV,
                )
            )

        def _peak_search_method_changed(self) -> None:
            method = self.peak_search_selector.current_key()
            if method:
                self.workspace_controller.set_peak_search_method(str(method))

        def _background_method_changed(self) -> None:
            method = self.background_selector.current_key()
            if method:
                self.workspace_controller.set_roi_background_method(str(method))

        def _active_foreground_with_background(self) -> tuple[GammaSpectrum | None, GammaSpectrum | None]:
            foreground = self.workspace_controller.spectrum()
            background = None
            if self.workspace_controller.state.active_spectrum_key != "background":
                background_slot = self.workspace_controller.slot("background")
                if background_slot is not None:
                    background = background_slot.spectrum
            return foreground, background

        def _roi_bounds(self) -> tuple[float, float]:
            return tuple(sorted((float(self.roi_left.value()), float(self.roi_right.value()))))

        def _analyze_roi(self) -> None:
            foreground, background = self._active_foreground_with_background()
            if foreground is None:
                self.summary.setHtml("<p>No active spectrum available for ROI analysis.</p>")
                return
            result = analyze_roi_region(
                foreground,
                roi_bounds_keV=self._roi_bounds(),
                label=(foreground.spectrum_id or "ROI"),
                background_method=self.background_selector.current_key()
                or self.workspace_controller.state.roi_background_method,
                peak_search_method=self.peak_search_selector.current_key()
                or self.workspace_controller.state.peak_search_method,
                sideband_width_keV=float(self.sideband_width.value()),
                background_spectrum=background,
                background_mode=self.workspace_controller.state.background_mode,
                background_scale=self.workspace_controller.state.background_scale,
                decompose_overlaps=bool(self.overlap_checkbox.isChecked()),
                max_components=int(self.max_components.value()),
                registries=self.registries,
            )
            self.workspace_controller.set_roi_analysis(result)
            self.selection_bus.publish(
                SelectionState(
                    roi_bounds_keV=result.roi_bounds_keV,
                    peak_energy_keV=result.centroid_keV,
                )
            )

        def _statistics_inputs(self) -> tuple[tuple[str, GammaSpectrum], ...]:
            background_source = None
            background_slot = self.workspace_controller.slot("background")
            if background_slot is not None:
                background_source = background_slot.source_key
            records = []
            for record in self.workspace_controller.loaded_spectrum_records():
                if background_source is not None and record.key == background_source:
                    continue
                records.append((record.label, record.spectrum))
            if not records:
                for slot in self.workspace_controller.spectrum_slots():
                    if slot.key == "background":
                        continue
                    records.append((slot.source_label or slot.label, slot.spectrum))
            return tuple(records)

        def _analyze_roi_statistics(self) -> None:
            inputs = self._statistics_inputs()
            if not inputs:
                self.summary.setHtml("<p>No loaded spectra available for ROI statistics.</p>")
                return
            result = compute_roi_statistics(
                inputs,
                roi_bounds_keV=self._roi_bounds(),
                label="ROI Statistics",
                background_method=self.background_selector.current_key()
                or self.workspace_controller.state.roi_background_method,
                peak_search_method=self.peak_search_selector.current_key()
                or self.workspace_controller.state.peak_search_method,
                sideband_width_keV=float(self.sideband_width.value()),
                registries=self.registries,
            )
            self.workspace_controller.set_roi_statistics(result)

        def _sync_state(self, state) -> None:
            if self.peak_search_selector.current_key() != state.peak_search_method:
                self.peak_search_selector.combo.blockSignals(True)
                self.peak_search_selector.set_current_key(state.peak_search_method)
                self.peak_search_selector.combo.blockSignals(False)
                self.peak_search_selector._sync_badge()
            if self.background_selector.current_key() != state.roi_background_method:
                self.background_selector.combo.blockSignals(True)
                self.background_selector.set_current_key(state.roi_background_method)
                self.background_selector.combo.blockSignals(False)
                self.background_selector._sync_badge()

            if state.roi_analysis is None:
                self.summary.setHtml("<p>No ROI analysis has been run yet.</p>")
                self.component_table.setRowCount(0)
            else:
                analysis = state.roi_analysis
                self.summary.setHtml(
                    (
                        f"<h3>{analysis.label}</h3>"
                        f"<p><strong>ROI:</strong> {analysis.roi_bounds_keV[0]:.3f}-{analysis.roi_bounds_keV[1]:.3f} keV<br/>"
                        f"<strong>Gross:</strong> {analysis.gross_counts:.3f} ± {analysis.gross_counts_uncertainty:.3f}<br/>"
                        f"<strong>Background:</strong> {analysis.background_counts:.3f} ± {analysis.background_counts_uncertainty:.3f} ({analysis.background_method})<br/>"
                        f"<strong>Net:</strong> {analysis.net_counts:.3f} ± {analysis.net_counts_uncertainty:.3f}<br/>"
                        f"<strong>Centroid:</strong> {analysis.centroid_keV:.3f} ± {analysis.centroid_uncertainty_keV:.3f} keV<br/>"
                        f"<strong>Significance:</strong> {analysis.significance:.2f}σ<br/>"
                        f"<strong>Peak search:</strong> {analysis.peak_search_method}<br/>"
                        f"<strong>Notes:</strong> {' '.join(analysis.notes)}</p>"
                    )
                )
                self.component_table.setRowCount(len(analysis.overlap_components))
                for row, component in enumerate(analysis.overlap_components):
                    values = (
                        f"{component.centroid_keV:.3f}",
                        f"{component.net_counts:.3f}",
                        f"{component.net_counts_uncertainty:.3f}",
                        f"{component.fwhm_channels:.3f}",
                        f"{component.reduced_chi_squared:.3f}",
                    )
                    for column, value in enumerate(values):
                        item = QTableWidgetItem(value)
                        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                        self.component_table.setItem(row, column, item)

            stats = state.roi_statistics
            if stats is None:
                self.statistics_table.setRowCount(0)
                return
            self.statistics_table.setRowCount(len(stats.samples))
            for row, sample in enumerate(stats.samples):
                values = (
                    sample.label,
                    f"{sample.net_counts:.3f}",
                    f"{sample.net_counts_uncertainty:.3f}",
                    f"{sample.centroid_keV:.3f}",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    self.statistics_table.setItem(row, column, item)
            self.summary.append(
                (
                    f"<p><strong>ROI stats:</strong> n={stats.sample_count} | "
                    f"mean net {stats.mean_net_counts:.3f} | "
                    f"σ {stats.stdev_net_counts:.3f} | "
                    f"RSD {stats.relative_std * 100.0:.2f}%</p>"
                )
            )


    class SurveyMapPanel(QWidget):
        """Survey map summary for spectra carrying GPS metadata."""

        def __init__(
            self,
            *,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.workspace_controller = workspace_controller

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Survey coordinates extracted from loaded spectra are surfaced here "
                    "for offline review and report generation."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.browser = QTextBrowser(self)
            self.browser.setObjectName("SurveyMapBrowser")
            layout.addWidget(self.browser, 1)

            self.workspace_controller.subscribe(self._sync_state)
            self._sync_state(self.workspace_controller.state)

        def _sync_state(self, state) -> None:
            if not state.survey_points:
                self.browser.setHtml("<p>No GPS-enabled spectra are loaded.</p>")
                return
            items = []
            for point in state.survey_points:
                items.append(
                    f"<li><strong>{point.label}</strong> ({point.source_role}) "
                    f"@ {point.latitude:.5f}, {point.longitude:.5f}</li>"
                )
            self.browser.setHtml(
                "<h3>Survey Map Points</h3><ul>" + "".join(items) + "</ul>"
            )


    from fluxforge.gui.panels.modern_shell_center import CentralWorkspaceTabs

    from fluxforge.gui.panels.modern_shell_sidebar import SidebarPanel

    class BatchQueuePanel(QWidget):
        """Queued batch-analysis panel backed by ProcessPoolExecutor workers."""

        def __init__(
            self,
            *,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.workspace_controller = workspace_controller
            self.queued_jobs: tuple[BatchAnalysisJob, ...] = ()
            self.results = ()
            self.last_output_dir = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Run loaded spectra through the batch queue, then emit per-spectrum JSON "
                    "artifacts plus one aggregate CSV."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            action_row = QHBoxLayout()
            self.queue_button = QPushButton("Queue Loaded Spectra", self)
            self.queue_button.setObjectName("QueueLoadedSpectraButton")
            self.queue_button.clicked.connect(self.queue_loaded_spectra)
            action_row.addWidget(self.queue_button)

            self.run_button = QPushButton("Run Batch Queue", self)
            self.run_button.setObjectName("RunBatchQueueButton")
            self.run_button.clicked.connect(self.run_queue)
            action_row.addWidget(self.run_button)

            self.prefer_gpu_checkbox = QCheckBox("Prefer GPU backend", self)
            self.prefer_gpu_checkbox.setObjectName("BatchPreferGpuCheck")
            action_row.addWidget(self.prefer_gpu_checkbox)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self.output_dir = QLineEdit(self)
            self.output_dir.setObjectName("BatchOutputDirectoryInput")
            self.output_dir.setText("artifacts/batch_analysis")
            layout.addWidget(self.output_dir)

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setObjectName("BatchQueueProgressBar")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            self.table = QTableWidget(0, 6, self)
            self.table.setObjectName("BatchQueueResultsTable")
            self.table.setHorizontalHeaderLabels(
                ("Label", "Counts", "σ Counts", "Peaks", "Dominant Nuclide", "Backend")
            )
            self.table.verticalHeader().setVisible(False)
            layout.addWidget(self.table, 1)

            self.summary = QLabel("No queued spectra.", self)
            self.summary.setObjectName("PanelBody")
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)

        def queue_loaded_spectra(self) -> None:
            jobs: list[BatchAnalysisJob] = []
            records = self.workspace_controller.loaded_spectrum_records()
            if records:
                for index, record in enumerate(records):
                    jobs.append(
                        BatchAnalysisJob(
                            job_id=f"batch-{index + 1}",
                            label=record.label,
                            spectrum=record.spectrum,
                        )
                    )
            else:
                for index, slot in enumerate(self.workspace_controller.spectrum_slots()):
                    jobs.append(
                        BatchAnalysisJob(
                            job_id=f"slot-{slot.key}",
                            label=slot.source_label or slot.label,
                            spectrum=slot.spectrum,
                        )
                    )
            self.queued_jobs = tuple(jobs)
            self.progress_bar.setValue(0)
            self.summary.setText(f"Queued {len(self.queued_jobs)} spectra for batch analysis.")

        def run_queue(self) -> None:
            if not self.queued_jobs:
                self.queue_loaded_spectra()
            if not self.queued_jobs:
                self.summary.setText("No spectra are available for the batch queue.")
                return

            def _progress_update(completed: int, total: int) -> None:
                if total <= 0:
                    self.progress_bar.setValue(0)
                    return
                percent = int(round(100.0 * float(completed) / float(total)))
                self.progress_bar.setValue(max(0, min(percent, 100)))
                self.summary.setText(
                    f"Processed {completed} of {total} spectra in the batch queue."
                )
                QApplication.processEvents()

            self.results = run_batch_analysis_queue(
                self.queued_jobs,
                max_workers=1,
                prefer_gpu=self.prefer_gpu_checkbox.isChecked(),
                progress_callback=_progress_update,
            )
            output_dir = self.output_dir.text().strip() or "artifacts/batch_analysis"
            aggregate_path, _json_paths = write_batch_outputs(self.results, output_dir)
            self.last_output_dir = aggregate_path.parent
            self.table.setRowCount(len(self.results))
            for row, result in enumerate(self.results):
                values = (
                    result.label,
                    result.total_counts,
                    result.total_uncertainty,
                    result.peak_count,
                    result.dominant_nuclide or "N/A",
                    result.backend,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(
                        value if isinstance(value, str) else f"{float(value):.4f}".rstrip("0").rstrip(".")
                    )
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.table.setItem(row, column, item)
            self.progress_bar.setValue(100)
            self.summary.setText(
                f"Processed {len(self.results)} spectra. Aggregate CSV: {aggregate_path}"
            )


    class BottomWorkspaceTabs(QTabWidget):
        """Bottom-zone tab set for tables, ROI tools, calibration, activity, batch, and logs."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager,
            undo_stack: QUndoStack | None = None,
            open_calibration_workspace: Callable[[], None] | None = None,
            open_quick_slider_calibration_workspace: Callable[[], None] | None = None,
            open_manual_calibration_workspace: Callable[[], None] | None = None,
            open_standards_calibration_workspace: Callable[[], None] | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self.undo_stack = undo_stack
            self._open_calibration_workspace = open_calibration_workspace
            self._open_quick_slider_calibration_workspace = (
                open_quick_slider_calibration_workspace
            )
            self._open_manual_calibration_workspace = open_manual_calibration_workspace
            self._open_standards_calibration_workspace = open_standards_calibration_workspace
            self.peak_table_panel = PeakTablePanel(
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                workspace_controller=self.workspace_controller,
                library_manager=self.library_manager,
                undo_stack=self.undo_stack,
                parent=self,
            )
            self.addTab(self.peak_table_panel, "Peak Table")
            self.roi_tools_panel = RoiToolsPanel(
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                workspace_controller=self.workspace_controller,
                parent=self,
            )
            self.addTab(self.roi_tools_panel, "ROI Tools")
            self.addTab(
                self._build_calibration_panel(),
                "Calibration",
            )
            self.activity_results_panel = ActivityResultsPanel(
                mode_manager=self.mode_manager,
                selection_bus=self.selection_bus,
                workspace_controller=self.workspace_controller,
                library_manager=self.library_manager,
                parent=self,
            )
            self.addTab(self.activity_results_panel, "Activity Results")
            self.inventory_timeline_panel = InventoryTimelinePanel(
                mode_manager=self.mode_manager,
                workspace_controller=self.workspace_controller,
                library_manager=self.library_manager,
                parent=self,
            )
            self.addTab(self.inventory_timeline_panel, "Inventory / Time Evolution")
            self.masking_review_panel = MaskingReviewPanel(
                mode_manager=self.mode_manager,
                activity_review_provider=self.activity_results_panel.current_activity_review,
                activity_results_provider=lambda: self.workspace_controller.state.activity_results,
                parent=self,
            )
            self.addTab(self.masking_review_panel, "Line Interference / Masking")
            self.optimization_workspace_panel = OptimizationWorkspacePanel(
                mode_manager=self.mode_manager,
                activity_review_provider=self.activity_results_panel.current_activity_review,
                activity_results_provider=lambda: self.workspace_controller.state.activity_results,
                parent=self,
            )
            self.addTab(self.optimization_workspace_panel, "Irradiation Optimizer")
            self.second_irradiation_panel = SecondIrradiationPlannerPanel(
                mode_manager=self.mode_manager,
                activity_review_provider=self.activity_results_panel.current_activity_review,
                activity_results_provider=lambda: self.workspace_controller.state.activity_results,
                parent=self,
            )
            self.addTab(self.second_irradiation_panel, "Second Irradiation")
            self.batch_queue_panel = BatchQueuePanel(
                workspace_controller=self.workspace_controller,
                parent=self,
            )
            self.addTab(self.batch_queue_panel, "Batch Queue")
            self.addTab(
                self._text_panel(
                    "Spectrogram",
                    "Reserved for the future time-energy color-map view and interim multi-spectrum history heatmap.",
                ),
                "Spectrogram",
            )
            self.survey_map_panel = SurveyMapPanel(
                workspace_controller=self.workspace_controller,
                parent=self,
            )
            self.addTab(self.survey_map_panel, "Survey Map")
            self.addTab(self._log_panel(), "Log")

        def _text_panel(self, heading: str, body: str) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(16, 16, 16, 16)
            title = QLabel(heading, widget)
            title.setObjectName("PanelHeading")
            layout.addWidget(title)
            text = QLabel(body, widget)
            text.setWordWrap(True)
            text.setObjectName("PanelBody")
            layout.addWidget(text)
            layout.addStretch(1)
            return widget

        def _log_panel(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            self.log = QPlainTextEdit(widget)
            self.log.setReadOnly(True)
            self.log.setObjectName("RunLog")
            self.log.setPlainText("\n".join(MODERN_LOG_LINES))
            layout.addWidget(self.log)
            self.selection_bus.subscribe(self._append_selection_event)
            self.mode_manager.subscribe(self._append_mode_event)
            return widget

        def _build_calibration_panel(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QLabel("Calibration", widget)
            title.setObjectName("PanelHeading")
            layout.addWidget(title)

            body = QLabel(
                (
                    "Calibration now uses the dedicated Qt workspace rather "
                    "than expanding the old bottom-tab editor."
                ),
                widget,
            )
            body.setObjectName("PanelBody")
            body.setWordWrap(True)
            layout.addWidget(body)

            self.calibration_mode_note = QLabel(widget)
            self.calibration_mode_note.setObjectName("HeroCardAccent")
            self.calibration_mode_note.setWordWrap(True)
            layout.addWidget(self.calibration_mode_note)

            action_row = QHBoxLayout()
            action_row.setSpacing(10)

            manual_button = QPushButton("Manual Workflow", widget)
            manual_button.setObjectName("ManualCalibrationWorkflowButton")
            manual_button.setEnabled(self._open_manual_calibration_workspace is not None)
            if self._open_manual_calibration_workspace is not None:
                manual_button.clicked.connect(self._open_manual_calibration_workspace)
            action_row.addWidget(manual_button)

            quick_button = QPushButton("Quick Slider", widget)
            quick_button.setObjectName("QuickSliderCalibrationWorkflowButton")
            quick_button.setEnabled(
                self._open_quick_slider_calibration_workspace is not None
            )
            if self._open_quick_slider_calibration_workspace is not None:
                quick_button.clicked.connect(self._open_quick_slider_calibration_workspace)
            action_row.addWidget(quick_button)

            standards_button = QPushButton("Standards Workflow", widget)
            standards_button.setObjectName("StandardsCalibrationWorkflowButton")
            standards_button.setEnabled(
                self._open_standards_calibration_workspace is not None
            )
            if self._open_standards_calibration_workspace is not None:
                standards_button.clicked.connect(self._open_standards_calibration_workspace)
            action_row.addWidget(standards_button)

            launch_button = QPushButton("Open Shared Workspace", widget)
            launch_button.setObjectName("SharedCalibrationWorkspaceButton")
            launch_button.setEnabled(self._open_calibration_workspace is not None)
            if self._open_calibration_workspace is not None:
                launch_button.clicked.connect(self._open_calibration_workspace)
            action_row.addWidget(launch_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)
            layout.addStretch(1)

            self.mode_manager.subscribe(self._sync_calibration_note)
            self._sync_calibration_note(self.mode_manager.state)
            return widget

        def _append_selection_event(self, state: SelectionState) -> None:
            self.log.appendPlainText("Selection → " + _selection_summary(state))

        def _append_mode_event(self, state) -> None:
            summary = state.mode.value.title()
            if state.standard:
                summary += f" / {state.standard}"
            self.log.appendPlainText("Mode → " + summary)

        def run_auto_peak_search(self) -> None:
            self.setCurrentWidget(self.peak_table_panel)
            self.peak_table_panel.run_auto_peak_search()

        def _sync_calibration_note(self, state) -> None:
            if state.mode is GUIMode.STANDARDS and state.standard:
                self.calibration_mode_note.setText(
                    f"Standards lock active: {state.standard} forces the energy fit to order 2."
                )
                return
            self.calibration_mode_note.setText(
                "Expert mode keeps the order selection editable while the ASTM acceptance band remains visible."
            )

        def workflow_state(self) -> dict[str, object]:
            return {
                "current_tab": current_tab_label(self),
                "activity_results": self.activity_results_panel.workflow_state(),
                "inventory_timeline": self.inventory_timeline_panel.workflow_state(),
                "masking_review": self.masking_review_panel.workflow_state(),
                "optimization_workspace": self.optimization_workspace_panel.workflow_state(),
                "second_irradiation": self.second_irradiation_panel.workflow_state(),
            }

        def apply_workflow_state(self, payload: dict[str, object] | None) -> None:
            if not payload:
                return
            activity_payload = payload.get("activity_results")
            if isinstance(activity_payload, dict):
                self.activity_results_panel.apply_workflow_state(activity_payload)
            inventory_payload = payload.get("inventory_timeline")
            if isinstance(inventory_payload, dict):
                self.inventory_timeline_panel.apply_workflow_state(inventory_payload)
            masking_payload = payload.get("masking_review")
            if isinstance(masking_payload, dict):
                self.masking_review_panel.apply_workflow_state(masking_payload)
            optimization_payload = payload.get("optimization_workspace")
            if isinstance(optimization_payload, dict):
                self.optimization_workspace_panel.apply_workflow_state(
                    optimization_payload
                )
            second_payload = payload.get("second_irradiation")
            if isinstance(second_payload, dict):
                self.second_irradiation_panel.apply_workflow_state(second_payload)
            if "current_tab" in payload:
                set_tab_label(self, str(payload["current_tab"]))


    from fluxforge.gui.panels.modern_shell_context import ToolContextPanel

else:

    class CentralWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.parent = parent
            self._current_spectrum = build_demo_spectrum()

        def current_spectrum(self):
            return self._current_spectrum


    class SidebarPanel:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager | None = None,
            qa_monitor: QAMonitor | None = None,
            open_qa_history: Callable[[], None] | None = None,
            open_standards_review: Callable[[], None] | None = None,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self.qa_monitor = qa_monitor
            self._open_qa_history = open_qa_history
            self._open_standards_review = open_standards_review
            self.parent = parent


    class BottomWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            library_manager: DataLibraryManager,
            undo_stack=None,
            open_calibration_workspace: Callable[[], None] | None = None,
            open_quick_slider_calibration_workspace: Callable[[], None] | None = None,
            open_manual_calibration_workspace: Callable[[], None] | None = None,
            open_standards_calibration_workspace: Callable[[], None] | None = None,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self.undo_stack = undo_stack
            self.open_calibration_workspace = open_calibration_workspace
            self.open_quick_slider_calibration_workspace = open_quick_slider_calibration_workspace
            self.open_manual_calibration_workspace = open_manual_calibration_workspace
            self.open_standards_calibration_workspace = open_standards_calibration_workspace
            self.parent = parent


    class ToolContextPanel:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.parent = parent
