"""Modern panels for the next-generation GUI shell."""

from __future__ import annotations

from typing import Callable

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.phase2_analysis import (
    PeakCandidate,
    bayesian_match_peak_candidates,
    calculate_peak_activity,
    compute_cascade_sum_lines,
    detect_peak_candidates,
    extract_survey_points,
    subtract_background_counts,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, pyqtgraph_backend_status
from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.nuclide_search import NuclideSearchController
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.phase2_workspace import Phase2WorkspaceController, SpectrumSlot, WorkspaceStateCommand
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import SpectrumTrace
from fluxforge.io.spe import GammaSpectrum

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.backends import PyQtGraphSpectrumCanvas
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
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
        QPushButton,
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
    return " | ".join(fragments) if fragments else "No active selection"


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
    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        channels=np.asarray(channels, dtype=float),
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="demo_hpge_workspace",
        detector_id="demo-hpge",
        gps={"latitude": 43.0731, "longitude": -89.4012},
        metadata={"source": "phase2-demo", "gps": {"latitude": 43.0731, "longitude": -89.4012}},
    )


def build_demo_background_spectrum() -> GammaSpectrum:
    """Return the background companion spectrum for the Phase 2 demo shell."""

    counts = np.asarray(_demo_counts(), dtype=float) * 0.16
    channels = np.arange(len(counts), dtype=float)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="demo_hpge_background",
        detector_id="demo-hpge",
        gps={"latitude": 43.0736, "longitude": -89.4019},
        metadata={"source": "phase2-demo-background", "gps": {"latitude": 43.0736, "longitude": -89.4019}},
    )


def build_demo_overlay_spectrum() -> GammaSpectrum:
    """Return the secondary overlay companion spectrum for the Phase 2 demo shell."""

    counts = np.asarray(_demo_counts(), dtype=float) * 0.62
    counts[540:620] *= 1.18
    channels = np.arange(len(counts), dtype=float)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="demo_hpge_overlay",
        detector_id="demo-hpge",
        gps={"latitude": 43.0742, "longitude": -89.4024},
        metadata={"source": "phase2-demo-overlay", "gps": {"latitude": 43.0742, "longitude": -89.4024}},
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
            workspace_controller: Phase2WorkspaceController,
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

            self.match_button = QPushButton("Bayesian Match", self)
            self.match_button.setObjectName("BayesianMatchPeaksButton")
            self.match_button.clicked.connect(self.run_bayesian_match)
            action_row.addWidget(self.match_button)

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

            self.workspace_controller.subscribe(self._sync_state)
            self._sync_state(self.workspace_controller.state)

        def run_auto_peak_search(self) -> None:
            spectrum = self.workspace_controller.spectrum()
            if spectrum is None:
                self.summary.setText("No active spectrum is available for peak search.")
                return
            peaks = detect_peak_candidates(spectrum)
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
                    }
                ),
            )

        def run_bayesian_match(self) -> None:
            state = self.workspace_controller.state
            if not state.peaks:
                self.summary.setText("Run peak search before Bayesian matching.")
                return
            matched = bayesian_match_peak_candidates(
                state.peaks,
                source_id=self.library_manager.state.gamma_identification_source_id,
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

        def _pin_selected_nuclide(self) -> None:
            peak = self.workspace_controller.selected_peak()
            if peak is None or not peak.nuclide:
                return
            state = self.workspace_controller.state
            pinned = list(state.pinned_nuclides)
            if peak.nuclide not in pinned:
                pinned.append(peak.nuclide)
            cascade_sum_lines_keV = compute_cascade_sum_lines(
                pinned,
                source_id=self.library_manager.state.gamma_identification_source_id,
                custom_path=self.library_manager.state.custom_gamma_path,
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
                self.summary.setText(
                    f"{len(state.peaks)} peaks tracked. {matched} currently matched. "
                    f"Pinned nuclides: {', '.join(state.pinned_nuclides) or 'none'}."
                )
            else:
                self.summary.setText("No peaks have been added yet.")

        def _publish_selected_peak(self) -> None:
            row = self.table.currentRow()
            if row < 0 or row >= len(self.workspace_controller.state.peaks):
                return
            peak = self.workspace_controller.state.peaks[row]
            self.workspace_controller.select_peak(peak.peak_id)
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=peak.energy_keV,
                    roi_bounds_keV=peak.roi_bounds_keV,
                    nuclide=peak.nuclide,
                    reference_lines_keV=peak.reference_lines_keV,
                )
            )


    class ActivityResultsPanel(QWidget):
        """Efficiency, activity, source age, and background workflow surface."""

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
            library_manager: DataLibraryManager,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Activity calculations consume the selected peak, the active efficiency "
                    "fit, the chosen data library, and the current background mode."
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
            self.source_age_hours.setSuffix(" h age")
            control_row.addWidget(self.source_age_hours)
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

        def _compute_activity(self) -> None:
            peak = self.workspace_controller.selected_peak()
            fit = self.workspace_controller.state.efficiency_fit
            spectrum = self.workspace_controller.spectrum()
            if peak is None or fit is None or spectrum is None:
                self.summary.setText(
                    "Select a peak and fit an efficiency curve before computing activity."
                )
                return
            result = calculate_peak_activity(
                peak,
                spectrum,
                efficiency_curve=fit.curve,
                gamma_intensity=1.0,
                half_life_s=5.27 * 365.25 * 24.0 * 3600.0 if peak.nuclide and "co60" in peak.nuclide.lower() else 30.17 * 365.25 * 24.0 * 3600.0,
                source_age_s=float(self.source_age_hours.value()) * 3600.0,
            )
            self.workspace_controller.set_activity_results((result,))

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
                self.summary.setText(
                    f"{state.efficiency_fit.model_label} active. RMSE {state.efficiency_fit.rmse:.6f}."
                )
            if state.activity_results:
                result = state.activity_results[0]
                self.results.setPlainText(
                    "\n".join(
                        [
                            f"Nuclide: {result.nuclide}",
                            f"Line: {result.line_energy_keV:.3f} keV",
                            f"Activity: {result.activity_bq:.6g} Bq ± {result.uncertainty_bq:.3g}",
                            f"Age corrected: {result.age_corrected_activity_bq:.6g} Bq",
                            f"MDA: {result.mda_bq:.6g} Bq",
                            result.chain_summary,
                        ]
                    )
                )
            else:
                self.results.setPlainText("No activity result has been calculated yet.")


    class SurveyMapPanel(QWidget):
        """Survey map summary for spectra carrying GPS metadata."""

        def __init__(
            self,
            *,
            workspace_controller: Phase2WorkspaceController,
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


    class CentralWorkspaceTabs(QTabWidget):
        """Center-zone tab stack with the modern spectrum and survey surfaces."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self._current_spectrum = workspace_controller.spectrum() or build_demo_spectrum()
            self.setObjectName("CentralWorkspaceTabs")
            self.addTab(self._build_spectrum_tab(), "Spectrum")
            self.addTab(self._build_dashboard_tab(), "Dashboard")
            self.mode_manager.subscribe(self._sync_mode_banner)
            self.mode_manager.subscribe(self._sync_workspace_mode)
            self.workspace_controller.subscribe(self._sync_workspace_state)
            self._sync_mode_banner(self.mode_manager.state)
            self._sync_workspace_state(self.workspace_controller.state)

        def _build_spectrum_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(18)

            self.standards_banner = QLabel(widget)
            self.standards_banner.setObjectName("StandardsBanner")
            self.standards_banner.setVisible(False)
            layout.addWidget(self.standards_banner)

            hero = QFrame(widget)
            hero.setObjectName("HeroCanvas")
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(28, 28, 28, 28)
            hero_layout.setSpacing(10)

            eyebrow = QLabel("FluxForge Next", hero)
            eyebrow.setObjectName("HeroEyebrow")
            hero_layout.addWidget(eyebrow)

            title = QLabel("Native, dockable HPGe workspace", hero)
            title.setObjectName("HeroHeader")
            hero_layout.addWidget(title)

            subtitle = QLabel(
                (
                    "PySide6 shell with a PyQtGraph-first spectrum canvas, standards-aware "
                    "workflow modes, and a clearly separated legacy fallback."
                ),
                hero,
            )
            subtitle.setWordWrap(True)
            subtitle.setObjectName("HeroSubhead")
            hero_layout.addWidget(subtitle)
            layout.addWidget(hero)

            self.spectrum_slot_tabs = QTabBar(widget)
            self.spectrum_slot_tabs.setObjectName("CanvasSpectrumTabs")
            self.spectrum_slot_tabs.currentChanged.connect(self._slot_tab_changed)
            layout.addWidget(self.spectrum_slot_tabs)

            body = QHBoxLayout()
            body.setSpacing(18)
            layout.addLayout(body, 1)

            if PYQTGRAPH_AVAILABLE:
                self.canvas = PyQtGraphSpectrumCanvas(
                    selection_bus=self.selection_bus,
                    parent=widget,
                )
                body.addWidget(self.canvas, 3)
            else:
                status = pyqtgraph_backend_status()
                reason = status["reason"] or "Install the native GUI extras."
                body.addWidget(
                    _card(
                        "Renderer pending optional extras",
                        "The production canvas is wired, but this local environment does not have PySide6 + PyQtGraph installed.",
                        f"Import status: {reason}",
                    ),
                    3,
                )

            rail = QWidget(widget)
            rail_layout = QVBoxLayout(rail)
            rail_layout.setContentsMargins(0, 0, 0, 0)
            rail_layout.setSpacing(18)
            rail_layout.addWidget(
                _card(
                    "Visual Feedback First",
                    "Peak fits, calibration, results, and standards context stay visible around the canvas instead of hiding behind modal-only flows.",
                    "Design target: bGamma polish with InterSpec-grade canvas interaction.",
                )
            )
            rail_layout.addWidget(
                _card(
                    "Shared Analytical State",
                    "SelectionBus synchronizes the peak table, sidebar, and tool inspector around the same active ROI or nuclide.",
                    "Current shell wiring already reflects cross-panel selection state.",
                )
            )
            rail_layout.addWidget(
                _card(
                    "Offline-First Reporting",
                    "Session provenance, native exports, and report templates stay local and reproducible.",
                    "Roadmap target: Jinja2 templates with residuals embedded by default.",
                )
            )
            rail_layout.addStretch(1)
            body.addWidget(rail, 2)

            return widget

        def load_spectrum(
            self,
            spectrum,
            *,
            slot_key: str | None = None,
            source_key: str | None = None,
            source_label: str | None = None,
            source_path: str | None = None,
        ) -> None:
            """Load a GammaSpectrum-like object into the primary canvas."""

            self._current_spectrum = spectrum
            target_slot = slot_key or self.workspace_controller.state.active_spectrum_key
            self.workspace_controller.replace_spectrum_slot(
                target_slot,
                spectrum,
                source_key=source_key,
                source_label=source_label,
                source_path=source_path,
            )

        def current_spectrum(self):
            """Return the current spectrum object shown in the central workspace."""

            return self.workspace_controller.spectrum() or self._current_spectrum

        def _slot_tab_changed(self, index: int) -> None:
            if index < 0 or index >= len(self.workspace_controller.state.spectra):
                return
            self.workspace_controller.select_spectrum(
                self.workspace_controller.state.spectra[index].key
            )

        def _build_dashboard_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(18)

            title = QLabel("Digital twin hardware dashboard", widget)
            title.setObjectName("HeroHeader")
            layout.addWidget(title)

            subtitle = QLabel(
                "Reserved for MCA device discovery, status telemetry, live acquisition, and the future spectrogram surface.",
                widget,
            )
            subtitle.setWordWrap(True)
            subtitle.setObjectName("HeroSubhead")
            layout.addWidget(subtitle)

            cards = QGridLayout()
            cards.setHorizontalSpacing(18)
            cards.setVerticalSpacing(18)
            cards.addWidget(
                _card(
                    "Device Cards",
                    "Auto-discovery, live rate, and health telemetry will be surfaced as native cards instead of hidden in dialogs.",
                    "Current phase: layout reserved to avoid later shell rewrites.",
                ),
                0,
                0,
            )
            cards.addWidget(
                _card(
                    "Acquisition Timeline",
                    "The Dashboard tab will host count-rate history and the future spectrogram view without displacing the spectrum tab.",
                    "Current phase: placeholder only, but the tab is permanent.",
                ),
                0,
                1,
            )
            layout.addLayout(cards)
            layout.addStretch(1)
            return widget

        def _sync_workspace_state(self, state) -> None:
            self._current_spectrum = self.workspace_controller.spectrum() or self._current_spectrum
            self._sync_slot_tabs(state)
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return

            foreground_slot = None
            background_slot = None
            overlay_slot = None
            for slot in state.spectra:
                if slot.key == state.active_spectrum_key:
                    foreground_slot = slot
                elif slot.key == "background":
                    background_slot = slot
                elif slot.key == "overlay":
                    overlay_slot = slot
            foreground = (
                foreground_slot.spectrum if foreground_slot is not None else self._current_spectrum
            )
            background = background_slot.spectrum if background_slot is not None else None
            overlay = overlay_slot.spectrum if overlay_slot is not None else None
            traces: list[SpectrumTrace] = []
            if foreground is not None:
                primary_counts = np.asarray(foreground.counts, dtype=float)
                if (
                    background is not None
                    and state.active_spectrum_key != "background"
                    and state.background_mode in {"simple", "scaled", "statistical"}
                ):
                    primary_counts = subtract_background_counts(
                        foreground,
                        background,
                        mode=state.background_mode,
                        scale=state.background_scale,
                    )
                traces.append(
                    SpectrumTrace(
                        label=(
                            foreground_slot.source_label
                            if foreground_slot is not None and foreground_slot.source_label
                            else "Foreground"
                        ),
                        counts=tuple(float(value) for value in primary_counts),
                        channels=tuple(float(value) for value in np.asarray(foreground.channels, dtype=float)),
                        color="#72d6ff",
                    )
                )
            if background is not None and state.background_visible:
                traces.append(
                    SpectrumTrace(
                        label=(
                            background_slot.source_label
                            if background_slot is not None and background_slot.source_label
                            else "Background"
                        ),
                        counts=tuple(float(value) for value in np.asarray(background.counts, dtype=float)),
                        channels=tuple(float(value) for value in np.asarray(background.channels, dtype=float)),
                        color="#f59e0b",
                    )
                )
            if overlay is not None:
                traces.append(
                    SpectrumTrace(
                        label=(
                            overlay_slot.source_label
                            if overlay_slot is not None and overlay_slot.source_label
                            else "Secondary Overlay"
                        ),
                        counts=tuple(float(value) for value in np.asarray(overlay.counts, dtype=float)),
                        channels=tuple(float(value) for value in np.asarray(overlay.channels, dtype=float)),
                        color="#10b981",
                    )
                )
            self.canvas.set_traces(traces)
            self.canvas.set_peak_candidates(state.peaks)
            self.canvas.set_cascade_sum_lines(state.cascade_sum_lines_keV)
            self.canvas.set_peak_residuals(
                state.peaks[:3],
                visible=self.mode_manager.state.mode is not GUIMode.SIMPLE,
            )
            if state.peaks and state.selected_peak_id:
                selected_peak = next(
                    (peak for peak in state.peaks if peak.peak_id == state.selected_peak_id),
                    None,
                )
                if selected_peak is not None:
                    self.selection_bus.publish(
                        SelectionState(
                            peak_energy_keV=selected_peak.energy_keV,
                            roi_bounds_keV=selected_peak.roi_bounds_keV,
                            nuclide=selected_peak.nuclide,
                            reference_lines_keV=selected_peak.reference_lines_keV,
                        )
                    )

        def _sync_slot_tabs(self, state) -> None:
            self.spectrum_slot_tabs.blockSignals(True)
            while self.spectrum_slot_tabs.count():
                self.spectrum_slot_tabs.removeTab(0)
            current_index = 0
            for index, slot in enumerate(state.spectra):
                self.spectrum_slot_tabs.addTab(slot.label)
                self.spectrum_slot_tabs.setTabData(index, slot.key)
                if slot.key == state.active_spectrum_key:
                    current_index = index
            self.spectrum_slot_tabs.setCurrentIndex(current_index)
            self.spectrum_slot_tabs.blockSignals(False)

        def _sync_workspace_mode(self, state) -> None:
            if PYQTGRAPH_AVAILABLE and hasattr(self, "canvas"):
                self.canvas.set_peak_residuals(
                    self.workspace_controller.state.peaks[:3],
                    visible=state.mode is not GUIMode.SIMPLE,
                )

        def _sync_mode_banner(self, state) -> None:
            locked = state.mode is GUIMode.STANDARDS and state.standard
            self.standards_banner.setVisible(bool(locked))
            if locked:
                self.standards_banner.setText(
                    f"Standards mode locked to {state.standard}. Alternate methods remain available in Expert mode."
                )


    class SidebarPanel(QWidget):
        """Left-side shell for files, devices, libraries, results, and QA."""

        def __init__(
            self,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
            library_manager: DataLibraryManager | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager or DataLibraryManager()
            self.nuclide_controller = NuclideSearchController(
                selection_bus,
                library_manager=self.library_manager,
            )

            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            self.files = QTreeWidget(self)
            self.files.setHeaderLabels(["Loaded Spectra", "Assigned Roles"])
            self.files.setObjectName("SidebarTree")
            self.files.setSelectionMode(QAbstractItemView.SingleSelection)
            layout.addWidget(self.files, 2)

            layout.addWidget(self._build_spectrum_role_panel(), 2)

            layout.addWidget(self._build_library_panel(), 2)

            devices = QListWidget(self)
            devices.setObjectName("SidebarList")
            for text in ("Mock MCA · Offline", "Dashboard reserved", "Spectrogram planned"):
                QListWidgetItem(text, devices)
            layout.addWidget(devices, 1)

            self.nuclide_query = QLineEdit(self)
            self.nuclide_query.setObjectName("NuclideSearchInput")
            self.nuclide_query.setPlaceholderText("Nuclide search...")
            layout.addWidget(self.nuclide_query)

            self.nuclides = QListWidget(self)
            self.nuclides.setObjectName("SidebarList")
            layout.addWidget(self.nuclides, 1)
            self._refresh_nuclide_results("cs")

            pin_row = QHBoxLayout()
            self.pin_selected_nuclide_button = QPushButton("Pin Selected Nuclide", self)
            self.pin_selected_nuclide_button.clicked.connect(self._pin_selected_nuclide)
            pin_row.addWidget(self.pin_selected_nuclide_button)
            pin_row.addStretch(1)
            layout.addLayout(pin_row)

            self.pinned_nuclides = QListWidget(self)
            self.pinned_nuclides.setObjectName("PinnedNuclidesList")
            layout.addWidget(self.pinned_nuclides, 1)

            self.selection_note = QTextEdit(self)
            self.selection_note.setObjectName("SidebarNote")
            self.selection_note.setReadOnly(True)
            self.selection_note.setPlainText(
                "Selection sync\n\nNo active selection"
            )
            layout.addWidget(self.selection_note, 1)

            qa = QTextEdit(self)
            qa.setObjectName("SidebarNote")
            qa.setReadOnly(True)
            qa.setPlainText(
                "QA & Standards\n\nASTM status dots, drift alerts, and standards locking details will be surfaced here."
            )
            layout.addWidget(qa, 1)

            self.selection_bus.subscribe(self._sync_selection)
            self.library_manager.subscribe(self._sync_library_state)
            self.workspace_controller.subscribe(self._sync_workspace_state)
            self.nuclide_query.textChanged.connect(self._refresh_nuclide_results)
            self.nuclides.itemSelectionChanged.connect(self._activate_selected_nuclide)
            self.custom_gamma_path.editingFinished.connect(self._apply_custom_gamma_path)
            self.gamma_source_combo.currentIndexChanged.connect(self._gamma_source_changed)
            self.calibration_source_combo.currentIndexChanged.connect(
                self._calibration_source_changed
            )
            self.foreground_spectrum_combo.currentIndexChanged.connect(
                self._foreground_spectrum_changed
            )
            self.background_spectrum_combo.currentIndexChanged.connect(
                self._background_spectrum_changed
            )
            self.overlay_spectrum_combo.currentIndexChanged.connect(
                self._overlay_spectrum_changed
            )
            self.naa_source_combo.currentIndexChanged.connect(self._naa_source_changed)
            self.dosimetry_source_combo.currentIndexChanged.connect(
                self._dosimetry_source_changed
            )
            self.activation_source_combo.currentIndexChanged.connect(
                self._activation_source_changed
            )
            self._populate_library_combos()
            self._sync_library_state(self.library_manager.state)
            self._sync_workspace_state(self.workspace_controller.state)

        def _build_spectrum_role_panel(self) -> QWidget:
            group = QGroupBox("Spectrum Roles", self)
            group.setObjectName("SidebarSpectrumRolePanel")
            layout = QVBoxLayout(group)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Open or drag spectra into the modern Qt shell, then assign them to the "
                    "foreground, background, or overlay lanes here. Background subtraction and "
                    "overlay plotting update directly from these selectors."
                ),
                group,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            layout.addWidget(QLabel("Foreground spectrum", group))
            self.foreground_spectrum_combo = QComboBox(group)
            self.foreground_spectrum_combo.setObjectName("ForegroundSpectrumSelector")
            layout.addWidget(self.foreground_spectrum_combo)

            layout.addWidget(QLabel("Background spectrum", group))
            self.background_spectrum_combo = QComboBox(group)
            self.background_spectrum_combo.setObjectName("BackgroundSpectrumSelector")
            layout.addWidget(self.background_spectrum_combo)

            layout.addWidget(QLabel("Overlay spectrum", group))
            self.overlay_spectrum_combo = QComboBox(group)
            self.overlay_spectrum_combo.setObjectName("OverlaySpectrumSelector")
            layout.addWidget(self.overlay_spectrum_combo)

            self.spectrum_role_summary = QPlainTextEdit(group)
            self.spectrum_role_summary.setObjectName("SpectrumRoleSummary")
            self.spectrum_role_summary.setReadOnly(True)
            layout.addWidget(self.spectrum_role_summary)
            return group

        def _build_library_panel(self) -> QWidget:
            group = QGroupBox("Data Libraries", self)
            group.setObjectName("SidebarLibraryPanel")
            layout = QVBoxLayout(group)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "The modern Qt shell reads from the governed library registry. "
                    "Manual and standards workflows use these selectors instead of legacy Tk state."
                ),
                group,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            layout.addWidget(QLabel("Identification library", group))
            self.gamma_source_combo = QComboBox(group)
            self.gamma_source_combo.setObjectName("GammaLibraryCombo")
            layout.addWidget(self.gamma_source_combo)

            self.custom_gamma_path = QLineEdit(group)
            self.custom_gamma_path.setObjectName("CustomGammaPathInput")
            self.custom_gamma_path.setPlaceholderText(
                "Optional custom gamma locator (JSON/CSV/YAML/sqlite://.../python://...)"
            )
            layout.addWidget(self.custom_gamma_path)

            layout.addWidget(QLabel("Calibration sources", group))
            self.calibration_source_combo = QComboBox(group)
            self.calibration_source_combo.setObjectName("CalibrationLibraryCombo")
            layout.addWidget(self.calibration_source_combo)

            layout.addWidget(QLabel("Standards / monitors", group))
            self.naa_source_combo = QComboBox(group)
            self.naa_source_combo.setObjectName("NaaMonitorLibraryCombo")
            layout.addWidget(self.naa_source_combo)

            layout.addWidget(QLabel("Dosimetry catalog", group))
            self.dosimetry_source_combo = QComboBox(group)
            self.dosimetry_source_combo.setObjectName("DosimetryLibraryCombo")
            layout.addWidget(self.dosimetry_source_combo)

            layout.addWidget(QLabel("Activation catalog", group))
            self.activation_source_combo = QComboBox(group)
            self.activation_source_combo.setObjectName("ActivationLibraryCombo")
            layout.addWidget(self.activation_source_combo)

            self.library_summary = QPlainTextEdit(group)
            self.library_summary.setObjectName("LibrarySummary")
            self.library_summary.setReadOnly(True)
            layout.addWidget(self.library_summary)
            return group

        def _populate_library_combos(self) -> None:
            self._populate_combo(
                self.gamma_source_combo,
                self.library_manager.available_sources("gamma_identification"),
            )
            self._populate_combo(
                self.calibration_source_combo,
                self.library_manager.available_sources("calibration"),
            )
            self._populate_combo(
                self.naa_source_combo,
                self.library_manager.available_sources("naa_monitor"),
            )
            self._populate_combo(
                self.dosimetry_source_combo,
                self.library_manager.available_sources("dosimetry"),
            )
            self._populate_combo(
                self.activation_source_combo,
                self.library_manager.available_sources("activation"),
            )

        def _populate_combo(self, combo: QComboBox, records) -> None:
            combo.blockSignals(True)
            combo.clear()
            for record in records:
                combo.addItem(record.label, record.source_id)
            combo.blockSignals(False)

        def _sync_library_state(self, state) -> None:
            self._set_combo_value(
                self.gamma_source_combo,
                state.gamma_identification_source_id,
            )
            self._set_combo_value(
                self.calibration_source_combo,
                state.calibration_source_id,
            )
            self._set_combo_value(self.naa_source_combo, state.naa_monitor_source_id)
            self._set_combo_value(self.dosimetry_source_combo, state.dosimetry_source_id)
            self._set_combo_value(
                self.activation_source_combo,
                state.activation_catalog_source_id,
            )
            self.custom_gamma_path.blockSignals(True)
            self.custom_gamma_path.setText(state.custom_gamma_path or "")
            self.custom_gamma_path.setEnabled(
                state.gamma_identification_source_id == "custom_gamma_file"
            )
            self.custom_gamma_path.blockSignals(False)
            self.library_summary.setPlainText(
                "\n\n".join(
                    [
                        "Identification\n"
                        + self.library_manager.summary_for_category("gamma_identification"),
                        "Calibration\n"
                        + self.library_manager.summary_for_category("calibration"),
                        "Standards / monitors\n"
                        + self.library_manager.summary_for_category("naa_monitor"),
                        "Dosimetry\n"
                        + self.library_manager.summary_for_category("dosimetry"),
                        "Activation\n"
                        + self.library_manager.summary_for_category("activation"),
                    ]
                )
            )
            self._refresh_nuclide_results(self.nuclide_query.text())

        def _set_combo_value(self, combo: QComboBox, source_id: str) -> None:
            index = combo.findData(source_id)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        def _populate_spectrum_role_combo(self, combo: QComboBox, state, slot_key: str) -> None:
            current_slot = next(
                (slot for slot in state.spectra if slot.key == slot_key),
                None,
            )
            combo.blockSignals(True)
            combo.clear()
            for record in state.loaded_spectra:
                combo.addItem(record.label, record.key)
            if current_slot is not None and current_slot.source_key:
                index = combo.findData(current_slot.source_key)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)

        def _assign_selected_spectrum_role(self, combo: QComboBox, slot_key: str) -> None:
            loaded_key = combo.currentData()
            if not loaded_key:
                return
            self.workspace_controller.assign_loaded_spectrum_to_slot(
                str(loaded_key),
                slot_key,
            )

        def _foreground_spectrum_changed(self) -> None:
            self._assign_selected_spectrum_role(self.foreground_spectrum_combo, "foreground")

        def _background_spectrum_changed(self) -> None:
            self._assign_selected_spectrum_role(self.background_spectrum_combo, "background")

        def _overlay_spectrum_changed(self) -> None:
            self._assign_selected_spectrum_role(self.overlay_spectrum_combo, "overlay")

        def _gamma_source_changed(self) -> None:
            source_id = self.gamma_source_combo.currentData()
            if source_id:
                self.library_manager.set_gamma_identification_source(
                    str(source_id),
                    custom_gamma_path=self.custom_gamma_path.text().strip() or None,
                )

        def _calibration_source_changed(self) -> None:
            source_id = self.calibration_source_combo.currentData()
            if source_id:
                self.library_manager.set_calibration_source(str(source_id))

        def _naa_source_changed(self) -> None:
            source_id = self.naa_source_combo.currentData()
            if source_id:
                self.library_manager.set_naa_monitor_source(str(source_id))

        def _dosimetry_source_changed(self) -> None:
            source_id = self.dosimetry_source_combo.currentData()
            if source_id:
                self.library_manager.set_dosimetry_source(str(source_id))

        def _activation_source_changed(self) -> None:
            source_id = self.activation_source_combo.currentData()
            if source_id:
                self.library_manager.set_activation_catalog_source(str(source_id))

        def _apply_custom_gamma_path(self) -> None:
            if self.gamma_source_combo.currentData() != "custom_gamma_file":
                return
            self.library_manager.set_gamma_identification_source(
                "custom_gamma_file",
                custom_gamma_path=self.custom_gamma_path.text().strip() or None,
            )

        def _sync_selection(self, state: SelectionState) -> None:
            self.selection_note.setPlainText(
                "Selection sync\n\n" + _selection_summary(state)
            )

        def _refresh_nuclide_results(self, query: str) -> None:
            self.nuclides.clear()
            for hit in self.nuclide_controller.search(query or "c"):
                label = hit.display_name
                if hit.strongest_lines_keV:
                    label += (
                        " · "
                        + " / ".join(f"{energy:.3f}" for energy in hit.strongest_lines_keV)
                        + " keV"
                    )
                item = QListWidgetItem(label, self.nuclides)
                item.setData(0x0100, hit.nuclide)

        def _activate_selected_nuclide(self) -> None:
            item = self.nuclides.currentItem()
            if item is None:
                return
            nuclide = item.data(0x0100)
            if nuclide:
                self.nuclide_controller.activate(str(nuclide))

        def _pin_selected_nuclide(self) -> None:
            item = self.nuclides.currentItem()
            if item is None:
                return
            nuclide = item.data(0x0100)
            if nuclide:
                pinned = list(self.workspace_controller.state.pinned_nuclides)
                if str(nuclide) in pinned:
                    pinned.remove(str(nuclide))
                else:
                    pinned.append(str(nuclide))
                self.workspace_controller.set_state(
                    self.workspace_controller.state.__class__(
                        **{
                            **self.workspace_controller.state.__dict__,
                            "pinned_nuclides": tuple(pinned),
                            "cascade_sum_lines_keV": compute_cascade_sum_lines(
                                pinned,
                                source_id=self.library_manager.state.gamma_identification_source_id,
                                custom_path=self.library_manager.state.custom_gamma_path,
                            ),
                        }
                    )
                )

        def _sync_workspace_state(self, state) -> None:
            self.files.clear()
            assignments: dict[str, list[str]] = {}
            for slot in state.spectra:
                if slot.source_key:
                    assignments.setdefault(slot.source_key, []).append(slot.label)
            for record in state.loaded_spectra:
                item = QTreeWidgetItem(
                    [
                        record.label,
                        ", ".join(assignments.get(record.key, ())) or "Available",
                    ]
                )
                item.setData(0, 0x0100, record.key)
                if record.source_path:
                    item.setToolTip(0, record.source_path)
                self.files.addTopLevelItem(item)
            self.files.resizeColumnToContents(0)
            self.files.expandAll()

            self._populate_spectrum_role_combo(
                self.foreground_spectrum_combo,
                state,
                "foreground",
            )
            self._populate_spectrum_role_combo(
                self.background_spectrum_combo,
                state,
                "background",
            )
            self._populate_spectrum_role_combo(
                self.overlay_spectrum_combo,
                state,
                "overlay",
            )
            self.spectrum_role_summary.setPlainText(
                "\n".join(
                    [
                        "Foreground: "
                        + next(
                            (
                                slot.source_label or slot.label
                                for slot in state.spectra
                                if slot.key == "foreground"
                            ),
                            "None",
                        ),
                        "Background: "
                        + next(
                            (
                                slot.source_label or slot.label
                                for slot in state.spectra
                                if slot.key == "background"
                            ),
                            "None",
                        ),
                        "Overlay: "
                        + next(
                            (
                                slot.source_label or slot.label
                                for slot in state.spectra
                                if slot.key == "overlay"
                            ),
                            "None",
                        ),
                        "",
                        (
                            f"Mode: {state.background_mode} subtraction | "
                            f"Background {'visible' if state.background_visible else 'hidden'}"
                        ),
                    ]
                )
            )
            self.pinned_nuclides.clear()
            for nuclide in state.pinned_nuclides:
                QListWidgetItem(nuclide, self.pinned_nuclides)


    class BottomWorkspaceTabs(QTabWidget):
        """Bottom-zone tab set for tables, calibration, activity, batch, and logs."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
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
            self.addTab(
                self._text_panel(
                    "Batch Queue",
                    "Queued spectra, progress, and aggregate reporting placeholder.",
                ),
                "Batch Queue",
            )
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
                    "Phase 2.1 now uses the dedicated Qt calibration workspace rather "
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


    class ToolContextPanel(QWidget):
        """Right-side contextual tool panel placeholder."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QLabel("Tool Context", self)
            title.setObjectName("PanelHeading")
            layout.addWidget(title)

            self.mode_summary = QLabel(self)
            self.mode_summary.setWordWrap(True)
            self.mode_summary.setObjectName("PanelBody")
            layout.addWidget(self.mode_summary)

            self.selection_summary = QLabel("No active selection", self)
            self.selection_summary.setWordWrap(True)
            self.selection_summary.setObjectName("PanelBody")
            layout.addWidget(self.selection_summary)

            self.workspace_summary = QLabel(self)
            self.workspace_summary.setWordWrap(True)
            self.workspace_summary.setObjectName("PanelBody")
            layout.addWidget(self.workspace_summary)

            layout.addWidget(
                _card(
                    "Current focus",
                    "ROI bounds, calibration points, peak model settings, and standards locks will appear here.",
                    "This panel stays contextual instead of burying settings in modal-only flows.",
                )
            )
            layout.addWidget(
                _card(
                    "Recommended defaults",
                    "Expert mode keeps all defensible methods available while still surfacing a documented recommended path.",
                    "Standards mode replaces recommendations with locked requirements.",
                )
            )
            layout.addStretch(1)

            self.mode_manager.subscribe(self._sync_mode)
            self.selection_bus.subscribe(self._sync_selection)
            self.workspace_controller.subscribe(self._sync_workspace)
            self._sync_mode(self.mode_manager.state)
            self._sync_selection(self.selection_bus.state)
            self._sync_workspace(self.workspace_controller.state)

        def _sync_mode(self, state) -> None:
            description = f"Mode: {state.mode.value.title()}"
            if state.standard:
                description += f" ({state.standard})"
            description += f"\nTheme: {state.theme}"
            self.mode_summary.setText(description)

        def _sync_selection(self, state: SelectionState) -> None:
            self.selection_summary.setText(_selection_summary(state))

        def _sync_workspace(self, state) -> None:
            foreground = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "foreground"
                ),
                "none",
            )
            background = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "background"
                ),
                "none",
            )
            overlay = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "overlay"
                ),
                "none",
            )
            self.workspace_summary.setText(
                (
                    f"Active spectrum: {state.active_spectrum_key}\n"
                    f"Foreground source: {foreground}\n"
                    f"Background source: {background}\n"
                    f"Overlay source: {overlay}\n"
                    f"Peaks: {len(state.peaks)}\n"
                    f"Pinned nuclides: {', '.join(state.pinned_nuclides) or 'none'}\n"
                    f"Background: {state.background_mode} "
                    f"({'visible' if state.background_visible else 'hidden'})"
                )
            )


else:

    class CentralWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
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
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
            library_manager: DataLibraryManager | None = None,
            parent=None,
        ) -> None:
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager
            self.parent = parent


    class BottomWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: Phase2WorkspaceController,
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
            workspace_controller: Phase2WorkspaceController,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.parent = parent
