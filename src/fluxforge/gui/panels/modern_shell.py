"""Modern placeholder panels for the next-generation GUI shell."""

from __future__ import annotations

from typing import Callable

import numpy as np

from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, pyqtgraph_backend_status
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.nuclide_search import NuclideSearchController
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.io.spe import GammaSpectrum

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.backends import PyQtGraphSpectrumCanvas
    from fluxforge.gui.qt_compat import (
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
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
        metadata={"source": "phase2-demo"},
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


    class CentralWorkspaceTabs(QTabWidget):
        """Center-zone tab stack with the spectrum and dashboard placeholders."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self._current_spectrum = build_demo_spectrum()
            self.setObjectName("CentralWorkspaceTabs")
            self.addTab(self._build_spectrum_tab(), "Spectrum")
            self.addTab(self._build_dashboard_tab(), "Dashboard")
            self.mode_manager.subscribe(self._sync_mode_banner)
            self._sync_mode_banner(self.mode_manager.state)

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

            body = QHBoxLayout()
            body.setSpacing(18)
            layout.addLayout(body, 1)

            if PYQTGRAPH_AVAILABLE:
                self.canvas = PyQtGraphSpectrumCanvas(
                    selection_bus=self.selection_bus,
                    parent=widget,
                )
                self.canvas.set_spectrum(self._current_spectrum.counts)
                self.canvas.set_reference_lines((661.657, 1173.228, 1332.492))
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

        def load_spectrum(self, spectrum) -> None:
            """Load a GammaSpectrum-like object into the primary canvas."""

            self._current_spectrum = spectrum
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return
            self.canvas.set_spectrum(tuple(float(value) for value in spectrum.counts))
            if getattr(spectrum, "gps", {}):
                self.selection_bus.publish(
                    SelectionState(
                        peak_energy_keV=self.selection_bus.state.peak_energy_keV,
                        roi_bounds_keV=self.selection_bus.state.roi_bounds_keV,
                        nuclide=self.selection_bus.state.nuclide,
                        reference_lines_keV=self.selection_bus.state.reference_lines_keV,
                    )
                )

        def current_spectrum(self):
            """Return the current spectrum object shown in the central workspace."""

            return self._current_spectrum

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
            library_manager: DataLibraryManager | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.selection_bus = selection_bus
            self.library_manager = library_manager or DataLibraryManager()
            self.nuclide_controller = NuclideSearchController(
                selection_bus,
                library_manager=self.library_manager,
            )

            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            files = QTreeWidget(self)
            files.setHeaderLabels(["Open Spectra", "State"])
            files.setObjectName("SidebarTree")
            root = QTreeWidgetItem(["demo_hpge.n42", "Ready"])
            root.addChild(QTreeWidgetItem(["background_overlay.n42", "Standby"]))
            files.addTopLevelItem(root)
            files.expandAll()
            layout.addWidget(files, 2)

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
            self.nuclide_query.textChanged.connect(self._refresh_nuclide_results)
            self.nuclides.itemSelectionChanged.connect(self._activate_selected_nuclide)
            self.custom_gamma_path.editingFinished.connect(self._apply_custom_gamma_path)
            self.gamma_source_combo.currentIndexChanged.connect(self._gamma_source_changed)
            self.calibration_source_combo.currentIndexChanged.connect(
                self._calibration_source_changed
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


    class BottomWorkspaceTabs(QTabWidget):
        """Bottom-zone tab set for tables, calibration, activity, batch, and logs."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            open_calibration_workspace: Callable[[], None] | None = None,
            open_quick_slider_calibration_workspace: Callable[[], None] | None = None,
            open_manual_calibration_workspace: Callable[[], None] | None = None,
            open_standards_calibration_workspace: Callable[[], None] | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self._open_calibration_workspace = open_calibration_workspace
            self._open_quick_slider_calibration_workspace = (
                open_quick_slider_calibration_workspace
            )
            self._open_manual_calibration_workspace = open_manual_calibration_workspace
            self._open_standards_calibration_workspace = open_standards_calibration_workspace
            self.addTab(
                self._text_panel(
                    "Peak Table",
                    "Peak rows, assignments, and fit quality will land here with SelectionBus synchronization.",
                ),
                "Peak Table",
            )
            self.addTab(
                self._build_calibration_panel(),
                "Calibration",
            )
            self.addTab(
                self._text_panel(
                    "Activity Results",
                    "Quantified activities, uncertainty badges, and standards-derived flags placeholder.",
                ),
                "Activity Results",
            )
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
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus

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
            self._sync_mode(self.mode_manager.state)
            self._sync_selection(self.selection_bus.state)

        def _sync_mode(self, state) -> None:
            description = f"Mode: {state.mode.value.title()}"
            if state.standard:
                description += f" ({state.standard})"
            description += f"\nTheme: {state.theme}"
            self.mode_summary.setText(description)

        def _sync_selection(self, state: SelectionState) -> None:
            self.selection_summary.setText(_selection_summary(state))


else:

    class CentralWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(self, mode_manager: ModeManager, selection_bus: SelectionBus, parent=None) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.parent = parent
            self._current_spectrum = build_demo_spectrum()

        def current_spectrum(self):
            return self._current_spectrum


    class SidebarPanel:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            selection_bus: SelectionBus,
            library_manager: DataLibraryManager | None = None,
            parent=None,
        ) -> None:
            self.selection_bus = selection_bus
            self.library_manager = library_manager
            self.parent = parent


    class BottomWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            open_calibration_workspace: Callable[[], None] | None = None,
            open_manual_calibration_workspace: Callable[[], None] | None = None,
            open_standards_calibration_workspace: Callable[[], None] | None = None,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.open_calibration_workspace = open_calibration_workspace
            self.open_manual_calibration_workspace = open_manual_calibration_workspace
            self.open_standards_calibration_workspace = open_standards_calibration_workspace
            self.parent = parent


    class ToolContextPanel:  # pragma: no cover - placeholder without Qt
        def __init__(self, mode_manager: ModeManager, selection_bus: SelectionBus, parent=None) -> None:
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.parent = parent
