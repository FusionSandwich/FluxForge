"""Sidebar modern-shell panel module."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from fluxforge.core.analysis_workspace import compute_cascade_sum_lines
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_dead_time_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.nuclide_search import NuclideSearchController
from fluxforge.gui.panels.modern_shell_shared import (
    current_tab_label,
    format_duration as _format_duration,
    format_percent as _format_percent,
    selection_summary as _selection_summary,
    set_tab_label,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import ReferenceLine
from fluxforge.plugins import bootstrap_builtin_registries
from fluxforge.standards import (
    QAMonitor,
    StandardsEvaluationContext,
    register_builtin_standards_modules,
)

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
        Qt,
    )

    class SidebarPanel(QWidget):
        """Left-side shell for files, devices, libraries, results, and QA."""

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
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.library_manager = library_manager or DataLibraryManager()
            self.qa_monitor = qa_monitor or QAMonitor()
            self._open_qa_history = open_qa_history
            self._open_standards_review = open_standards_review
            self.qa_monitor.seed_demo_history()
            self.registries = bootstrap_builtin_registries()
            register_builtin_standards_modules(self.registries)
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

            layout.addWidget(self._build_reference_workbench_panel(), 3)
            self._refresh_nuclide_results("cs")

            self.selection_note = QTextEdit(self)
            self.selection_note.setObjectName("SidebarNote")
            self.selection_note.setReadOnly(True)
            self.selection_note.setPlainText(
                "Selection sync\n\nNo active selection"
            )
            layout.addWidget(self.selection_note, 1)

            self.qa_note = QTextEdit(self)
            self.qa_note.setObjectName("QaStandardsSummary")
            self.qa_note.setReadOnly(True)
            layout.addWidget(self.qa_note, 1)

            qa_actions = QHBoxLayout()
            self.qa_history_button = QPushButton("View QA History", self)
            self.qa_history_button.setObjectName("QaHistoryShortcutButton")
            self.qa_history_button.clicked.connect(self._open_qa_history_clicked)
            qa_actions.addWidget(self.qa_history_button)
            self.astm_check_button = QPushButton("Run ASTM Check", self)
            self.astm_check_button.setObjectName("RunAstmCheckButton")
            self.astm_check_button.clicked.connect(self._open_standards_review_clicked)
            qa_actions.addWidget(self.astm_check_button)
            layout.addLayout(qa_actions)

            self.selection_bus.subscribe(self._sync_selection)
            self.library_manager.subscribe(self._sync_library_state)
            self.workspace_controller.subscribe(self._sync_workspace_state)
            self.mode_manager.subscribe(lambda _state: self._sync_qa_summary())
            self.mode_manager.subscribe(lambda _state: self._sync_library_state(self.library_manager.state))
            self.nuclide_query.textChanged.connect(self._refresh_nuclide_results)
            self.nuclides.itemSelectionChanged.connect(self._activate_selected_nuclide)
            self.nuclide_age_days.valueChanged.connect(self._refresh_nuclide_details)
            self.save_selected_nuclide_button.clicked.connect(self._save_selected_nuclide)
            self.remove_saved_nuclide_button.clicked.connect(self._remove_saved_nuclide)
            self.clear_saved_nuclide_button.clicked.connect(self._clear_saved_nuclides)
            self.apply_saved_overlay_button.clicked.connect(self._apply_saved_list_overlay)
            self.add_selected_mixture_button.clicked.connect(self._add_selected_to_mixture)
            self.remove_mixture_row_button.clicked.connect(self._remove_selected_mixture_row)
            self.clear_mixture_button.clicked.connect(self._clear_mixture)
            self.normalize_mixture_button.clicked.connect(self._normalize_mixture_weights)
            self.apply_mixture_overlay_button.clicked.connect(self._apply_mixture_overlay)
            self.mixture_table.itemChanged.connect(self._update_mixture_summary)
            self.custom_gamma_path.editingFinished.connect(self._apply_custom_gamma_path)
            self.register_custom_gamma_button.clicked.connect(
                self._register_user_gamma_source
            )
            self.remove_registered_gamma_button.clicked.connect(
                self._remove_registered_gamma_source
            )
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
            self._sync_qa_summary()

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

            self.custom_gamma_alias = QLineEdit(group)
            self.custom_gamma_alias.setObjectName("CustomGammaAliasInput")
            self.custom_gamma_alias.setPlaceholderText(
                "Registered alias for the library above"
            )
            layout.addWidget(self.custom_gamma_alias)

            custom_actions = QHBoxLayout()
            self.register_custom_gamma_button = QPushButton("Register Library", group)
            self.register_custom_gamma_button.setObjectName("RegisterCustomGammaButton")
            custom_actions.addWidget(self.register_custom_gamma_button)
            self.remove_registered_gamma_button = QPushButton("Remove Selected", group)
            self.remove_registered_gamma_button.setObjectName(
                "RemoveRegisteredGammaButton"
            )
            custom_actions.addWidget(self.remove_registered_gamma_button)
            custom_actions.addStretch(1)
            layout.addLayout(custom_actions)

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

        def _build_reference_workbench_panel(self) -> QWidget:
            group = QGroupBox("Nuclide Workbench", self)
            group.setObjectName("NuclideWorkbenchPanel")
            layout = QVBoxLayout(group)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "PeakEasy-style reference workbench: search isotopes, review half-life and "
                    "line details, save analyst lists, build mixtures, and publish overlays "
                    "without leaving the modern Qt shell."
                ),
                group,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.nuclide_query = QLineEdit(group)
            self.nuclide_query.setObjectName("NuclideSearchInput")
            self.nuclide_query.setPlaceholderText("Nuclide search...")
            layout.addWidget(self.nuclide_query)

            self.nuclides = QListWidget(group)
            self.nuclides.setObjectName("SidebarList")
            layout.addWidget(self.nuclides, 1)

            action_row = QHBoxLayout()
            self.pin_selected_nuclide_button = QPushButton("Pin Selected Nuclide", group)
            self.pin_selected_nuclide_button.clicked.connect(self._pin_selected_nuclide)
            action_row.addWidget(self.pin_selected_nuclide_button)
            self.save_selected_nuclide_button = QPushButton("Save To List", group)
            self.save_selected_nuclide_button.setObjectName("SaveNuclideToUserListButton")
            action_row.addWidget(self.save_selected_nuclide_button)
            self.add_selected_mixture_button = QPushButton("Add To Mixture", group)
            self.add_selected_mixture_button.setObjectName("AddNuclideToMixtureButton")
            action_row.addWidget(self.add_selected_mixture_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self.reference_tabs = QTabWidget(group)
            self.reference_tabs.setObjectName("NuclideWorkbenchTabs")
            layout.addWidget(self.reference_tabs, 2)

            details_tab = QWidget(self.reference_tabs)
            details_layout = QVBoxLayout(details_tab)
            details_layout.setContentsMargins(8, 8, 8, 8)
            details_layout.setSpacing(8)

            age_row = QHBoxLayout()
            age_row.addWidget(QLabel("Nuclide age (days)", details_tab))
            self.nuclide_age_days = QDoubleSpinBox(details_tab)
            self.nuclide_age_days.setObjectName("NuclideAgeDaysSpin")
            self.nuclide_age_days.setRange(0.0, 36500.0)
            self.nuclide_age_days.setDecimals(2)
            self.nuclide_age_days.setValue(0.0)
            age_row.addWidget(self.nuclide_age_days)
            age_row.addStretch(1)
            details_layout.addLayout(age_row)

            self.nuclide_details_browser = QTextBrowser(details_tab)
            self.nuclide_details_browser.setObjectName("NuclideDetailsBrowser")
            details_layout.addWidget(self.nuclide_details_browser, 1)

            self.nuclide_line_table = QTableWidget(0, 3, details_tab)
            self.nuclide_line_table.setObjectName("NuclideLineTable")
            self.nuclide_line_table.setHorizontalHeaderLabels(("Energy (keV)", "Yield", "Age Adj."))
            self.nuclide_line_table.verticalHeader().setVisible(False)
            self.nuclide_line_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.nuclide_line_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            details_layout.addWidget(self.nuclide_line_table, 1)

            details_layout.addWidget(QLabel("Pinned nuclides", details_tab))
            self.pinned_nuclides = QListWidget(details_tab)
            self.pinned_nuclides.setObjectName("PinnedNuclidesList")
            details_layout.addWidget(self.pinned_nuclides, 1)

            self.reference_tabs.addTab(details_tab, "Details")

            saved_tab = QWidget(self.reference_tabs)
            saved_layout = QVBoxLayout(saved_tab)
            saved_layout.setContentsMargins(8, 8, 8, 8)
            saved_layout.setSpacing(8)

            self.saved_nuclides = QListWidget(saved_tab)
            self.saved_nuclides.setObjectName("SavedNuclideList")
            saved_layout.addWidget(self.saved_nuclides, 1)

            saved_actions = QHBoxLayout()
            self.remove_saved_nuclide_button = QPushButton("Remove", saved_tab)
            self.remove_saved_nuclide_button.setObjectName("RemoveSavedNuclideButton")
            saved_actions.addWidget(self.remove_saved_nuclide_button)
            self.clear_saved_nuclide_button = QPushButton("Clear List", saved_tab)
            self.clear_saved_nuclide_button.setObjectName("ClearSavedNuclideButton")
            saved_actions.addWidget(self.clear_saved_nuclide_button)
            self.apply_saved_overlay_button = QPushButton("Apply Overlay", saved_tab)
            self.apply_saved_overlay_button.setObjectName("ApplySavedNuclideOverlayButton")
            saved_actions.addWidget(self.apply_saved_overlay_button)
            saved_actions.addStretch(1)
            saved_layout.addLayout(saved_actions)

            self.saved_nuclide_summary = QTextBrowser(saved_tab)
            self.saved_nuclide_summary.setObjectName("SavedNuclideSummary")
            saved_layout.addWidget(self.saved_nuclide_summary, 1)

            self.reference_tabs.addTab(saved_tab, "User List")

            mixture_tab = QWidget(self.reference_tabs)
            mixture_layout = QVBoxLayout(mixture_tab)
            mixture_layout.setContentsMargins(8, 8, 8, 8)
            mixture_layout.setSpacing(8)

            self.mixture_table = QTableWidget(0, 2, mixture_tab)
            self.mixture_table.setObjectName("NuclideMixtureTable")
            self.mixture_table.setHorizontalHeaderLabels(("Nuclide", "Weight"))
            self.mixture_table.verticalHeader().setVisible(False)
            self.mixture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.mixture_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            mixture_layout.addWidget(self.mixture_table, 1)

            mixture_actions = QHBoxLayout()
            self.remove_mixture_row_button = QPushButton("Remove", mixture_tab)
            self.remove_mixture_row_button.setObjectName("RemoveMixtureRowButton")
            mixture_actions.addWidget(self.remove_mixture_row_button)
            self.clear_mixture_button = QPushButton("Clear Mixture", mixture_tab)
            self.clear_mixture_button.setObjectName("ClearMixtureButton")
            mixture_actions.addWidget(self.clear_mixture_button)
            self.normalize_mixture_button = QPushButton("Normalize", mixture_tab)
            self.normalize_mixture_button.setObjectName("NormalizeMixtureButton")
            mixture_actions.addWidget(self.normalize_mixture_button)
            self.apply_mixture_overlay_button = QPushButton("Apply Mixture Overlay", mixture_tab)
            self.apply_mixture_overlay_button.setObjectName("ApplyMixtureOverlayButton")
            mixture_actions.addWidget(self.apply_mixture_overlay_button)
            mixture_actions.addStretch(1)
            mixture_layout.addLayout(mixture_actions)

            self.mixture_summary = QTextBrowser(mixture_tab)
            self.mixture_summary.setObjectName("NuclideMixtureSummary")
            mixture_layout.addWidget(self.mixture_summary, 1)

            self.reference_tabs.addTab(mixture_tab, "Mixtures")
            return group

        def _populate_library_combos(self) -> None:
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode is GUIMode.STANDARDS
                else None
            )
            self._populate_combo(
                self.gamma_source_combo,
                self.library_manager.available_sources(
                    "gamma_identification",
                    standard=standard,
                ),
            )
            self._populate_combo(
                self.calibration_source_combo,
                self.library_manager.available_sources("calibration", standard=standard),
            )
            self._populate_combo(
                self.naa_source_combo,
                self.library_manager.available_sources("naa_monitor", standard=standard),
            )
            self._populate_combo(
                self.dosimetry_source_combo,
                self.library_manager.available_sources("dosimetry", standard=standard),
            )
            self._populate_combo(
                self.activation_source_combo,
                self.library_manager.available_sources("activation", standard=standard),
            )

        def _populate_combo(self, combo: QComboBox, records) -> None:
            combo.blockSignals(True)
            combo.clear()
            for record in records:
                combo.addItem(record.label, record.source_id)
            combo.blockSignals(False)

        def _sync_library_state(self, state) -> None:
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode is GUIMode.STANDARDS
                else None
            )
            resolved_state = self.library_manager.resolved_state(standard=standard)
            self.nuclide_controller.set_source(
                resolved_state.gamma_identification_source_id,
                custom_path=resolved_state.custom_gamma_path,
            )
            self._populate_library_combos()
            self._set_combo_value(
                self.gamma_source_combo,
                resolved_state.gamma_identification_source_id,
            )
            self._set_combo_value(
                self.calibration_source_combo,
                resolved_state.calibration_source_id,
            )
            self._set_combo_value(self.naa_source_combo, resolved_state.naa_monitor_source_id)
            self._set_combo_value(self.dosimetry_source_combo, resolved_state.dosimetry_source_id)
            self._set_combo_value(
                self.activation_source_combo,
                resolved_state.activation_catalog_source_id,
            )
            self.custom_gamma_path.blockSignals(True)
            self.custom_gamma_path.setText(state.custom_gamma_path or "")
            self.custom_gamma_path.setEnabled(
                standard is None
                and state.gamma_identification_source_id == "custom_gamma_file"
            )
            self.custom_gamma_path.blockSignals(False)
            self.custom_gamma_alias.setEnabled(standard is None)
            self.register_custom_gamma_button.setEnabled(standard is None)
            self.remove_registered_gamma_button.setEnabled(
                standard is None
                and str(state.gamma_identification_source_id).startswith("user_gamma_")
            )
            self.gamma_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "gamma_identification",
                    standard=standard,
                )
                is None
                and self.gamma_source_combo.count() > 1
            )
            self.calibration_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "calibration",
                    standard=standard,
                )
                is None
                and self.calibration_source_combo.count() > 1
            )
            self.naa_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "naa_monitor",
                    standard=standard,
                )
                is None
                and self.naa_source_combo.count() > 1
            )
            self.dosimetry_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "dosimetry",
                    standard=standard,
                )
                is None
                and self.dosimetry_source_combo.count() > 1
            )
            self.activation_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "activation",
                    standard=standard,
                )
                is None
                and self.activation_source_combo.count() > 1
            )
            sections = [
                "Identification\n"
                + self.library_manager.summary_for_category(
                    "gamma_identification",
                    standard=standard,
                ),
                "Calibration\n"
                + self.library_manager.summary_for_category(
                    "calibration",
                    standard=standard,
                ),
                "Standards / monitors\n"
                + self.library_manager.summary_for_category(
                    "naa_monitor",
                    standard=standard,
                ),
                "Dosimetry\n"
                + self.library_manager.summary_for_category(
                    "dosimetry",
                    standard=standard,
                ),
                "Activation\n"
                + self.library_manager.summary_for_category(
                    "activation",
                    standard=standard,
                ),
            ]
            registered = self.library_manager.registered_user_gamma_sources()
            if registered:
                sections.append(
                    "Registered User Libraries\n"
                    + "\n".join(
                        f"{record.source_id} -> {record.path_hint}"
                        for record in registered
                    )
                )
            self.library_summary.setPlainText("\n\n".join(sections))
            self._refresh_nuclide_results(self.nuclide_query.text())
            self._refresh_nuclide_details()
            self._update_saved_summary()
            self._update_mixture_summary()

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

        def _register_user_gamma_source(self) -> None:
            alias = self.custom_gamma_alias.text().strip()
            locator = self.custom_gamma_path.text().strip()
            if not alias or not locator:
                self.library_summary.setPlainText(
                    "Provide both a library alias and locator to register a user library."
                )
                return
            try:
                record = self.library_manager.register_user_gamma_source(alias, locator)
            except Exception as exc:
                self.library_summary.setPlainText(str(exc))
                return
            self.custom_gamma_alias.clear()
            self._set_combo_value(self.gamma_source_combo, record.source_id)

        def _remove_registered_gamma_source(self) -> None:
            source_id = str(self.gamma_source_combo.currentData() or "")
            if not source_id.startswith("user_gamma_"):
                self.library_summary.setPlainText(
                    "Select a registered user library before attempting removal."
                )
                return
            if not self.library_manager.remove_user_gamma_source(source_id):
                self.library_summary.setPlainText(
                    f"Registered library not found: {source_id}"
                )

        def _sync_selection(self, state: SelectionState) -> None:
            self.selection_note.setPlainText(
                "Selection sync\n\n" + _selection_summary(state)
            )
            self._sync_qa_summary()

        def _refresh_nuclide_results(self, query: str) -> None:
            previous = self._selected_nuclide_name()
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
            if self.nuclides.count():
                selected_index = 0
                if previous:
                    for index in range(self.nuclides.count()):
                        item = self.nuclides.item(index)
                        if item is not None and item.data(0x0100) == previous:
                            selected_index = index
                            break
                self.nuclides.setCurrentRow(selected_index)
            else:
                self._refresh_nuclide_details()

        def _activate_selected_nuclide(self) -> None:
            nuclide = self._selected_nuclide_name()
            if nuclide:
                self.nuclide_controller.activate(str(nuclide))
            self._refresh_nuclide_details()

        def _pin_selected_nuclide(self) -> None:
            nuclide = self._selected_nuclide_name()
            if nuclide:
                pinned = list(self.workspace_controller.state.pinned_nuclides)
                if str(nuclide) in pinned:
                    pinned.remove(str(nuclide))
                else:
                    pinned.append(str(nuclide))
                standard = (
                    self.mode_manager.state.standard
                    if self.mode_manager.state.mode is GUIMode.STANDARDS
                    else None
                )
                resolved_state = self.library_manager.resolved_state(standard=standard)
                self.workspace_controller.set_state(
                    self.workspace_controller.state.__class__(
                        **{
                            **self.workspace_controller.state.__dict__,
                            "pinned_nuclides": tuple(pinned),
                            "cascade_sum_lines_keV": compute_cascade_sum_lines(
                                pinned,
                                source_id=resolved_state.gamma_identification_source_id,
                                custom_path=resolved_state.custom_gamma_path,
                            ),
                        }
                    )
                )
            self._update_saved_summary()

        def _selected_nuclide_name(self) -> str | None:
            item = self.nuclides.currentItem()
            if item is None:
                return None
            value = item.data(0x0100)
            return str(value) if value else None

        def _refresh_nuclide_details(self) -> None:
            nuclide = self._selected_nuclide_name()
            if not nuclide:
                self.nuclide_details_browser.setHtml("<p>No nuclide selected.</p>")
                self.nuclide_line_table.setRowCount(0)
                return
            details = self.nuclide_controller.nuclide_details(
                nuclide,
                age_s=float(self.nuclide_age_days.value()) * 86400.0,
                limit=8,
            )
            parents = (
                ", ".join(
                    f"{item.display_name} ({item.decay_mode or 'decay'})"
                    for item in details.parents
                )
                or "No parent-chain data in this source."
            )
            daughters = (
                ", ".join(
                    f"{item.display_name} ({item.decay_mode or 'decay'})"
                    for item in details.daughters
                )
                or "No daughter-chain data in this source."
            )
            gamma_lines = (
                ", ".join(f"{line.energy_keV:.3f}" for line in details.gamma_lines[:4])
                or "None"
            )
            xray_lines = (
                ", ".join(f"{line.energy_keV:.3f}" for line in details.xray_lines[:4])
                or "None"
            )
            age_note = (
                f"{self.nuclide_age_days.value():.2f} d"
                if self.nuclide_age_days.value() > 0.0
                else "fresh reference"
            )
            self.nuclide_details_browser.setHtml(
                (
                    f"<h3>{details.display_name}</h3>"
                    f"<p><strong>Source:</strong> {self.nuclide_controller.source_label()}<br/>"
                    f"<strong>Half-life:</strong> {details.half_life_s:.3e} s<br/>"
                    f"<strong>Specific activity:</strong> {details.specific_activity_bq_g:.3e} Bq/g<br/>"
                    f"<strong>Dose @ 1 µCi / 1 m:</strong> {details.dose_rate_uSv_h_per_uCi_at_1m:.4f} µSv/h<br/>"
                    f"<strong>Age context:</strong> {age_note}</p>"
                    f"<p><strong>Strongest gamma lines:</strong> {gamma_lines}<br/>"
                    f"<strong>Strongest X-rays:</strong> {xray_lines}</p>"
                    f"<p><strong>Parents:</strong> {parents}<br/>"
                    f"<strong>Daughters:</strong> {daughters}</p>"
                )
            )
            lines = details.gamma_lines or details.xray_lines
            self.nuclide_line_table.setRowCount(len(lines))
            for row, line in enumerate(lines):
                values = (
                    f"{line.energy_keV:.3f}",
                    f"{line.intensity:.6f}",
                    f"{line.age_adjusted_intensity:.6f}",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    self.nuclide_line_table.setItem(row, column, item)

        def _save_selected_nuclide(self) -> None:
            nuclide = self._selected_nuclide_name()
            if not nuclide:
                return
            existing = {
                self.saved_nuclides.item(index).data(0x0100)
                for index in range(self.saved_nuclides.count())
            }
            if nuclide in existing:
                return
            details = self.nuclide_controller.nuclide_details(nuclide, limit=4)
            item = QListWidgetItem(details.display_name, self.saved_nuclides)
            item.setData(0x0100, details.nuclide)
            self._update_saved_summary()
            self.reference_tabs.setCurrentWidget(self.saved_nuclides.parentWidget())

        def _remove_saved_nuclide(self) -> None:
            row = self.saved_nuclides.currentRow()
            if row >= 0:
                self.saved_nuclides.takeItem(row)
            self._update_saved_summary()

        def _clear_saved_nuclides(self) -> None:
            self.saved_nuclides.clear()
            self._update_saved_summary()

        def _apply_saved_list_overlay(self) -> None:
            nuclides = self._saved_nuclide_names()
            if nuclides:
                self._publish_reference_overlay(nuclides, label_prefix="Saved")

        def _saved_nuclide_names(self) -> tuple[str, ...]:
            names = []
            for index in range(self.saved_nuclides.count()):
                item = self.saved_nuclides.item(index)
                if item is None:
                    continue
                value = item.data(0x0100)
                if value:
                    names.append(str(value))
            return tuple(names)

        def _update_saved_summary(self) -> None:
            saved = self._saved_nuclide_names()
            if not saved:
                self.saved_nuclide_summary.setHtml(
                    "<p>No user-defined nuclide list yet.</p>"
                )
                return
            items = []
            for nuclide in saved:
                details = self.nuclide_controller.nuclide_details(nuclide, limit=3)
                lines = ", ".join(f"{line.energy_keV:.3f}" for line in details.gamma_lines[:3]) or "None"
                items.append(f"<li><strong>{details.display_name}</strong>: {lines} keV</li>")
            self.saved_nuclide_summary.setHtml(
                "<h3>User Define List</h3><ul>" + "".join(items) + "</ul>"
            )

        def _add_selected_to_mixture(self) -> None:
            nuclide = self._selected_nuclide_name()
            if not nuclide:
                return
            for row in range(self.mixture_table.rowCount()):
                item = self.mixture_table.item(row, 0)
                if item is not None and item.data(0x0100) == nuclide:
                    self.mixture_table.setCurrentCell(row, 0)
                    return
            self.mixture_table.blockSignals(True)
            row = self.mixture_table.rowCount()
            self.mixture_table.insertRow(row)
            details = self.nuclide_controller.nuclide_details(nuclide, limit=3)
            name_item = QTableWidgetItem(details.display_name)
            name_item.setData(0x0100, details.nuclide)
            self.mixture_table.setItem(row, 0, name_item)
            weight_item = QTableWidgetItem("1.0")
            self.mixture_table.setItem(row, 1, weight_item)
            self.mixture_table.blockSignals(False)
            self._update_mixture_summary()

        def _remove_selected_mixture_row(self) -> None:
            row = self.mixture_table.currentRow()
            if row >= 0:
                self.mixture_table.removeRow(row)
            self._update_mixture_summary()

        def _clear_mixture(self) -> None:
            self.mixture_table.setRowCount(0)
            self._update_mixture_summary()

        def _normalize_mixture_weights(self) -> None:
            entries = self._mixture_entries()
            total = sum(weight for _nuclide, weight in entries)
            if total <= 0.0:
                return
            self.mixture_table.blockSignals(True)
            for row, (_nuclide, weight) in enumerate(entries):
                normalized = weight / total
                self.mixture_table.setItem(row, 1, QTableWidgetItem(f"{normalized:.6f}"))
            self.mixture_table.blockSignals(False)
            self._update_mixture_summary()

        def _apply_mixture_overlay(self) -> None:
            entries = self._mixture_entries()
            if entries:
                self._publish_reference_overlay(
                    tuple(nuclide for nuclide, _weight in entries),
                    label_prefix="Mixture",
                )

        def _mixture_entries(self) -> tuple[tuple[str, float], ...]:
            entries: list[tuple[str, float]] = []
            for row in range(self.mixture_table.rowCount()):
                name_item = self.mixture_table.item(row, 0)
                weight_item = self.mixture_table.item(row, 1)
                if name_item is None:
                    continue
                nuclide = name_item.data(0x0100)
                if not nuclide:
                    continue
                try:
                    weight = float(weight_item.text()) if weight_item is not None else 0.0
                except (TypeError, ValueError):
                    weight = 0.0
                entries.append((str(nuclide), float(weight)))
            return tuple(entries)

        def _update_mixture_summary(self) -> None:
            entries = self._mixture_entries()
            if not entries:
                self.mixture_summary.setHtml("<p>No nuclide mixture defined.</p>")
                return
            total = sum(weight for _nuclide, weight in entries) or 1.0
            items = []
            for nuclide, weight in entries:
                details = self.nuclide_controller.nuclide_details(nuclide, limit=3)
                items.append(
                    f"<li><strong>{details.display_name}</strong>: {weight:.4f} "
                    f"({(weight / total) * 100.0:.1f}%)</li>"
                )
            self.mixture_summary.setHtml(
                "<h3>Mixture</h3><ul>" + "".join(items) + "</ul>"
            )

        def _publish_reference_overlay(
            self,
            nuclides: tuple[str, ...],
            *,
            label_prefix: str,
        ) -> None:
            annotation_lines: list[ReferenceLine] = []
            reference_lines: list[float] = []
            for nuclide in nuclides:
                details = self.nuclide_controller.nuclide_details(
                    nuclide,
                    age_s=float(self.nuclide_age_days.value()) * 86400.0,
                    limit=self.nuclide_controller.overlay_limit,
                )
                for line in details.gamma_lines[: self.nuclide_controller.overlay_limit]:
                    reference_lines.append(float(line.energy_keV))
                    annotation_lines.append(
                        ReferenceLine(
                            energy_keV=float(line.energy_keV),
                            label=f"{details.display_name} ref",
                            color="#f59e0b",
                        )
                    )
            if not reference_lines:
                return
            summary = ", ".join(
                self.nuclide_controller.nuclide_details(nuclide, limit=1).display_name
                for nuclide in nuclides[:4]
            )
            if len(nuclides) > 4:
                summary += f" +{len(nuclides) - 4}"
            self.selection_bus.publish(
                SelectionState(
                    nuclide=f"{label_prefix}: {summary}",
                    reference_lines_keV=tuple(reference_lines),
                    annotation_lines=tuple(annotation_lines),
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
            self._sync_qa_summary()

        def _sync_qa_summary(self) -> None:
            statuses = self.qa_monitor.status_snapshot()
            latest_status = statuses[0] if statuses else None
            active_spectrum = self.workspace_controller.spectrum()
            records = list(self.workspace_controller.loaded_spectrum_records())
            records.sort(
                key=lambda record: (
                    record.spectrum.start_time or datetime.max,
                    record.label,
                )
            )
            history_spectra = tuple(record.spectrum for record in records) or (
                (active_spectrum,) if active_spectrum is not None else ()
            )
            context = StandardsEvaluationContext(
                calibration_order=2,
                max_residual_keV=0.18,
                efficiency_uncertainty_pct=2.6,
                fwhm_at_413_keV=1.08,
                qa_centroid_drift_keV=(
                    latest_status.centroid_drift_keV if latest_status else 0.0
                ),
                qa_fwhm_degradation_pct=(
                    latest_status.fwhm_degradation_pct if latest_status else 0.0
                ),
                before_calibration=(
                    latest_status.last_check if latest_status is not None else None
                ),
                measured_at=(
                    latest_status.last_check if latest_status is not None else None
                ),
                after_calibration=(
                    latest_status.last_check if latest_status is not None else None
                ),
                net_counts={"primary": 1200.0, "Pu-240 160.3": 1205.0},
            )
            summary_bits = []
            for key in (
                "ASTM E181",
                "ASTM E1297",
                "ASTM E1218",
                "ASTM C1232",
                "ASTM C1030",
            ):
                module = self.registries.standards_modules.get(key)
                evaluation = module.evaluate(context)
                dot = {"green": "●", "amber": "◐", "red": "◆"}.get(
                    evaluation.overall_status,
                    "○",
                )
                summary_bits.append(f"{module.display_name} [{dot}]")

            lines = ["<h3>QA &amp; Standards</h3>"]
            lines.append(
                "<p><strong>ASTM Status:</strong> " + "  ".join(summary_bits) + "</p>"
            )
            if latest_status is not None:
                lines.append(
                    "<p><strong>QA Monitor:</strong> "
                    f"FWHM @ {latest_status.energy_keV:.2f} keV: {latest_status.fwhm_degradation_pct:+.2f}% | "
                    f"Drift: {latest_status.centroid_drift_keV:+.3f} keV<br/>"
                    f"Last check: {latest_status.last_check.isoformat(sep=' ', timespec='minutes')}</p>"
                )
            if active_spectrum is not None:
                roi_bounds = self.selection_bus.state.roi_bounds_keV
                count_forecast = estimate_count_target_forecast(
                    active_spectrum,
                    roi_bounds_keV=roi_bounds,
                    target_counts=10000.0,
                    history_spectra=history_spectra,
                )
                dead_time_forecast = estimate_dead_time_forecast(history_spectra)
                lines.append(
                    (
                        "<p><strong>Predictive:</strong> "
                        f"ROI ETA {_format_duration(count_forecast.eta_seconds)} | "
                        f"Dead time {_format_percent(dead_time_forecast.current_dead_time_fraction)}"
                    )
                    + (
                        f" → {_format_percent(dead_time_forecast.projected_dead_time_fraction_1h)} in 1h"
                    )
                    + "</p>"
                )
            recalibration_forecast = estimate_recalibration_forecast(
                self.qa_monitor.history()
            )
            if recalibration_forecast is not None:
                lines.append(
                    "<p><strong>Recalibration forecast:</strong> "
                    + (
                        recalibration_forecast.predicted_recalibration_at.strftime(
                            "%Y-%m-%d"
                        )
                        if recalibration_forecast.predicted_recalibration_at is not None
                        else "Stable"
                    )
                    + (
                        f" ({recalibration_forecast.days_until_recalibration:.1f} d)"
                        if recalibration_forecast.days_until_recalibration is not None
                        else ""
                    )
                    + "</p>"
                )
            active_standard = self.mode_manager.state.standard
            if active_standard and active_standard in self.registries.standards_modules:
                module = self.registries.standards_modules.get(active_standard)
                locks = "<br/>".join(
                    f"[locked] {setting.field_id}: {setting.value} ({setting.standard_section})"
                    for setting in module.locked_settings()
                ) or "No workflow locks."
                lines.append(
                    f"<p><strong>Active standard:</strong> {active_standard}<br/>{locks}</p>"
                )
            self.qa_note.setHtml("".join(lines))

        def _open_qa_history_clicked(self) -> None:
            if callable(self._open_qa_history):
                self._open_qa_history()

        def _open_standards_review_clicked(self) -> None:
            if callable(self._open_standards_review):
                self._open_standards_review()

        def _display_name_for_nuclide(self, nuclide: str) -> str:
            try:
                return self.nuclide_controller.nuclide_details(nuclide, limit=1).display_name
            except Exception:
                return str(nuclide)

        def workflow_state(self) -> dict[str, object]:
            return {
                "nuclide_query": self.nuclide_query.text(),
                "nuclide_age_days": float(self.nuclide_age_days.value()),
                "reference_tab": current_tab_label(self.reference_tabs),
                "saved_nuclides": list(self._saved_nuclide_names()),
                "mixture_entries": [
                    {"nuclide": nuclide, "weight": float(weight)}
                    for nuclide, weight in self._mixture_entries()
                ],
            }

        def apply_workflow_state(self, payload: dict[str, object] | None) -> None:
            if not payload:
                return
            if "nuclide_query" in payload:
                self.nuclide_query.setText(str(payload["nuclide_query"] or ""))
            if "nuclide_age_days" in payload:
                self.nuclide_age_days.setValue(float(payload["nuclide_age_days"]))

            saved = payload.get("saved_nuclides")
            if isinstance(saved, list):
                self.saved_nuclides.clear()
                for nuclide in saved:
                    value = str(nuclide or "").strip()
                    if not value:
                        continue
                    item = QListWidgetItem(
                        self._display_name_for_nuclide(value),
                        self.saved_nuclides,
                    )
                    item.setData(0x0100, value)
                self._update_saved_summary()

            mixture_entries = payload.get("mixture_entries")
            if isinstance(mixture_entries, list):
                self.mixture_table.blockSignals(True)
                self.mixture_table.setRowCount(0)
                for entry in mixture_entries:
                    if not isinstance(entry, dict):
                        continue
                    nuclide = str(entry.get("nuclide") or "").strip()
                    if not nuclide:
                        continue
                    weight = float(entry.get("weight") or 0.0)
                    row = self.mixture_table.rowCount()
                    self.mixture_table.insertRow(row)
                    name_item = QTableWidgetItem(self._display_name_for_nuclide(nuclide))
                    name_item.setData(0x0100, nuclide)
                    self.mixture_table.setItem(row, 0, name_item)
                    self.mixture_table.setItem(row, 1, QTableWidgetItem(f"{weight:.6g}"))
                self.mixture_table.blockSignals(False)
                self._update_mixture_summary()

            if "reference_tab" in payload:
                set_tab_label(self.reference_tabs, str(payload["reference_tab"]))
            self._refresh_nuclide_details()

else:

    class SidebarPanel:  # pragma: no cover - placeholder without Qt
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


__all__ = ["SidebarPanel"]
