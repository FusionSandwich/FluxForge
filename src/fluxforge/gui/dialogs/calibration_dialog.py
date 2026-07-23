"""Unified calibration workspace for the modern Qt GUI stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from fluxforge.core.analysis_workspace import detect_peak_candidates
from fluxforge.core.calibration import (
    ASTM_E181_ENERGY_LIMIT_KEV,
    EnergyCalibrationFit,
    EnergyCalibrationPoint,
    EnergyDeviationPair,
    FWHMCalibrationFit,
    FWHMCalibrationPoint,
    energy_calibration_slope,
    estimate_local_fwhm_channels,
    fit_energy_calibration,
    fit_fwhm_calibration,
    fit_quick_slider_calibration,
    resolve_energy_calibration_order,
)
from fluxforge.core.peak_fitting import (
    InteractivePeakFitResult,
    available_background_models,
    fit_roi_peak,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.backends.pyqtgraph_backend import catalog_pyqtgraph_export_action
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.nuclide_search import NuclideSearchController
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus
from fluxforge.gui.theme_manager import theme_tokens
from fluxforge.gui.widgets import MethodSelectorWidget
from fluxforge.io.spe import GammaSpectrum
from fluxforge.plugins import bootstrap_builtin_registries

if (
    QT_AVAILABLE and PYQTGRAPH_AVAILABLE
):  # pragma: no cover - optional dependency branch
    import pyqtgraph as pg

    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QComboBox,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        Qt,
    )


REFERENCE_LINES_KEV = (
    ("Cs-137", 661.657),
    ("Co-60 (1173)", 1173.228),
    ("Co-60 (1332)", 1332.492),
)

DETECTOR_SLOT_NAMES = (
    "Primary HPGe",
    "Field HPGe",
    "Low-Energy HPGe",
    "Well Detector",
)


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Preserved calibration state used for fine-tuning and detector-slot recall."""

    label: str
    energy_points: tuple[EnergyCalibrationPoint, ...]
    fwhm_points: tuple[FWHMCalibrationPoint, ...]
    deviation_pairs: tuple[EnergyDeviationPair, ...]


if (
    QT_AVAILABLE and PYQTGRAPH_AVAILABLE
):  # pragma: no cover - optional dependency branch

    class CalibrationWorkspaceDialog(QDialog):
        """Modern calibration workspace with embedded plots and live diagnostics."""

        ENERGY_HEADERS = (
            "Label",
            "Channel",
            "Observed keV",
            "Reference keV",
            "Unc keV",
            "Residual",
            "Status",
        )
        FWHM_HEADERS = (
            "Label",
            "Energy keV",
            "FWHM keV",
            "Unc keV",
            "Residual",
        )

        def __init__(
            self,
            *,
            spectrum: GammaSpectrum | None,
            mode_manager: ModeManager,
            selection_bus: SelectionBus | None = None,
            library_manager: DataLibraryManager | None = None,
            on_apply: (
                Callable[
                    [GammaSpectrum, EnergyCalibrationFit, FWHMCalibrationFit | None],
                    None,
                ]
                | None
            ) = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("CalibrationWorkspaceDialog")
            self.setWindowTitle("FluxForge — Calibration Workspace")
            self.resize(1540, 960)
            self.setModal(False)

            self.mode_manager = mode_manager
            self.selection_bus = selection_bus or SelectionBus.shared()
            self.library_manager = library_manager or DataLibraryManager()
            self.nuclide_controller = NuclideSearchController(
                self.selection_bus,
                library_manager=self.library_manager,
            )
            self.on_apply = on_apply
            self._syncing_energy_table = False
            self._syncing_fwhm_table = False
            self._syncing_deviation_table = False
            self._syncing_quick_controls = False
            if spectrum is None:
                raise ValueError("Calibration requires an explicitly loaded spectrum.")
            self._spectrum = spectrum
            self._energy_fit: EnergyCalibrationFit | None = None
            self._fwhm_fit: FWHMCalibrationFit | None = None
            self._quick_fit = None
            self._roi_fit: InteractivePeakFitResult | None = None
            self._peak_fitter_registry = bootstrap_builtin_registries().peak_fitters
            self._preserved_snapshot: CalibrationSnapshot | None = None
            self._detector_slots: dict[str, CalibrationSnapshot] = {}

            self._build_ui()
            self.mode_manager.subscribe(self._on_mode_state_changed)
            self.destroyed.connect(self._cleanup)

            self._seed_tables()
            self._plot_spectrum()
            self._on_mode_state_changed(self.mode_manager.state)
            self._refresh_energy_fit()
            self._refresh_fwhm_fit()

        def _cleanup(self, *_args) -> None:
            self.mode_manager.unsubscribe(self._on_mode_state_changed)
            self.library_manager.unsubscribe(self._sync_library_state)

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(14)

            header = QFrame(self)
            header.setObjectName("CalibrationHeaderCard")
            header_layout = QVBoxLayout(header)
            header_layout.setContentsMargins(24, 20, 24, 20)
            header_layout.setSpacing(10)

            eyebrow = QLabel("Calibration", header)
            eyebrow.setObjectName("HeroEyebrow")
            header_layout.addWidget(eyebrow)

            title = QLabel("Unified energy + FWHM calibration workspace", header)
            title.setObjectName("HeroHeader")
            header_layout.addWidget(title)

            subtitle = QLabel(
                (
                    "Residuals stay visible while the spectrum, calibration points, "
                    "and fit diagnostics update in one place."
                ),
                header,
            )
            subtitle.setObjectName("HeroSubhead")
            subtitle.setWordWrap(True)
            header_layout.addWidget(subtitle)

            kpi_row = QHBoxLayout()
            kpi_row.setSpacing(12)
            self.spectrum_kpi = self._build_kpi_card("Spectrum", "No spectrum")
            self.energy_kpi = self._build_kpi_card("Energy Fit", "Waiting")
            self.fwhm_kpi = self._build_kpi_card("Resolution Fit", "Waiting")
            kpi_row.addWidget(self.spectrum_kpi[0], 1)
            kpi_row.addWidget(self.energy_kpi[0], 1)
            kpi_row.addWidget(self.fwhm_kpi[0], 1)
            header_layout.addLayout(kpi_row)
            root.addWidget(header)

            splitter = QSplitter(Qt.Horizontal, self)
            splitter.setChildrenCollapsible(False)
            root.addWidget(splitter, 1)

            plots_panel = QWidget(splitter)
            plots_layout = QVBoxLayout(plots_panel)
            plots_layout.setContentsMargins(0, 0, 0, 0)
            plots_layout.setSpacing(12)

            self.spectrum_plot = self._create_plot_widget(
                "Spectrum canvas",
                "Channel",
                "Counts",
                "CalibrationSpectrumPlot",
            )
            self.spectrum_curve = self.spectrum_plot.plot(
                pen=pg.mkPen(width=2),
                fillLevel=0.0,
            )
            self.spectrum_markers = self.spectrum_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=10,
            )
            self.spectrum_roi_fit_curve = self.spectrum_plot.plot(
                pen=pg.mkPen(width=2),
            )
            self.spectrum_roi_background_curve = self.spectrum_plot.plot(
                pen=pg.mkPen(width=1, style=Qt.DashLine),
            )
            self.quick_anchor_a_line = pg.InfiniteLine(angle=90, movable=False)
            self.quick_anchor_b_line = pg.InfiniteLine(angle=90, movable=False)
            self.roi_centroid_line = pg.InfiniteLine(angle=90, movable=False)
            self.quick_anchor_a_line.setVisible(False)
            self.quick_anchor_b_line.setVisible(False)
            self.roi_centroid_line.setVisible(False)
            self.roi_region = pg.LinearRegionItem(
                values=self._initial_roi_bounds(),
                movable=True,
            )
            self.spectrum_plot.addItem(self.quick_anchor_a_line)
            self.spectrum_plot.addItem(self.quick_anchor_b_line)
            self.spectrum_plot.addItem(self.roi_region)
            self.spectrum_plot.addItem(self.roi_centroid_line)
            self.roi_region.sigRegionChanged.connect(self._handle_roi_region_changed)
            self.spectrum_plot.scene().sigMouseClicked.connect(
                self._handle_spectrum_click
            )
            plots_layout.addWidget(
                self._wrap_plot_card(
                    "Embedded spectrum canvas",
                    (
                        "Click the spectrum to update the selected calibration row. "
                        "Drag the ROI band for live Gaussian or skew fitting."
                    ),
                    self.spectrum_plot,
                ),
                5,
            )

            lower_splitter = QSplitter(Qt.Vertical, plots_panel)
            lower_splitter.setChildrenCollapsible(False)
            plots_layout.addWidget(lower_splitter, 4)

            self.energy_residual_plot = self._create_plot_widget(
                "Energy residuals",
                "Reference energy (keV)",
                "Residual (keV)",
                "CalibrationEnergyResidualPlot",
            )
            self.energy_residual_in_spec = self.energy_residual_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=9,
            )
            self.energy_residual_out_spec = self.energy_residual_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=10,
            )
            self.energy_residual_zero = pg.InfiniteLine(
                pos=0.0,
                angle=0,
                movable=False,
            )
            self.energy_residual_upper = pg.InfiniteLine(
                pos=ASTM_E181_ENERGY_LIMIT_KEV,
                angle=0,
                movable=False,
            )
            self.energy_residual_lower = pg.InfiniteLine(
                pos=-ASTM_E181_ENERGY_LIMIT_KEV,
                angle=0,
                movable=False,
            )
            self.energy_residual_band = pg.LinearRegionItem(
                values=[-ASTM_E181_ENERGY_LIMIT_KEV, ASTM_E181_ENERGY_LIMIT_KEV],
                orientation=pg.LinearRegionItem.Horizontal,
                movable=False,
            )
            self.energy_residual_plot.addItem(self.energy_residual_band)
            self.energy_residual_plot.addItem(self.energy_residual_zero)
            self.energy_residual_plot.addItem(self.energy_residual_upper)
            self.energy_residual_plot.addItem(self.energy_residual_lower)
            lower_splitter.addWidget(
                self._wrap_plot_card(
                    "Residuals first",
                    "ASTM E181 acceptance band is shown at +/- 0.5 keV.",
                    self.energy_residual_plot,
                )
            )

            self.fwhm_plot = self._create_plot_widget(
                "FWHM resolution fit",
                "Energy (keV)",
                "FWHM (keV)",
                "CalibrationFwhmPlot",
            )
            self.fwhm_measured_curve = self.fwhm_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=9,
            )
            self.fwhm_fit_curve = self.fwhm_plot.plot(
                pen=pg.mkPen(width=2),
            )
            lower_splitter.addWidget(
                self._wrap_plot_card(
                    "Resolution curve",
                    "The FWHM model updates alongside the energy fit.",
                    self.fwhm_plot,
                )
            )

            self.roi_detail_plot = self._create_plot_widget(
                "ROI fit preview",
                "Channel",
                "Counts",
                "CalibrationRoiFitPlot",
            )
            self.roi_detail_observed_curve = self.roi_detail_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=7,
            )
            self.roi_detail_fit_curve = self.roi_detail_plot.plot(
                pen=pg.mkPen(width=2),
            )
            self.roi_detail_background_curve = self.roi_detail_plot.plot(
                pen=pg.mkPen(width=1, style=Qt.DashLine),
            )
            self.roi_residual_plot = self._create_plot_widget(
                "ROI residuals",
                "Channel",
                "(data-model) / sqrt(model)",
                "CalibrationRoiResidualPlot",
            )
            self.roi_residual_curve = self.roi_residual_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=7,
            )
            self.roi_residual_zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False)
            self.roi_residual_plot.addItem(self.roi_residual_zero)
            roi_plot_container = QWidget(plots_panel)
            roi_plot_layout = QVBoxLayout(roi_plot_container)
            roi_plot_layout.setContentsMargins(0, 0, 0, 0)
            roi_plot_layout.setSpacing(10)
            roi_plot_layout.addWidget(self.roi_detail_plot, 3)
            roi_plot_layout.addWidget(self.roi_residual_plot, 2)
            lower_splitter.addWidget(
                self._wrap_plot_card(
                    "ROI peak fitter",
                    "Drag the ROI band on the canvas and compare data, fit, and normalized residuals live.",
                    roi_plot_container,
                )
            )

            controls_panel = QWidget(splitter)
            controls_scroll = QScrollArea(splitter)
            controls_scroll.setObjectName("CalibrationControlsScrollArea")
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setWidget(controls_panel)
            controls_layout = QVBoxLayout(controls_panel)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(12)

            energy_group = QGroupBox("Energy calibration", controls_panel)
            energy_group.setObjectName("CalibrationGroup")
            energy_layout = QVBoxLayout(energy_group)
            energy_layout.setSpacing(10)

            order_row = QHBoxLayout()
            order_row.setSpacing(8)
            order_label = QLabel("Polynomial order", energy_group)
            order_row.addWidget(order_label)
            self.energy_order = QSpinBox(energy_group)
            self.energy_order.setObjectName("CalibrationEnergyPolynomialOrderSpin")
            self.energy_order.setRange(1, 4)
            self.energy_order.setValue(2)
            self.energy_order.valueChanged.connect(self._refresh_energy_fit)
            order_row.addWidget(self.energy_order)
            self.energy_lock_label = QLabel("", energy_group)
            self.energy_lock_label.setObjectName("CalibrationLockLabel")
            self.energy_lock_label.setWordWrap(True)
            order_row.addWidget(self.energy_lock_label, 1)
            energy_layout.addLayout(order_row)

            energy_button_row = QHBoxLayout()
            self.add_energy_point_button = QPushButton("Add point", energy_group)
            self.add_energy_point_button.setObjectName(
                "AddEnergyCalibrationPointButton"
            )
            self.add_energy_point_button.clicked.connect(self._add_empty_energy_row)
            energy_button_row.addWidget(self.add_energy_point_button)
            self.remove_energy_point_button = QPushButton(
                "Remove selected", energy_group
            )
            self.remove_energy_point_button.setObjectName(
                "RemoveEnergyCalibrationPointButton"
            )
            self.remove_energy_point_button.clicked.connect(
                self._remove_selected_energy_rows
            )
            energy_button_row.addWidget(self.remove_energy_point_button)
            self.reset_energy_points_button = QPushButton(
                "Reset seeded points", energy_group
            )
            self.reset_energy_points_button.setObjectName(
                "ResetEnergyCalibrationPointsButton"
            )
            self.reset_energy_points_button.clicked.connect(self._seed_energy_table)
            energy_button_row.addWidget(self.reset_energy_points_button)
            energy_button_row.addStretch(1)
            energy_layout.addLayout(energy_button_row)

            self.energy_table = QTableWidget(0, len(self.ENERGY_HEADERS), energy_group)
            self.energy_table.setObjectName("CalibrationEnergyPointsTable")
            self.energy_table.setHorizontalHeaderLabels(self.ENERGY_HEADERS)
            self.energy_table.verticalHeader().setVisible(False)
            self.energy_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.energy_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.energy_table.itemChanged.connect(self._handle_energy_table_change)
            energy_header = self.energy_table.horizontalHeader()
            energy_header.setSectionResizeMode(QHeaderView.Stretch)
            energy_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            energy_layout.addWidget(self.energy_table, 1)

            self.energy_summary = QLabel(
                "Waiting for enough calibration points to fit the energy polynomial.",
                energy_group,
            )
            self.energy_summary.setObjectName("PanelBody")
            self.energy_summary.setWordWrap(True)
            energy_layout.addWidget(self.energy_summary)
            controls_layout.addWidget(energy_group, 3)

            fwhm_group = QGroupBox("FWHM calibration", controls_panel)
            fwhm_group.setObjectName("CalibrationGroup")
            fwhm_layout = QVBoxLayout(fwhm_group)
            fwhm_layout.setSpacing(10)

            fwhm_button_row = QHBoxLayout()
            self.add_fwhm_point_button = QPushButton("Add point", fwhm_group)
            self.add_fwhm_point_button.setObjectName("AddFwhmCalibrationPointButton")
            self.add_fwhm_point_button.clicked.connect(self._add_empty_fwhm_row)
            fwhm_button_row.addWidget(self.add_fwhm_point_button)
            self.remove_fwhm_point_button = QPushButton("Remove selected", fwhm_group)
            self.remove_fwhm_point_button.setObjectName(
                "RemoveFwhmCalibrationPointButton"
            )
            self.remove_fwhm_point_button.clicked.connect(
                self._remove_selected_fwhm_rows
            )
            fwhm_button_row.addWidget(self.remove_fwhm_point_button)
            self.reset_fwhm_points_button = QPushButton(
                "Reset seeded points", fwhm_group
            )
            self.reset_fwhm_points_button.setObjectName(
                "ResetFwhmCalibrationPointsButton"
            )
            self.reset_fwhm_points_button.clicked.connect(self._seed_fwhm_table)
            fwhm_button_row.addWidget(self.reset_fwhm_points_button)
            fwhm_button_row.addStretch(1)
            fwhm_layout.addLayout(fwhm_button_row)

            self.fwhm_table = QTableWidget(0, len(self.FWHM_HEADERS), fwhm_group)
            self.fwhm_table.setObjectName("CalibrationFwhmPointsTable")
            self.fwhm_table.setHorizontalHeaderLabels(self.FWHM_HEADERS)
            self.fwhm_table.verticalHeader().setVisible(False)
            self.fwhm_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.fwhm_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.fwhm_table.itemChanged.connect(self._handle_fwhm_table_change)
            fwhm_header = self.fwhm_table.horizontalHeader()
            fwhm_header.setSectionResizeMode(QHeaderView.Stretch)
            fwhm_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            fwhm_layout.addWidget(self.fwhm_table, 1)

            self.fwhm_summary = QLabel(
                "Waiting for enough FWHM points to fit the detector resolution curve.",
                fwhm_group,
            )
            self.fwhm_summary.setObjectName("PanelBody")
            self.fwhm_summary.setWordWrap(True)
            fwhm_layout.addWidget(self.fwhm_summary)
            controls_layout.addWidget(fwhm_group, 2)

            library_group = QGroupBox(
                "Library-assisted point assignment", controls_panel
            )
            library_group.setObjectName("CalibrationGroup")
            library_layout = QVBoxLayout(library_group)
            library_layout.setSpacing(10)

            library_intro = QLabel(
                (
                    "Search the active identification library, overlay its reference lines, "
                    "and assign a selected line to the current calibration row."
                ),
                library_group,
            )
            library_intro.setObjectName("PanelBody")
            library_intro.setWordWrap(True)
            library_layout.addWidget(library_intro)

            self.library_source_combo = QComboBox(library_group)
            self.library_source_combo.setObjectName("CalibrationLibrarySourceCombo")
            library_layout.addWidget(self.library_source_combo)

            self.library_search = QLineEdit(library_group)
            self.library_search.setObjectName("CalibrationLibrarySearchInput")
            self.library_search.setPlaceholderText("Search calibration library...")
            library_layout.addWidget(self.library_search)

            library_lists = QHBoxLayout()
            library_lists.setSpacing(10)

            self.library_results = QListWidget(library_group)
            self.library_results.setObjectName("CalibrationLibraryResults")
            library_lists.addWidget(self.library_results, 3)

            self.library_lines = QListWidget(library_group)
            self.library_lines.setObjectName("CalibrationLibraryLines")
            library_lists.addWidget(self.library_lines, 2)

            library_layout.addLayout(library_lists)

            self.assign_line_button = QPushButton(
                "Assign line to selected row",
                library_group,
            )
            self.assign_line_button.setObjectName("AssignCalibrationLibraryLineButton")
            self.assign_line_button.clicked.connect(self._assign_selected_library_line)
            library_layout.addWidget(self.assign_line_button)

            self.library_summary = QLabel("", library_group)
            self.library_summary.setObjectName("PanelBody")
            self.library_summary.setWordWrap(True)
            library_layout.addWidget(self.library_summary)
            controls_layout.addWidget(library_group, 2)

            advanced_group = QGroupBox("Advanced calibration tools", controls_panel)
            advanced_group.setObjectName("CalibrationGroup")
            advanced_layout = QVBoxLayout(advanced_group)
            advanced_layout.setSpacing(10)
            self.advanced_tabs = QTabWidget(advanced_group)
            self.advanced_tabs.setObjectName("CalibrationAdvancedTabs")
            self.quick_slider_tab = self._build_quick_slider_tab(self.advanced_tabs)
            self.deviation_pairs_tab = self._build_deviation_pairs_tab(
                self.advanced_tabs
            )
            self.preserve_slots_tab = self._build_preserve_slots_tab(self.advanced_tabs)
            self.roi_fit_tab = self._build_roi_fit_tab(self.advanced_tabs)
            self.advanced_tabs.addTab(self.quick_slider_tab, "Quick Slider")
            self.advanced_tabs.addTab(self.deviation_pairs_tab, "Fine Tuning")
            self.advanced_tabs.addTab(self.preserve_slots_tab, "Preserve && Slots")
            self.advanced_tabs.addTab(self.roi_fit_tab, "ROI Fit")
            self.advanced_tabs.setProperty(
                "fluxforgeTabIds",
                {
                    "Quick Slider": "calibration.quick_slider.open",
                    "Fine Tuning": "calibration.fine_tuning.open",
                    "Preserve && Slots": "calibration.preserve_slots.open",
                    "ROI Fit": "calibration.roi_fit.open",
                },
            )
            advanced_layout.addWidget(self.advanced_tabs)
            controls_layout.addWidget(advanced_group, 3)

            provenance = QFrame(controls_panel)
            provenance.setObjectName("HeroCard")
            provenance_layout = QVBoxLayout(provenance)
            provenance_layout.setContentsMargins(18, 18, 18, 18)
            provenance_layout.setSpacing(6)
            provenance_title = QLabel("Workflow provenance", provenance)
            provenance_title.setObjectName("HeroCardTitle")
            provenance_layout.addWidget(provenance_title)
            self.provenance_summary = QLabel("", provenance)
            self.provenance_summary.setObjectName("HeroCardBody")
            self.provenance_summary.setWordWrap(True)
            provenance_layout.addWidget(self.provenance_summary)
            controls_layout.addWidget(provenance)

            action_row = QHBoxLayout()
            action_row.addStretch(1)
            self.apply_button = QPushButton(
                "Apply calibration to spectrum", controls_panel
            )
            self.apply_button.setObjectName("ApplyCalibrationButton")
            self.apply_button.clicked.connect(self._apply_workspace_results)
            action_row.addWidget(self.apply_button)
            self.close_button = QPushButton("Close", controls_panel)
            self.close_button.setObjectName("CloseCalibrationDialogButton")
            self.close_button.clicked.connect(self.close)
            action_row.addWidget(self.close_button)
            controls_layout.addLayout(action_row)

            splitter.addWidget(plots_panel)
            splitter.addWidget(controls_scroll)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)

            self.library_manager.subscribe(self._sync_library_state)
            self.library_source_combo.currentIndexChanged.connect(
                self._library_source_changed
            )
            self.library_search.textChanged.connect(self._refresh_library_results)
            self.library_results.itemSelectionChanged.connect(
                self._library_result_selected
            )
            self.library_lines.itemDoubleClicked.connect(
                lambda _item: self._assign_selected_library_line()
            )
            self._populate_library_sources()
            self._sync_library_state(self.library_manager.state)

        def _build_quick_slider_tab(self, parent: QWidget) -> QWidget:
            widget = QWidget(parent)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "PeakEasy-style quick anchors provide a fast linear preview that can "
                    "be promoted into the full calibration table without replacing it."
                ),
                widget,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(8)

            self.quick_anchor_a_combo = QComboBox(widget)
            self.quick_anchor_a_combo.setObjectName("CalibrationQuickAnchorALineCombo")
            self.quick_anchor_a_combo.currentIndexChanged.connect(
                self._refresh_quick_slider_preview
            )
            form.addRow("Anchor A line", self.quick_anchor_a_combo)

            self.quick_anchor_a_slider = QSlider(Qt.Horizontal, widget)
            self.quick_anchor_a_slider.setObjectName(
                "CalibrationQuickAnchorAChannelSlider"
            )
            self.quick_anchor_a_slider.valueChanged.connect(
                self._refresh_quick_slider_preview
            )
            form.addRow("Anchor A channel", self.quick_anchor_a_slider)

            self.quick_anchor_a_label = QLabel("--", widget)
            self.quick_anchor_a_label.setObjectName("PanelBody")
            form.addRow("Anchor A preview", self.quick_anchor_a_label)

            self.quick_anchor_b_combo = QComboBox(widget)
            self.quick_anchor_b_combo.setObjectName("CalibrationQuickAnchorBLineCombo")
            self.quick_anchor_b_combo.currentIndexChanged.connect(
                self._refresh_quick_slider_preview
            )
            form.addRow("Anchor B line", self.quick_anchor_b_combo)

            self.quick_anchor_b_slider = QSlider(Qt.Horizontal, widget)
            self.quick_anchor_b_slider.setObjectName(
                "CalibrationQuickAnchorBChannelSlider"
            )
            self.quick_anchor_b_slider.valueChanged.connect(
                self._refresh_quick_slider_preview
            )
            form.addRow("Anchor B channel", self.quick_anchor_b_slider)

            self.quick_anchor_b_label = QLabel("--", widget)
            self.quick_anchor_b_label.setObjectName("PanelBody")
            form.addRow("Anchor B preview", self.quick_anchor_b_label)
            layout.addLayout(form)

            button_row = QHBoxLayout()
            button_row.setSpacing(8)
            self.quick_reset_button = QPushButton("Reset to seeded anchors", widget)
            self.quick_reset_button.setObjectName("ResetCalibrationQuickAnchorsButton")
            self.quick_reset_button.clicked.connect(self._reset_quick_slider_anchors)
            button_row.addWidget(self.quick_reset_button)
            self.quick_promote_button = QPushButton("Promote anchors to table", widget)
            self.quick_promote_button.setObjectName(
                "PromoteCalibrationQuickAnchorsButton"
            )
            self.quick_promote_button.clicked.connect(
                self._promote_quick_slider_to_energy_rows
            )
            button_row.addWidget(self.quick_promote_button)
            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.quick_preview_summary = QLabel("", widget)
            self.quick_preview_summary.setObjectName("PanelBody")
            self.quick_preview_summary.setWordWrap(True)
            layout.addWidget(self.quick_preview_summary)
            self._populate_quick_reference_options()
            self._reset_quick_slider_anchors()
            return widget

        def _build_deviation_pairs_tab(self, parent: QWidget) -> QWidget:
            widget = QWidget(parent)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Fine calibration pairs apply piecewise-linear energy corrections on "
                    "top of the polynomial fit for InterSpec-style local cleanup."
                ),
                widget,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            button_row = QHBoxLayout()
            button_row.setSpacing(8)
            self.add_deviation_pair_button = QPushButton("Add pair", widget)
            self.add_deviation_pair_button.setObjectName(
                "AddCalibrationDeviationPairButton"
            )
            self.add_deviation_pair_button.clicked.connect(
                self._add_empty_deviation_row
            )
            button_row.addWidget(self.add_deviation_pair_button)
            self.remove_deviation_pair_button = QPushButton("Remove selected", widget)
            self.remove_deviation_pair_button.setObjectName(
                "RemoveCalibrationDeviationPairButton"
            )
            self.remove_deviation_pair_button.clicked.connect(
                self._remove_selected_deviation_rows
            )
            button_row.addWidget(self.remove_deviation_pair_button)
            self.seed_deviation_pairs_button = QPushButton(
                "Seed from residuals", widget
            )
            self.seed_deviation_pairs_button.setObjectName(
                "SeedCalibrationDeviationPairsButton"
            )
            self.seed_deviation_pairs_button.clicked.connect(
                self._seed_deviation_pairs_from_residuals
            )
            button_row.addWidget(self.seed_deviation_pairs_button)
            self.clear_deviation_pairs_button = QPushButton("Clear", widget)
            self.clear_deviation_pairs_button.setObjectName(
                "ClearCalibrationDeviationPairsButton"
            )
            self.clear_deviation_pairs_button.clicked.connect(
                self._clear_deviation_pairs
            )
            button_row.addWidget(self.clear_deviation_pairs_button)
            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.deviation_table = QTableWidget(0, 3, widget)
            self.deviation_table.setObjectName("CalibrationDeviationPairsTable")
            self.deviation_table.setHorizontalHeaderLabels(
                ("Energy keV", "Correction keV", "Label")
            )
            self.deviation_table.verticalHeader().setVisible(False)
            self.deviation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.deviation_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.deviation_table.itemChanged.connect(
                self._handle_deviation_table_change
            )
            deviation_header = self.deviation_table.horizontalHeader()
            deviation_header.setSectionResizeMode(QHeaderView.Stretch)
            deviation_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            layout.addWidget(self.deviation_table, 1)

            self.deviation_summary = QLabel(
                "No deviation pairs are active. The polynomial fit is currently unwarped.",
                widget,
            )
            self.deviation_summary.setObjectName("PanelBody")
            self.deviation_summary.setWordWrap(True)
            layout.addWidget(self.deviation_summary)
            return widget

        def _build_preserve_slots_tab(self, parent: QWidget) -> QWidget:
            widget = QWidget(parent)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Preserve the current calibration, recall named detector slots, and "
                    "fine-tune a prior solution against the current spectrum without "
                    "rebuilding every row by hand."
                ),
                widget,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(8)

            self.detector_slot_combo = QComboBox(widget)
            self.detector_slot_combo.setObjectName("CalibrationDetectorSlotCombo")
            for slot_name in DETECTOR_SLOT_NAMES:
                self.detector_slot_combo.addItem(slot_name, slot_name)
            self.detector_slot_combo.currentIndexChanged.connect(
                self._refresh_snapshot_summary
            )
            form.addRow("Detector slot", self.detector_slot_combo)
            layout.addLayout(form)

            button_row = QHBoxLayout()
            button_row.setSpacing(8)

            self.preserve_current_button = QPushButton("Preserve current", widget)
            self.preserve_current_button.setObjectName("PreserveCalibrationButton")
            self.preserve_current_button.clicked.connect(
                self._preserve_current_calibration
            )
            button_row.addWidget(self.preserve_current_button)

            self.fine_tune_preserved_button = QPushButton("Fine-tune preserved", widget)
            self.fine_tune_preserved_button.setObjectName("FineTunePreservedButton")
            self.fine_tune_preserved_button.clicked.connect(
                self._fine_tune_from_preserved
            )
            button_row.addWidget(self.fine_tune_preserved_button)

            self.save_slot_button = QPushButton("Save to slot", widget)
            self.save_slot_button.setObjectName("SaveDetectorSlotButton")
            self.save_slot_button.clicked.connect(self._save_current_to_detector_slot)
            button_row.addWidget(self.save_slot_button)

            self.load_slot_button = QPushButton("Load slot", widget)
            self.load_slot_button.setObjectName("LoadDetectorSlotButton")
            self.load_slot_button.clicked.connect(self._load_detector_slot)
            button_row.addWidget(self.load_slot_button)

            self.nasa_smart_seed_button = QPushButton("NASA smart seed", widget)
            self.nasa_smart_seed_button.setObjectName("NasaSmartSeedButton")
            self.nasa_smart_seed_button.clicked.connect(self._apply_nasa_smart_seed)
            button_row.addWidget(self.nasa_smart_seed_button)

            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.snapshot_summary = QLabel("", widget)
            self.snapshot_summary.setObjectName("CalibrationSnapshotSummary")
            self.snapshot_summary.setWordWrap(True)
            layout.addWidget(self.snapshot_summary)

            self._refresh_snapshot_summary()
            return widget

        def _build_roi_fit_tab(self, parent: QWidget) -> QWidget:
            widget = QWidget(parent)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            intro = QLabel(
                (
                    "Drag the ROI directly on the spectrum canvas. The fitter updates in "
                    "real time and can push the centroid or FWHM into the calibration tables."
                ),
                widget,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.roi_method_selector = MethodSelectorWidget(
                self._peak_fitter_registry,
                self.mode_manager,
                title="Peak fitter",
                parent=widget,
            )
            self.roi_method_selector.combo.setObjectName(
                "CalibrationRoiPeakFitterCombo"
            )
            self.roi_method_selector.combo.currentIndexChanged.connect(
                self._handle_roi_fitter_changed
            )
            layout.addWidget(self.roi_method_selector)

            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(8)
            self.roi_background_combo = QComboBox(widget)
            self.roi_background_combo.setObjectName("CalibrationRoiBackgroundCombo")
            self.roi_background_combo.currentIndexChanged.connect(self._refresh_roi_fit)
            form.addRow("Background", self.roi_background_combo)
            layout.addLayout(form)

            button_row = QHBoxLayout()
            button_row.setSpacing(8)
            self.snap_roi_button = QPushButton("Snap ROI to selected point", widget)
            self.snap_roi_button.setObjectName("SnapCalibrationRoiButton")
            self.snap_roi_button.clicked.connect(self._snap_roi_to_selected_energy_row)
            button_row.addWidget(self.snap_roi_button)
            self.apply_roi_energy_button = QPushButton(
                "Use centroid for energy row", widget
            )
            self.apply_roi_energy_button.setObjectName(
                "ApplyCalibrationRoiCentroidButton"
            )
            self.apply_roi_energy_button.clicked.connect(
                self._apply_roi_fit_to_selected_energy_row
            )
            button_row.addWidget(self.apply_roi_energy_button)
            self.apply_roi_fwhm_button = QPushButton(
                "Use FWHM for resolution row", widget
            )
            self.apply_roi_fwhm_button.setObjectName("ApplyCalibrationRoiFwhmButton")
            self.apply_roi_fwhm_button.clicked.connect(
                self._apply_roi_fit_to_selected_fwhm_row
            )
            button_row.addWidget(self.apply_roi_fwhm_button)
            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.roi_fit_summary = QLabel(
                "Move the ROI band to generate a live fit.",
                widget,
            )
            self.roi_fit_summary.setObjectName("PanelBody")
            self.roi_fit_summary.setWordWrap(True)
            layout.addWidget(self.roi_fit_summary)

            self._refresh_roi_background_models()
            return widget

        def _build_kpi_card(self, label: str, value: str) -> tuple[QFrame, QLabel]:
            card = QFrame(self)
            card.setObjectName("CalibrationKpiCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(4)
            label_widget = QLabel(label, card)
            label_widget.setObjectName("CalibrationKpiLabel")
            layout.addWidget(label_widget)
            value_widget = QLabel(value, card)
            value_widget.setObjectName("CalibrationKpiValue")
            value_widget.setWordWrap(True)
            layout.addWidget(value_widget)
            return card, value_widget

        def _wrap_plot_card(
            self,
            title: str,
            body: str,
            plot_widget: QWidget,
        ) -> QFrame:
            card = QFrame(self)
            card.setObjectName("HeroCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(10)
            title_label = QLabel(title, card)
            title_label.setObjectName("HeroCardTitle")
            layout.addWidget(title_label)
            body_label = QLabel(body, card)
            body_label.setObjectName("HeroCardBody")
            body_label.setWordWrap(True)
            layout.addWidget(body_label)
            layout.addWidget(plot_widget, 1)
            return card

        def _create_plot_widget(
            self,
            title: str,
            bottom_label: str,
            left_label: str,
            object_name: str,
        ) -> pg.PlotWidget:
            widget = pg.PlotWidget(self)
            widget.setObjectName(object_name)
            catalog_pyqtgraph_export_action(widget, object_name)
            widget.showGrid(x=True, y=True, alpha=0.12)
            widget.setMenuEnabled(False)
            widget.setMouseEnabled(x=True, y=True)
            widget.setTitle(title)
            widget.setLabel("bottom", bottom_label)
            widget.setLabel("left", left_label)
            return widget

        def _initial_roi_bounds(self) -> tuple[float, float]:
            counts = np.asarray(self._spectrum.counts, dtype=float)
            if counts.size == 0:
                return (0.0, 32.0)
            reference_channel = self._snap_channel_to_peak(min(661.0, counts.size - 1))
            half_width = max(min(counts.size / 40.0, 24.0), 12.0)
            return (
                float(max(reference_channel - half_width, 0.0)),
                float(min(reference_channel + half_width, counts.size - 1)),
            )

        def set_active_advanced_tab(self, key: str) -> None:
            widget_map = {
                "quick_slider": self.quick_slider_tab,
                "deviation_pairs": self.deviation_pairs_tab,
                "preserve_slots": self.preserve_slots_tab,
                "roi_fit": self.roi_fit_tab,
            }
            widget = widget_map.get(key)
            if widget is not None:
                self.advanced_tabs.setCurrentWidget(widget)

        def _on_mode_state_changed(self, state) -> None:
            resolved = resolve_energy_calibration_order(
                self.energy_order.value(),
                standard=state.standard,
            )
            self.energy_order.blockSignals(True)
            self.energy_order.setValue(resolved.order)
            self.energy_order.setEnabled(resolved.locked_by is None)
            self.energy_order.blockSignals(False)
            self.energy_lock_label.setText(
                resolved.locked_by or "Expert-mode order selection is available."
            )
            self.provenance_summary.setText(
                self._format_provenance_summary(resolved.locked_by)
            )
            simple_mode = state.mode.value == "simple"
            if hasattr(self.advanced_tabs, "setTabVisible"):
                self.advanced_tabs.setTabVisible(
                    self.advanced_tabs.indexOf(self.deviation_pairs_tab),
                    not simple_mode,
                )
                self.advanced_tabs.setTabVisible(
                    self.advanced_tabs.indexOf(self.preserve_slots_tab),
                    not simple_mode,
                )
                self.advanced_tabs.setTabVisible(
                    self.advanced_tabs.indexOf(self.roi_fit_tab),
                    not simple_mode,
                )
            if simple_mode and self.advanced_tabs.currentWidget() in (
                self.deviation_pairs_tab,
                self.preserve_slots_tab,
                self.roi_fit_tab,
            ):
                self.advanced_tabs.setCurrentWidget(self.quick_slider_tab)
            self._apply_plot_palette()
            self._populate_library_sources()
            self._sync_library_state(self.library_manager.state)
            self._refresh_energy_fit()
            self._refresh_roi_background_models()
            self._refresh_quick_slider_preview()
            self._refresh_snapshot_summary()
            self._refresh_roi_fit()

        def _format_provenance_summary(self, locked_by: str | None) -> str:
            spectrum_name = self._spectrum.spectrum_id or "Untitled spectrum"
            mode_state = self.mode_manager.state
            standard = (
                mode_state.standard if mode_state.mode.value == "standards" else None
            )
            calibration_library = self.library_manager.record_for_category(
                "calibration",
                standard=standard,
            ).label
            identification_library = self.library_manager.record_for_category(
                "gamma_identification",
                standard=standard,
            ).label
            lines = [
                f"Spectrum: {spectrum_name}",
                f"Mode: {mode_state.mode.value.title()}",
                (
                    f"Standard: {mode_state.standard}"
                    if mode_state.standard
                    else "Standard: none"
                ),
            ]
            if locked_by:
                lines.append(f"Order lock: {locked_by}")
            else:
                lines.append("Order lock: none")
            lines.append(f"Calibration sources: {calibration_library}")
            lines.append(f"Identification library: {identification_library}")
            lines.append(
                "Residual-first rule: energy residuals remain visible with the ASTM band."
            )
            return "\n".join(lines)

        def _populate_library_sources(self) -> None:
            self.library_source_combo.blockSignals(True)
            self.library_source_combo.clear()
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode.value == "standards"
                else None
            )
            for record in self.library_manager.available_sources(
                "gamma_identification",
                standard=standard,
            ):
                self.library_source_combo.addItem(record.label, record.source_id)
            self.library_source_combo.blockSignals(False)

        def _reference_line_options(self) -> tuple[tuple[str, float], ...]:
            options: list[tuple[str, float]] = list(REFERENCE_LINES_KEV)
            for energy in self.selection_bus.state.reference_lines_keV or ():
                options.append((f"Active line {energy:.3f} keV", float(energy)))
            unique: dict[float, tuple[str, float]] = {}
            for label, energy in options:
                unique[round(float(energy), 6)] = (label, float(energy))
            return tuple(sorted(unique.values(), key=lambda item: item[1]))

        def _populate_quick_reference_options(self) -> None:
            current_a = (
                float(self.quick_anchor_a_combo.currentData())
                if self.quick_anchor_a_combo.count()
                else None
            )
            current_b = (
                float(self.quick_anchor_b_combo.currentData())
                if self.quick_anchor_b_combo.count()
                else None
            )
            options = self._reference_line_options()
            for combo, fallback in (
                (self.quick_anchor_a_combo, REFERENCE_LINES_KEV[0][1]),
                (self.quick_anchor_b_combo, REFERENCE_LINES_KEV[-1][1]),
            ):
                combo.blockSignals(True)
                combo.clear()
                for label, energy in options:
                    combo.addItem(f"{label} · {energy:.3f} keV", float(energy))
                combo.blockSignals(False)
                target = current_a if combo is self.quick_anchor_a_combo else current_b
                if target is None:
                    target = float(fallback)
                index = combo.findData(float(target))
                if index < 0 and combo.count() > 0:
                    index = (
                        0 if combo is self.quick_anchor_a_combo else combo.count() - 1
                    )
                if index >= 0:
                    combo.setCurrentIndex(index)

        def _sync_library_state(self, state) -> None:
            standard = (
                self.mode_manager.state.standard
                if self.mode_manager.state.mode.value == "standards"
                else None
            )
            resolved_state = self.library_manager.resolved_state(standard=standard)
            self.nuclide_controller.set_source(
                resolved_state.gamma_identification_source_id,
                custom_path=resolved_state.custom_gamma_path,
            )
            index = self.library_source_combo.findData(
                resolved_state.gamma_identification_source_id
            )
            if index >= 0:
                self.library_source_combo.blockSignals(True)
                self.library_source_combo.setCurrentIndex(index)
                self.library_source_combo.blockSignals(False)
            self.library_source_combo.setEnabled(
                self.library_manager.locked_source_for_category(
                    "gamma_identification",
                    standard=standard,
                )
                is None
                and self.library_source_combo.count() > 1
            )
            self.library_summary.setText(
                self.library_manager.summary_for_category(
                    "gamma_identification",
                    standard=standard,
                )
            )
            self._refresh_library_results(self.library_search.text())
            self._populate_quick_reference_options()
            self.provenance_summary.setText(
                self._format_provenance_summary(
                    resolve_energy_calibration_order(
                        self.energy_order.value(),
                        standard=self.mode_manager.state.standard,
                    ).locked_by
                )
            )

        def _library_source_changed(self) -> None:
            source_id = self.library_source_combo.currentData()
            if source_id:
                self.library_manager.set_gamma_identification_source(str(source_id))

        def _refresh_library_results(self, query: str) -> None:
            self.library_results.clear()
            self.library_lines.clear()
            for hit in self.nuclide_controller.search(query or "c", limit=18):
                label = hit.display_name
                if hit.strongest_lines_keV:
                    label += " · " + ", ".join(
                        f"{energy:.3f}" for energy in hit.strongest_lines_keV[:3]
                    )
                    label += " keV"
                item = QListWidgetItem(label, self.library_results)
                item.setData(Qt.UserRole, hit.nuclide)

        def _library_result_selected(self) -> None:
            item = self.library_results.currentItem()
            if item is None:
                self.library_lines.clear()
                return
            nuclide = str(item.data(Qt.UserRole))
            state = self.nuclide_controller.activate(nuclide)
            self.library_lines.clear()
            for energy in state.reference_lines_keV:
                line_item = QListWidgetItem(
                    f"{nuclide} · {energy:.3f} keV",
                    self.library_lines,
                )
                line_item.setData(Qt.UserRole, (nuclide, float(energy)))
            self._populate_quick_reference_options()

        def _assign_selected_library_line(self) -> None:
            item = self.library_lines.currentItem()
            if item is None:
                return
            nuclide, energy = item.data(Qt.UserRole)
            row = self.energy_table.currentRow()
            if row < 0:
                row = self.energy_table.rowCount()
                self._add_empty_energy_row(select_row=False)
            self._set_energy_cell(row, 0, str(nuclide), editable=True)
            self._set_energy_cell(row, 3, float(energy), editable=True)
            if not self._table_text(self.energy_table, row, 4):
                self._set_energy_cell(row, 4, 0.12, editable=True)
            self.energy_table.selectRow(row)
            self._refresh_energy_fit()

        def _apply_plot_palette(self) -> None:
            tokens = theme_tokens(self.mode_manager.state.theme)
            base = tokens["canvas_background"]
            accent = tokens["accent"]
            accent_warm = tokens["accent_warm"]
            text = tokens["text_primary"]
            success = tokens["success"]
            error = tokens["error"]

            for plot in (
                self.spectrum_plot,
                self.energy_residual_plot,
                self.fwhm_plot,
            ):
                plot.setBackground(base)
                axis_pen = pg.mkPen(color=text)
                plot.getAxis("bottom").setTextPen(axis_pen)
                plot.getAxis("left").setTextPen(axis_pen)
                plot.getAxis("bottom").setPen(axis_pen)
                plot.getAxis("left").setPen(axis_pen)

            self.spectrum_curve.setPen(pg.mkPen(color=accent, width=2))
            self.spectrum_curve.setFillBrush(pg.mkBrush((0, 0, 0, 0)))
            self.spectrum_markers.setSymbolBrush(pg.mkBrush(accent_warm))
            self.spectrum_markers.setSymbolPen(pg.mkPen(color=accent_warm, width=1.5))

            self.energy_residual_in_spec.setSymbolBrush(pg.mkBrush(success))
            self.energy_residual_in_spec.setSymbolPen(
                pg.mkPen(color=success, width=1.5)
            )
            self.energy_residual_out_spec.setSymbolBrush(pg.mkBrush(error))
            self.energy_residual_out_spec.setSymbolPen(pg.mkPen(color=error, width=1.5))
            self.energy_residual_zero.setPen(pg.mkPen(color=text, width=1))
            self.energy_residual_upper.setPen(pg.mkPen(color=accent_warm, width=1))
            self.energy_residual_lower.setPen(pg.mkPen(color=accent_warm, width=1))
            self.energy_residual_band.setBrush(pg.mkBrush(244, 184, 96, 32))
            for line in self.energy_residual_band.lines:
                line.setPen(pg.mkPen(color=accent_warm, width=1))

            self.fwhm_measured_curve.setSymbolBrush(pg.mkBrush(accent_warm))
            self.fwhm_measured_curve.setSymbolPen(
                pg.mkPen(color=accent_warm, width=1.5)
            )
            self.fwhm_fit_curve.setPen(pg.mkPen(color=accent, width=2))
            self.spectrum_roi_fit_curve.setPen(pg.mkPen(color=success, width=2))
            self.spectrum_roi_background_curve.setPen(
                pg.mkPen(color=accent_warm, width=1, style=Qt.DashLine)
            )
            self.quick_anchor_a_line.setPen(pg.mkPen(color=accent_warm, width=1.5))
            self.quick_anchor_b_line.setPen(pg.mkPen(color=accent, width=1.5))
            self.roi_centroid_line.setPen(pg.mkPen(color=success, width=1))
            self.roi_region.setBrush(pg.mkBrush(26, 108, 192, 26))
            for line in self.roi_region.lines:
                line.setPen(pg.mkPen(color=accent, width=1))
            self.roi_detail_observed_curve.setSymbolBrush(pg.mkBrush(accent_warm))
            self.roi_detail_observed_curve.setSymbolPen(
                pg.mkPen(color=accent_warm, width=1.25)
            )
            self.roi_detail_fit_curve.setPen(pg.mkPen(color=success, width=2))
            self.roi_detail_background_curve.setPen(
                pg.mkPen(color=accent_warm, width=1, style=Qt.DashLine)
            )
            self.roi_residual_curve.setSymbolBrush(pg.mkBrush(accent))
            self.roi_residual_curve.setSymbolPen(pg.mkPen(color=accent, width=1.25))
            self.roi_residual_zero.setPen(pg.mkPen(color=text, width=1))

        def _reset_quick_slider_anchors(self) -> None:
            points = self._read_energy_points()[:2]
            if len(points) < 2:
                points = self._seed_energy_points()[:2]
            max_channel = max(len(self._spectrum.counts) - 1, 1)
            self._syncing_quick_controls = True
            self.quick_anchor_a_slider.setRange(0, max_channel)
            self.quick_anchor_b_slider.setRange(0, max_channel)
            anchor_a = float(points[0].channel) if len(points) >= 1 else 128.0
            anchor_b = (
                float(points[1].channel) if len(points) >= 2 else max_channel * 0.65
            )
            self.quick_anchor_a_slider.setValue(
                int(np.clip(round(anchor_a), 0, max_channel))
            )
            self.quick_anchor_b_slider.setValue(
                int(np.clip(round(anchor_b), 0, max_channel))
            )
            self._syncing_quick_controls = False
            self._refresh_quick_slider_preview()

        def _refresh_quick_slider_preview(self, *_args) -> None:
            if self._syncing_quick_controls:
                return
            if (
                self.quick_anchor_a_combo.count() == 0
                or self.quick_anchor_b_combo.count() == 0
            ):
                return
            anchor_channels = (
                float(self.quick_anchor_a_slider.value()),
                float(self.quick_anchor_b_slider.value()),
            )
            reference_energies = (
                float(self.quick_anchor_a_combo.currentData()),
                float(self.quick_anchor_b_combo.currentData()),
            )
            self.quick_anchor_a_line.setVisible(True)
            self.quick_anchor_b_line.setVisible(True)
            self.quick_anchor_a_line.setPos(anchor_channels[0])
            self.quick_anchor_b_line.setPos(anchor_channels[1])
            try:
                self._quick_fit = fit_quick_slider_calibration(
                    anchor_channels,
                    reference_energies,
                )
            except ValueError as exc:
                self.quick_preview_summary.setText(str(exc))
                self.quick_promote_button.setEnabled(False)
                return

            self.quick_promote_button.setEnabled(True)
            predicted = self._quick_fit.evaluate(
                np.asarray(anchor_channels, dtype=float)
            )
            self.quick_anchor_a_label.setText(
                f"ch {anchor_channels[0]:.0f} -> {predicted[0]:.3f} keV"
            )
            self.quick_anchor_b_label.setText(
                f"ch {anchor_channels[1]:.0f} -> {predicted[1]:.3f} keV"
            )
            standards_note = ""
            if self.mode_manager.state.standard:
                standards_note = " Standards mode still requires the locked polynomial workflow on apply."
            self.quick_preview_summary.setText(
                (
                    f"Linear preview: E = {self._quick_fit.coefficients[0]:.4f} + "
                    f"{self._quick_fit.slope_keV_per_channel:.6f} ch. "
                    f"Anchor separation: {abs(anchor_channels[1] - anchor_channels[0]):.0f} channels."
                    f"{standards_note}"
                )
            )

        def _promote_quick_slider_to_energy_rows(self) -> None:
            if self._quick_fit is None:
                return
            while self.energy_table.rowCount() < 2:
                self._add_empty_energy_row(select_row=False)
            previews = self._quick_fit.evaluate(
                np.asarray(self._quick_fit.anchor_channels, dtype=float)
            )
            for row, label, channel, preview_energy, reference_energy in (
                (
                    0,
                    "Quick Anchor A",
                    self._quick_fit.anchor_channels[0],
                    float(previews[0]),
                    self._quick_fit.reference_energies_keV[0],
                ),
                (
                    1,
                    "Quick Anchor B",
                    self._quick_fit.anchor_channels[1],
                    float(previews[1]),
                    self._quick_fit.reference_energies_keV[1],
                ),
            ):
                self._set_energy_cell(row, 0, label, editable=True)
                self._set_energy_cell(row, 1, channel)
                self._set_energy_cell(row, 2, preview_energy)
                self._set_energy_cell(row, 3, reference_energy)
                if not self._table_text(self.energy_table, row, 4):
                    self._set_energy_cell(row, 4, 0.12, editable=True)
            self.energy_table.selectRow(0)
            self._refresh_energy_fit()

        def _selected_detector_slot_name(self) -> str:
            return str(
                self.detector_slot_combo.currentData()
                or self.detector_slot_combo.currentText()
            )

        def _capture_current_snapshot(self, label: str) -> CalibrationSnapshot:
            return CalibrationSnapshot(
                label=label,
                energy_points=self._read_energy_points(),
                fwhm_points=self._read_fwhm_points(),
                deviation_pairs=self._read_deviation_pairs(),
            )

        def _seed_energy_table_from_points(
            self,
            points: tuple[EnergyCalibrationPoint, ...],
        ) -> None:
            self._syncing_energy_table = True
            self.energy_table.setRowCount(0)
            for point in points:
                self._append_energy_row(point)
            self._syncing_energy_table = False

        def _seed_fwhm_table_from_points(
            self,
            points: tuple[FWHMCalibrationPoint, ...],
        ) -> None:
            self._syncing_fwhm_table = True
            self.fwhm_table.setRowCount(0)
            for point in points:
                self._append_fwhm_row(point)
            self._syncing_fwhm_table = False

        def _seed_deviation_table_from_pairs(
            self,
            pairs: tuple[EnergyDeviationPair, ...],
        ) -> None:
            self._syncing_deviation_table = True
            self.deviation_table.setRowCount(0)
            for pair in pairs:
                row = self.deviation_table.rowCount()
                self.deviation_table.insertRow(row)
                self._set_deviation_cell(row, 0, pair.energy_keV)
                self._set_deviation_cell(row, 1, pair.correction_keV)
                self._set_deviation_cell(row, 2, pair.label)
            self._syncing_deviation_table = False

        def _restore_snapshot(
            self,
            snapshot: CalibrationSnapshot,
            *,
            message: str,
        ) -> None:
            self._seed_energy_table_from_points(snapshot.energy_points)
            self._seed_fwhm_table_from_points(snapshot.fwhm_points)
            self._seed_deviation_table_from_pairs(snapshot.deviation_pairs)
            if self.energy_table.rowCount():
                self.energy_table.selectRow(0)
            if self.fwhm_table.rowCount():
                self.fwhm_table.selectRow(0)
            self._reset_quick_slider_anchors()
            self._refresh_energy_fit()
            self._refresh_fwhm_fit()
            self._refresh_snapshot_summary(message=message)

        def _refresh_snapshot_summary(
            self,
            *_args,
            message: str | None = None,
        ) -> None:
            slot_name = self._selected_detector_slot_name()
            slot_snapshot = self._detector_slots.get(slot_name)
            preserved_label = (
                self._preserved_snapshot.label
                if self._preserved_snapshot is not None
                else "none"
            )
            if slot_snapshot is None:
                slot_text = f"{slot_name}: empty"
            else:
                slot_text = (
                    f"{slot_name}: {len(slot_snapshot.energy_points)} energy points, "
                    f"{len(slot_snapshot.fwhm_points)} resolution points, "
                    f"{len(slot_snapshot.deviation_pairs)} deviation pairs"
                )
            lines = [
                f"Preserved calibration: {preserved_label}",
                f"Detector slot status: {slot_text}",
                "NASA smart seed reassigns the seeded references using nasa_peaksearch.",
            ]
            if message:
                lines.append(message)
            self.snapshot_summary.setText("\n".join(lines))
            self.fine_tune_preserved_button.setEnabled(
                self._preserved_snapshot is not None
            )
            self.load_slot_button.setEnabled(slot_snapshot is not None)

        def _preserve_current_calibration(self) -> None:
            self._preserved_snapshot = self._capture_current_snapshot(
                "Current workspace"
            )
            self._refresh_snapshot_summary(
                message="Current calibration preserved for later fine-tuning."
            )

        def _save_current_to_detector_slot(self) -> None:
            slot_name = self._selected_detector_slot_name()
            self._detector_slots[slot_name] = self._capture_current_snapshot(slot_name)
            self._refresh_snapshot_summary(
                message=f"Stored the current calibration in detector slot '{slot_name}'."
            )

        def _load_detector_slot(self) -> None:
            slot_name = self._selected_detector_slot_name()
            snapshot = self._detector_slots.get(slot_name)
            if snapshot is None:
                self._refresh_snapshot_summary(
                    message=f"Detector slot '{slot_name}' is empty."
                )
                return
            self._restore_snapshot(
                snapshot,
                message=f"Loaded detector slot '{slot_name}' into the workspace tables.",
            )

        def _best_channel_guess(
            self,
            reference_energy_keV: float,
            *,
            fallback_channel: float,
        ) -> float:
            max_channel = max(len(self._spectrum.counts) - 1, 0)
            predicted_channel = float(fallback_channel)
            try:
                guessed = float(self._spectrum.energy_to_channel(reference_energy_keV))
                if np.isfinite(guessed) and abs(guessed - fallback_channel) <= 256.0:
                    predicted_channel = guessed
            except Exception:
                pass
            return float(np.clip(predicted_channel, 0.0, max_channel))

        def _fine_tune_from_preserved(self) -> None:
            if self._preserved_snapshot is None:
                self._refresh_snapshot_summary(
                    message="Preserve a calibration before requesting a fine-tune pass."
                )
                return
            tuned_energy_points: list[EnergyCalibrationPoint] = []
            for point in self._preserved_snapshot.energy_points:
                channel_guess = self._best_channel_guess(
                    float(point.reference_energy_keV),
                    fallback_channel=float(point.channel),
                )
                snapped_channel = self._snap_channel_to_peak(channel_guess)
                tuned_energy_points.append(
                    EnergyCalibrationPoint(
                        label=point.label,
                        channel=snapped_channel,
                        observed_energy_keV=float(
                            self._spectrum.channel_to_energy(snapped_channel)
                        ),
                        reference_energy_keV=float(point.reference_energy_keV),
                        uncertainty_keV=point.uncertainty_keV,
                    )
                )

            tuned_fwhm_points: list[FWHMCalibrationPoint] = []
            for point in self._preserved_snapshot.fwhm_points:
                channel_guess = self._best_channel_guess(
                    float(point.energy_keV),
                    fallback_channel=float(point.energy_keV),
                )
                snapped_channel = self._snap_channel_to_peak(channel_guess)
                local_fwhm_channels = estimate_local_fwhm_channels(
                    self._spectrum.counts,
                    int(round(snapped_channel)),
                )
                tuned_fwhm_points.append(
                    FWHMCalibrationPoint(
                        label=point.label,
                        energy_keV=float(point.energy_keV),
                        fwhm_keV=max(
                            local_fwhm_channels
                            * self._energy_scale_at_channel(snapped_channel),
                            0.05,
                        ),
                        uncertainty_keV=point.uncertainty_keV,
                    )
                )

            self._restore_snapshot(
                CalibrationSnapshot(
                    label=f"{self._preserved_snapshot.label} (fine-tuned)",
                    energy_points=tuple(tuned_energy_points),
                    fwhm_points=tuple(tuned_fwhm_points),
                    deviation_pairs=self._preserved_snapshot.deviation_pairs,
                ),
                message="Fine-tuned the preserved calibration against peaks in the current spectrum.",
            )

        def _apply_nasa_smart_seed(self) -> None:
            detected_peaks = list(
                detect_peak_candidates(self._spectrum, method="nasa_peaksearch")
            )
            assigned_peak_ids: set[str] = set()
            smart_points: list[EnergyCalibrationPoint] = []
            for index, (label, reference_energy) in enumerate(REFERENCE_LINES_KEV):
                selected_peak = None
                selected_delta = float("inf")
                for peak in detected_peaks:
                    if peak.peak_id in assigned_peak_ids:
                        continue
                    delta = abs(float(peak.energy_keV) - float(reference_energy))
                    if delta < selected_delta:
                        selected_peak = peak
                        selected_delta = delta
                if selected_peak is not None and selected_delta <= 80.0:
                    assigned_peak_ids.add(selected_peak.peak_id)
                    snapped_channel = float(selected_peak.channel)
                else:
                    snapped_channel = self._snap_channel_to_peak(
                        self._best_channel_guess(
                            float(reference_energy),
                            fallback_channel=float(reference_energy),
                        )
                    )
                smart_points.append(
                    EnergyCalibrationPoint(
                        label=label,
                        channel=snapped_channel,
                        observed_energy_keV=float(
                            self._spectrum.channel_to_energy(snapped_channel)
                        ),
                        reference_energy_keV=float(reference_energy),
                        uncertainty_keV=0.15 if index == 0 else 0.12,
                    )
                )

            smart_fwhm_points: list[FWHMCalibrationPoint] = []
            for point in smart_points:
                local_fwhm_channels = estimate_local_fwhm_channels(
                    self._spectrum.counts,
                    int(round(point.channel)),
                )
                smart_fwhm_points.append(
                    FWHMCalibrationPoint(
                        label=point.label,
                        energy_keV=float(point.reference_energy_keV),
                        fwhm_keV=max(
                            local_fwhm_channels
                            * self._energy_scale_at_channel(point.channel),
                            0.05,
                        ),
                        uncertainty_keV=0.05,
                    )
                )

            self._restore_snapshot(
                CalibrationSnapshot(
                    label="NASA smart seed",
                    energy_points=tuple(smart_points),
                    fwhm_points=tuple(smart_fwhm_points),
                    deviation_pairs=self._read_deviation_pairs(),
                ),
                message="NASA smart seed refreshed the calibration anchors from detected peaks.",
            )

        def _read_deviation_pairs(self) -> tuple[EnergyDeviationPair, ...]:
            pairs: list[EnergyDeviationPair] = []
            for row in range(self.deviation_table.rowCount()):
                energy = self._table_float(self.deviation_table, row, 0)
                correction = self._table_float(self.deviation_table, row, 1)
                label = self._table_text(self.deviation_table, row, 2)
                if energy is None or correction is None:
                    continue
                pairs.append(
                    EnergyDeviationPair(
                        energy_keV=float(energy),
                        correction_keV=float(correction),
                        label=label,
                    )
                )
            return tuple(sorted(pairs, key=lambda pair: pair.energy_keV))

        def _add_empty_deviation_row(self) -> None:
            row = self.deviation_table.rowCount()
            self._syncing_deviation_table = True
            self.deviation_table.insertRow(row)
            for column in range(3):
                self._set_deviation_cell(row, column, "")
            self._syncing_deviation_table = False

        def _set_deviation_cell(self, row: int, column: int, value) -> None:
            item = QTableWidgetItem(
                "" if value is None else self._format_table_value(value)
            )
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            self.deviation_table.setItem(row, column, item)

        def _remove_selected_deviation_rows(self) -> None:
            row = self.deviation_table.currentRow()
            if row < 0:
                return
            self.deviation_table.removeRow(row)
            self._refresh_energy_fit()

        def _clear_deviation_pairs(self) -> None:
            self._syncing_deviation_table = True
            self.deviation_table.setRowCount(0)
            self._syncing_deviation_table = False
            self._refresh_energy_fit()

        def _seed_deviation_pairs_from_residuals(self) -> None:
            if self._energy_fit is None:
                return
            points = self._read_energy_points()
            candidate_pairs = [
                EnergyDeviationPair(
                    energy_keV=float(point.reference_energy_keV),
                    correction_keV=float(self._energy_fit.residuals_keV[index]),
                    label=point.label or f"Pair {index + 1}",
                )
                for index, point in enumerate(points)
                if index < len(self._energy_fit.residuals_keV)
                and abs(float(self._energy_fit.residuals_keV[index])) >= 0.02
            ]
            if not candidate_pairs and points:
                candidate_pairs = [
                    EnergyDeviationPair(
                        energy_keV=float(points[0].reference_energy_keV),
                        correction_keV=float(self._energy_fit.residuals_keV[0]),
                        label=points[0].label or "Pair 1",
                    )
                ]
            self._syncing_deviation_table = True
            self.deviation_table.setRowCount(0)
            for pair in candidate_pairs:
                row = self.deviation_table.rowCount()
                self.deviation_table.insertRow(row)
                self._set_deviation_cell(row, 0, pair.energy_keV)
                self._set_deviation_cell(row, 1, pair.correction_keV)
                self._set_deviation_cell(row, 2, pair.label)
            self._syncing_deviation_table = False
            self._refresh_energy_fit()

        def _handle_deviation_table_change(self, _item) -> None:
            if self._syncing_deviation_table:
                return
            self._refresh_energy_fit()

        def _update_deviation_pairs_summary(
            self,
            pairs: tuple[EnergyDeviationPair, ...],
        ) -> None:
            if not pairs:
                self.deviation_summary.setText(
                    "No deviation pairs are active. The polynomial fit is currently unwarped."
                )
                return
            max_correction = max(abs(pair.correction_keV) for pair in pairs)
            self.deviation_summary.setText(
                (
                    f"{len(pairs)} deviation pairs active. "
                    f"Max local correction = {max_correction:.4f} keV."
                )
            )

        def _refresh_roi_background_models(self) -> None:
            fitter_key = (
                self.roi_method_selector.current_key()
                or self._peak_fitter_registry.default_key
                or "gaussian"
            )
            current = self.roi_background_combo.currentData()
            self.roi_background_combo.blockSignals(True)
            self.roi_background_combo.clear()
            for model in available_background_models(fitter_key):
                self.roi_background_combo.addItem(
                    model.replace("_", " ").title(), model
                )
            self.roi_background_combo.blockSignals(False)
            if current is not None:
                index = self.roi_background_combo.findData(current)
                if index >= 0:
                    self.roi_background_combo.setCurrentIndex(index)
            if (
                self.roi_background_combo.currentIndex() < 0
                and self.roi_background_combo.count()
            ):
                self.roi_background_combo.setCurrentIndex(0)

        def _handle_roi_fitter_changed(self, *_args) -> None:
            self._refresh_roi_background_models()
            self._refresh_roi_fit()

        def _handle_roi_region_changed(self, *_args) -> None:
            self._refresh_roi_fit()

        def _snap_roi_to_selected_energy_row(self) -> None:
            row = self.energy_table.currentRow()
            channel = None
            if row >= 0:
                channel = self._table_float(self.energy_table, row, 1)
            if channel is None and self.selection_bus.state.peak_energy_keV is not None:
                try:
                    channel = float(
                        self._spectrum.energy_to_channel(
                            self.selection_bus.state.peak_energy_keV
                        )
                    )
                except Exception:
                    channel = None
            if channel is None:
                return
            half_width = max(
                estimate_local_fwhm_channels(self._spectrum.counts, int(round(channel)))
                * 3.0,
                10.0,
            )
            self.roi_region.setRegion((channel - half_width, channel + half_width))
            self._refresh_roi_fit()

        def _clear_roi_fit_visuals(self, message: str) -> None:
            self._roi_fit = None
            self.roi_centroid_line.setVisible(False)
            for curve in (
                self.spectrum_roi_fit_curve,
                self.spectrum_roi_background_curve,
                self.roi_detail_observed_curve,
                self.roi_detail_fit_curve,
                self.roi_detail_background_curve,
                self.roi_residual_curve,
            ):
                curve.setData([], [])
            self.roi_fit_summary.setText(message)

        def _refresh_roi_fit(self, *_args) -> None:
            counts = np.asarray(self._spectrum.counts, dtype=float)
            channels = np.asarray(self._spectrum.channels, dtype=float)
            if channels.size == 0:
                channels = np.arange(len(counts), dtype=float)
            if counts.size < 5:
                self._clear_roi_fit_visuals(
                    "Not enough channels are available for ROI fitting."
                )
                return
            roi_bounds = tuple(float(value) for value in self.roi_region.getRegion())
            try:
                prior_fwhm_channels = None
                if self._fwhm_fit is not None:
                    centroid_estimate = float(np.mean(roi_bounds))
                    centroid_energy = float(
                        self._spectrum.channel_to_energy(centroid_estimate)
                    )
                    fwhm_keV = float(
                        np.asarray(
                            self._fwhm_fit.curve.fwhm(
                                np.asarray([centroid_energy], dtype=float)
                            )
                        )[0]
                    )
                    scale_keV = max(
                        self._energy_scale_at_channel(centroid_estimate), 1e-6
                    )
                    prior_fwhm_channels = fwhm_keV / scale_keV
                fit = fit_roi_peak(
                    channels,
                    counts,
                    roi_bounds,
                    fitter_key=self.roi_method_selector.current_key() or "gaussian",
                    background_model=str(
                        self.roi_background_combo.currentData() or "linear"
                    ),
                    prior_fwhm_channels=prior_fwhm_channels,
                )
            except Exception as exc:
                self._clear_roi_fit_visuals(str(exc))
                return

            self._roi_fit = fit
            normalized_residuals = (fit.observed_counts - fit.fit_counts) / np.sqrt(
                np.clip(fit.fit_counts, 1.0, None)
            )
            self.spectrum_roi_fit_curve.setData(fit.channels, fit.fit_counts)
            self.spectrum_roi_background_curve.setData(
                fit.channels,
                fit.background_counts,
            )
            self.roi_centroid_line.setVisible(True)
            self.roi_centroid_line.setPos(fit.centroid_channel)
            self.roi_detail_observed_curve.setData(fit.channels, fit.observed_counts)
            self.roi_detail_fit_curve.setData(fit.channels, fit.fit_counts)
            self.roi_detail_background_curve.setData(
                fit.channels, fit.background_counts
            )
            self.roi_residual_curve.setData(fit.channels, normalized_residuals)
            centroid_energy = float(
                self._spectrum.channel_to_energy(fit.centroid_channel)
            )
            roi_energy_bounds = (
                float(self._spectrum.channel_to_energy(min(fit.roi_bounds))),
                float(self._spectrum.channel_to_energy(max(fit.roi_bounds))),
            )
            self.selection_bus.publish_roi(*roi_energy_bounds)
            self.selection_bus.publish_peak(centroid_energy)
            self.roi_fit_summary.setText(
                (
                    f"{fit.fitter_label} with {fit.background_model} background | "
                    f"centroid {fit.centroid_channel:.3f} ch ({centroid_energy:.3f} keV) | "
                    f"FWHM {fit.fwhm_channels:.3f} ch | area {fit.area_counts:.1f} counts | "
                    f"reduced chi^2 {fit.peak_result.reduced_chi_squared:.3f}"
                )
            )

        def _energy_scale_at_channel(self, channel: float) -> float:
            low = max(float(channel) - 0.5, 0.0)
            high = min(float(channel) + 0.5, max(len(self._spectrum.channels) - 1, 1))
            return abs(
                float(self._spectrum.channel_to_energy(high))
                - float(self._spectrum.channel_to_energy(low))
            )

        def _apply_roi_fit_to_selected_energy_row(self) -> None:
            if self._roi_fit is None:
                return
            row = self.energy_table.currentRow()
            if row < 0:
                row = self.energy_table.rowCount()
                self._add_empty_energy_row(select_row=False)
            channel = float(self._roi_fit.centroid_channel)
            observed_energy = float(self._spectrum.channel_to_energy(channel))
            self._set_energy_cell(row, 0, self._roi_fit.fitter_label, editable=True)
            self._set_energy_cell(row, 1, channel)
            self._set_energy_cell(row, 2, observed_energy)
            if not self._table_text(self.energy_table, row, 3):
                reference_lines = tuple(
                    self.selection_bus.state.reference_lines_keV or ()
                )
                if reference_lines:
                    nearest = min(
                        reference_lines, key=lambda value: abs(value - observed_energy)
                    )
                    self._set_energy_cell(row, 3, nearest)
            if not self._table_text(self.energy_table, row, 4):
                self._set_energy_cell(row, 4, 0.12, editable=True)
            self.energy_table.selectRow(row)
            self._refresh_energy_fit()

        def _apply_roi_fit_to_selected_fwhm_row(self) -> None:
            if self._roi_fit is None:
                return
            row = self.fwhm_table.currentRow()
            if row < 0:
                self._add_empty_fwhm_row()
                row = self.fwhm_table.currentRow()
            channel = float(self._roi_fit.centroid_channel)
            observed_energy = float(self._spectrum.channel_to_energy(channel))
            fwhm_keV = max(
                self._roi_fit.fwhm_channels * self._energy_scale_at_channel(channel),
                0.05,
            )
            self._set_fwhm_cell(row, 0, self._roi_fit.fitter_label)
            self._set_fwhm_cell(row, 1, observed_energy)
            self._set_fwhm_cell(row, 2, fwhm_keV)
            if not self._table_text(self.fwhm_table, row, 3):
                self._set_fwhm_cell(row, 3, max(fwhm_keV * 0.05, 0.05), editable=True)
            self.fwhm_table.selectRow(row)
            self._refresh_fwhm_fit()

        def _seed_tables(self) -> None:
            self._seed_energy_table()
            self._seed_fwhm_table()
            self._reset_quick_slider_anchors()

        def _seed_energy_table(self) -> None:
            self._syncing_energy_table = True
            self.energy_table.setRowCount(0)
            for point in self._seed_energy_points():
                self._append_energy_row(point)
            self._syncing_energy_table = False
            self._refresh_energy_fit()

        def _seed_fwhm_table(self) -> None:
            self._syncing_fwhm_table = True
            self.fwhm_table.setRowCount(0)
            for point in self._seed_fwhm_points():
                self._append_fwhm_row(point)
            self._syncing_fwhm_table = False
            self._refresh_fwhm_fit()

        def _seed_energy_points(self) -> tuple[EnergyCalibrationPoint, ...]:
            points: list[EnergyCalibrationPoint] = []
            counts = np.asarray(self._spectrum.counts, dtype=float)
            coefficients = self._spectrum.calibration.get("energy", [0.0, 1.0])
            use_calibration = len(coefficients) >= 2 and counts.size > 0

            for index, (label, reference_energy) in enumerate(REFERENCE_LINES_KEV):
                if use_calibration:
                    try:
                        channel = float(
                            self._spectrum.energy_to_channel(reference_energy)
                        )
                    except Exception:
                        channel = float(reference_energy)
                else:
                    channel = float(reference_energy)
                snapped_channel = self._snap_channel_to_peak(channel)
                observed_energy = float(
                    self._spectrum.channel_to_energy(snapped_channel)
                )
                points.append(
                    EnergyCalibrationPoint(
                        label=label,
                        channel=snapped_channel,
                        observed_energy_keV=observed_energy,
                        reference_energy_keV=float(reference_energy),
                        uncertainty_keV=0.15 if index == 0 else 0.12,
                    )
                )
            return tuple(points)

        def _seed_fwhm_points(self) -> tuple[FWHMCalibrationPoint, ...]:
            points: list[FWHMCalibrationPoint] = []
            coefficients = self._spectrum.calibration.get("energy", [0.0, 1.0])
            for label, reference_energy in REFERENCE_LINES_KEV:
                try:
                    channel = float(self._spectrum.energy_to_channel(reference_energy))
                except Exception:
                    channel = float(reference_energy)
                snapped_channel = self._snap_channel_to_peak(channel)
                fwhm_channels = estimate_local_fwhm_channels(
                    self._spectrum.counts,
                    int(round(snapped_channel)),
                )
                slope = abs(energy_calibration_slope(coefficients, snapped_channel))
                slope = slope if slope > 1e-6 else 1.0
                fwhm_keV = max(fwhm_channels * slope, 0.05)
                points.append(
                    FWHMCalibrationPoint(
                        label=label,
                        energy_keV=float(reference_energy),
                        fwhm_keV=float(fwhm_keV),
                        uncertainty_keV=max(float(fwhm_keV) * 0.05, 0.05),
                    )
                )
            return tuple(points)

        def _snap_channel_to_peak(self, candidate_channel: float) -> float:
            counts = np.asarray(self._spectrum.counts, dtype=float)
            if counts.size == 0:
                return float(candidate_channel)
            center = int(np.clip(int(round(candidate_channel)), 0, counts.size - 1))
            left = max(center - 12, 0)
            right = min(center + 12, counts.size - 1)
            local_offset = int(np.argmax(counts[left : right + 1]))
            return float(left + local_offset)

        def _plot_spectrum(self) -> None:
            channels = np.asarray(self._spectrum.channels, dtype=float)
            counts = np.asarray(self._spectrum.counts, dtype=float)
            if channels.size == 0:
                channels = np.arange(len(counts), dtype=float)
            self.spectrum_curve.setData(channels, counts)
            self.spectrum_plot.setXRange(
                float(np.min(channels)) if channels.size else 0.0,
                float(np.max(channels)) if channels.size else 2048.0,
                padding=0.02,
            )
            self.spectrum_kpi[1].setText(
                self._spectrum.spectrum_id or "Loaded spectrum"
            )
            self._refresh_spectrum_markers()
            self._refresh_quick_slider_preview()
            self._refresh_roi_fit()

        def _refresh_spectrum_markers(self) -> None:
            points = self._read_energy_points()
            counts = np.asarray(self._spectrum.counts, dtype=float)
            marker_x: list[float] = []
            marker_y: list[float] = []
            for point in points:
                channel = int(
                    np.clip(int(round(point.channel)), 0, max(len(counts) - 1, 0))
                )
                if counts.size == 0:
                    continue
                marker_x.append(point.channel)
                marker_y.append(float(counts[channel]))
            self.spectrum_markers.setData(marker_x, marker_y)

        def _handle_spectrum_click(self, event) -> None:
            if event.button() != Qt.LeftButton:
                return
            scene_point = event.scenePos()
            plot_point = self.spectrum_plot.plotItem.vb.mapSceneToView(scene_point)
            clicked_channel = self._snap_channel_to_peak(plot_point.x())
            half_width = max(
                estimate_local_fwhm_channels(
                    self._spectrum.counts, int(round(clicked_channel))
                )
                * 3.0,
                10.0,
            )
            self.roi_region.setRegion(
                (clicked_channel - half_width, clicked_channel + half_width)
            )
            row = self.energy_table.currentRow()
            if row < 0:
                row = self.energy_table.rowCount()
                self._add_empty_energy_row(select_row=False)
            observed_energy = float(self._spectrum.channel_to_energy(clicked_channel))
            self._set_energy_cell(row, 1, clicked_channel)
            self._set_energy_cell(row, 2, observed_energy)
            label_item = self.energy_table.item(row, 0)
            if label_item is None or not label_item.text().strip():
                self._set_energy_cell(row, 0, f"Point {row + 1}", editable=True)
            reference_item = self.energy_table.item(row, 3)
            if reference_item is None or not reference_item.text().strip():
                reference_lines = tuple(
                    self.selection_bus.state.reference_lines_keV or ()
                )
                if reference_lines:
                    nearest = min(
                        reference_lines, key=lambda value: abs(value - observed_energy)
                    )
                    self._set_energy_cell(row, 3, nearest)
            self.energy_table.selectRow(row)
            self.selection_bus.publish_peak(observed_energy)
            self._refresh_energy_fit()
            self._refresh_roi_fit()

        def _read_energy_points(self) -> tuple[EnergyCalibrationPoint, ...]:
            points: list[EnergyCalibrationPoint] = []
            for row in range(self.energy_table.rowCount()):
                label = self._table_text(self.energy_table, row, 0)
                channel = self._table_float(self.energy_table, row, 1)
                observed = self._table_float(self.energy_table, row, 2)
                reference = self._table_float(self.energy_table, row, 3)
                uncertainty = self._table_float(self.energy_table, row, 4)
                if channel is None or reference is None:
                    continue
                points.append(
                    EnergyCalibrationPoint(
                        label=label,
                        channel=channel,
                        observed_energy_keV=observed,
                        reference_energy_keV=reference,
                        uncertainty_keV=uncertainty,
                    )
                )
            return tuple(points)

        def _read_fwhm_points(self) -> tuple[FWHMCalibrationPoint, ...]:
            points: list[FWHMCalibrationPoint] = []
            for row in range(self.fwhm_table.rowCount()):
                label = self._table_text(self.fwhm_table, row, 0)
                energy = self._table_float(self.fwhm_table, row, 1)
                fwhm = self._table_float(self.fwhm_table, row, 2)
                uncertainty = self._table_float(self.fwhm_table, row, 3)
                if energy is None or fwhm is None:
                    continue
                points.append(
                    FWHMCalibrationPoint(
                        label=label,
                        energy_keV=energy,
                        fwhm_keV=fwhm,
                        uncertainty_keV=uncertainty,
                    )
                )
            return tuple(points)

        def _refresh_energy_fit(self) -> None:
            points = self._read_energy_points()
            self._refresh_spectrum_markers()
            try:
                fit = fit_energy_calibration(
                    points,
                    order=self.energy_order.value(),
                    standard=self.mode_manager.state.standard,
                    deviation_pairs=self._read_deviation_pairs(),
                )
            except ValueError as exc:
                self._energy_fit = None
                self.energy_summary.setText(str(exc))
                self.energy_kpi[1].setText("Waiting")
                self.energy_residual_in_spec.setData([], [])
                self.energy_residual_out_spec.setData([], [])
                self.apply_button.setEnabled(False)
                self._update_energy_table_diagnostics(None)
                self._update_deviation_pairs_summary(self._read_deviation_pairs())
                return

            self._energy_fit = fit
            self.energy_kpi[1].setText(f"Order {fit.order} | RMS {fit.rms_keV:.3f} keV")
            coefficients = ", ".join(f"{value:.6g}" for value in fit.coefficients)
            self.energy_summary.setText(
                (
                    f"Coefficients: [{coefficients}] | R^2={fit.r_squared:.6f} | "
                    f"chi^2={fit.chi_squared:.3f} | reduced chi^2={fit.reduced_chi_squared:.3f} | "
                    f"RMS={fit.rms_keV:.4f} keV | deviation pairs={len(fit.deviation_pairs)}"
                )
            )
            reference_energies = np.asarray(
                [point.reference_energy_keV for point in points],
                dtype=float,
            )
            residuals = np.asarray(fit.residuals_keV, dtype=float)
            in_spec_mask = np.asarray(
                [not value for value in fit.out_of_tolerance],
                dtype=bool,
            )
            self.energy_residual_in_spec.setData(
                reference_energies[in_spec_mask],
                residuals[in_spec_mask],
            )
            self.energy_residual_out_spec.setData(
                reference_energies[~in_spec_mask],
                residuals[~in_spec_mask],
            )
            self._update_energy_table_diagnostics(fit)
            self._update_deviation_pairs_summary(fit.deviation_pairs)
            self.apply_button.setEnabled(True)

        def _refresh_fwhm_fit(self) -> None:
            points = self._read_fwhm_points()
            try:
                fit = fit_fwhm_calibration(points)
            except ValueError as exc:
                self._fwhm_fit = None
                self.fwhm_summary.setText(str(exc))
                self.fwhm_kpi[1].setText("Waiting")
                self.fwhm_measured_curve.setData([], [])
                self.fwhm_fit_curve.setData([], [])
                self._update_fwhm_table_diagnostics(None)
                return

            self._fwhm_fit = fit
            self.fwhm_kpi[1].setText(f"{fit.model} | RMS {fit.rms_keV:.3f} keV")
            coefficients = ", ".join(f"{value:.6g}" for value in fit.coefficients)
            self.fwhm_summary.setText(
                (
                    f"Coefficients: [{coefficients}] | chi^2={fit.chi_squared:.3f} | "
                    f"reduced chi^2={fit.reduced_chi_squared:.3f} | RMS={fit.rms_keV:.4f} keV"
                )
            )
            energies = np.asarray([point.energy_keV for point in points], dtype=float)
            fitted = np.asarray(fit.fitted_fwhm_keV, dtype=float)
            self.fwhm_measured_curve.setData(
                energies, [point.fwhm_keV for point in points]
            )
            order = np.argsort(energies)
            self.fwhm_fit_curve.setData(energies[order], fitted[order])
            self._update_fwhm_table_diagnostics(fit)

        def _update_energy_table_diagnostics(
            self,
            fit: EnergyCalibrationFit | None,
        ) -> None:
            self._syncing_energy_table = True
            for row in range(self.energy_table.rowCount()):
                residual_text = ""
                status_text = ""
                if fit is not None and row < len(fit.residuals_keV):
                    residual = float(fit.residuals_keV[row])
                    residual_text = f"{residual:.4f}"
                    status_text = "FAIL" if fit.out_of_tolerance[row] else "PASS"
                self._set_energy_cell(row, 5, residual_text, editable=False)
                self._set_energy_cell(row, 6, status_text, editable=False)
            self._syncing_energy_table = False

        def _update_fwhm_table_diagnostics(
            self,
            fit: FWHMCalibrationFit | None,
        ) -> None:
            self._syncing_fwhm_table = True
            for row in range(self.fwhm_table.rowCount()):
                residual_text = ""
                if fit is not None and row < len(fit.residuals_keV):
                    residual_text = f"{float(fit.residuals_keV[row]):.4f}"
                self._set_fwhm_cell(row, 4, residual_text, editable=False)
            self._syncing_fwhm_table = False

        def _apply_workspace_results(self) -> None:
            if self._energy_fit is None:
                return
            calibration = dict(self._spectrum.calibration)
            calibration["energy"] = list(self._energy_fit.coefficients)
            calibration["deviation_pairs"] = [
                {
                    "energy_keV": pair.energy_keV,
                    "correction_keV": pair.correction_keV,
                    "label": pair.label,
                }
                for pair in self._energy_fit.deviation_pairs
            ]
            if self._fwhm_fit is not None:
                calibration["fwhm"] = {
                    "model": self._fwhm_fit.model,
                    "coefficients": list(self._fwhm_fit.coefficients),
                }
            # GammaSpectrum is mutable for parser compatibility, so applying a
            # calibration must create a new spectrum value.  ``replace`` keeps
            # the large count/channel arrays by identity while ``__post_init__``
            # derives a fresh energy axis from the new calibration mapping.
            updated_spectrum = replace(
                self._spectrum,
                calibration=calibration,
                energies=None,
            )
            self.selection_bus.publish(
                self.selection_bus.state.__class__(
                    peak_energy_keV=self.selection_bus.state.peak_energy_keV,
                    roi_bounds_keV=self.selection_bus.state.roi_bounds_keV,
                    nuclide=self.selection_bus.state.nuclide,
                    reference_lines_keV=tuple(
                        point.reference_energy_keV
                        for point in self._read_energy_points()
                    ),
                )
            )
            if self.on_apply is not None:
                self.on_apply(updated_spectrum, self._energy_fit, self._fwhm_fit)
            self._spectrum = updated_spectrum
            self.energy_summary.setText(
                self.energy_summary.text()
                + "\nApplied to the workspace spectrum and broadcast to the shell."
            )

        def _handle_energy_table_change(self, _item) -> None:
            if self._syncing_energy_table:
                return
            self._refresh_energy_fit()

        def _handle_fwhm_table_change(self, _item) -> None:
            if self._syncing_fwhm_table:
                return
            self._refresh_fwhm_fit()

        def _add_empty_energy_row(self, *, select_row: bool = True) -> None:
            row = self.energy_table.rowCount()
            self._syncing_energy_table = True
            self.energy_table.insertRow(row)
            for column in range(len(self.ENERGY_HEADERS)):
                self._set_energy_cell(row, column, "", editable=column < 5)
            self._set_energy_cell(row, 0, f"Point {row + 1}")
            self._syncing_energy_table = False
            if select_row:
                self.energy_table.selectRow(row)

        def _add_empty_fwhm_row(self) -> None:
            row = self.fwhm_table.rowCount()
            self._syncing_fwhm_table = True
            self.fwhm_table.insertRow(row)
            for column in range(len(self.FWHM_HEADERS)):
                self._set_fwhm_cell(row, column, "", editable=column < 4)
            self._set_fwhm_cell(row, 0, f"Point {row + 1}")
            self._syncing_fwhm_table = False
            self.fwhm_table.selectRow(row)

        def _remove_selected_energy_rows(self) -> None:
            row = self.energy_table.currentRow()
            if row < 0:
                return
            self.energy_table.removeRow(row)
            self._refresh_energy_fit()

        def _remove_selected_fwhm_rows(self) -> None:
            row = self.fwhm_table.currentRow()
            if row < 0:
                return
            self.fwhm_table.removeRow(row)
            self._refresh_fwhm_fit()

        def _append_energy_row(self, point: EnergyCalibrationPoint) -> None:
            row = self.energy_table.rowCount()
            self.energy_table.insertRow(row)
            self._set_energy_cell(row, 0, point.label or f"Point {row + 1}")
            self._set_energy_cell(row, 1, point.channel)
            self._set_energy_cell(row, 2, point.observed_energy_keV)
            self._set_energy_cell(row, 3, point.reference_energy_keV)
            self._set_energy_cell(row, 4, point.uncertainty_keV)
            self._set_energy_cell(row, 5, "", editable=False)
            self._set_energy_cell(row, 6, "", editable=False)

        def _append_fwhm_row(self, point: FWHMCalibrationPoint) -> None:
            row = self.fwhm_table.rowCount()
            self.fwhm_table.insertRow(row)
            self._set_fwhm_cell(row, 0, point.label or f"Point {row + 1}")
            self._set_fwhm_cell(row, 1, point.energy_keV)
            self._set_fwhm_cell(row, 2, point.fwhm_keV)
            self._set_fwhm_cell(row, 3, point.uncertainty_keV)
            self._set_fwhm_cell(row, 4, "", editable=False)

        def _set_energy_cell(
            self,
            row: int,
            column: int,
            value,
            *,
            editable: bool = True,
        ) -> None:
            item = QTableWidgetItem(
                "" if value is None else self._format_table_value(value)
            )
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if editable:
                flags |= Qt.ItemIsEditable
            item.setFlags(flags)
            self.energy_table.setItem(row, column, item)

        def _set_fwhm_cell(
            self,
            row: int,
            column: int,
            value,
            *,
            editable: bool = True,
        ) -> None:
            item = QTableWidgetItem(
                "" if value is None else self._format_table_value(value)
            )
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if editable:
                flags |= Qt.ItemIsEditable
            item.setFlags(flags)
            self.fwhm_table.setItem(row, column, item)

        def _format_table_value(self, value) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6f}".rstrip("0").rstrip(".")
            return str(value)

        def _table_text(self, table: QTableWidget, row: int, column: int) -> str:
            item = table.item(row, column)
            if item is None:
                return ""
            return item.text().strip()

        def _table_float(
            self,
            table: QTableWidget,
            row: int,
            column: int,
        ) -> float | None:
            text = self._table_text(table, row, column)
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

else:

    class CalibrationWorkspaceDialog:  # pragma: no cover - placeholder without GUI deps
        """Import-safe placeholder when Qt or PyQtGraph is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "The calibration workspace requires the native Qt GUI extras."
            )
