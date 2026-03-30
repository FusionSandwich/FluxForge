"""Unified energy and FWHM calibration workspace for Phase 2.1."""

from __future__ import annotations

from typing import Callable

import numpy as np

from fluxforge.core.calibration import (
    ASTM_E181_ENERGY_LIMIT_KEV,
    EnergyCalibrationFit,
    EnergyCalibrationPoint,
    FWHMCalibrationFit,
    FWHMCalibrationPoint,
    energy_calibration_slope,
    estimate_local_fwhm_channels,
    fit_energy_calibration,
    fit_fwhm_calibration,
    resolve_energy_calibration_order,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus
from fluxforge.gui.theme_manager import theme_tokens
from fluxforge.io.spe import GammaSpectrum

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # pragma: no cover - optional dependency branch
    import pyqtgraph as pg

    from fluxforge.gui.panels.modern_shell import build_demo_spectrum
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QSpinBox,
        QSplitter,
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


if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # pragma: no cover - optional dependency branch

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
            on_apply: Callable[
                [GammaSpectrum, EnergyCalibrationFit, FWHMCalibrationFit | None], None
            ]
            | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("CalibrationWorkspaceDialog")
            self.setWindowTitle("FluxForge Next - Unified Calibration Workspace")
            self.resize(1540, 960)
            self.setModal(False)

            self.mode_manager = mode_manager
            self.selection_bus = selection_bus or SelectionBus.shared()
            self.on_apply = on_apply
            self._syncing_energy_table = False
            self._syncing_fwhm_table = False
            self._spectrum = spectrum or build_demo_spectrum()
            self._energy_fit: EnergyCalibrationFit | None = None
            self._fwhm_fit: FWHMCalibrationFit | None = None

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

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(14)

            header = QFrame(self)
            header.setObjectName("CalibrationHeaderCard")
            header_layout = QVBoxLayout(header)
            header_layout.setContentsMargins(24, 20, 24, 20)
            header_layout.setSpacing(10)

            eyebrow = QLabel("Phase 2.1", header)
            eyebrow.setObjectName("HeroEyebrow")
            header_layout.addWidget(eyebrow)

            title = QLabel("Unified energy + FWHM calibration workspace", header)
            title.setObjectName("HeroHeader")
            header_layout.addWidget(title)

            subtitle = QLabel(
                (
                    "Residuals stay visible while the spectrum, calibration points, "
                    "and fit diagnostics update in one place. This dialog uses the "
                    "new Qt calibration stack only."
                ),
                header,
            )
            subtitle.setObjectName("HeroSubhead")
            subtitle.setWordWrap(True)
            header_layout.addWidget(subtitle)

            kpi_row = QHBoxLayout()
            kpi_row.setSpacing(12)
            self.spectrum_kpi = self._build_kpi_card("Spectrum", "Demo")
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

            self.spectrum_plot = self._create_plot_widget("Spectrum canvas", "Channel", "Counts")
            self.spectrum_curve = self.spectrum_plot.plot(
                pen=pg.mkPen(width=2),
                fillLevel=0.0,
            )
            self.spectrum_markers = self.spectrum_plot.plot(
                pen=None,
                symbol="o",
                symbolSize=10,
            )
            self.spectrum_plot.scene().sigMouseClicked.connect(self._handle_spectrum_click)
            plots_layout.addWidget(
                self._wrap_plot_card(
                    "Embedded spectrum canvas",
                    "Click the spectrum to update the selected calibration row.",
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

            controls_panel = QWidget(splitter)
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
            self.add_energy_point_button.clicked.connect(self._add_empty_energy_row)
            energy_button_row.addWidget(self.add_energy_point_button)
            self.remove_energy_point_button = QPushButton("Remove selected", energy_group)
            self.remove_energy_point_button.clicked.connect(self._remove_selected_energy_rows)
            energy_button_row.addWidget(self.remove_energy_point_button)
            self.reset_energy_points_button = QPushButton("Reset seeded points", energy_group)
            self.reset_energy_points_button.clicked.connect(self._seed_energy_table)
            energy_button_row.addWidget(self.reset_energy_points_button)
            energy_button_row.addStretch(1)
            energy_layout.addLayout(energy_button_row)

            self.energy_table = QTableWidget(0, len(self.ENERGY_HEADERS), energy_group)
            self.energy_table.setObjectName("CalibrationTable")
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
            self.add_fwhm_point_button.clicked.connect(self._add_empty_fwhm_row)
            fwhm_button_row.addWidget(self.add_fwhm_point_button)
            self.remove_fwhm_point_button = QPushButton("Remove selected", fwhm_group)
            self.remove_fwhm_point_button.clicked.connect(self._remove_selected_fwhm_rows)
            fwhm_button_row.addWidget(self.remove_fwhm_point_button)
            self.reset_fwhm_points_button = QPushButton("Reset seeded points", fwhm_group)
            self.reset_fwhm_points_button.clicked.connect(self._seed_fwhm_table)
            fwhm_button_row.addWidget(self.reset_fwhm_points_button)
            fwhm_button_row.addStretch(1)
            fwhm_layout.addLayout(fwhm_button_row)

            self.fwhm_table = QTableWidget(0, len(self.FWHM_HEADERS), fwhm_group)
            self.fwhm_table.setObjectName("CalibrationTable")
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
            self.apply_button = QPushButton("Apply calibration to spectrum", controls_panel)
            self.apply_button.setObjectName("PrimaryAction")
            self.apply_button.clicked.connect(self._apply_workspace_results)
            action_row.addWidget(self.apply_button)
            self.close_button = QPushButton("Close", controls_panel)
            self.close_button.clicked.connect(self.close)
            action_row.addWidget(self.close_button)
            controls_layout.addLayout(action_row)

            splitter.addWidget(plots_panel)
            splitter.addWidget(controls_panel)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)

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
        ) -> pg.PlotWidget:
            widget = pg.PlotWidget(self)
            widget.setObjectName("CalibrationPlot")
            widget.showGrid(x=True, y=True, alpha=0.12)
            widget.setMenuEnabled(False)
            widget.setMouseEnabled(x=True, y=True)
            widget.setTitle(title)
            widget.setLabel("bottom", bottom_label)
            widget.setLabel("left", left_label)
            return widget

        def _on_mode_state_changed(self, state) -> None:
            resolved = resolve_energy_calibration_order(
                self.energy_order.value(),
                standard=state.standard,
            )
            self.energy_order.blockSignals(True)
            self.energy_order.setValue(resolved.order)
            self.energy_order.setEnabled(resolved.locked_by is None)
            self.energy_order.blockSignals(False)
            self.energy_lock_label.setText(resolved.locked_by or "Expert-mode order selection is available.")
            self.provenance_summary.setText(
                self._format_provenance_summary(resolved.locked_by)
            )
            self._apply_plot_palette()
            self._refresh_energy_fit()

        def _format_provenance_summary(self, locked_by: str | None) -> str:
            spectrum_name = self._spectrum.spectrum_id or "Untitled spectrum"
            mode_state = self.mode_manager.state
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
            lines.append(
                "Residual-first rule: energy residuals remain visible with the ASTM band."
            )
            return "\n".join(lines)

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
            self.energy_residual_in_spec.setSymbolPen(pg.mkPen(color=success, width=1.5))
            self.energy_residual_out_spec.setSymbolBrush(pg.mkBrush(error))
            self.energy_residual_out_spec.setSymbolPen(pg.mkPen(color=error, width=1.5))
            self.energy_residual_zero.setPen(pg.mkPen(color=text, width=1))
            self.energy_residual_upper.setPen(pg.mkPen(color=accent_warm, width=1))
            self.energy_residual_lower.setPen(pg.mkPen(color=accent_warm, width=1))
            self.energy_residual_band.setBrush(pg.mkBrush(244, 184, 96, 32))
            for line in self.energy_residual_band.lines:
                line.setPen(pg.mkPen(color=accent_warm, width=1))

            self.fwhm_measured_curve.setSymbolBrush(pg.mkBrush(accent_warm))
            self.fwhm_measured_curve.setSymbolPen(pg.mkPen(color=accent_warm, width=1.5))
            self.fwhm_fit_curve.setPen(pg.mkPen(color=accent, width=2))

        def _seed_tables(self) -> None:
            self._seed_energy_table()
            self._seed_fwhm_table()

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
                        channel = float(self._spectrum.energy_to_channel(reference_energy))
                    except Exception:
                        channel = float(reference_energy)
                else:
                    channel = float(reference_energy)
                snapped_channel = self._snap_channel_to_peak(channel)
                observed_energy = float(self._spectrum.channel_to_energy(snapped_channel))
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
            self.spectrum_kpi[1].setText(self._spectrum.spectrum_id or "Demo spectrum")
            self._refresh_spectrum_markers()

        def _refresh_spectrum_markers(self) -> None:
            points = self._read_energy_points()
            counts = np.asarray(self._spectrum.counts, dtype=float)
            marker_x: list[float] = []
            marker_y: list[float] = []
            for point in points:
                channel = int(np.clip(int(round(point.channel)), 0, max(len(counts) - 1, 0)))
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
                reference_lines = tuple(self.selection_bus.state.reference_lines_keV or ())
                if reference_lines:
                    nearest = min(reference_lines, key=lambda value: abs(value - observed_energy))
                    self._set_energy_cell(row, 3, nearest)
            self.energy_table.selectRow(row)
            self.selection_bus.publish_peak(observed_energy)
            self._refresh_energy_fit()

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
                )
            except ValueError as exc:
                self._energy_fit = None
                self.energy_summary.setText(str(exc))
                self.energy_kpi[1].setText("Waiting")
                self.energy_residual_in_spec.setData([], [])
                self.energy_residual_out_spec.setData([], [])
                self.apply_button.setEnabled(False)
                self._update_energy_table_diagnostics(None)
                return

            self._energy_fit = fit
            self.energy_kpi[1].setText(
                f"Order {fit.order} | RMS {fit.rms_keV:.3f} keV"
            )
            coefficients = ", ".join(f"{value:.6g}" for value in fit.coefficients)
            self.energy_summary.setText(
                (
                    f"Coefficients: [{coefficients}] | R^2={fit.r_squared:.6f} | "
                    f"chi^2={fit.chi_squared:.3f} | reduced chi^2={fit.reduced_chi_squared:.3f} | "
                    f"RMS={fit.rms_keV:.4f} keV"
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
            self.fwhm_kpi[1].setText(
                f"{fit.model} | RMS {fit.rms_keV:.3f} keV"
            )
            coefficients = ", ".join(f"{value:.6g}" for value in fit.coefficients)
            self.fwhm_summary.setText(
                (
                    f"Coefficients: [{coefficients}] | chi^2={fit.chi_squared:.3f} | "
                    f"reduced chi^2={fit.reduced_chi_squared:.3f} | RMS={fit.rms_keV:.4f} keV"
                )
            )
            energies = np.asarray([point.energy_keV for point in points], dtype=float)
            fitted = np.asarray(fit.fitted_fwhm_keV, dtype=float)
            self.fwhm_measured_curve.setData(energies, [point.fwhm_keV for point in points])
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
                    status_text = (
                        "FAIL"
                        if fit.out_of_tolerance[row]
                        else "PASS"
                    )
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
            self._spectrum.calibration["energy"] = list(self._energy_fit.coefficients)
            self._spectrum.energies = self._spectrum.calibrate_channels(
                list(self._energy_fit.coefficients)
            )
            self.selection_bus.publish(
                self.selection_bus.state.__class__(
                    peak_energy_keV=self.selection_bus.state.peak_energy_keV,
                    roi_bounds_keV=self.selection_bus.state.roi_bounds_keV,
                    nuclide=self.selection_bus.state.nuclide,
                    reference_lines_keV=tuple(
                        point.reference_energy_keV for point in self._read_energy_points()
                    ),
                )
            )
            if self.on_apply is not None:
                self.on_apply(self._spectrum, self._energy_fit, self._fwhm_fit)
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
            item = QTableWidgetItem("" if value is None else self._format_table_value(value))
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
            item = QTableWidgetItem("" if value is None else self._format_table_value(value))
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
                "The Phase 2.1 calibration workspace requires the native Qt GUI extras."
            )
