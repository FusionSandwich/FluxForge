"""Modern unfolding workspace dialog for the Qt shell."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fluxforge.analysis.flux_unfold import _make_response_row
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.backends.pyqtgraph_backend import catalog_pyqtgraph_export_action
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.io import read_reaction_rates
from fluxforge.plugins import bootstrap_builtin_registries
from fluxforge.unfolding import register_builtin_unfolders
from fluxforge.unfolding.base import UnfoldingResult
from fluxforge.unfolding.response_matrix import (
    build_analytical_hpge_response,
    load_response_matrix,
)

if (
    QT_AVAILABLE and PYQTGRAPH_AVAILABLE
):  # pragma: no cover - optional dependency branch
    import pyqtgraph as pg

    from fluxforge.gui.qt_compat import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        Qt,
    )
    from fluxforge.gui.widgets import MethodSelectorWidget


@dataclass(frozen=True)
class UnfoldingWorkspaceInput:
    """Input bundle for the native unfolding workspace."""

    label: str
    measured_rates: np.ndarray
    measurement_uncertainty: np.ndarray
    response_matrix: np.ndarray
    energy_edges: np.ndarray
    initial_flux: np.ndarray
    measurement_labels: tuple[str, ...] = ()
    energy_unit: str = "MeV"


def build_demo_unfolding_workspace_input() -> UnfoldingWorkspaceInput:
    """Return a deterministic sample case for the unfolding workspace."""

    energy_edges = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], dtype=float)
    response_matrix = np.array(
        [
            [0.92, 0.26, 0.08, 0.02, 0.00, 0.00],
            [0.18, 0.84, 0.30, 0.09, 0.02, 0.00],
            [0.06, 0.22, 0.88, 0.31, 0.08, 0.02],
            [0.02, 0.09, 0.28, 0.90, 0.34, 0.10],
            [0.00, 0.03, 0.10, 0.32, 0.86, 0.28],
            [0.00, 0.00, 0.03, 0.10, 0.26, 0.81],
            [0.44, 0.36, 0.18, 0.07, 0.02, 0.00],
            [0.00, 0.08, 0.22, 0.40, 0.48, 0.32],
        ],
        dtype=float,
    )
    true_flux = np.array([18.0, 44.0, 82.0, 34.0, 15.0, 6.0], dtype=float)
    measured_rates = response_matrix @ true_flux
    measurement_uncertainty = np.sqrt(np.maximum(measured_rates, 1.0))
    initial_flux = np.full(true_flux.shape, float(np.mean(true_flux)), dtype=float)
    return UnfoldingWorkspaceInput(
        label="Demo Flux-Wire Response",
        measured_rates=measured_rates,
        measurement_uncertainty=measurement_uncertainty,
        response_matrix=response_matrix,
        energy_edges=energy_edges,
        initial_flux=initial_flux,
        measurement_labels=tuple(
            f"M{index + 1}" for index in range(measured_rates.size)
        ),
    )


if (
    QT_AVAILABLE and PYQTGRAPH_AVAILABLE
):  # pragma: no cover - optional dependency branch

    class UnfoldingWorkspaceDialog(QDialog):
        """Qt unfolding workspace with method comparison and response-matrix review."""

        def __init__(
            self,
            *,
            mode_manager: ModeManager | None = None,
            workspace_input: UnfoldingWorkspaceInput | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("FluxForge — Unfolding Workspace")
            self.resize(1420, 940)

            self.mode_manager = mode_manager or ModeManager()
            self.workspace_input = (
                workspace_input or build_demo_unfolding_workspace_input()
            )
            self.registries = bootstrap_builtin_registries()
            register_builtin_unfolders(self.registries)

            self.current_result: UnfoldingResult | None = None
            self.comparison_results: dict[str, UnfoldingResult] = {}
            self._flux_curves: dict[str, object] = {}
            self._convergence_curves: dict[str, object] = {}

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            controls_panel = QWidget(self)
            controls_panel.setObjectName("UnfoldingControlsPanel")
            controls_panel.setLayout(self._build_controls())
            controls_scroll = QScrollArea(self)
            controls_scroll.setObjectName("UnfoldingControlsScrollArea")
            controls_scroll.setWidgetResizable(True)
            # The controls use a responsive grid and must shrink to the viewport.
            # Leaving the horizontal policy as ``AsNeeded`` creates a feedback
            # loop on some themes: the vertical bar reduces the viewport just
            # enough to summon a second, unnecessary horizontal bar.
            controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            controls_scroll.setMinimumHeight(190)
            controls_scroll.setMaximumHeight(520)
            controls_scroll.setWidget(controls_panel)
            root.addWidget(controls_scroll)

            splitter = QSplitter(Qt.Horizontal, self)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self._build_left_column())
            splitter.addWidget(self._build_right_column())
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            root.addWidget(splitter, 1)

            self.results_table = QTableWidget(0, 4, self)
            self.results_table.setObjectName("UnfoldingResultsTable")
            self.results_table.setHorizontalHeaderLabels(
                (
                    "Group",
                    f"Energy Range ({self.workspace_input.energy_unit})",
                    "Flux",
                    "Uncertainty",
                )
            )
            self.results_table.verticalHeader().setVisible(False)
            root.addWidget(self.results_table)

            self.summary_label = QLabel("", self)
            self.summary_label.setObjectName("PanelBody")
            self.summary_label.setWordWrap(True)
            root.addWidget(self.summary_label)

            self.negative_bin_label = QLabel("", self)
            self.negative_bin_label.setObjectName("PanelBody")
            self.negative_bin_label.setWordWrap(True)
            root.addWidget(self.negative_bin_label)

            self.method_selector.combo.currentIndexChanged.connect(
                self._sync_method_controls
            )
            self.compare_selector.combo.currentIndexChanged.connect(
                self._sync_method_controls
            )
            self.compare_mode_checkbox.toggled.connect(self._sync_method_controls)
            self.rmle_auto_checkbox.toggled.connect(self._sync_method_controls)
            self.use_ml_seed_checkbox.toggled.connect(self._sync_method_controls)
            self.rmle_lambda_spin.valueChanged.connect(self._sync_lambda_slider)
            self.rmle_lambda_slider.valueChanged.connect(self._sync_lambda_spin)
            self.show_uncertainty_bands_checkbox.toggled.connect(self._refresh_plots)
            self.log_energy_checkbox.toggled.connect(self._apply_plot_modes)
            self.log_flux_checkbox.toggled.connect(self._apply_plot_modes)
            self.reset_plots_button.clicked.connect(self._reset_plot_views)
            self.run_button.clicked.connect(self._run_selected_method)
            self.compare_button.clicked.connect(self._run_comparison)
            self._populate_measurement_table()
            self._update_response_matrix_view()
            self._sync_method_controls()
            self._run_selected_method()

        def _build_controls(self):
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)

            self.method_selector = MethodSelectorWidget(
                self.registries.unfolders,
                self.mode_manager,
                title="Method A",
                parent=self,
            )
            grid.addWidget(self.method_selector, 0, 0)

            self.compare_selector = MethodSelectorWidget(
                self.registries.unfolders,
                self.mode_manager,
                title="Method B",
                parent=self,
            )
            if self.compare_selector.combo.count() > 1:
                self.compare_selector.combo.setCurrentIndex(1)
            grid.addWidget(self.compare_selector, 0, 1)

            response_group = QGroupBox("Response Matrix", self)
            response_layout = QGridLayout(response_group)
            response_layout.addWidget(QLabel("Source", response_group), 0, 0)
            self.response_source_combo = QComboBox(response_group)
            self.response_source_combo.setObjectName("ResponseSourceCombo")
            self.response_source_combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            self.response_source_combo.setMinimumContentsLength(14)
            self.response_source_combo.addItem("Demo Response", "demo")
            self.response_source_combo.addItem("User CSV", "user_csv")
            self.response_source_combo.addItem("MCNP / GEANT4 Table", "mcnp_geant4")
            self.response_source_combo.addItem("Analytical HPGe", "analytical_hpge")
            self.response_source_combo.addItem(
                "UWNR RAFM Simplified Response",
                "uwnr_flux_wires",
            )
            response_layout.addWidget(self.response_source_combo, 0, 1)
            response_layout.addWidget(QLabel("Path / note", response_group), 1, 0)
            self.response_path_input = QLineEdit(response_group)
            self.response_path_input.setObjectName("ResponsePathInput")
            self.response_path_input.setPlaceholderText(
                "CSV/TSV path for file-backed response matrices"
            )
            response_layout.addWidget(self.response_path_input, 1, 1)
            self.response_load_button = QPushButton("Load Response", response_group)
            self.response_load_button.setObjectName("LoadResponseMatrixButton")
            self.response_load_button.clicked.connect(
                self._load_selected_response_matrix
            )
            response_layout.addWidget(self.response_load_button, 2, 0, 1, 2)
            response_layout.addWidget(QLabel("Measured rates", response_group), 3, 0)
            self.rates_path_input = QLineEdit(response_group)
            self.rates_path_input.setObjectName("ReactionRatesPathInput")
            self.rates_path_input.setPlaceholderText(
                "FluxForge reaction-rates JSON or RAFM CSV"
            )
            response_layout.addWidget(self.rates_path_input, 3, 1)
            self.rates_load_button = QPushButton("Load Measured Rates", response_group)
            self.rates_load_button.setObjectName("LoadMeasuredRatesButton")
            self.rates_load_button.clicked.connect(self._load_measured_rates)
            response_layout.addWidget(self.rates_load_button, 4, 0, 1, 2)
            grid.addWidget(response_group, 1, 0, 1, 2)

            maxed_group = QGroupBox("MAXED Controls", self)
            maxed_layout = QGridLayout(maxed_group)
            maxed_layout.addWidget(QLabel("Entropy Weight", maxed_group), 0, 0)
            self.entropy_weight_spin = QDoubleSpinBox(maxed_group)
            self.entropy_weight_spin.setObjectName("MaxedEntropyWeightSpin")
            self.entropy_weight_spin.setDecimals(3)
            self.entropy_weight_spin.setRange(0.01, 10.0)
            self.entropy_weight_spin.setSingleStep(0.05)
            self.entropy_weight_spin.setValue(0.02)
            maxed_layout.addWidget(self.entropy_weight_spin, 0, 1)
            grid.addWidget(maxed_group, 2, 0)

            rmle_group = QGroupBox("RMLE Controls", self)
            rmle_layout = QGridLayout(rmle_group)
            self.rmle_auto_checkbox = QCheckBox("Auto λ (L-curve)", rmle_group)
            self.rmle_auto_checkbox.setObjectName("RmleAutoLambdaCheck")
            self.rmle_auto_checkbox.setChecked(True)
            rmle_layout.addWidget(self.rmle_auto_checkbox, 0, 0, 1, 2)
            rmle_layout.addWidget(QLabel("Regularization λ", rmle_group), 1, 0)
            self.rmle_lambda_spin = QDoubleSpinBox(rmle_group)
            self.rmle_lambda_spin.setObjectName("RmleRegularizationSpin")
            self.rmle_lambda_spin.setDecimals(4)
            self.rmle_lambda_spin.setRange(0.0001, 1000.0)
            self.rmle_lambda_spin.setSingleStep(0.1)
            self.rmle_lambda_spin.setValue(1.0)
            rmle_layout.addWidget(self.rmle_lambda_spin, 1, 1)
            rmle_layout.addWidget(QLabel("λ Slider", rmle_group), 2, 0)
            self.rmle_lambda_slider = QSlider(Qt.Horizontal, rmle_group)
            self.rmle_lambda_slider.setObjectName("RmleLambdaSlider")
            self.rmle_lambda_slider.setRange(0, 1000)
            self.rmle_lambda_slider.setValue(
                self._lambda_to_slider(self.rmle_lambda_spin.value())
            )
            rmle_layout.addWidget(self.rmle_lambda_slider, 2, 1)
            rmle_layout.addWidget(QLabel("Penalty", rmle_group), 3, 0)
            self.rmle_penalty_combo = QComboBox(rmle_group)
            self.rmle_penalty_combo.setObjectName("RmlePenaltyCombo")
            self.rmle_penalty_combo.addItem("2nd derivative", "second_derivative")
            self.rmle_penalty_combo.addItem("1st derivative", "first_derivative")
            self.rmle_penalty_combo.addItem("L2", "l2")
            rmle_layout.addWidget(self.rmle_penalty_combo, 3, 1)
            grid.addWidget(rmle_group, 2, 1)

            ml_seed_group = QGroupBox("ML Seed Controls", self)
            ml_seed_layout = QGridLayout(ml_seed_group)
            self.use_ml_seed_checkbox = QCheckBox(
                "Use ML Seed initializer for GRAVEL/RMLE",
                ml_seed_group,
            )
            self.use_ml_seed_checkbox.setObjectName("UseMlSeedCheck")
            ml_seed_layout.addWidget(self.use_ml_seed_checkbox, 0, 0, 1, 2)
            ml_seed_layout.addWidget(
                QLabel("Confidence Threshold", ml_seed_group), 1, 0
            )
            self.ml_seed_threshold_spin = QDoubleSpinBox(ml_seed_group)
            self.ml_seed_threshold_spin.setObjectName("MlSeedThresholdSpin")
            self.ml_seed_threshold_spin.setDecimals(2)
            self.ml_seed_threshold_spin.setRange(0.0, 1.0)
            self.ml_seed_threshold_spin.setSingleStep(0.05)
            self.ml_seed_threshold_spin.setValue(0.6)
            ml_seed_layout.addWidget(self.ml_seed_threshold_spin, 1, 1)
            grid.addWidget(ml_seed_group, 3, 0, 1, 2)

            action_grid = QGridLayout()
            action_grid.setContentsMargins(0, 0, 0, 0)
            action_grid.setHorizontalSpacing(12)
            action_grid.setVerticalSpacing(6)

            self.compare_mode_checkbox = QCheckBox("Algorithm comparison mode", self)
            self.compare_mode_checkbox.setObjectName("UnfoldingCompareModeCheck")
            action_grid.addWidget(self.compare_mode_checkbox, 0, 0)

            self.show_uncertainty_bands_checkbox = QCheckBox(
                "Show uncertainty bands",
                self,
            )
            self.show_uncertainty_bands_checkbox.setObjectName(
                "UnfoldingUncertaintyBandsCheck"
            )
            self.show_uncertainty_bands_checkbox.setChecked(True)
            self.show_uncertainty_bands_checkbox.setToolTip(
                "Show the one-sigma uncertainty estimate returned by each method."
            )
            action_grid.addWidget(self.show_uncertainty_bands_checkbox, 0, 1)

            self.log_energy_checkbox = QCheckBox("Log energy", self)
            self.log_energy_checkbox.setObjectName("UnfoldingLogEnergyCheck")
            action_grid.addWidget(self.log_energy_checkbox, 0, 2)

            self.log_flux_checkbox = QCheckBox("Log flux", self)
            self.log_flux_checkbox.setObjectName("UnfoldingLogFluxCheck")
            action_grid.addWidget(self.log_flux_checkbox, 0, 3)

            self.reset_plots_button = QPushButton("Reset Plot Views", self)
            self.reset_plots_button.setObjectName("ResetUnfoldingPlotsButton")
            action_grid.addWidget(self.reset_plots_button, 1, 1)

            self.run_button = QPushButton("Run Selected", self)
            self.run_button.setObjectName("RunSelectedUnfoldingButton")
            action_grid.addWidget(self.run_button, 1, 2)

            self.compare_button = QPushButton("Run Comparison", self)
            self.compare_button.setObjectName("RunComparisonUnfoldingButton")
            action_grid.addWidget(self.compare_button, 1, 3)

            grid.addLayout(action_grid, 4, 0, 1, 2)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            return grid

        def _build_left_column(self) -> QWidget:
            panel = QWidget(self)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)

            measurements_group = QGroupBox("Measured Rates", panel)
            measurements_layout = QVBoxLayout(measurements_group)
            self.measurements_table = QTableWidget(0, 3, measurements_group)
            self.measurements_table.setObjectName("UnfoldingMeasurementsTable")
            self.measurements_table.setHorizontalHeaderLabels(
                ("Measurement", "Rate", "Uncertainty")
            )
            self.measurements_table.verticalHeader().setVisible(False)
            measurements_layout.addWidget(self.measurements_table)
            layout.addWidget(measurements_group)

            comparison_group = QGroupBox("Method Comparison", panel)
            comparison_layout = QVBoxLayout(comparison_group)
            self.comparison_table = QTableWidget(0, 5, comparison_group)
            self.comparison_table.setObjectName("UnfoldingComparisonTable")
            self.comparison_table.setHorizontalHeaderLabels(
                ("Method", "Chi²/dof", "Iterations", "Neg Bins", "Uncertainty")
            )
            self.comparison_table.verticalHeader().setVisible(False)
            comparison_layout.addWidget(self.comparison_table)
            layout.addWidget(comparison_group)

            return panel

        def _build_right_column(self) -> QWidget:
            panel = QWidget(self)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)

            self.flux_plot = pg.PlotWidget(panel)
            self.flux_plot.setObjectName("UnfoldingFluxPlot")
            catalog_pyqtgraph_export_action(self.flux_plot, "UnfoldingFluxPlot")
            self.flux_plot.setBackground("#0f172a")
            self.flux_plot.showGrid(x=True, y=True, alpha=0.12)
            self.flux_plot.setLabel(
                "bottom", "Energy", units=self.workspace_input.energy_unit
            )
            self.flux_plot.setLabel("left", "Flux")
            layout.addWidget(self.flux_plot, 2)

            self.convergence_plot = pg.PlotWidget(panel)
            self.convergence_plot.setObjectName("UnfoldingConvergencePlot")
            catalog_pyqtgraph_export_action(
                self.convergence_plot,
                "UnfoldingConvergencePlot",
            )
            self.convergence_plot.setBackground("#0f172a")
            self.convergence_plot.showGrid(x=True, y=True, alpha=0.12)
            self.convergence_plot.setLabel("bottom", "Iteration")
            self.convergence_plot.setLabel("left", "Objective / Chi²")
            layout.addWidget(self.convergence_plot, 1)

            self.response_plot = pg.PlotWidget(panel)
            self.response_plot.setObjectName("UnfoldingResponseMatrixPlot")
            catalog_pyqtgraph_export_action(
                self.response_plot,
                "UnfoldingResponseMatrixPlot",
            )
            self.response_plot.setBackground("#0f172a")
            self.response_plot.setLabel("bottom", "Measurement")
            self.response_plot.setLabel("left", "Energy Group")
            self.response_image = pg.ImageItem()
            self.response_plot.getPlotItem().addItem(self.response_image)
            layout.addWidget(self.response_plot, 1)

            return panel

        def _sync_method_controls(self, *_args) -> None:
            compare_enabled = self.compare_mode_checkbox.isChecked()
            self.compare_selector.combo.setEnabled(
                compare_enabled and self.compare_selector.combo.count() > 1
            )
            uses_maxed = self.method_selector.current_key() == "maxed"
            uses_rmle = self.method_selector.current_key() == "rmle"
            uses_gravel_or_rmle = self.method_selector.current_key() in {
                "gravel",
                "rmle",
            }
            uses_ml_seed = self.method_selector.current_key() == "ml_seed"
            if compare_enabled:
                uses_maxed = (
                    uses_maxed or self.compare_selector.current_key() == "maxed"
                )
                uses_rmle = uses_rmle or self.compare_selector.current_key() == "rmle"
                uses_gravel_or_rmle = uses_gravel_or_rmle or (
                    self.compare_selector.current_key() in {"gravel", "rmle"}
                )
                uses_ml_seed = (
                    uses_ml_seed or self.compare_selector.current_key() == "ml_seed"
                )
            self.entropy_weight_spin.setEnabled(uses_maxed)
            self.rmle_auto_checkbox.setEnabled(uses_rmle)
            self.rmle_penalty_combo.setEnabled(uses_rmle)
            self.rmle_lambda_spin.setEnabled(
                uses_rmle and not self.rmle_auto_checkbox.isChecked()
            )
            self.rmle_lambda_slider.setEnabled(
                uses_rmle and not self.rmle_auto_checkbox.isChecked()
            )
            self.use_ml_seed_checkbox.setEnabled(uses_gravel_or_rmle)
            self.ml_seed_threshold_spin.setEnabled(
                uses_ml_seed
                or (uses_gravel_or_rmle and self.use_ml_seed_checkbox.isChecked())
            )
            self.show_uncertainty_bands_checkbox.setEnabled(True)
            self.compare_button.setEnabled(compare_enabled)

        @staticmethod
        def _slider_to_lambda(value: int) -> float:
            exponent = -4.0 + (float(value) / 1000.0) * 7.0
            return float(10.0**exponent)

        @classmethod
        def _lambda_to_slider(cls, value: float) -> int:
            clamped = min(max(float(value), 1e-4), 1000.0)
            position = (np.log10(clamped) + 4.0) / 7.0
            return int(round(position * 1000.0))

        def _sync_lambda_slider(self, value: float) -> None:
            slider_value = self._lambda_to_slider(value)
            if self.rmle_lambda_slider.value() == slider_value:
                return
            self.rmle_lambda_slider.blockSignals(True)
            self.rmle_lambda_slider.setValue(slider_value)
            self.rmle_lambda_slider.blockSignals(False)

        def _sync_lambda_spin(self, value: int) -> None:
            lambda_value = self._slider_to_lambda(value)
            if np.isclose(
                self.rmle_lambda_spin.value(),
                lambda_value,
                rtol=1e-6,
                atol=1e-8,
            ):
                return
            self.rmle_lambda_spin.blockSignals(True)
            self.rmle_lambda_spin.setValue(lambda_value)
            self.rmle_lambda_spin.blockSignals(False)

        def _selected_method_keys(self) -> tuple[str, ...]:
            primary = (
                self.method_selector.current_key()
                or self.registries.unfolders.default_key
            )
            assert primary is not None
            if not self.compare_mode_checkbox.isChecked():
                return (primary,)
            secondary = self.compare_selector.current_key() or primary
            if secondary == primary:
                return (primary,)
            return (primary, secondary)

        def _populate_measurement_table(self) -> None:
            measured = self.workspace_input.measured_rates
            sigma = self.workspace_input.measurement_uncertainty
            labels = self.workspace_input.measurement_labels
            self.measurements_table.setRowCount(len(measured))
            for row, (value, uncertainty) in enumerate(zip(measured, sigma)):
                self._set_table_item(
                    self.measurements_table,
                    row,
                    0,
                    labels[row] if row < len(labels) else f"M{row + 1}",
                )
                self._set_table_item(self.measurements_table, row, 1, value)
                self._set_table_item(self.measurements_table, row, 2, uncertainty)

        def _update_response_matrix_view(self) -> None:
            image = np.asarray(self.workspace_input.response_matrix, dtype=float).T
            self.response_image.setImage(image)
            rows, columns = self.workspace_input.response_matrix.shape
            self.response_plot.setTitle(
                f"{self.workspace_input.label} ({rows}×{columns})"
            )

        def _load_measured_rates(self) -> None:
            path = self.rates_path_input.text().strip()
            if not path:
                self.summary_label.setText(
                    "Choose a FluxForge reaction-rates JSON or RAFM CSV file first."
                )
                return
            try:
                source_path = Path(path)
                is_rafm_csv = source_path.suffix.lower() == ".csv"
                if is_rafm_csv:
                    with source_path.open("r", encoding="utf-8", newline="") as handle:
                        rows = [
                            row
                            for row in csv.DictReader(handle)
                            if not str(row.get("reaction_id", "")).startswith(
                                "Unknown("
                            )
                        ]
                    rate_key = "reaction_rate"
                    uncertainty_key = "reaction_rate_unc"
                else:
                    payload = read_reaction_rates(source_path)
                    rows = list(payload.get("rates", []))
                    rate_key = "rate"
                    uncertainty_key = "uncertainty"
                rates = np.asarray(
                    [float(row[rate_key]) for row in rows],
                    dtype=float,
                )
                uncertainties = np.asarray(
                    [float(row.get(uncertainty_key, 0.0)) for row in rows],
                    dtype=float,
                )
                if rates.size == 0:
                    raise ValueError("reaction-rates artifact contains no rates")
                if np.any(rates < 0.0) or np.any(uncertainties < 0.0):
                    raise ValueError("rates and uncertainties must be non-negative")
                labels = tuple(
                    str(
                        row.get("reaction_id") or row.get("reaction") or f"M{index + 1}"
                    )
                    for index, row in enumerate(rows)
                )
                if is_rafm_csv:
                    self.response_source_combo.setCurrentIndex(
                        self.response_source_combo.findData("uwnr_flux_wires")
                    )
                    self.workspace_input = self._uwnr_flux_wire_input(
                        rates,
                        uncertainties,
                        labels,
                    )
                else:
                    if rates.size != self.workspace_input.response_matrix.shape[0]:
                        raise ValueError(
                            f"rate count {rates.size} does not match response rows "
                            f"{self.workspace_input.response_matrix.shape[0]}"
                        )
                    self.workspace_input = UnfoldingWorkspaceInput(
                        label=self.workspace_input.label,
                        measured_rates=rates,
                        measurement_uncertainty=uncertainties,
                        response_matrix=self.workspace_input.response_matrix,
                        energy_edges=self.workspace_input.energy_edges,
                        initial_flux=self.workspace_input.initial_flux,
                        measurement_labels=labels,
                        energy_unit=self.workspace_input.energy_unit,
                    )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                self.summary_label.setText(f"Measured rates were not loaded: {exc}")
                return
            self._populate_measurement_table()
            self._update_response_matrix_view()
            self._run_selected_method()

        def _load_selected_response_matrix(self) -> None:
            try:
                source_key = str(self.response_source_combo.currentData() or "demo")
                if source_key == "demo":
                    self.workspace_input = build_demo_unfolding_workspace_input()
                elif source_key == "analytical_hpge":
                    loaded = build_analytical_hpge_response(
                        n_channels=int(self.workspace_input.measured_rates.size),
                        energy_edges=self.workspace_input.energy_edges,
                    )
                    self.workspace_input = self._workspace_input_from_loaded_response(
                        loaded
                    )
                elif source_key == "uwnr_flux_wires":
                    if not self.workspace_input.measurement_labels or all(
                        label.startswith("M")
                        for label in self.workspace_input.measurement_labels
                    ):
                        raise ValueError(
                            "Load the UWNR RAFM reaction-rate CSV before selecting its response."
                        )
                    self.workspace_input = self._uwnr_flux_wire_input(
                        self.workspace_input.measured_rates,
                        self.workspace_input.measurement_uncertainty,
                        self.workspace_input.measurement_labels,
                    )
                else:
                    path = self.response_path_input.text().strip()
                    if not path:
                        self.summary_label.setText(
                            "Choose a response-matrix file first."
                        )
                        return
                    loaded = load_response_matrix(
                        path,
                        source_format=(
                            "mcnp_geant4_table"
                            if source_key == "mcnp_geant4"
                            else "user_csv"
                        ),
                        energy_edges=self.workspace_input.energy_edges,
                    )
                    self.workspace_input = self._workspace_input_from_loaded_response(
                        loaded
                    )
            except (OSError, TypeError, ValueError) as exc:
                self.summary_label.setText(f"Response matrix was not loaded: {exc}")
                return
            self._populate_measurement_table()
            self._update_response_matrix_view()
            self._run_selected_method()

        def _uwnr_flux_wire_input(
            self,
            rates: np.ndarray,
            uncertainties: np.ndarray,
            labels: tuple[str, ...],
        ) -> UnfoldingWorkspaceInput:
            # Match the 20-group structure and simplified response curves used by
            # the repository's committed UWNR RAFM unfolding artifacts.  This
            # path is deliberately labelled as simplified: production work can
            # still load an evaluated IRDFF response matrix through User CSV.
            energy_edges = np.logspace(np.log10(0.0253), np.log10(20.0e6), 21)
            matrix = np.asarray(
                [
                    _make_response_row(label, energy_edges, energy_edges.size - 1)
                    for label in labels
                ],
                dtype=float,
            )
            response_scale = max(float(np.mean(matrix)), np.finfo(float).tiny)
            initial_level = max(
                float(np.mean(rates)) / response_scale / matrix.shape[1],
                np.finfo(float).tiny,
            )
            return UnfoldingWorkspaceInput(
                label="UWNR RAFM Simplified Flux-Wire Response",
                measured_rates=np.asarray(rates, dtype=float),
                measurement_uncertainty=np.asarray(uncertainties, dtype=float),
                response_matrix=np.asarray(matrix, dtype=float),
                energy_edges=np.asarray(energy_edges, dtype=float),
                initial_flux=np.full(matrix.shape[1], initial_level, dtype=float),
                measurement_labels=labels,
                energy_unit="eV",
            )

        def _workspace_input_from_loaded_response(
            self, loaded
        ) -> UnfoldingWorkspaceInput:
            matrix = np.asarray(loaded.matrix, dtype=float)
            energy_edges = np.asarray(loaded.energy_edges, dtype=float)
            initial_flux = np.asarray(self.workspace_input.initial_flux, dtype=float)
            if initial_flux.size != matrix.shape[1]:
                initial_flux = np.full(
                    matrix.shape[1],
                    float(np.mean(self.workspace_input.initial_flux)),
                    dtype=float,
                )
            measured_rates = np.asarray(
                self.workspace_input.measured_rates, dtype=float
            )
            measurement_uncertainty = np.asarray(
                self.workspace_input.measurement_uncertainty, dtype=float
            )
            if measured_rates.size != matrix.shape[0]:
                raise ValueError(
                    f"response rows {matrix.shape[0]} do not match the active "
                    f"measured-rate count {measured_rates.size}"
                )
            return UnfoldingWorkspaceInput(
                label=loaded.source_label,
                measured_rates=measured_rates,
                measurement_uncertainty=measurement_uncertainty,
                response_matrix=matrix,
                energy_edges=energy_edges,
                initial_flux=initial_flux,
                measurement_labels=self.workspace_input.measurement_labels,
                energy_unit=self.workspace_input.energy_unit,
            )

        def _run_selected_method(self) -> None:
            method_keys = self._selected_method_keys()
            self.comparison_results = {
                method_key: self._run_method(method_key) for method_key in method_keys
            }
            self.current_result = self.comparison_results[method_keys[0]]
            self._refresh_view()

        def _run_comparison(self) -> None:
            method_keys = self._selected_method_keys()
            self.comparison_results = {
                method_key: self._run_method(method_key) for method_key in method_keys
            }
            self.current_result = self.comparison_results[method_keys[0]]
            self._refresh_view()

        def _run_method(self, method_key: str) -> UnfoldingResult:
            unfolder = self.registries.unfolders.get(method_key)
            kwargs = {
                "initial_flux": self.workspace_input.initial_flux,
                "measurement_uncertainty": self.workspace_input.measurement_uncertainty,
            }
            if method_key == "maxed":
                kwargs["entropy_weight"] = self.entropy_weight_spin.value()
            if method_key == "rmle":
                kwargs["regularization_strength"] = self.rmle_lambda_spin.value()
                kwargs["regularization_type"] = (
                    self.rmle_penalty_combo.currentData() or "second_derivative"
                )
                kwargs["auto_regularization"] = self.rmle_auto_checkbox.isChecked()
            if (
                method_key in {"gravel", "rmle"}
                and self.use_ml_seed_checkbox.isChecked()
            ):
                kwargs["seed_with_ml"] = True
                kwargs["confidence_threshold"] = self.ml_seed_threshold_spin.value()
            if method_key == "ml_seed":
                kwargs["confidence_threshold"] = self.ml_seed_threshold_spin.value()
            return unfolder.unfold(
                self.workspace_input.measured_rates,
                self.workspace_input.response_matrix,
                **kwargs,
            )

        def _refresh_view(self) -> None:
            self._refresh_comparison_table()
            self._refresh_plots()
            if self.current_result is not None:
                self._refresh_results_table(self.current_result)
                uncertainty_note = (
                    "visible"
                    if self.current_result.uncertainties is not None
                    else "not available"
                )
                seed_confidence = self.current_result.parameters_used.get(
                    "seed_confidence_score"
                )
                if self.current_result.method_used == "ML Seed":
                    seed_confidence = self.current_result.parameters_used.get(
                        "confidence_score"
                    )
                confidence_note = ""
                if seed_confidence is not None:
                    confidence_note = f" Seed confidence {float(seed_confidence):.2f}."
                self.summary_label.setText(
                    f"{self.current_result.method_used} completed in "
                    f"{self.current_result.iterations} iterations with "
                    f"chi²/dof {self.current_result.chi_squared:.4f}. "
                    f"Flux values and uncertainties are {uncertainty_note} in the table below."
                    f"{confidence_note}"
                )
                if self.current_result.negative_bin_count > 0:
                    self.negative_bin_label.setText(
                        f"{self.current_result.negative_bin_count} negative bins reported. "
                        "Review the comparison table and flux plot before accepting the solution."
                    )
                else:
                    self.negative_bin_label.setText(
                        "No negative bins were reported for the current unfolding solution."
                    )

        def _refresh_comparison_table(self) -> None:
            rows = list(self.comparison_results.items())
            self.comparison_table.setRowCount(len(rows))
            for row, (key, result) in enumerate(rows):
                uncertainty_state = (
                    "Shown" if result.uncertainties is not None else "N/A"
                )
                self._set_table_item(self.comparison_table, row, 0, key.upper())
                self._set_table_item(self.comparison_table, row, 1, result.chi_squared)
                self._set_table_item(self.comparison_table, row, 2, result.iterations)
                self._set_table_item(
                    self.comparison_table,
                    row,
                    3,
                    result.negative_bin_count,
                )
                self._set_table_item(self.comparison_table, row, 4, uncertainty_state)

        def _refresh_results_table(self, result: UnfoldingResult) -> None:
            energy_edges = np.asarray(self.workspace_input.energy_edges, dtype=float)
            self.results_table.setHorizontalHeaderLabels(
                (
                    "Group",
                    f"Energy Range ({self.workspace_input.energy_unit})",
                    "Flux",
                    "Uncertainty",
                )
            )
            self.flux_plot.setLabel(
                "bottom",
                "Energy",
                units=self.workspace_input.energy_unit,
            )
            self.results_table.setRowCount(len(result.flux))
            for row, flux_value in enumerate(result.flux):
                band = f"{energy_edges[row]:.3f}-{energy_edges[row + 1]:.3f}"
                uncertainty = (
                    result.uncertainties[row]
                    if result.uncertainties is not None
                    else "N/A"
                )
                self._set_table_item(self.results_table, row, 0, row + 1)
                self._set_table_item(self.results_table, row, 1, band)
                self._set_table_item(self.results_table, row, 2, flux_value)
                self._set_table_item(self.results_table, row, 3, uncertainty)

        def _refresh_plots(self) -> None:
            edges = np.asarray(self.workspace_input.energy_edges, dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            positive = (edges[:-1] > 0.0) & (edges[1:] > 0.0)
            centers[positive] = np.sqrt(edges[:-1][positive] * edges[1:][positive])
            self.flux_plot.clear()
            self.flux_plot.addLegend(offset=(10, 10))
            self.convergence_plot.clear()
            self.convergence_plot.addLegend(offset=(10, 10))

            palette = {
                "gravel": "#72d6ff",
                "maxed": "#f59e0b",
                "ml_seed": "#a78bfa",
                "rmle": "#34d399",
            }
            for key, result in self.comparison_results.items():
                color = palette.get(key, "#cbd5e1")
                flux_curve = self.flux_plot.plot(
                    centers,
                    result.flux,
                    pen=pg.mkPen(color=color, width=2),
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=pg.mkBrush(color),
                    name=result.method_used,
                )
                if (
                    self.show_uncertainty_bands_checkbox.isChecked()
                    and result.uncertainties is not None
                ):
                    upper = np.asarray(result.flux, dtype=float) + np.asarray(
                        result.uncertainties,
                        dtype=float,
                    )
                    lower = np.maximum(
                        np.asarray(result.flux, dtype=float)
                        - np.asarray(result.uncertainties, dtype=float),
                        0.0,
                    )
                    upper_curve = self.flux_plot.plot(
                        centers,
                        upper,
                        pen=pg.mkPen(color=color, width=1, style=Qt.DashLine),
                    )
                    lower_curve = self.flux_plot.plot(
                        centers,
                        lower,
                        pen=pg.mkPen(color=color, width=1, style=Qt.DashLine),
                    )
                    band = pg.FillBetweenItem(
                        upper_curve,
                        lower_curve,
                        brush=pg.mkBrush(color + "33"),
                    )
                    self.flux_plot.addItem(band)
                history = np.asarray(result.convergence_history, dtype=float)
                if history.size > 0:
                    self.convergence_plot.plot(
                        np.arange(history.size, dtype=float),
                        history,
                        pen=pg.mkPen(color=color, width=1.8),
                        name=result.method_used,
                    )
            self._apply_plot_modes()

        def _apply_plot_modes(self, *_args) -> None:
            self.flux_plot.setLogMode(
                x=self.log_energy_checkbox.isChecked(),
                y=self.log_flux_checkbox.isChecked(),
            )
            self.convergence_plot.setLogMode(
                x=False,
                y=self.log_flux_checkbox.isChecked(),
            )

        def _reset_plot_views(self) -> None:
            for plot in (self.flux_plot, self.convergence_plot, self.response_plot):
                plot.enableAutoRange()

        def _set_table_item(
            self, table: QTableWidget, row: int, column: int, value
        ) -> None:
            if isinstance(value, str):
                text = value
            elif isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = f"{float(value):.6f}".rstrip("0").rstrip(".")
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, column, item)

else:

    class UnfoldingWorkspaceDialog:  # pragma: no cover - placeholder without GUI deps
        """Import-safe placeholder when Qt or PyQtGraph is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "The unfolding workspace requires the native Qt GUI extras."
            )


__all__ = [
    "UnfoldingWorkspaceDialog",
    "UnfoldingWorkspaceInput",
    "build_demo_unfolding_workspace_input",
]
