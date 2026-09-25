"""Efficiency calibration dialog for the modern analysis workspace."""

from __future__ import annotations

import math
import csv
from pathlib import Path
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.analysis_workspace import (
    EfficiencyCalibrationFitResult,
    register_builtin_efficiency_models,
    fit_efficiency_model,
)
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.backends.pyqtgraph_backend import PYQTGRAPH_AVAILABLE, catalog_pyqtgraph_export_action
from fluxforge.gui.widgets import MethodSelectorWidget
from fluxforge.io.flux_wire import EfficiencyCalibration
from fluxforge.plugins import bootstrap_builtin_registries

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    if PYQTGRAPH_AVAILABLE:
        import pyqtgraph as pg
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class EfficiencyCalibrationDialog(QDialog):
        """Fit and apply registered detector-efficiency models."""

        # Exact CSV header contract; import is atomic if any row is invalid.
        CSV_COLUMNS = (
            "energy_keV", "net_counts", "count_uncertainty", "live_time_s",
            "activity_bq", "activity_rel_unc", "emission_probability",
            "probability_uncertainty", "geometry_factor",
        )

        HEADERS = (
            "Energy keV",
            "Net Counts",
            "Count Unc.",
            "Live Time s",
            "Activity Bq",
            "Activity Rel. Unc.",
            "Gamma Intensity",
            "Intensity Unc.",
            "Geometry Factor",
            "Activity Source ID",
        )

        DETECTOR_FIELDS = (
            ("C1", "Efficiency C1", -1.0e6, 1.0e6),
            ("C2", "Efficiency C2", -1.0e6, 1.0e6),
            ("C3", "Efficiency C3", -1.0e6, 1.0e6),
            ("C4", "Efficiency C4", -1.0e6, 1.0e6),
            ("geometry_factor_A", "Geometry factor A", 0.0, 1.0e6),
            ("al_window_T1_um", "Al window T1 (um)", 0.0, 1.0e6),
            ("detector_thickness_DI_cm", "Detector thickness DI (cm)", 0.0, 1.0e4),
            ("dead_layer_DL_um", "Dead layer DL (um)", 0.0, 1.0e6),
            ("incident_angle_AI_deg", "Incident angle AI (deg)", -360.0, 360.0),
            ("detector_diameter_cm", "Detector diameter (cm)", 0.0, 1.0e4),
            ("source_distance_cm", "Source distance (cm)", 0.0, 1.0e6),
            ("relative_uncertainty", "Relative uncertainty (fraction)", 0.0, 1.0),
        )

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            points: Sequence[EfficiencyPoint] | None = None,
            detector_calibration: EfficiencyCalibration | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Efficiency Calibration")
            self.resize(1200, 940)

            registries = bootstrap_builtin_registries()
            if len(registries.calibration_models) == 0:
                register_builtin_efficiency_models(registries)
            self._registry = registries.calibration_models
            self._fit_result: EfficiencyCalibrationFitResult | None = None
            self._fitted_points: tuple[EfficiencyPoint, ...] = ()
            self._detector_calibration = detector_calibration or EfficiencyCalibration(
                relative_uncertainty=0.05
            )

            root = QVBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(10)

            intro = QLabel(
                (
                    "Fit one of the registered efficiency models. The resulting curve "
                    "is used by the activity workflow."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            root.addWidget(intro)

            self.method_selector = MethodSelectorWidget(
                self._registry,
                mode_manager,
                title="Efficiency model",
                parent=self,
            )
            root.addWidget(self.method_selector)

            detector_group = QGroupBox("Detector geometry / certificate metadata (not fitted)", self)
            detector_group.setObjectName("HpgeDetectorParametersGroup")
            detector_layout = QGridLayout(detector_group)
            detector_layout.addWidget(QLabel("Detector ID", detector_group), 0, 0)
            self.detector_id_edit = QLineEdit(detector_group)
            self.detector_id_edit.setObjectName("HpgeDetectorIdEdit")
            self.detector_id_edit.setText(self._detector_calibration.detector_id)
            detector_layout.addWidget(self.detector_id_edit, 0, 1, 1, 3)

            self.detector_fields: dict[str, QDoubleSpinBox] = {}
            for index, (attribute, label, minimum, maximum) in enumerate(
                self.DETECTOR_FIELDS
            ):
                row = 1 + index // 2
                column = (index % 2) * 2
                detector_layout.addWidget(QLabel(label, detector_group), row, column)
                field = QDoubleSpinBox(detector_group)
                field.setObjectName(f"Hpge{attribute}Spin")
                field.setDecimals(8)
                field.setRange(minimum, maximum)
                field.setValue(float(getattr(self._detector_calibration, attribute)))
                field.valueChanged.connect(self._invalidate_fit)
                detector_layout.addWidget(field, row, column + 1)
                self.detector_fields[attribute] = field
            root.addWidget(detector_group)

            self.table = QTableWidget(0, len(self.HEADERS), self)
            self.table.setObjectName("EfficiencyCalibrationTable")
            self.table.setHorizontalHeaderLabels(self.HEADERS)
            self.table.setMinimumHeight(145)
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setStretchLastSection(False)
            root.addWidget(self.table, 1)

            edit_actions = QHBoxLayout()
            self.add_button = QPushButton("Add row", self)
            self.add_button.setObjectName("AddEfficiencyPointButton")
            self.add_button.clicked.connect(self._add_row)
            edit_actions.addWidget(self.add_button)
            self.delete_button = QPushButton("Delete selected rows", self)
            self.delete_button.setObjectName("DeleteEfficiencyPointsButton")
            self.delete_button.clicked.connect(self._delete_selected_rows)
            edit_actions.addWidget(self.delete_button)
            self.import_button = QPushButton("Import CSV…", self)
            self.import_button.setObjectName("ImportEfficiencyCsvButton")
            self.import_button.clicked.connect(self._choose_csv)
            edit_actions.addWidget(self.import_button)
            seed_button = QPushButton("Seed demo points", self)
            seed_button.setObjectName("SeedEfficiencyExamplePointsButton")
            seed_button.clicked.connect(self._seed_demo_points)
            edit_actions.addWidget(seed_button)
            root.addLayout(edit_actions)
            self.table.itemChanged.connect(self._invalidate_fit)
            self.detector_id_edit.textChanged.connect(self._invalidate_fit)
            self.method_selector.combo.currentIndexChanged.connect(self._invalidate_fit)

            contract = QLabel(
                "CSV only (certificate/PDF formats unsupported). Required headers: "
                + ", ".join(self.CSV_COLUMNS)
                + ". Every row must be valid. Enter a common Activity Source ID "
                  "for lines sharing one source's activity uncertainty.", self,
            )
            contract.setObjectName("EfficiencyCsvContractLabel")
            contract.setWordWrap(True)
            root.addWidget(contract)

            if PYQTGRAPH_AVAILABLE:
                plots = QHBoxLayout()
                self.efficiency_plot = pg.PlotWidget(self)
                self.efficiency_plot.setObjectName("EfficiencyMeasuredFittedPlot")
                catalog_pyqtgraph_export_action(self.efficiency_plot, "EfficiencyMeasuredFittedPlot")
                self.efficiency_plot.setLabel("bottom", "Energy", units="keV")
                self.efficiency_plot.setLabel("left", "Efficiency", units="fraction")
                self.efficiency_plot.getAxis("bottom").enableAutoSIPrefix(False)
                self.efficiency_plot.getAxis("left").enableAutoSIPrefix(False)
                self.efficiency_plot.addLegend()
                self.efficiency_plot.setMinimumHeight(175)
                plots.addWidget(self.efficiency_plot, 1)
                self.residual_plot = pg.PlotWidget(self)
                self.residual_plot.setObjectName("EfficiencyPercentResidualPlot")
                catalog_pyqtgraph_export_action(self.residual_plot, "EfficiencyPercentResidualPlot")
                self.residual_plot.setLabel("bottom", "Energy", units="keV")
                self.residual_plot.setLabel("left", "Measured − fitted", units="%")
                self.residual_plot.getAxis("bottom").enableAutoSIPrefix(False)
                self.residual_plot.setMinimumHeight(175)
                plots.addWidget(self.residual_plot, 1)
                root.addLayout(plots, 1)
            else:
                self.efficiency_plot = None
                self.residual_plot = None

            self.point_diagnostics = QTableWidget(0, 5, self)
            self.point_diagnostics.setObjectName("EfficiencyPointDiagnosticsTable")
            self.point_diagnostics.setHorizontalHeaderLabels(
                ("Energy keV", "Measured", "Fitted", "Residual %", "Status")
            )
            self.point_diagnostics.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.point_diagnostics.setMinimumHeight(145)
            root.addWidget(self.point_diagnostics, 1)
            self.model_comparison = QTextBrowser(self)
            self.model_comparison.setObjectName("EfficiencyModelComparison")
            self.model_comparison.setMaximumHeight(95)
            self.model_comparison.setMinimumHeight(65)
            root.addWidget(self.model_comparison)

            self.summary = QLabel("Waiting for a fit.", self)
            self.summary.setObjectName("EfficiencyFitDiagnosticsLabel")
            self.summary.setWordWrap(True)
            root.addWidget(self.summary)

            buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                parent=self,
            )
            buttons.button(QDialogButtonBox.Ok).setObjectName(
                "AcceptEfficiencyCalibrationButton"
            )
            self.accept_button = buttons.button(QDialogButtonBox.Ok)
            self.accept_button.setEnabled(False)
            buttons.button(QDialogButtonBox.Cancel).setObjectName(
                "CancelEfficiencyCalibrationButton"
            )
            self.fit_button = buttons.addButton(
                "Fit Model", QDialogButtonBox.ActionRole
            )
            self.fit_button.setObjectName("FitEfficiencyModelButton")
            self.fit_button.clicked.connect(self._fit_model)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

            self._seed_points(points if points is not None else ())

        def _seed_points(self, points: Sequence[EfficiencyPoint]) -> None:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for point in points:
                row = self.table.rowCount()
                self.table.insertRow(row)
                for column, value in enumerate(
                    (
                        point.energy_keV,
                        point.net_counts,
                        (
                            point.count_uncertainty
                            if point.count_uncertainty is not None
                            else max(math.sqrt(abs(point.net_counts)), 1.0)
                        ),
                        point.live_time_s,
                        point.activity_bq,
                        point.activity_rel_unc or 0.0,
                        point.emission_probability,
                        point.probability_uncertainty or 0.0,
                        point.geometry_factor,
                        point.activity_source_id or "",
                    )
                ):
                    self.table.setItem(
                        row, column, QTableWidgetItem(
                            f"{float(value):.6g}" if column < 9 else str(value)
                        )
                    )
            self.table.blockSignals(False)
            self._invalidate_fit()
            self.fit_button.setEnabled(self.table.rowCount() > 0)
            if self.table.rowCount() == 0:
                self.summary.setText(
                    "Add calibration points or choose Seed demo points explicitly."
                )

        def _seed_demo_points(self) -> None:
            self._seed_points(self._demo_points())

        def _demo_points(self) -> tuple[EfficiencyPoint, ...]:
            energies = (121.78, 356.01, 661.657, 1173.228, 1332.492)
            values = (0.085, 0.041, 0.019, 0.012, 0.010)
            points: list[EfficiencyPoint] = []
            for energy, efficiency in zip(energies, values):
                points.append(
                    EfficiencyPoint(
                        energy_keV=energy,
                        net_counts=efficiency * 1e6,
                        live_time_s=100.0,
                        activity_bq=1e5,
                        emission_probability=1.0,
                        count_uncertainty=max(math.sqrt(efficiency * 1e6), 1.0),
                    )
                )
            return tuple(points)

        def _read_points(self) -> tuple[EfficiencyPoint, ...]:
            points: list[EfficiencyPoint] = []
            for row in range(self.table.rowCount()):
                try:
                    values = [float(self.table.item(row, column).text()) for column in range(9)]
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Row {row + 1}: all nine numeric fields are required") from exc
                self._validate_values(values, row + 1)
                source_item = self.table.item(row, 9)
                source_id = source_item.text().strip() if source_item is not None else ""
                points.append(EfficiencyPoint(*values[:2], values[3], values[4], values[6],
                    geometry_factor=values[8], count_uncertainty=values[2],
                    activity_rel_unc=values[5], probability_uncertainty=values[7],
                    activity_source_id=source_id or None))
            return tuple(points)

        @staticmethod
        def _validate_values(values: Sequence[float], row: int) -> None:
            if len(values) != 9 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Row {row}: nine finite numbers are required")
            for index in (0, 1, 3, 4, 6, 8):
                if values[index] <= 0:
                    raise ValueError(f"Row {row}: {EfficiencyCalibrationDialog.CSV_COLUMNS[index]} must be positive")
            for index in (2, 5, 7):
                if values[index] < 0:
                    raise ValueError(f"Row {row}: {EfficiencyCalibrationDialog.CSV_COLUMNS[index]} cannot be negative")
            if values[6] > 1 or values[5] > 1 or values[7] > 1:
                raise ValueError(f"Row {row}: relative uncertainties and emission_probability cannot exceed 1")

        def _add_row(self) -> None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column in range(len(self.HEADERS)):
                self.table.setItem(row, column, QTableWidgetItem(""))
            self.table.setCurrentCell(row, 0)
            self.fit_button.setEnabled(True)
            self._invalidate_fit()

        def _delete_selected_rows(self) -> None:
            for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
                self.table.removeRow(row)
            self.fit_button.setEnabled(self.table.rowCount() > 0)
            self._invalidate_fit()

        def _choose_csv(self) -> None:
            path, _filter = QFileDialog.getOpenFileName(self, "Import efficiency points", "", "CSV files (*.csv)")
            if path:
                try:
                    self.import_csv(path)
                except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                    self.summary.setText(f"CSV import failed: {exc}")

        def import_csv(self, path: str | Path) -> None:
            """Import an exact-header CSV, replacing the table only after full validation."""
            path = Path(path)
            if path.suffix.lower() != ".csv":
                raise ValueError("Only CSV is supported; certificate/PDF formats have no defined parser")
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream, strict=True)
                if reader.fieldnames is None or len(reader.fieldnames) != 9 or set(reader.fieldnames) != set(self.CSV_COLUMNS):
                    raise ValueError("CSV headers must be exactly: " + ", ".join(self.CSV_COLUMNS))
                parsed = []
                for row_number, row in enumerate(reader, 2):
                    try:
                        values = [float(row[name]) for name in self.CSV_COLUMNS]
                    except (TypeError, ValueError, KeyError) as exc:
                        raise ValueError(f"CSV line {row_number}: nine numeric values required") from exc
                    if None in row:
                        raise ValueError(f"CSV line {row_number}: extra columns")
                    self._validate_values(values, row_number)
                    parsed.append(values)
            if not parsed:
                raise ValueError("CSV contains no calibration rows")
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for values in parsed:
                row = self.table.rowCount()
                self.table.insertRow(row)
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(f"{value:.12g}"))
                self.table.setItem(row, 9, QTableWidgetItem(""))
            self.table.blockSignals(False)
            self.fit_button.setEnabled(True)
            self._invalidate_fit()
            self.summary.setText(f"Imported {len(parsed)} valid CSV points. Fit a model to review diagnostics.")

        def _invalidate_fit(self, *_args) -> None:
            self._fit_result = None
            self._fitted_points = ()
            if hasattr(self, "accept_button"):
                self.accept_button.setEnabled(False)
            self.point_diagnostics.setRowCount(0)
            self.model_comparison.clear()
            if self.efficiency_plot is not None:
                self.efficiency_plot.clear()
                self.residual_plot.clear()
            if hasattr(self, "summary"):
                self.summary.setText("Points changed; fit a model before accepting.")

        def _fit_model(self) -> None:
            try:
                points = self._read_points()
            except ValueError as exc:
                self._invalidate_fit()
                self.summary.setText(f"Fit blocked: {exc}")
                return
            if not points:
                self.summary.setText("No efficiency points are available.")
                return
            selected = self.method_selector.current_key() or self._registry.default_key or "log_poly_2"
            comparisons = []
            selected_result = None
            for entry in self._registry.entries():
                try:
                    result = fit_efficiency_model(points, model_key=entry.key)
                    rmse = float(result.rmse)
                    if not math.isfinite(rmse):
                        raise ValueError("non-finite RMSE")
                    comparisons.append(f"{result.model_label}: RMSE {rmse:.6g}" + (" (selected)" if entry.key == selected else ""))
                    if entry.key == selected:
                        selected_result = result
                except (ValueError, ArithmeticError, RuntimeError, TypeError, np.linalg.LinAlgError) as exc:
                    comparisons.append(f"{entry.key}: unavailable ({exc})")
            self.model_comparison.setPlainText("Model comparison (absolute efficiency RMSE):\n" + "\n".join(comparisons))
            if selected_result is None:
                self._fit_result = None
                self._fitted_points = ()
                self.summary.setText(f"Selected model {selected} could not be fitted. Review the comparison and input rows.")
                return
            try:
                self._show_fit_diagnostics(points, selected_result)
            except (ValueError, TypeError, ArithmeticError) as exc:
                self.summary.setText(f"Fit diagnostics failed: {exc}")
                return
            self._fit_result = selected_result
            self._fitted_points = points
            detector = self.detector_calibration()
            self._detector_calibration = detector
            self._fit_result.curve.detector_id = detector.detector_id
            self._fit_result.curve.geometry = {
                "geometry_factor_A": detector.geometry_factor_A,
                "al_window_T1_um": detector.al_window_T1_um,
                "detector_thickness_DI_cm": detector.detector_thickness_DI_cm,
                "dead_layer_DL_um": detector.dead_layer_DL_um,
                "incident_angle_AI_deg": detector.incident_angle_AI_deg,
                "detector_diameter_cm": detector.detector_diameter_cm,
                "source_distance_cm": detector.source_distance_cm,
            }
            self._fit_result.curve.uncertainty_model = {
                "type": "constant",
                "value": detector.relative_uncertainty,
            }
            # Metadata remains separate from model coefficients in the UI; retain
            # the legacy payload for downstream consumers of accepted_fit().
            self._fit_result.curve.parameters["detector_calibration"] = asdict(detector)
            covariance = getattr(self._fit_result, "covariance", None)
            covariance_names = getattr(self._fit_result, "covariance_parameters", ())
            covariance_text = (
                f"{len(covariance)}×{len(covariance)} for {', '.join(covariance_names)}"
                if covariance is not None else "not provided by fit API"
            )
            quality = getattr(self._fit_result, "fit_quality", {}) or {}
            p_value = quality.get("chi_square_p_value")
            p_text = "n/a" if p_value is None else f"{p_value:.3g}"
            reduced = quality.get("reduced_chi_squared")
            reduced_text = "n/a" if reduced is None else f"{reduced:.3g}"
            fixed = quality.get("fixed_parameters", ())
            conditional = (
                f" Fixed assumptions: {', '.join(fixed)}."
                if fixed else ""
            )
            self.summary.setText(
                (
                    f"{self._fit_result.model_label}: {self._fit_result.points_used} points; "
                    f"RMSE {self._fit_result.rmse:.6g}; Covariance: {covariance_text}.\n"
                    f"Review: {quality.get('review_status', 'unknown')}; "
                    f"max |residual| {quality.get('max_absolute_percentage_residual', float('nan')):.3g}%; "
                    f"reduced χ² {reduced_text}; p {p_text}.\n"
                    f"Parameters: {self._fit_parameters_text()}.{conditional} "
                    "Detector fields are metadata, not fit coefficients. "
                    "Source IDs correlate activity errors; blank IDs leave that correlation unknown."
                )
            )
            self.accept_button.setEnabled(True)

        def _fit_parameters_text(self) -> str:
            parameters = self._fit_result.curve.parameters
            return ", ".join(f"{key}={value:.6g}" if isinstance(value, (int, float))
                else f"{key}={np.array2string(np.asarray(value), precision=5, separator=',')}"
                for key, value in parameters.items()
                if key != "detector_calibration") or "none exposed"

        def _show_fit_diagnostics(self, points, result) -> None:
            energies = np.asarray([point.energy_keV for point in points], dtype=float)
            measured = np.asarray(getattr(result, "measured_efficiencies", None)
                if getattr(result, "measured_efficiencies", None)
                else [point.efficiency()[0] for point in points], dtype=float)
            fitted = np.asarray(getattr(result, "fitted_efficiencies", None)
                if getattr(result, "fitted_efficiencies", None)
                else result.curve.efficiency(energies), dtype=float)
            if measured.size != len(points) or fitted.size != len(points) or not np.all(np.isfinite(fitted)):
                raise ValueError("Fit returned invalid point diagnostics")
            residuals = getattr(result, "percentage_residuals", None)
            residuals = np.asarray(residuals if residuals
                else 100.0 * (measured - fitted) / measured, dtype=float)
            if residuals.size != len(points) or not np.all(np.isfinite(residuals)):
                raise ValueError("Fit returned invalid percentage residuals")
            statuses = getattr(result, "point_status", None)
            if not statuses:
                statuses = ["review" if abs(value) > 3 else "within ±3%" for value in residuals]
            else:
                statuses = ["review (>3%)" if status == "outside_3_percent" else
                    "within ±3%" if status == "within_3_percent" else status
                    for status in statuses]
            if len(statuses) != len(points):
                raise ValueError("Fit returned invalid point statuses")
            self.point_diagnostics.setRowCount(len(points))
            for row, (energy, obs, pred, residual, status) in enumerate(
                zip(energies, measured, fitted, residuals, statuses)
            ):
                for column, value in enumerate((f"{energy:.6g}", f"{obs:.6g}",
                    f"{pred:.6g}", f"{residual:+.3f}", str(status))):
                    self.point_diagnostics.setItem(row, column, QTableWidgetItem(value))
            if self.efficiency_plot is not None:
                self.efficiency_plot.clear()
                self.residual_plot.clear()
                order = np.argsort(energies)
                self.efficiency_plot.plot(energies, measured, pen=None, symbol="o",
                    symbolBrush="#29b6f6", name="Measured")
                self.efficiency_plot.plot(energies[order], fitted[order],
                    pen=pg.mkPen("#f5b041", width=2), name="Fitted")
                self.residual_plot.addItem(pg.LinearRegionItem(values=(-3, 3),
                    orientation="horizontal", movable=False,
                    brush=pg.mkBrush(56, 142, 60, 45)))
                self.residual_plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                    pen=pg.mkPen("#888888", style=pg.QtCore.Qt.DashLine)))
                self.residual_plot.plot(energies, residuals, pen=None, symbol="o",
                    symbolBrush="#29b6f6")

        def accepted_fit(self) -> EfficiencyCalibrationFitResult | None:
            return self._fit_result

        def calibration_points(self) -> tuple[EfficiencyPoint, ...]:
            """Return exactly the validated points of the current successful fit."""
            return self._fitted_points

        def detector_calibration(self) -> EfficiencyCalibration:
            return EfficiencyCalibration(
                detector_id=self.detector_id_edit.text().strip(),
                **{
                    attribute: float(field.value())
                    for attribute, field in self.detector_fields.items()
                },
            )

else:

    class EfficiencyCalibrationDialog:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            points: Sequence[EfficiencyPoint] | None = None,
            detector_calibration: EfficiencyCalibration | None = None,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.points = tuple(points or ())
            self._detector_calibration = detector_calibration or EfficiencyCalibration(
                relative_uncertainty=0.05
            )
            self.parent = parent

        def accepted_fit(self) -> EfficiencyCalibrationFitResult | None:
            return None

        def calibration_points(self) -> tuple[EfficiencyPoint, ...]:
            return ()

        def detector_calibration(self) -> EfficiencyCalibration:
            return self._detector_calibration
