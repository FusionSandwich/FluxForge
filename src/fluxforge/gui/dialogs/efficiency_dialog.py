"""Efficiency calibration dialog for the modern analysis workspace."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.analysis_workspace import (
    EfficiencyCalibrationFitResult,
    register_builtin_efficiency_models,
    fit_efficiency_model,
)
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.widgets import MethodSelectorWidget
from fluxforge.io.flux_wire import EfficiencyCalibration
from fluxforge.plugins import bootstrap_builtin_registries

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class EfficiencyCalibrationDialog(QDialog):
        """Fit and apply registered detector-efficiency models."""

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
            self.resize(1120, 760)

            registries = bootstrap_builtin_registries()
            if len(registries.calibration_models) == 0:
                register_builtin_efficiency_models(registries)
            self._registry = registries.calibration_models
            self._fit_result: EfficiencyCalibrationFitResult | None = None
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

            detector_group = QGroupBox("HPGe detector parameters", self)
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
                detector_layout.addWidget(field, row, column + 1)
                self.detector_fields[attribute] = field
            root.addWidget(detector_group)

            self.table = QTableWidget(0, len(self.HEADERS), self)
            self.table.setObjectName("EfficiencyCalibrationTable")
            self.table.setHorizontalHeaderLabels(self.HEADERS)
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setStretchLastSection(False)
            root.addWidget(self.table, 1)

            seed_button = QPushButton("Seed demo points", self)
            seed_button.setObjectName("SeedEfficiencyExamplePointsButton")
            seed_button.clicked.connect(self._seed_demo_points)
            root.addWidget(seed_button)

            self.summary = QLabel("Waiting for a fit.", self)
            self.summary.setObjectName("PanelBody")
            self.summary.setWordWrap(True)
            root.addWidget(self.summary)

            buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                parent=self,
            )
            buttons.button(QDialogButtonBox.Ok).setObjectName(
                "AcceptEfficiencyCalibrationButton"
            )
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
                    )
                ):
                    self.table.setItem(
                        row, column, QTableWidgetItem(f"{float(value):.6g}")
                    )
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
                    points.append(
                        EfficiencyPoint(
                            energy_keV=float(self.table.item(row, 0).text()),
                            net_counts=float(self.table.item(row, 1).text()),
                            count_uncertainty=max(
                                float(self.table.item(row, 2).text()), 0.0
                            ),
                            live_time_s=float(self.table.item(row, 3).text()),
                            activity_bq=float(self.table.item(row, 4).text()),
                            activity_rel_unc=max(
                                float(self.table.item(row, 5).text()), 0.0
                            ),
                            emission_probability=float(self.table.item(row, 6).text()),
                            probability_uncertainty=max(
                                float(self.table.item(row, 7).text()), 0.0
                            ),
                            geometry_factor=max(
                                float(self.table.item(row, 8).text()), 1.0e-12
                            ),
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
            return tuple(points)

        def _fit_model(self) -> None:
            points = self._read_points()
            if not points:
                self.summary.setText("No valid efficiency points are available.")
                return
            self._fit_result = fit_efficiency_model(
                points,
                model_key=self.method_selector.current_key()
                or self._registry.default_key
                or "log_poly_2",
            )
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
            self._fit_result.curve.parameters["detector_calibration"] = asdict(detector)
            self.summary.setText(
                (
                    f"{self._fit_result.model_label} fit complete. "
                    f"RMSE = {self._fit_result.rmse:.6f} using {self._fit_result.points_used} points."
                )
            )

        def accepted_fit(self) -> EfficiencyCalibrationFitResult | None:
            return self._fit_result

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

        def detector_calibration(self) -> EfficiencyCalibration:
            return self._detector_calibration
