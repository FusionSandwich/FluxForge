"""Efficiency calibration dialog for the modern Phase 2 workspace."""

from __future__ import annotations

import math
from collections.abc import Sequence

from fluxforge.analysis.detector_calibration import EfficiencyPoint
from fluxforge.core.phase2_analysis import (
    EfficiencyCalibrationFitResult,
    register_builtin_efficiency_models,
    fit_efficiency_model,
)
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.widgets import MethodSelectorWidget
from fluxforge.plugins import bootstrap_builtin_registries

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QAbstractItemView,
        QDialog,
        QDialogButtonBox,
        QHeaderView,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class EfficiencyCalibrationDialog(QDialog):
        """Fit and apply Phase 2 efficiency models from the modern Qt stack."""

        HEADERS = (
            "Energy keV",
            "Net Counts",
            "Live Time s",
            "Activity Bq",
            "Gamma Intensity",
        )

        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            points: Sequence[EfficiencyPoint] | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Efficiency Calibration")
            self.resize(860, 520)

            registries = bootstrap_builtin_registries()
            if len(registries.calibration_models) == 0:
                register_builtin_efficiency_models(registries)
            self._registry = registries.calibration_models
            self._fit_result: EfficiencyCalibrationFitResult | None = None

            root = QVBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(10)

            intro = QLabel(
                (
                    "Fit one of the registered efficiency models. The resulting curve "
                    "is used by the Phase 2 activity workflow."
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

            self.table = QTableWidget(0, len(self.HEADERS), self)
            self.table.setObjectName("EfficiencyCalibrationTable")
            self.table.setHorizontalHeaderLabels(self.HEADERS)
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Stretch)
            root.addWidget(self.table, 1)

            seed_button = QPushButton("Seed demo points", self)
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
            fit_button = buttons.addButton("Fit Model", QDialogButtonBox.ActionRole)
            fit_button.clicked.connect(self._fit_model)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

            self._seed_points(points or self._demo_points())

        def _seed_points(self, points: Sequence[EfficiencyPoint]) -> None:
            self.table.setRowCount(0)
            for point in points:
                row = self.table.rowCount()
                self.table.insertRow(row)
                for column, value in enumerate(
                    (
                        point.energy_keV,
                        point.net_counts,
                        point.live_time_s,
                        point.activity_bq,
                        point.emission_probability,
                    )
                ):
                    self.table.setItem(row, column, QTableWidgetItem(f"{float(value):.6g}"))

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
                            live_time_s=float(self.table.item(row, 2).text()),
                            activity_bq=float(self.table.item(row, 3).text()),
                            emission_probability=float(self.table.item(row, 4).text()),
                            count_uncertainty=max(
                                math.sqrt(abs(float(self.table.item(row, 1).text()))),
                                1.0,
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
                model_key=self.method_selector.current_key() or self._registry.default_key or "log_poly_2",
            )
            self.summary.setText(
                (
                    f"{self._fit_result.model_label} fit complete. "
                    f"RMSE = {self._fit_result.rmse:.6f} using {self._fit_result.points_used} points."
                )
            )

        def accepted_fit(self) -> EfficiencyCalibrationFitResult | None:
            return self._fit_result


else:

    class EfficiencyCalibrationDialog:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            *,
            mode_manager: ModeManager,
            points: Sequence[EfficiencyPoint] | None = None,
            parent=None,
        ) -> None:
            self.mode_manager = mode_manager
            self.points = tuple(points or ())
            self.parent = parent

        def accepted_fit(self) -> EfficiencyCalibrationFitResult | None:
            return None
