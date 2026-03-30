"""ASTM C1030 plutonium isotopics wizard."""

from __future__ import annotations

from math import sqrt

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.standards import (
    C1030Module,
    PuLineObservation,
    StandardsEvaluationContext,
    compute_pu_isotopics,
)

if QT_AVAILABLE:  # pragma: no cover - optional GUI branch
    from fluxforge.gui.qt_compat import (
        QCheckBox,
        QDialog,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )


def _build_observations_from_peaks(peaks) -> tuple[PuLineObservation, ...]:
    observations: list[PuLineObservation] = []
    for peak in peaks:
        if peak.nuclide not in {"Pu-239", "Pu-240", "Pu-241", "Am-241"}:
            continue
        observations.append(
            PuLineObservation(
                nuclide=peak.nuclide,
                energy_keV=float(peak.energy_keV),
                net_counts=float(peak.net_counts),
                efficiency=0.1,
                uncertainty_counts=sqrt(max(float(peak.net_counts), 0.0)),
            )
        )
    if observations:
        return tuple(observations)
    return (
        PuLineObservation("Pu-239", 129.3, 4200.0, 0.12, 64.8),
        PuLineObservation("Pu-240", 160.3, 520.0, 0.11, 22.8),
        PuLineObservation("Pu-241", 148.6, 330.0, 0.10, 18.1),
        PuLineObservation("Am-241", 59.5, 210.0, 0.16, 14.5),
    )


if QT_AVAILABLE:  # pragma: no cover - optional GUI branch

    class PuIsotopicsDialog(QDialog):
        """Four-step C1030 wizard with isotopic ratios and report summary."""

        def __init__(
            self,
            *,
            peaks,
            efficiency_uncertainty_pct: float = 2.5,
            fwhm_at_413_keV: float = 1.1,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("FluxForge Next - Pu Isotopics Wizard")
            self.resize(980, 760)
            self._observations = _build_observations_from_peaks(peaks)
            self._efficiency_uncertainty_pct = efficiency_uncertainty_pct
            self._fwhm_at_413_keV = fwhm_at_413_keV
            self._module = C1030Module()

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QLabel("ASTM C1030 Pu Isotopics Wizard", self)
            title.setObjectName("PanelHeading")
            root.addWidget(title)

            self.tabs = QTabWidget(self)
            self.tabs.setObjectName("PuIsotopicsWizardTabs")
            self.tabs.addTab(self._build_line_review_tab(), "1. Peak Review")
            self.tabs.addTab(self._build_age_tab(), "2. Source Age")
            self.tabs.addTab(self._build_ratio_tab(), "3. Ratios")
            self.tabs.addTab(self._build_report_tab(), "4. Report")
            root.addWidget(self.tabs, 1)

            action_row = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh", self)
            self.refresh_button.setObjectName("PuIsotopicsRefreshButton")
            self.refresh_button.clicked.connect(self.refresh)
            action_row.addWidget(self.refresh_button)
            action_row.addStretch(1)
            root.addLayout(action_row)

            self.refresh()

        def _build_line_review_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            self.line_table = QTableWidget(0, 4, widget)
            self.line_table.setObjectName("PuLineObservationTable")
            self.line_table.setHorizontalHeaderLabels(
                ("Nuclide", "Energy", "Net Counts", "σ Counts")
            )
            layout.addWidget(self.line_table)
            return widget

        def _build_age_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            self.estimate_age_checkbox = QCheckBox(
                "Estimate from Am-241 / Pu-241 ratio",
                widget,
            )
            self.estimate_age_checkbox.setObjectName("PuEstimateAgeCheck")
            self.estimate_age_checkbox.setChecked(True)
            self.estimate_age_checkbox.toggled.connect(self.refresh)
            layout.addWidget(self.estimate_age_checkbox)
            self.age_spin = QDoubleSpinBox(widget)
            self.age_spin.setObjectName("PuSourceAgeYearsSpin")
            self.age_spin.setRange(0.0, 100.0)
            self.age_spin.setValue(5.0)
            self.age_spin.valueChanged.connect(self.refresh)
            layout.addWidget(self.age_spin)
            return widget

        def _build_ratio_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            self.ratio_table = QTableWidget(0, 2, widget)
            self.ratio_table.setObjectName("PuRatioTable")
            self.ratio_table.setHorizontalHeaderLabels(("Metric", "Value"))
            layout.addWidget(self.ratio_table)
            return widget

        def _build_report_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            self.report_browser = QTextBrowser(widget)
            self.report_browser.setObjectName("PuIsotopicsReportBrowser")
            layout.addWidget(self.report_browser)
            return widget

        def refresh(self) -> None:
            for row, observation in enumerate(self._observations):
                self.line_table.setRowCount(len(self._observations))
                values = (
                    observation.nuclide,
                    f"{observation.energy_keV:.2f} keV",
                    f"{observation.net_counts:.1f}",
                    f"{observation.uncertainty_counts:.1f}",
                )
                for column, value in enumerate(values):
                    self.line_table.setItem(row, column, QTableWidgetItem(value))

            source_age = None if self.estimate_age_checkbox.isChecked() else self.age_spin.value()
            result = compute_pu_isotopics(self._observations, source_age_years=source_age)

            ratio_rows = (
                ("Pu-240 / Pu-239", f"{result.pu240_to_pu239:.4f}"),
                ("Pu-241 / Pu-239", f"{result.pu241_to_pu239:.4f}"),
                ("Am-241 / Pu-239", f"{result.am241_to_pu239:.4f}"),
                ("Age (years)", "estimated" if source_age is None else f"{result.age_years:.2f}"),
                ("Classification", result.classification),
                ("Ratio uncertainty", f"{result.ratio_uncertainty:.4f}"),
            )
            self.ratio_table.setRowCount(len(ratio_rows))
            for row, (label, value) in enumerate(ratio_rows):
                self.ratio_table.setItem(row, 0, QTableWidgetItem(label))
                self.ratio_table.setItem(row, 1, QTableWidgetItem(value))

            evaluation = self._module.evaluate(
                context=StandardsEvaluationContext(
                    efficiency_uncertainty_pct=self._efficiency_uncertainty_pct,
                    fwhm_at_413_keV=self._fwhm_at_413_keV,
                    net_counts={
                        "Pu-240 160.3": next(
                            (
                                obs.net_counts
                                for obs in self._observations
                                if obs.nuclide == "Pu-240"
                            ),
                            0.0,
                        )
                    },
                    line_observations={"c1030": self._observations},
                    extra={
                        "source_age_years": (
                            None
                            if self.estimate_age_checkbox.isChecked()
                            else self.age_spin.value()
                        )
                    },
                )
            )
            lines = [
                "<h2>ASTM C1030 Report</h2>",
                f"<p><strong>Classification:</strong> {result.classification}</p>",
                f"<p><strong>Estimated age:</strong> {result.age_years:.2f} years</p>" if result.age_years is not None else "<p><strong>Estimated age:</strong> unavailable</p>",
                "<ul>",
            ]
            for check in evaluation.checks:
                lines.append(
                    f"<li><strong>{check.label}</strong>: {check.status} — {check.message} ({check.value})</li>"
                )
            lines.append("</ul>")
            self.report_browser.setHtml("".join(lines))


else:

    class PuIsotopicsDialog:  # pragma: no cover - import-safe fallback without Qt
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PuIsotopicsDialog requires PySide6")
