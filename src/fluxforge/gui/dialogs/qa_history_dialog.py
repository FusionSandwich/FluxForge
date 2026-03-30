"""QA history review dialog for Module 3."""

from __future__ import annotations

from collections import defaultdict

from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.standards import QAMonitor

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # pragma: no cover - optional GUI branch
    import pyqtgraph as pg

    from fluxforge.gui.qt_compat import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )


if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # pragma: no cover - optional GUI branch

    class QAHistoryDialog(QDialog):
        """Stacked QA plots with history export and drift summary."""

        def __init__(self, qa_monitor: QAMonitor, parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle("FluxForge Next - QA History")
            self.resize(1180, 900)
            self.qa_monitor = qa_monitor

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QLabel("ASTM QA History", self)
            title.setObjectName("PanelHeading")
            root.addWidget(title)

            self.summary = QPlainTextEdit(self)
            self.summary.setObjectName("QaHistorySummary")
            self.summary.setReadOnly(True)
            root.addWidget(self.summary)

            self.centroid_plot = pg.PlotWidget(self)
            self.centroid_plot.setObjectName("QaCentroidPlot")
            self.centroid_plot.setBackground("#0f172a")
            self.centroid_plot.setLabel("left", "Centroid (keV)")
            self.centroid_plot.setLabel("bottom", "History Index")
            root.addWidget(self.centroid_plot, 1)

            self.fwhm_plot = pg.PlotWidget(self)
            self.fwhm_plot.setObjectName("QaFwhmPlot")
            self.fwhm_plot.setBackground("#0f172a")
            self.fwhm_plot.setLabel("left", "FWHM (keV)")
            self.fwhm_plot.setLabel("bottom", "History Index")
            root.addWidget(self.fwhm_plot, 1)

            self.efficiency_plot = pg.PlotWidget(self)
            self.efficiency_plot.setObjectName("QaEfficiencyPlot")
            self.efficiency_plot.setBackground("#0f172a")
            self.efficiency_plot.setLabel("left", "Efficiency")
            self.efficiency_plot.setLabel("bottom", "History Index")
            root.addWidget(self.efficiency_plot, 1)

            self.table = QTableWidget(0, 6, self)
            self.table.setObjectName("QaHistoryTable")
            self.table.setHorizontalHeaderLabels(
                ("Nuclide", "Energy", "Drift", "FWHM Δ%", "Eff Δ%", "Status")
            )
            root.addWidget(self.table)

            self.export_path = QPlainTextEdit(self)
            self.export_path.setObjectName("QaExportPathInput")
            self.export_path.setMaximumHeight(52)
            self.export_path.setPlainText("artifacts/qa_history.csv")
            root.addWidget(self.export_path)

            action_row = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh", self)
            self.refresh_button.setObjectName("QaRefreshButton")
            self.refresh_button.clicked.connect(self.refresh)
            action_row.addWidget(self.refresh_button)
            self.export_button = QPushButton("Export CSV", self)
            self.export_button.setObjectName("QaExportButton")
            self.export_button.clicked.connect(self._export_csv)
            action_row.addWidget(self.export_button)
            action_row.addStretch(1)
            root.addLayout(action_row)

            self.last_export_path = None
            self.refresh()

        def refresh(self) -> None:
            history = self.qa_monitor.grouped_history()
            statuses = self.qa_monitor.status_snapshot()
            self.table.setRowCount(len(statuses))
            self.centroid_plot.clear()
            self.fwhm_plot.clear()
            self.efficiency_plot.clear()

            color_map = {"green": "#22c55e", "amber": "#f59e0b", "red": "#ef4444"}
            summary_lines = []
            for row, status in enumerate(statuses):
                values = (
                    status.nuclide,
                    f"{status.energy_keV:.2f} keV",
                    f"{status.centroid_drift_keV:+.3f} keV",
                    f"{status.fwhm_degradation_pct:+.2f}%",
                    f"{status.efficiency_deviation_pct:+.2f}%",
                    status.status,
                )
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
                summary_lines.append(
                    f"{status.nuclide} {status.energy_keV:.2f} keV | "
                    f"drift {status.centroid_drift_keV:+.3f} keV | "
                    f"FWHM {status.fwhm_degradation_pct:+.2f}% | "
                    f"eff {status.efficiency_deviation_pct:+.2f}% | {status.status}"
                )

            for (nuclide, energy), series in history.items():
                indices = list(range(len(series)))
                label = f"{nuclide} {energy:.1f} keV"
                color = color_map.get(
                    next(
                        (status.status for status in statuses if status.nuclide == nuclide and status.energy_keV == energy),
                        "green",
                    ),
                    "#22c55e",
                )
                pen = pg.mkPen(color=color, width=2)
                brush = pg.mkBrush(color)
                self.centroid_plot.plot(
                    indices,
                    [record.measured_centroid_keV for record in series],
                    pen=pen,
                    symbol="d",
                    symbolBrush=brush,
                    name=label,
                )
                self.fwhm_plot.plot(
                    indices,
                    [record.measured_fwhm_keV for record in series],
                    pen=pen,
                    symbol="d",
                    symbolBrush=brush,
                    name=label,
                )
                self.efficiency_plot.plot(
                    indices,
                    [record.efficiency for record in series],
                    pen=pen,
                    symbol="d",
                    symbolBrush=brush,
                    name=label,
                )

            self.summary.setPlainText("\n".join(summary_lines) or "No QA history recorded.")

        def _export_csv(self) -> None:
            target = self.export_path.toPlainText().strip()
            if not target:
                return
            path = self.qa_monitor.db_path.parent / target if not target.startswith("/") else target
            aggregate = ["timestamp,nuclide,energy_keV,centroid_keV,fwhm_keV,efficiency,status"]
            status_lookup = {
                (status.nuclide, status.energy_keV): status.status
                for status in self.qa_monitor.status_snapshot()
            }
            for record in self.qa_monitor.history():
                aggregate.append(
                    ",".join(
                        (
                            record.timestamp.isoformat(),
                            record.nuclide,
                            f"{record.energy_keV:.3f}",
                            f"{record.measured_centroid_keV:.3f}",
                            f"{record.measured_fwhm_keV:.3f}",
                            f"{record.efficiency:.6f}",
                            status_lookup.get((record.nuclide, record.energy_keV), "green"),
                        )
                    )
                )
            from pathlib import Path

            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("\n".join(aggregate), encoding="utf-8")
            self.last_export_path = resolved


else:

    class QAHistoryDialog:  # pragma: no cover - import-safe fallback without Qt
        def __init__(self, qa_monitor: QAMonitor, parent=None) -> None:
            raise RuntimeError("QAHistoryDialog requires PySide6 and pyqtgraph")
