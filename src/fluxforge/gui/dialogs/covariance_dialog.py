"""Covariance and correlation heatmaps for declared analytical artifacts."""

from pathlib import Path

import numpy as np

from fluxforge.gui.covariance_view import prepare_covariance_view, read_covariance_view
from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:
    from matplotlib import colormaps
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )
    from PySide6.QtCore import Qt

    class CovarianceDialog(QDialog):
        """Load a matrix, inspect exact values and export the plotted view."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Covariance and Correlation")
            self.resize(960, 700)
            self.current_view = None
            layout = QVBoxLayout(self)
            controls = QHBoxLayout()
            self.load_button = QPushButton("Load covariance JSON…", self)
            self.load_button.setObjectName("CovarianceLoadButton")
            self.load_button.clicked.connect(self._choose_file)
            controls.addWidget(self.load_button)
            self.view_combo = QComboBox(self)
            self.view_combo.setObjectName("CovarianceViewCombo")
            self.view_combo.addItems(["Correlation", "Covariance"])
            self.view_combo.currentIndexChanged.connect(self._render)
            controls.addWidget(self.view_combo)
            self.export_button = QPushButton("Export PNG…", self)
            self.export_button.setObjectName("CovarianceExportButton")
            self.export_button.clicked.connect(self._choose_export)
            self.export_button.setEnabled(False)
            controls.addWidget(self.export_button)
            controls.addStretch(1)
            layout.addLayout(controls)
            self.summary_label = QLabel(
                "Load a declared covariance matrix to inspect its correlations.", self
            )
            self.summary_label.setWordWrap(True)
            layout.addWidget(self.summary_label)
            self.figure = Figure(layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.table = QTableWidget(self)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            splitter = QSplitter(Qt.Vertical, self)
            splitter.addWidget(self.canvas)
            splitter.addWidget(self.table)
            splitter.setSizes([450, 180])
            layout.addWidget(splitter)
            note = QLabel(
                "N/A means correlation is undefined because variance is zero. Displaying a matrix does not qualify its scientific source.",
                self,
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        def set_covariance(
            self, covariance, labels=None, *, title="Covariance", source=""
        ):
            self.current_view = prepare_covariance_view(
                covariance, labels, title=title, source=source
            )
            self._render()

        def load_file(self, path):
            view = read_covariance_view(path)
            self.current_view = view
            self._render()

        def _choose_file(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Load covariance", "", "JSON artifacts (*.json)"
            )
            if path:
                try:
                    self.load_file(path)
                except (ValueError, TypeError, OSError) as exc:
                    QMessageBox.warning(self, "Cannot load covariance", str(exc))

        def _render(self, *_args):
            view = self.current_view
            if view is None:
                return
            is_correlation = self.view_combo.currentIndex() == 0
            values = view.correlation if is_correlation else view.covariance
            limit = 1.0 if is_correlation else float(np.max(np.abs(values))) or 1.0
            cmap = colormaps["coolwarm"].with_extremes(bad="#b0b0b0")
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            plot = axes.imshow(
                np.ma.masked_invalid(values),
                vmin=-limit,
                vmax=limit,
                cmap=cmap,
                interpolation="nearest",
            )
            ticks = np.unique(
                np.linspace(0, len(values) - 1, min(len(values), 20)).astype(int)
            )
            axes.set_xticks(
                ticks, [view.labels[i] for i in ticks], rotation=45, ha="right"
            )
            axes.set_yticks(ticks, [view.labels[i] for i in ticks])
            kind = "Correlation" if is_correlation else "Covariance"
            axes.set_title(f"{view.title} — {kind}")
            self.figure.colorbar(plot, ax=axes, label=kind)
            self.canvas.draw_idle()
            self.table.setRowCount(len(values))
            self.table.setColumnCount(len(values))
            self.table.setHorizontalHeaderLabels(view.labels)
            self.table.setVerticalHeaderLabels(view.labels)
            for i in range(len(values)):
                for j in range(len(values)):
                    value = values[i, j]
                    item = QTableWidgetItem(
                        f"{value:.6g}" if np.isfinite(value) else "N/A"
                    )
                    item.setToolTip(f"{view.labels[i]} × {view.labels[j]}: {value!r}")
                    rgba = (
                        cmap((value / limit + 1) / 2)
                        if np.isfinite(value)
                        else (0.69, 0.69, 0.69, 1)
                    )
                    item.setBackground(QColor.fromRgbF(*rgba))
                    luminance = sum(
                        w * c for w, c in zip((0.2126, 0.7152, 0.0722), rgba[:3])
                    )
                    item.setForeground(QColor("white" if luminance < 0.4 else "black"))
                    self.table.setItem(i, j, item)
            self.summary_label.setText(
                f"{view.title}: {len(values)} × {len(values)}. {view.source}"
            )
            self.export_button.setEnabled(True)

        def export_png(self, path):
            if self.current_view is None:
                raise ValueError("Load a covariance matrix before exporting")
            self.figure.savefig(Path(path), dpi=150, format="png")

        def _choose_export(self):
            path, _ = QFileDialog.getSaveFileName(
                self, "Export heatmap", "covariance.png", "PNG (*.png)"
            )
            if path:
                try:
                    self.export_png(path)
                except (ValueError, OSError) as exc:
                    QMessageBox.warning(self, "Cannot export heatmap", str(exc))

else:

    class CovarianceDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Covariance inspection requires the native GUI extras")
