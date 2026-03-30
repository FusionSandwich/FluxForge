"""Auto peak review dialog for the modern Phase 2 workspace."""

from __future__ import annotations

from collections.abc import Sequence

from fluxforge.core.phase2_analysis import PeakCandidate
from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QDialog,
        QDialogButtonBox,
        QHeaderView,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        Qt,
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class AutoPeakReviewDialog(QDialog):
        """Review auto-detected peaks before they are committed to the peak table."""

        HEADERS = ("Use", "Energy keV", "Significance", "Net Counts", "ROI keV")

        def __init__(self, peaks: Sequence[PeakCandidate], parent=None) -> None:
            super().__init__(parent)
            self._peaks = tuple(peaks)
            self.setWindowTitle("Auto Peak Review")
            self.resize(760, 460)

            root = QVBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(10)

            intro = QLabel(
                (
                    "Review detected peaks before they are added to the modern Qt peak "
                    "table. Uncheck any peaks you do not want to keep."
                ),
                self,
            )
            intro.setWordWrap(True)
            intro.setObjectName("PanelBody")
            root.addWidget(intro)

            self.table = QTableWidget(len(self._peaks), len(self.HEADERS), self)
            self.table.setObjectName("AutoPeakReviewTable")
            self.table.setHorizontalHeaderLabels(self.HEADERS)
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Stretch)
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            root.addWidget(self.table, 1)

            buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                parent=self,
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

            self._populate_rows()

        def _populate_rows(self) -> None:
            for row, peak in enumerate(self._peaks):
                check_item = QTableWidgetItem("")
                check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                check_item.setCheckState(Qt.Checked)
                self.table.setItem(row, 0, check_item)

                self.table.setItem(row, 1, QTableWidgetItem(f"{peak.energy_keV:.3f}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{peak.significance:.2f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{peak.net_counts:.1f}"))
                self.table.setItem(
                    row,
                    4,
                    QTableWidgetItem(
                        f"{peak.roi_bounds_keV[0]:.2f} - {peak.roi_bounds_keV[1]:.2f}"
                    ),
                )

        def accepted_peaks(self) -> tuple[PeakCandidate, ...]:
            peaks: list[PeakCandidate] = []
            for row, peak in enumerate(self._peaks):
                item = self.table.item(row, 0)
                if item is not None and item.checkState() == Qt.Checked:
                    peaks.append(peak)
            return tuple(peaks)


else:

    class AutoPeakReviewDialog:  # pragma: no cover - placeholder without Qt
        def __init__(self, peaks: Sequence[PeakCandidate], parent=None) -> None:
            self._peaks = tuple(peaks)
            self.parent = parent

        def accepted_peaks(self) -> tuple[PeakCandidate, ...]:
            return self._peaks
