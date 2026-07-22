"""Standards review dialog for ASTM-oriented checks in the modern Qt shell."""

from __future__ import annotations

from collections.abc import Callable

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.standards.base import StandardsEvaluation

if QT_AVAILABLE:  # pragma: no cover - optional GUI branch
    from fluxforge.gui.qt_compat import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional GUI branch

    class StandardsReviewDialog(QDialog):
        """Review ASTM standards evaluations against the current workspace."""

        def __init__(
            self,
            *,
            evaluation_factory: Callable[[], tuple[StandardsEvaluation, ...]],
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("FluxForge — ASTM Standards Review")
            self.resize(1180, 880)
            self.evaluation_factory = evaluation_factory
            self.evaluations: tuple[StandardsEvaluation, ...] = ()

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QLabel("ASTM Standards Review", self)
            title.setObjectName("PanelHeading")
            root.addWidget(title)

            self.summary = QTextBrowser(self)
            self.summary.setObjectName("StandardsReviewSummary")
            root.addWidget(self.summary)

            self.table = QTableWidget(0, 4, self)
            self.table.setObjectName("StandardsReviewTable")
            self.table.setHorizontalHeaderLabels(
                ("Standard", "Status", "Locked Settings", "Summary")
            )
            root.addWidget(self.table, 1)

            self.detail = QTextBrowser(self)
            self.detail.setObjectName("StandardsReviewDetail")
            root.addWidget(self.detail, 1)

            actions = QHBoxLayout()
            self.refresh_button = QLabel("Refresh from current workspace state", self)
            self.refresh_button.setObjectName("HeroCardAccent")
            actions.addWidget(self.refresh_button)
            actions.addStretch(1)
            root.addLayout(actions)

            self.refresh()

        def refresh(self) -> None:
            self.evaluations = tuple(self.evaluation_factory())
            self.table.setRowCount(len(self.evaluations))
            detail_sections = []
            summary_bits = []
            for row, evaluation in enumerate(self.evaluations):
                locks = (
                    ", ".join(setting.field_id for setting in evaluation.locked_settings)
                    or "none"
                )
                values = (
                    evaluation.display_name,
                    evaluation.overall_status,
                    locks,
                    evaluation.summary,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    self.table.setItem(row, column, item)
                summary_bits.append(
                    f"{evaluation.display_name}: {evaluation.overall_status.upper()}"
                )
                lines = [
                    f"<h3>{evaluation.display_name}</h3>",
                    f"<p><strong>Status:</strong> {evaluation.overall_status}</p>",
                    f"<p><strong>Summary:</strong> {evaluation.summary}</p>",
                ]
                if evaluation.locked_settings:
                    lines.append("<ul>")
                    for setting in evaluation.locked_settings:
                        lines.append(
                            f"<li>[locked] {setting.field_id}: {setting.value} ({setting.standard_section})</li>"
                        )
                    lines.append("</ul>")
                if evaluation.checks:
                    lines.append("<table><tr><th>Check</th><th>Status</th><th>Value</th><th>Limit</th></tr>")
                    for check in evaluation.checks:
                        lines.append(
                            f"<tr><td>{check.label}</td><td>{check.status}</td><td>{check.value}</td><td>{check.limit}</td></tr>"
                        )
                    lines.append("</table>")
                detail_sections.append("".join(lines))

            self.summary.setHtml(
                "<p><strong>Current ASTM sweep:</strong> " + " | ".join(summary_bits) + "</p>"
            )
            self.detail.setHtml("".join(detail_sections))


else:

    class StandardsReviewDialog:  # pragma: no cover - import-safe fallback without Qt
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("StandardsReviewDialog requires PySide6")
