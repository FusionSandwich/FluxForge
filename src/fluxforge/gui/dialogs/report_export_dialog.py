"""Report export dialog for the modern Qt shell."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.reporting.engine import ReportingEngine

if QT_AVAILABLE:  # pragma: no cover - optional GUI branch
    from fluxforge.gui.qt_compat import (
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional GUI branch

    class ReportExportDialog(QDialog):
        """Preview and export bundled Jinja2 reports."""

        def __init__(
            self,
            *,
            engine: ReportingEngine | None = None,
            context_factory: Callable[[str], dict[str, object]] | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("FluxForge — Export Report")
            self.resize(1080, 840)
            self.engine = engine or ReportingEngine()
            self.context_factory = context_factory or (lambda _name: {})
            self.last_export_path = None
            self.last_pdf_export_path = None

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QLabel("Report Export", self)
            title.setObjectName("PanelHeading")
            root.addWidget(title)

            controls = QHBoxLayout()
            self.template_combo = QComboBox(self)
            self.template_combo.setObjectName("ReportTemplateCombo")
            for template in self.engine.bundled_templates():
                self.template_combo.addItem(template.name, template.name)
            controls.addWidget(self.template_combo, 1)

            self.path_input = QLineEdit(self)
            self.path_input.setObjectName("ReportExportPathInput")
            self.path_input.setText("artifacts/reports/fluxforge_report.html")
            controls.addWidget(self.path_input, 2)

            self.render_button = QPushButton("Render Preview", self)
            self.render_button.setObjectName("RenderReportPreviewButton")
            self.render_button.clicked.connect(self.render_preview)
            controls.addWidget(self.render_button)

            self.export_button = QPushButton("Export HTML", self)
            self.export_button.setObjectName("ExportReportHtmlButton")
            self.export_button.clicked.connect(self.export_html)
            controls.addWidget(self.export_button)

            self.pdf_button = QPushButton("Export PDF", self)
            self.pdf_button.setObjectName("ExportReportPdfButton")
            self.pdf_button.clicked.connect(self.export_pdf)
            controls.addWidget(self.pdf_button)
            root.addLayout(controls)

            self.pdf_status = QLabel(self)
            self.pdf_status.setObjectName("ReportPdfStatus")
            self.pdf_status.setWordWrap(True)
            root.addWidget(self.pdf_status)

            self.preview = QTextBrowser(self)
            self.preview.setObjectName("ReportPreviewBrowser")
            root.addWidget(self.preview, 1)

            self._sync_template_status()
            self._sync_pdf_status()
            if self.engine.template_backend_available():
                self.render_preview()
            else:
                self.preview.setPlainText(
                    "HTML report rendering requires the optional reporting extra (`Jinja2`)."
                )

        def current_template_name(self) -> str:
            data = self.template_combo.currentData()
            return str(data) if data is not None else str(self.template_combo.currentText())

        def render_preview(self) -> None:
            if not self.engine.template_backend_available():
                self.preview.setPlainText(
                    "HTML report rendering requires the optional reporting extra (`Jinja2`)."
                )
                return
            template_name = self.current_template_name()
            context = self.context_factory(template_name)
            rendered = self.engine.render(template_name, context)
            self.preview.setHtml(rendered.html)

        def export_html(self) -> None:
            if not self.engine.template_backend_available():
                return
            template_name = self.current_template_name()
            context = self.context_factory(template_name)
            path = Path(self.path_input.text().strip())
            if not path.is_absolute():
                path = Path.cwd() / path
            self.last_export_path = self.engine.export_html(template_name, context, path)
            self.render_preview()

        def export_pdf(self) -> None:
            if not self.engine.template_backend_available():
                return
            template_name = self.current_template_name()
            context = self.context_factory(template_name)
            path = Path(self.path_input.text().strip())
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.suffix.lower() != ".pdf":
                path = path.with_suffix(".pdf")
            self.last_pdf_export_path = self.engine.export_pdf(
                template_name, context, path
            )
            self.render_preview()

        def _sync_template_status(self) -> None:
            available = self.engine.template_backend_available()
            self.template_combo.setEnabled(available)
            self.path_input.setEnabled(available)
            self.render_button.setEnabled(available)
            self.export_button.setEnabled(available)

        def _sync_pdf_status(self) -> None:
            available = self.engine.can_export_pdf()
            self.pdf_button.setEnabled(available)
            if not self.engine.template_backend_available():
                self.pdf_status.setText(
                    "HTML/PDF report export requires the optional reporting extra (`Jinja2`)."
                )
            elif available:
                self.pdf_status.setText(
                    "PDF export is available through the installed WeasyPrint backend."
                )
            else:
                self.pdf_status.setText(
                    "PDF export requires the optional reporting extra (`WeasyPrint`)."
                )


else:

    class ReportExportDialog:  # pragma: no cover - import-safe fallback without Qt
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("ReportExportDialog requires PySide6")
