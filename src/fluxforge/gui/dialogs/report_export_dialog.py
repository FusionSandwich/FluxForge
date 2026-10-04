"""Report export dialog for the modern Qt shell."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.reporting.engine import ReportingEngine

if QT_AVAILABLE:  # pragma: no cover - optional GUI branch
    from fluxforge.gui.qt_compat import (
        QComboBox,
        QCheckBox,
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
            self.setObjectName("ReportExportDialog")
            self.setWindowTitle("FluxForge — Export Report")
            self.resize(1080, 840)
            self.engine = engine or ReportingEngine()
            self.context_factory = context_factory or (lambda _name: {})
            self.last_export_path = None
            self.last_pdf_export_path = None
            self.last_bundle_path = None
            self.last_context = None

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

            bundle_controls = QHBoxLayout()
            self.bundle_pdf_check = QCheckBox("Include PDF in run bundle", self)
            self.bundle_pdf_check.setObjectName("ReportBundlePdfCheck")
            bundle_controls.addWidget(self.bundle_pdf_check)
            bundle_controls.addStretch(1)
            self.generate_button = QPushButton("Generate Report", self)
            self.generate_button.setObjectName("GenerateReportBundleButton")
            self.generate_button.setToolTip(
                "Capture the current Qt tables, visible plots and inputs into a ZIP "
                "beside the export path. Existing bundles are preserved."
            )
            self.generate_button.clicked.connect(self.generate_report)
            bundle_controls.addWidget(self.generate_button)
            root.addLayout(bundle_controls)

            self.export_status = QLabel(self)
            self.export_status.setObjectName("ReportExportStatus")
            self.export_status.setWordWrap(True)
            root.addWidget(self.export_status)

            self.pdf_status = QLabel(self)
            self.pdf_status.setObjectName("ReportPdfStatus")
            self.pdf_status.setWordWrap(True)
            root.addWidget(self.pdf_status)

            self.preview = QTextBrowser(self)
            self.preview.setObjectName("ReportPreviewBrowser")
            self.preview.setStyleSheet("background-color: white; color: #10253c;")
            root.addWidget(self.preview, 1)

            self._sync_template_status()
            self._sync_pdf_status()
            if self.engine.template_backend_available():
                self.render_preview()
            else:
                self.preview.setPlainText(
                    "HTML report rendering requires the optional reporting extra "
                    "(`Jinja2`)."
                )

        def current_template_name(self) -> str:
            data = self.template_combo.currentData()
            return (
                str(data)
                if data is not None
                else str(self.template_combo.currentText())
            )

        def render_preview(self) -> None:
            if not self.engine.template_backend_available():
                self.preview.setPlainText(
                    "HTML report rendering requires the optional reporting extra "
                    "(`Jinja2`)."
                )
                return
            try:
                self._show_context(self._capture_context())
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.export_status.setText(f"Report capture failed: {exc}")

        def _capture_context(self) -> dict[str, object]:
            return deepcopy(self.context_factory(self.current_template_name()))

        def _show_context(self, context: dict[str, object]) -> None:
            rendered = self.engine.render(self.current_template_name(), context)
            self.last_context = context
            self.preview.setHtml(rendered.html)

        def _output_path(self, suffix: str | None = None) -> Path:
            text = self.path_input.text().strip()
            if not text:
                raise ValueError("Enter an export file path.")
            path = Path(text)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path.with_suffix(suffix) if suffix is not None else path

        def export_html(self) -> None:
            if not self.engine.template_backend_available():
                return
            self.last_export_path = None
            try:
                context = self._capture_context()
                self.last_export_path = self.engine.export_html(
                    self.current_template_name(), context, self._output_path()
                )
                self._show_context(context)
                self.export_status.setText(f"Exported HTML: {self.last_export_path}")
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.export_status.setText(f"HTML export failed: {exc}")

        def export_pdf(self) -> None:
            if not self.engine.template_backend_available():
                return
            self.last_pdf_export_path = None
            try:
                context = self._capture_context()
                self.last_pdf_export_path = self.engine.export_pdf(
                    self.current_template_name(), context, self._output_path(".pdf")
                )
                self._show_context(context)
                self.export_status.setText(f"Exported PDF: {self.last_pdf_export_path}")
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.export_status.setText(f"PDF export failed: {exc}")

        def generate_report(self) -> None:
            self.last_bundle_path = None
            try:
                context = self._capture_context()
                self.last_bundle_path = self.engine.export_bundle(
                    self.current_template_name(),
                    context,
                    self._output_path(".zip"),
                    include_pdf=self.bundle_pdf_check.isChecked(),
                )
                self._show_context(context)
                self.export_status.setText(
                    f"Generated report bundle: {self.last_bundle_path}"
                )
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.export_status.setText(f"Report bundle failed: {exc}")

        def _sync_template_status(self) -> None:
            available = self.engine.template_backend_available()
            self.template_combo.setEnabled(available)
            self.path_input.setEnabled(available)
            self.render_button.setEnabled(available)
            self.export_button.setEnabled(available)
            self.generate_button.setEnabled(available)

        def _sync_pdf_status(self) -> None:
            available = self.engine.can_export_pdf()
            self.pdf_button.setEnabled(available)
            self.bundle_pdf_check.setEnabled(available)
            if not self.engine.template_backend_available():
                self.pdf_status.setText(
                    "HTML/PDF report export requires the optional reporting extra "
                    "(`Jinja2`)."
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
