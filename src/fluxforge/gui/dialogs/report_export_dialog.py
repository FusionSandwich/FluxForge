"""Report export dialog for the modern Qt shell."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.reporting.engine import ReportingEngine
from fluxforge.reporting.instrument_provenance import (
    SETTING_FIELDS,
    instrument_provenance,
)
import json

if QT_AVAILABLE:  # pragma: no cover - optional GUI branch
    from PySide6.QtCore import QThread, Signal
    from fluxforge.gui.qt_compat import (
        QComboBox,
        QCheckBox,
        QDialog,
        QHBoxLayout,
        QGridLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QPushButton,
        QProgressBar,
        QTextBrowser,
        QVBoxLayout,
    )


if QT_AVAILABLE:  # pragma: no cover - optional GUI branch

    class _ReportExportWorker(QThread):
        """Render an isolated snapshot; never access a Qt widget from this thread."""

        def __init__(
            self, engine, template, context, operation, path, include_pdf, parent
        ):
            super().__init__(parent)
            self.engine = engine
            self.template = template
            self.context = context
            self.operation = operation
            self.path = path
            self.include_pdf = include_pdf
            self.output = None
            self.html = None
            self.error = None

        def run(self):
            try:
                self.html = self.engine.render(self.template, self.context).html
                if self.operation == "bundle":
                    self.output = self.engine.export_bundle(
                        self.template,
                        self.context,
                        self.path,
                        include_pdf=self.include_pdf,
                    )
                elif self.operation == "pdf":
                    self.output = self.engine.export_pdf(
                        self.template, self.context, self.path
                    )
                else:
                    self.output = self.engine.export_html(
                        self.template, self.context, self.path
                    )
            except Exception as exc:
                self.error = str(exc)

    class ReportExportDialog(QDialog):
        """Preview and export bundled Jinja2 reports."""

        export_finished = Signal(bool)

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
            self._instrument_spectrum_id = None
            self._instrument_overrides = {}
            self._instrument_sources = {}
            self._instrument_document_id = None
            self.worker = None

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

            provenance_group = QGroupBox(
                "Acquisition settings for current spectrum", self
            )
            provenance_layout = QGridLayout(provenance_group)
            self.instrument_inputs = {}
            for index, (name, label, object_name) in enumerate(
                (
                    (
                        "amplifier_gain",
                        "Amplifier gain (×)",
                        "ReportAmplifierGainInput",
                    ),
                    ("shaping_time_us", "Shaping time (µs)", "ReportShapingTimeInput"),
                    ("high_voltage_v", "High voltage (V)", "ReportHighVoltageInput"),
                    ("count_geometry", "Count geometry", "ReportCountGeometryInput"),
                )
            ):
                editor = QLineEdit(provenance_group)
                editor.setObjectName(object_name)
                editor.setPlaceholderText(
                    "Recorded value; blank uses imported metadata"
                )
                provenance_layout.addWidget(
                    QLabel(label, provenance_group), index // 2, (index % 2) * 2
                )
                provenance_layout.addWidget(editor, index // 2, (index % 2) * 2 + 1)
                self.instrument_inputs[name] = editor
            self.instructional_check = QCheckBox(
                "Require complete instrument settings (instructional report)",
                provenance_group,
            )
            self.instructional_check.setObjectName("ReportInstructionalCheck")
            provenance_layout.addWidget(self.instructional_check, 2, 0, 1, 4)
            self.instrument_status = QLabel(provenance_group)
            self.instrument_status.setWordWrap(True)
            provenance_layout.addWidget(self.instrument_status, 3, 0, 1, 4)
            root.addWidget(provenance_group)

            self.export_status = QLabel(self)
            self.export_status.setObjectName("ReportExportStatus")
            self.export_status.setWordWrap(True)
            root.addWidget(self.export_status)
            self.export_progress = QProgressBar(self)
            self.export_progress.setObjectName("ReportExportProgress")
            self.export_progress.setRange(0, 0)
            self.export_progress.hide()
            root.addWidget(self.export_progress)

            self.pdf_status = QLabel(self)
            self.pdf_status.setObjectName("ReportPdfStatus")
            self.pdf_status.setWordWrap(True)
            root.addWidget(self.pdf_status)

            self.preview = QTextBrowser(self)
            self.preview.setObjectName("ReportPreviewBrowser")
            self.preview.setStyleSheet("background-color: white; color: #10253c;")
            root.addWidget(self.preview, 1)

            self._workspace_controller = getattr(parent, "analysis_workspace", None)
            if self._workspace_controller is not None:
                self._workspace_controller.subscribe_document(self._on_document_changed)
                self._on_document_changed(self._workspace_controller.document)

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
            if self.worker is not None:
                return
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
            context = deepcopy(self.context_factory(self.current_template_name()))
            snapshot = context.get("run_snapshot")
            if snapshot is not None:
                workspace = snapshot["workspace"]
                active = workspace.get("active_spectrum_id")
                self._bind_instrument_inputs(active)
                self._save_instrument_inputs(active)
                provenance = instrument_provenance(
                    workspace,
                    self._instrument_overrides,
                    require_complete=False,
                )
                snapshot["instrument_provenance"] = provenance
                snapshot["instructional_report"] = self.instructional_check.isChecked()
                context["instrument_provenance_json"] = json.dumps(provenance, indent=2)
                missing = provenance["missing_required"]
                active_record = next(
                    (
                        record
                        for record in provenance["records"]
                        if record["spectrum_id"] == active
                    ),
                    None,
                )
                incomplete = sum(
                    bool(record["missing_required"]) for record in provenance["records"]
                )
                labels = {
                    "amplifier_gain": "amplifier gain",
                    "shaping_time_us": "shaping time",
                    "high_voltage_v": "high voltage",
                    "count_geometry": "count geometry",
                    "live_time_s": "live time",
                    "real_time_s": "real time",
                    "counting_time_order": "valid live/real time order",
                    "mca_calibration": "MCA calibration",
                }
                active_missing = (
                    active_record["missing_required"] if active_record else []
                )
                current_label = (
                    (active_record.get("label") or active) if active_record else "none"
                )
                self.instrument_status.setText(
                    f"Current spectrum: {current_label}. "
                    + (
                        "All loaded spectra have the required recorded settings."
                        if not missing
                        else (
                            "Missing here: "
                            + ", ".join(
                                labels.get(name, name) for name in active_missing
                            )
                            + ". "
                            if active_missing
                            else ""
                        )
                        + f"{incomplete} acquisitions need recorded settings. "
                        "Select each spectrum in the workspace to enter its settings."
                    )
                )
                if self.instructional_check.isChecked() and missing:
                    raise ValueError(
                        "Instructional report needs complete recorded acquisition settings. "
                        "Review the current spectrum's missing settings above."
                    )
            return context

        def _save_instrument_inputs(self, spectrum_id) -> None:
            if spectrum_id is None:
                return
            values = {name: self.instrument_inputs[name].text() for name in SETTING_FIELDS}
            if any(value.strip() for value in values.values()):
                self._instrument_overrides[spectrum_id] = values
            else:
                self._instrument_overrides.pop(spectrum_id, None)

        def _bind_instrument_inputs(self, active) -> None:
            self._save_instrument_inputs(self._instrument_spectrum_id)
            if active != self._instrument_spectrum_id:
                restored = self._instrument_overrides.get(active, {})
                for name, editor in self.instrument_inputs.items():
                    editor.setText(restored.get(name, ""))
                self._instrument_spectrum_id = active
            self.instrument_status.setText(
                f"Acquisition settings are bound to spectrum: {active or 'none'}."
            )

        def _on_document_changed(self, document) -> None:
            sources = {
                item.spectrum_id: item.spectrum.counts for item in document.spectra
            }
            if document.document_id != self._instrument_document_id:
                changed = set(self._instrument_sources)
            else:
                changed = {
                    key
                    for key, counts in self._instrument_sources.items()
                    if sources.get(key) is not counts
                }
            for key in changed:
                self._instrument_overrides.pop(key, None)
            if self._instrument_spectrum_id in changed:
                for editor in self.instrument_inputs.values():
                    editor.clear()
                self._instrument_spectrum_id = None
            self._instrument_sources = sources
            self._instrument_document_id = document.document_id
            self._bind_instrument_inputs(document.active_spectrum_id)

        def _detach_workspace(self):
            if self._workspace_controller is not None:
                self._workspace_controller.unsubscribe_document(
                    self._on_document_changed
                )
                self._workspace_controller = None

        def closeEvent(self, event) -> None:
            if self.worker is not None:
                self.export_status.setText(
                    "Report export is running. Close after it finishes."
                )
                event.ignore()
                return
            self._detach_workspace()
            super().closeEvent(event)

        def done(self, result):
            # Escape/accept/reject bypass closeEvent in QDialog.
            if self.worker is not None:
                self.export_status.setText(
                    "Report export is running. Close after it finishes."
                )
                return
            self._detach_workspace()
            super().done(result)

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
            self._start_export("html")

        def export_pdf(self) -> None:
            self._start_export("pdf")

        def generate_report(self) -> None:
            self._start_export("bundle")

        def _start_export(self, operation):
            if self.worker is not None or not self.engine.template_backend_available():
                return
            attributes = {
                "html": "last_export_path",
                "pdf": "last_pdf_export_path",
                "bundle": "last_bundle_path",
            }
            setattr(self, attributes[operation], None)
            try:
                template = self.current_template_name()
                path = self._output_path(
                    {"html": None, "pdf": ".pdf", "bundle": ".zip"}[operation]
                )
                # Widget/table/plot capture belongs on the GUI thread. The worker
                # receives a detached value and the path/options from this click.
                context = self._capture_context()
                self.worker = _ReportExportWorker(
                    self.engine,
                    template,
                    context,
                    operation,
                    path,
                    self.bundle_pdf_check.isChecked(),
                    self,
                )
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.export_status.setText(f"Report {operation} export failed: {exc}")
                return
            for control in self._export_controls():
                control.setEnabled(False)
            self.export_progress.show()
            self.export_status.setText("Exporting the captured report…")
            self.worker.finished.connect(self._finish_export)
            self.worker.start()

        def _export_controls(self):
            return (
                self.template_combo,
                self.path_input,
                self.render_button,
                self.export_button,
                self.pdf_button,
                self.generate_button,
                self.bundle_pdf_check,
                self.instructional_check,
                *self.instrument_inputs.values(),
            )

        def _finish_export(self):
            worker = self.worker
            if worker is None:
                return
            self.worker = None
            self.export_progress.hide()
            for control in self._export_controls():
                control.setEnabled(True)
            self._sync_template_status()
            self._sync_pdf_status()
            if worker.error is not None:
                self.export_status.setText(
                    f"Report {worker.operation} export failed: {worker.error}"
                )
            else:
                attribute, label = {
                    "html": ("last_export_path", "Exported HTML"),
                    "pdf": ("last_pdf_export_path", "Exported PDF"),
                    "bundle": ("last_bundle_path", "Generated report bundle"),
                }[worker.operation]
                setattr(self, attribute, worker.output)
                self.last_context = worker.context
                self.preview.setHtml(worker.html)
                self.export_status.setText(f"{label}: {worker.output}")
            succeeded = worker.error is None
            worker.deleteLater()
            self.export_finished.emit(succeeded)

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
