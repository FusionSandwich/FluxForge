"""Background folder/file queue for append and covariance-preserving summing."""

from pathlib import Path

from fluxforge.core.spectrum_file_queue import (
    convert_spectrum_file_queue,
    discover_spectrum_files,
    unique_spectrum_paths,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.io.reader_factory import create_reader_factory

if QT_AVAILABLE:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    class _ConversionWorker(QThread):
        progress = Signal(int, int)
        completed = Signal(object)
        failed = Signal(str)

        def __init__(self, paths, output, mode, independent, parent):
            super().__init__(parent)
            self.paths, self.output, self.mode, self.independent = (
                paths,
                output,
                mode,
                independent,
            )

        def run(self):
            try:
                result = convert_spectrum_file_queue(
                    self.paths,
                    self.output,
                    mode=self.mode,
                    independent_acquisitions=self.independent,
                    progress_callback=self.progress.emit,
                )
                self.completed.emit(result)
            except Exception as exc:
                self.failed.emit(str(exc))

    class SpectrumFileQueueDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Spectrum Summing and Conversion")
            self.resize(840, 540)
            self.paths = ()
            self.worker = None
            self.last_result = None
            layout = QVBoxLayout(self)
            note = QLabel(
                "Queue raw files or folders. Append preserves separate spectra "
                "in one native .ffs session; Sum combines only identical bins "
                "and calibration. Open the output with File → Open Session.",
                self,
            )
            note.setWordWrap(True)
            layout.addWidget(note)
            buttons = QHBoxLayout()
            self.files_button = QPushButton("Add files…", self)
            self.files_button.setObjectName("SpectrumQueueFilesButton")
            self.files_button.clicked.connect(self._choose_files)
            self.folder_button = QPushButton("Add folder…", self)
            self.folder_button.setObjectName("SpectrumQueueFolderButton")
            self.folder_button.clicked.connect(self._choose_folder)
            self.remove_button = QPushButton("Remove selected", self)
            self.remove_button.setObjectName("SpectrumQueueRemoveButton")
            self.remove_button.clicked.connect(self.remove_selected)
            self.recursive_check = QCheckBox("Include subfolders", self)
            self.recursive_check.setObjectName("SpectrumQueueRecursiveCheck")
            for widget in (
                self.files_button,
                self.folder_button,
                self.remove_button,
                self.recursive_check,
            ):
                buttons.addWidget(widget)
            layout.addLayout(buttons)
            self.table = QTableWidget(0, 1, self)
            self.table.setObjectName("SpectrumFileQueueTable")
            self.table.setHorizontalHeaderLabels(["Queued source files"])
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.table)
            options = QHBoxLayout()
            self.mode_combo = QComboBox(self)
            self.mode_combo.setObjectName("SpectrumQueueModeCombo")
            self.mode_combo.addItem("Append separate spectra", "append")
            self.mode_combo.addItem("Sum matching spectra", "sum")
            self.independent_check = QCheckBox(
                "Inputs are independent acquisitions", self
            )
            self.independent_check.setObjectName("SpectrumQueueIndependentCheck")
            self.mode_combo.currentIndexChanged.connect(self._mode_changed)
            options.addWidget(self.mode_combo)
            options.addWidget(self.independent_check)
            layout.addLayout(options)
            output = QHBoxLayout()
            self.output_input = QLineEdit(self)
            self.output_input.setObjectName("SpectrumQueueOutputInput")
            self.output_input.setPlaceholderText("New output file (.ffs)")
            self.output_button = QPushButton("Choose output…", self)
            self.output_button.setObjectName("SpectrumQueueOutputButton")
            self.output_button.clicked.connect(self._choose_output)
            output.addWidget(self.output_input)
            output.addWidget(self.output_button)
            layout.addLayout(output)
            self.run_button = QPushButton("Convert queue", self)
            self.run_button.setObjectName("SpectrumQueueRunButton")
            self.run_button.clicked.connect(self.start_conversion)
            layout.addWidget(self.run_button)
            self.progress_bar = QProgressBar(self)
            layout.addWidget(self.progress_bar)
            self.status_label = QLabel(
                "No files queued. Existing output files are preserved.", self
            )
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)
            self._mode_changed()

        def _mode_changed(self):
            self.independent_check.setEnabled(self.mode_combo.currentData() == "sum")

        def add_paths(self, paths):
            if self.worker is not None:
                return
            self.paths = unique_spectrum_paths((*self.paths, *paths))
            self._refresh()

        def add_folder(self, folder):
            self.add_paths(
                discover_spectrum_files(
                    folder, recursive=self.recursive_check.isChecked()
                )
            )

        def remove_selected(self):
            indices = {item.row() for item in self.table.selectedIndexes()}
            self.paths = tuple(
                path for index, path in enumerate(self.paths) if index not in indices
            )
            self._refresh()

        def _refresh(self):
            self.table.setRowCount(len(self.paths))
            for index, path in enumerate(self.paths):
                self.table.setItem(index, 0, QTableWidgetItem(str(path)))
            self.status_label.setText(f"Queued {len(self.paths)} unique files.")

        def _choose_files(self):
            extensions = " ".join(
                f"*{suffix}"
                for suffix in create_reader_factory().supported_extensions()
            )
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Add spectra", "", f"Spectra ({extensions})"
            )
            self.add_paths(paths)

        def _choose_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Add spectrum folder")
            if folder:
                try:
                    self.add_folder(folder)
                except OSError as exc:
                    self.status_label.setText(str(exc))

        def _choose_output(self):
            path, _ = QFileDialog.getSaveFileName(
                self, "New combined output", "", "FluxForge session (*.ffs)"
            )
            if path:
                self.output_input.setText(path)

        def start_conversion(self):
            if self.worker is not None:
                return
            output = self.output_input.text().strip()
            if not self.paths or not output:
                self.status_label.setText(
                    "Queue input files and choose a new .ffs output."
                )
                return
            if Path(output).exists():
                self.status_label.setText(
                    "Output exists. Choose a new file to preserve it."
                )
                return
            mode = self.mode_combo.currentData()
            if mode == "sum" and not self.independent_check.isChecked():
                self.status_label.setText(
                    "Declare independent acquisitions before summing."
                )
                return
            self.last_result = None
            for widget in self._controls():
                widget.setEnabled(False)
            self.progress_bar.setRange(0, len(self.paths))
            self.progress_bar.setValue(0)
            self.status_label.setText(
                "Reading the queue. Close after conversion finishes."
            )
            self.worker = _ConversionWorker(
                self.paths, output, mode, self.independent_check.isChecked(), self
            )
            self.worker.progress.connect(
                lambda done, total: self.progress_bar.setValue(done)
            )
            self.worker.completed.connect(self._completed)
            self.worker.failed.connect(
                lambda message: self.status_label.setText(
                    f"Conversion failed: {message}"
                )
            )
            self.worker.finished.connect(self._finished)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.start()

        def _controls(self):
            return (
                self.files_button,
                self.folder_button,
                self.remove_button,
                self.recursive_check,
                self.table,
                self.mode_combo,
                self.independent_check,
                self.output_input,
                self.output_button,
                self.run_button,
            )

        def _completed(self, result):
            self.last_result = result
            self.status_label.setText(
                f"Saved {result['output_spectra']} spectra from "
                f"{result['input_count']} inputs to {result['output']}."
            )

        def _finished(self):
            self.worker = None
            for widget in self._controls():
                widget.setEnabled(True)
            self._mode_changed()

        def closeEvent(self, event):
            if self.worker is not None:
                event.ignore()
            else:
                super().closeEvent(event)

else:

    class SpectrumFileQueueDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Native Qt is unavailable.")
