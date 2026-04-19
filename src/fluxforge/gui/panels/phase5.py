"""Phase 5 GUI panels for writeup-crosswalk and parity execution review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from fluxforge.gui.qt_compat import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from fluxforge.validation import (
    load_phase5_crosswalk,
    run_reference_parity_suite,
    summarize_phase5_crosswalk,
)


def _short_path(paths: Sequence[str]) -> str:
    if not paths:
        return ""
    return str(paths[0])


class Phase5ParityPanel(QWidget):
    """Qt surface for Phase 5 crosswalk status and parity run summaries."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self._last_parity_summary: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            (
                "Phase 5 Parity tracks the writeup crosswalk, replay-state coverage, and "
                "algorithm/workflow fixture execution summaries."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QGridLayout()
        controls.addWidget(QLabel("Crosswalk JSON", self), 0, 0)
        self.crosswalk_path_edit = QLineEdit(
            ".github/project-management/phase5_crosswalk.json",
            self,
        )
        self.crosswalk_path_edit.setObjectName("Phase5CrosswalkPathEdit")
        controls.addWidget(self.crosswalk_path_edit, 0, 1, 1, 3)

        controls.addWidget(QLabel("Replay filter", self), 1, 0)
        self.replay_filter_combo = QComboBox(self)
        self.replay_filter_combo.setObjectName("Phase5ReplayFilterCombo")
        self.replay_filter_combo.addItems(
            [
                "all",
                "replay-now",
                "adapter-required",
                "reference-only",
            ]
        )
        controls.addWidget(self.replay_filter_combo, 1, 1)

        self.refresh_button = QPushButton("Refresh Crosswalk", self)
        self.refresh_button.setObjectName("Phase5RefreshButton")
        self.refresh_button.clicked.connect(self.refresh_crosswalk)
        controls.addWidget(self.refresh_button, 1, 2)

        self.run_parity_button = QPushButton("Run Parity Suite", self)
        self.run_parity_button.setObjectName("Phase5RunParityButton")
        self.run_parity_button.clicked.connect(self.run_parity_suite)
        controls.addWidget(self.run_parity_button, 1, 3)
        layout.addLayout(controls)

        self.summary_label = QLabel("Crosswalk summary has not been loaded yet.", self)
        self.summary_label.setObjectName("Phase5SummaryLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.parity_label = QLabel("Parity suite has not been executed from this panel.", self)
        self.parity_label.setObjectName("Phase5ParityStatusLabel")
        self.parity_label.setWordWrap(True)
        layout.addWidget(self.parity_label)

        self.table = QTableWidget(self)
        self.table.setObjectName("Phase5CrosswalkTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        footer.addWidget(
            QLabel(
                "Columns show writeup section, replay state, and first landing/fixture anchor.",
                self,
            )
        )
        footer.addStretch(1)
        layout.addLayout(footer)

        self.replay_filter_combo.currentTextChanged.connect(self._refresh_table_only)
        self.refresh_crosswalk()

    def _crosswalk_path(self) -> Path:
        raw = str(self.crosswalk_path_edit.text() or "").strip()
        if not raw:
            raw = ".github/project-management/phase5_crosswalk.json"
        return Path(raw)

    def refresh_crosswalk(self) -> dict[str, Any] | None:
        path = self._crosswalk_path()
        try:
            payload = load_phase5_crosswalk(path)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self._entries = []
            self._summary = {}
            self.summary_label.setText(f"Crosswalk load failed: {type(exc).__name__}: {exc}")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return None

        self._entries = [dict(item) for item in payload.get("entries") or []]
        self._summary = summarize_phase5_crosswalk(payload, workspace_root=Path.cwd())

        replay = self._summary.get("by_replay_state") or {}
        self.summary_label.setText(
            (
                f"Loaded {self._summary.get('total_entries', 0)} entries; "
                f"replay-now={replay.get('replay-now', 0)}, "
                f"adapter-required={replay.get('adapter-required', 0)}, "
                f"reference-only={replay.get('reference-only', 0)}"
            )
        )
        self._refresh_table_only()
        return self._summary

    def _refresh_table_only(self) -> None:
        rows = self._filtered_entries()
        headers = [
            "section",
            "source_family",
            "replay_state",
            "data_class",
            "backend",
            "cli",
            "gui",
            "fixture",
        ]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for row_index, entry in enumerate(rows):
            values = [
                str(entry.get("writeup_section") or ""),
                str(entry.get("source_family") or ""),
                str(entry.get("replay_state") or ""),
                str(entry.get("data_class") or ""),
                _short_path([str(item) for item in entry.get("backend_paths") or []]),
                _short_path([str(item) for item in entry.get("cli_paths") or []]),
                _short_path([str(item) for item in entry.get("gui_paths") or []]),
                _short_path([str(item) for item in entry.get("fixture_paths") or []]),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))

    def _filtered_entries(self) -> list[dict[str, Any]]:
        replay_filter = str(self.replay_filter_combo.currentText() or "all")
        entries = sorted(
            self._entries,
            key=lambda item: int(item.get("writeup_section") or 0),
        )
        if replay_filter == "all":
            return entries
        return [
            entry
            for entry in entries
            if str(entry.get("replay_state") or "") == replay_filter
        ]

    def run_parity_suite(self) -> dict[str, Any] | None:
        try:
            payload = run_reference_parity_suite(
                reference_root=Path("tests/spectra/reference_parity"),
                activation_root=Path("tests/activation_inventory/fixtures"),
                scope="all",
                include_activation=True,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.parity_label.setText(f"Parity run failed: {type(exc).__name__}: {exc}")
            return None

        summary = payload.get("summary") or {}
        self._last_parity_summary = dict(summary)
        self.parity_label.setText(
            (
                f"Parity suite: {summary.get('passed', 0)} passed, "
                f"{summary.get('failed', 0)} failed, total={summary.get('total', 0)}"
            )
        )
        return payload

    def workflow_state(self) -> dict[str, Any]:
        return {
            "crosswalk_path": str(self.crosswalk_path_edit.text() or ""),
            "replay_state_filter": str(self.replay_filter_combo.currentText() or "all"),
        }

    def apply_workflow_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if "crosswalk_path" in payload:
            self.crosswalk_path_edit.setText(str(payload.get("crosswalk_path") or ""))
        self.refresh_crosswalk()
        if "replay_state_filter" in payload:
            replay_filter = str(payload.get("replay_state_filter") or "all")
            index = self.replay_filter_combo.findText(replay_filter)
            if index >= 0:
                self.replay_filter_combo.setCurrentIndex(index)
            self._refresh_table_only()


__all__ = ["Phase5ParityPanel"]
