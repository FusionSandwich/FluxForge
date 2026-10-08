"""Protect unsaved scientific work at destructive GUI boundaries."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from fluxforge.core.workspace_document import (  # noqa: E402
    AnalysisROI,
    WorkspaceDocument,
)
from fluxforge.gui import FluxForgeMainWindow, ModeManager, SelectionBus  # noqa: E402
from fluxforge.io import (  # noqa: E402
    FluxForgeSession,
    read_ffs_session,
    write_ffs_session,
)


class MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        settings=settings,
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.analysis_workspace.upsert_roi(
        AnalysisROI(
            roi_id="unsaved",
            spectrum_id=window.analysis_workspace.document.active_spectrum_id,
            signal_range=(100.0, 200.0),
            left_background_range=(80.0, 100.0),
            right_background_range=(200.0, 220.0),
        )
    )
    assert window._document_dirty
    yield window
    window._document_dirty = False
    window.close()
    app.processEvents()


def _act(window, action, path):
    if action == "close":
        event = QCloseEvent()
        window.closeEvent(event)
        return event.isAccepted()
    if action == "open":
        window.open_path(path)
    elif action == "example":
        window._load_example_workspace()
    else:
        window._reset_analysis_workspace()


@pytest.mark.parametrize("action", ["close", "open", "example", "reset"])
def test_cancel_preserves_document_history_and_session(
    window, tmp_path, monkeypatch, action
):
    path = tmp_path / "next.ffs"
    write_ffs_session(
        path, FluxForgeSession(document=WorkspaceDocument(document_id="next"))
    )
    before = window.analysis_workspace.document.to_dict()
    old_path = window._session_path
    prompts = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: (prompts.append(args), QMessageBox.Cancel)[1],
    )
    assert _act(window, action, path) is not True
    assert len(prompts) == 1
    assert window.analysis_workspace.document.to_dict() == before
    assert window._session_path == old_path
    assert window._document_dirty


def test_save_as_cancel_keeps_close_pending(window, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Save)
    monkeypatch.setattr(window, "_save_session_as_dialog", lambda: False)
    assert _act(window, "close", None) is False
    assert window.analysis_workspace.document.roi_by_id("unsaved") is not None
    assert window._document_dirty


def test_save_before_replacement_preserves_original_edits(
    window, tmp_path, monkeypatch
):
    saved = tmp_path / "saved.ffs"
    next_path = tmp_path / "next.ffs"
    write_ffs_session(
        next_path, FluxForgeSession(document=WorkspaceDocument(document_id="next"))
    )
    window._session_path = saved
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Save)
    window.open_path(next_path)
    assert read_ffs_session(saved).document.roi_by_id("unsaved") is not None
    assert window.analysis_workspace.document.document_id == "next"
    assert window._session_path == next_path.resolve()
    assert not window._document_dirty


def test_save_failure_does_not_close(window, tmp_path, monkeypatch):
    from fluxforge.gui import main_window

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Save)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: None)

    def failed_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(main_window, "write_ffs_session", failed_write)
    window._session_path = tmp_path / "failed.ffs"
    assert _act(window, "close", None) is False
    assert window._document_dirty
    assert not window._session_path.exists()


def test_discard_replaces_without_saving(window, tmp_path, monkeypatch):
    path = tmp_path / "next.ffs"
    write_ffs_session(
        path, FluxForgeSession(document=WorkspaceDocument(document_id="next"))
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Discard)
    monkeypatch.setattr(
        window, "save_session", lambda *args: pytest.fail("Discard must not save")
    )
    window.open_path(path)
    assert window.analysis_workspace.document.document_id == "next"
