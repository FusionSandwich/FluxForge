"""Compatibility helpers for the optional PySide6 GUI stack."""

from __future__ import annotations

QT_AVAILABLE = False
QT_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional GUI extras
    from PySide6.QtCore import QByteArray, QSettings, Qt
    from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QComboBox,
        QDockWidget,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMenuBar,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency branch
    QT_IMPORT_ERROR = exc

__all__ = [
    "QT_AVAILABLE",
    "QT_IMPORT_ERROR",
]

if QT_AVAILABLE:  # pragma: no cover - export names only when available
    __all__ += [
        "QAction",
        "QApplication",
        "QByteArray",
        "QButtonGroup",
        "QComboBox",
        "QDockWidget",
        "QFrame",
        "QGridLayout",
        "QGuiApplication",
        "QHBoxLayout",
        "QKeySequence",
        "QLabel",
        "QLineEdit",
        "QListWidget",
        "QListWidgetItem",
        "QMainWindow",
        "QMenu",
        "QMenuBar",
        "QPlainTextEdit",
        "QProgressBar",
        "QPushButton",
        "QSettings",
        "QSizePolicy",
        "QStatusBar",
        "QTabWidget",
        "QTextEdit",
        "QToolBar",
        "QToolButton",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QVBoxLayout",
        "QWidget",
        "Qt",
    ]
