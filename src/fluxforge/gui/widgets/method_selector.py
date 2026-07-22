"""Reusable method selector for registry-backed GUI workflows."""

from __future__ import annotations

from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.plugins import PluginRegistry

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class MethodSelectorWidget(QWidget):
        """Small registry-backed selector with a mode-aware badge."""

        def __init__(
            self,
            registry: PluginRegistry,
            mode_manager: ModeManager,
            *,
            title: str = "Method",
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.registry = registry
            self.mode_manager = mode_manager
            object_stem = "".join(
                character for character in title.title() if character.isalnum()
            )
            self.setObjectName(f"{object_stem}MethodSelector")

            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(6)

            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            self.title_label = QLabel(title, self)
            self.title_label.setObjectName("MethodSelectorTitle")
            title_row.addWidget(self.title_label)
            title_row.addStretch(1)
            self.badge_label = QLabel("Recommended", self)
            self.badge_label.setObjectName("MethodSelectorBadge")
            title_row.addWidget(self.badge_label)
            root.addLayout(title_row)

            self.combo = QComboBox(self)
            self.combo.setObjectName(f"{object_stem}MethodSelectorCombo")
            self.combo.currentIndexChanged.connect(self._sync_badge)
            root.addWidget(self.combo)

            self.detail_label = QLabel("", self)
            self.detail_label.setObjectName("PanelBody")
            self.detail_label.setWordWrap(True)
            root.addWidget(self.detail_label)

            self.mode_manager.subscribe(self._sync_mode)
            self._sync_mode(self.mode_manager.state)

        @staticmethod
        def _entry_label(entry) -> str:
            implementation = entry.implementation
            if hasattr(implementation, "label"):
                return str(implementation.label)
            definition = getattr(type(implementation), "definition", None)
            if callable(definition):
                resolved = definition()
                label = getattr(resolved, "label", None)
                if label:
                    return str(label)
            return str(entry.key)

        def current_key(self) -> str | None:
            data = self.combo.currentData()
            return str(data) if data is not None else None

        def set_current_key(self, key: str) -> None:
            index = self.combo.findData(key)
            if index >= 0:
                self.combo.setCurrentIndex(index)

        def _sync_mode(self, state) -> None:
            current_key = self.current_key() or self.registry.default_key
            entries = list(self.registry.entries())
            if state.mode is GUIMode.STANDARDS:
                locked_entries = [
                    entry for entry in entries if entry.metadata.standards_locked
                ]
                if locked_entries:
                    entries = locked_entries
            self.combo.blockSignals(True)
            self.combo.clear()
            for entry in entries:
                self.combo.addItem(self._entry_label(entry), entry.key)
            self.combo.blockSignals(False)
            if current_key is not None:
                index = self.combo.findData(current_key)
                if index >= 0:
                    self.combo.setCurrentIndex(index)
            if self.combo.currentIndex() < 0 and self.combo.count() > 0:
                self.combo.setCurrentIndex(0)
            self.combo.setEnabled(
                state.mode is not GUIMode.STANDARDS and self.combo.count() > 1
            )
            self._sync_badge()

        def _sync_badge(self, *_args) -> None:
            key = self.current_key()
            if key is None:
                self.badge_label.setText("Unavailable")
                self.detail_label.setText("")
                return
            entry = self.registry.get_entry(key)
            if (
                self.mode_manager.state.mode is GUIMode.STANDARDS
                and entry.metadata.standards_locked
            ):
                badge = "🔒 Standards Locked"
            elif entry.metadata.recommended:
                badge = "Recommended"
            else:
                badge = "Alternative"
            self.badge_label.setText(badge)
            self.detail_label.setText(entry.metadata.description)

else:

    class MethodSelectorWidget:  # pragma: no cover - placeholder without Qt
        def __init__(
            self,
            registry: PluginRegistry,
            mode_manager: ModeManager,
            *,
            title: str = "Method",
            parent=None,
        ) -> None:
            self.registry = registry
            self.mode_manager = mode_manager
            self.title = title
            self.parent = parent
