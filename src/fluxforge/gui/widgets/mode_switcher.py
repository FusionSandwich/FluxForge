"""Modern mode and theme switcher widget."""

from __future__ import annotations

from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.theme_manager import available_themes

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import (
        QButtonGroup,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QToolButton,
        QWidget,
    )


STANDARD_CHOICES = (
    "ASTM E181",
    "ASTM E261",
    "ASTM E1297",
    "ASTM C1030",
)


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class ModeSwitcherWidget(QWidget):
        """Toolbar widget for GUI mode and theme selection."""

        def __init__(self, mode_manager: ModeManager, parent=None) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager

            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)

            self.mode_group = QButtonGroup(self)
            self.mode_group.setExclusive(True)
            self.mode_buttons: dict[GUIMode, QToolButton] = {}
            for label, mode in (
                ("Simple", GUIMode.SIMPLE),
                ("Expert", GUIMode.EXPERT),
                ("Standards", GUIMode.STANDARDS),
            ):
                button = QToolButton(self)
                button.setObjectName("ModeButton")
                button.setText(label)
                button.setCheckable(True)
                button.clicked.connect(
                    lambda checked=False, selected=mode: self._set_mode(selected)
                )
                self.mode_group.addButton(button)
                self.mode_buttons[mode] = button
                layout.addWidget(button)

            self.standard_label = QLabel("Standard", self)
            self.standard_label.setObjectName("ModeMetaLabel")
            layout.addWidget(self.standard_label)

            self.standard_combo = QComboBox(self)
            self.standard_combo.setObjectName("StandardCombo")
            self.standard_combo.addItems(list(STANDARD_CHOICES))
            self.standard_combo.currentTextChanged.connect(self._standard_changed)
            layout.addWidget(self.standard_combo)

            self.theme_label = QLabel("Theme", self)
            self.theme_label.setObjectName("ModeMetaLabel")
            layout.addWidget(self.theme_label)

            self.theme_combo = QComboBox(self)
            self.theme_combo.setObjectName("ThemeCombo")
            self.theme_combo.addItems(list(available_themes()))
            self.theme_combo.currentTextChanged.connect(self.mode_manager.set_theme)
            layout.addWidget(self.theme_combo)

            self.mode_manager.subscribe(self._sync_from_state)
            self._sync_from_state(self.mode_manager.state)

        def _set_mode(self, mode: GUIMode) -> None:
            if mode is GUIMode.STANDARDS:
                self.mode_manager.set_standard(self.standard_combo.currentText())
            else:
                self.mode_manager.set_mode(mode)

        def _standard_changed(self, standard: str) -> None:
            if self.mode_manager.state.mode is GUIMode.STANDARDS and standard:
                self.mode_manager.set_standard(standard)

        def _sync_from_state(self, state) -> None:
            for mode, button in self.mode_buttons.items():
                button.blockSignals(True)
                button.setChecked(mode is state.mode)
                button.blockSignals(False)

            self.standard_label.setVisible(state.mode is GUIMode.STANDARDS)
            self.standard_combo.setVisible(state.mode is GUIMode.STANDARDS)

            if state.standard:
                index = self.standard_combo.findText(state.standard)
                if index >= 0:
                    self.standard_combo.blockSignals(True)
                    self.standard_combo.setCurrentIndex(index)
                    self.standard_combo.blockSignals(False)

            index = self.theme_combo.findText(state.theme)
            if index >= 0:
                self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentIndex(index)
                self.theme_combo.blockSignals(False)

else:

    class ModeSwitcherWidget:  # pragma: no cover - placeholder without Qt
        """Import-safe placeholder when Qt is unavailable."""

        def __init__(self, mode_manager: ModeManager, parent=None) -> None:
            self.mode_manager = mode_manager
            self.parent = parent
