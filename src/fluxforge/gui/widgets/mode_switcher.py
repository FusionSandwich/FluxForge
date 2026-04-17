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
        QInputDialog,
        QLabel,
        QPushButton,
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
            self.theme_combo.currentTextChanged.connect(self._theme_changed)
            layout.addWidget(self.theme_combo)

            self.theme_profile_label = QLabel("Profile", self)
            self.theme_profile_label.setObjectName("ModeMetaLabel")
            layout.addWidget(self.theme_profile_label)

            self.theme_profile_combo = QComboBox(self)
            self.theme_profile_combo.setObjectName("ThemeProfileCombo")
            self.theme_profile_combo.currentTextChanged.connect(self._profile_changed)
            layout.addWidget(self.theme_profile_combo)

            self.save_theme_profile_button = QPushButton("Save Profile", self)
            self.save_theme_profile_button.setObjectName("SaveThemeProfileButton")
            self.save_theme_profile_button.clicked.connect(self._save_profile)
            layout.addWidget(self.save_theme_profile_button)

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

        def _theme_changed(self, theme: str) -> None:
            if theme:
                self.mode_manager.set_theme(theme)

        def _profile_changed(self, profile_name: str) -> None:
            profile = profile_name.strip()
            if not profile:
                return
            resolved_theme = self.mode_manager.theme_for_profile(profile)
            if resolved_theme is None:
                return
            if (
                self.mode_manager.state.theme_profile == profile
                and self.mode_manager.state.theme == resolved_theme
            ):
                return
            self.mode_manager.set_theme_profile(profile)

        def _save_profile(self) -> None:
            current = self.mode_manager.state
            default_name = current.theme_profile or f"{current.theme}-profile"
            name, accepted = QInputDialog.getText(
                self,
                "Save Theme Profile",
                "Profile name:",
                text=default_name,
            )
            if not accepted:
                return
            profile_name = name.strip()
            if not profile_name:
                return
            self.mode_manager.save_theme_profile(profile_name, theme=current.theme)
            self.mode_manager.set_theme_profile(profile_name)

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

            self.theme_profile_combo.blockSignals(True)
            self.theme_profile_combo.clear()
            for profile_name in self.mode_manager.available_theme_profiles():
                profile_theme = self.mode_manager.theme_for_profile(profile_name)
                self.theme_profile_combo.addItem(f"{profile_name} ({profile_theme})", profile_name)
            if state.theme_profile:
                profile_index = self.theme_profile_combo.findData(state.theme_profile)
                if profile_index >= 0:
                    self.theme_profile_combo.setCurrentIndex(profile_index)
            elif self.theme_profile_combo.count() > 0:
                self.theme_profile_combo.setCurrentIndex(-1)
            self.theme_profile_combo.blockSignals(False)

else:

    class ModeSwitcherWidget:  # pragma: no cover - placeholder without Qt
        """Import-safe placeholder when Qt is unavailable."""

        def __init__(self, mode_manager: ModeManager, parent=None) -> None:
            self.mode_manager = mode_manager
            self.parent = parent
