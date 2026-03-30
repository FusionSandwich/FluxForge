"""Mode management for Simple, Expert, and Standards workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class GUIMode(str, Enum):
    """Supported GUI modes."""

    SIMPLE = "simple"
    EXPERT = "expert"
    STANDARDS = "standards"


@dataclass(frozen=True)
class ModeState:
    """Serializable GUI mode state."""

    mode: GUIMode = GUIMode.EXPERT
    standard: str | None = None
    theme: str = "dark"


ModeListener = Callable[[ModeState], None]


class ModeManager:
    """Pure-Python mode manager with persistence and subscriber hooks."""

    MODE_KEY = "gui/mode"
    STANDARD_KEY = "gui/standard"
    THEME_KEY = "gui/theme"

    def __init__(
        self,
        initial_state: ModeState | None = None,
        settings=None,
    ) -> None:
        self._settings = settings
        self._state = initial_state or self._load_state()
        self._listeners: list[ModeListener] = []

    @property
    def state(self) -> ModeState:
        return self._state

    def subscribe(self, listener: ModeListener) -> None:
        """Register a listener for mode-state changes."""

        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: ModeListener) -> None:
        """Remove a previously registered listener."""

        if listener in self._listeners:
            self._listeners.remove(listener)

    def _publish(self, state: ModeState) -> ModeState:
        self._state = state
        self._save_state(state)
        for listener in tuple(self._listeners):
            listener(state)
        return state

    def _load_state(self) -> ModeState:
        """Load the persisted state when a settings backend is available."""

        if self._settings is None:
            return ModeState()

        theme = str(self._settings_value(self.THEME_KEY, "dark")).strip().lower()
        if theme not in {"dark", "light", "system"}:
            theme = "dark"

        mode_token = str(self._settings_value(self.MODE_KEY, GUIMode.EXPERT.value)).strip().lower()
        try:
            mode = GUIMode(mode_token)
        except ValueError:
            mode = GUIMode.EXPERT

        standard = self._settings_value(self.STANDARD_KEY, None)
        standard = standard or None
        if mode is not GUIMode.STANDARDS:
            standard = None

        return ModeState(mode=mode, standard=standard, theme=theme)

    def _save_state(self, state: ModeState) -> None:
        """Persist the current state when a settings backend is available."""

        if self._settings is None:
            return

        self._settings_set_value(self.MODE_KEY, state.mode.value)
        self._settings_set_value(self.STANDARD_KEY, state.standard or "")
        self._settings_set_value(self.THEME_KEY, state.theme)
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()

    def _settings_value(self, key: str, default):
        getter = getattr(self._settings, "value", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _settings_set_value(self, key: str, value) -> None:
        setter = getattr(self._settings, "setValue", None)
        if callable(setter):
            setter(key, value)

    def set_mode(self, mode: GUIMode, standard: str | None = None) -> ModeState:
        """Update the active mode, requiring a standard in standards mode."""

        if mode is GUIMode.STANDARDS and not standard:
            raise ValueError("Standards mode requires an active standard name")
        if mode is not GUIMode.STANDARDS:
            standard = None
        return self._publish(
            ModeState(
                mode=mode,
                standard=standard,
                theme=self._state.theme,
            )
        )

    def set_standard(self, standard: str) -> ModeState:
        """Convenience helper for switching to standards mode."""

        return self.set_mode(GUIMode.STANDARDS, standard=standard)

    def set_theme(self, theme: str) -> ModeState:
        """Update the preferred theme token."""

        normalized = theme.strip().lower()
        if normalized not in {"dark", "light", "system"}:
            raise ValueError(f"Unsupported theme {theme!r}")
        return self._publish(
            ModeState(
                mode=self._state.mode,
                standard=self._state.standard,
                theme=normalized,
            )
        )

    def describe(self) -> dict[str, str | None]:
        """Return a small serializable description of the current state."""

        return {
            "mode": self._state.mode.value,
            "standard": self._state.standard,
            "theme": self._state.theme,
        }
