"""Mode management for Simple, Expert, and Standards workflows."""

from __future__ import annotations

import json
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
    theme_profile: str | None = None


ModeListener = Callable[[ModeState], None]


class ModeManager:
    """Pure-Python mode manager with persistence and subscriber hooks."""

    MODE_KEY = "gui/mode"
    STANDARD_KEY = "gui/standard"
    THEME_KEY = "gui/theme"
    THEME_PROFILE_KEY = "gui/theme_profile"
    THEME_PROFILES_KEY = "gui/theme_profiles"

    DEFAULT_THEME_PROFILES = {
        "night-lab": "dark",
        "paper-light": "light",
        "system-default": "system",
    }

    def __init__(
        self,
        initial_state: ModeState | None = None,
        settings=None,
    ) -> None:
        self._settings = settings
        self._theme_profiles = dict(self.DEFAULT_THEME_PROFILES)
        self._load_theme_profiles()
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

        theme_profile = self._settings_value(self.THEME_PROFILE_KEY, None)
        theme_profile = str(theme_profile).strip() if theme_profile else None
        if theme_profile and theme_profile in self._theme_profiles:
            theme = self._theme_profiles[theme_profile]
        elif theme_profile:
            theme_profile = None

        mode_token = str(self._settings_value(self.MODE_KEY, GUIMode.EXPERT.value)).strip().lower()
        try:
            mode = GUIMode(mode_token)
        except ValueError:
            mode = GUIMode.EXPERT

        standard = self._settings_value(self.STANDARD_KEY, None)
        standard = standard or None
        if mode is not GUIMode.STANDARDS:
            standard = None

        return ModeState(
            mode=mode,
            standard=standard,
            theme=theme,
            theme_profile=theme_profile,
        )

    def _save_state(self, state: ModeState) -> None:
        """Persist the current state when a settings backend is available."""

        if self._settings is None:
            return

        self._settings_set_value(self.MODE_KEY, state.mode.value)
        self._settings_set_value(self.STANDARD_KEY, state.standard or "")
        self._settings_set_value(self.THEME_KEY, state.theme)
        self._settings_set_value(self.THEME_PROFILE_KEY, state.theme_profile or "")
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()

    def _load_theme_profiles(self) -> None:
        if self._settings is None:
            return
        raw_profiles = self._settings_value(self.THEME_PROFILES_KEY, "")
        if not raw_profiles:
            return
        try:
            parsed = json.loads(str(raw_profiles))
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(parsed, dict):
            return
        merged = dict(self.DEFAULT_THEME_PROFILES)
        for name, theme in parsed.items():
            key = str(name).strip()
            value = str(theme).strip().lower()
            if key and value in {"dark", "light", "system"}:
                merged[key] = value
        self._theme_profiles = merged

    def _save_theme_profiles(self) -> None:
        if self._settings is None:
            return
        self._settings_set_value(
            self.THEME_PROFILES_KEY,
            json.dumps(self._theme_profiles, sort_keys=True),
        )
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
                theme_profile=self._state.theme_profile,
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
        profile = self._state.theme_profile
        if profile and self._theme_profiles.get(profile) != normalized:
            profile = None
        return self._publish(
            ModeState(
                mode=self._state.mode,
                standard=self._state.standard,
                theme=normalized,
                theme_profile=profile,
            )
        )

    def apply_state(self, state: ModeState | dict[str, object]) -> ModeState:
        """Replace the current mode state from a serialized payload."""

        if isinstance(state, ModeState):
            next_state = state
        else:
            mode_token = str(state.get("mode") or self._state.mode.value).strip().lower()
            try:
                mode = GUIMode(mode_token)
            except ValueError:
                mode = self._state.mode
            standard = state.get("standard")
            standard = str(standard).strip() if standard else None
            if mode is not GUIMode.STANDARDS:
                standard = None
            theme_profile = state.get("theme_profile")
            theme_profile = str(theme_profile).strip() if theme_profile else None
            if theme_profile and theme_profile not in self._theme_profiles:
                theme_profile = None
            theme = (
                self._theme_profiles.get(theme_profile)
                if theme_profile
                else str(state.get("theme") or self._state.theme).strip().lower()
            )
            if theme not in {"dark", "light", "system"}:
                theme = self._state.theme
            next_state = ModeState(
                mode=mode,
                standard=standard,
                theme=theme,
                theme_profile=theme_profile,
            )
        return self._publish(next_state)

    def available_theme_profiles(self) -> tuple[str, ...]:
        """Return all known profile names in stable order."""

        return tuple(self._theme_profiles.keys())

    def theme_for_profile(self, profile_name: str) -> str | None:
        """Resolve a profile name to a theme token."""

        return self._theme_profiles.get(profile_name)

    def save_theme_profile(self, profile_name: str, *, theme: str | None = None) -> None:
        """Persist a named profile that maps to one of the known theme tokens."""

        name = profile_name.strip()
        if not name:
            raise ValueError("Theme profile name cannot be empty")
        resolved_theme = (theme or self._state.theme).strip().lower()
        if resolved_theme not in {"dark", "light", "system"}:
            raise ValueError(f"Unsupported theme {resolved_theme!r}")
        self._theme_profiles[name] = resolved_theme
        self._save_theme_profiles()

    def set_theme_profile(self, profile_name: str) -> ModeState:
        """Switch to the theme represented by a saved profile."""

        if profile_name not in self._theme_profiles:
            raise ValueError(f"Unknown theme profile {profile_name!r}")
        return self._publish(
            ModeState(
                mode=self._state.mode,
                standard=self._state.standard,
                theme=self._theme_profiles[profile_name],
                theme_profile=profile_name,
            )
        )

    def describe(self) -> dict[str, str | None]:
        """Return a small serializable description of the current state."""

        return {
            "mode": self._state.mode.value,
            "standard": self._state.standard,
            "theme": self._state.theme,
            "theme_profile": self._state.theme_profile,
        }
