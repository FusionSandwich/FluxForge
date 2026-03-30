"""Theme helpers for the next-generation FluxForge GUI shell."""

from __future__ import annotations

from pathlib import Path

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.themes.color_tokens import THEME_TOKENS

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import QGuiApplication, Qt


THEME_DIR = Path(__file__).with_name("themes")
THEME_FILES = {
    "dark": THEME_DIR / "dark.qss",
    "light": THEME_DIR / "light.qss",
}


def available_themes() -> tuple[str, ...]:
    """Return the supported theme tokens."""

    return ("dark", "light", "system")


def resolve_theme(theme: str) -> str:
    """Resolve `system` into a concrete theme token."""

    normalized = theme.strip().lower()
    if normalized not in available_themes():
        raise ValueError(f"Unsupported theme {theme!r}")
    if normalized != "system":
        return normalized
    if not QT_AVAILABLE:
        return "dark"
    app = QGuiApplication.instance()
    if app is None:
        return "dark"
    style_hints = app.styleHints()
    try:
        color_scheme = style_hints.colorScheme()
    except Exception:
        return "dark"
    return "light" if color_scheme == Qt.ColorScheme.Light else "dark"


def load_stylesheet(theme: str) -> str:
    """Return the QSS stylesheet text for the requested theme."""

    concrete = resolve_theme(theme)
    return THEME_FILES[concrete].read_text(encoding="utf-8")


def theme_tokens(theme: str) -> dict[str, str]:
    """Return the color-token map for the resolved theme."""

    return THEME_TOKENS[resolve_theme(theme)]
