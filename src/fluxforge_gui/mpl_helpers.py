"""Matplotlib helpers shared by FluxForge GUI render/export code."""

from __future__ import annotations

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    FigureCanvasAgg = None
    Figure = None


def ensure_agg_renderer(figure):
    """Attach an Agg canvas for figures that would otherwise use a fallback."""

    if figure is None or FigureCanvasAgg is None:
        return figure
    canvas = getattr(figure, "canvas", None)
    if canvas is None or not hasattr(canvas, "get_renderer"):
        FigureCanvasAgg(figure)
    return figure


def create_offscreen_figure(
    *,
    figsize: tuple[float, float],
    dpi: int = 100,
    figure=None,
):
    """Return a figure that is safe for off-screen layout and export helpers."""

    if figure is None:
        if Figure is None:
            return None
        figure = Figure(figsize=figsize, dpi=dpi)
    return ensure_agg_renderer(figure)


def apply_tight_layout(figure) -> None:
    """Run tight_layout with a stable renderer-backed canvas."""

    ensure_agg_renderer(figure)
    figure.tight_layout()


__all__ = [
    "apply_tight_layout",
    "create_offscreen_figure",
    "ensure_agg_renderer",
]
