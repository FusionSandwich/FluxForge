"""Renderer backend exports for the next-generation GUI shell."""

from __future__ import annotations

from fluxforge.gui.backends.pyqtgraph_backend import (
    PYQTGRAPH_AVAILABLE,
    PYQTGRAPH_IMPORT_ERROR,
    PyQtGraphSpectrumCanvas,
    pyqtgraph_backend_status,
)
from fluxforge.gui.backends.vispy_backend import (
    VISPY_AVAILABLE,
    VISPY_IMPORT_ERROR,
    VispySpectrumCanvas,
    vispy_backend_status,
)
from fluxforge.plugins import PluginRegistries, bootstrap_builtin_registries


def available_renderer_status() -> tuple[dict[str, object], ...]:
    """Return import-safe backend availability metadata."""

    return (
        pyqtgraph_backend_status(),
        vispy_backend_status(),
    )


def register_builtin_render_backends(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in renderer backends in a registry container."""

    registries.render_backends.clear()
    registries.render_backends.register(
        "pyqtgraph",
        PyQtGraphSpectrumCanvas,
        description="Primary production renderer for the native Qt shell.",
        recommended=True,
        tags=("qt", "production", "phase1"),
        set_default=True,
    )
    registries.render_backends.register(
        "vispy",
        VispySpectrumCanvas,
        description="Optional high-performance renderer stub for future expansion.",
        tags=("qt", "optional", "phase1"),
    )
    return registries


__all__ = [
    "PYQTGRAPH_AVAILABLE",
    "PYQTGRAPH_IMPORT_ERROR",
    "PyQtGraphSpectrumCanvas",
    "VISPY_AVAILABLE",
    "VISPY_IMPORT_ERROR",
    "VispySpectrumCanvas",
    "available_renderer_status",
    "pyqtgraph_backend_status",
    "register_builtin_render_backends",
    "vispy_backend_status",
]


_shared_registries = bootstrap_builtin_registries()
if len(_shared_registries.render_backends) == 0:
    register_builtin_render_backends(_shared_registries)
