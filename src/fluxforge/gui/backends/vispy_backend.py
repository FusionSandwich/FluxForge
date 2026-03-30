"""Optional Vispy renderer stub for the next-generation GUI shell."""

from __future__ import annotations

from typing import Sequence

from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.spectrum_canvas import RendererCapabilities, SpectrumCanvas


VISPY_AVAILABLE = False
VISPY_IMPORT_ERROR = QT_IMPORT_ERROR

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    try:
        import vispy  # noqa: F401

        VISPY_AVAILABLE = True
        VISPY_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - optional dependency branch
        VISPY_IMPORT_ERROR = exc


def vispy_backend_status() -> dict[str, object]:
    """Return import-safe availability metadata for the Vispy backend."""

    reason = None
    if VISPY_IMPORT_ERROR is not None:
        reason = f"{type(VISPY_IMPORT_ERROR).__name__}: {VISPY_IMPORT_ERROR}"
    return {
        "key": "vispy",
        "display_name": "Vispy",
        "recommended": False,
        "available": VISPY_AVAILABLE,
        "reason": reason,
    }


class VispySpectrumCanvas(SpectrumCanvas):
    """Future renderer stub kept additive alongside the PyQtGraph backend."""

    backend_key = "vispy"
    capabilities = RendererCapabilities(
        supports_context_menus=False,
    )

    def set_spectrum(self, counts: Sequence[float]) -> None:
        del counts
        raise RuntimeError(
            "Vispy renderer is a Phase 1 stub and is not yet implemented in this workspace."
        )

    def set_reference_lines(self, energies_keV: Sequence[float]) -> None:
        del energies_keV
        raise RuntimeError(
            "Vispy renderer is a Phase 1 stub and is not yet implemented in this workspace."
        )

    def clear(self) -> None:
        raise RuntimeError(
            "Vispy renderer is a Phase 1 stub and is not yet implemented in this workspace."
        )
