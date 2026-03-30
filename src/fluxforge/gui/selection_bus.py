"""Selection-bus scaffold for coordinated spectrum views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fluxforge.gui.spectrum_canvas import ReferenceLine


@dataclass(frozen=True)
class SelectionState:
    """Shared selection state between coordinated GUI surfaces."""

    peak_energy_keV: float | None = None
    roi_bounds_keV: tuple[float, float] | None = None
    nuclide: str | None = None
    reference_lines_keV: tuple[float, ...] = ()
    annotation_lines: tuple[ReferenceLine, ...] = ()


SelectionListener = Callable[[SelectionState], None]


class SelectionBus:
    """Simple publish/subscribe state container for GUI scaffolding."""

    _shared_instance: "SelectionBus | None" = None

    def __init__(self) -> None:
        self._state = SelectionState()
        self._listeners: list[SelectionListener] = []

    @classmethod
    def shared(cls) -> "SelectionBus":
        """Return the shared selection bus instance."""

        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    @property
    def state(self) -> SelectionState:
        return self._state

    def subscribe(self, listener: SelectionListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: SelectionListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def publish(self, state: SelectionState) -> SelectionState:
        self._state = state
        for listener in tuple(self._listeners):
            listener(state)
        return state

    def publish_peak(
        self,
        peak_energy_keV: float,
        *,
        nuclide: str | None = None,
    ) -> SelectionState:
        """Publish a peak-centric selection update."""

        return self.publish(
            SelectionState(
                peak_energy_keV=float(peak_energy_keV),
                roi_bounds_keV=self._state.roi_bounds_keV,
                nuclide=nuclide or self._state.nuclide,
                reference_lines_keV=self._state.reference_lines_keV,
                annotation_lines=self._state.annotation_lines,
            )
        )

    def publish_roi(self, start_keV: float, end_keV: float) -> SelectionState:
        """Publish a region-of-interest selection update."""

        lower, upper = sorted((float(start_keV), float(end_keV)))
        return self.publish(
            SelectionState(
                peak_energy_keV=self._state.peak_energy_keV,
                roi_bounds_keV=(lower, upper),
                nuclide=self._state.nuclide,
                reference_lines_keV=self._state.reference_lines_keV,
                annotation_lines=self._state.annotation_lines,
            )
        )

    def publish_nuclide(
        self,
        nuclide: str,
        *,
        reference_lines_keV: tuple[float, ...] = (),
        annotation_lines: tuple[ReferenceLine, ...] = (),
    ) -> SelectionState:
        """Publish a nuclide-centric selection update."""

        return self.publish(
            SelectionState(
                peak_energy_keV=self._state.peak_energy_keV,
                roi_bounds_keV=self._state.roi_bounds_keV,
                nuclide=nuclide,
                reference_lines_keV=reference_lines_keV,
                annotation_lines=annotation_lines,
            )
        )

    def clear(self) -> SelectionState:
        """Reset the shared selection state."""

        return self.publish(SelectionState())

    def describe(self) -> dict[str, object]:
        """Return a small serializable description of the current selection."""

        return {
            "peak_energy_keV": self._state.peak_energy_keV,
            "roi_bounds_keV": self._state.roi_bounds_keV,
            "nuclide": self._state.nuclide,
            "reference_lines_keV": self._state.reference_lines_keV,
            "annotation_line_count": len(self._state.annotation_lines),
            "listener_count": len(self._listeners),
        }
