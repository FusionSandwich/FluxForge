"""Spectrum canvas abstractions and shared renderer-side data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.core.workspace_document import CanvasViewport


@dataclass(frozen=True)
class RendererCapabilities:
    """Declares which interaction surfaces a backend supports."""

    supports_crosshair: bool = True
    supports_context_menus: bool = True
    supports_overlay_layers: bool = True
    supports_y_zoom: bool = True
    supports_incremental_updates: bool = True
    supports_multi_spectrum_tabs: bool = True
    supports_hierarchical_buffer: bool = True


@dataclass(frozen=True)
class SpectrumTrace:
    """A single spectrum-like trace shown on the canvas."""

    label: str
    counts: tuple[float, ...]
    channels: tuple[float, ...]
    color: str = "#72d6ff"
    visible: bool = True
    x_axis_label: str = "Channel"


@dataclass(frozen=True)
class ReferenceLine:
    """Line overlay metadata for nuclide or standards references."""

    energy_keV: float
    label: str = ""
    color: str = "#f59e0b"


@dataclass(frozen=True)
class HierarchicalSpectrumLevel:
    """One pre-computed resolution level for a spectrum."""

    stride: int
    counts: tuple[float, ...]

    @property
    def sample_count(self) -> int:
        return len(self.counts)


@dataclass(frozen=True)
class HierarchicalSpectrumBuffer:
    """Multi-resolution spectrum representation for fluid zooming."""

    full_resolution: tuple[float, ...]
    levels: tuple[HierarchicalSpectrumLevel, ...]

    @classmethod
    def from_counts(
        cls,
        counts: Sequence[float],
        *,
        max_levels: int = 5,
    ) -> "HierarchicalSpectrumBuffer":
        """Build a multi-resolution pyramid from raw spectrum counts."""

        normalized = tuple(float(value) for value in counts)
        levels = [HierarchicalSpectrumLevel(stride=1, counts=normalized)]
        current = normalized
        stride = 1

        while len(levels) < max_levels and len(current) > 1:
            stride *= 2
            current = _downsample_counts(current, factor=2)
            levels.append(HierarchicalSpectrumLevel(stride=stride, counts=current))

        return cls(full_resolution=normalized, levels=tuple(levels))

    def choose_level(self, pixel_width: int) -> HierarchicalSpectrumLevel:
        """Choose the lowest-cost level that still exceeds the viewport density."""

        target = max(int(pixel_width), 1) * 2
        for level in self.levels:
            if level.sample_count <= target:
                return level
        return self.levels[-1]

    def describe(self) -> dict[str, object]:
        """Return a small serializable summary of the hierarchy."""

        return {
            "channel_count": len(self.full_resolution),
            "level_count": len(self.levels),
            "strides": [level.stride for level in self.levels],
            "sample_counts": [level.sample_count for level in self.levels],
        }


class SpectrumCanvas:
    """Abstract rendering surface used by the planned GUI."""

    backend_key: str
    capabilities: RendererCapabilities

    def set_spectrum(self, counts: Sequence[float]) -> None:
        """Load the primary spectrum."""
        raise NotImplementedError

    def set_reference_lines(self, energies_keV: Sequence[float]) -> None:
        """Load reference-line overlays."""
        raise NotImplementedError

    def set_annotation_lines(self, lines: Sequence[ReferenceLine]) -> None:
        """Optional bulk update for labeled analytical overlays."""

        del lines

    def clear(self) -> None:
        """Reset the canvas state."""
        raise NotImplementedError

    def set_traces(self, traces: Sequence[SpectrumTrace]) -> None:
        """Optional bulk update for primary and overlay traces."""

        if not traces:
            self.clear()
            return
        self.set_spectrum(traces[0].counts)

    def set_peak_candidates(self, peaks: Sequence[PeakCandidate]) -> None:
        """Optional bulk update for detected peak overlays."""

        del peaks

    def set_cascade_sum_lines(self, energies_keV: Sequence[float]) -> None:
        """Optional bulk update for cascade-sum overlays."""

        del energies_keV

    def set_peak_residuals(
        self, peaks: Sequence[PeakCandidate], *, visible: bool
    ) -> None:
        """Optional update for mini residual subplots."""

        del peaks
        del visible

    def viewport_state(
        self,
        *,
        viewport_id: str = "primary-spectrum",
        spectrum_id: str | None = None,
        selected_roi_id: str | None = None,
    ) -> CanvasViewport:
        """Return renderer-independent persisted view state."""

        raise NotImplementedError

    def apply_viewport_state(self, viewport: CanvasViewport) -> None:
        """Restore renderer-independent persisted view state."""

        raise NotImplementedError


def _downsample_counts(counts: Sequence[float], *, factor: int) -> tuple[float, ...]:
    """Reduce a spectrum by taking the bucket maximum at a lower resolution."""

    if factor <= 1:
        return tuple(float(value) for value in counts)

    reduced = []
    for offset in range(0, len(counts), factor):
        bucket = counts[offset : offset + factor]
        if bucket:
            reduced.append(max(float(value) for value in bucket))
    return tuple(reduced)
