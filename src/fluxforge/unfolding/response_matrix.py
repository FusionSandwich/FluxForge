"""Response-matrix loading helpers for the unfolding workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fluxforge.solvers.rmle import create_gaussian_response_matrix


@dataclass(frozen=True)
class LoadedResponseMatrix:
    """Loaded or synthesized response matrix payload."""

    matrix: np.ndarray
    energy_edges: np.ndarray
    source_label: str
    source_format: str
    metadata: dict[str, str] = field(default_factory=dict)


def _detect_delimiter(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if "\t" in first_line:
        return "\t"
    return ","


def load_response_matrix(
    path: str | Path,
    *,
    source_format: str = "auto",
    energy_edges: np.ndarray | None = None,
) -> LoadedResponseMatrix:
    """Load a response matrix from a CSV or tab-delimited file."""

    resolved = Path(path)
    if source_format == "auto":
        source_format = "tab_delimited" if resolved.suffix.lower() in {".tsv", ".txt"} else "user_csv"
    delimiter = _detect_delimiter(resolved)
    matrix = np.loadtxt(resolved, delimiter=delimiter, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("response matrix must be a 2-D table")
    edges = (
        np.asarray(energy_edges, dtype=float)
        if energy_edges is not None
        else np.linspace(0.01, float(matrix.shape[1]), matrix.shape[1] + 1, dtype=float)
    )
    return LoadedResponseMatrix(
        matrix=np.asarray(matrix, dtype=float),
        energy_edges=edges,
        source_label=resolved.name,
        source_format=source_format,
        metadata={"path": str(resolved)},
    )


def build_analytical_hpge_response(
    *,
    n_channels: int,
    energy_edges: np.ndarray,
    energy_range_keV: tuple[float, float] = (0.0, 3000.0),
    fwhm_at_661_keV: float = 1.8,
    efficiency_scale: float = 0.9,
) -> LoadedResponseMatrix:
    """Build an analytical HPGe response matrix for the unfolding dialog."""

    n_bins = int(np.asarray(energy_edges, dtype=float).size - 1)

    def fwhm_function(energy_keV: float) -> float:
        reference = max(float(energy_keV), 1.0) / 661.0
        return max(fwhm_at_661_keV * np.sqrt(reference), 0.75)

    def efficiency_function(energy_keV: float) -> float:
        normalized = max(float(energy_keV), 1.0) / max(energy_range_keV[1], 1.0)
        return max(efficiency_scale * (1.0 - 0.35 * normalized), 0.05)

    response = create_gaussian_response_matrix(
        n_channels=n_channels,
        n_energy_bins=n_bins,
        fwhm_function=fwhm_function,
        efficiency_function=efficiency_function,
        energy_range=energy_range_keV,
    )
    return LoadedResponseMatrix(
        matrix=np.asarray(response.matrix, dtype=float),
        energy_edges=np.asarray(energy_edges, dtype=float),
        source_label="Analytical HPGe",
        source_format="analytical_hpge",
        metadata={
            "fwhm_at_661_keV": f"{float(fwhm_at_661_keV):.3f}",
            "efficiency_scale": f"{float(efficiency_scale):.3f}",
        },
    )


__all__ = [
    "LoadedResponseMatrix",
    "build_analytical_hpge_response",
    "load_response_matrix",
]
