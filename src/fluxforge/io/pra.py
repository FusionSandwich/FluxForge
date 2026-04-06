"""
PRA Histogram Reader.

Parses ASCII pulse-height histogram exports from PRA-like tools.
The file format is typically two columns: channel (or height) and counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum


@dataclass
class PRAHistogram:
    """Parsed PRA histogram data."""

    channels: np.ndarray
    counts: np.ndarray
    live_time_s: float
    header: Optional[str] = None


def _parse_pra_lines(
    lines: Iterable[str],
) -> Tuple[List[float], List[float], Optional[str]]:
    channels: List[float] = []
    counts: List[float] = []
    header: Optional[str] = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            # Header or malformed line.
            if idx == 0:
                header = stripped
            continue
        try:
            ch = float(parts[0])
            cnt = float(parts[1])
        except ValueError:
            if idx == 0:
                header = stripped
            continue
        channels.append(ch)
        counts.append(cnt)

    if not channels:
        raise ValueError("No numeric channel/count data found in PRA histogram.")

    return channels, counts, header


def read_pra_histogram(
    path: Union[str, Path],
    live_time_s: float,
) -> PRAHistogram:
    """
    Read a PRA ASCII histogram file.

    Parameters
    ----------
    path : str or Path
        Histogram file path.
    live_time_s : float
        Acquisition live time in seconds.

    Returns
    -------
    PRAHistogram
        Parsed histogram data.
    """
    path = Path(path)
    if live_time_s <= 0:
        raise ValueError("live_time_s must be positive.")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    channels, counts, header = _parse_pra_lines(lines)
    return PRAHistogram(
        channels=np.array(channels, dtype=float),
        counts=np.array(counts, dtype=float),
        live_time_s=float(live_time_s),
        header=header,
    )


def pra_to_gamma_spectrum(
    histogram: PRAHistogram,
    spectrum_id: Optional[str] = None,
) -> GammaSpectrum:
    """
    Convert PRAHistogram to GammaSpectrum.

    Counts are retained as raw counts; rate can be computed from live_time.
    """
    spectrum_id = spectrum_id or "pra_histogram"
    return GammaSpectrum(
        counts=histogram.counts,
        channels=histogram.channels,
        live_time=histogram.live_time_s,
        real_time=histogram.live_time_s,
        spectrum_id=spectrum_id,
        metadata={
            "format": "pra_histogram",
            "header": histogram.header or "",
        },
    )


def read_pra_as_spectrum(
    path: Union[str, Path],
    live_time_s: float,
) -> GammaSpectrum:
    """
    Convenience wrapper to read a PRA histogram as GammaSpectrum.
    """
    histogram = read_pra_histogram(path, live_time_s)
    return pra_to_gamma_spectrum(histogram, spectrum_id=Path(path).stem)
