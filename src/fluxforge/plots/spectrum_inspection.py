"""Headless-safe calibrated gamma-spectrum inspection plots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional at import time
    plt = None

from fluxforge.io.spe import GammaSpectrum


def _spectrum_energies(spectrum: GammaSpectrum) -> np.ndarray:
    if spectrum.energies is not None:
        return np.asarray(spectrum.energies, dtype=float)
    if spectrum.calibration:
        return np.asarray(spectrum.calibrate_channels(), dtype=float)
    return np.asarray(spectrum.channels, dtype=float)


def _region_bounds_in_energy(
    spectrum: GammaSpectrum,
    region: Dict[str, Any],
) -> Tuple[float, float]:
    energies = _spectrum_energies(spectrum)
    if region.get("left_keV") is not None and region.get("right_keV") is not None:
        lo = float(region["left_keV"])
        hi = float(region["right_keV"])
    elif (
        region.get("left_channel") is not None
        and region.get("right_channel") is not None
    ):
        lo = float(energies[int(round(float(region["left_channel"])))])
        hi = float(energies[int(round(float(region["right_channel"])))])
    else:
        raise ValueError(
            "Manual peak regions require left/right channels or left/right energies."
        )
    return tuple(sorted((lo, hi)))


def plot_gamma_spectrum(
    spectrum: GammaSpectrum,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    manual_regions: Optional[List[Dict[str, Any]]] = None,
    x_min_keV: Optional[float] = None,
    x_max_keV: Optional[float] = None,
    y_log: bool = False,
    figsize: Tuple[float, float] = (12.0, 6.0),
):
    """Plot counts vs energy with optional manual ROI overlays."""
    if plt is None:
        raise ImportError("matplotlib required for plotting")

    energies = _spectrum_energies(spectrum)
    counts = np.asarray(spectrum.counts, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(energies, counts, linewidth=1.0, color="#1f4e79")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if y_log:
        ax.set_yscale("log")
    if x_min_keV is not None or x_max_keV is not None:
        ax.set_xlim(left=x_min_keV, right=x_max_keV)

    if title:
        ax.set_title(title)
    if subtitle:
        ax.text(
            0.01,
            0.98,
            subtitle,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    if manual_regions:
        y_max = float(np.nanmax(counts)) if counts.size else 0.0
        label_y = y_max * 0.92 if y_max > 0.0 else 1.0
        for idx, region in enumerate(manual_regions):
            lo_keV, hi_keV = _region_bounds_in_energy(spectrum, region)
            label = (
                region.get("label")
                or region.get("name")
                or region.get("isotope")
                or f"manual_peak_{idx + 1}"
            )
            ax.axvspan(lo_keV, hi_keV, color="#c44e52", alpha=0.12)
            ax.axvline(lo_keV, color="#c44e52", alpha=0.35, linewidth=0.9)
            ax.axvline(hi_keV, color="#c44e52", alpha=0.35, linewidth=0.9)
            ax.text(
                (lo_keV + hi_keV) * 0.5,
                label_y,
                str(label),
                rotation=90,
                va="top",
                ha="center",
                fontsize=8,
                color="#7a1f24",
            )

    fig.tight_layout()
    return fig, ax
