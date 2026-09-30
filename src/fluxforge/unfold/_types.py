"""
Shared data-class types for the unfold subpackage.

These mirror the FluxForge domain concepts but are kept here so the
unfold modules are self-contained and testable independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ReactionRates:
    """Measured reaction-rate (SigPhi) data from activation foils/wires.

    Attributes
    ----------
    values : np.ndarray
        Measured reaction rates, shape (N_monitors,).
    uncertainties : np.ndarray
        Absolute 1-sigma uncertainties, same shape.
    quantity : str
        Defaults to activation rates. PyUnfold requires explicit
        ``effect_counts``; this marker does not convert rates to counts.
    """

    values: np.ndarray  # Shape (N_monitors,)
    uncertainties: np.ndarray  # Shape (N_monitors,)
    quantity: str = "activation_rate"


@dataclass
class ResponseBundle:
    """Cross-section response matrix bundled with energy grid.

    Attributes
    ----------
    matrix : np.ndarray
        Response / mixing matrix, shape (N_monitors, N_groups).
    energy_bins : np.ndarray
        Energy group boundaries in eV, shape (N_groups + 1,).
    quantity : str
        Defaults to dimensionful activation response. PyUnfold requires
        ``conditional_probability``; this marker does not normalize the matrix.
    """

    matrix: np.ndarray  # Shape (N_monitors, N_groups)
    energy_bins: np.ndarray  # Shape (N_groups + 1,)
    quantity: str = "activation_rate_response"


@dataclass
class SpectrumFile:
    """Simple container for a gamma spectrum (counts per channel).

    Attributes
    ----------
    counts : np.ndarray
        Channel counts, shape (N_channels,).
    live_time_s : float, optional
        Live time in seconds.
    """

    counts: np.ndarray  # Shape (N_channels,)
    live_time_s: Optional[float] = None
