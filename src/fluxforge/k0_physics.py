"""Shared k0 physics helpers.

This module centralizes small, pure functions reused by multiple k0 workflows
(analysis and TRIGA-specific wrappers) so equations are implemented once.
"""

from __future__ import annotations

import math
from typing import Tuple


def calculate_q0_alpha(
    q0: float,
    alpha: float,
    e_res_ev: float,
    e_cd_ev: float = 0.55,
) -> float:
    """Return alpha-corrected Q0 for epithermal spectra."""
    if alpha == 0.0 or e_res_ev <= 0.0:
        return q0
    return (q0 - 0.429) / (e_res_ev**alpha) + 0.429 / (
        (2.0 * alpha + 1.0) * (e_cd_ev**alpha)
    )


def saturation_factor(decay_const_s: float, irradiation_time_s: float) -> float:
    """Saturation factor S = 1 - exp(-lambda * t_irr)."""
    return 1.0 - math.exp(-decay_const_s * irradiation_time_s)


def decay_factor(decay_const_s: float, decay_time_s: float) -> float:
    """Decay factor D = exp(-lambda * t_decay)."""
    return math.exp(-decay_const_s * decay_time_s)


def counting_factor(decay_const_s: float, counting_time_s: float) -> float:
    """Counting factor C = (1 - exp(-lambda * t_count)) / (lambda * t_count)."""
    if decay_const_s * counting_time_s < 1.0e-6:
        return 1.0
    return (1.0 - math.exp(-decay_const_s * counting_time_s)) / (
        decay_const_s * counting_time_s
    )


def sdc_factor(
    half_life_s: float,
    irradiation_time_s: float,
    decay_time_s: float,
    counting_time_s: float,
) -> Tuple[float, float, float, float]:
    """Return (S, D, C, SDC) from half-life and timing metadata."""
    if half_life_s <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    decay_const_s = math.log(2.0) / half_life_s
    sat = saturation_factor(decay_const_s, irradiation_time_s)
    dec = decay_factor(decay_const_s, decay_time_s)
    cnt = counting_factor(decay_const_s, counting_time_s)
    return sat, dec, cnt, sat * dec * cnt
