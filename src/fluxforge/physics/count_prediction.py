"""Forward activation-to-native-count operators with explicit clock semantics."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from fluxforge.physics.activation import count_decay_factor, irradiation_buildup_factor


def _nonnegative(values, name):
    value = np.asarray(values, dtype=float)
    if value.size == 0 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite, nonempty and nonnegative")
    return value


def gaussian_bin_probabilities(energy_edges, line_energies, sigma):
    """Line x native-bin probabilities, integrated over supplied bin edges.

    Probability outside the supplied window remains outside; no renormalization
    to an ROI or midpoint-density approximation is applied.
    """
    edges = _nonnegative(energy_edges, "energy_edges")
    centers = _nonnegative(line_energies, "line_energies")
    widths = np.asarray(sigma, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges must be a strictly increasing vector")
    if centers.ndim != 1 or widths.shape != centers.shape:
        raise ValueError("line_energies and sigma must be matching vectors")
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    cdf = ndtr((edges[None, :] - centers[:, None]) / widths[:, None])
    return np.diff(cdf, axis=1)


def activation_count_operator(
    rate_response,
    *,
    target_atoms,
    half_life_s,
    irradiation_segments,
    cooling_time_s,
    real_time_s,
    live_time_s,
    emission_probabilities,
    efficiencies,
    line_bin_probabilities,
):
    """Return bins x groups mapping integrated flux to accepted line counts.

    One directly produced radioactive isotope, no feeding/depletion/burnup.
    Response is cm² per target atom; target_atoms includes isotopic abundance.
    Time-varying exposure is represented by existing IrradiationSegment objects.
    Decay runs on REAL time. Live/real acceptance is applied exactly once.
    Efficiencies include declared geometry/attenuation/coincidence conventions.
    Add other isotopes' operators before folding a shared flux draw.
    """
    response = _nonnegative(rate_response, "rate_response")
    yields = _nonnegative(emission_probabilities, "emission_probabilities")
    efficiency = _nonnegative(efficiencies, "efficiencies")
    probabilities = _nonnegative(line_bin_probabilities, "line_bin_probabilities")
    if response.ndim != 1 or yields.ndim != 1 or efficiency.shape != yields.shape:
        raise ValueError(
            "response and matched line yields/efficiencies must be vectors"
        )
    if probabilities.ndim != 2 or probabilities.shape[0] != yields.size:
        raise ValueError("line_bin_probabilities must have one row per gamma line")
    # Gamma yields are photons/decay and can exceed one (e.g. annihilation).
    if np.any(efficiency > 1):
        raise ValueError("efficiencies must be fractions")
    if np.any(probabilities.sum(axis=1) > 1 + 1e-12):
        raise ValueError("line-bin probabilities may not exceed unit probability")
    for name, value in (
        ("target_atoms", target_atoms),
        ("half_life_s", half_life_s),
        ("real_time_s", real_time_s),
        ("live_time_s", live_time_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not math.isfinite(cooling_time_s) or cooling_time_s < 0:
        raise ValueError("cooling_time_s must be finite and nonnegative")
    if live_time_s > real_time_s:
        raise ValueError("live_time_s cannot exceed real_time_s")
    buildup = irradiation_buildup_factor(irradiation_segments, half_life_s)
    if buildup <= 0:
        raise ValueError("irradiation must have a positive production buildup")
    clock = (
        target_atoms
        * buildup
        * math.exp(-math.log(2) * cooling_time_s / half_life_s)
        * live_time_s
        * count_decay_factor(half_life_s, real_time_s)
    )
    operator = np.outer((yields * efficiency) @ probabilities, response) * clock
    if not np.isfinite(operator).all():
        raise ValueError("count operator is not finite")
    return operator


def expected_count_draws(flux_draws, operator, *, additive_counts=0):
    """Fold JOINT posterior draws of flux/operator/continuum into count means.

    operator may be fixed (bins, groups) or draw-aligned (draws, bins, groups).
    additive_counts is scalar or broadcastable to (draws, bins), in accepted
    counts. Shared nuisance parameters must use the same draw index. Sampling
    their prior independently after fitting is not generally posterior prediction.
    """
    flux = _nonnegative(flux_draws, "flux_draws")
    matrix = _nonnegative(operator, "operator")
    additive = _nonnegative(additive_counts, "additive_counts")
    if flux.ndim != 2:
        raise ValueError("flux_draws must have shape (draws, groups)")
    if matrix.ndim == 2 and matrix.shape[1] == flux.shape[1]:
        means = flux @ matrix.T
    elif (
        matrix.ndim == 3
        and matrix.shape[0] == flux.shape[0]
        and matrix.shape[2] == flux.shape[1]
    ):
        means = np.einsum("dg,dbg->db", flux, matrix)
    else:
        raise ValueError("operator must match the flux draw and group dimensions")
    try:
        means = means + np.broadcast_to(additive, means.shape)
    except ValueError as exc:
        raise ValueError("additive_counts must broadcast to draws x bins") from exc
    if not np.isfinite(means).all():
        raise ValueError("expected counts must be finite")
    return means
