"""Cross-energy detector-efficiency covariance for measured curve fits."""

import numpy as np

from fluxforge.core.probability import covariance_matrix


def efficiency_evaluation_covariance(
    curve, energies_keV, *, systematic_correlation=None
):
    """Return absolute efficiency covariance, conditional on an applicable fit.

    Supports log-polynomial and Gray fit_covariance curves. Missing covariance
    is unavailable and raises; no historical default 5% is substituted.
    A nonzero relative_systematic requires an explicit 'shared' or 'independent'
    cross-energy correlation convention. Unknown correlations are not inferred.
    Extrapolation outside the declared calibration interval is rejected.
    """
    energy = np.asarray(energies_keV, dtype=float)
    if (
        energy.ndim != 1
        or not energy.size
        or not np.isfinite(energy).all()
        or np.any(energy <= 0)
    ):
        raise ValueError("energies_keV must be a finite positive vector")
    if np.any(energy < curve.energy_range[0]) or np.any(energy > curve.energy_range[1]):
        raise ValueError(
            "efficiency covariance cannot extrapolate beyond calibration range"
        )
    uncertainty = curve.uncertainty_model
    if not uncertainty or uncertainty.get("type") != "fit_covariance":
        raise ValueError("a declared measured-fit parameter covariance is required")
    if curve.model_type == "polynomial":
        n = len(curve.parameters.get("coefficients", ()))
        names = tuple(f"a{i}" for i in range(n))
        basis = np.column_stack([np.log(energy) ** i for i in range(n)]) if n else None
    elif curve.model_type == "functional" and curve.parameters.get("form") == "gray":
        names = ("a", "b", "c", "d")
        basis = np.column_stack(
            (np.ones_like(energy), np.log(energy), np.log(energy) ** 2, 1 / energy)
        )
    else:
        raise ValueError("cross-energy covariance unsupported for this curve model")
    if not names or tuple(uncertainty.get("parameter_names", ())) != names:
        raise ValueError("fit covariance parameter order must match curve coefficients")
    covariance = covariance_matrix(
        uncertainty.get("covariance"), len(names), "fit_covariance"
    )
    efficiency = np.atleast_1d(curve.efficiency(energy))
    jacobian = efficiency[:, None] * basis
    result = jacobian @ covariance @ jacobian.T
    systematic = float(uncertainty.get("relative_systematic", 0))
    if not np.isfinite(systematic) or systematic < 0:
        raise ValueError("relative systematic must be finite and nonnegative")
    if systematic:
        if systematic_correlation == "shared":
            result += np.outer(efficiency, efficiency) * systematic**2
        elif systematic_correlation == "independent":
            result += np.diag(efficiency**2 * systematic**2)
        else:
            raise ValueError("declare systematic_correlation as shared or independent")
    return result
