"""Conservative, opt-in ROI qualification using existing peakfit responses.

This is a local high-count weighted least-squares diagnostic, not an isotope
identifier or activity calculation. Areas are full Gaussian integrals in counts;
input x must be unit-spaced channel coordinates. Doublet separation is supplied
independently (e.g. a calibrated library); only a common shift and width are fit.
Covariance is conditional on that separation and on the declared response model.
No saved vendor settings or residual improvement establish nuclide identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import warnings

import numpy as np
from scipy import optimize, stats

from .peakfit import FWHM_SIG_RATIO, gaussian, gaussian_with_step_bg


@dataclass(frozen=True)
class QualificationPolicy:
    """Fixed screening thresholds; these are not a spectroscopy standard."""

    min_separation_fwhm: float = 0.5
    max_area_correlation: float = 0.95
    max_covariance_condition: float = 1e8
    min_component_snr: float = 3.0
    min_delta_bic: float = 10.0
    min_goodness_p: float = 0.001
    min_expected_count: float = 5.0
    min_dof: int = 8


@dataclass
class ResponseFit:
    status: str
    reasons: list[str] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)
    parameters: list[float] = field(default_factory=list)
    covariance: list[list[float]] | None = None
    component_areas: list[float] = field(default_factory=list)
    area_covariance: list[list[float]] | None = None
    total_area: float | None = None
    total_area_uncertainty: float | None = None
    predicted: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    whitened_residuals: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class MultipletQualification:
    status: str
    reasons: list[str]
    single: ResponseFit | None = None
    doublet: ResponseFit | None = None
    broad_single: ResponseFit | None = None
    component_admission: list[bool] = field(default_factory=lambda: [False, False])
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _fit_response(
    x, y, centers, sigma_bounds, shift_limit, continuum, noise, policy, max_evaluations
):
    """Fit the same ROI/noise/response family for both competing models."""
    n = len(centers)
    bg_n = 1 if continuum == "constant" else 2
    k = n + 2 + bg_n + (continuum == "step")
    dof = len(x) - k
    if dof < policy.min_dof:
        return ResponseFit("overparameterized", ["insufficient_residual_dof"])
    t = (x - x[0]) / (x[-1] - x[0])
    anchor = float(np.mean(centers))
    sigma0 = float(np.mean(sigma_bounds))
    background = max(float(np.median(np.r_[y[:3], y[-3:]])), 0.1)
    area0 = max(float(np.sum(y - background)) / n, 1.0)
    p0 = [area0] * n + [0.0, sigma0] + [background] * bg_n
    lower = [0.0] * n + [-shift_limit, sigma_bounds[0]] + [0.0] * bg_n
    upper = [np.inf] * n + [shift_limit, sigma_bounds[1]] + [np.inf] * bg_n
    names = [f"area_{i}" for i in range(n)] + ["common_shift", "shared_sigma"]
    names += ["continuum"] if bg_n == 1 else ["continuum_left", "continuum_right"]
    if continuum == "step":
        p0.append(0.1)
        lower.append(0.0)
        upper.append(np.inf)
        names.append("step_height")

    def model(_x, *p):
        shift, sigma = p[n : n + 2]
        bg = (
            np.full_like(x, p[n + 2])
            if bg_n == 1
            else ((1 - t) * p[n + 2] + t * p[n + 3])
        )
        if continuum == "step":
            # Existing erfc step helper; zero peak amplitude makes this a
            # continuum term. It is excluded from Gaussian component areas.
            bg = bg + gaussian_with_step_bg(
                x, 0.0, anchor + shift, sigma, p[-1] / 2, -p[-1] / 2
            )
        for area, center in zip(p[:n], centers):
            bg = bg + gaussian(
                x, area / (sigma * np.sqrt(2 * np.pi)), center + shift, sigma
            )
        return bg

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", optimize.OptimizeWarning)
            p, cov = optimize.curve_fit(
                model,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                sigma=noise,
                absolute_sigma=True,
                maxfev=max_evaluations,
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(cov)):
            return ResponseFit("fit_failed", ["nonfinite_parameters_or_covariance"])
        if np.any(np.diag(cov) <= 0):
            return ResponseFit("non_identifiable", ["invalid_covariance"])
        scales = np.sqrt(np.diag(cov))
        correlation = cov / np.outer(scales, scales)
        condition = float(np.linalg.cond(correlation))
        if np.min(np.linalg.eigvalsh(correlation)) <= 0:
            return ResponseFit("non_identifiable", ["non_positive_covariance"])
        predicted = model(x, *p)
        residuals = y - predicted
        white = (
            np.linalg.solve(np.linalg.cholesky(noise), residuals)
            if noise.ndim == 2
            else residuals / noise
        )
        chi2 = float(white @ white)
        acov = cov[:n, :n]
        total_variance = float(np.sum(acov))
        if total_variance <= 0:
            return ResponseFit("non_identifiable", ["invalid_total_variance"])
        reasons = []
        # Active bounds invalidate the unconstrained local covariance.
        for value, lo, hi, name in zip(p, lower, upper, names):
            tolerance = 1e-5 * max(1.0, abs(value))
            if value - lo <= tolerance or hi - value <= tolerance:
                reasons.append(f"parameter_at_bound:{name}")
        if condition > policy.max_covariance_condition:
            reasons.append("ill_conditioned_covariance")
        goodness = float(stats.chi2.sf(chi2, dof))
        if goodness < policy.min_goodness_p:
            reasons.append("structured_or_excess_residuals")
        if np.min(predicted) < policy.min_expected_count:
            reasons.append("low_count_wls_not_qualified")
        signs = np.sign(white)
        runs = int(1 + np.count_nonzero(np.diff(signs)))
        lag1 = float(white[:-1] @ white[1:] / (white @ white)) if chi2 > 1e-20 else 0.0
        return ResponseFit(
            "rejected" if reasons else "qualified",
            reasons,
            names,
            p.tolist(),
            cov.tolist(),
            p[:n].tolist(),
            acov.tolist(),
            float(np.sum(p[:n])),
            float(np.sqrt(total_variance)),
            predicted.tolist(),
            residuals.tolist(),
            white.tolist(),
            dict(
                chi_squared=chi2,
                dof=dof,
                bic=chi2 + k * np.log(len(x)),
                goodness_p=goodness,
                covariance_condition=condition,
                residual_lag1=lag1,
                residual_sign_runs=runs,
            ),
        )
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
        optimize.OptimizeWarning,
    ) as exc:
        return ResponseFit("fit_failed", [f"{type(exc).__name__}: {exc}"])


def qualify_doublet(
    channels,
    counts,
    candidate_channels,
    *,
    sigma_bounds,
    resolution_evidence: str,
    component_evidence=(None, None),
    continuum="linear",
    response="gaussian",
    response_evidence=None,
    shift_limit=0.5,
    counts_uncertainty=None,
    counts_covariance=None,
    count_basis="raw_sample",
    policy=None,
    max_evaluations=4000,
) -> MultipletQualification:
    """Compare a justified single peak to a constrained known-separation doublet.

    ``sigma_bounds`` are external resolution limits in channels, common to both
    models. ``resolution_evidence`` must identify their independent source.
    The single control centroid can move across the candidate interval, whereas
    the doublet has only a common calibration shift. This prevents a fictitious
    second component merely accommodating a displaced single peak.

    ``component_evidence`` records independent line/nuclide support supplied by
    the caller, never evidence obtained by this fit. Statistical qualification
    alone does not admit components. Admission still requires downstream
    nuclide/yield/efficiency checks. Tail responses remain explicitly unsupported
    in this layer: existing tail helpers lack qualified joint area semantics.
    A step continuum uses an existing response only with independent evidence.
    ``raw_sample`` means unscaled, integer Poisson observations; supplied
    variance must not undercut the observed-count noise used by this method.
    Simulated/expected data use the explicit ``synthetic`` count basis and
    declared noise. Its component admission is scoped to simulation only.
    Signed/background-adjusted data require declared noise, including shared
    bin covariance where applicable. No counts are silently clipped/subtracted.
    """
    policy = policy or QualificationPolicy()
    try:
        evidence = list(component_evidence)
    except TypeError:
        return MultipletQualification("invalid_input", ["invalid_component_evidence"])
    provenance = dict(
        response=response,
        continuum=continuum,
        count_basis=count_basis,
        resolution_evidence=resolution_evidence,
        response_evidence=response_evidence,
        component_evidence=evidence,
        policy=asdict(policy),
        covariance_scope="conditional_on_line_separation_and_response",
        area_basis="full_gaussian_integral_counts",
        method="absolute_sigma_weighted_least_squares",
    )

    def stop(status, reason):
        return MultipletQualification(status, [reason], provenance=provenance)

    if response != "gaussian" or continuum not in {"constant", "linear", "step"}:
        return stop("unsupported", "response_or_continuum_not_jointly_qualified")
    if continuum == "step" and not response_evidence:
        return stop("unsupported", "step_requires_independent_response_evidence")
    try:
        x, y = np.asarray(channels, float), np.asarray(counts, float)
        centers = np.asarray(candidate_channels, float)
        bounds = np.asarray(sigma_bounds, float)
        if (
            x.ndim != 1
            or y.shape != x.shape
            or len(x) < 3
            or np.any(~np.isfinite(x))
            or np.any(~np.isfinite(y))
            or not np.allclose(np.diff(x), 1.0, atol=1e-9, rtol=0)
        ):
            return stop("invalid_input", "require_finite_unit_spaced_channel_counts")
        if (
            centers.shape != (2,)
            or np.any(~np.isfinite(centers))
            or np.any(np.diff(centers) < 0)
            or centers[0] < x[0]
            or centers[1] > x[-1]
        ):
            return stop("invalid_input", "require_two_ordered_in_roi_candidates")
        if (
            bounds.shape != (2,)
            or np.any(~np.isfinite(bounds))
            or not 0 < bounds[0] < bounds[1]
            or not isinstance(resolution_evidence, str)
            or not resolution_evidence.strip()
            or not np.isfinite(shift_limit)
            or shift_limit <= 0
            or len(evidence) != 2
            or not isinstance(max_evaluations, int)
            or max_evaluations < 1
        ):
            return stop("invalid_input", "invalid_resolution_evidence_or_constraints")
        if bounds[0] < 1.0:
            return stop("unsupported", "undersampled_response_requires_bin_integration")
        thresholds = np.asarray(
            [
                policy.min_separation_fwhm,
                policy.max_area_correlation,
                policy.max_covariance_condition,
                policy.min_component_snr,
                policy.min_delta_bic,
                policy.min_goodness_p,
                policy.min_expected_count,
                policy.min_dof,
            ],
            float,
        )
        if (
            np.any(~np.isfinite(thresholds))
            or np.any(thresholds <= 0)
            or policy.max_area_correlation >= 1
            or policy.min_goodness_p >= 1
        ):
            return stop("invalid_input", "invalid_policy")
        if count_basis not in {"raw_sample", "background_adjusted", "synthetic"}:
            return stop("unsupported", "unknown_count_basis")
        if count_basis == "raw_sample" and np.any(y < 0):
            return stop("invalid_input", "raw_sample_counts_must_be_nonnegative")
        if count_basis == "raw_sample" and not np.all(y == np.floor(y)):
            return stop(
                "invalid_input", "raw_sample_requires_unscaled_integer_observations"
            )
        if (
            count_basis in {"background_adjusted", "synthetic"}
            and counts_uncertainty is None
            and counts_covariance is None
        ):
            return stop("invalid_input", "adjusted_counts_require_declared_noise")
        uncertainty = (
            np.asarray(counts_uncertainty, float)
            if counts_uncertainty is not None
            else None
        )
        if uncertainty is not None and (
            uncertainty.shape != y.shape
            or np.any(~np.isfinite(uncertainty))
            or np.any(uncertainty <= 0)
        ):
            return stop("invalid_input", "invalid_count_uncertainty")
        if counts_covariance is not None:
            noise = np.asarray(counts_covariance, float)
            if (
                noise.shape != (len(x), len(x))
                or np.any(~np.isfinite(noise))
                or not np.allclose(noise, noise.T, rtol=1e-10, atol=1e-10)
            ):
                return stop("invalid_input", "invalid_count_covariance")
            np.linalg.cholesky(noise)
            if uncertainty is not None and not np.allclose(
                np.diag(noise), uncertainty**2
            ):
                return stop("invalid_input", "count_noise_diagonal_mismatch")
        else:
            noise = (
                uncertainty if uncertainty is not None else np.sqrt(np.maximum(y, 1.0))
            )
        variance = np.diag(noise) if noise.ndim == 2 else noise**2
        if count_basis == "raw_sample" and np.any(
            variance < np.maximum(y, 1.0) * (1 - 1e-10)
        ):
            return stop(
                "invalid_input", "raw_sample_variance_below_observed_poisson_noise"
            )
        if count_basis == "raw_sample" and noise.ndim == 2:
            extra_noise = noise - np.diag(np.maximum(y, 1.0))
            if np.min(np.linalg.eigvalsh(extra_noise)) < -1e-9 * max(
                1.0, float(np.max(variance))
            ):
                return stop(
                    "invalid_input",
                    "raw_covariance_undercuts_independent_poisson_noise",
                )
        provenance.update(
            roi_channels=[float(x[0]), float(x[-1])],
            candidate_channels=centers.tolist(),
            sigma_bounds=bounds.tolist(),
            shift_limit=float(shift_limit),
            max_evaluations=max_evaluations,
            noise_basis=(
                "declared_covariance"
                if counts_covariance is not None
                else (
                    "declared_sigma"
                    if uncertainty is not None
                    else "sqrt_observed_counts_floor_1"
                )
            ),
        )
        # Full response must be observable: prevent extrapolated areas from
        # clipped candidates using the broadest permitted width and shift.
        if (
            centers[0] - shift_limit - 3 * bounds[1] < x[0]
            or centers[1] + shift_limit + 3 * bounds[1] > x[-1]
        ):
            return stop("invalid_input", "roi_clips_candidate_response")
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return stop("invalid_input", f"{type(exc).__name__}: {exc}")

    single = _fit_response(
        x,
        y,
        [np.mean(centers)],
        bounds,
        max(shift_limit, (centers[1] - centers[0]) / 2 + shift_limit),
        continuum,
        noise,
        policy,
        max_evaluations,
    )
    doublet = _fit_response(
        x, y, centers, bounds, shift_limit, continuum, noise, policy, max_evaluations
    )
    # A doublet can mimic a single broadened response. This diagnostic
    # relaxes the upper width to a fixed multiple; it never establishes a
    # physically valid resolution. An adequate broad single is ambiguity,
    # not evidence for two components, even if the nominal control fails.
    broad_single = _fit_response(
        x,
        y,
        [np.mean(centers)],
        (bounds[0], 3 * bounds[1]),
        max(shift_limit, (centers[1] - centers[0]) / 2 + shift_limit),
        continuum,
        noise,
        policy,
        max_evaluations,
    )
    result = MultipletQualification(
        doublet.status,
        list(doublet.reasons),
        single,
        doublet,
        broad_single,
        provenance=provenance,
    )
    if doublet.covariance is None:
        return result
    sigma = doublet.parameters[3]
    separation = float((centers[1] - centers[0]) / (FWHM_SIG_RATIO * sigma))
    acov = np.asarray(doublet.area_covariance)
    area_corr = float(acov[0, 1] / np.sqrt(acov[0, 0] * acov[1, 1]))
    snr = np.asarray(doublet.component_areas) / np.sqrt(np.diag(acov))
    doublet.diagnostics.update(
        separation_fwhm=separation,
        area_correlation=area_corr,
        component_snr=snr.tolist(),
    )
    if (
        separation < policy.min_separation_fwhm
        or abs(area_corr) > policy.max_area_correlation
    ):
        result.status = "non_identifiable"
        result.reasons.append("unresolved_or_correlated_components")
        return result
    if doublet.status != "qualified":
        return result
    if broad_single.covariance is None:
        result.status = "control_failed"
        result.reasons.append("broad_single_control_did_not_produce_valid_covariance")
        return result
    if broad_single.status == "qualified":
        broad_delta = float(
            broad_single.diagnostics["bic"] - doublet.diagnostics["bic"]
        )
        doublet.diagnostics["delta_bic_broad_single_minus_doublet"] = broad_delta
        if broad_delta < policy.min_delta_bic:
            result.status = "response_ambiguous"
            result.reasons.append("broader_single_explains_roi_resolution_conflict")
            return result
    if single.covariance is None:
        result.status = "control_failed"
        result.reasons.append("single_control_did_not_produce_valid_covariance")
        return result
    delta = float(single.diagnostics["bic"] - doublet.diagnostics["bic"])
    doublet.diagnostics["delta_bic_single_minus_doublet"] = delta
    if delta < policy.min_delta_bic or np.any(snr < policy.min_component_snr):
        result.status = "single_preferred"
        result.reasons.append("extra_component_not_supported")
        return result
    if not all(isinstance(e, str) and e.strip() for e in evidence):
        result.status = "evidence_required"
        result.reasons.append("fit_improvement_does_not_establish_component_identity")
        return result
    result.status = "qualified"
    result.component_admission = [True, True]
    return result
