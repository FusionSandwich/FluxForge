"""Opt-in native-bin sample/ambient Poisson inference (issue #239).

For live times t_s,t_b and declared normalization k:
  mu_s = A P_s + C_s c + k B_s b; mu_b = (t_b/t_s) B_b b.
A is a nonnegative full-response area in sample live-time counts. The ambient
b includes its own continuum and declared peaks, including a peak coincident
with the sample peak. Observations are never subtracted, aligned or smoothed.
Each response is integrated on its own calibrated native bin edges. The
continuum coefficients are nonnegative integrated counts at sample exposure.

This bounded primitive fixes calibrated centroid/width and model choices. It
does not choose backgrounds, infer applicability, fit efficiency, or alter any
workflow default. A later ambient measurement must be declared conditional.
Profile sets use the chi-square(1) likelihood-ratio cutoff, clipped at A=0:
asymptotic (conservative at the signal boundary with regular nuisances), not
calibrated coverage for sparse counts or nuisance boundaries.
They exclude response, efficiency and undeclared systematic uncertainties.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.special import ndtr, kl_div, xlogy
from scipy.stats import chi2

from .peakfit import gauss_with_erf, gaussian, poisson_neg_log_likelihood


@dataclass(frozen=True)
class CountObservation:
    counts: np.ndarray
    energy_edges_keV: np.ndarray
    live_time_s: float
    acquisition_id: str
    acquired_at: Optional[str] = None
    count_basis: str = "original_native_counts"

    def validated(self):
        y = np.asarray(self.counts, dtype=float)
        edges = np.asarray(self.energy_edges_keV, dtype=float)
        if self.count_basis != "original_native_counts":
            raise ValueError("Poisson observations require original_native_counts")
        if (
            y.ndim != 1
            or not y.size
            or not np.all(np.isfinite(y))
            or np.any(y < 0)
            or np.any(y != np.floor(y))
        ):
            raise ValueError("counts must be finite nonnegative integer native counts")
        if (
            edges.shape != (len(y) + 1,)
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
            or not np.all(np.isfinite(np.diff(edges)))
        ):
            raise ValueError("energy edges must be finite, increasing and match counts")
        if not np.isfinite(self.live_time_s) or self.live_time_s <= 0:
            raise ValueError("live exposure must be finite and positive")
        if not isinstance(self.acquisition_id, str) or not self.acquisition_id.strip():
            raise ValueError("acquisition_id is required")
        if self.acquired_at is not None:
            try:
                datetime.fromisoformat(self.acquired_at)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "acquired_at must be an ISO timestamp or None (unknown)"
                ) from exc
        return y.copy(), edges.copy()


@dataclass(frozen=True)
class PeakResponse:
    centroid_keV: float
    sigma_keV: float

    def integrated(self, edges):
        if (
            not np.isfinite(self.centroid_keV)
            or not np.isfinite(self.sigma_keV)
            or self.sigma_keV <= 0
        ):
            raise ValueError("peak centroid/width must be finite with positive sigma")
        z = (edges - self.centroid_keV) / self.sigma_keV
        # Use the survival side above the mean to avoid subtracting two ones.
        return np.where(
            z[:-1] >= 0, ndtr(-z[:-1]) - ndtr(-z[1:]), ndtr(z[1:]) - ndtr(z[:-1])
        )


@dataclass(frozen=True)
class BackgroundChoice:
    applicability: str
    rationale: str
    normalization: str = "fixed"  # fixed, free, gaussian_auxiliary
    scale: float = 1.0  # multiplier of live-time ratio, not the ratio itself
    scale_sigma: Optional[float] = None  # independent auxiliary measurement
    continuum: str = "linear"
    peaks: tuple[PeakResponse, ...] = ()


@dataclass
class ProfileInterval:
    lower: Optional[float]
    upper: Optional[float]
    confidence: float
    success: bool
    message: str
    method: str = "asymptotic_chi2_1_profile_clipped_at_zero"


@dataclass
class JointPoissonResult:
    success: bool
    status: str
    message: str
    area: float
    normalization: Optional[float]
    exposure_scale: Optional[float]
    sample_expected: np.ndarray
    background_expected: np.ndarray
    sample_residuals: np.ndarray
    background_residuals: np.ndarray
    deviance: float
    negative_log_likelihood: float
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    identifiability_ratio: float
    provenance: dict
    model_diagnostics: dict
    interval: Optional[ProfileInterval] = None
    _problem: object = field(default=None, repr=False)


def _continuum(edges, kind, domain, peak):
    lo, hi = domain
    width = np.diff(edges)
    if kind == "none":
        return np.empty((len(width), 0))
    if kind == "constant":
        return (width / (hi - lo))[:, None]
    if kind == "linear":
        u = ((edges[:-1] + edges[1:]) / 2 - lo) / (hi - lo)
        return np.column_stack((2 * width * (1 - u), 2 * width * u)) / (hi - lo)
    if kind == "step":
        # Analytic primitive of the existing error-function step, accurate
        # even when a native bin is much wider than the peak response.
        u = edges - peak.centroid_keV
        left = gauss_with_erf(edges, 0, 1, peak.centroid_keV, peak.sigma_keV)
        primitive = u * left - peak.sigma_keV * gaussian(
            edges, 1 / np.sqrt(2 * np.pi), peak.centroid_keV, peak.sigma_keV
        )
        integral = np.clip(np.diff(primitive), 0, width)
        return np.column_stack((integral, width - integral)) * 2 / (hi - lo)
    raise ValueError("continuum must be none, constant, linear or step")


def _loss(mu, y):
    """Exact zero-mean extension of the existing Poisson helper."""
    if np.any(mu < 0) or not np.all(np.isfinite(mu)):
        return np.inf
    positive = mu > 0
    if np.any(y[~positive] > 0):
        return np.inf
    regular = mu >= 1e-10
    tiny = positive & ~regular
    return float(
        poisson_neg_log_likelihood(mu[regular], y[regular])
        + np.sum(mu[tiny] - xlogy(y[tiny], mu[tiny]))
    )


class _Problem:
    def __init__(self, sample, ambient, peak, choice, continuum, maxiter):
        self.y, self.es = sample.validated()
        self.z, self.eb = ambient.validated() if ambient else (np.array([]), None)
        if ambient and sample.acquisition_id == ambient.acquisition_id:
            raise ValueError(
                "sample and background must have distinct acquisition identities"
            )
        peak.integrated(self.es)  # validates even if a basis is absent
        if choice.applicability not in {
            "later_conditional",
            "earlier_conditional",
            "contemporaneous_declared",
            "no_separate_ambient_vendor",
        }:
            raise ValueError("declare supported background applicability explicitly")
        if not isinstance(choice.rationale, str) or not choice.rationale.strip():
            raise ValueError("background applicability rationale is required")
        if (ambient is None) != (choice.applicability == "no_separate_ambient_vendor"):
            raise ValueError(
                "no_separate_ambient_vendor requires no ambient observation"
            )
        self.chronology = "unknown"
        if (
            ambient
            and sample.acquired_at is not None
            and ambient.acquired_at is not None
        ):
            try:
                background_date = datetime.fromisoformat(ambient.acquired_at)
                sample_date = datetime.fromisoformat(sample.acquired_at)
                later = background_date > sample_date
            except TypeError as exc:
                raise ValueError(
                    "acquisition timestamps must use compatible timezone conventions"
                ) from exc
            self.chronology = "background_later" if later else "background_not_later"
            if later and choice.applicability != "later_conditional":
                raise ValueError(
                    "later background requires later_conditional applicability"
                )
            if (
                background_date < sample_date
                and choice.applicability != "earlier_conditional"
            ):
                raise ValueError(
                    "earlier background requires earlier_conditional applicability"
                )
        if choice.normalization not in {"fixed", "free", "gaussian_auxiliary"}:
            raise ValueError("unknown normalization choice")
        if not np.isfinite(choice.scale) or choice.scale <= 0:
            raise ValueError("normalization scale must be finite and positive")
        if choice.normalization == "gaussian_auxiliary":
            if (
                choice.scale_sigma is None
                or not np.isfinite(choice.scale_sigma)
                or choice.scale_sigma <= 0
            ):
                raise ValueError("auxiliary scale_sigma must be finite and positive")
        elif choice.scale_sigma is not None:
            raise ValueError("scale_sigma requires gaussian_auxiliary normalization")
        if not ambient and (
            choice.normalization != "fixed"
            or choice.peaks
            or choice.continuum != "none"
        ):
            raise ValueError(
                "vendor scenario cannot contain unused ambient nuisance choices"
            )
        # Native supports must overlap; each likelihood still sees its own bins.
        if ambient and min(self.es[-1], self.eb[-1]) <= max(self.es[0], self.eb[0]):
            raise ValueError("native energy supports do not overlap")
        domain = (
            min(self.es[0], self.eb[0]) if ambient else self.es[0],
            max(self.es[-1], self.eb[-1]) if ambient else self.es[-1],
        )
        self.p = peak.integrated(self.es)
        if self.p.sum() <= 1e-8:
            raise ValueError("peak response has negligible coverage in sample ROI")
        self.c = _continuum(self.es, continuum, domain, peak)
        self.bs = (
            _continuum(self.es, choice.continuum, domain, peak)
            if ambient
            else np.empty((len(self.y), 0))
        )
        self.bb = (
            _continuum(self.eb, choice.continuum, domain, peak)
            if ambient
            else np.empty((0, 0))
        )
        for bp in choice.peaks:
            self.bs = np.column_stack((self.bs, bp.integrated(self.es)))
            self.bb = np.column_stack((self.bb, bp.integrated(self.eb)))
        if ambient and not self.bs.shape[1]:
            raise ValueError("ambient model needs a continuum or peak")
        self.choice, self.maxiter = choice, maxiter
        self.nc, self.nb = self.c.shape[1], self.bs.shape[1]
        self.free = ambient is not None and choice.normalization != "fixed"
        self.ratio = ambient.live_time_s / sample.live_time_s if ambient else 1.0
        if (
            not np.isfinite(self.ratio)
            or self.ratio <= 0
            or not np.isfinite(1 / self.ratio)
        ):
            raise ValueError("exposure ratio must be finite and positive")
        self.names = (
            ("sample_peak_area",)
            + tuple(f"sample_continuum_{i}" for i in range(self.nc))
            + tuple(f"ambient_{i}" for i in range(self.nb))
            + (("normalization",) if self.free else ())
        )
        n = len(self.names)
        totals = (float(self.y.sum()), float(self.z.sum() / self.ratio))
        if not np.all(np.isfinite(totals)):
            raise ValueError("count totals/exposure-scaled counts must be finite")
        unit = max(*totals, 1.0) / max(n, 1)
        self.units = np.full(n, max(unit, 1.0))
        if self.free:
            self.units[-1] = 1.0
        self.initial = np.full(n, 0.5)
        if self.free:
            self.initial[-1] = choice.scale

    def model(self, q):
        v = q * self.units
        b = v[1 + self.nc : 1 + self.nc + self.nb]
        k = v[-1] if self.free else self.choice.scale
        mus = v[0] * self.p + self.c @ v[1 : 1 + self.nc] + k * (self.bs @ b)
        mub = self.ratio * (self.bb @ b)
        js = np.column_stack((self.p, self.c, k * self.bs))
        jb = np.column_stack(
            (np.zeros((len(self.z), 1 + self.nc)), self.ratio * self.bb)
        )
        if self.free:
            js = np.column_stack((js, self.bs @ b))
            jb = np.column_stack((jb, np.zeros(len(self.z))))
        return mus, mub, js * self.units, jb * self.units

    def objective(self, q):
        mus, mub, js, jb = self.model(q)
        # Same Poisson likelihood, expressed relative to the saturated model
        # to avoid cancellation when profiling very close to the optimum.
        loss = float(np.sum(kl_div(self.y, mus)) + np.sum(kl_div(self.z, mub)))
        ds = 1 - np.divide(self.y, mus, out=np.zeros_like(mus), where=mus > 0)
        db = 1 - np.divide(self.z, mub, out=np.zeros_like(mub), where=mub > 0)
        grad = js.T @ ds + jb.T @ db
        if self.choice.normalization == "gaussian_auxiliary":
            delta = q[-1] - self.choice.scale
            loss += 0.5 * (delta / self.choice.scale_sigma) ** 2
            grad[-1] += delta / self.choice.scale_sigma**2
        return loss, grad

    def solve(self, start=None, fixed_area=None):
        start = self.initial.copy() if start is None else start.copy()
        # Tiny positive nuisance floors keep trial means inside the log domain
        # even when a line search would otherwise set every ambient term to 0.
        # The signal area remains exactly zero-capable. These floors are below
        # the KKT active-set tolerance and are recorded in the provenance.
        bounds = [(0.0, None)] + [(1e-12, None)] * (len(start) - 1)
        if fixed_area is not None:
            start[0] = fixed_area / self.units[0]
            bounds[0] = (start[0], start[0])
        try:
            result = optimize.minimize(
                self.objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.maxiter,
                    "ftol": 1e-13,
                    "gtol": 1e-4,
                    "maxls": 40,
                },
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            result = optimize.OptimizeResult(
                x=start,
                fun=self.objective(start)[0],
                success=False,
                message="optimizer exception: " + str(exc),
            )
        if (
            np.shape(result.x) != np.shape(start)
            or not np.all(np.isfinite(result.x))
            or np.any(result.x < 0)
        ):
            return (
                optimize.OptimizeResult(
                    x=start,
                    fun=self.objective(start)[0],
                    success=False,
                    message="optimizer returned invalid parameters",
                ),
                False,
            )
        value, grad = self.objective(result.x)
        projected = np.where(result.x > 1e-9, grad, np.minimum(grad, 0))
        if fixed_area is not None:
            projected[0] = 0
        valid = (
            result.success
            and np.isfinite(value)
            and np.all(np.isfinite(result.x))
            and np.isfinite(result.fun)
            and np.isclose(result.fun, value, rtol=1e-8, atol=1e-8)
            and np.all(result.x >= 0)
            and np.max(np.abs(projected)) < 1e-3
        )
        if result.success and not valid:
            result.message = "optimizer objective/stationarity check failed: " + str(
                result.message
            )
        return result, valid

    def rank_ratio(self, q):
        ms, mb, js, jb = self.model(q)
        j = np.vstack(
            (
                js / np.sqrt(np.maximum(ms, 1))[:, None],
                jb / np.sqrt(np.maximum(mb, 1))[:, None],
            )
        )
        if self.choice.normalization == "gaussian_auxiliary":
            aux = np.zeros((1, len(q)))
            aux[0, -1] = 1 / self.choice.scale_sigma
            j = np.vstack((j, aux))
        norms = np.linalg.norm(j, axis=0)
        if np.any(norms == 0) or j.shape[0] < j.shape[1]:
            return 0.0
        singular = np.linalg.svd(j / norms, compute_uv=False)
        return float(singular[-1] / singular[0])


def fit_joint_poisson(
    sample: CountObservation,
    background: Optional[CountObservation],
    peak: PeakResponse,
    choice: BackgroundChoice,
    *,
    sample_continuum: str = "linear",
    confidence: Optional[float] = 0.95,
    maxiter: int = 2000,
) -> JointPoissonResult:
    """Fit declared original counts; invalid observations raise ValueError.

    ``scale`` multiplies t_s/t_b. Free scaling requires identifiable shapes;
    ``gaussian_auxiliary`` adds an independently justified Gaussian measurement
    of scale, not an invented prior chosen to reproduce a vendor result.
    Failure/unidentifiable results retain diagnostics but are not accepted fits.
    Success is numerical conditional inference; model adequacy is reported
    separately and is required before interpreting estimates physically.
    """
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if confidence is not None and (
        not np.isfinite(confidence) or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be between zero and one")
    problem = _Problem(sample, background, peak, choice, sample_continuum, maxiter)
    opt, ok = problem.solve()
    ms, mb, _, _ = problem.model(opt.x)
    v = opt.x * problem.units
    rank = (
        problem.rank_ratio(opt.x)
        if np.all(np.isfinite(ms)) and np.all(np.isfinite(mb))
        else 0.0
    )
    status = "converged" if ok else "optimizer_failed"
    message = str(opt.message)
    if ok and rank < 1e-6:
        ok, status = False, "unidentifiable"
        message = "Normalization/peak/continuum tradeoff: rank-deficient or ill-conditioned response"
    norm = (v[-1] if problem.free else choice.scale) if background else None
    provenance = {
        "method": "native_joint_poisson_fixed_response",
        "sample_acquisition_id": sample.acquisition_id,
        "background_acquisition_id": background.acquisition_id if background else None,
        "sample_acquired_at": sample.acquired_at,
        "background_acquired_at": background.acquired_at if background else None,
        "sample_live_time_s": sample.live_time_s,
        "background_live_time_s": background.live_time_s if background else None,
        "count_basis": "original_native_counts",
        "area_basis": "sample_live_time_full_gaussian_response_counts",
        "grid_policy": "integrated_response_on_each_native_energy_grid_no_observation_rebin",
        "applicability": choice.applicability,
        "rationale": choice.rationale,
        "normalization_choice": choice.normalization,
        "normalization_declared": choice.scale,
        "normalization_auxiliary_sigma": choice.scale_sigma,
        "sample_continuum": sample_continuum,
        "ambient_continuum": choice.continuum,
        "ambient_peaks": [vars(p) for p in choice.peaks],
        "peak_response": vars(peak),
        "calibration_uncertainty": "not_included",
        "nuisance_floor_scaled": 1e-12,
        "optimizer": "L-BFGS-B_with_projected_gradient_check",
        "sample_energy_edges_keV": problem.es.tolist(),
        "background_energy_edges_keV": problem.eb.tolist() if background else None,
        "acquisition_chronology": problem.chronology,
        "parameter_scale_units": problem.units.tolist(),
        "projected_gradient_acceptance_threshold": 1e-3,
        "identifiability_ratio_threshold": 1e-6,
        "active_nuisance_boundaries": [
            problem.names[i] for i in range(1, len(v)) if opt.x[i] <= 1e-9
        ],
    }
    ds = float(2 * np.sum(kl_div(problem.y, ms)))
    db = float(2 * np.sum(kl_div(problem.z, mb)))
    dof = len(problem.y) + len(problem.z) - len(v)
    approximate_p = float(chi2.sf(ds + db, dof)) if dof > 0 and ok else None
    diagnostics = {
        "sample_poisson_deviance": ds,
        "background_poisson_deviance": db,
        "counts_only_nominal_dof": dof,
        "approximate_chi2_tail_probability": approximate_p,
        "adequacy_flag": (
            "strong_lack_of_fit"
            if approximate_p is not None and approximate_p < 0.01
            else "not_qualified"
        ),
        "qualification": "asymptotic screening only; sparse bins, nuisance boundaries and auxiliary constraints invalidate calibrated tail probabilities",
        "physical_interpretation": "requires model adequacy and applicability review",
    }
    fit = JointPoissonResult(
        ok,
        status,
        message,
        float(v[0]),
        float(norm) if norm is not None else None,
        float(norm / problem.ratio) if norm is not None else None,
        ms,
        mb,
        problem.y - ms,
        problem.z - mb,
        float(2 * (np.sum(kl_div(problem.y, ms)) + np.sum(kl_div(problem.z, mb)))),
        _loss(ms, problem.y) + _loss(mb, problem.z),
        problem.names,
        v,
        rank,
        provenance,
        diagnostics,
        _problem=problem,
    )
    if ok and confidence is not None:
        fit.interval = profile_area_interval(fit, confidence)
        if not fit.interval.success:
            fit.success, fit.status = False, "profile_failed"
            fit.message = fit.interval.message
    return fit


def profile_area_interval(fit: JointPoissonResult, confidence=0.95) -> ProfileInterval:
    """Reoptimize all nuisance parameters at each tested nonnegative area."""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not fit.success or fit._problem is None:
        return ProfileInterval(
            None, None, confidence, False, "requires an accepted identifiable fit"
        )
    p = fit._problem
    q = fit.parameters / p.units
    best = p.objective(q)[0]
    cutoff = float(chi2.ppf(confidence, 1)) / 2

    def crossing(area):
        if len(q) == 1:
            # No nuisance optimization exists. A=0 with positive counts has
            # zero likelihood (+infinite LR), a valid exclusion, not a failure.
            fixed = np.array([area / p.units[0]])
            return float(p.objective(fixed)[0] - best - cutoff)
        opt, ok = p.solve(q, fixed_area=area)
        if not ok:
            raise RuntimeError("profile nuisance optimizer failed: " + str(opt.message))
        delta = float(opt.fun - best)
        if delta < -1e-5:
            raise RuntimeError("profile found a lower optimum than fitted solution")
        return delta - cutoff

    try:
        lower = (
            0.0
            if crossing(0) <= 0
            else float(optimize.brentq(crossing, 0, fit.area, xtol=1e-5))
        )
        upper = fit.area + max(np.sqrt(float(p.y.sum())), 1.0)
        for _ in range(32):
            if crossing(upper) >= 0:
                break
            upper = fit.area + 2 * (upper - fit.area)
        else:
            raise RuntimeError("profile upper limit is unbounded within search budget")
        upper = float(optimize.brentq(crossing, fit.area, upper, xtol=1e-5))
        return ProfileInterval(
            lower,
            upper,
            confidence,
            True,
            "Asymptotic LR set; clipped at zero; sparse-count/nuisance-boundary coverage not calibrated; requires adequate model",
        )
    except (RuntimeError, ValueError) as exc:
        return ProfileInterval(None, None, confidence, False, str(exc))
