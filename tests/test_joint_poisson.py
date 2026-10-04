"""Independent planted controls and failure counterexamples for issue #239."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.stats import chi2

from fluxforge.analysis.joint_poisson import (
    BackgroundChoice,
    CountObservation,
    PeakResponse,
    fit_joint_poisson,
    profile_area_interval,
)


EDGES = np.linspace(-8, 8, 65)
PEAK = PeakResponse(0, 0.8)


def obs(y, edges=EDGES, exposure=1, identity="sample"):
    return CountObservation(np.asarray(y), edges, exposure, identity)


def response(edges, mean=0, sigma=0.8):
    return np.diff(ndtr((edges - mean) / sigma))


def vendor():
    return BackgroundChoice(
        "no_separate_ambient_vendor", "declared vendor scenario", continuum="none"
    )


def ambient(**kwargs):
    return BackgroundChoice(
        "later_conditional",
        "background acquired after sample; conditional only",
        **kwargs,
    )


@pytest.mark.parametrize("area", [0, 50, 4000])
def test_fixed_scale_recovers_zero_weak_and_strong_with_ambient_peak(area):
    # Rounded Asimov counts remain integer native observations. No subtraction
    # is fed to likelihood. Background contains the exact same line response.
    ts, tb, k = 10, 40, 1.4
    p = response(EDGES)
    b = 40 * np.diff(EDGES) + 800 * p
    y = np.rint(area * p + k * b + 10 * np.diff(EDGES))
    z = np.rint(tb / ts * b)
    fit = fit_joint_poisson(
        obs(y, exposure=ts),
        obs(z, exposure=tb, identity="ambient"),
        PEAK,
        ambient(scale=k, continuum="constant", peaks=(PEAK,)),
        sample_continuum="constant",
    )
    assert fit.success, fit.message
    assert fit.area == pytest.approx(area, abs=6)
    assert fit.exposure_scale == pytest.approx(ts / tb * k)
    assert fit.interval.success and fit.interval.lower <= area <= fit.interval.upper
    if area == 0:
        assert fit.interval.lower == 0
        assert fit.interval.upper > fit.area
    np.testing.assert_allclose(fit.sample_residuals, y - fit.sample_expected)
    assert fit.provenance["applicability"] == "later_conditional"


def test_free_scale_recovers_known_exposure_and_area_on_unequal_native_grids():
    eb = np.linspace(-8.1, 8.1, 82)
    ts, tb, k, area = 2, 9, 1.7, 2000
    bp_s = response(EDGES, mean=3)
    bp_b = response(eb, mean=3)
    y = np.rint(area * response(EDGES) + k * (1000 * np.diff(EDGES) + 3000 * bp_s))
    z = np.rint(tb / ts * (1000 * np.diff(eb) + 3000 * bp_b))
    fit = fit_joint_poisson(
        obs(y, exposure=ts),
        obs(z, eb, tb, "ambient"),
        PEAK,
        ambient(
            normalization="free", continuum="constant", peaks=(PeakResponse(3, 0.8),)
        ),
        sample_continuum="none",
    )
    assert fit.success, fit.message
    assert fit.area == pytest.approx(area, abs=5)
    assert fit.normalization == pytest.approx(k, abs=0.002)
    assert fit.exposure_scale == pytest.approx(k * ts / tb, abs=0.001)
    assert len(fit.background_expected) == len(z)
    assert "no_observation_rebin" in fit.provenance["grid_policy"]


def test_poisson_seeded_truth_and_interval():
    rng = np.random.default_rng(239)
    ts, tb = 5, 20
    b = 80 * np.diff(EDGES) + 200 * response(EDGES)
    y = rng.poisson(500 * response(EDGES) + b)
    z = rng.poisson(tb / ts * b)
    fit = fit_joint_poisson(
        obs(y, exposure=ts),
        obs(z, exposure=tb, identity="ambient"),
        PEAK,
        ambient(continuum="constant", peaks=(PEAK,)),
        sample_continuum="none",
    )
    assert fit.success, fit.message
    assert fit.interval.lower < 500 < fit.interval.upper


def test_signed_residual_never_used_as_observation(monkeypatch):
    import fluxforge.analysis.joint_poisson as module

    seen = []
    original = module.poisson_neg_log_likelihood

    def guarded(mu, y):
        assert np.all(np.isfinite(y)) and np.all(y >= 0)
        seen.append(y.copy())
        return original(mu, y)

    monkeypatch.setattr(module, "poisson_neg_log_likelihood", guarded)
    y, z = np.full(64, 2), np.full(64, 8)
    fit = fit_joint_poisson(
        obs(y),
        obs(z, identity="ambient"),
        PEAK,
        ambient(continuum="constant"),
        sample_continuum="none",
    )
    assert np.any(y - z < 0) and seen
    assert fit.success, fit.message
    assert fit.area == 0
    assert fit.interval.lower == 0 and fit.interval.upper > 0


def test_empty_boundary_matches_analytic_profile_limit():
    fit = fit_joint_poisson(
        obs(np.zeros(64)), None, PEAK, vendor(), sample_continuum="none"
    )
    assert fit.success, fit.message
    assert fit.area == 0 and fit.interval.lower == 0
    assert fit.interval.upper == pytest.approx(
        chi2.ppf(0.95, 1) / (2 * response(EDGES).sum()), rel=1e-5
    )
    assert "sparse-count" in fit.interval.message


def test_positive_signal_without_nuisances_has_analytic_profile():
    y = np.rint(100 * response(EDGES))
    fit = fit_joint_poisson(obs(y), None, PEAK, vendor(), sample_continuum="none")
    assert fit.success, fit.message
    assert fit.area == pytest.approx(y.sum(), abs=1e-4)
    for endpoint in (fit.interval.lower, fit.interval.upper):
        expected_lr = 2 * (endpoint - y.sum() + y.sum() * np.log(y.sum() / endpoint))
        assert expected_lr == pytest.approx(chi2.ppf(0.95, 1), abs=1e-4)


def test_step_integral_on_coarse_irregular_bins_matches_independent_quadrature():
    from fluxforge.analysis.joint_poisson import _continuum

    edges = np.array([-8, -0.1, 100.0])
    peak = PeakResponse(0, 0.1)
    basis = _continuum(edges, "step", (edges[0], edges[-1]), peak)
    oracle = quad(lambda x: ndtr(-x / 0.1), -0.1, 100, epsabs=1e-11)[0]
    assert basis[1, 0] * 108 / 2 == pytest.approx(oracle, abs=1e-10)
    np.testing.assert_allclose(basis.sum(axis=1), 2 * np.diff(edges) / 108)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, 0.5])
@pytest.mark.parametrize("where", ["sample", "background"])
def test_invalid_counts_fail_explicitly(bad, where):
    y = np.ones(64)
    y[12] = bad
    sample = obs(y if where == "sample" else np.ones(64))
    background = obs(y if where == "background" else np.ones(64), identity="ambient")
    with pytest.raises(ValueError, match="counts must"):
        fit_joint_poisson(sample, background, PEAK, ambient())


@pytest.mark.parametrize("exposure", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("where", ["sample", "background"])
def test_invalid_live_exposure_fails(exposure, where):
    sample = obs(np.ones(64), exposure=exposure if where == "sample" else 1)
    background = obs(
        np.ones(64),
        exposure=exposure if where == "background" else 1,
        identity="ambient",
    )
    with pytest.raises(ValueError, match="exposure"):
        fit_joint_poisson(sample, background, PEAK, ambient())


def test_rebinned_counts_and_reused_identity_rejected():
    sample = obs(np.ones(64))
    with pytest.raises(ValueError, match="original_native"):
        fit_joint_poisson(
            replace(sample, count_basis="rebinned_counts"), None, PEAK, vendor()
        )
    with pytest.raises(ValueError, match="distinct acquisition"):
        fit_joint_poisson(sample, sample, PEAK, ambient())


def test_later_background_cannot_be_declared_contemporaneous():
    sample = replace(obs(np.ones(64)), acquired_at="2025-08-28T12:55:00")
    background = replace(
        obs(np.ones(64), identity="ambient"), acquired_at="2026-03-02T17:59:00"
    )
    choice = replace(ambient(), applicability="contemporaneous_declared")
    with pytest.raises(ValueError, match="later_conditional"):
        fit_joint_poisson(sample, background, PEAK, choice)


def test_earlier_background_requires_truthful_conditional_label():
    sample = replace(obs(np.ones(64)), acquired_at="2026-03-02T17:59:00")
    background = replace(
        obs(np.ones(64), identity="ambient"), acquired_at="2025-08-28T12:55:00"
    )
    with pytest.raises(ValueError, match="earlier_conditional"):
        fit_joint_poisson(sample, background, PEAK, ambient())


def test_unknown_acquisition_dates_remain_unknown():
    fit = fit_joint_poisson(obs(np.ones(64)), None, PEAK, vendor())
    assert fit.provenance["sample_acquired_at"] is None
    assert fit.provenance["acquisition_chronology"] == "unknown"


@pytest.mark.parametrize("stamp", ["not-a-date", 123])
def test_invalid_timestamp_rejected(stamp):
    with pytest.raises(ValueError, match="ISO timestamp"):
        fit_joint_poisson(
            replace(obs(np.ones(64)), acquired_at=stamp), None, PEAK, vendor()
        )


@pytest.mark.parametrize("edges", [np.zeros(65), np.arange(64), np.full(65, np.nan)])
def test_invalid_native_energy_grid(edges):
    with pytest.raises(ValueError, match="energy edges"):
        fit_joint_poisson(obs(np.ones(64), edges), None, PEAK, vendor())


def test_unidentifiable_normalization_peak_continuum_tradeoff_not_success():
    p = response(EDGES)
    fit = fit_joint_poisson(
        obs(np.rint(1000 * p + 40)),
        obs(np.rint(200 * p + 10), identity="ambient"),
        PEAK,
        ambient(normalization="free", continuum="constant", peaks=(PEAK,)),
        sample_continuum="constant",
    )
    assert not fit.success
    assert fit.status == "unidentifiable" and fit.interval is None
    assert fit.identifiability_ratio < 1e-6
    assert not profile_area_interval(fit).success


def test_independent_auxiliary_normalization_breaks_declared_tradeoff():
    p = response(EDGES)
    y, z = np.rint(1200 * p + 30), np.rint(200 * p + 10)
    fit = fit_joint_poisson(
        obs(y),
        obs(z, identity="ambient"),
        PEAK,
        ambient(
            normalization="gaussian_auxiliary",
            scale_sigma=0.1,
            continuum="constant",
            peaks=(PEAK,),
        ),
        sample_continuum="constant",
    )
    assert fit.success, fit.message
    assert fit.area == pytest.approx(1000, abs=6)
    assert fit.normalization == pytest.approx(1, abs=0.001)
    assert fit.provenance["normalization_auxiliary_sigma"] == 0.1


def test_optimizer_failure_not_accepted(monkeypatch):
    def failed(fun, start, **kwargs):
        return SimpleNamespace(
            x=start, fun=fun(start)[0], success=False, message="test failure"
        )

    monkeypatch.setattr("fluxforge.analysis.joint_poisson.optimize.minimize", failed)
    fit = fit_joint_poisson(obs(np.ones(64)), None, PEAK, vendor())
    assert not fit.success and fit.status == "optimizer_failed"
    assert fit.interval is None and "test failure" in fit.message


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_false_optimizer_success_with_invalid_parameters_not_accepted(
    monkeypatch, invalid
):
    def failed(fun, start, **kwargs):
        return SimpleNamespace(
            x=np.full_like(start, invalid), fun=0, success=True, message="false success"
        )

    monkeypatch.setattr("fluxforge.analysis.joint_poisson.optimize.minimize", failed)
    fit = fit_joint_poisson(obs(np.ones(64)), None, PEAK, vendor())
    assert not fit.success and fit.status == "optimizer_failed" and fit.interval is None


def test_optimizer_exception_reported(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr("fluxforge.analysis.joint_poisson.optimize.minimize", failed)
    fit = fit_joint_poisson(obs(np.ones(64)), None, PEAK, vendor())
    assert not fit.success and "injected solver failure" in fit.message


def test_profile_optimizer_failure_invalidates_fit(monkeypatch):
    import fluxforge.analysis.joint_poisson as module

    original = module._Problem.solve

    def fail_profile(self, start=None, fixed_area=None):
        result, valid = original(self, start, fixed_area)
        return result, valid and fixed_area is None

    monkeypatch.setattr(module._Problem, "solve", fail_profile)
    fit = fit_joint_poisson(obs(np.ones(64)), None, PEAK, vendor())
    assert not fit.success and fit.status == "profile_failed"
    assert not fit.interval.success and fit.interval.upper is None


@pytest.mark.parametrize(
    "choice",
    [
        ambient(scale=-1),
        ambient(normalization="mystery"),
        ambient(normalization="gaussian_auxiliary", scale_sigma=0),
        ambient(scale_sigma=0.1),
    ],
)
def test_invalid_declared_nuisance_choices_fail(choice):
    with pytest.raises(ValueError):
        fit_joint_poisson(
            obs(np.ones(64)), obs(np.ones(64), identity="ambient"), PEAK, choice
        )


@pytest.mark.parametrize("continuum", ["linear", "step"])
def test_local_continuum_response_nonnegative_and_finite(continuum):
    y = np.rint(2000 * response(EDGES) + 200 * np.diff(EDGES))
    fit = fit_joint_poisson(obs(y), None, PEAK, vendor(), sample_continuum=continuum)
    assert fit.success, fit.message
    assert fit.area == pytest.approx(2000, abs=6)
    assert np.all(fit.sample_expected >= 0)


def test_co_cd_pilot_preserves_originals_and_separates_adequacy_from_convergence():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/RAFM_irradiation/joint_poisson_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("joint_poisson_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.run_pilot()
    assert payload["originals_byte_identical_after_run"]
    assert len(payload["rows"]) == 10
    fits = [r for r in payload["rows"] if "joint" in r]
    assert len(fits) == 8
    assert all(r["joint"]["success"] for r in fits)
    assert all(
        r["joint"]["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit"
        for r in fits
    )
    for row in fits:
        assert row["roi_sample_channels_inclusive"] == list(
            row["current_gaussian"]["fit_region"]
        )
        assert row["vendor_report_counts"] is None
        assert np.all(np.array(row["sample_original_counts"]) >= 0)
        assert np.all(np.array(row["sample_original_counts"]) % 1 == 0)
    assert all(
        r["status"] == "unidentifiable" and not r["success"]
        for r in payload["rows"]
        if "joint" not in r
    )
