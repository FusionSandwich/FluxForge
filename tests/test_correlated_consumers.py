import numpy as np
import pytest
from scipy import sparse

from fluxforge.analysis.peakfit import fit_single_peak, fit_multiple_peaks, calculate_activity
from fluxforge.core.analysis_workspace import analyze_roi_region
from fluxforge.core.peak_fitting import fit_roi_peak
from fluxforge.io.spe import GammaSpectrum


def test_parity_reader_retains_signed_counts_and_full_covariance():
    from fluxforge.validation.reference_parity import _spectrum_from_payload

    original = GammaSpectrum(counts=[-2, 5], energies=np.array([1., 2.]),
                             counts_covariance=sparse.csr_matrix([[4., 1.], [1., 9.]]),
                             metadata={"operation": "background subtraction"})
    restored = _spectrum_from_payload(original.to_dict())
    np.testing.assert_array_equal(restored.counts, original.counts)
    np.testing.assert_array_equal(restored.energies, original.energies)
    np.testing.assert_array_equal(restored.counts_covariance.toarray(),
                                  original.counts_covariance.toarray())
    assert restored.metadata == original.metadata


def test_rafm_legacy_export_rejects_covariance_before_touching_file(tmp_path):
    from fluxforge.examples.rafm_workflow import save_counts_csv, compute_final_corrected

    spectrum = GammaSpectrum(counts=[1, 2], counts_covariance=sparse.eye(2))
    destination = tmp_path / "existing.csv"
    destination.write_text("preserve me")
    with pytest.raises(ValueError, match="counts_covariance"):
        save_counts_csv(spectrum, spectrum, None, [1, 2], [1, 1], destination)
    assert destination.read_text() == "preserve me"
    with pytest.raises(ValueError, match="counts_covariance"):
        compute_final_corrected(spectrum, None)


def test_roi_shared_sidebands_against_explicit_weight_vector():
    # ROI 2..4, inclusive sidebands 1..2 and 4..5; their shared endpoints
    # must be included once with net weights, not two independent variances.
    c = np.diag(np.arange(1., 8.)) + np.ones((7,7)) * 0.5
    s = GammaSpectrum(counts=[2,3,10,20,12,5,2], calibration={'energy':[0,1]},
                      counts_covariance=sparse.csr_matrix(c))
    r = analyze_roi_region(s, roi_bounds_keV=(2,4), sideband_width_keV=1)
    w = np.array([0,-.75,.25,1,.25,-.75,0])
    assert r.net_counts == pytest.approx(w @ s.counts)
    assert r.net_counts_uncertainty**2 == pytest.approx(w @ c @ w)
    gross = np.array([0,0,1,1,1,0,0])
    assert r.gross_counts_uncertainty**2 == pytest.approx(gross @ c @ gross)
    # Independent finite-difference check of the displayed centroid derivative.
    def centroid(y):
        continuum = np.linspace((y[1]+y[2])/2, (y[4]+y[5])/2, 3)
        weights=np.maximum(y[2:5]-continuum,0)
        return np.arange(2,5) @ weights / weights.sum()
    h=1e-4
    jac=np.array([(centroid(s.counts+h*np.eye(7)[i])-centroid(s.counts-h*np.eye(7)[i]))/(2*h) for i in range(7)])
    assert r.centroid_uncertainty_keV**2 == pytest.approx(jac @ c @ jac, rel=1e-6)
    with pytest.raises(ValueError, match='continuum uncertainty'):
        analyze_roi_region(s, roi_bounds_keV=(2,4), background_method='snip')


def test_gaussian_gls_against_analytic_jacobian_and_area_gradient():
    x=np.arange(61., dtype=float)
    amp,mu,sigma=80.,30.,3.
    g=np.exp(-.5*((x-mu)/sigma)**2)
    y=amp*g + .1*x + 12
    c=np.diag(np.full(61,9.)) + np.ones((61,61))*2
    result=fit_single_peak(x,y,30,fit_width=12, counts_covariance=sparse.csr_matrix(c))
    assert result.success
    sl=slice(18,43); xx=x[sl]; gg=g[sl]
    j=np.column_stack([gg,amp*gg*(xx-mu)/sigma**2,amp*gg*(xx-mu)**2/sigma**3,xx,np.ones(len(xx))])
    expected=np.linalg.inv(j.T @ np.linalg.solve(c[sl,sl],j))
    np.testing.assert_allclose(result.covariance,expected,rtol=2e-4,atol=1e-7)
    gradient=np.sqrt(2*np.pi)*np.array([sigma,0,amp,0,0])
    assert result.net_counts_uncertainty**2 == pytest.approx(gradient@expected@gradient,rel=2e-4)
    assert result.net_counts == pytest.approx(amp*sigma*np.sqrt(2*np.pi),rel=1e-5)
    activity,unc=calculate_activity(result.net_counts,result.net_counts_uncertainty,10,.2,0,.5,0)
    assert activity == pytest.approx(result.net_counts)
    assert unc == pytest.approx(result.net_counts_uncertainty)


def test_singular_fit_rejected_and_poisson_wrapper_rejected():
    x=np.arange(21.)
    y=10+50*np.exp(-.5*((x-10)/2)**2)
    c=sparse.csr_matrix(np.ones((21,21)))
    with pytest.raises(ValueError,match='Singular fit'):
        fit_single_peak(x,y,10,fit_width=8,counts_covariance=c)
    with pytest.raises(ValueError,match='correlated'):
        fit_roi_peak(x,y,(4,16),fitter_key='bayesian_gaussian',counts_covariance=c)


def test_shared_width_fit_area_uses_full_parameter_covariance():
    x=np.arange(81.)
    y=10+100*np.exp(-.5*((x-34)/3)**2)+75*np.exp(-.5*((x-43)/3)**2)
    c=sparse.diags(np.full(81,4.))+sparse.csr_matrix(np.ones((81,81)))
    results=fit_multiple_peaks(x,y,[34,43],fit_width=14,share_sigma=True,counts_covariance=c)
    assert len(results)==2 and all(r.success for r in results)
    for i,r in enumerate(results):
        gradient=np.zeros(7)
        gradient[2*i]=r.peak.sigma*np.sqrt(2*np.pi)
        gradient[4]=r.peak.amplitude*np.sqrt(2*np.pi)
        assert r.net_counts_uncertainty**2 == pytest.approx(gradient@r.covariance@gradient)
