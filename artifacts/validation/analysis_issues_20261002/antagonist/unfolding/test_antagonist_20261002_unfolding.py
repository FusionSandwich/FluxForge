"""Adversarial regression expectations for remaining unfolding contracts."""
from unittest.mock import patch
import numpy as np
import pytest
from fluxforge.unfolding import MLSeedUnfolder
from fluxforge.solvers import rmle

@pytest.mark.parametrize('warm', [0, 8])
def test_ml_seed_keeps_estimate_and_acceptance_under_measurement_unit_change(warm):
    outputs=[]
    for units in [1.0,1e-20]:
        outputs.append(MLSeedUnfolder().unfold(
            np.array([3.0,7.0])*units,np.eye(2)*units,
            initial_flux=np.ones(2),measurement_uncertainty=np.array([.1,.2])*units,
            warm_start_iterations=warm))
    np.testing.assert_allclose(outputs[1].flux,outputs[0].flux,rtol=1e-10,atol=0)
    np.testing.assert_allclose(outputs[1].chi_squared,outputs[0].chi_squared,rtol=1e-10,atol=0)
    np.testing.assert_allclose(outputs[1].parameters_used['confidence_score'],
                               outputs[0].parameters_used['confidence_score'],rtol=1e-10,atol=0)


def test_gaussian_rmle_keeps_regularized_estimate_under_measurement_unit_change():
    outputs=[]
    for units in [1.0,1e-40]:
        outputs.append(rmle.rmle_unfolding(
            rmle.SpectrumData(counts=np.array([3.0,7.0])*units,
                              uncertainty=np.array([.1,.2])*units),
            rmle.ResponseMatrix(matrix=np.eye(2)*units),
            regularization=rmle.RegularizationType.TIKHONOV,
            reg_param=1.0,enforce_positivity=False))
    np.testing.assert_allclose(outputs[1].solution,outputs[0].solution,rtol=1e-10,atol=0)


def test_gaussian_nnls_fallback_meets_constrained_optimum_or_marks_nonconvergence():
    response=np.array([[1.0,1.0],[1.0,2.0]])
    observed=np.array([1.0,0.0])
    with patch.object(rmle.optimize,'nnls',side_effect=RuntimeError('forced NNLS failure')):
        result=rmle.rmle_unfolding(
            rmle.SpectrumData(counts=observed,uncertainty=np.ones(2)),
            rmle.ResponseMatrix(matrix=response),regularization=rmle.RegularizationType.NONE)
    if result.converged:
        np.testing.assert_allclose(result.solution,[.5,0],rtol=1e-10,atol=1e-10)
        gradient=response.T@(response@result.solution-observed)
        assert np.max(np.abs(result.solution*gradient)) < 1e-10
