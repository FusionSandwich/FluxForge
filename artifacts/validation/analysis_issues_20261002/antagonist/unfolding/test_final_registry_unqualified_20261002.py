"""Final independent registry uncertainty qualification checks."""
import numpy as np
import pytest
from fluxforge.unfolding import GravelUnfolder,MaxedUnfolder,MLSeedUnfolder,RMLEUnfolder

@pytest.mark.parametrize('factory',[GravelUnfolder,MaxedUnfolder,MLSeedUnfolder,RMLEUnfolder])
def test_registry_never_qualifies_unimplemented_estimator_uncertainty(factory):
    counts=np.array([25.,400.])
    result=factory().unfold(counts,np.eye(2),initial_flux=counts,
        measurement_uncertainty=np.sqrt(counts),max_iterations=20,
        regularization_type='l2',regularization_strength=0,auto_regularization=False)
    assert result.uncertainties is None
    assert result.parameters_used['uncertainty_qualified'] is False
    assert result.parameters_used['uncertainty_status']=='unavailable'
    assert result.parameters_used['uncertainty_unavailable_reason']
    assert factory.definition().supports_uncertainties is False
