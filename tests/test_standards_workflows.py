import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge.physics.dose import gamma_dose_rate
from fluxforge.physics.decay_chain import DecayChain
from fluxforge.physics.stacked_target import StackedTarget
from fluxforge.physics.stopping_power import Projectile
from fluxforge.solvers.advanced import PPPCorrectionMethod, apply_ppp_correction
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder


def test_iaea_irdff_unfolder_initializes_with_energy_groups():
    unfolder = SpectrumUnfolder(verbose=False)

    assert unfolder.n_groups > 0
    assert unfolder.energy_edges[0] < unfolder.energy_edges[-1]
    assert unfolder.irdff_db is not None


def test_iaea_ppp_correction_preserves_covariance_shape():
    measurements = np.array([100.0, 120.0, 95.0])
    covariance = np.diag([4.0, 9.0, 16.0])

    corrected_y, corrected_cov = apply_ppp_correction(
        measurements,
        covariance,
        method=PPPCorrectionMethod.CHIBA_SMITH,
    )

    assert corrected_y.shape == measurements.shape
    assert corrected_cov.shape == covariance.shape
    assert np.all(np.isfinite(corrected_y))
    assert np.all(np.diag(corrected_cov) > 0.0)


def test_curie_like_decay_chain_and_stacked_target_smoke():
    chain = DecayChain(
        "X-1",
        nuclide_data={
            "X-1": {"half_life_s": 10.0, "decay_products": {"Y-1": 1.0}},
            "Y-1": {"half_life_s": float("inf"), "decay_products": {}},
        },
    )
    result = chain.decay(initial_activity={"X-1": 100.0}, times=[0.0, 10.0])

    assert result.activities["X-1"][1] < result.activities["X-1"][0]
    assert result.atoms["Y-1"][1] >= result.atoms["Y-1"][0]

    stack = StackedTarget(projectile=Projectile.PROTON, beam_energy_MeV=15.0)
    stack.add_foil("aluminum", 50.0)
    stack.add_foil("copper", 25.0)
    energies = stack.calculate_energies()

    assert len(energies) == 2
    assert energies[1].energy_in_MeV < energies[0].energy_in_MeV


def test_curie_like_gamma_dose_rate_positive():
    dose = gamma_dose_rate(
        energy_keV=661.7,
        intensity=0.851,
        activity_Bq=1.0e6,
        distance_cm=30.0,
    )

    assert dose > 0.0
