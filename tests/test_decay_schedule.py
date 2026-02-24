import math

from fluxforge.physics.decay_inventory import (
    CountObservation,
    ProductionSegment,
    DecayNetwork,
    schedule_from_rates,
    evolve_with_schedule,
    fit_schedule_scale,
)
from fluxforge.physics.decay_library import DecayDataset


def test_fit_schedule_scale_recovers_factor():
    dataset = DecayDataset(
        half_lives_s={"X-1": 10.0, "Y-1": 1e30},
        atomic_masses_g_mol={"X-1": 1.0, "Y-1": 1.0},
        progeny={"X-1": ["Y-1"], "Y-1": []},
        branching={"X-1": [1.0], "Y-1": []},
        modes={"X-1": ["beta"], "Y-1": []},
    )

    network = DecayNetwork.from_roots(["X-1"], dataset)
    schedule = schedule_from_rates({"X-1": [(100.0, 10.0), (50.0, 30.0)]}, timestamp=True)

    atoms_end = evolve_with_schedule(network, atoms={}, schedule=schedule, units="s")
    hl = dataset.half_life_s("X-1")
    lam = math.log(2) / hl
    expected = atoms_end["X-1"] * (math.exp(-lam * 0.0) - math.exp(-lam * 5.0))

    scale_factor = 1.2
    obs = [
        CountObservation(
            nuclide="X-1",
            start=0.0,
            stop=5.0,
            decays=expected * scale_factor,
            uncertainty=expected * 0.05,
        )
    ]

    fitted = fit_schedule_scale(network, atoms={}, schedule=schedule, observations=obs, units="s")
    assert math.isclose(fitted["X-1"], scale_factor, rel_tol=0.1)
