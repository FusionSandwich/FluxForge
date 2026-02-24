import json
from pathlib import Path

from fluxforge.physics.decay_library import DecayDataset
from fluxforge.physics.decay_inventory import (
    CountObservation,
    DecayNetwork,
    schedule_from_rates,
    evolve_with_schedule,
    fit_schedule_scale,
)


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/decay_schedule")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(
        "testing/radioactivedecay/radioactivedecay/icrp107_ame2020_nubase2020/decay_data.npz"
    )
    dataset = DecayDataset.from_radioactivedecay_npz(dataset_path)

    network = DecayNetwork.from_roots(["Ra-225"], dataset)
    schedule = schedule_from_rates({"Ra-225": [(9.0, 0.5), (2.0, 1.5), (5.0, 4.5)]}, timestamp=True)

    atoms_end = evolve_with_schedule(network, atoms={}, schedule=schedule, units="d")

    observations = [
        CountObservation(nuclide="Ra-225", start=5.0, stop=5.1, decays=1.0e5, uncertainty=2.0e4),
        CountObservation(nuclide="Ra-225", start=6.0, stop=6.1, decays=1.1e5, uncertainty=2.5e4),
    ]
    scales = fit_schedule_scale(network, atoms={}, schedule=schedule, observations=observations, units="d")

    payload = {
        "atoms_end": atoms_end,
        "scale_factors": scales,
        "observations": [obs.__dict__ for obs in observations],
    }
    (output_dir / "decay_schedule_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
