"""Plot raw sample/background channels for the six original zero-net QG ROIs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluxforge.examples.rafm_workflow import (
    build_generic_gamma_library,
    default_paths,
    load_rafm_example_metadata,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_raw_asc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.root.resolve()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metadata = load_rafm_example_metadata(root / "examples/RAFM_irradiation")
    paths = default_paths(root / "examples/RAFM_irradiation")
    options = dict(
        energy_calibration_override=workflow_profile_energy_calibration(
            metadata.config
        ),
        profile_name=metadata.config["profile_name"],
    )
    library, _ = build_generic_gamma_library(metadata)
    background = read_raw_asc(paths.background_path, **options)
    reports = {row["report"]: row for row in audit["reports"]}
    rows = [row for row in audit["rows"] if row["status"] == "reference_nondetection"]
    assert len(rows) == 6
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        raw = reports[row["report"]]["raw_file"]
        if raw is None:
            raise ValueError("Reference-zero plot requires all six raw acquisitions")
        data = read_raw_asc(root / raw, **options)
        line = min(
            (line for line in library if line.isotope == row["reported_isotope"]),
            key=lambda line: abs(line.energy_keV - row["reported_energy_keV"]),
        )
        energy = np.array([data.channel_to_energy(c) for c in data.spectrum.channels])
        lo = min(line.energy_keV, row["reported_energy_keV"]) - 8
        hi = max(line.energy_keV, row["reported_energy_keV"]) + 8
        window = (energy >= lo) & (energy <= hi)
        scaled = background.spectrum.counts * data.live_time / background.live_time
        axis.step(
            energy[window],
            data.spectrum.counts[window],
            where="mid",
            label="Observed sample",
        )
        axis.step(
            energy[window],
            scaled[window],
            where="mid",
            label="Measured background × live-time ratio",
        )
        axis.axvline(
            line.energy_keV,
            color="forestgreen",
            linestyle="--",
            label=f"Library {line.energy_keV:.3f} keV",
        )
        axis.axvline(
            row["reported_energy_keV"],
            color="firebrick",
            linestyle=":",
            label=f"QG zero ROI {row['reported_energy_keV']:.2f} keV",
        )
        axis.set(
            title=Path(raw).stem + " / " + line.isotope,
            xlabel="Energy (keV)",
            ylabel="Channel counts",
        )
        axis.legend(fontsize=8)
    figure.suptitle(
        "Six source zero-net ROIs: observed channel evidence\nNative counts retained; original reference centers preserved",
        fontsize=14,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
