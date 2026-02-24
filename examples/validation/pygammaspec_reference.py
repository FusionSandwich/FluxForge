import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "testing/PyGammaSpec/src")

from pygammaspec.analysis import peak_search  # type: ignore
from pygammaspec.spectrum import Calibration, GammaSpectrum  # type: ignore


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/pra_spectrum")
    output_dir.mkdir(parents=True, exist_ok=True)

    background = GammaSpectrum.from_PRA_histogram("testing/PyGammaSpec/docs/utils/background.txt", 25851)
    sample = GammaSpectrum.from_PRA_histogram("testing/PyGammaSpec/docs/utils/weak_radium.txt", 25851)
    calibration = Calibration.from_calibration_file("testing/PyGammaSpec/docs/utils/calibration.txt")

    spectrum = (sample - background).average_smoothing(10)
    spectrum.calibration = calibration

    peaks = peak_search(spectrum, prominence=0.001)

    rows = []
    for idx, (channel, counts, energy) in peaks.items():
        rows.append([idx, channel, counts, energy])

    with (output_dir / "pygammaspec_reference_peaks.json").open("w", encoding="utf-8") as handle:
        json.dump({"peaks": rows}, handle, indent=2)

    with (output_dir / "pygammaspec_reference_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peak_index", "channel", "counts", "energy_keV"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
