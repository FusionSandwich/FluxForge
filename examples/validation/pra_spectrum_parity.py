import csv
import json
from pathlib import Path

import numpy as np

from fluxforge.analysis.spectrum_math import subtract_spectra, moving_average
from fluxforge.analysis.spectroscopy_tools import prominence_peaks
from fluxforge.io.calibration_text import read_pygammaspec_calibration, pygammaspec_to_calibration
from fluxforge.io.genie import read_genie_spectrum
from fluxforge.io.pra import read_pra_as_spectrum


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peak_index", "channel", "counts", "energy_keV"])
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _peaks_to_rows(peaks):
    rows = []
    for idx, peak in peaks.items():
        rows.append([idx, peak.channel, peak.counts, peak.energy_keV])
    return rows


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/pra_spectrum")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference dataset: PyGammaSpec PRA files
    background = read_pra_as_spectrum("testing/PyGammaSpec/docs/utils/background.txt", live_time_s=25851)
    sample = read_pra_as_spectrum("testing/PyGammaSpec/docs/utils/weak_radium.txt", live_time_s=25851)
    diff = subtract_spectra(sample, background)
    smoothed = moving_average(diff, width=10)

    cal_data = read_pygammaspec_calibration("testing/PyGammaSpec/docs/utils/calibration.txt")
    cal = pygammaspec_to_calibration(cal_data)

    peaks = prominence_peaks(
        smoothed.channels,
        smoothed.counts,
        prominence=0.001,
        calibration=cal,
    )
    rows = _peaks_to_rows(peaks)
    _write_csv(output_dir / "pygammaspec_peaks.csv", rows)
    _write_json(output_dir / "pygammaspec_peaks.json", {"peaks": rows})

    # UWNR dataset: RAFM1 raw gamma spectrum (Genie ASC)
    rafm_path = Path("rafm_irradiation_ldrd/raw_gamma_spec/RAFM1/RAFM1_Long_72h_EOI.ASC")
    if rafm_path.exists():
        rafm_spec = read_genie_spectrum(rafm_path)
        counts = rafm_spec.counts.astype(float)
        channels = rafm_spec.channels.astype(float)
        smooth = np.convolve(counts, np.ones(9), mode="valid") / 9.0
        smooth_channels = channels[4:-4]
        rafm_peaks = prominence_peaks(smooth_channels, smooth, prominence=10.0)
        rows = _peaks_to_rows(rafm_peaks)
        _write_csv(output_dir / "rafm_peaks.csv", rows)
        _write_json(output_dir / "rafm_peaks.json", {"peaks": rows})


if __name__ == "__main__":
    main()
