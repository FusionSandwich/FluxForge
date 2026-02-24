import json
import runpy
from pathlib import Path

import numpy as np

from fluxforge.analysis.peak_finders import (
    ScipyPeakFinder,
    WaveletPeakFinder,
    RelativeExtremaPeakFinder,
    DirectScipyPeakFinder,
)
from fluxforge.io.genie import read_genie_spectrum


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/pyfindpeaks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference dataset: py-findpeaks vector
    data = runpy.run_path("testing/py-findpeaks/tests/vector.py")
    vector = np.array(data["vector"], dtype=float)

    finders = {
        "direct_scipy": DirectScipyPeakFinder(),
        "scipy": ScipyPeakFinder(threshold_factor=0.0, prominence=1.0, distance=2, smooth_window=9),
        "wavelet": WaveletPeakFinder(),
        "argrelextrema": RelativeExtremaPeakFinder(order=2),
    }

    results = {}
    for name, finder in finders.items():
        peaks = finder.find_peaks(vector)
        results[name] = [p.index for p in peaks]

    _write_json(output_dir / "pyfindpeaks_vector_peaks.json", results)

    # UWNR dataset: RAFM1 raw spectrum
    rafm_path = Path("rafm_irradiation_ldrd/raw_gamma_spec/RAFM1/RAFM1_Long_72h_EOI.ASC")
    if rafm_path.exists():
        rafm = read_genie_spectrum(rafm_path)
        counts = rafm.counts.astype(float)
        rafm_results = {}
        for name, finder in finders.items():
            peaks = finder.find_peaks(counts)
            rafm_results[name] = [p.index for p in peaks[:20]]
        _write_json(output_dir / "rafm_peaks.json", rafm_results)


if __name__ == "__main__":
    main()
