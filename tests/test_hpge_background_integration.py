from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from fluxforge.analysis.hpge_processor import HPGeProcessor
from fluxforge.io.genie import read_genie_spectrum


RAFM_ROOT = Path(__file__).resolve().parents[1] / "examples" / "RAFM_irradiation"
SAMPLE = RAFM_ROOT / "raw_gamma_spec" / "flux_wires" / "Co-Cd-RAFM-1_25cm.ASC"
BACKGROUND = RAFM_ROOT / "background.ASC"


@pytest.mark.skipif(
    not SAMPLE.exists() or not BACKGROUND.exists(), reason="RAFM example data absent"
)
def test_hpge_processor_uses_measured_background_and_propagated_roi_uncertainty(
    independent_background_sum_variance,
):
    sample = read_genie_spectrum(SAMPLE)
    background = read_genie_spectrum(BACKGROUND)
    processor = HPGeProcessor()

    with pytest.warns(RuntimeWarning, match="requires non-negative counts"):
        corrected = processor.analyze(
            sample, known_isotopes=["Co-60"], background_spectrum=background
        )
    assert corrected.spectrum.metadata["background_subtraction"]["scale_factor"] == 9.0
    assert corrected.spectrum.metadata["background_subtraction"]["negative_bins"] > 0
    assert (corrected.spectrum.counts < 0.0).any()
    assert len(corrected.gamma_lines) == 2
    for line in corrected.gamma_lines:
        fit_lo, fit_hi = line.fit_result.fit_region
        mask = (corrected.spectrum.channels >= fit_lo) & (
            corrected.spectrum.channels <= fit_hi
        )
        roi_uncertainty = np.sqrt(
            independent_background_sum_variance(
                sample,
                background,
                mask.astype(float),
                9.0,
            )
        )
        assert roi_uncertainty > np.sqrt(
            np.sum(corrected.spectrum.counts_uncertainty[mask] ** 2)
        )
        assert line.net_counts_unc >= roi_uncertainty
        assert line.activity_unc >= line.activity * roi_uncertainty / line.net_counts

    with pytest.warns(RuntimeWarning, match="no background spectrum was provided"):
        uncorrected = processor.analyze(sample, known_isotopes=["Co-60"])
    np.testing.assert_array_equal(uncorrected.spectrum.counts, sample.counts)
    assert len(uncorrected.gamma_lines) == 2
