"""
ASTM INL Dosimetry Workflow.

Implements the ASTM/INL-aligned gamma spectrometry peak finding, 
ROI / net-peak area determination and activity data sources.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from fluxforge.analysis.peak_finders import SimplePeakFinder, savitzky_golay_smooth
from fluxforge.data.k0_library import get_k0_library_record


@dataclass
class DecayDataRecord:
    nuclide: str
    energy_keV: float
    emission_prob: float
    half_life_s: float
    source_name: str = "k0_library"


@dataclass
class DetectorCalibrationRecord:
    detector_id: str
    energy_poly: List[float]  # e.g. [intercept, slope]
    fwhm_poly: List[float]  # e.g. a + b * sqrt(E)
    efficiency_curve: Any


@dataclass
class PeakAreaResult:
    method: str
    classification: str
    roi_left: int
    roi_right: int
    centroid_keV: float
    net_area: float
    net_area_unc: float


@dataclass
class ActivityResult:
    nuclide: str
    line_energy_keV: float
    activity: float
    uncertainty: float
    reference_time: float
    net_area: float
    efficiency: float
    emission_prob: float


class INLDosimetryWorkflow:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def fwhm_at_energy(
        self, energy_keV: float, cal: DetectorCalibrationRecord
    ) -> float:
        # Assuming FWHM = p[0] + p[1] * sqrt(E)
        if energy_keV <= 0:
            return cal.fwhm_poly[0]
        return cal.fwhm_poly[0] + cal.fwhm_poly[1] * math.sqrt(energy_keV)

    def channel_to_energy(
        self, channel: float, cal: DetectorCalibrationRecord
    ) -> float:
        energy = 0.0
        for i, coeff in enumerate(cal.energy_poly):
            energy += coeff * (channel**i)
        return energy

    def energy_to_channel(self, energy: float, cal: DetectorCalibrationRecord) -> float:
        # Simple inversion for linear cal: E = p0 + p1*ch => ch = (E-p0)/p1
        return (energy - cal.energy_poly[0]) / cal.energy_poly[1]

    def covell_linear(
        self, raw_counts: np.ndarray, roi_left: int, roi_right: int
    ) -> Tuple[float, float]:
        # Simple Covell style linear background interpolation from edges
        if roi_right <= roi_left:
            return 0.0, 0.0

        N = roi_right - roi_left + 1
        G = sum(raw_counts[roi_left : roi_right + 1])

        # Estimate background from edges (simplest endpoints)
        B1 = float(raw_counts[roi_left])
        B2 = float(raw_counts[roi_right])
        B = (N / 2.0) * (B1 + B2)

        net = G - B
        unc = math.sqrt(max(G + B, 1.0))
        return net, unc

    def analyze_spectrum(
        self, raw_counts: np.ndarray, cal: DetectorCalibrationRecord
    ) -> List[PeakAreaResult]:
        # Two-stream spectrum pipeline
        # Search copy is smoothed
        search_counts = savitzky_golay_smooth(raw_counts, window_size=5, order=2)

        # Detect peaks on SEARCH stream
        finder = SimplePeakFinder(threshold_sigma=4.0)
        seed_peaks = finder.find_peaks(search_counts)

        results = []
        for p in seed_peaks:
            ch = p.index
            energy_keV = self.channel_to_energy(ch, cal)
            # FWHM based ROI logic
            fwhm_E = self.fwhm_at_energy(energy_keV, cal)
            fwhm_ch = fwhm_E / cal.energy_poly[1]

            # ordinary singlet initial width ~ +/- 1.25 FWHM
            roi_width = int(math.ceil(1.25 * fwhm_ch))
            left = max(0, ch - roi_width)
            right = min(len(raw_counts) - 1, ch + roi_width)

            # Final net extracted from RAW stream using Covell
            net, unc = self.covell_linear(raw_counts, left, right)

            res = PeakAreaResult(
                method="covell_linear",
                classification="singlet",
                roi_left=left,
                roi_right=right,
                centroid_keV=energy_keV,
                net_area=net,
                net_area_unc=unc,
            )
            results.append(res)
        return results

    def compute_activity(
        self,
        peak: PeakAreaResult,
        cal: DetectorCalibrationRecord,
        live_time: float,
        nuclide: str,
        record: DecayDataRecord,
    ) -> ActivityResult:
        if peak.net_area <= 0:
            return None

        eff = (
            cal.efficiency_curve(peak.centroid_keV)
            if callable(cal.efficiency_curve)
            else cal.efficiency_curve.efficiency(peak.centroid_keV)
        )
        I_gamma = record.emission_prob

        # A = N / (eff * I * t)
        activity = peak.net_area / (eff * I_gamma * live_time)
        unc = (peak.net_area_unc / peak.net_area) * activity if peak.net_area > 0 else 0

        return ActivityResult(
            nuclide=nuclide,
            line_energy_keV=peak.centroid_keV,
            activity=activity,
            uncertainty=unc,
            reference_time=0.0,
            net_area=peak.net_area,
            efficiency=eff,
            emission_prob=I_gamma,
        )
