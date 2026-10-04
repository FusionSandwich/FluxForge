"""GUI data models for FluxForge desktop app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint


@dataclass(frozen=True)
class StandardsGuiPreset:
    """Preset metadata for standards-oriented GUI workflows."""

    key: str
    label: str
    default_profile: str | None
    peaks_sensitivity: str
    peak_counting_method: str
    background_subtracted: bool
    reaction_category: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GuiSpectrumSeries:
    """One plotted spectrum series in the GUI preview."""

    label: str
    channels: np.ndarray
    energies_keV: np.ndarray
    counts: np.ndarray
    calibration_coeffs: tuple[float, ...] = ()


@dataclass(frozen=True)
class GuiSpectrumPeak:
    """Peak marker row shown in the GUI preview and table."""

    energy_keV: float
    area: float
    channel: int | None = None
    label: str = ""


@dataclass(frozen=True)
class GuiSpectrumPreview:
    """Spectrum preview state rendered by the GUI viewer tab."""

    primary: GuiSpectrumSeries
    overlays: tuple[GuiSpectrumSeries, ...] = ()
    peaks: tuple[GuiSpectrumPeak, ...] = ()


@dataclass(frozen=True)
class GuiManualRegion:
    """User-defined manual ROI region for GUI editing."""

    label: str
    left_keV: float
    right_keV: float
    notes: str = ""


@dataclass(frozen=True)
class GuiCalibrationPoint:
    """Manual calibration point selected in the GUI."""

    channel: float
    observed_energy_keV: float
    reference_energy_keV: float
    label: str = ""


@dataclass(frozen=True)
class GuiCalibrationFit:
    """Polynomial energy-calibration fit result."""

    coefficients: tuple[float, ...]
    fitted_keV: np.ndarray
    residuals_keV: np.ndarray
    r_squared: float


@dataclass(frozen=True)
class GuiPeakCountingResult:
    """Peak counting summary for the selected method."""

    method: str
    net_counts: float
    net_uncertainty: float
    gross_counts: float
    gross_uncertainty: float
    roi_bounds: tuple[int, int]
    diagnostic_channels: np.ndarray | None = None
    diagnostic_counts: np.ndarray | None = None
    diagnostic_model: np.ndarray | None = None
    diagnostic_residuals: np.ndarray | None = None


@dataclass(frozen=True)
class GuiDiagnosticSeries:
    """One diagnostic plot series."""

    label: str
    x: np.ndarray
    y: np.ndarray
    style: str = "line"
    color: str = "#1f77b4"


@dataclass(frozen=True)
class GuiDiagnosticPlot:
    """Diagnostic plot shown below the main spectrum."""

    title: str
    x_label: str
    y_label: str
    series: tuple[GuiDiagnosticSeries, ...]
    reference_y: float | None = None
    x_log: bool = False
    y_log: bool = False


@dataclass(frozen=True)
class GuiEfficiencyCalibrationPoint:
    """Efficiency calibration point captured from GUI workflows."""

    source_name: str
    reference_energy_keV: float
    measured_energy_keV: float
    net_counts: float
    live_time_s: float
    activity_bq: float
    emission_probability: float
    geometry_factor: float
    efficiency: float
    efficiency_uncertainty: float
    label: str = ""

    def to_efficiency_point(self) -> EfficiencyPoint:
        return EfficiencyPoint(
            energy_keV=self.reference_energy_keV,
            net_counts=self.net_counts,
            live_time_s=self.live_time_s,
            activity_bq=self.activity_bq,
            emission_probability=self.emission_probability,
            geometry_factor=self.geometry_factor,
            count_uncertainty=np.sqrt(max(self.net_counts, 1.0)),
        )
