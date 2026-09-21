"""
SPE File Parser for HPGe Gamma Spectroscopy

This module provides standalone SPE file parsing for gamma spectra from
HPGe detectors. It supports both standard SPE format and "$" prefixed format
commonly used by ORTEC MAESTRO and similar software.

This is a standalone implementation that does not require PyNE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from fluxforge.core.count_covariance import (
    covariance_from_payload, covariance_to_payload, linear_variance,
    validate_count_covariance,
)

from fluxforge.core.calibration import (
    EnergyDeviationPair,
    apply_energy_deviation_pairs,
)


@dataclass
class GammaSpectrum:
    """
    Container for gamma spectrum data from HPGe detector.

    Attributes
    ----------
    counts : np.ndarray
        Channel counts array
    counts_uncertainty : Optional[np.ndarray]
        Per-channel 1-sigma uncertainty (sqrt(counts) if not supplied)
    channels : np.ndarray
        Channel numbers
    energies : Optional[np.ndarray]
        Energy values for each channel (if calibrated)
    live_time : float
        Live time in seconds
    real_time : float
        Real time in seconds
    start_time : Optional[datetime]
        Acquisition start time
    spectrum_id : str
        Spectrum identifier or filename
    detector_id : str
        Detector identifier
    calibration : Dict[str, Any]
        Energy and shape calibration parameters
    metadata : Dict[str, Any]
        Additional metadata from file

    Examples
    --------
    >>> from fluxforge.io.spe import read_spe_file
    >>> spectrum = read_spe_file("sample.spe")
    >>> print(f"Live time: {spectrum.live_time} s")
    >>> print(f"Total counts: {spectrum.counts.sum()}")
    """

    counts: np.ndarray
    counts_uncertainty: Optional[np.ndarray] = None
    channels: np.ndarray = field(default_factory=lambda: np.array([]))
    energies: Optional[np.ndarray] = None
    live_time: float = 0.0
    real_time: float = 0.0
    start_time: Optional[datetime] = None
    spectrum_id: str = ""
    detector_id: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "file"
    device_id: str = ""
    device_label: str = ""
    gps: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    counts_covariance: Optional[sparse.csr_matrix] = None

    def __post_init__(self):
        """Initialize derived fields."""
        self.counts = np.asarray(self.counts, dtype=float)

        if (np.any(self.counts < 0.0)
                and self.counts_uncertainty is None
                and self.counts_covariance is None):
            raise ValueError(
                "Signed counts require counts_uncertainty or counts_covariance"
            )

        if self.counts_covariance is not None:
            self.counts_covariance = validate_count_covariance(self.counts_covariance, len(self.counts))
            diagonal_unc = np.sqrt(self.counts_covariance.diagonal())
            if self.counts_uncertainty is None:
                self.counts_uncertainty = diagonal_unc

        if self.counts_uncertainty is None:
            self.counts_uncertainty = np.sqrt(np.maximum(self.counts, 0.0))
        else:
            self.counts_uncertainty = np.asarray(self.counts_uncertainty, dtype=float)
            if self.counts_uncertainty.shape != self.counts.shape:
                raise ValueError(
                    "counts_uncertainty must have the same shape as counts."
                )
            if (not np.all(np.isfinite(self.counts_uncertainty))
                    or np.any(self.counts_uncertainty < 0.0)):
                raise ValueError(
                    "counts_uncertainty must be finite and non-negative."
                )
            if (self.counts_covariance is not None
                    and not np.allclose(
                        self.counts_uncertainty,
                        diagonal_unc,
                        rtol=1e-10,
                        atol=1e-12,
                    )):
                raise ValueError("counts_uncertainty must match counts_covariance diagonal")

        raw_channels = np.asarray(self.channels)
        if len(raw_channels) == 0 and len(self.counts) > 0:
            self.channels = np.arange(len(self.counts))
        elif np.allclose(raw_channels, np.round(raw_channels)):
            self.channels = np.round(raw_channels).astype(int)
        else:
            self.channels = raw_channels.astype(float)

        # Apply energy calibration if available
        if self.energies is None and self.calibration:
            self.energies = self.calibrate_channels()

    def calibrate_channels(
        self, coefficients: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Apply energy calibration to channels.

        Parameters
        ----------
        coefficients : list of float, optional
            Polynomial coefficients [a0, a1, a2, ...] where
            E = a0 + a1*ch + a2*ch^2 + ...
            If None, uses self.calibration['energy']

        Returns
        -------
        np.ndarray
            Energy values for each channel in keV
        """
        if coefficients is None:
            coefficients = self.calibration.get("energy", [0.0, 1.0])

        energies = np.zeros_like(self.channels, dtype=float)
        for i, coeff in enumerate(coefficients):
            energies += coeff * (np.asarray(self.channels, dtype=float) ** i)

        return apply_energy_deviation_pairs(energies, self._deviation_pairs())

    def channel_to_energy(
        self, channel: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert channel number to energy using calibration."""
        coeffs = self.calibration.get("energy", [0.0, 1.0])
        result = np.polynomial.polynomial.polyval(
            np.asarray(channel, dtype=float), coeffs
        )
        corrected = apply_energy_deviation_pairs(result, self._deviation_pairs())
        if np.isscalar(channel):
            return float(np.asarray(corrected, dtype=float))
        return corrected

    def energy_to_channel(
        self, energy: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
        """Invert a finite, unambiguous calibration on the recorded channel domain.

        Linear calibrations retain extrapolation for out-of-range ROI bounds.
        Nonlinear calibrations require a monotone mapping on the stored domain.
        """
        values = np.asarray(energy, dtype=float)
        coeffs = np.asarray(self.calibration.get("energy", [0.0, 1.0]), dtype=float)
        if coeffs.ndim != 1 or not coeffs.size or not np.all(np.isfinite(coeffs)):
            raise ValueError("Energy calibration coefficients must be finite.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Target energies must be finite.")
        coeffs = np.trim_zeros(coeffs, trim="b")
        if coeffs.size < 2:
            raise ValueError("Constant energy calibration cannot be inverted.")
        pairs = self._deviation_pairs()
        if coeffs.size == 2 and not pairs:
            channels = (values - coeffs[0]) / coeffs[1]
        else:
            from scipy.optimize import brentq

            domain = np.asarray(self.channels, dtype=float)
            if domain.size < 2 or not np.all(np.isfinite(domain)):
                raise ValueError(
                    "Nonlinear inversion requires a finite channel domain."
                )
            lo, hi = float(np.min(domain)), float(np.max(domain))
            if lo == hi:
                raise ValueError("Nonlinear inversion requires distinct channels.")
            derivative = np.polynomial.polynomial.polyder(coeffs)
            roots = np.polynomial.polynomial.polyroots(derivative)
            turns = sorted(
                [lo, hi]
                + [
                    float(z.real)
                    for z in roots
                    if abs(z.imag) < 1e-10 and lo < z.real < hi
                ]
            )
            probes = np.asarray(
                [(left + right) / 2 for left, right in zip(turns, turns[1:])]
            )
            slopes = np.polynomial.polynomial.polyval(probes, derivative)
            if not (np.all(slopes > 0) or np.all(slopes < 0)):
                raise ValueError(
                    "Energy calibration is not monotone on the channel domain."
                )
            if pairs:
                anchors = np.array(
                    sorted((p.energy_keV, p.correction_keV) for p in pairs)
                )
                if not np.all(np.isfinite(anchors)):
                    raise ValueError("Energy deviation pairs must be finite.")
                if len(anchors) > 1 and (
                    np.any(np.diff(anchors[:, 0]) <= 0)
                    or np.any(np.diff(anchors.sum(axis=1)) <= 0)
                ):
                    raise ValueError(
                        "Energy deviation mapping must be strictly increasing."
                    )
            bounds = sorted(
                (float(self.channel_to_energy(lo)), float(self.channel_to_energy(hi)))
            )
            if np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError(
                    "Target energy is outside the calibrated channel domain."
                )
            channels = np.asarray(
                [
                    brentq(
                        lambda ch: float(self.channel_to_energy(ch)) - target, lo, hi
                    )
                    for target in values.ravel()
                ]
            ).reshape(values.shape)
        if not np.all(np.isfinite(channels)) or np.any(
            np.abs(channels) >= np.iinfo(np.int64).max
        ):
            raise ValueError("Inverted channel is outside the supported integer range.")
        rounded = np.rint(channels).astype(np.int64)
        return int(rounded) if values.ndim == 0 else rounded

    def _deviation_pairs(self) -> tuple[EnergyDeviationPair, ...]:
        raw_pairs = self.calibration.get("deviation_pairs") or ()
        pairs: list[EnergyDeviationPair] = []
        for raw in raw_pairs:
            if isinstance(raw, EnergyDeviationPair):
                pairs.append(raw)
                continue
            if isinstance(raw, dict):
                pairs.append(
                    EnergyDeviationPair(
                        energy_keV=float(raw.get("energy_keV", 0.0)),
                        correction_keV=float(raw.get("correction_keV", 0.0)),
                        label=str(raw.get("label", "")),
                    )
                )
        return tuple(pairs)

    def counts_in_range(
        self, e_min: float, e_max: float, use_energy: bool = True
    ) -> Tuple[float, float]:
        """
        Get total counts and uncertainty in energy/channel range.

        Parameters
        ----------
        e_min, e_max : float
            Range bounds (energy in keV if use_energy=True, else channels)
        use_energy : bool
            If True, interpret bounds as energy; else as channels

        Returns
        -------
        counts : float
            Total counts in range
        uncertainty : float
            Poisson uncertainty (sqrt(counts))
        """
        if use_energy:
            ch_min = self.energy_to_channel(e_min)
            ch_max = self.energy_to_channel(e_max)
        else:
            ch_min, ch_max = int(e_min), int(e_max)

        mask = (self.channels >= ch_min) & (self.channels <= ch_max)
        total = self.counts[mask].sum()
        total_unc = np.sqrt(self.linear_variance(mask.astype(float)))
        return float(total), float(total_unc)

    def linear_variance(self, weights) -> float:
        """Variance of a fixed weighted count sum, including correlations."""
        return linear_variance(weights, self.counts_uncertainty, self.counts_covariance)

    def covariance_matrix(self):
        """Return full count covariance, interpreting absent storage as diagonal."""
        if self.counts_covariance is not None:
            return self.counts_covariance.copy()
        return sparse.diags(self.counts_uncertainty ** 2, format="csr")

    def require_diagonal(self, operation):
        """Fail explicitly when a legacy quantitative operation lacks support."""
        if self.counts_covariance is not None:
            raise ValueError(f"{operation} does not support counts_covariance; use a covariance-aware workflow")

    @property
    def dead_time_fraction(self) -> float:
        """Calculate dead time fraction."""
        if self.real_time > 0:
            return 1.0 - (self.live_time / self.real_time)
        return 0.0

    @property
    def count_rate(self) -> float:
        """Calculate total count rate in counts per second."""
        if self.live_time > 0:
            return self.counts.sum() / self.live_time
        return 0.0

    @property
    def energy_calibration(self) -> tuple[float, ...]:
        """Compatibility alias for callers that expect an energy tuple."""

        values = self.calibration.get("energy", ())
        return tuple(float(value) for value in values)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        if self.counts_covariance is not None:
            checked = validate_count_covariance(self.counts_covariance, len(self.counts))
            if not np.allclose(self.counts_uncertainty ** 2, checked.diagonal(), rtol=1e-10, atol=1e-12):
                raise ValueError("counts_uncertainty must match counts_covariance diagonal")
        return {
            "counts": self.counts.tolist(),
            "counts_covariance": covariance_to_payload(self.counts_covariance),
            "counts_uncertainty": (
                self.counts_uncertainty.tolist()
                if self.counts_uncertainty is not None
                else None
            ),
            "channels": self.channels.tolist(),
            "energies": self.energies.tolist() if self.energies is not None else None,
            "live_time": self.live_time,
            "real_time": self.real_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "spectrum_id": self.spectrum_id,
            "detector_id": self.detector_id,
            "calibration": self.calibration,
            "source_type": self.source_type,
            "device_id": self.device_id,
            "device_label": self.device_label,
            "gps": self.gps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GammaSpectrum":
        """Create GammaSpectrum from dictionary."""
        if (np.any(np.asarray(data["counts"]) < 0) and data.get("counts_uncertainty") is None
                and data.get("counts_covariance") is None):
            raise ValueError("Signed counts require counts_uncertainty or counts_covariance")
        return cls(
            counts=np.array(data["counts"]),
            counts_covariance=covariance_from_payload(data.get("counts_covariance"), len(data["counts"])),
            counts_uncertainty=(
                np.array(data["counts_uncertainty"])
                if data.get("counts_uncertainty") is not None
                else None
            ),
            channels=np.array(data.get("channels", [])),
            energies=np.array(data["energies"]) if data.get("energies") else None,
            live_time=data.get("live_time", 0.0),
            real_time=data.get("real_time", 0.0),
            start_time=(
                datetime.fromisoformat(data["start_time"])
                if data.get("start_time")
                else None
            ),
            spectrum_id=data.get("spectrum_id", ""),
            detector_id=data.get("detector_id", ""),
            calibration=data.get("calibration", {}),
            source_type=data.get("source_type", "file"),
            device_id=data.get("device_id", ""),
            device_label=data.get("device_label", ""),
            gps=data.get("gps", {}),
            metadata=data.get("metadata", {}),
        )


def read_spe_file(
    filepath: Union[str, Path], format_hint: Optional[str] = None
) -> GammaSpectrum:
    """
    Read SPE file and return GammaSpectrum object.

    Automatically detects SPE format (standard or "$" prefixed).

    Parameters
    ----------
    filepath : str or Path
        Path to SPE file
    format_hint : str, optional
        Force format: 'standard', 'dollar', or None for auto-detect

    Returns
    -------
    GammaSpectrum
        Parsed spectrum data

    Examples
    --------
    >>> spectrum = read_spe_file("Na22_calibration.spe")
    >>> print(f"Channels: {len(spectrum.counts)}")
    >>> print(f"Live time: {spectrum.live_time} s")

    Notes
    -----
    SPE file format sections:
    - $SPEC_ID: Spectrum identifier
    - $DATE_MEA: Measurement date
    - $MEAS_TIM: Live and real time
    - $DATA: Channel counts
    - $ENER_FIT: Energy calibration coefficients
    - $MCA_CAL: Multi-channel analyzer calibration
    - $SHAPE_CAL: Peak shape calibration
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8-sig", errors="strict") as f:
        content = f.read()

    # Auto-detect format
    if format_hint is None:
        if content.lstrip().startswith("$"):
            format_hint = "dollar"
        else:
            format_hint = "standard"

    if format_hint not in {"dollar", "standard"}:
        raise ValueError(f"Unsupported SPE format: {format_hint}")
    if format_hint == "dollar":
        return _parse_dollar_spe(content, str(filepath))
    else:
        return _parse_standard_spe(content, str(filepath))


def _finite_spe_numbers(text: str, section: str) -> np.ndarray:
    try:
        values = np.asarray([float(token) for token in text.split()], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric token in SPE {section}.") from exc
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"SPE {section} requires finite numeric values.")
    return values


def _parse_dollar_spe(content: str, filename: str) -> GammaSpectrum:
    """
    Parse "$" prefixed SPE format (ORTEC MAESTRO style).

    Format example:
        $SPEC_ID:
        Sample spectrum
        $DATE_MEA:
        10/15/2024 14:30:00
        $MEAS_TIM:
        3600 3650
        $DATA:
        0 8191
        0
        15
        42
        ...
        $ENER_FIT:
        0.000000E+000 5.000000E-001
    """
    sections = {}
    current_section = None
    section_content = []

    for line in content.split("\n"):
        line = line.strip()

        if line.startswith("$"):
            # Save previous section
            if current_section:
                sections[current_section] = section_content

            # Start new section
            current_section = line.rstrip(":")
            section_content = []
        elif current_section:
            section_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = section_content

    # Parse spectrum ID
    spectrum_id = ""
    if "$SPEC_ID" in sections:
        spectrum_id = " ".join(sections["$SPEC_ID"]).strip()

    # Parse date
    start_time = None
    if "$DATE_MEA" in sections and sections["$DATE_MEA"]:
        date_str = sections["$DATE_MEA"][0].strip()
        for fmt in [
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%m-%d-%Y %H:%M:%S",
        ]:
            try:
                start_time = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

    # Parse times
    live_time = 0.0
    real_time = 0.0
    if "$MEAS_TIM" in sections and sections["$MEAS_TIM"]:
        times = sections["$MEAS_TIM"][0].split()
        if len(times) >= 2:
            live_time = float(times[0])
            real_time = float(times[1])
        elif len(times) == 1:
            live_time = real_time = float(times[0])

    data_lines = [line for line in sections.get("$DATA", []) if line.strip()]
    if not data_lines:
        raise ValueError("SPE is missing a nonempty $DATA section.")
    parts = data_lines[0].split()
    if len(parts) != 2:
        raise ValueError("SPE $DATA requires an inclusive start/end channel range.")
    try:
        start_channel, end_channel = map(int, parts)
    except ValueError as exc:
        raise ValueError("SPE channel bounds must be integers.") from exc
    if start_channel < 0 or end_channel < start_channel:
        raise ValueError("SPE channel range is invalid.")
    counts = _finite_spe_numbers(" ".join(data_lines[1:]), "$DATA")
    if len(counts) != end_channel - start_channel + 1:
        raise ValueError("SPE count length does not match its declared channel range.")
    channels = np.arange(start_channel, end_channel + 1)

    calibration = {}
    if "$ENER_FIT" in sections:
        calibration["energy"] = _finite_spe_numbers(
            " ".join(sections["$ENER_FIT"]), "$ENER_FIT"
        ).tolist()
    for section, key in (("$MCA_CAL", "energy"), ("$SHAPE_CAL", "shape")):
        if section in sections:
            rows = [line for line in sections[section] if line.strip()]
            if len(rows) < 2 or not re.fullmatch(r"[0-9]+", rows[0]):
                raise ValueError(
                    f"{section} requires a coefficient count followed by coefficients."
                )
            text = " ".join(rows[1:])
            if section == "$MCA_CAL":
                text = re.sub(r"\s+keV\s*$", "", text, flags=re.IGNORECASE)
            coefficients = _finite_spe_numbers(text, section)
            if len(coefficients) != int(rows[0]):
                raise ValueError(
                    f"{section} coefficient count does not match its header."
                )
            calibration.setdefault(key, coefficients.tolist())

    # Collect metadata
    metadata = {
        key.lstrip("$"): " ".join(val)
        for key, val in sections.items()
        if key
        not in [
            "$DATA",
            "$ENER_FIT",
            "$MCA_CAL",
            "$SHAPE_CAL",
            "$SPEC_ID",
            "$DATE_MEA",
            "$MEAS_TIM",
        ]
    }

    # Parse detector ID if available
    detector_id = ""
    if "$DET_ID" in sections:
        detector_id = " ".join(sections["$DET_ID"]).strip()

    return GammaSpectrum(
        counts=counts,
        channels=channels,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        spectrum_id=spectrum_id or Path(filename).stem,
        detector_id=detector_id,
        calibration=calibration,
        metadata=metadata,
    )


def _parse_standard_spe(content: str, filename: str) -> GammaSpectrum:
    """
    Parse standard SPE format (numeric header style).

    Format varies but typically:
    - Header lines with counts, times
    - Followed by channel data
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    markers = [
        i
        for i, line in enumerate(lines)
        if line.lower() in {"data:", "counts:", "spectrum:"}
    ]
    live_time = real_time = 0.0
    calibration = {}
    if markers:
        index = markers[0]
        for line in lines[:index]:
            label, _, value = line.partition(":")
            if label.lower() in {"live time", "real time"}:
                parsed = _finite_spe_numbers(value, label)
                if len(parsed) != 1:
                    raise ValueError("SPE timing headers require one value.")
                if label.lower() == "live time":
                    live_time = float(parsed[0])
                else:
                    real_time = float(parsed[0])
            elif label.lower() in {"energy", "calibration"}:
                calibration["energy"] = _finite_spe_numbers(value, label).tolist()
            else:
                raise ValueError(f"Unsupported untagged SPE header: {line}")
        lines = lines[index + 1 :]
    counts = _finite_spe_numbers(" ".join(lines), "data")
    return GammaSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        spectrum_id=Path(filename).stem,
        calibration=calibration,
    )


def write_spe_file(
    spectrum: GammaSpectrum, filepath: Union[str, Path], format_type: str = "dollar"
) -> None:
    """
    Write GammaSpectrum to SPE file.

    Parameters
    ----------
    spectrum : GammaSpectrum
        Spectrum to write
    filepath : str or Path
        Output file path
    format_type : str
        Format type: 'dollar' for ORTEC-style
    """
    if format_type != "dollar":
        raise ValueError("Only dollar-tagged SPE export is supported.")
    spectrum.require_diagonal("SPE export")
    counts = np.asarray(spectrum.counts, dtype=float)
    channels = np.asarray(spectrum.channels, dtype=float)
    if counts.ndim != 1 or not counts.size or not np.all(np.isfinite(counts)):
        raise ValueError("SPE export requires a nonempty finite count vector.")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError(
            "SPE export requires nonnegative integer counts; use a spectrum artifact for processed data."
        )
    if (
        channels.shape != counts.shape
        or not np.all(np.isfinite(channels))
        or np.any(channels < 0)
        or np.any(channels != np.floor(channels))
        or np.any(np.diff(channels) != 1)
    ):
        raise ValueError("SPE export requires contiguous nonnegative integer channels.")
    if not np.allclose(
        np.asarray(spectrum.counts_uncertainty), np.sqrt(counts), rtol=1e-12, atol=0
    ):
        raise ValueError(
            "SPE cannot store custom count uncertainties; use a spectrum artifact."
        )
    if spectrum.calibration.get("deviation_pairs"):
        raise ValueError(
            "SPE cannot store energy deviation pairs; use a spectrum artifact."
        )
    for key in ("energy", "shape"):
        if key in spectrum.calibration:
            values = np.asarray(spectrum.calibration[key], dtype=float)
            if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid SPE {key} calibration coefficients.")
    for value in (spectrum.spectrum_id, spectrum.detector_id):
        if any(c in value for c in "\r\n"):
            raise ValueError("SPE identifiers cannot contain line breaks.")
    if any(
        not np.isfinite(t) or t < 0 for t in (spectrum.live_time, spectrum.real_time)
    ):
        raise ValueError("SPE times must be finite and nonnegative.")
    filepath = Path(filepath)

    lines = []

    # Spectrum ID
    lines.append("$SPEC_ID:")
    lines.append(spectrum.spectrum_id or filepath.stem)

    # Detector ID
    if spectrum.detector_id:
        lines.append("$DET_ID:")
        lines.append(spectrum.detector_id)

    # Date
    lines.append("$DATE_MEA:")
    if spectrum.start_time:
        lines.append(spectrum.start_time.strftime("%m/%d/%Y %H:%M:%S"))
    else:
        lines.pop()  # A missing acquisition time must remain missing.

    # Times
    lines.append("$MEAS_TIM:")
    lines.append(f"{spectrum.live_time:.17g} {spectrum.real_time:.17g}")

    # Data
    lines.append("$DATA:")
    start_ch = int(spectrum.channels[0]) if len(spectrum.channels) > 0 else 0
    end_ch = (
        int(spectrum.channels[-1])
        if len(spectrum.channels) > 0
        else len(spectrum.counts) - 1
    )
    lines.append(f"{start_ch} {end_ch}")

    for count in spectrum.counts:
        lines.append(f"{int(count)}")

    # Energy calibration
    if "energy" in spectrum.calibration:
        lines.append("$ENER_FIT:")
        coeffs = spectrum.calibration["energy"]
        lines.append(" ".join(f"{c:.17g}" for c in coeffs))

        lines.append("$MCA_CAL:")
        lines.append(str(len(coeffs)))
        lines.append(" ".join(f"{c:.17g}" for c in coeffs) + " keV")

    # Shape calibration
    if "shape" in spectrum.calibration:
        lines.append("$SHAPE_CAL:")
        coeffs = spectrum.calibration["shape"]
        lines.append(str(len(coeffs)))
        lines.append(" ".join(f"{c:.17g}" for c in coeffs))

    # End marker
    lines.append("$ENDRECORD:")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_multiple_spe(
    filepaths: List[Union[str, Path]], sum_spectra: bool = False
) -> Union[List[GammaSpectrum], GammaSpectrum]:
    """
    Read multiple SPE files.

    Parameters
    ----------
    filepaths : list of str or Path
        Paths to SPE files
    sum_spectra : bool
        If True, return summed spectrum

    Returns
    -------
    list of GammaSpectrum or GammaSpectrum
        Parsed spectra (or summed if sum_spectra=True)
    """
    spectra = [read_spe_file(fp) for fp in filepaths]

    if sum_spectra and len(spectra) > 0:
        # Sum counts, add times
        total_counts = np.zeros_like(spectra[0].counts)
        total_live = 0.0
        total_real = 0.0

        for sp in spectra:
            first = spectra[0]
            if (
                not np.array_equal(sp.channels, first.channels)
                or sp.calibration != first.calibration
            ):
                raise ValueError(
                    "SPE summation requires identical channel grids and calibrations."
                )
            total_counts += sp.counts
            total_live += sp.live_time
            total_real += sp.real_time

        return GammaSpectrum(
            counts=total_counts,
            counts_uncertainty=np.sqrt(sum(sp.counts_uncertainty**2 for sp in spectra)),
            channels=spectra[0].channels.copy(),
            live_time=total_live,
            real_time=total_real,
            spectrum_id="summed_spectrum",
            calibration=spectra[0].calibration.copy(),
            metadata={"source_files": [str(fp) for fp in filepaths]},
        )

    return spectra
