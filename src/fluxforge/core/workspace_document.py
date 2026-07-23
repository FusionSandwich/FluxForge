"""Versioned, Qt-independent persisted analysis state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids core/io import cycles
    from fluxforge.io.spe import GammaSpectrum


WORKSPACE_DOCUMENT_SCHEMA = "fluxforge.workspace_document.v2"
WORKSPACE_DOCUMENT_VERSION = 2


class WorkspaceValidationError(ValueError):
    """A field-specific invalid workspace value."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_json(item) for key, item in dict(value or {}).items()}
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkspaceValidationError(path, "must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkspaceValidationError(path, "must be an array")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    non_string = [key for key in payload if not isinstance(key, str)]
    if non_string:
        raise WorkspaceValidationError(path, "object field names must be strings")
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise WorkspaceValidationError(
            path, "contains unknown fields: " + ", ".join(unknown)
        )


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceValidationError(path, "must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkspaceValidationError(path, "must be a string or null")
    text = value.strip()
    return text or None


def _text(value: Any, path: str) -> str:
    """Return a string without accepting or normalizing another JSON type."""

    if not isinstance(value, str):
        raise WorkspaceValidationError(path, "must be a string")
    return value


def _text_items(value: Any, path: str, *, non_empty: bool = True) -> tuple[str, ...]:
    items = _sequence(value, path)
    validator = _required_text if non_empty else _text
    return tuple(
        validator(item, f"{path}[{index}]") for index, item in enumerate(items)
    )


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WorkspaceValidationError(path, "must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise WorkspaceValidationError(path, "must be finite")
    if minimum is not None and result < minimum:
        raise WorkspaceValidationError(path, f"must be at least {minimum}")
    return result


def _optional_number(
    value: Any, path: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum)


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    """Return a strict integer, rejecting booleans and lossy conversions."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkspaceValidationError(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise WorkspaceValidationError(path, f"must be at least {minimum}")
    return value


def _pair(value: Any, path: str) -> tuple[float, float]:
    items = _sequence(value, path)
    if len(items) != 2:
        raise WorkspaceValidationError(path, "must contain exactly two values")
    return (_number(items[0], f"{path}[0]"), _number(items[1], f"{path}[1]"))


def _optional_pair(value: Any, path: str) -> tuple[float, float] | None:
    return None if value is None else _pair(value, path)


def _validate_range(value: tuple[float, float] | None, path: str) -> None:
    if value is None:
        return
    if not all(isfinite(float(item)) for item in value) or value[0] >= value[1]:
        raise WorkspaceValidationError(path, "must be a finite increasing range")


def _matrix(value: Any, path: str) -> tuple[tuple[float, ...], ...]:
    if value is None:
        return ()
    rows = _sequence(value, path)
    return tuple(
        tuple(
            _number(item, f"{path}[{row_index}][{column_index}]")
            for column_index, item in enumerate(_sequence(row, f"{path}[{row_index}]"))
        )
        for row_index, row in enumerate(rows)
    )


def _validate_covariance(value: tuple[tuple[float, ...], ...], path: str) -> None:
    if not value:
        return
    size = len(value)
    if any(len(row) != size for row in value):
        raise WorkspaceValidationError(path, "must be square")
    matrix = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise WorkspaceValidationError(path, "must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise WorkspaceValidationError(path, "must be symmetric")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1.0e-10
    if float(np.min(eigenvalues)) < -tolerance:
        raise WorkspaceValidationError(path, "must be positive semidefinite")


def _validate_json_finite(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkspaceValidationError(path, "object keys must be strings")
            _validate_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
    elif isinstance(value, float):
        if not isfinite(value):
            raise WorkspaceValidationError(path, "must be finite")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    else:
        raise WorkspaceValidationError(
            path,
            f"must contain JSON-compatible values, not {type(value).__name__}",
        )


def _validate_spectrum_payload(value: Any, path: str) -> dict[str, Any]:
    """Validate the persisted GammaSpectrum shape before its legacy reader runs."""

    data = dict(_mapping(value, path))
    allowed = {
        "counts",
        "counts_uncertainty",
        "channels",
        "energies",
        "live_time",
        "real_time",
        "start_time",
        "spectrum_id",
        "detector_id",
        "calibration",
        "source_type",
        "device_id",
        "device_label",
        "gps",
        "metadata",
    }
    _reject_unknown(data, allowed, path)
    if "counts" not in data:
        raise WorkspaceValidationError(f"{path}.counts", "is required")
    for name in ("counts", "channels"):
        for index, item in enumerate(_sequence(data.get(name, ()), f"{path}.{name}")):
            minimum = 0.0 if name == "counts" else None
            _number(item, f"{path}.{name}[{index}]", minimum=minimum)
    for name in ("counts_uncertainty", "energies"):
        raw = data.get(name)
        if raw is None:
            continue
        for index, item in enumerate(_sequence(raw, f"{path}.{name}")):
            minimum = 0.0 if name == "counts_uncertainty" else None
            _number(item, f"{path}.{name}[{index}]", minimum=minimum)
    _number(data.get("live_time", 0.0), f"{path}.live_time", minimum=0.0)
    _number(data.get("real_time", 0.0), f"{path}.real_time", minimum=0.0)
    _optional_text(data.get("start_time"), f"{path}.start_time")
    for name, default in (
        ("spectrum_id", ""),
        ("detector_id", ""),
        ("source_type", "file"),
        ("device_id", ""),
        ("device_label", ""),
    ):
        _text(data.get(name, default), f"{path}.{name}")
    for name in ("calibration", "gps", "metadata"):
        mapping = _mapping(data.get(name, {}), f"{path}.{name}")
        _validate_json_finite(mapping, f"{path}.{name}")
    return data


@dataclass(frozen=True)
class WorkspaceSpectrum:
    spectrum_id: str
    spectrum: "GammaSpectrum"
    label: str = ""
    source_path: str | None = None
    source_hash: str | None = None
    detector_profile_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))
        object.__setattr__(self, "extensions", _frozen_mapping(self.extensions))

    def validate(self, path: str = "spectra[]") -> None:
        _required_text(self.spectrum_id, f"{path}.spectrum_id")
        _text(self.label, f"{path}.label")
        _optional_text(self.source_path, f"{path}.source_path")
        _optional_text(self.source_hash, f"{path}.source_hash")
        _optional_text(self.detector_profile_id, f"{path}.detector_profile_id")
        if not hasattr(self.spectrum, "counts") or not hasattr(
            self.spectrum, "to_dict"
        ):
            raise WorkspaceValidationError(
                f"{path}.spectrum", "must be a GammaSpectrum"
            )
        counts = np.asarray(self.spectrum.counts, dtype=float)
        uncertainty = np.asarray(self.spectrum.counts_uncertainty, dtype=float)
        channels = np.asarray(self.spectrum.channels, dtype=float)
        if counts.ndim != 1 or not np.all(np.isfinite(counts)):
            raise WorkspaceValidationError(
                f"{path}.spectrum.counts", "must be a finite one-dimensional array"
            )
        if np.any(counts < 0.0):
            raise WorkspaceValidationError(
                f"{path}.spectrum.counts", "must not contain negative counts"
            )
        if (
            uncertainty.shape != counts.shape
            or np.any(uncertainty < 0.0)
            or not np.all(np.isfinite(uncertainty))
        ):
            raise WorkspaceValidationError(
                f"{path}.spectrum.counts_uncertainty",
                "must be finite, non-negative, and match counts",
            )
        if channels.shape != counts.shape or not np.all(np.isfinite(channels)):
            raise WorkspaceValidationError(
                f"{path}.spectrum.channels", "must be finite and match counts"
            )
        energies = getattr(self.spectrum, "energies", None)
        if energies is not None:
            energy_values = np.asarray(energies, dtype=float)
            if energy_values.shape != counts.shape or not np.all(
                np.isfinite(energy_values)
            ):
                raise WorkspaceValidationError(
                    f"{path}.spectrum.energies", "must be finite and match counts"
                )
        _number(self.spectrum.live_time, f"{path}.spectrum.live_time", minimum=0.0)
        _number(self.spectrum.real_time, f"{path}.spectrum.real_time", minimum=0.0)
        if self.spectrum.start_time is not None and not isinstance(
            self.spectrum.start_time, datetime
        ):
            raise WorkspaceValidationError(
                f"{path}.spectrum.start_time", "must be a datetime or null"
            )
        for name in (
            "spectrum_id",
            "detector_id",
            "source_type",
            "device_id",
            "device_label",
        ):
            _text(getattr(self.spectrum, name), f"{path}.spectrum.{name}")
        for name in ("calibration", "gps", "metadata"):
            mapping = _mapping(getattr(self.spectrum, name), f"{path}.spectrum.{name}")
            _validate_json_finite(mapping, f"{path}.spectrum.{name}")
        _validate_json_finite(self.provenance, f"{path}.provenance")
        _validate_json_finite(self.extensions, f"{path}.extensions")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "spectrum_id": self.spectrum_id,
            "label": self.label,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "detector_profile_id": self.detector_profile_id,
            "spectrum": self.spectrum.to_dict(),
            "provenance": _thaw_json(self.provenance),
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkspaceSpectrum":
        data = _mapping(payload, "spectrum")
        _reject_unknown(
            data,
            {
                "spectrum_id",
                "label",
                "source_path",
                "source_hash",
                "detector_profile_id",
                "spectrum",
                "provenance",
                "extensions",
            },
            "spectrum",
        )
        from fluxforge.io.spe import GammaSpectrum

        try:
            spectrum = GammaSpectrum.from_dict(
                _validate_spectrum_payload(data.get("spectrum"), "spectrum.spectrum")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkspaceValidationError("spectrum.spectrum", str(exc)) from exc
        result = cls(
            spectrum_id=_required_text(data.get("spectrum_id"), "spectrum.spectrum_id"),
            spectrum=spectrum,
            label=_text(data.get("label", ""), "spectrum.label"),
            source_path=_optional_text(data.get("source_path"), "spectrum.source_path"),
            source_hash=_optional_text(data.get("source_hash"), "spectrum.source_hash"),
            detector_profile_id=_optional_text(
                data.get("detector_profile_id"), "spectrum.detector_profile_id"
            ),
            provenance=_mapping(data.get("provenance", {}), "spectrum.provenance"),
            extensions=_mapping(data.get("extensions", {}), "spectrum.extensions"),
        )
        result.validate("spectrum")
        return result


@dataclass(frozen=True)
class SpectrumRoleAssignment:
    role: str
    spectrum_ids: tuple[str, ...]

    def validate(self, path: str = "spectrum_roles[]") -> None:
        _required_text(self.role, f"{path}.role")
        if not self.spectrum_ids:
            raise WorkspaceValidationError(f"{path}.spectrum_ids", "must not be empty")
        if len(set(self.spectrum_ids)) != len(self.spectrum_ids):
            raise WorkspaceValidationError(
                f"{path}.spectrum_ids", "must contain unique IDs"
            )
        for index, spectrum_id in enumerate(self.spectrum_ids):
            _required_text(spectrum_id, f"{path}.spectrum_ids[{index}]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"role": self.role, "spectrum_ids": list(self.spectrum_ids)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpectrumRoleAssignment":
        data = _mapping(payload, "spectrum_role")
        _reject_unknown(data, {"role", "spectrum_ids"}, "spectrum_role")
        ids = tuple(
            _required_text(value, f"spectrum_role.spectrum_ids[{index}]")
            for index, value in enumerate(
                _sequence(data.get("spectrum_ids"), "spectrum_role.spectrum_ids")
            )
        )
        result = cls(
            role=_required_text(data.get("role"), "spectrum_role.role"),
            spectrum_ids=ids,
        )
        result.validate("spectrum_role")
        return result


@dataclass(frozen=True)
class AnalysisROI:
    roi_id: str
    spectrum_id: str
    signal_range: tuple[float, float]
    left_background_range: tuple[float, float]
    right_background_range: tuple[float, float]
    associated_peak_ids: tuple[str, ...] = ()
    fit_revision: int = 0
    label: str = ""
    color: str = "#2dd4bf"

    def validate(self, path: str = "rois[]") -> None:
        _required_text(self.roi_id, f"{path}.roi_id")
        _required_text(self.spectrum_id, f"{path}.spectrum_id")
        _validate_range(self.signal_range, f"{path}.signal_range")
        _validate_range(self.left_background_range, f"{path}.left_background_range")
        _validate_range(self.right_background_range, f"{path}.right_background_range")
        if self.left_background_range[1] > self.signal_range[0]:
            raise WorkspaceValidationError(
                f"{path}.left_background_range", "must end before the signal range"
            )
        if self.right_background_range[0] < self.signal_range[1]:
            raise WorkspaceValidationError(
                f"{path}.right_background_range", "must start after the signal range"
            )
        _integer(self.fit_revision, f"{path}.fit_revision", minimum=0)
        if len(set(self.associated_peak_ids)) != len(self.associated_peak_ids):
            raise WorkspaceValidationError(
                f"{path}.associated_peak_ids", "must contain unique IDs"
            )
        for index, peak_id in enumerate(self.associated_peak_ids):
            _required_text(peak_id, f"{path}.associated_peak_ids[{index}]")
        if not isinstance(self.label, str):
            raise WorkspaceValidationError(f"{path}.label", "must be a string")
        _required_text(self.color, f"{path}.color")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "roi_id": self.roi_id,
            "spectrum_id": self.spectrum_id,
            "signal_range": list(self.signal_range),
            "left_background_range": list(self.left_background_range),
            "right_background_range": list(self.right_background_range),
            "associated_peak_ids": list(self.associated_peak_ids),
            "fit_revision": self.fit_revision,
            "label": self.label,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisROI":
        data = _mapping(payload, "roi")
        _reject_unknown(
            data,
            {
                "roi_id",
                "spectrum_id",
                "signal_range",
                "left_background_range",
                "right_background_range",
                "associated_peak_ids",
                "fit_revision",
                "label",
                "color",
            },
            "roi",
        )
        result = cls(
            roi_id=_required_text(data.get("roi_id"), "roi.roi_id"),
            spectrum_id=_required_text(data.get("spectrum_id"), "roi.spectrum_id"),
            signal_range=_pair(data.get("signal_range"), "roi.signal_range"),
            left_background_range=_pair(
                data.get("left_background_range"), "roi.left_background_range"
            ),
            right_background_range=_pair(
                data.get("right_background_range"), "roi.right_background_range"
            ),
            associated_peak_ids=_text_items(
                data.get("associated_peak_ids", ()), "roi.associated_peak_ids"
            ),
            fit_revision=_integer(
                data.get("fit_revision", 0), "roi.fit_revision", minimum=0
            ),
            label=(
                data.get("label", "")
                if isinstance(data.get("label", ""), str)
                else (_raise_validation("roi.label", "must be a string"))
            ),
            color=_required_text(data.get("color", "#2dd4bf"), "roi.color"),
        )
        result.validate("roi")
        return result


@dataclass(frozen=True)
class PeakComponent:
    component_id: str
    shape: str
    centroid: float
    area: float = 0.0
    amplitude: float = 0.0
    fwhm: float = 0.0
    uncertainty: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    def validate(self, path: str = "components[]") -> None:
        _required_text(self.component_id, f"{path}.component_id")
        _required_text(self.shape, f"{path}.shape")
        _number(self.centroid, f"{path}.centroid")
        for name in ("area", "amplitude", "fwhm", "uncertainty"):
            _number(getattr(self, name), f"{path}.{name}", minimum=0.0)
        _validate_json_finite(self.parameters, f"{path}.parameters")
        _validate_json_finite(self.provenance, f"{path}.provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "component_id": self.component_id,
            "shape": self.shape,
            "centroid": self.centroid,
            "area": self.area,
            "amplitude": self.amplitude,
            "fwhm": self.fwhm,
            "uncertainty": self.uncertainty,
            "parameters": _thaw_json(self.parameters),
            "provenance": _thaw_json(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeakComponent":
        data = _mapping(payload, "component")
        _reject_unknown(
            data,
            {
                "component_id",
                "shape",
                "centroid",
                "area",
                "amplitude",
                "fwhm",
                "uncertainty",
                "parameters",
                "provenance",
            },
            "component",
        )
        result = cls(
            component_id=_required_text(
                data.get("component_id"), "component.component_id"
            ),
            shape=_required_text(data.get("shape"), "component.shape"),
            centroid=_number(data.get("centroid"), "component.centroid"),
            area=_number(data.get("area", 0.0), "component.area", minimum=0.0),
            amplitude=_number(
                data.get("amplitude", 0.0), "component.amplitude", minimum=0.0
            ),
            fwhm=_number(data.get("fwhm", 0.0), "component.fwhm", minimum=0.0),
            uncertainty=_number(
                data.get("uncertainty", 0.0), "component.uncertainty", minimum=0.0
            ),
            parameters=_mapping(data.get("parameters", {}), "component.parameters"),
            provenance=_mapping(data.get("provenance", {}), "component.provenance"),
        )
        result.validate("component")
        return result


@dataclass(frozen=True)
class NuclideAssignment:
    nuclide: str
    line_energy_keV: float
    library_id: str = ""
    confidence: float | None = None
    manual: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    def validate(self, path: str = "assignments[]") -> None:
        _required_text(self.nuclide, f"{path}.nuclide")
        _text(self.library_id, f"{path}.library_id")
        _number(self.line_energy_keV, f"{path}.line_energy_keV", minimum=0.0)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise WorkspaceValidationError(f"{path}.confidence", "must be within 0..1")
        if not isinstance(self.manual, bool):
            raise WorkspaceValidationError(f"{path}.manual", "must be a boolean")
        _validate_json_finite(self.provenance, f"{path}.provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "nuclide": self.nuclide,
            "line_energy_keV": self.line_energy_keV,
            "library_id": self.library_id,
            "confidence": self.confidence,
            "manual": self.manual,
            "provenance": _thaw_json(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NuclideAssignment":
        data = _mapping(payload, "assignment")
        _reject_unknown(
            data,
            {
                "nuclide",
                "line_energy_keV",
                "library_id",
                "confidence",
                "manual",
                "provenance",
            },
            "assignment",
        )
        manual = data.get("manual", False)
        if not isinstance(manual, bool):
            raise WorkspaceValidationError("assignment.manual", "must be a boolean")
        result = cls(
            nuclide=_required_text(data.get("nuclide"), "assignment.nuclide"),
            line_energy_keV=_number(
                data.get("line_energy_keV"), "assignment.line_energy_keV", minimum=0.0
            ),
            library_id=_text(data.get("library_id", ""), "assignment.library_id"),
            confidence=_optional_number(
                data.get("confidence"), "assignment.confidence", minimum=0.0
            ),
            manual=manual,
            provenance=_mapping(data.get("provenance", {}), "assignment.provenance"),
        )
        result.validate("assignment")
        return result


@dataclass(frozen=True)
class PeakModel:
    peak_id: str
    spectrum_id: str
    roi_id: str | None
    centroid_channel: float
    centroid_energy_keV: float
    shape: str = "gaussian"
    components: tuple[PeakComponent, ...] = ()
    assignments: tuple[NuclideAssignment, ...] = ()
    tags: tuple[str, ...] = ()
    status: str = "candidate"
    manual_overrides: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    net_counts: float = 0.0
    significance: float = 0.0
    fit_quality: float = 0.0
    candidate_nuclides: tuple[str, ...] = ()
    reference_lines_keV: tuple[float, ...] = ()
    normalized_residuals: tuple[float, ...] = ()
    residual_channels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manual_overrides", _frozen_mapping(self.manual_overrides)
        )
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    def validate(self, path: str = "peaks[]") -> None:
        _required_text(self.peak_id, f"{path}.peak_id")
        _required_text(self.spectrum_id, f"{path}.spectrum_id")
        _optional_text(self.roi_id, f"{path}.roi_id")
        _required_text(self.shape, f"{path}.shape")
        _required_text(self.status, f"{path}.status")
        _number(self.centroid_channel, f"{path}.centroid_channel", minimum=0.0)
        _number(self.centroid_energy_keV, f"{path}.centroid_energy_keV", minimum=0.0)
        # Background-subtracted net areas may be negative.  Preserve them so an
        # analyst can review the invalid/non-detection state instead of clipping.
        _number(self.net_counts, f"{path}.net_counts")
        _number(self.significance, f"{path}.significance", minimum=0.0)
        _number(self.fit_quality, f"{path}.fit_quality", minimum=0.0)
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise WorkspaceValidationError(f"{path}.components", "IDs must be unique")
        for index, component in enumerate(self.components):
            component.validate(f"{path}.components[{index}]")
        for index, assignment in enumerate(self.assignments):
            assignment.validate(f"{path}.assignments[{index}]")
        for name, values in (
            ("tags", self.tags),
            ("candidate_nuclides", self.candidate_nuclides),
        ):
            for index, value in enumerate(values):
                _required_text(value, f"{path}.{name}[{index}]")
        if len(self.normalized_residuals) != len(self.residual_channels):
            raise WorkspaceValidationError(
                f"{path}.normalized_residuals",
                "must match residual_channels length",
            )
        for name, values in (
            ("reference_lines_keV", self.reference_lines_keV),
            ("normalized_residuals", self.normalized_residuals),
            ("residual_channels", self.residual_channels),
        ):
            for index, value in enumerate(values):
                _number(value, f"{path}.{name}[{index}]")
        _validate_json_finite(self.manual_overrides, f"{path}.manual_overrides")
        _validate_json_finite(self.provenance, f"{path}.provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "peak_id": self.peak_id,
            "spectrum_id": self.spectrum_id,
            "roi_id": self.roi_id,
            "centroid_channel": self.centroid_channel,
            "centroid_energy_keV": self.centroid_energy_keV,
            "shape": self.shape,
            "components": [item.to_dict() for item in self.components],
            "assignments": [item.to_dict() for item in self.assignments],
            "tags": list(self.tags),
            "status": self.status,
            "manual_overrides": _thaw_json(self.manual_overrides),
            "provenance": _thaw_json(self.provenance),
            "net_counts": self.net_counts,
            "significance": self.significance,
            "fit_quality": self.fit_quality,
            "candidate_nuclides": list(self.candidate_nuclides),
            "reference_lines_keV": list(self.reference_lines_keV),
            "normalized_residuals": list(self.normalized_residuals),
            "residual_channels": list(self.residual_channels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeakModel":
        data = _mapping(payload, "peak")
        allowed = {
            "peak_id",
            "spectrum_id",
            "roi_id",
            "centroid_channel",
            "centroid_energy_keV",
            "shape",
            "components",
            "assignments",
            "tags",
            "status",
            "manual_overrides",
            "provenance",
            "net_counts",
            "significance",
            "fit_quality",
            "candidate_nuclides",
            "reference_lines_keV",
            "normalized_residuals",
            "residual_channels",
        }
        _reject_unknown(data, allowed, "peak")
        result = cls(
            peak_id=_required_text(data.get("peak_id"), "peak.peak_id"),
            spectrum_id=_required_text(data.get("spectrum_id"), "peak.spectrum_id"),
            roi_id=_optional_text(data.get("roi_id"), "peak.roi_id"),
            centroid_channel=_number(
                data.get("centroid_channel"), "peak.centroid_channel", minimum=0.0
            ),
            centroid_energy_keV=_number(
                data.get("centroid_energy_keV"),
                "peak.centroid_energy_keV",
                minimum=0.0,
            ),
            shape=_required_text(data.get("shape", "gaussian"), "peak.shape"),
            components=tuple(
                PeakComponent.from_dict(item)
                for item in _sequence(data.get("components", ()), "peak.components")
            ),
            assignments=tuple(
                NuclideAssignment.from_dict(item)
                for item in _sequence(data.get("assignments", ()), "peak.assignments")
            ),
            tags=_text_items(data.get("tags", ()), "peak.tags"),
            status=_required_text(data.get("status", "candidate"), "peak.status"),
            manual_overrides=_mapping(
                data.get("manual_overrides", {}), "peak.manual_overrides"
            ),
            provenance=_mapping(data.get("provenance", {}), "peak.provenance"),
            net_counts=_number(data.get("net_counts", 0.0), "peak.net_counts"),
            significance=_number(
                data.get("significance", 0.0), "peak.significance", minimum=0.0
            ),
            fit_quality=_number(
                data.get("fit_quality", 0.0), "peak.fit_quality", minimum=0.0
            ),
            candidate_nuclides=_text_items(
                data.get("candidate_nuclides", ()), "peak.candidate_nuclides"
            ),
            reference_lines_keV=tuple(
                _number(item, f"peak.reference_lines_keV[{index}]")
                for index, item in enumerate(
                    _sequence(
                        data.get("reference_lines_keV", ()), "peak.reference_lines_keV"
                    )
                )
            ),
            normalized_residuals=tuple(
                _number(item, f"peak.normalized_residuals[{index}]")
                for index, item in enumerate(
                    _sequence(
                        data.get("normalized_residuals", ()),
                        "peak.normalized_residuals",
                    )
                )
            ),
            residual_channels=tuple(
                _number(item, f"peak.residual_channels[{index}]")
                for index, item in enumerate(
                    _sequence(
                        data.get("residual_channels", ()), "peak.residual_channels"
                    )
                )
            ),
        )
        result.validate("peak")
        return result


@dataclass(frozen=True)
class CalibrationModel:
    model_key: str
    coefficients: tuple[float, ...]
    calibration_points: tuple[tuple[float, float], ...] = ()
    deviation_pairs: tuple[tuple[float, float], ...] = ()
    covariance: tuple[tuple[float, ...], ...] = ()
    valid_range: tuple[float, float] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    def validate(self, path: str = "calibration") -> None:
        _required_text(self.model_key, f"{path}.model_key")
        if not self.coefficients:
            raise WorkspaceValidationError(f"{path}.coefficients", "must not be empty")
        for index, value in enumerate(self.coefficients):
            _number(value, f"{path}.coefficients[{index}]")
        for name, values in (
            ("calibration_points", self.calibration_points),
            ("deviation_pairs", self.deviation_pairs),
        ):
            for index, value in enumerate(values):
                if len(value) != 2:
                    raise WorkspaceValidationError(
                        f"{path}.{name}[{index}]", "must contain exactly two values"
                    )
                _number(value[0], f"{path}.{name}[{index}][0]")
                _number(value[1], f"{path}.{name}[{index}][1]")
        _validate_covariance(self.covariance, f"{path}.covariance")
        if self.covariance and len(self.covariance) != len(self.coefficients):
            raise WorkspaceValidationError(
                f"{path}.covariance", "dimension must match coefficients"
            )
        _validate_range(self.valid_range, f"{path}.valid_range")
        _validate_json_finite(self.provenance, f"{path}.provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "model_key": self.model_key,
            "coefficients": list(self.coefficients),
            "calibration_points": [list(item) for item in self.calibration_points],
            "deviation_pairs": [list(item) for item in self.deviation_pairs],
            "covariance": [list(row) for row in self.covariance],
            "valid_range": list(self.valid_range) if self.valid_range else None,
            "provenance": _thaw_json(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationModel":
        data = _mapping(payload, "calibration")
        _reject_unknown(
            data,
            {
                "model_key",
                "coefficients",
                "calibration_points",
                "deviation_pairs",
                "covariance",
                "valid_range",
                "provenance",
            },
            "calibration",
        )
        result = cls(
            model_key=_required_text(data.get("model_key"), "calibration.model_key"),
            coefficients=tuple(
                _number(item, f"calibration.coefficients[{index}]")
                for index, item in enumerate(
                    _sequence(data.get("coefficients"), "calibration.coefficients")
                )
            ),
            calibration_points=tuple(
                _pair(item, f"calibration.calibration_points[{index}]")
                for index, item in enumerate(
                    _sequence(
                        data.get("calibration_points", ()),
                        "calibration.calibration_points",
                    )
                )
            ),
            deviation_pairs=tuple(
                _pair(item, f"calibration.deviation_pairs[{index}]")
                for index, item in enumerate(
                    _sequence(
                        data.get("deviation_pairs", ()), "calibration.deviation_pairs"
                    )
                )
            ),
            covariance=_matrix(data.get("covariance", ()), "calibration.covariance"),
            valid_range=_optional_pair(
                data.get("valid_range"), "calibration.valid_range"
            ),
            provenance=_mapping(data.get("provenance", {}), "calibration.provenance"),
        )
        result.validate("calibration")
        return result


@dataclass(frozen=True)
class EfficiencyModelState:
    model_key: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    points: tuple[Mapping[str, Any], ...] = ()
    covariance: tuple[tuple[float, ...], ...] = ()
    uncertainty_model: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))
        object.__setattr__(
            self, "points", tuple(_frozen_mapping(item) for item in self.points)
        )
        object.__setattr__(
            self, "uncertainty_model", _frozen_mapping(self.uncertainty_model)
        )

    def validate(self, path: str = "efficiency_model") -> None:
        _required_text(self.model_key, f"{path}.model_key")
        _validate_json_finite(self.parameters, f"{path}.parameters")
        _validate_json_finite(self.points, f"{path}.points")
        _validate_covariance(self.covariance, f"{path}.covariance")
        _validate_json_finite(self.uncertainty_model, f"{path}.uncertainty_model")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "model_key": self.model_key,
            "parameters": _thaw_json(self.parameters),
            "points": [_thaw_json(item) for item in self.points],
            "covariance": [list(row) for row in self.covariance],
            "uncertainty_model": _thaw_json(self.uncertainty_model),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EfficiencyModelState":
        data = _mapping(payload, "efficiency_model")
        _reject_unknown(
            data,
            {"model_key", "parameters", "points", "covariance", "uncertainty_model"},
            "efficiency_model",
        )
        result = cls(
            model_key=_required_text(
                data.get("model_key"), "efficiency_model.model_key"
            ),
            parameters=_mapping(
                data.get("parameters", {}), "efficiency_model.parameters"
            ),
            points=tuple(
                _mapping(item, f"efficiency_model.points[{index}]")
                for index, item in enumerate(
                    _sequence(data.get("points", ()), "efficiency_model.points")
                )
            ),
            covariance=_matrix(
                data.get("covariance", ()), "efficiency_model.covariance"
            ),
            uncertainty_model=_mapping(
                data.get("uncertainty_model", {}),
                "efficiency_model.uncertainty_model",
            ),
        )
        result.validate("efficiency_model")
        return result


@dataclass(frozen=True)
class DetectorGeometry:
    crystal_diameter_cm: float | None = None
    crystal_length_cm: float | None = None
    dead_layer_um: float | None = None
    window_material: str | None = None
    window_thickness_um: float | None = None
    distance_cm: float | None = None
    angle_deg: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def validate(self, path: str = "geometry") -> None:
        _optional_text(self.window_material, f"{path}.window_material")
        for name in (
            "crystal_diameter_cm",
            "crystal_length_cm",
            "dead_layer_um",
            "window_thickness_um",
            "distance_cm",
        ):
            _optional_number(getattr(self, name), f"{path}.{name}", minimum=0.0)
        _optional_number(self.angle_deg, f"{path}.angle_deg")
        _validate_json_finite(self.metadata, f"{path}.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "crystal_diameter_cm": self.crystal_diameter_cm,
            "crystal_length_cm": self.crystal_length_cm,
            "dead_layer_um": self.dead_layer_um,
            "window_material": self.window_material,
            "window_thickness_um": self.window_thickness_um,
            "distance_cm": self.distance_cm,
            "angle_deg": self.angle_deg,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DetectorGeometry":
        data = _mapping(payload, "geometry")
        allowed = {
            "crystal_diameter_cm",
            "crystal_length_cm",
            "dead_layer_um",
            "window_material",
            "window_thickness_um",
            "distance_cm",
            "angle_deg",
            "metadata",
        }
        _reject_unknown(data, allowed, "geometry")
        result = cls(
            crystal_diameter_cm=_optional_number(
                data.get("crystal_diameter_cm"),
                "geometry.crystal_diameter_cm",
                minimum=0.0,
            ),
            crystal_length_cm=_optional_number(
                data.get("crystal_length_cm"),
                "geometry.crystal_length_cm",
                minimum=0.0,
            ),
            dead_layer_um=_optional_number(
                data.get("dead_layer_um"), "geometry.dead_layer_um", minimum=0.0
            ),
            window_material=_optional_text(
                data.get("window_material"), "geometry.window_material"
            ),
            window_thickness_um=_optional_number(
                data.get("window_thickness_um"),
                "geometry.window_thickness_um",
                minimum=0.0,
            ),
            distance_cm=_optional_number(
                data.get("distance_cm"), "geometry.distance_cm", minimum=0.0
            ),
            angle_deg=_optional_number(data.get("angle_deg"), "geometry.angle_deg"),
            metadata=_mapping(data.get("metadata", {}), "geometry.metadata"),
        )
        result.validate("geometry")
        return result


@dataclass(frozen=True)
class CorrectionSettings:
    attenuation_enabled: bool = False
    self_shielding_enabled: bool = False
    summing_enabled: bool = False
    dead_time_enabled: bool = True
    pileup_enabled: bool = False
    geometry_enabled: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))

    def validate(self, path: str = "corrections") -> None:
        for name in (
            "attenuation_enabled",
            "self_shielding_enabled",
            "summing_enabled",
            "dead_time_enabled",
            "pileup_enabled",
            "geometry_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise WorkspaceValidationError(f"{path}.{name}", "must be a boolean")
        _validate_json_finite(self.parameters, f"{path}.parameters")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "attenuation_enabled": self.attenuation_enabled,
            "self_shielding_enabled": self.self_shielding_enabled,
            "summing_enabled": self.summing_enabled,
            "dead_time_enabled": self.dead_time_enabled,
            "pileup_enabled": self.pileup_enabled,
            "geometry_enabled": self.geometry_enabled,
            "parameters": _thaw_json(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorrectionSettings":
        data = _mapping(payload, "corrections")
        allowed = {
            "attenuation_enabled",
            "self_shielding_enabled",
            "summing_enabled",
            "dead_time_enabled",
            "pileup_enabled",
            "geometry_enabled",
            "parameters",
        }
        _reject_unknown(data, allowed, "corrections")
        bool_values = {}
        defaults = {
            "attenuation_enabled": False,
            "self_shielding_enabled": False,
            "summing_enabled": False,
            "dead_time_enabled": True,
            "pileup_enabled": False,
            "geometry_enabled": True,
        }
        for key, default in defaults.items():
            value = data.get(key, default)
            if not isinstance(value, bool):
                raise WorkspaceValidationError(
                    f"corrections.{key}", "must be a boolean"
                )
            bool_values[key] = value
        result = cls(
            **bool_values,
            parameters=_mapping(data.get("parameters", {}), "corrections.parameters"),
        )
        result.validate("corrections")
        return result


@dataclass(frozen=True)
class DetectorProfile:
    detector_profile_id: str
    detector_id: str = ""
    energy_calibration: CalibrationModel | None = None
    fwhm_calibration: CalibrationModel | None = None
    efficiency_model: EfficiencyModelState | None = None
    geometry: DetectorGeometry = field(default_factory=DetectorGeometry)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    covariance: tuple[tuple[float, ...], ...] = ()
    corrections: CorrectionSettings = field(default_factory=CorrectionSettings)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uncertainty", _frozen_mapping(self.uncertainty))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))
        object.__setattr__(self, "extensions", _frozen_mapping(self.extensions))

    def validate(self, path: str = "detector_profiles[]") -> None:
        _required_text(self.detector_profile_id, f"{path}.detector_profile_id")
        _text(self.detector_id, f"{path}.detector_id")
        if self.energy_calibration:
            self.energy_calibration.validate(f"{path}.energy_calibration")
        if self.fwhm_calibration:
            self.fwhm_calibration.validate(f"{path}.fwhm_calibration")
        if self.efficiency_model:
            self.efficiency_model.validate(f"{path}.efficiency_model")
        self.geometry.validate(f"{path}.geometry")
        self.corrections.validate(f"{path}.corrections")
        _validate_covariance(self.covariance, f"{path}.covariance")
        _validate_json_finite(self.uncertainty, f"{path}.uncertainty")
        _validate_json_finite(self.provenance, f"{path}.provenance")
        _validate_json_finite(self.extensions, f"{path}.extensions")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "detector_profile_id": self.detector_profile_id,
            "detector_id": self.detector_id,
            "energy_calibration": (
                self.energy_calibration.to_dict() if self.energy_calibration else None
            ),
            "fwhm_calibration": (
                self.fwhm_calibration.to_dict() if self.fwhm_calibration else None
            ),
            "efficiency_model": (
                self.efficiency_model.to_dict() if self.efficiency_model else None
            ),
            "geometry": self.geometry.to_dict(),
            "uncertainty": _thaw_json(self.uncertainty),
            "covariance": [list(row) for row in self.covariance],
            "corrections": self.corrections.to_dict(),
            "provenance": _thaw_json(self.provenance),
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DetectorProfile":
        data = _mapping(payload, "detector_profile")
        allowed = {
            "detector_profile_id",
            "detector_id",
            "energy_calibration",
            "fwhm_calibration",
            "efficiency_model",
            "geometry",
            "uncertainty",
            "covariance",
            "corrections",
            "provenance",
            "extensions",
        }
        _reject_unknown(data, allowed, "detector_profile")
        result = cls(
            detector_profile_id=_required_text(
                data.get("detector_profile_id"),
                "detector_profile.detector_profile_id",
            ),
            detector_id=_text(
                data.get("detector_id", ""), "detector_profile.detector_id"
            ),
            energy_calibration=(
                CalibrationModel.from_dict(data["energy_calibration"])
                if data.get("energy_calibration") is not None
                else None
            ),
            fwhm_calibration=(
                CalibrationModel.from_dict(data["fwhm_calibration"])
                if data.get("fwhm_calibration") is not None
                else None
            ),
            efficiency_model=(
                EfficiencyModelState.from_dict(data["efficiency_model"])
                if data.get("efficiency_model") is not None
                else None
            ),
            geometry=DetectorGeometry.from_dict(data.get("geometry", {})),
            uncertainty=_mapping(
                data.get("uncertainty", {}), "detector_profile.uncertainty"
            ),
            covariance=_matrix(
                data.get("covariance", ()), "detector_profile.covariance"
            ),
            corrections=CorrectionSettings.from_dict(data.get("corrections", {})),
            provenance=_mapping(
                data.get("provenance", {}), "detector_profile.provenance"
            ),
            extensions=_mapping(
                data.get("extensions", {}), "detector_profile.extensions"
            ),
        )
        result.validate("detector_profile")
        return result


@dataclass(frozen=True)
class FitDiagnostics:
    diagnostic_id: str
    spectrum_id: str
    fit_revision: int
    status: str
    roi_id: str | None = None
    peak_id: str | None = None
    x: tuple[float, ...] = ()
    observed: tuple[float, ...] = ()
    model: tuple[float, ...] = ()
    uncertainty: tuple[float, ...] = ()
    normalized_residuals: tuple[float, ...] = ()
    goodness_of_fit: Mapping[str, Any] = field(default_factory=dict)
    warning_flags: tuple[str, ...] = ()
    method: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "goodness_of_fit", _frozen_mapping(self.goodness_of_fit)
        )
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    def validate(self, path: str = "fit_diagnostics[]") -> None:
        _required_text(self.diagnostic_id, f"{path}.diagnostic_id")
        _required_text(self.spectrum_id, f"{path}.spectrum_id")
        _optional_text(self.roi_id, f"{path}.roi_id")
        _optional_text(self.peak_id, f"{path}.peak_id")
        _text(self.method, f"{path}.method")
        if self.status not in {"valid", "invalid"}:
            raise WorkspaceValidationError(f"{path}.status", "must be valid or invalid")
        _integer(self.fit_revision, f"{path}.fit_revision", minimum=0)
        lengths = {
            len(self.x),
            len(self.observed),
            len(self.model),
            len(self.uncertainty),
            len(self.normalized_residuals),
        }
        if self.status == "valid" and (not self.observed or len(lengths) != 1):
            raise WorkspaceValidationError(
                path,
                "valid diagnostics require equal non-empty x/observed/model/uncertainty/residual arrays",
            )
        if self.status == "invalid" and any(
            (
                self.x,
                self.observed,
                self.model,
                self.uncertainty,
                self.normalized_residuals,
            )
        ):
            populated_lengths = {
                len(values)
                for values in (
                    self.x,
                    self.observed,
                    self.model,
                    self.uncertainty,
                    self.normalized_residuals,
                )
                if values
            }
            if len(populated_lengths) != 1:
                raise WorkspaceValidationError(
                    path, "populated diagnostic arrays must have equal lengths"
                )
        for name in ("x", "observed", "model", "uncertainty", "normalized_residuals"):
            for index, value in enumerate(getattr(self, name)):
                minimum = 0.0 if name == "uncertainty" else None
                _number(value, f"{path}.{name}[{index}]", minimum=minimum)
        for index, flag in enumerate(self.warning_flags):
            _required_text(flag, f"{path}.warning_flags[{index}]")
        _validate_json_finite(self.goodness_of_fit, f"{path}.goodness_of_fit")
        _validate_json_finite(self.provenance, f"{path}.provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "diagnostic_id": self.diagnostic_id,
            "spectrum_id": self.spectrum_id,
            "roi_id": self.roi_id,
            "peak_id": self.peak_id,
            "fit_revision": self.fit_revision,
            "status": self.status,
            "x": list(self.x),
            "observed": list(self.observed),
            "model": list(self.model),
            "uncertainty": list(self.uncertainty),
            "normalized_residuals": list(self.normalized_residuals),
            "goodness_of_fit": _thaw_json(self.goodness_of_fit),
            "warning_flags": list(self.warning_flags),
            "method": self.method,
            "provenance": _thaw_json(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FitDiagnostics":
        data = _mapping(payload, "fit_diagnostics")
        allowed = {
            "diagnostic_id",
            "spectrum_id",
            "roi_id",
            "peak_id",
            "fit_revision",
            "status",
            "x",
            "observed",
            "model",
            "uncertainty",
            "normalized_residuals",
            "goodness_of_fit",
            "warning_flags",
            "method",
            "provenance",
        }
        _reject_unknown(data, allowed, "fit_diagnostics")

        def values(name: str) -> tuple[float, ...]:
            return tuple(
                _number(item, f"fit_diagnostics.{name}[{index}]")
                for index, item in enumerate(
                    _sequence(data.get(name, ()), f"fit_diagnostics.{name}")
                )
            )

        result = cls(
            diagnostic_id=_required_text(
                data.get("diagnostic_id"), "fit_diagnostics.diagnostic_id"
            ),
            spectrum_id=_required_text(
                data.get("spectrum_id"), "fit_diagnostics.spectrum_id"
            ),
            roi_id=_optional_text(data.get("roi_id"), "fit_diagnostics.roi_id"),
            peak_id=_optional_text(data.get("peak_id"), "fit_diagnostics.peak_id"),
            fit_revision=_integer(
                data.get("fit_revision", 0),
                "fit_diagnostics.fit_revision",
                minimum=0,
            ),
            status=_required_text(
                data.get("status", "invalid"), "fit_diagnostics.status"
            ),
            x=values("x"),
            observed=values("observed"),
            model=values("model"),
            uncertainty=values("uncertainty"),
            normalized_residuals=values("normalized_residuals"),
            goodness_of_fit=_mapping(
                data.get("goodness_of_fit", {}),
                "fit_diagnostics.goodness_of_fit",
            ),
            warning_flags=_text_items(
                data.get("warning_flags", ()), "fit_diagnostics.warning_flags"
            ),
            method=_text(data.get("method", ""), "fit_diagnostics.method"),
            provenance=_mapping(
                data.get("provenance", {}), "fit_diagnostics.provenance"
            ),
        )
        result.validate("fit_diagnostics")
        return result


@dataclass(frozen=True)
class CanvasViewport:
    viewport_id: str
    spectrum_id: str | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    x_unit: str = "channel"
    y_unit: str = "counts"
    log_x: bool = False
    log_y: bool = False
    overlays: tuple[str, ...] = ()
    residual_mode: str = "off"
    selected_roi_id: str | None = None
    crosshair_enabled: bool = False
    labels_visible: bool = True

    def validate(self, path: str = "viewports[]") -> None:
        _required_text(self.viewport_id, f"{path}.viewport_id")
        _optional_text(self.spectrum_id, f"{path}.spectrum_id")
        _required_text(self.x_unit, f"{path}.x_unit")
        _required_text(self.y_unit, f"{path}.y_unit")
        _optional_text(self.selected_roi_id, f"{path}.selected_roi_id")
        _validate_range(self.x_range, f"{path}.x_range")
        _validate_range(self.y_range, f"{path}.y_range")
        if self.residual_mode not in {"off", "compact", "full"}:
            raise WorkspaceValidationError(
                f"{path}.residual_mode", "must be off, compact, or full"
            )
        for index, overlay in enumerate(self.overlays):
            _required_text(overlay, f"{path}.overlays[{index}]")
        for name in ("log_x", "log_y", "crosshair_enabled", "labels_visible"):
            if not isinstance(getattr(self, name), bool):
                raise WorkspaceValidationError(f"{path}.{name}", "must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "viewport_id": self.viewport_id,
            "spectrum_id": self.spectrum_id,
            "x_range": list(self.x_range) if self.x_range else None,
            "y_range": list(self.y_range) if self.y_range else None,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "overlays": list(self.overlays),
            "residual_mode": self.residual_mode,
            "selected_roi_id": self.selected_roi_id,
            "crosshair_enabled": self.crosshair_enabled,
            "labels_visible": self.labels_visible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanvasViewport":
        data = _mapping(payload, "viewport")
        allowed = {
            "viewport_id",
            "spectrum_id",
            "x_range",
            "y_range",
            "x_unit",
            "y_unit",
            "log_x",
            "log_y",
            "overlays",
            "residual_mode",
            "selected_roi_id",
            "crosshair_enabled",
            "labels_visible",
        }
        _reject_unknown(data, allowed, "viewport")
        bool_values = {}
        for key, default in (
            ("log_x", False),
            ("log_y", False),
            ("crosshair_enabled", False),
            ("labels_visible", True),
        ):
            value = data.get(key, default)
            if not isinstance(value, bool):
                raise WorkspaceValidationError(f"viewport.{key}", "must be a boolean")
            bool_values[key] = value
        result = cls(
            viewport_id=_required_text(data.get("viewport_id"), "viewport.viewport_id"),
            spectrum_id=_optional_text(data.get("spectrum_id"), "viewport.spectrum_id"),
            x_range=_optional_pair(data.get("x_range"), "viewport.x_range"),
            y_range=_optional_pair(data.get("y_range"), "viewport.y_range"),
            x_unit=_required_text(data.get("x_unit", "channel"), "viewport.x_unit"),
            y_unit=_required_text(data.get("y_unit", "counts"), "viewport.y_unit"),
            overlays=_text_items(data.get("overlays", ()), "viewport.overlays"),
            residual_mode=_required_text(
                data.get("residual_mode", "off"), "viewport.residual_mode"
            ),
            selected_roi_id=_optional_text(
                data.get("selected_roi_id"), "viewport.selected_roi_id"
            ),
            **bool_values,
        )
        result.validate("viewport")
        return result


@dataclass(frozen=True)
class WorkspaceDocument:
    document_id: str = "workspace"
    title: str = ""
    created_at: str = field(default_factory=_utc_timestamp)
    updated_at: str = field(default_factory=_utc_timestamp)
    spectra: tuple[WorkspaceSpectrum, ...] = ()
    spectrum_roles: tuple[SpectrumRoleAssignment, ...] = ()
    active_spectrum_id: str | None = None
    rois: tuple[AnalysisROI, ...] = ()
    peaks: tuple[PeakModel, ...] = ()
    detector_profiles: tuple[DetectorProfile, ...] = ()
    fit_diagnostics: tuple[FitDiagnostics, ...] = ()
    viewports: tuple[CanvasViewport, ...] = ()
    pinned_nuclides: tuple[str, ...] = ()
    nuclide_tags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    plot_settings: Mapping[str, Any] = field(default_factory=dict)
    workflow_state: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema: str = WORKSPACE_DOCUMENT_SCHEMA
    schema_version: int = WORKSPACE_DOCUMENT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nuclide_tags",
            MappingProxyType(
                {key: tuple(value) for key, value in dict(self.nuclide_tags).items()}
            ),
        )
        for name in ("plot_settings", "workflow_state", "provenance", "extensions"):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))

    def spectrum_by_id(self, spectrum_id: str) -> WorkspaceSpectrum | None:
        return next(
            (item for item in self.spectra if item.spectrum_id == spectrum_id), None
        )

    def roi_by_id(self, roi_id: str) -> AnalysisROI | None:
        return next((item for item in self.rois if item.roi_id == roi_id), None)

    def peak_by_id(self, spectrum_id: str, peak_id: str) -> PeakModel | None:
        return next(
            (
                item
                for item in self.peaks
                if item.spectrum_id == spectrum_id and item.peak_id == peak_id
            ),
            None,
        )

    def detector_profile_by_id(self, profile_id: str) -> DetectorProfile | None:
        return next(
            (
                item
                for item in self.detector_profiles
                if item.detector_profile_id == profile_id
            ),
            None,
        )

    def viewport_by_id(self, viewport_id: str) -> CanvasViewport | None:
        return next(
            (item for item in self.viewports if item.viewport_id == viewport_id), None
        )

    def validate(self) -> None:
        if self.schema != WORKSPACE_DOCUMENT_SCHEMA:
            raise WorkspaceValidationError("schema", "is unsupported")
        if self.schema_version != WORKSPACE_DOCUMENT_VERSION:
            raise WorkspaceValidationError("schema_version", "is unsupported")
        _required_text(self.document_id, "document_id")
        _text(self.title, "title")
        _required_text(self.created_at, "created_at")
        _required_text(self.updated_at, "updated_at")

        spectrum_ids = _validate_unique(
            [item.spectrum_id for item in self.spectra], "spectra", "spectrum_id"
        )
        profile_ids = _validate_unique(
            [item.detector_profile_id for item in self.detector_profiles],
            "detector_profiles",
            "detector_profile_id",
        )
        roi_ids = _validate_unique(
            [item.roi_id for item in self.rois], "rois", "roi_id"
        )
        diagnostic_ids = _validate_unique(
            [item.diagnostic_id for item in self.fit_diagnostics],
            "fit_diagnostics",
            "diagnostic_id",
        )
        viewport_ids = _validate_unique(
            [item.viewport_id for item in self.viewports], "viewports", "viewport_id"
        )
        del diagnostic_ids, viewport_ids
        peak_keys: set[tuple[str, str]] = set()

        for index, item in enumerate(self.spectra):
            item.validate(f"spectra[{index}]")
            if item.detector_profile_id and item.detector_profile_id not in profile_ids:
                raise WorkspaceValidationError(
                    f"spectra[{index}].detector_profile_id",
                    "references an unknown profile",
                )
        if (
            self.active_spectrum_id is not None
            and self.active_spectrum_id not in spectrum_ids
        ):
            raise WorkspaceValidationError(
                "active_spectrum_id", "references an unknown spectrum"
            )

        roles: set[str] = set()
        for index, role in enumerate(self.spectrum_roles):
            role.validate(f"spectrum_roles[{index}]")
            if role.role in roles:
                raise WorkspaceValidationError(
                    f"spectrum_roles[{index}].role", "must be unique"
                )
            roles.add(role.role)
            for spectrum_id in role.spectrum_ids:
                if spectrum_id not in spectrum_ids:
                    raise WorkspaceValidationError(
                        f"spectrum_roles[{index}].spectrum_ids",
                        "references an unknown spectrum",
                    )

        for index, profile in enumerate(self.detector_profiles):
            profile.validate(f"detector_profiles[{index}]")
        for index, peak in enumerate(self.peaks):
            peak.validate(f"peaks[{index}]")
            key = (peak.spectrum_id, peak.peak_id)
            if key in peak_keys:
                raise WorkspaceValidationError(
                    f"peaks[{index}].peak_id", "must be unique within its spectrum"
                )
            peak_keys.add(key)
            if peak.spectrum_id not in spectrum_ids:
                raise WorkspaceValidationError(
                    f"peaks[{index}].spectrum_id", "references an unknown spectrum"
                )
            if peak.roi_id is not None and peak.roi_id not in roi_ids:
                raise WorkspaceValidationError(
                    f"peaks[{index}].roi_id", "references an unknown ROI"
                )

        for index, roi in enumerate(self.rois):
            roi.validate(f"rois[{index}]")
            if roi.spectrum_id not in spectrum_ids:
                raise WorkspaceValidationError(
                    f"rois[{index}].spectrum_id", "references an unknown spectrum"
                )
            for peak_id in roi.associated_peak_ids:
                if (roi.spectrum_id, peak_id) not in peak_keys:
                    raise WorkspaceValidationError(
                        f"rois[{index}].associated_peak_ids",
                        "references an unknown peak for the ROI spectrum",
                    )

        for index, diagnostic in enumerate(self.fit_diagnostics):
            diagnostic.validate(f"fit_diagnostics[{index}]")
            if diagnostic.spectrum_id not in spectrum_ids:
                raise WorkspaceValidationError(
                    f"fit_diagnostics[{index}].spectrum_id",
                    "references an unknown spectrum",
                )
            if diagnostic.roi_id and diagnostic.roi_id not in roi_ids:
                raise WorkspaceValidationError(
                    f"fit_diagnostics[{index}].roi_id", "references an unknown ROI"
                )
            if (
                diagnostic.peak_id
                and (
                    diagnostic.spectrum_id,
                    diagnostic.peak_id,
                )
                not in peak_keys
            ):
                raise WorkspaceValidationError(
                    f"fit_diagnostics[{index}].peak_id", "references an unknown peak"
                )

        for index, viewport in enumerate(self.viewports):
            viewport.validate(f"viewports[{index}]")
            if viewport.spectrum_id and viewport.spectrum_id not in spectrum_ids:
                raise WorkspaceValidationError(
                    f"viewports[{index}].spectrum_id", "references an unknown spectrum"
                )
            if viewport.selected_roi_id and viewport.selected_roi_id not in roi_ids:
                raise WorkspaceValidationError(
                    f"viewports[{index}].selected_roi_id", "references an unknown ROI"
                )

        if len(set(self.pinned_nuclides)) != len(self.pinned_nuclides):
            raise WorkspaceValidationError(
                "pinned_nuclides", "must contain unique values"
            )
        for index, value in enumerate(self.pinned_nuclides):
            _required_text(value, f"pinned_nuclides[{index}]")
        for nuclide, tags in self.nuclide_tags.items():
            _required_text(nuclide, f"nuclide_tags.{nuclide}")
            if len(set(tags)) != len(tags):
                raise WorkspaceValidationError(
                    f"nuclide_tags.{nuclide}", "must contain unique tags"
                )
            for index, tag in enumerate(tags):
                _required_text(tag, f"nuclide_tags.{nuclide}[{index}]")
        for name in ("plot_settings", "workflow_state", "provenance", "extensions"):
            _validate_json_finite(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "spectra": [item.to_dict() for item in self.spectra],
            "spectrum_roles": [item.to_dict() for item in self.spectrum_roles],
            "active_spectrum_id": self.active_spectrum_id,
            "rois": [item.to_dict() for item in self.rois],
            "peaks": [item.to_dict() for item in self.peaks],
            "detector_profiles": [item.to_dict() for item in self.detector_profiles],
            "fit_diagnostics": [item.to_dict() for item in self.fit_diagnostics],
            "viewports": [item.to_dict() for item in self.viewports],
            "pinned_nuclides": list(self.pinned_nuclides),
            "nuclide_tags": {
                key: list(values) for key, values in self.nuclide_tags.items()
            },
            "plot_settings": _thaw_json(self.plot_settings),
            "workflow_state": _thaw_json(self.workflow_state),
            "provenance": _thaw_json(self.provenance),
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkspaceDocument":
        data = _mapping(payload, "document")
        allowed = {
            "schema",
            "schema_version",
            "document_id",
            "title",
            "created_at",
            "updated_at",
            "spectra",
            "spectrum_roles",
            "active_spectrum_id",
            "rois",
            "peaks",
            "detector_profiles",
            "fit_diagnostics",
            "viewports",
            "pinned_nuclides",
            "nuclide_tags",
            "plot_settings",
            "workflow_state",
            "provenance",
            "extensions",
        }
        _reject_unknown(data, allowed, "document")
        schema = data.get("schema")
        version = data.get("schema_version")
        if schema != WORKSPACE_DOCUMENT_SCHEMA:
            raise WorkspaceValidationError("schema", "is unsupported")
        if version != WORKSPACE_DOCUMENT_VERSION:
            raise WorkspaceValidationError("schema_version", "is unsupported")
        raw_tags = _mapping(data.get("nuclide_tags", {}), "document.nuclide_tags")
        result = cls(
            document_id=_required_text(data.get("document_id"), "document.document_id"),
            title=_text(data.get("title", ""), "document.title"),
            created_at=_required_text(data.get("created_at"), "document.created_at"),
            updated_at=_required_text(data.get("updated_at"), "document.updated_at"),
            spectra=tuple(
                WorkspaceSpectrum.from_dict(item)
                for item in _sequence(data.get("spectra", ()), "document.spectra")
            ),
            spectrum_roles=tuple(
                SpectrumRoleAssignment.from_dict(item)
                for item in _sequence(
                    data.get("spectrum_roles", ()), "document.spectrum_roles"
                )
            ),
            active_spectrum_id=_optional_text(
                data.get("active_spectrum_id"), "document.active_spectrum_id"
            ),
            rois=tuple(
                AnalysisROI.from_dict(item)
                for item in _sequence(data.get("rois", ()), "document.rois")
            ),
            peaks=tuple(
                PeakModel.from_dict(item)
                for item in _sequence(data.get("peaks", ()), "document.peaks")
            ),
            detector_profiles=tuple(
                DetectorProfile.from_dict(item)
                for item in _sequence(
                    data.get("detector_profiles", ()), "document.detector_profiles"
                )
            ),
            fit_diagnostics=tuple(
                FitDiagnostics.from_dict(item)
                for item in _sequence(
                    data.get("fit_diagnostics", ()), "document.fit_diagnostics"
                )
            ),
            viewports=tuple(
                CanvasViewport.from_dict(item)
                for item in _sequence(data.get("viewports", ()), "document.viewports")
            ),
            pinned_nuclides=_text_items(
                data.get("pinned_nuclides", ()), "document.pinned_nuclides"
            ),
            nuclide_tags={
                _required_text(key, "document.nuclide_tags key"): _text_items(
                    value, f"document.nuclide_tags.{key}"
                )
                for key, value in raw_tags.items()
            },
            plot_settings=_mapping(
                data.get("plot_settings", {}), "document.plot_settings"
            ),
            workflow_state=_mapping(
                data.get("workflow_state", {}), "document.workflow_state"
            ),
            provenance=_mapping(data.get("provenance", {}), "document.provenance"),
            extensions=_mapping(data.get("extensions", {}), "document.extensions"),
            schema=_required_text(schema, "schema"),
            schema_version=_integer(version, "schema_version", minimum=1),
        )
        result.validate()
        return result


def _validate_unique(values: list[str], path: str, field_name: str) -> set[str]:
    seen: set[str] = set()
    for index, value in enumerate(values):
        _required_text(value, f"{path}[{index}].{field_name}")
        if value in seen:
            raise WorkspaceValidationError(
                f"{path}[{index}].{field_name}", "must be unique"
            )
        seen.add(value)
    return seen


def _raise_validation(path: str, message: str) -> Any:
    """Expression helper used while constructing frozen dataclasses."""

    raise WorkspaceValidationError(path, message)


__all__ = [
    "WORKSPACE_DOCUMENT_SCHEMA",
    "WORKSPACE_DOCUMENT_VERSION",
    "AnalysisROI",
    "CalibrationModel",
    "CanvasViewport",
    "CorrectionSettings",
    "DetectorGeometry",
    "DetectorProfile",
    "EfficiencyModelState",
    "FitDiagnostics",
    "NuclideAssignment",
    "PeakComponent",
    "PeakModel",
    "SpectrumRoleAssignment",
    "WorkspaceDocument",
    "WorkspaceSpectrum",
    "WorkspaceValidationError",
]
