"""Source-bound historical QuantumGold configuration, never physical defaults.

Observed offsets are validated for the 32 rev4 study files in the 46096eb
investigation, not a universal ANS decoder. Saved state may precede report
processing. Missing GammaLib yields and corrections remain unknown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
import struct
from types import MappingProxyType
from typing import Any, Mapping


LAYOUT_ID = "qg-study-rev4-observed-1548-v1"
LAYOUT = {
    "revision": (0, "<h"),
    "live_time_s": (104, "<d"),
    "real_time_s": (96, "<d"),
    "analysis_ctrl": (860, "<H"),
    "use_library_efficiencies": (1292, "<h"),
    "roi_width_fwhm": (852, "<f"),
    "background_width_channels": (846, "<h"),
    "background_gap_fwhm": (848, "<f"),
    "first_channel": (1020, "<h"),
    "last_channel": (1022, "<h"),
    "roi_count": (1026, "<h"),
}
SAVED_QUALIFICATION = "SAVED_HEADER_ONLY; final report processing state not established"


def verified_sha256(data: bytes, expected: str) -> str:
    """Verify bytes against an explicitly supplied source identity."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected a lowercase SHA-256 source identity")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError("Source SHA-256 mismatch")
    return actual


def _validate_saved_values(values: Mapping[str, Any], byte_length: int) -> None:
    if set(values) != set(LAYOUT):
        raise ValueError("Missing or unrecognized saved-header fields")
    for key, (_, fmt) in LAYOUT.items():
        allowed = {float, int} if fmt in {"<d", "<f"} else {int}
        if type(values[key]) not in allowed:
            raise ValueError(f"Invalid saved-header type: {key}")
    if values["revision"] != 4:
        raise ValueError("Unsupported revision")
    if (values["first_channel"], values["last_channel"]) != (0, 8191):
        raise ValueError("Unsupported channel bounds")
    if (
        type(byte_length) is not int
        or values["roi_count"] < 0
        or values["roi_count"] > 32767
        or byte_length != 1548 + 8192 * 4 + values["roi_count"] * 50
    ):
        raise ValueError("Header/count/ROI record length does not close")
    if not all(math.isfinite(values[k]) for k in ("live_time_s", "real_time_s")):
        raise ValueError("Nonfinite timing anchor")
    if not 0 < values["live_time_s"] <= values["real_time_s"]:
        raise ValueError("Invalid timing anchor")
    if values["analysis_ctrl"] < 0 or values["analysis_ctrl"] & ~3:
        raise ValueError("Unexpected analysis-control bits")
    if values["use_library_efficiencies"] not in (0, -1, 1):
        raise ValueError("Invalid library-efficiency boolean")
    if (
        not math.isfinite(values["roi_width_fwhm"])
        or values["roi_width_fwhm"] <= 0
        or not math.isfinite(values["background_gap_fwhm"])
        or values["background_gap_fwhm"] < 0
        or values["background_width_channels"] < 1
        or values["background_width_channels"] > 32767
    ):
        raise ValueError("Invalid ROI/continuum parameters")


@dataclass(frozen=True)
class SavedState:
    source_sha256: str
    byte_length: int
    values: Mapping[str, Any]
    library_name: str
    report_sha256: str | None
    report_confirmed: Mapping[str, Any]

    def __post_init__(self):
        for digest in (self.source_sha256, self.report_sha256):
            if digest is not None and (
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("Invalid saved-header source identity")
        if self.source_sha256 is None:
            raise ValueError("Missing saved-header source identity")
        _validate_saved_values(self.values, self.byte_length)
        if self.library_name != "GammaLib.mdb":
            raise ValueError("Unsupported study library anchor")
        if not set(self.report_confirmed) <= {
            "gamma_library_name",
            "use_library_efficiencies",
            "activity_reference",
        }:
            raise ValueError("Unrecognized report-confirmed fields")
        if self.report_confirmed and self.report_sha256 is None:
            raise ValueError("Report-confirmed fields require a report identity")
        if "gamma_library_name" in self.report_confirmed and (
            not isinstance(self.report_confirmed["gamma_library_name"], str)
            or self.report_confirmed["gamma_library_name"].lower()
            != self.library_name.lower()
        ):
            raise ValueError("Report library anchor contradiction")
        if "use_library_efficiencies" in self.report_confirmed:
            if (
                self.report_confirmed["use_library_efficiencies"] is not False
                or self.values["use_library_efficiencies"] != 0
            ):
                raise ValueError("Library-efficiency flag contradicts report")
        if (
            self.report_confirmed.get("activity_reference", "measurement_date")
            != "measurement_date"
        ):
            raise ValueError("Unsupported report activity-reference assertion")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(
            self, "report_confirmed", MappingProxyType(dict(self.report_confirmed))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "byte_length": self.byte_length,
            "values": dict(self.values),
            "library_name": self.library_name,
            "report_sha256": self.report_sha256,
            "report_confirmed": dict(self.report_confirmed),
            "layout_id": LAYOUT_ID,
            "layout_scope": "32 study files; additional identities require separate validation",
            "layout_offsets": {
                k: {"offset": o, "format": f} for k, (o, f) in LAYOUT.items()
            },
            "saved_state": {
                "ambient_enabled": not bool(self.values["analysis_ctrl"] & 1),
                "continuum_enabled": not bool(self.values["analysis_ctrl"] & 2),
            },
            "qualification": SAVED_QUALIFICATION,
        }


def parse_saved_state(
    data: bytes,
    *,
    expected_sha256: str,
    layout_id: str,
    report: bytes | None = None,
    expected_report_sha256: str | None = None,
) -> SavedState:
    """Decode an explicitly selected, hash-bound observed study layout.

    Revision, channels, record closure, timings, ROI parameters, control bits
    and library descriptor are independent structural anchors. A supplied
    report must also be hash-bound and match its library/timing anchors.
    """
    source_hash = verified_sha256(data, expected_sha256)
    if layout_id != LAYOUT_ID:
        raise ValueError("Unsupported ANS layout; manual offsets are not universal")
    if len(data) < 1548:
        raise ValueError("Truncated observed study header")
    values = {k: struct.unpack_from(f, data, o)[0] for k, (o, f) in LAYOUT.items()}
    _validate_saved_values(values, len(data))
    try:
        library = data[1294:1306].decode("ascii").strip(" \0")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid library descriptor anchor") from exc
    if library != "GammaLib.mdb":
        raise ValueError("Unsupported study library anchor")
    confirmed: dict[str, Any] = {}
    report_hash = None
    if report is None and expected_report_sha256 is not None:
        raise ValueError("Report hash supplied without report bytes")
    if report is not None:
        report_hash = verified_sha256(report, expected_report_sha256)
        text = report.decode("utf-8", errors="replace")
        library_match = re.search(r"^Library:\s*(\S+)\s*$", text, re.MULTILINE)
        times = re.search(r"LT:\s*([\d,]+\.\d+)\s+RT:\s*([\d,]+\.\d+)", text)
        if (
            not library_match
            or library_match[1].lower() != library.lower()
            or not times
        ):
            raise ValueError("Missing or mismatched report library/timing anchors")
        for key, group in (("live_time_s", 1), ("real_time_s", 2)):
            if abs(float(times[group].replace(",", "")) - values[key]) > 0.005:
                raise ValueError("Report timing anchor contradiction")
        confirmed["gamma_library_name"] = library_match[1]
        if re.search(r"^Library efficiencies were ignored\s*$", text, re.MULTILINE):
            if values["use_library_efficiencies"] != 0:
                raise ValueError("Library-efficiency flag contradicts report")
            confirmed["use_library_efficiencies"] = False
        if re.search(
            r"^Activities reported as of Measurement Date\.\s*$", text, re.MULTILINE
        ):
            confirmed["activity_reference"] = "measurement_date"
    return SavedState(source_hash, len(data), values, library, report_hash, confirmed)


@dataclass(frozen=True)
class ProtocolField:
    """One scalar choice with its evidentiary status and source or rationale."""

    value: str | bool | float | int | None
    status: str
    source: str

    def __post_init__(self):
        if self.status not in {
            "saved_state",
            "report_confirmed",
            "assumed",
            "unknown",
            "unsupported",
        }:
            raise ValueError("Invalid evidence status")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("A field needs a source or assumption rationale")
        if self.value is not None and type(self.value) not in {str, bool, float, int}:
            raise ValueError("Protocol fields must be JSON scalars")
        if type(self.value) in {float, int} and not math.isfinite(self.value):
            raise ValueError("Protocol fields must be finite")
        if (self.status == "unknown") != (self.value is None):
            raise ValueError(
                "Unknown fields must be null; established choices must have values"
            )


FIELD_TYPES = {
    "ambient_enabled": bool,
    "continuum_enabled": bool,
    "roi_width_fwhm": float,
    "background_width_channels": int,
    "background_gap_fwhm": float,
    "continuum_method": str,
    "gamma_library_name": str,
    "gamma_library_sha256": str,
    "gamma_library_revision": str,
    "gamma_intensities": str,
    "gamma_corrections": str,
    "efficiency_identity": str,
    "efficiency_sha256": str,
    "efficiency_units": str,
    "vendor_software_version": str,
    "use_library_efficiencies": bool,
    "aggregation": str,
    "activity_reference": str,
    "activity_reference_timestamp": str,
    "activity_reference_timezone": str,
    "ambient_identity": str,
    "ambient_sha256": str,
    "ambient_normalization": str,
}


@dataclass(frozen=True)
class HistoricalProtocol:
    """Explicit comparison scenario, independently selectable from physical analysis.

    Configuration persists unsupported choices but never silently maps them to
    another method. Calculation/aggregation implementations remain engine-owned.
    """

    scenario_name: str
    fields: Mapping[str, ProtocolField]
    saved_header: SavedState

    def __post_init__(self):
        if not isinstance(self.scenario_name, str) or not self.scenario_name.strip():
            raise ValueError("A comparison scenario needs a name")
        if set(self.fields) != set(FIELD_TYPES):
            raise ValueError("Protocol fields are missing or unrecognized")
        for name, expected_type in FIELD_TYPES.items():
            field = self.fields[name]
            if not isinstance(field, ProtocolField):
                raise ValueError("Expected ProtocolField entries")
            if field.value is None:
                continue
            allowed = {int, float} if expected_type is float else {expected_type}
            if type(field.value) not in allowed:
                raise ValueError(f"Invalid field type: {name}")
            if expected_type is str and not field.value.strip():
                raise ValueError(f"Blank established string: {name}")
            if name.endswith("sha256") and not re.fullmatch(
                r"[0-9a-f]{64}", field.value
            ):
                raise ValueError(f"Invalid SHA-256 identity: {name}")
            if (
                name in {"roi_width_fwhm", "background_width_channels"}
                and field.value <= 0
            ):
                raise ValueError(f"Invalid positive width: {name}")
            if name == "background_gap_fwhm" and field.value < 0:
                raise ValueError("Negative continuum gap")
            if field.status == "saved_state":
                saved = _saved_choices(self.saved_header)
                if name not in saved or field.value != saved[name]:
                    raise ValueError(f"Choice contradicts saved-state evidence: {name}")
                if field.source != _saved_source(self.saved_header):
                    raise ValueError("Saved-state source identity differs from header")
            if field.status == "report_confirmed":
                if (
                    name not in self.saved_header.report_confirmed
                    or field.value != self.saved_header.report_confirmed[name]
                ):
                    raise ValueError(f"Choice lacks report confirmation: {name}")
                if field.source != f"Report SHA256:{self.saved_header.report_sha256}":
                    raise ValueError("Report source identity differs from header")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def verify_sources(self, data: bytes, *, report: bytes | None = None) -> SavedState:
        """Rebind a restored configuration to bytes before any reproduction.

        A JSON round trip checks configuration consistency, not the truth of
        serialized observations. This check rejects a tampered header snapshot
        even if its claimed SHA-256 and derived flags were left self-consistent.
        """
        observed = parse_saved_state(
            data,
            expected_sha256=self.saved_header.source_sha256,
            layout_id=LAYOUT_ID,
            report=report,
            expected_report_sha256=self.saved_header.report_sha256,
        )
        if observed.to_dict() != self.saved_header.to_dict():
            raise ValueError("Serialized saved header differs from source bytes")
        return observed

    @classmethod
    def from_saved_state(
        cls, state: SavedState, *, scenario_name: str
    ) -> HistoricalProtocol:
        """Keep the header and report evidence separate from scenario choices."""
        fields = {
            name: ProtocolField(None, "unknown", "Not established by supplied sources")
            for name in FIELD_TYPES
        }
        saved = _saved_choices(state)
        for name, value in saved.items():
            fields[name] = ProtocolField(
                value,
                "saved_state",
                _saved_source(state),
            )
        for name, value in state.report_confirmed.items():
            fields[name] = ProtocolField(
                value, "report_confirmed", f"Report SHA256:{state.report_sha256}"
            )
        return cls(scenario_name, fields, state)

    def with_assumptions(
        self, *, scenario_name: str, rationale: str, **choices
    ) -> HistoricalProtocol:
        """Declare a new scenario without rewriting the observed saved header."""
        if not set(choices) <= set(FIELD_TYPES):
            raise ValueError("Unrecognized scenario choice")
        fields = dict(self.fields)
        fields.update(
            {k: ProtocolField(v, "assumed", rationale) for k, v in choices.items()}
        )
        return replace(self, scenario_name=scenario_name, fields=fields)

    def roi_parameters(self) -> dict[str, float | int]:
        """Arguments for the current engine's local sideband ROI estimator.

        Require an explicit approximation: header continuum-on alone does not
        establish the exact algorithm used for the exported vendor report.
        """
        method = self.fields["continuum_method"]
        if (
            method.status == "unsupported"
            or method.value != "current_engine_local_sidebands"
        ):
            raise ValueError("Unknown or unsupported continuum method")
        names = ("roi_width_fwhm", "background_width_channels", "background_gap_fwhm")
        if any(
            self.fields[n].value is None or self.fields[n].status == "unsupported"
            for n in names
        ):
            raise ValueError("Unknown or unsupported ROI parameters")
        return {n: self.fields[n].value for n in names}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "purpose": "historical_comparison",
            "scenario_name": self.scenario_name,
            "fields": {k: asdict(v) for k, v in self.fields.items()},
            "evidence_by_status": {
                status: [k for k, v in self.fields.items() if v.status == status]
                for status in (
                    "saved_state",
                    "report_confirmed",
                    "assumed",
                    "unknown",
                    "unsupported",
                )
            },
            "saved_header": self.saved_header.to_dict(),
            "exact_vendor_parity": False,
            "physical_defaults_changed": False,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalProtocol:
        """Load configuration; serialized claims are not a new source-byte audit."""
        if (
            type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("purpose") != "historical_comparison"
        ):
            raise ValueError("Unsupported historical protocol schema")
        if (
            payload.get("exact_vendor_parity") is not False
            or payload.get("physical_defaults_changed") is not False
        ):
            raise ValueError(
                "Historical protocol cannot claim parity or change physical defaults"
            )
        header = payload["saved_header"]
        if header.get("layout_id") != LAYOUT_ID:
            raise ValueError("Unsupported saved-header layout")
        state = SavedState(
            **{
                k: header[k]
                for k in (
                    "source_sha256",
                    "byte_length",
                    "values",
                    "library_name",
                    "report_sha256",
                    "report_confirmed",
                )
            }
        )
        result = cls(
            payload["scenario_name"],
            {k: ProtocolField(**v) for k, v in payload["fields"].items()},
            state,
        )
        if result.to_dict() != payload:
            raise ValueError("Noncanonical or inconsistent protocol evidence")
        return result


def _saved_choices(state: SavedState) -> dict[str, Any]:
    return {
        "ambient_enabled": not bool(state.values["analysis_ctrl"] & 1),
        "continuum_enabled": not bool(state.values["analysis_ctrl"] & 2),
        "use_library_efficiencies": bool(state.values["use_library_efficiencies"]),
        "gamma_library_name": state.library_name,
        **{
            name: state.values[name]
            for name in (
                "roi_width_fwhm",
                "background_width_channels",
                "background_gap_fwhm",
            )
        },
    }


def _saved_source(state: SavedState) -> str:
    return f"ANS SHA256:{state.source_sha256}; {SAVED_QUALIFICATION}"
