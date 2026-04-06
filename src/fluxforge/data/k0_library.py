"""Governed starter libraries for k0-NAA workflows.

This module provides a provenance-bearing starter library for standard k0
constants and a separate auxiliary library for threshold / fast-flux
interference workflows. The built-in library is intentionally partial: it
covers the nuclides currently used by FluxForge examples and tests, and makes
its scope explicit in metadata instead of implying full TECDOC-grade coverage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


@dataclass(frozen=True)
class K0LibraryRecord:
    """One k0 nuclear-data record.

    Attributes mirror the existing `K0NuclideData` payload but add provenance
    fields needed for governed library selection and reporting.
    """

    product_isotope: str
    target_isotope: str
    element: str
    gamma_energy_keV: float
    gamma_intensity: float
    half_life_s: float
    k0_Au: float
    k0_unc_percent: float = 0.0
    Q0: float = 1.0
    Q0_unc_percent: float = 0.0
    E_res_eV: float = 0.0
    sigma_0_barn: float = 0.0
    I0_barn: float = 0.0
    isotopic_abundance: float = 1.0
    atomic_mass_g_mol: float = 0.0
    reaction_family: str = "thermal_capture"
    data_status: str = "starter_partial"
    source_note: str = (
        "FluxForge bundled starter library derived from existing add_gui seed values"
    )
    additional_gammas: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["additional_gammas"] = [list(item) for item in self.additional_gammas]
        return payload


@dataclass(frozen=True)
class AuxiliaryCorrectionRecord:
    """Auxiliary threshold / fast-flux correction record.

    These records intentionally identify workflow-relevant reactions and
    provenance without inventing numerical correction data that is not already
    validated in FluxForge.
    """

    reaction_id: str
    category: str
    monitor_role: str
    source_library: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedLibrary:
    """Versioned library bundle used by k0 workflows."""

    library_id: str
    version: str
    scope: str
    status: str
    provenance_note: str
    records: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        if not self.records:
            serialized_records: Dict[str, Any] = {}
        else:
            serialized_records = {
                key: value.to_dict() if hasattr(value, "to_dict") else dict(value)
                for key, value in self.records.items()
            }
        return {
            "library_id": self.library_id,
            "version": self.version,
            "scope": self.scope,
            "status": self.status,
            "provenance_note": self.provenance_note,
            "records": serialized_records,
        }


_K0_STARTER_RECORDS: tuple[K0LibraryRecord, ...] = (
    K0LibraryRecord(
        "Au-198",
        "Au-197",
        "Au",
        411.8,
        0.9558,
        2.6944 * 24 * 3600,
        1.0,
        0.0,
        15.71,
        1.5,
        5.65,
        98.65,
        1550.0,
        1.0,
        196.967,
    ),
    K0LibraryRecord(
        "Co-60",
        "Co-59",
        "Co",
        1332.5,
        0.9998,
        5.2714 * 365.25 * 24 * 3600,
        1.320,
        0.9,
        1.99,
        2.0,
        132.0,
        37.18,
        74.0,
        1.0,
        58.933,
        additional_gammas=((1173.2, 0.9985),),
    ),
    K0LibraryRecord(
        "Sc-46",
        "Sc-45",
        "Sc",
        889.3,
        0.99984,
        83.79 * 24 * 3600,
        8.79e-3,
        1.3,
        0.435,
        4.0,
        4.5,
        27.5,
        12.0,
        1.0,
        44.956,
        additional_gammas=((1120.5, 0.99987),),
    ),
    K0LibraryRecord(
        "Fe-59",
        "Fe-58",
        "Fe",
        1099.2,
        0.565,
        44.50 * 24 * 3600,
        5.34e-5,
        2.0,
        0.91,
        8.0,
        230.0,
        1.31,
        1.19,
        0.00282,
        55.845,
        additional_gammas=((1291.6, 0.432),),
    ),
    K0LibraryRecord(
        "Cu-64",
        "Cu-63",
        "Cu",
        1345.8,
        0.00473,
        12.701 * 3600,
        3.88e-3,
        2.5,
        1.11,
        4.0,
        580.0,
        4.5,
        5.0,
        0.6917,
        63.546,
    ),
    K0LibraryRecord(
        "In-114m",
        "In-113",
        "In",
        190.3,
        0.1556,
        49.51 * 24 * 3600,
        3.16e-2,
        2.0,
        31.5,
        3.0,
        1.45,
        4.0,
        126.0,
        0.0429,
        114.818,
    ),
    K0LibraryRecord(
        "Mn-56",
        "Mn-55",
        "Mn",
        846.8,
        0.9887,
        2.5789 * 3600,
        4.88e-3,
        1.2,
        0.67,
        3.5,
        337.0,
        13.3,
        8.9,
        1.0,
        54.938,
        additional_gammas=((1810.7, 0.272), (2113.1, 0.143)),
    ),
    K0LibraryRecord(
        "Na-24",
        "Na-23",
        "Na",
        1368.6,
        0.9999,
        14.997 * 3600,
        4.68e-2,
        1.3,
        0.59,
        4.0,
        2850.0,
        0.530,
        0.31,
        1.0,
        22.990,
        additional_gammas=((2754.0, 0.9994),),
    ),
    K0LibraryRecord(
        "Cr-51",
        "Cr-50",
        "Cr",
        320.1,
        0.0991,
        27.701 * 24 * 3600,
        8.41e-4,
        2.0,
        2.22,
        6.0,
        1820.0,
        15.8,
        35.0,
        0.04345,
        51.996,
    ),
    K0LibraryRecord(
        "Zn-65",
        "Zn-64",
        "Zn",
        1115.5,
        0.506,
        243.66 * 24 * 3600,
        3.05e-4,
        2.0,
        1.91,
        5.0,
        2560.0,
        0.76,
        1.45,
        0.4863,
        65.38,
    ),
    K0LibraryRecord(
        "As-76",
        "As-75",
        "As",
        559.1,
        0.451,
        26.32 * 3600,
        1.42e-2,
        2.5,
        2.50,
        6.0,
        1360.0,
        4.30,
        10.8,
        1.0,
        74.922,
    ),
    K0LibraryRecord(
        "W-187",
        "W-186",
        "W",
        685.8,
        0.273,
        23.72 * 3600,
        1.05e-4,
        3.0,
        19.3,
        7.0,
        18.8,
        39.5,
        762.0,
        0.2843,
        183.84,
    ),
)


BUILTIN_K0_LIBRARY = GovernedLibrary(
    library_id="fluxforge.k0.starter",
    version="2026.03-starter-v1",
    scope="Partial bundled starter library for thermal k0-NAA workflows used by current FluxForge examples/tests.",
    status="partial",
    provenance_note=(
        "Uses the existing add_gui branch seed values and is not a full TECDOC-2026"
        " replacement library. Future work should add the external benchmark dataset"
        " and broader governed source imports."
    ),
    records={record.product_isotope: record for record in _K0_STARTER_RECORDS},
)


BUILTIN_AUXILIARY_CORRECTION_LIBRARY = GovernedLibrary(
    library_id="fluxforge.k0.auxiliary.threshold",
    version="2026.03-aux-v1",
    scope="Auxiliary threshold / fast-flux workflow identifiers kept separate from standard k0 constants.",
    status="partial",
    provenance_note=(
        "These records identify monitor and interference pathways only. Numerical"
        " correction factors remain workflow-specific and must not be inferred without"
        " validated data."
    ),
    records={
        record.reaction_id: record
        for record in (
            AuxiliaryCorrectionRecord(
                "54Fe(n,p)54Mn",
                "threshold",
                "fast_flux_monitor",
                "IRDFF-II",
                "Common fast-flux monitor pathway for threshold-interference bookkeeping.",
            ),
            AuxiliaryCorrectionRecord(
                "58Ni(n,p)58Co",
                "threshold",
                "fast_flux_monitor",
                "IRDFF-II",
                "Optional fast-flux monitor pathway kept separate from standard k0 constants.",
            ),
            AuxiliaryCorrectionRecord(
                "27Al(n,p)27Mg",
                "threshold",
                "fast_flux_monitor",
                "IRDFF-II",
                "Short-lived fast-flux support pathway; numerical corrections are workflow-specific.",
            ),
        )
    },
)


_ACTIVE_K0_LIBRARY = BUILTIN_K0_LIBRARY
_ACTIVE_AUXILIARY_CORRECTION_LIBRARY = BUILTIN_AUXILIARY_CORRECTION_LIBRARY


def _coerce_k0_record(payload: Dict[str, Any]) -> K0LibraryRecord:
    return K0LibraryRecord(
        product_isotope=str(payload.get("product_isotope") or ""),
        target_isotope=str(payload.get("target_isotope") or ""),
        element=str(payload.get("element") or ""),
        gamma_energy_keV=float(payload.get("gamma_energy_keV", 0.0) or 0.0),
        gamma_intensity=float(payload.get("gamma_intensity", 0.0) or 0.0),
        half_life_s=float(payload.get("half_life_s", 0.0) or 0.0),
        k0_Au=float(payload.get("k0_Au", 0.0) or 0.0),
        k0_unc_percent=float(payload.get("k0_unc_percent", 0.0) or 0.0),
        Q0=float(payload.get("Q0", 1.0) or 1.0),
        Q0_unc_percent=float(payload.get("Q0_unc_percent", 0.0) or 0.0),
        E_res_eV=float(payload.get("E_res_eV", 0.0) or 0.0),
        sigma_0_barn=float(payload.get("sigma_0_barn", 0.0) or 0.0),
        I0_barn=float(payload.get("I0_barn", 0.0) or 0.0),
        isotopic_abundance=float(payload.get("isotopic_abundance", 1.0) or 1.0),
        atomic_mass_g_mol=float(payload.get("atomic_mass_g_mol", 0.0) or 0.0),
        reaction_family=str(payload.get("reaction_family") or "thermal_capture"),
        data_status=str(payload.get("data_status") or "external"),
        source_note=str(payload.get("source_note") or ""),
        additional_gammas=tuple(
            (float(item[0]), float(item[1]))
            for item in (payload.get("additional_gammas") or [])
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ),
    )


def _coerce_auxiliary_record(payload: Dict[str, Any]) -> AuxiliaryCorrectionRecord:
    return AuxiliaryCorrectionRecord(
        reaction_id=str(payload.get("reaction_id") or ""),
        category=str(payload.get("category") or ""),
        monitor_role=str(payload.get("monitor_role") or ""),
        source_library=str(payload.get("source_library") or ""),
        notes=str(payload.get("notes") or ""),
    )


def governed_library_from_dict(
    payload: Dict[str, Any], *, library_kind: str = "standard"
) -> GovernedLibrary:
    """Build a `GovernedLibrary` from a JSON/YAML payload."""

    record_items = payload.get("records") or {}
    if isinstance(record_items, list):
        record_iterable = []
        for item in record_items:
            if isinstance(item, dict):
                key = str(
                    item.get("product_isotope")
                    or item.get("reaction_id")
                    or f"record_{len(record_iterable) + 1}"
                )
                record_iterable.append((key, item))
    else:
        record_iterable = list(dict(record_items).items())

    coerced_records: Dict[str, Any] = {}
    for key, value in record_iterable:
        row = dict(value or {})
        if library_kind == "standard":
            if not row.get("product_isotope"):
                row["product_isotope"] = str(key)
            record = _coerce_k0_record(row)
            coerced_records[record.product_isotope] = record
        else:
            if not row.get("reaction_id"):
                row["reaction_id"] = str(key)
            record = _coerce_auxiliary_record(row)
            coerced_records[record.reaction_id] = record

    return GovernedLibrary(
        library_id=str(
            payload.get("library_id") or f"fluxforge.k0.{library_kind}.external"
        ),
        version=str(payload.get("version") or "external"),
        scope=str(payload.get("scope") or "Externally supplied governed library."),
        status=str(payload.get("status") or "external"),
        provenance_note=str(
            payload.get("provenance_note") or "Loaded from external file."
        ),
        records=coerced_records,
    )


def load_governed_library(
    path: str | Path, *, library_kind: str = "standard"
) -> GovernedLibrary:
    """Load a governed library from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Governed library input must be a JSON object.")
    return governed_library_from_dict(payload, library_kind=library_kind)


def set_active_k0_library(library: GovernedLibrary | None = None) -> GovernedLibrary:
    """Set the active standard k0 library and return it."""

    global _ACTIVE_K0_LIBRARY
    _ACTIVE_K0_LIBRARY = library or BUILTIN_K0_LIBRARY
    return _ACTIVE_K0_LIBRARY


def set_active_auxiliary_correction_library(
    library: GovernedLibrary | None = None,
) -> GovernedLibrary:
    """Set the active auxiliary correction library and return it."""

    global _ACTIVE_AUXILIARY_CORRECTION_LIBRARY
    _ACTIVE_AUXILIARY_CORRECTION_LIBRARY = (
        library or BUILTIN_AUXILIARY_CORRECTION_LIBRARY
    )
    return _ACTIVE_AUXILIARY_CORRECTION_LIBRARY


@contextmanager
def use_governed_libraries(
    *,
    standard_library: GovernedLibrary | None = None,
    auxiliary_library: GovernedLibrary | None = None,
) -> Iterator[tuple[GovernedLibrary, GovernedLibrary]]:
    """Temporarily activate governed k0 libraries for one workflow block."""

    previous_standard = get_active_k0_library()
    previous_auxiliary = get_active_auxiliary_correction_library()
    set_active_k0_library(standard_library or previous_standard)
    set_active_auxiliary_correction_library(auxiliary_library or previous_auxiliary)
    try:
        yield get_active_k0_library(), get_active_auxiliary_correction_library()
    finally:
        set_active_k0_library(previous_standard)
        set_active_auxiliary_correction_library(previous_auxiliary)


def get_active_k0_library() -> GovernedLibrary:
    """Return the bundled starter k0 library."""

    return _ACTIVE_K0_LIBRARY


def get_active_auxiliary_correction_library() -> GovernedLibrary:
    """Return the bundled auxiliary threshold / fast-flux library."""

    return _ACTIVE_AUXILIARY_CORRECTION_LIBRARY


def get_k0_library_record(product_isotope: str) -> K0LibraryRecord | None:
    """Return one starter-library record by product isotope."""

    return get_active_k0_library().records.get(product_isotope)


def iter_k0_library_records() -> Iterable[K0LibraryRecord]:
    """Iterate over bundled starter-library records."""

    return get_active_k0_library().records.values()


def get_library_summary() -> Dict[str, Any]:
    """Return report-friendly summaries for the active k0 libraries."""

    return {
        "standard_k0_library": get_active_k0_library().to_dict(),
        "auxiliary_correction_library": get_active_auxiliary_correction_library().to_dict(),
    }
