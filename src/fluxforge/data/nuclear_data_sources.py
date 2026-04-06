"""Nuclear data source registry and adapters for FluxForge.

This module keeps distinct nuclear-data sources separate so GUI and workflow
code can expose provenance-aware selection instead of treating everything as a
single bundled database.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from fluxforge.core.runtime import (
    is_remote_locator,
    offline_mode_enabled,
    require_network_access,
)
from fluxforge.data.efficiency import CALIBRATION_SOURCES
from fluxforge.data.flux_wire_catalog import list_flux_wire_isotopes
from fluxforge.data.gamma_database import (
    DecayData,
    GammaDatabase,
    GammaLine,
    FLUXFORGE_GAMMA_DATA,
    find_actigamma_data,
)
from fluxforge.data.irdff_access import get_default_library
from fluxforge.data.kayzero_k0 import load_kayzero_half_life_table
from fluxforge.data.nndc import GAMMA_LINES as NNDC_GAMMA_LINES
from fluxforge.data.nndc import HALF_LIVES_S as NNDC_HALF_LIVES_S
from fluxforge.physics.decay_library import DecayDataset, normalize_nuclide_label
from fluxforge.triga.cd_ratio import STANDARD_MONITORS


@dataclass(frozen=True)
class NuclearDataSourceRecord:
    """Description of a selectable FluxForge nuclear-data source."""

    source_id: str
    label: str
    kind: str
    description: str
    builtin: bool = True
    path_hint: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


_DATA_DIR = Path(__file__).resolve().parent
_ICRP107_DECAY_DATA = _DATA_DIR / "icrp107_decay_data.npz"
_KAYZERO_K0_2020 = _DATA_DIR / "kayzero_k0_2020.uT12"
_KAYZERO_K0_2023 = _DATA_DIR / "kayzero_k0_2023.uT12"
_GSA_V4_LIBEDIT = _DATA_DIR / "gsa_v4_libedit.dat"
_GSA_V4_LIB_GAMMA_NATURAL = _DATA_DIR / "gsa_v4_lib_gamma_natural.dat"
_NASA_COMMON_LAB_SOURCES = _DATA_DIR / "nasa_common_lab_sources.csv"
_NASA_NATURAL_RADIATION = _DATA_DIR / "nasa_natural_radiation.csv"
_NASA_CAPTURE_CAPGAM = _DATA_DIR / "nasa_capture_capgam.csv"
_NASA_CAPTURE_IAEA = _DATA_DIR / "nasa_capture_iaea.csv"
_NASA_DELAYED_ACTIVATION_IAEA = _DATA_DIR / "nasa_delayed_activation_iaea.csv"
_NASA_INELASTIC_BAGHDAD = _DATA_DIR / "nasa_inelastic_baghdad.csv"
_NASA_INELASTIC_14MEV_TALYS = _DATA_DIR / "nasa_inelastic_14mev_talys.csv"
_NASA_TALYS_14MEV = _DATA_DIR / "nasa_talys_14mev.csv"
_ENDFB8_DECAY_SUPPLEMENT = _DATA_DIR / "endfb8_decay_supplement.json"
_USER_LIBRARY_REGISTRY_ENV = "FLUXFORGE_LIBRARY_REGISTRY"
_USER_LIBRARY_REGISTRY_VERSION = 1


def _empty_gamma_database() -> GammaDatabase:
    return GammaDatabase(datafile="")


def _build_gamma_database_from_rows(rows: Iterable[dict[str, Any]]) -> GammaDatabase:
    database = _empty_gamma_database()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        nuclide = str(row.get("nuclide") or row.get("isotope") or "").strip()
        if not nuclide:
            continue
        half_life_s = float(row.get("half_life_s", row.get("halflife_s", 0.0)) or 0.0)
        grouped.setdefault(nuclide, {"half_life_s": half_life_s, "lines": []})
        grouped[nuclide]["half_life_s"] = half_life_s or grouped[nuclide]["half_life_s"]
        grouped[nuclide]["lines"].append(
            GammaLine(
                energy=float(row["energy_keV"]) * 1000.0,
                energy_unc=float(row.get("energy_unc_keV", 0.0) or 0.0) * 1000.0,
                intensity=float(row.get("intensity", 0.0) or 0.0),
                intensity_unc=float(row.get("intensity_unc", 0.0) or 0.0),
                norm=float(row.get("norm", 1.0) or 1.0),
                norm_unc=float(row.get("norm_unc", 0.0) or 0.0),
            )
        )
    for nuclide, data in grouped.items():
        database._nuclides[nuclide] = DecayData(
            nuclide=nuclide,
            zai=0,
            halflife=float(data["half_life_s"] or 0.0),
            gamma_lines=sorted(data["lines"], key=lambda item: item.energy),
        )
    return database


def _build_nndc_gamma_database() -> GammaDatabase:
    rows: list[dict[str, Any]] = []
    for nuclide, lines in NNDC_GAMMA_LINES.items():
        compact_name = nuclide.replace("-", "")
        for energy_keV, intensity in lines:
            rows.append(
                {
                    "nuclide": compact_name,
                    "energy_keV": energy_keV,
                    "intensity": intensity,
                    "half_life_s": NNDC_HALF_LIVES_S.get(nuclide, 0.0),
                }
            )
    return _build_gamma_database_from_rows(rows)


def _compact_nuclide_name(label: str) -> str:
    normalized = normalize_nuclide_label(label)
    return normalized.replace("-", "")


def _parse_half_life_seconds(
    value: Any,
    unit: str | None = None,
) -> float:
    raw_text = str(value or "").strip()
    if not raw_text:
        return 0.0
    half_life = float(raw_text)
    unit_text = str(unit or "").strip().lower()
    if unit_text in {"", "s", "sec", "second", "seconds"}:
        return half_life
    if unit_text in {"m", "min", "minute", "minutes"}:
        return half_life * 60.0
    if unit_text in {"h", "hr", "hour", "hours"}:
        return half_life * 3600.0
    if unit_text in {"d", "day", "days"}:
        return half_life * 86400.0
    if unit_text in {"y", "yr", "year", "years"}:
        return half_life * 365.25 * 24.0 * 3600.0
    return half_life


def _load_gsa_gamma_source(path: Path) -> GammaDatabase:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for columns in reader:
            if len(columns) < 5:
                continue
            nuclide_text, energy_text, intensity_text, half_life_text, half_life_unit = (
                str(columns[0]).strip(),
                str(columns[1]).strip(),
                str(columns[2]).strip(),
                str(columns[3]).strip(),
                str(columns[4]).strip(),
            )
            if not nuclide_text or not energy_text or not intensity_text:
                continue
            rows.append(
                {
                    "nuclide": _compact_nuclide_name(nuclide_text),
                    "energy_keV": float(energy_text),
                    "intensity": float(intensity_text) / 100.0,
                    "half_life_s": _parse_half_life_seconds(
                        half_life_text, half_life_unit
                    ),
                }
            )
    return _build_gamma_database_from_rows(rows)


def _load_nasa_gamma_source(path: Path) -> GammaDatabase:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            nuclide_text = str(
                source_row.get("Isotope")
                or source_row.get("isotope")
                or source_row.get("Nuclide")
                or source_row.get("nuclide")
                or ""
            ).strip()
            energy_text = str(
                source_row.get("Energy (keV)")
                or source_row.get("energy_keV")
                or source_row.get("energy")
                or ""
            ).strip()
            intensity_text = str(
                source_row.get("Intensity (%)")
                or source_row.get("intensity")
                or source_row.get("Sigma (mb)")
                or source_row.get("sigma_mb")
                or source_row.get("XS (mb)")
                or source_row.get("xs_mb")
                or ""
            ).strip()
            if not nuclide_text or not energy_text or not intensity_text:
                continue
            intensity_value = float(intensity_text)
            intensity_uncertainty = float(
                source_row.get("Intensity uncertainty (%)")
                or source_row.get("intensity_unc")
                or source_row.get("DSigma (mb)")
                or source_row.get("dSigma_mb")
                or source_row.get("d_sigma_mb")
                or source_row.get("DXS (mb)")
                or source_row.get("d_xs_mb")
                or 0.0
                or 0.0
            )
            if source_row.get("Intensity (%)") is not None or source_row.get("intensity") is not None:
                intensity_value /= 100.0
                intensity_uncertainty /= 100.0
            rows.append(
                {
                    "nuclide": _compact_nuclide_name(nuclide_text),
                    "energy_keV": float(energy_text),
                    "intensity": intensity_value,
                    "intensity_unc": intensity_uncertainty,
                    "half_life_s": _parse_half_life_seconds(
                        source_row.get("Half life (s)")
                        or source_row.get("half_life_s")
                        or 0.0
                    ),
                }
            )
    return _build_gamma_database_from_rows(rows)


def _load_custom_gamma_source(path: Path) -> GammaDatabase:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsn"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and all(
            isinstance(value, dict) for value in payload.values()
        ):
            return GammaDatabase(str(path))
        if isinstance(payload, dict):
            rows = payload.get("lines") or payload.get("rows") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unsupported custom gamma JSON payload.")
        return _build_gamma_database_from_rows(rows)
    if suffix in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return _build_gamma_database_from_rows(rows)
    raise ValueError("Custom gamma sources must be JSON or CSV files.")


def _normalize_source_locator(locator: str | Path) -> str:
    text = str(locator).strip()
    if not text:
        raise ValueError("Empty external data source locator.")
    return text


def _user_library_registry_path() -> Path:
    override = str(os.getenv(_USER_LIBRARY_REGISTRY_ENV, "")).strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fluxforge" / "nuclear_data_sources.json"


def _load_user_library_registry_payload() -> dict[str, Any]:
    path = _user_library_registry_path()
    if not path.exists():
        return {
            "version": _USER_LIBRARY_REGISTRY_VERSION,
            "user_gamma_sources": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("User nuclear-data registry must be a JSON object.")
    if not isinstance(payload.get("user_gamma_sources", []), list):
        raise ValueError("User nuclear-data registry must contain user_gamma_sources[].")
    return payload


def _save_user_library_registry_payload(payload: dict[str, Any]) -> None:
    path = _user_library_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slugify_user_gamma_alias(alias: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(alias).strip().lower()).strip("_")
    if not slug:
        raise ValueError("User library alias must contain letters or numbers.")
    return f"user_gamma_{slug}"


def _builtin_nuclear_data_sources() -> tuple[NuclearDataSourceRecord, ...]:
    actigamma_path = find_actigamma_data()
    return (
        NuclearDataSourceRecord(
            source_id="actigamma_2012",
            label="actigamma decay-2012 (legacy id)",
            kind="gamma-library",
            description="Backward-compatible alias for decay_2012 used by older GUI/tests.",
            path_hint=actigamma_path,
            capabilities=("peak-identification", "calibration-sources"),
            metadata={"nuclide_count": "external", "alias_for": "decay_2012"},
        ),
        NuclearDataSourceRecord(
            source_id="decay_2012",
            label="actigamma decay-2012",
            kind="gamma-library",
            description="Primary high-coverage gamma-line source used by FluxForge when actigamma data are installed.",
            path_hint=actigamma_path,
            capabilities=("peak-identification", "calibration-sources"),
            metadata={"nuclide_count": "external"},
        ),
        NuclearDataSourceRecord(
            source_id="fluxforge_bundled_gamma",
            label="FluxForge bundled gamma lines",
            kind="gamma-library",
            description="Bundled offline gamma library shipped with FluxForge.",
            path_hint=FLUXFORGE_GAMMA_DATA,
            capabilities=("peak-identification",),
        ),
        NuclearDataSourceRecord(
            source_id="gsa_v4_edit_library",
            label="GSA-v4 edited gamma library",
            kind="gamma-library",
            description="Bundled GSA-v4 line library imported from the newer libEdit.dat table.",
            path_hint=str(_GSA_V4_LIBEDIT),
            capabilities=("peak-identification", "activation-reference"),
            metadata={"source_family": "GSA-v4"},
        ),
        NuclearDataSourceRecord(
            source_id="gsa_v4_natural_library",
            label="GSA-v4 natural gamma library",
            kind="gamma-library",
            description="Bundled GSA-v4 natural-background gamma reference table from Lib-gamma-natur.dat.",
            path_hint=str(_GSA_V4_LIB_GAMMA_NATURAL),
            capabilities=("peak-identification", "activation-reference", "natural-background"),
            metadata={"source_family": "GSA-v4", "line_family": "natural"},
        ),
        NuclearDataSourceRecord(
            source_id="nasa_common_lab_sources",
            label="NASA common lab sources",
            kind="gamma-library",
            description="Bundled NASA-gamma common laboratory source library with energies, intensities, and half-lives.",
            path_hint=str(_NASA_COMMON_LAB_SOURCES),
            capabilities=("peak-identification", "activation-reference"),
            metadata={"source_family": "NASA-gamma"},
        ),
        NuclearDataSourceRecord(
            source_id="nasa_natural_radiation",
            label="NASA natural radiation library",
            kind="gamma-library",
            description="Bundled NASA-gamma natural-radiation reference lines for common environmental contributors.",
            path_hint=str(_NASA_NATURAL_RADIATION),
            capabilities=("peak-identification", "activation-reference", "natural-background"),
            metadata={"source_family": "NASA-gamma"},
        ),
        NuclearDataSourceRecord(
            source_id="nasa_capture_capgam",
            label="NASA capture CapGam library",
            kind="gamma-library",
            description="Bundled NASA-gamma capture-gamma reference lines from CapGam.",
            path_hint=str(_NASA_CAPTURE_CAPGAM),
            capabilities=("activation-reference", "capture-gamma"),
            metadata={"source_family": "NASA-gamma", "line_family": "capture"},
        ),
        NuclearDataSourceRecord(
            source_id="nasa_capture_iaea",
            label="NASA capture IAEA library",
            kind="gamma-library",
            description="Bundled NASA-gamma IAEA thermal capture-gamma reference lines with sigma and sigma-uncertainty columns.",
            path_hint=str(_NASA_CAPTURE_IAEA),
            capabilities=("activation-reference", "capture-gamma"),
            metadata={
                "source_family": "NASA-gamma",
                "line_family": "capture",
                "reference_metric": "sigma_mb",
            },
        ),
        NuclearDataSourceRecord(
            source_id="nasa_delayed_activation_iaea",
            label="NASA delayed-activation IAEA library",
            kind="gamma-library",
            description="Bundled NASA-gamma delayed-activation reference lines with cross sections and half-lives.",
            path_hint=str(_NASA_DELAYED_ACTIVATION_IAEA),
            capabilities=("activation-reference", "delayed-activation"),
            metadata={
                "source_family": "NASA-gamma",
                "line_family": "delayed-activation",
                "reference_metric": "sigma_mb",
            },
        ),
        NuclearDataSourceRecord(
            source_id="nasa_inelastic_baghdad",
            label="NASA inelastic Baghdad atlas",
            kind="gamma-library",
            description="Bundled NASA-gamma inelastic-scattering gamma atlas imported from the Baghdad reference table.",
            path_hint=str(_NASA_INELASTIC_BAGHDAD),
            capabilities=("activation-reference", "reaction-gamma"),
            metadata={
                "source_family": "NASA-gamma",
                "line_family": "inelastic",
                "reference_metric": "xs_mb",
            },
        ),
        NuclearDataSourceRecord(
            source_id="nasa_inelastic_14mev_talys",
            label="NASA inelastic 14 MeV TALYS",
            kind="gamma-library",
            description="Bundled NASA-gamma 14 MeV inelastic reference lines generated from TALYS.",
            path_hint=str(_NASA_INELASTIC_14MEV_TALYS),
            capabilities=("activation-reference", "reaction-gamma"),
            metadata={
                "source_family": "NASA-gamma",
                "line_family": "inelastic",
                "reference_metric": "xs_mb",
            },
        ),
        NuclearDataSourceRecord(
            source_id="nasa_talys_14mev",
            label="NASA TALYS 14 MeV reaction library",
            kind="gamma-library",
            description="Bundled NASA-gamma mixed 14 MeV reaction-gamma reference lines from TALYS.",
            path_hint=str(_NASA_TALYS_14MEV),
            capabilities=("activation-reference", "reaction-gamma"),
            metadata={
                "source_family": "NASA-gamma",
                "line_family": "reaction",
                "reference_metric": "xs_mb",
            },
        ),
        NuclearDataSourceRecord(
            source_id="endfb8_decay_supplement",
            label="ENDF/B-VIII decay supplement",
            kind="gamma-library",
            description="Small bundled ENDF/B-VIII-derived decay-line supplement used for residual legacy-library gaps.",
            path_hint=str(_ENDFB8_DECAY_SUPPLEMENT),
            capabilities=("peak-identification", "activation-reference"),
            metadata={"source_family": "ENDF/B-VIII", "coverage": "supplemental"},
        ),
        NuclearDataSourceRecord(
            source_id="nndc_offline_activation",
            label="NNDC offline activation subset",
            kind="gamma-library",
            description="Offline NNDC-inspired activation subset for common NAA and dosimetry products.",
            capabilities=("peak-identification", "activation-reference"),
            metadata={"nuclides": len(NNDC_GAMMA_LINES)},
        ),
        NuclearDataSourceRecord(
            source_id="radioactivedecay_icrp107",
            label="ICRP-107 decay network",
            kind="decay-library",
            description="Bundled radioactivedecay ICRP-107/AME2020/NUBASE2020 decay-network dataset for full daughter chains and branching fractions.",
            path_hint=str(_ICRP107_DECAY_DATA),
            capabilities=("decay-chains", "inventory-decay"),
            metadata={"source_family": "radioactivedecay"},
        ),
        NuclearDataSourceRecord(
            source_id="radioactivedecay_icrp107_kayzero_2023",
            label="ICRP-107 + Kayzero 2023 half-life overlay",
            kind="decay-library",
            description="Bundled ICRP-107 decay network with Kayzero 2023 half-life values and uncertainty overlay.",
            path_hint=str(_KAYZERO_K0_2023),
            capabilities=("decay-chains", "inventory-decay", "half-life-uncertainty"),
            metadata={
                "source_family": "radioactivedecay+Kayzero",
                "overlay": "k0-2023.uT12",
            },
        ),
        NuclearDataSourceRecord(
            source_id="radioactivedecay_icrp107_kayzero_2020",
            label="ICRP-107 + Kayzero 2020 half-life overlay",
            kind="decay-library",
            description="Bundled ICRP-107 decay network with Kayzero 2020 half-life values and uncertainty overlay.",
            path_hint=str(_KAYZERO_K0_2020),
            capabilities=("decay-chains", "inventory-decay", "half-life-uncertainty"),
            metadata={
                "source_family": "radioactivedecay+Kayzero",
                "overlay": "k0-2020.uT12",
            },
        ),
        NuclearDataSourceRecord(
            source_id="calibration_standard_sources",
            label="Standard calibration source lines",
            kind="calibration-library",
            description="Reference gamma energies and intensities for common HPGe calibration sources.",
            capabilities=("efficiency-calibration",),
            metadata={"source_count": len(CALIBRATION_SOURCES)},
        ),
        NuclearDataSourceRecord(
            source_id="k0_naa_monitors",
            label="k0-NAA monitor constants",
            kind="naa-reference",
            description="Comparator and Cd-ratio monitor constants used in FluxForge k0 workflows.",
            capabilities=("k0-naa", "monitor-selection"),
            metadata={"monitor_count": len(STANDARD_MONITORS)},
        ),
        NuclearDataSourceRecord(
            source_id="irdff_ii_dosimetry",
            label="IRDFF-II dosimetry catalog",
            kind="dosimetry-library",
            description="IAEA IRDFF-II dosimetry reaction catalog exposed as its own selectable source.",
            capabilities=("dosimetry", "monitor-selection"),
            metadata={
                "reaction_count": len(get_default_library().to_dict()["reactions"])
            },
        ),
        NuclearDataSourceRecord(
            source_id="flux_wire_catalog",
            label="Flux-wire monitor catalog",
            kind="activation-catalog",
            description="Bundled flux-wire isotope/reference catalog for activation workflows.",
            capabilities=("monitor-selection", "activation-reference"),
            metadata={"isotope_count": len(list_flux_wire_isotopes())},
        ),
        NuclearDataSourceRecord(
            source_id="custom_gamma_file",
            label="User custom gamma file",
            kind="custom-gamma-library",
            description="User-supplied JSON or CSV gamma-line library loaded at runtime.",
            builtin=False,
            capabilities=("peak-identification", "user-supplied"),
        ),
    )


def list_registered_user_gamma_sources() -> tuple[NuclearDataSourceRecord, ...]:
    payload = _load_user_library_registry_payload()
    records: list[NuclearDataSourceRecord] = []
    for item in payload.get("user_gamma_sources", []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "") or "").strip()
        label = str(item.get("label", "") or "").strip()
        locator = str(item.get("locator", "") or "").strip()
        if not source_id or not label or not locator:
            continue
        capabilities = tuple(
            str(value).strip()
            for value in item.get("capabilities", ("peak-identification", "user-supplied"))
            if str(value).strip()
        ) or ("peak-identification", "user-supplied")
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("registry", "user")
        records.append(
            NuclearDataSourceRecord(
                source_id=source_id,
                label=label,
                kind=str(item.get("kind") or "custom-gamma-library"),
                description=str(
                    item.get("description")
                    or "User-registered gamma-line file."
                ),
                builtin=False,
                path_hint=locator,
                capabilities=capabilities,
                metadata=metadata,
            )
        )
    return tuple(records)


def register_user_gamma_source(
    alias: str,
    locator: str | Path,
    *,
    description: str | None = None,
) -> NuclearDataSourceRecord:
    resolved_alias = str(alias).strip()
    if not resolved_alias:
        raise ValueError("A non-empty alias is required for user libraries.")
    resolved_locator = _normalize_source_locator(locator)
    parsed = urlparse(resolved_locator)
    if parsed.scheme in {"", "file"}:
        local_path = Path(parsed.path if parsed.scheme == "file" else resolved_locator)
        if not local_path.exists():
            raise FileNotFoundError(f"User library path does not exist: {local_path}")
    if parsed.scheme == "sqlite":
        db_path = Path(parsed.path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite gamma source does not exist: {db_path}")

    source_id = _slugify_user_gamma_alias(resolved_alias)
    builtin_records = _builtin_nuclear_data_sources()
    reserved_ids = {record.source_id for record in builtin_records}
    reserved_aliases = {
        record.source_id.lower() for record in builtin_records
    } | {
        record.label.lower() for record in builtin_records
    }
    if source_id in reserved_ids:
        raise ValueError(
            f"Alias '{resolved_alias}' maps to reserved built-in source id '{source_id}'. "
            "Choose a distinct alias."
        )
    if resolved_alias.lower() in reserved_aliases:
        raise ValueError(
            f"Alias '{resolved_alias}' conflicts with a reserved built-in library name. "
            "Choose a distinct alias."
        )

    payload = _load_user_library_registry_payload()
    existing = [
        item
        for item in payload.get("user_gamma_sources", [])
        if isinstance(item, dict)
    ]
    for item in existing:
        existing_source_id = str(item.get("source_id", "") or "").strip()
        existing_label = str(item.get("label", "") or "").strip().lower()
        if existing_source_id == source_id or existing_label == resolved_alias.lower():
            raise ValueError(
                f"User library alias '{resolved_alias}' is already registered as '{existing_source_id}'."
            )

    entry = {
        "source_id": source_id,
        "label": resolved_alias,
        "locator": resolved_locator,
        "kind": "custom-gamma-library",
        "description": description or "User-registered gamma-line file.",
        "capabilities": ["peak-identification", "user-supplied"],
        "metadata": {"registry": "user"},
    }
    existing.append(entry)
    payload["version"] = _USER_LIBRARY_REGISTRY_VERSION
    payload["user_gamma_sources"] = existing
    _save_user_library_registry_payload(payload)
    return list_registered_user_gamma_sources()[-1]


def remove_user_gamma_source(source_id: str) -> bool:
    payload = _load_user_library_registry_payload()
    existing = [
        item
        for item in payload.get("user_gamma_sources", [])
        if isinstance(item, dict)
    ]
    filtered = [
        item
        for item in existing
        if str(item.get("source_id", "") or "").strip() != str(source_id).strip()
    ]
    if len(filtered) == len(existing):
        return False
    payload["version"] = _USER_LIBRARY_REGISTRY_VERSION
    payload["user_gamma_sources"] = filtered
    _save_user_library_registry_payload(payload)
    return True


def _rows_from_yaml_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("lines") or payload.get("rows") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("YAML source must contain a list of rows.")
    return [dict(item) for item in rows if isinstance(item, dict)]


def _build_sqlite_gamma_query(connection: sqlite3.Connection, table: str) -> str:
    column_rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    available_columns = {str(row[1]).lower() for row in column_rows}
    if not available_columns:
        raise ValueError(f"SQLite gamma source table not found: {table}")

    def _select_expr(column: str, default: float) -> str:
        if column.lower() in available_columns:
            return f"COALESCE({column}, {default}) AS {column}"
        return f"{default} AS {column}"

    return (
        "SELECT nuclide, energy_keV, intensity, "
        f"{_select_expr('half_life_s', 0.0)}, "
        f"{_select_expr('energy_unc_keV', 0.0)}, "
        f"{_select_expr('intensity_unc', 0.0)}, "
        f"{_select_expr('norm', 1.0)}, "
        f"{_select_expr('norm_unc', 0.0)} FROM {table}"
    )


def _load_custom_gamma_source_from_locator(locator: str | Path) -> GammaDatabase:
    source = _normalize_source_locator(locator)
    parsed = urlparse(source)
    scheme = parsed.scheme.lower()

    if scheme in {"", "file"}:
        local_path = Path(parsed.path if scheme == "file" else source)
        if local_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError(
                    "PyYAML is required to load YAML gamma sources."
                ) from exc
            payload = yaml.safe_load(local_path.read_text(encoding="utf-8"))
            return _build_gamma_database_from_rows(_rows_from_yaml_payload(payload))
        return _load_custom_gamma_source(local_path)

    if scheme in {"http", "https"}:
        require_network_access("Remote gamma-data source", source)
        with urlopen(
            source
        ) as response:  # nosec - user-supplied public source by design
            raw = response.read().decode("utf-8")
        if parsed.path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError(
                    "PyYAML is required to load YAML gamma sources."
                ) from exc
            payload = yaml.safe_load(raw)
            return _build_gamma_database_from_rows(_rows_from_yaml_payload(payload))
        if parsed.path.lower().endswith(".csv"):
            rows = list(csv.DictReader(raw.splitlines()))
            return _build_gamma_database_from_rows(rows)
        payload = json.loads(raw)
        if isinstance(payload, dict) and all(
            isinstance(value, dict) for value in payload.values()
        ):
            temp_path = Path("/tmp/fluxforge_remote_gamma_source.json")
            temp_path.write_text(json.dumps(payload), encoding="utf-8")
            return GammaDatabase(str(temp_path))
        rows = payload.get("lines") if isinstance(payload, dict) else payload
        return _build_gamma_database_from_rows(rows or [])

    if scheme == "sqlite":
        db_path = Path(parsed.path)
        params = parse_qs(parsed.query)
        table = params.get("table", ["gamma_lines"])[0]
        query = params.get("query", [""])[0]
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            sql = query or _build_sqlite_gamma_query(connection, table)
            rows = [dict(row) for row in connection.execute(sql)]
        return _build_gamma_database_from_rows(rows)

    if scheme == "python":
        module_name = parsed.netloc or parsed.path.strip("/")
        func_name = parsed.fragment or "load"
        if ":" in module_name:
            module_name, func_name = module_name.split(":", 1)
        import importlib

        module = importlib.import_module(module_name)
        loader = getattr(module, func_name)
        payload = loader()
        if isinstance(payload, GammaDatabase):
            return payload
        if isinstance(payload, dict) and all(
            isinstance(value, dict) for value in payload.values()
        ):
            database = _empty_gamma_database()
            for nuclide, item in payload.items():
                database._nuclides[str(nuclide)] = DecayData(
                    nuclide=str(nuclide),
                    zai=int(item.get("zai", 0) or 0),
                    halflife=float(item.get("half_life_s", 0.0) or 0.0),
                    gamma_lines=[
                        GammaLine(
                            energy=float(line["energy_keV"]) * 1000.0,
                            energy_unc=float(line.get("energy_unc_keV", 0.0) or 0.0)
                            * 1000.0,
                            intensity=float(line.get("intensity", 0.0) or 0.0),
                            intensity_unc=float(line.get("intensity_unc", 0.0) or 0.0),
                            norm=float(line.get("norm", 1.0) or 1.0),
                            norm_unc=float(line.get("norm_unc", 0.0) or 0.0),
                        )
                        for line in item.get("gamma_lines", [])
                    ],
                )
            return database
        if isinstance(payload, list):
            return _build_gamma_database_from_rows(payload)
        raise ValueError(
            "Python connector must return a GammaDatabase, a row list, or a nuclide dictionary payload."
        )

    raise ValueError(f"Unsupported external data source scheme: {scheme}")


def list_nuclear_data_sources(
    custom_paths: Iterable[str | Path] = (),
) -> tuple[NuclearDataSourceRecord, ...]:
    records = list(_builtin_nuclear_data_sources())
    records.extend(list_registered_user_gamma_sources())
    for index, custom_path in enumerate(custom_paths):
        path = Path(custom_path)
        records.append(
            NuclearDataSourceRecord(
                source_id=f"custom_gamma_{index + 1}",
                label=path.stem,
                kind="custom-gamma-library",
                description="Registered user gamma-line file.",
                builtin=False,
                path_hint=str(path),
                capabilities=("peak-identification", "user-supplied"),
            )
        )
    return tuple(records)


def list_nuclear_data_sources_by_capability(
    capability: str,
    custom_paths: Iterable[str | Path] = (),
) -> tuple[NuclearDataSourceRecord, ...]:
    return tuple(
        record
        for record in list_nuclear_data_sources(custom_paths)
        if capability in record.capabilities
    )


def get_nuclear_data_source(
    source_id: str, custom_paths: Iterable[str | Path] = ()
) -> NuclearDataSourceRecord:
    for record in list_nuclear_data_sources(custom_paths):
        if record.source_id == source_id:
            return record
    raise KeyError(f"Unknown nuclear data source: {source_id}")


def load_gamma_identification_source(
    source_id: str,
    *,
    custom_path: str | Path | None = None,
) -> GammaDatabase:
    if source_id in {"decay_2012", "actigamma_2012"}:
        datafile = find_actigamma_data()
        if datafile:
            return GammaDatabase(datafile)
        return GammaDatabase()
    if source_id == "fluxforge_bundled_gamma":
        return GammaDatabase(FLUXFORGE_GAMMA_DATA)
    if source_id == "gsa_v4_edit_library":
        return _load_gsa_gamma_source(_GSA_V4_LIBEDIT)
    if source_id == "gsa_v4_natural_library":
        return _load_gsa_gamma_source(_GSA_V4_LIB_GAMMA_NATURAL)
    if source_id == "nasa_common_lab_sources":
        return _load_nasa_gamma_source(_NASA_COMMON_LAB_SOURCES)
    if source_id == "nasa_natural_radiation":
        return _load_nasa_gamma_source(_NASA_NATURAL_RADIATION)
    if source_id == "nasa_capture_capgam":
        return _load_nasa_gamma_source(_NASA_CAPTURE_CAPGAM)
    if source_id == "nasa_capture_iaea":
        return _load_nasa_gamma_source(_NASA_CAPTURE_IAEA)
    if source_id == "nasa_delayed_activation_iaea":
        return _load_nasa_gamma_source(_NASA_DELAYED_ACTIVATION_IAEA)
    if source_id == "nasa_inelastic_baghdad":
        return _load_nasa_gamma_source(_NASA_INELASTIC_BAGHDAD)
    if source_id == "nasa_inelastic_14mev_talys":
        return _load_nasa_gamma_source(_NASA_INELASTIC_14MEV_TALYS)
    if source_id == "nasa_talys_14mev":
        return _load_nasa_gamma_source(_NASA_TALYS_14MEV)
    if source_id == "endfb8_decay_supplement":
        return _load_custom_gamma_source(_ENDFB8_DECAY_SUPPLEMENT)
    if source_id == "nndc_offline_activation":
        return _build_nndc_gamma_database()
    if source_id == "custom_gamma_file":
        if custom_path is None:
            raise ValueError(
                "A custom data-source path is required for custom gamma files."
            )
        return _load_custom_gamma_source_from_locator(custom_path)
    if source_id.startswith("custom_gamma_") or source_id.startswith("user_gamma_"):
        record = get_nuclear_data_source(
            source_id, custom_paths=[custom_path] if custom_path else ()
        )
        if not record.path_hint:
            raise ValueError("Custom gamma source path is not available.")
        return _load_custom_gamma_source_from_locator(record.path_hint)
    raise ValueError(f"Source '{source_id}' is not a gamma identification source.")


def load_decay_dataset_source(source_id: str) -> DecayDataset:
    if source_id == "radioactivedecay_icrp107":
        return DecayDataset.from_radioactivedecay_npz(_ICRP107_DECAY_DATA)
    if source_id in {
        "radioactivedecay_icrp107_kayzero_2020",
        "radioactivedecay_icrp107_kayzero_2023",
    }:
        half_life_table = load_kayzero_half_life_table(
            _KAYZERO_K0_2023
            if source_id.endswith("_2023")
            else _KAYZERO_K0_2020
        )
        half_life_overrides_s = {
            nuclide: row.half_life_s for nuclide, row in half_life_table.items()
        }
        half_life_uncertainties_s = {
            nuclide: row.half_life_uncertainty_s
            for nuclide, row in half_life_table.items()
            if row.half_life_uncertainty_s is not None
        }
        return DecayDataset.from_radioactivedecay_npz(
            _ICRP107_DECAY_DATA,
            half_life_overrides_s=half_life_overrides_s,
            half_life_uncertainties_s=half_life_uncertainties_s,
        )
    raise ValueError(f"Source '{source_id}' is not a decay dataset source.")


def summarize_nuclear_data_source(
    source_id: str,
    *,
    custom_path: str | Path | None = None,
) -> str:
    record = get_nuclear_data_source(
        source_id, custom_paths=[custom_path] if custom_path else ()
    )
    parts = [f"{record.label} ({record.kind})", record.description]
    if record.path_hint:
        parts.append(f"path={record.path_hint}")
    if record.metadata:
        details = ", ".join(f"{key}={value}" for key, value in record.metadata.items())
        parts.append(details)
    if record.capabilities:
        parts.append("capabilities=" + ", ".join(record.capabilities))
    if custom_path and is_remote_locator(str(custom_path)) and offline_mode_enabled():
        parts.append("offline_mode=remote HTTP(S) access disabled")
    return " | ".join(parts)


__all__ = [
    "NuclearDataSourceRecord",
    "get_nuclear_data_source",
    "list_nuclear_data_sources_by_capability",
    "list_nuclear_data_sources",
    "list_registered_user_gamma_sources",
    "load_decay_dataset_source",
    "load_gamma_identification_source",
    "register_user_gamma_source",
    "remove_user_gamma_source",
    "summarize_nuclear_data_source",
]
