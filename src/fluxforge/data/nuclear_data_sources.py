"""Nuclear data source registry and adapters for FluxForge.

This module keeps distinct nuclear-data sources separate so GUI and workflow
code can expose provenance-aware selection instead of treating everything as a
single bundled database.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

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
from fluxforge.data.nndc import GAMMA_LINES as NNDC_GAMMA_LINES
from fluxforge.data.nndc import HALF_LIVES_S as NNDC_HALF_LIVES_S
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
    actigamma_path = find_actigamma_data()
    records = [
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
            source_id="nndc_offline_activation",
            label="NNDC offline activation subset",
            kind="gamma-library",
            description="Offline NNDC-inspired activation subset for common NAA and dosimetry products.",
            capabilities=("peak-identification", "activation-reference"),
            metadata={"nuclides": len(NNDC_GAMMA_LINES)},
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
    ]
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
    if source_id == "nndc_offline_activation":
        return _build_nndc_gamma_database()
    if source_id == "custom_gamma_file":
        if custom_path is None:
            raise ValueError(
                "A custom data-source path is required for custom gamma files."
            )
        return _load_custom_gamma_source_from_locator(custom_path)
    if source_id.startswith("custom_gamma_"):
        record = get_nuclear_data_source(
            source_id, custom_paths=[custom_path] if custom_path else ()
        )
        if not record.path_hint:
            raise ValueError("Custom gamma source path is not available.")
        return _load_custom_gamma_source_from_locator(record.path_hint)
    raise ValueError(f"Source '{source_id}' is not a gamma identification source.")


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
    return " | ".join(parts)


__all__ = [
    "NuclearDataSourceRecord",
    "get_nuclear_data_source",
    "list_nuclear_data_sources",
    "load_gamma_identification_source",
    "summarize_nuclear_data_source",
]
