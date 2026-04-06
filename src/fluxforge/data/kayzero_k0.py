"""Import helpers for Kayzero k0 library data.

This module provides a one-way bridge from a user-supplied Kayzero library
folder or archive into FluxForge's governed k0 library JSON format. The current
import path intentionally prioritizes transparent provenance and unresolved-field
reporting over pretending to have a complete TECDOC-grade import.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List
import zipfile

from fluxforge.data.elements import atomic_mass as element_atomic_mass
from fluxforge.data.gamma_database import get_database
from fluxforge.data.k0_library import (
    GovernedLibrary,
    K0LibraryRecord,
    get_k0_library_record,
)
from fluxforge.data.nndc import ATOMIC_MASSES as ISOTOPE_ATOMIC_MASSES
from fluxforge.data.nndc import GAMMA_LINES as NNDC_GAMMA_LINES
from fluxforge.data.nndc import HALF_LIVES_S
from fluxforge.data.nndc import NATURAL_ABUNDANCES, format_isotope, parse_isotope


_SUPPLEMENTAL_GAMMA_LINES: dict[str, list[tuple[float, float, str]]] = {
    "Ag-109m": [(88.0336, 0.037, "endfb8_decay")],
    "Ge-77m": [(159.7, 0.1033296, "endfb8_decay")],
    "Hf-179m": [(214.335, 0.953, "endfb8_decay")],
    "Lu-176m": [(88.361, 0.08893943, "endfb8_decay")],
    "Re-188m": [(105.96, 0.10815, "endfb8_decay")],
    "Tm-170": [(84.25473, 0.02510443, "endfb8_decay")],
    "U-239": [(74.664, 0.492, "endfb8_decay")],
}

_SUPPLEMENTAL_ISOTOPIC_ABUNDANCES: dict[str, float] = {
    "U-238": 0.992745,
}


@dataclass(frozen=True)
class KayzeroImportResult:
    """Result payload for a Kayzero k0 library import."""

    library: GovernedLibrary
    report: Dict[str, Any]


@dataclass(frozen=True)
class KayzeroHalfLifeValue:
    """Resolved half-life value parsed from a Kayzero ``uT12`` table."""

    nuclide: str
    half_life_s: float
    half_life_uncertainty_s: float | None
    source_name: str


@dataclass(frozen=True)
class _KayzeroGammaLine:
    nuclide: str
    energy_keV: float
    k0: float
    dk0_percent: float | None
    k0_code: int | None


@dataclass(frozen=True)
class _KayzeroQ0Row:
    nuclide: str
    Q0: float
    dQ0_percent: float | None
    E_res_eV: float | None
    dE_res_eV: float | None


@dataclass(frozen=True)
class _KayzeroHalfLifeRow:
    nuclide: str
    raw_value: float
    raw_uncertainty: float | None


@dataclass(frozen=True)
class _IriReactionRow:
    product_isotope: str
    target_isotope: str
    element: str
    atomic_mass_g_mol: float | None
    isotopic_abundance: float | None
    sigma_0_barn: float | None
    Q0: float | None
    E_res_eV: float | None


@dataclass(frozen=True)
class _IriGammaLine:
    energy_keV: float
    intensity_fraction: float
    is_primary: bool


def _metastable_family(product_isotope: str) -> list[str]:
    element, mass_number, metastable = parse_isotope(product_isotope)
    variants = [format_isotope(element, mass_number)]
    if metastable <= 1:
        variants.append(format_isotope(element, mass_number, 1))
    else:
        variants.append(format_isotope(element, mass_number, metastable))
        variants.append(format_isotope(element, mass_number, 1))
    return list(dict.fromkeys([product_isotope, *variants]))


class _KayzeroSource:
    def __init__(self, source_path: str | Path):
        self.path = Path(source_path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)

    def iter_names(self) -> Iterator[str]:
        if self._zip is not None:
            yield from self._zip.namelist()
            return
        if self.path.is_dir():
            for child in self.path.rglob("*"):
                if child.is_file():
                    yield str(child.relative_to(self.path)).replace("\\", "/")
            return
        raise FileNotFoundError(f"Kayzero input path was not found: {self.path}")

    def read_text(self, name: str) -> str:
        if self._zip is not None:
            return self._zip.read(name).decode("utf-8", errors="ignore")
        return (self.path / name).read_text(encoding="utf-8", errors="ignore")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _normalize_kayzero_nuclide(label: str) -> str:
    text = str(label or "").strip().replace("*", "m")
    if not text:
        raise ValueError("Empty nuclide label")
    element, mass_number, metastable = parse_isotope(text)
    return format_isotope(element, mass_number, metastable)


def _find_versioned_member(
    names: Iterable[str], suffix: str, preferred_version: str | None = None
) -> str | None:
    candidates = []
    pattern = re.compile(r"k0-(\d{4})" + re.escape(suffix) + r"$", re.IGNORECASE)
    for name in names:
        normalized = name.replace("\\", "/")
        if "/__MACOSX/" in f"/{normalized}" or "/._" in f"/{normalized}":
            continue
        match = pattern.search(normalized)
        if match:
            candidates.append((int(match.group(1)), normalized))
    if not candidates:
        return None
    if preferred_version is not None:
        preferred = [
            name for year, name in candidates if str(year) == str(preferred_version)
        ]
        if preferred:
            return sorted(preferred)[-1]
    candidates.sort()
    return candidates[-1][1]


def _find_named_member(names: Iterable[str], basename: str) -> str | None:
    normalized_basename = basename.lower()
    candidates = []
    for name in names:
        normalized = name.replace("\\", "/")
        lowered = normalized.lower()
        if "/__MACOSX/" in f"/{normalized}" or "/._" in f"/{normalized}":
            continue
        if (
            lowered.endswith(f"/{normalized_basename}")
            or lowered == normalized_basename
        ):
            candidates.append(normalized)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _parse_k0_lines(text: str) -> list[_KayzeroGammaLine]:
    rows: list[_KayzeroGammaLine] = []
    if "\t" in text:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            nuclide_text = str(row.get("Nuclide") or "").strip()
            energy_text = str(row.get("E (keV)") or "").strip()
            k0_text = str(row.get("k0") or "").strip()
            if not nuclide_text or not energy_text or not k0_text:
                continue
            dk0_text = str(row.get("dk0") or "").strip()
            code_text = str(row.get("k0code") or "").strip()
            rows.append(
                _KayzeroGammaLine(
                    nuclide=_normalize_kayzero_nuclide(nuclide_text),
                    energy_keV=float(energy_text),
                    k0=float(k0_text),
                    dk0_percent=(
                        float(dk0_text) if dk0_text not in {"", "-", "--"} else None
                    ),
                    k0_code=(
                        int(float(code_text))
                        if code_text not in {"", "-", "--"}
                        else None
                    ),
                )
            )
        return rows

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("nuclide"):
            continue
        tokens = line.split()
        if len(tokens) < 4:
            continue
        nuclide = _normalize_kayzero_nuclide(tokens[0])
        energy_keV = float(tokens[1])
        k0 = float(tokens[2])
        if len(tokens) >= 5:
            dk0_percent = float(tokens[3]) if tokens[3] not in {"-", "--"} else None
            k0_code = int(float(tokens[4]))
        else:
            dk0_percent = None
            k0_code = int(float(tokens[3])) if tokens[3] not in {"-", "--"} else None
        rows.append(
            _KayzeroGammaLine(
                nuclide=nuclide,
                energy_keV=energy_keV,
                k0=k0,
                dk0_percent=dk0_percent,
                k0_code=k0_code,
            )
        )
    return rows


def _parse_q0_rows(text: str) -> dict[str, _KayzeroQ0Row]:
    rows: dict[str, _KayzeroQ0Row] = {}
    if "\t" in text:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            nuclide_text = str(row.get("Nuclide") or "").strip()
            q0_text = str(row.get("Q0") or "").strip()
            if not nuclide_text or not q0_text:
                continue
            rows[_normalize_kayzero_nuclide(nuclide_text)] = _KayzeroQ0Row(
                nuclide=_normalize_kayzero_nuclide(nuclide_text),
                Q0=float(q0_text),
                dQ0_percent=(
                    float(row["dQ0"])
                    if str(row.get("dQ0") or "").strip() not in {"", "-", "--"}
                    else None
                ),
                E_res_eV=(
                    float(row["Er"])
                    if str(row.get("Er") or "").strip() not in {"", "-", "--"}
                    else None
                ),
                dE_res_eV=(
                    float(row["dEr"])
                    if str(row.get("dEr") or "").strip() not in {"", "-", "--"}
                    else None
                ),
            )
        return rows

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("nuclide"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        nuclide = _normalize_kayzero_nuclide(tokens[0])
        q0 = float(tokens[1])
        d_q0 = (
            float(tokens[2])
            if len(tokens) >= 3 and tokens[2] not in {"-", "--"}
            else None
        )
        e_res = (
            float(tokens[3])
            if len(tokens) >= 4 and tokens[3] not in {"-", "--"}
            else None
        )
        d_e_res = (
            float(tokens[4])
            if len(tokens) >= 5 and tokens[4] not in {"-", "--"}
            else None
        )
        rows[nuclide] = _KayzeroQ0Row(
            nuclide=nuclide, Q0=q0, dQ0_percent=d_q0, E_res_eV=e_res, dE_res_eV=d_e_res
        )
    return rows


def _parse_half_life_rows(text: str) -> dict[str, _KayzeroHalfLifeRow]:
    rows: dict[str, _KayzeroHalfLifeRow] = {}
    if "\t" in text:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            nuclide_text = str(row.get("") or row.get("Nuclide") or "").strip()
            value_text = str(row.get("T1/2") or row.get("t1/2") or "").strip()
            if not nuclide_text or not value_text:
                continue
            uncertainty_text = str(
                row.get("dT") or row.get("dT12") or row.get("dT1/2") or ""
            ).strip()
            rows[_normalize_kayzero_nuclide(nuclide_text)] = _KayzeroHalfLifeRow(
                nuclide=_normalize_kayzero_nuclide(nuclide_text),
                raw_value=float(value_text),
                raw_uncertainty=(
                    float(uncertainty_text)
                    if uncertainty_text not in {"", "-", "--"}
                    else None
                ),
            )
        return rows

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("t1/2"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        nuclide = _normalize_kayzero_nuclide(tokens[0])
        value = float(tokens[1])
        uncertainty = (
            float(tokens[2])
            if len(tokens) >= 3 and tokens[2] not in {"-", "--"}
            else None
        )
        rows[nuclide] = _KayzeroHalfLifeRow(
            nuclide=nuclide, raw_value=value, raw_uncertainty=uncertainty
        )
    return rows


def _parse_md_codes(text: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            rows[_normalize_kayzero_nuclide(tokens[0])] = int(float(tokens[1]))
        except ValueError:
            continue
    return rows


def _parse_iri_mb1_rows(text: str) -> dict[str, _IriReactionRow]:
    rows: dict[str, _IriReactionRow] = {}
    current_element: str | None = None
    current_atomic_mass: float | None = None
    current_reaction: tuple[str, str] | None = None
    float_pattern = r"[-+]?\d+(?:\.\d+)?(?:E[-+]?\d+)?"
    reaction_pattern = re.compile(
        r"^\s*([A-Za-z]{1,2}-\d+(?:m\d?|\*)?)\s*->\s*([A-Za-z]{1,2}-\d+(?:m\d?|\*)?)"
    )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[A-Z][a-z]?", stripped):
            current_element = stripped
            current_atomic_mass = None
            current_reaction = None
            continue
        if stripped.startswith("M:"):
            mass_match = re.search(float_pattern, stripped)
            current_atomic_mass = float(mass_match.group(0)) if mass_match else None
            continue
        reaction_match = reaction_pattern.match(stripped)
        if reaction_match:
            current_reaction = (
                _normalize_kayzero_nuclide(reaction_match.group(1)),
                _normalize_kayzero_nuclide(reaction_match.group(2)),
            )
            continue
        if current_reaction is None or not stripped.startswith("("):
            continue
        q0_match = re.search(r"Q0:\s*(%s)" % float_pattern, stripped)
        e_res_match = re.search(r"Er:\s*(%s)" % float_pattern, stripped)
        sigma_match = re.search(
            r",\s*[^:,()]+:\s*(%s)\s*(m?b)" % float_pattern, stripped
        )
        if sigma_match is None or sigma_match.group(2).lower() != "b":
            continue
        numeric_tokens = re.findall(float_pattern, stripped)
        abundance = float(numeric_tokens[0]) if numeric_tokens else None
        target_isotope, product_isotope = current_reaction
        element, _, _ = parse_isotope(product_isotope)
        rows[product_isotope] = _IriReactionRow(
            product_isotope=product_isotope,
            target_isotope=target_isotope,
            element=current_element or element,
            atomic_mass_g_mol=current_atomic_mass,
            isotopic_abundance=abundance,
            sigma_0_barn=float(sigma_match.group(1)),
            Q0=float(q0_match.group(1)) if q0_match else None,
            E_res_eV=float(e_res_match.group(1)) if e_res_match else None,
        )
    return rows


def _parse_iri_mb1_parent_map(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    reaction_pattern = re.compile(
        r"^\s*([A-Za-z]{1,2}-\d+(?:m\d?|\*)?)\s*->\s*([A-Za-z]{1,2}-\d+(?:m\d?|\*)?)"
    )
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        reaction_match = reaction_pattern.match(stripped)
        if reaction_match is None:
            continue
        parent_isotope = _normalize_kayzero_nuclide(reaction_match.group(1))
        product_isotope = _normalize_kayzero_nuclide(reaction_match.group(2))
        parents = rows.setdefault(product_isotope, [])
        if parent_isotope not in parents:
            parents.append(parent_isotope)
    return rows


def _parse_iri_mb2_rows(text: str) -> dict[str, list[_IriGammaLine]]:
    rows: dict[str, list[_IriGammaLine]] = {}
    current_nuclide: str | None = None
    in_peaks = False
    nuclide_pattern = re.compile(r"^\s*([A-Za-z]{1,2}-\d+(?:m\d?|\*)?)\s*\(")
    peak_pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+keV:\s*(.*?)\s*$", re.IGNORECASE)
    float_pattern = r"[-+]?\d+(?:\.\d+)?(?:E[-+]?\d+)?"

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            in_peaks = False
            continue
        nuclide_match = nuclide_pattern.match(line)
        if nuclide_match:
            current_nuclide = _normalize_kayzero_nuclide(nuclide_match.group(1))
            rows.setdefault(current_nuclide, [])
            in_peaks = False
            continue
        if stripped == "Peaks:":
            in_peaks = True
            continue
        if not in_peaks or current_nuclide is None:
            continue
        peak_match = peak_pattern.match(line)
        if peak_match is None:
            continue
        energy_keV = float(peak_match.group(1))
        remainder = peak_match.group(2)
        intensity_match = re.search(float_pattern, remainder)
        if intensity_match is None or "(" in remainder:
            continue
        rows[current_nuclide].append(
            _IriGammaLine(
                energy_keV=energy_keV,
                intensity_fraction=float(intensity_match.group(0)) / 100.0,
                is_primary="*" in remainder,
            )
        )
    return rows


def _parse_iri_mb3_rows(text: str) -> dict[str, list[_IriGammaLine]]:
    rows: dict[str, list[_IriGammaLine]] = {}
    unit_tokens = {"s", "m", "h", "d", "y"}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 5:
            continue
        try:
            energy_keV = float(tokens[0])
        except ValueError:
            continue
        try:
            nuclide = _normalize_kayzero_nuclide(tokens[1])
        except ValueError:
            continue
        unit_index = None
        for index in range(2, len(tokens) - 1):
            if tokens[index].lower() in unit_tokens:
                unit_index = index
                break
        if unit_index is None or unit_index + 1 >= len(tokens):
            continue
        intensity_text = tokens[unit_index + 1]
        if intensity_text in {"(S)", "(E)"}:
            continue
        try:
            intensity_fraction = float(intensity_text) / 100.0
        except ValueError:
            continue
        rows.setdefault(nuclide, []).append(
            _IriGammaLine(
                energy_keV=energy_keV,
                intensity_fraction=float(intensity_fraction),
                is_primary=False,
            )
        )
    return rows


def _parse_fcd_rows(text: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "fcd":
            continue
        if "," not in line:
            continue
        nuclide, value = [part.strip() for part in line.split(",", 1)]
        try:
            rows[_normalize_kayzero_nuclide(nuclide)] = float(value)
        except ValueError:
            continue
    return rows


def _detect_half_life_scale(
    raw_half_lives: dict[str, _KayzeroHalfLifeRow]
) -> tuple[float, str]:
    minute_votes = 0
    second_votes = 0
    for nuclide, row in raw_half_lives.items():
        starter = get_k0_library_record(nuclide)
        if starter is None or row.raw_value <= 0.0:
            continue
        ratio = starter.half_life_s / row.raw_value
        if 50.0 <= ratio <= 70.0:
            minute_votes += 1
        elif 0.8 <= ratio <= 1.2:
            second_votes += 1
    if minute_votes >= max(second_votes, 1):
        return 60.0, "minutes_to_seconds"
    return 1.0, "seconds"


def load_kayzero_half_life_table(
    source_path: str | Path,
    *,
    preferred_version: str | None = None,
) -> dict[str, KayzeroHalfLifeValue]:
    """Load and scale half-life rows from a Kayzero ``uT12`` table or bundle."""

    path = Path(source_path)
    if path.is_file() and path.suffix.lower() == ".ut12":
        source_name = path.name
        rows = _parse_half_life_rows(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        source = _KayzeroSource(path)
        try:
            names = list(source.iter_names())
            member = _find_versioned_member(names, ".uT12", preferred_version)
            if member is None:
                raise FileNotFoundError(
                    "Kayzero half-life loading requires a uT12 table."
                )
            source_name = member
            rows = _parse_half_life_rows(source.read_text(member))
        finally:
            source.close()

    scale, _ = _detect_half_life_scale(rows)
    return {
        nuclide: KayzeroHalfLifeValue(
            nuclide=nuclide,
            half_life_s=float(row.raw_value) * scale,
            half_life_uncertainty_s=(
                float(row.raw_uncertainty) * scale
                if row.raw_uncertainty is not None
                else None
            ),
            source_name=source_name,
        )
        for nuclide, row in rows.items()
    }


def _product_and_target_isotopes(product_isotope: str) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    element, mass_number, _ = parse_isotope(product_isotope)
    if mass_number <= 1:
        return element, product_isotope, ["target_isotope_unresolved"]
    target_isotope = format_isotope(element, mass_number - 1)
    notes.append("target_isotope_inferred_as_n_gamma_parent")
    return element, target_isotope, notes


def _match_gamma_intensity(
    iri_lines: list[_IriGammaLine], energy_keV: float
) -> float | None:
    for line in iri_lines:
        if abs(float(line.energy_keV) - float(energy_keV)) <= 1.0:
            return float(line.intensity_fraction)
    return None


def _supplemental_gamma_intensity(
    product_isotope: str, energy_keV: float
) -> tuple[float, str] | None:
    for candidate in _metastable_family(product_isotope):
        for (
            line_energy_keV,
            intensity_fraction,
            source_name,
        ) in _SUPPLEMENTAL_GAMMA_LINES.get(candidate, []):
            if abs(float(line_energy_keV) - float(energy_keV)) <= 1.0:
                return float(intensity_fraction), source_name
        for line_energy_keV, intensity_fraction in NNDC_GAMMA_LINES.get(candidate, []):
            if abs(float(line_energy_keV) - float(energy_keV)) <= 1.0:
                return float(intensity_fraction), "nndc_offline"
    return None


def _resolve_iri_reaction_row(
    product_isotope: str,
    direct_rows: dict[str, _IriReactionRow],
    parent_map: dict[str, list[str]],
    *,
    _seen: set[str] | None = None,
) -> tuple[_IriReactionRow | None, str | None]:
    for candidate in _metastable_family(product_isotope):
        direct = direct_rows.get(candidate)
        if direct is not None:
            return direct, "iri_direct"

    seen = set() if _seen is None else set(_seen)
    if product_isotope in seen:
        return None, None
    seen.add(product_isotope)

    for candidate in _metastable_family(product_isotope):
        for parent_isotope in parent_map.get(candidate, []):
            resolved, source = _resolve_iri_reaction_row(
                parent_isotope, direct_rows, parent_map, _seen=seen
            )
            if resolved is not None:
                return resolved, "iri_ancestor" if source else "iri_ancestor"

    inferred_capture = _infer_capture_reaction_row(product_isotope, parent_map)
    if inferred_capture is not None:
        return inferred_capture, "iri_inferred_capture"

    element, mass_number, metastable = parse_isotope(product_isotope)
    sibling_candidates = (
        [format_isotope(element, mass_number, 1)]
        if metastable == 0
        else [format_isotope(element, mass_number)]
    )
    for sibling in sibling_candidates:
        if sibling == product_isotope:
            continue
        resolved, source = _resolve_iri_reaction_row(
            sibling, direct_rows, parent_map, _seen=seen
        )
        if resolved is not None:
            return resolved, "iri_sibling"
    return None, None


def _supplemental_isotopic_abundance(target_isotope: str) -> float | None:
    return _SUPPLEMENTAL_ISOTOPIC_ABUNDANCES.get(target_isotope)


def _infer_capture_reaction_row(
    product_isotope: str,
    parent_map: dict[str, list[str]],
    *,
    _seen: set[str] | None = None,
) -> _IriReactionRow | None:
    seen = set() if _seen is None else set(_seen)
    if product_isotope in seen:
        return None
    seen.add(product_isotope)

    element, mass_number, _ = parse_isotope(product_isotope)
    for candidate in _metastable_family(product_isotope):
        for parent_isotope in parent_map.get(candidate, []):
            parent_element, parent_mass_number, _ = parse_isotope(parent_isotope)
            if parent_element == element and parent_mass_number == mass_number - 1:
                target_atomic_mass = ISOTOPE_ATOMIC_MASSES.get(parent_isotope)
                if target_atomic_mass is None:
                    target_atomic_mass = element_atomic_mass(parent_element)
                return _IriReactionRow(
                    product_isotope=product_isotope,
                    target_isotope=parent_isotope,
                    element=parent_element,
                    atomic_mass_g_mol=(
                        float(target_atomic_mass)
                        if target_atomic_mass is not None
                        else None
                    ),
                    isotopic_abundance=NATURAL_ABUNDANCES.get(parent_isotope)
                    or _supplemental_isotopic_abundance(parent_isotope),
                    sigma_0_barn=None,
                    Q0=None,
                    E_res_eV=None,
                )
            inferred = _infer_capture_reaction_row(
                parent_isotope, parent_map, _seen=seen
            )
            if inferred is not None:
                return inferred
    return None


def _resolve_ancestor_q0_row(
    product_isotope: str,
    q0_rows: dict[str, _KayzeroQ0Row],
    parent_map: dict[str, list[str]],
    *,
    _seen: set[str] | None = None,
) -> tuple[_KayzeroQ0Row | None, str | None]:
    for candidate in _metastable_family(product_isotope):
        row = q0_rows.get(candidate)
        if row is not None:
            return row, "uq0_direct"

    seen = set() if _seen is None else set(_seen)
    if product_isotope in seen:
        return None, None
    seen.add(product_isotope)

    for candidate in _metastable_family(product_isotope):
        for parent_isotope in parent_map.get(candidate, []):
            row, source = _resolve_ancestor_q0_row(
                parent_isotope, q0_rows, parent_map, _seen=seen
            )
            if row is not None:
                return row, "uq0_ancestor" if source else "uq0_ancestor"

    element, mass_number, metastable = parse_isotope(product_isotope)
    sibling_candidates = (
        [format_isotope(element, mass_number, 1)]
        if metastable == 0
        else [format_isotope(element, mass_number)]
    )
    for sibling in sibling_candidates:
        if sibling == product_isotope:
            continue
        row, source = _resolve_ancestor_q0_row(sibling, q0_rows, parent_map, _seen=seen)
        if row is not None:
            return row, "uq0_sibling" if source else "uq0_sibling"
    return None, None


def _derive_sigma0_from_k0(
    *,
    k0_au: float,
    gamma_intensity: float,
    isotopic_abundance: float,
    atomic_mass_g_mol: float,
) -> float | None:
    if gamma_intensity <= 0.0 or isotopic_abundance <= 0.0 or atomic_mass_g_mol <= 0.0:
        return None
    gold_reference = get_k0_library_record("Au-198")
    if (
        gold_reference is None
        or gold_reference.sigma_0_barn <= 0.0
        or gold_reference.gamma_intensity <= 0.0
    ):
        return None
    return float(
        k0_au
        * gold_reference.sigma_0_barn
        * gold_reference.gamma_intensity
        * atomic_mass_g_mol
        / (gamma_intensity * isotopic_abundance * gold_reference.atomic_mass_g_mol)
    )


def _gamma_database_intensity(product_isotope: str, energy_keV: float) -> float | None:
    try:
        database = get_database()
    except Exception:
        return None
    candidate_keys = {
        product_isotope,
        product_isotope.replace("-", ""),
        product_isotope.replace("-", "").replace("m", "M"),
    }
    for key in candidate_keys:
        decay = database.get(key)
        if decay is None:
            continue
        for line in getattr(decay, "gamma_lines", []):
            if abs(float(line.energy) / 1000.0 - float(energy_keV)) <= 1.0:
                return float(line.intensity)
    return None


def _choose_primary_line(
    product_isotope: str,
    lines: list[_KayzeroGammaLine],
    *,
    iri_lines: list[_IriGammaLine] | None = None,
) -> _KayzeroGammaLine:
    starter = get_k0_library_record(product_isotope)
    if starter is not None:
        for line in lines:
            if abs(line.energy_keV - starter.gamma_energy_keV) <= 1.0:
                return line
    if iri_lines:
        preferred = [item for item in iri_lines if item.is_primary]
        for iri_line in preferred:
            for line in lines:
                if abs(line.energy_keV - iri_line.energy_keV) <= 1.0:
                    return line
    return sorted(
        lines,
        key=lambda item: (item.k0_code if item.k0_code is not None else 9999, -item.k0),
    )[0]


def import_kayzero_k0_library(
    source_path: str | Path, *, preferred_version: str | None = None
) -> KayzeroImportResult:
    """Import a Kayzero library folder or zip into a governed FluxForge k0 library.

    The import currently uses the text-discoverable sidecar files and emits a
    detailed report for unresolved fields that still appear to depend on opaque
    Kayzero binary library files.
    """

    source = _KayzeroSource(source_path)
    try:
        names = list(source.iter_names())
        uk0_member = _find_versioned_member(names, ".uk0", preferred_version)
        q0_member = _find_versioned_member(names, ".uQ0", preferred_version)
        t12_member = _find_versioned_member(names, ".uT12", preferred_version)
        md_member = _find_versioned_member(names, ".MDcode", preferred_version)
        fcd_member = _find_versioned_member(names, ".FCd", preferred_version)
        lb1_member = _find_versioned_member(names, ".LB1", preferred_version)
        lb2_member = _find_versioned_member(names, ".LB2", preferred_version)
        iri_mb1_member = _find_named_member(names, "IRI_MB1.TXT")
        iri_mb2_member = _find_named_member(names, "IRI_MB2.TXT")
        iri_mb3_member = _find_named_member(names, "IRI_MB3.TXT")

        if uk0_member is None or q0_member is None or t12_member is None:
            raise FileNotFoundError("Kayzero import requires uk0, uQ0, and uT12 files.")

        gamma_lines = _parse_k0_lines(source.read_text(uk0_member))
        q0_rows = _parse_q0_rows(source.read_text(q0_member))
        half_life_rows = _parse_half_life_rows(source.read_text(t12_member))
        md_codes = (
            {} if md_member is None else _parse_md_codes(source.read_text(md_member))
        )
        fcd_rows = (
            {} if fcd_member is None else _parse_fcd_rows(source.read_text(fcd_member))
        )
        iri_mb1_text = (
            None if iri_mb1_member is None else source.read_text(iri_mb1_member)
        )
        iri_reaction_rows = (
            {} if iri_mb1_text is None else _parse_iri_mb1_rows(iri_mb1_text)
        )
        iri_parent_map = (
            {} if iri_mb1_text is None else _parse_iri_mb1_parent_map(iri_mb1_text)
        )
        iri_gamma_rows = (
            {}
            if iri_mb2_member is None
            else _parse_iri_mb2_rows(source.read_text(iri_mb2_member))
        )
        iri_mb3_rows = (
            {}
            if iri_mb3_member is None
            else _parse_iri_mb3_rows(source.read_text(iri_mb3_member))
        )
        half_life_scale, half_life_scale_note = _detect_half_life_scale(half_life_rows)

        grouped_lines: dict[str, list[_KayzeroGammaLine]] = {}
        for line in gamma_lines:
            grouped_lines.setdefault(line.nuclide, []).append(line)

        records: dict[str, K0LibraryRecord] = {}
        unresolved_records: list[Dict[str, Any]] = []
        missing_field_counts: Counter[str] = Counter()
        fallback_counts: Counter[str] = Counter()

        for product_isotope, lines in sorted(grouped_lines.items()):
            iri_reaction, iri_reaction_source = _resolve_iri_reaction_row(
                product_isotope, iri_reaction_rows, iri_parent_map
            )
            iri_lines = [
                *iri_gamma_rows.get(product_isotope, []),
                *iri_mb3_rows.get(product_isotope, []),
            ]
            primary = _choose_primary_line(product_isotope, lines, iri_lines=iri_lines)
            q0_row, q0_row_source = _resolve_ancestor_q0_row(
                product_isotope, q0_rows, iri_parent_map
            )
            t12_row = half_life_rows.get(product_isotope)
            starter = get_k0_library_record(product_isotope)
            if iri_reaction is not None:
                element = iri_reaction.element
                target_isotope = iri_reaction.target_isotope
                notes = ["target_isotope_from_iri_mb1"]
                fallback_counts[iri_reaction_source or "iri_reaction"] += 1
            else:
                element, target_isotope, notes = _product_and_target_isotopes(
                    product_isotope
                )
            missing_fields: list[str] = []

            gamma_intensity = None
            gamma_source = None
            if iri_lines:
                gamma_intensity = _match_gamma_intensity(iri_lines, primary.energy_keV)
                if gamma_intensity is not None:
                    gamma_source = "iri_mb2"
            if starter is not None:
                if (
                    gamma_intensity is None
                    and abs(primary.energy_keV - starter.gamma_energy_keV) <= 1.0
                ):
                    gamma_intensity = starter.gamma_intensity
                    gamma_source = "starter_primary"
                elif gamma_intensity is None:
                    for energy_keV, intensity in starter.additional_gammas:
                        if abs(primary.energy_keV - float(energy_keV)) <= 1.0:
                            gamma_intensity = float(intensity)
                            gamma_source = "starter_additional"
                            break
            if gamma_intensity is None:
                supplemental_gamma = _supplemental_gamma_intensity(
                    product_isotope, primary.energy_keV
                )
                if supplemental_gamma is not None:
                    gamma_intensity, gamma_source = supplemental_gamma
            if gamma_intensity is None:
                gamma_intensity = _gamma_database_intensity(
                    product_isotope, primary.energy_keV
                )
                if gamma_intensity is not None:
                    gamma_source = "gamma_database"
            if gamma_intensity is None:
                gamma_intensity = 0.0
                missing_fields.append("gamma_intensity")
            else:
                fallback_counts[gamma_source or "gamma_intensity"] += 1

            half_life_s = None
            if t12_row is not None:
                half_life_s = t12_row.raw_value * half_life_scale
            elif product_isotope in HALF_LIVES_S:
                half_life_s = float(HALF_LIVES_S[product_isotope])
                fallback_counts["nndc_half_life"] += 1
            elif starter is not None:
                half_life_s = starter.half_life_s
                fallback_counts["starter_half_life"] += 1
            else:
                half_life_s = 0.0
                missing_fields.append("half_life_s")

            if q0_row is not None:
                q0_value = float(q0_row.Q0)
                if q0_row_source is not None:
                    fallback_counts[q0_row_source] += 1
            elif iri_reaction is not None and iri_reaction.Q0 is not None:
                q0_value = float(iri_reaction.Q0)
                fallback_counts["iri_q0"] += 1
            else:
                q0_value = float(starter.Q0 if starter is not None else 1.0)
            if q0_row is None and iri_reaction is None and starter is not None:
                fallback_counts["starter_q0"] += 1
            elif q0_row is None and iri_reaction is None:
                missing_fields.append("Q0")

            q0_unc_percent = (
                float(q0_row.dQ0_percent or 0.0)
                if q0_row is not None
                else float(starter.Q0_unc_percent if starter is not None else 0.0)
            )
            if q0_row is not None and q0_row.dQ0_percent is not None:
                fallback_counts["uq0_uncertainty"] += 1

            if q0_row is not None and q0_row.E_res_eV is not None:
                e_res_eV = float(q0_row.E_res_eV)
            elif iri_reaction is not None and iri_reaction.E_res_eV is not None:
                e_res_eV = float(iri_reaction.E_res_eV)
                fallback_counts["iri_eres"] += 1
            else:
                e_res_eV = float(starter.E_res_eV if starter is not None else 0.0)
            if q0_row is None and iri_reaction is None and starter is None:
                missing_fields.append("E_res_eV")

            if iri_reaction is not None and iri_reaction.sigma_0_barn is not None:
                sigma_0_barn = float(iri_reaction.sigma_0_barn)
                fallback_counts["iri_sigma0"] += 1
            else:
                sigma_0_barn = (
                    float(starter.sigma_0_barn) if starter is not None else 0.0
                )
            if iri_reaction is None and starter is not None:
                fallback_counts["starter_sigma_i0"] += 1

            isotopic_abundance = (
                iri_reaction.isotopic_abundance
                if iri_reaction is not None
                else NATURAL_ABUNDANCES.get(target_isotope)
            )
            if iri_reaction is not None and iri_reaction.isotopic_abundance is not None:
                fallback_counts["iri_abundance"] += 1
            if isotopic_abundance is None:
                isotopic_abundance = _supplemental_isotopic_abundance(target_isotope)
                if isotopic_abundance is not None:
                    fallback_counts["supplemental_abundance"] += 1
            if isotopic_abundance is None:
                if starter is not None:
                    isotopic_abundance = starter.isotopic_abundance
                    fallback_counts["starter_abundance"] += 1
                else:
                    isotopic_abundance = 1.0
                    missing_fields.append("isotopic_abundance")

            atomic_mass = (
                iri_reaction.atomic_mass_g_mol
                if iri_reaction is not None
                else ISOTOPE_ATOMIC_MASSES.get(target_isotope)
            )
            if iri_reaction is not None and iri_reaction.atomic_mass_g_mol is not None:
                fallback_counts["iri_atomic_mass"] += 1
            if atomic_mass is None:
                atomic_mass = element_atomic_mass(element)
                if atomic_mass > 0.0:
                    fallback_counts["element_atomic_mass"] += 1
                else:
                    missing_fields.append("atomic_mass_g_mol")
                    atomic_mass = 0.0

            if sigma_0_barn <= 0.0:
                derived_sigma = _derive_sigma0_from_k0(
                    k0_au=float(primary.k0),
                    gamma_intensity=float(gamma_intensity),
                    isotopic_abundance=float(isotopic_abundance),
                    atomic_mass_g_mol=float(atomic_mass),
                )
                if derived_sigma is not None:
                    sigma_0_barn = float(derived_sigma)
                    fallback_counts["derived_sigma0_from_k0"] += 1

            if sigma_0_barn > 0.0 and q0_value > 0.0:
                I0_barn = float(sigma_0_barn * q0_value)
                fallback_counts["derived_i0_from_sigma_q0"] += 1
            else:
                I0_barn = float(starter.I0_barn) if starter is not None else 0.0

            if sigma_0_barn <= 0.0:
                missing_fields.append("sigma_0_barn")
            if I0_barn <= 0.0:
                missing_fields.append("I0_barn")

            data_status = (
                "imported_kayzero_text_complete"
                if not missing_fields
                else "imported_kayzero_text_partial"
            )
            if primary.k0_code is not None:
                notes.append(f"kayzero_k0_code:{primary.k0_code}")
            if product_isotope in md_codes:
                notes.append(f"kayzero_md_code:{md_codes[product_isotope]}")
            if product_isotope in fcd_rows:
                notes.append(f"kayzero_fcd:{fcd_rows[product_isotope]:.6g}")
            if gamma_source is not None:
                notes.append(f"gamma_intensity_source:{gamma_source}")
            if missing_fields:
                notes.append("missing_fields:" + ",".join(sorted(set(missing_fields))))

            additional_gammas: list[tuple[float, float]] = []
            seen_additional_energies: set[int] = set()
            for iri_line in iri_lines:
                if abs(iri_line.energy_keV - primary.energy_keV) <= 1.0:
                    continue
                rounded_energy = int(round(iri_line.energy_keV * 10.0))
                if rounded_energy in seen_additional_energies:
                    continue
                additional_gammas.append(
                    (iri_line.energy_keV, iri_line.intensity_fraction)
                )
                seen_additional_energies.add(rounded_energy)
            if starter is not None:
                for energy_keV, fallback_intensity in starter.additional_gammas:
                    rounded_energy = int(round(float(energy_keV) * 10.0))
                    if (
                        rounded_energy in seen_additional_energies
                        or abs(float(energy_keV) - primary.energy_keV) <= 1.0
                    ):
                        continue
                    additional_gammas.append(
                        (float(energy_keV), float(fallback_intensity))
                    )
                    seen_additional_energies.add(rounded_energy)

            records[product_isotope] = K0LibraryRecord(
                product_isotope=product_isotope,
                target_isotope=target_isotope,
                element=element,
                gamma_energy_keV=primary.energy_keV,
                gamma_intensity=float(gamma_intensity),
                half_life_s=float(half_life_s),
                k0_Au=float(primary.k0),
                k0_unc_percent=float(primary.dk0_percent or 0.0),
                Q0=float(q0_value),
                Q0_unc_percent=float(q0_unc_percent),
                E_res_eV=float(e_res_eV),
                sigma_0_barn=float(sigma_0_barn),
                I0_barn=float(I0_barn),
                isotopic_abundance=float(isotopic_abundance),
                atomic_mass_g_mol=float(atomic_mass),
                reaction_family="thermal_capture",
                data_status=data_status,
                source_note=(
                    "Imported from Kayzero text library sidecar files; unresolved fields may still depend on opaque "
                    "Kayzero LB1/LB2 payloads. " + "; ".join(notes)
                ),
                additional_gammas=tuple(additional_gammas),
            )

            if missing_fields:
                for field_name in set(missing_fields):
                    missing_field_counts[field_name] += 1
                unresolved_records.append(
                    {
                        "product_isotope": product_isotope,
                        "primary_energy_keV": primary.energy_keV,
                        "missing_fields": sorted(set(missing_fields)),
                        "available_line_energies_keV": [
                            item.energy_keV for item in lines
                        ],
                    }
                )

        available_year_match = re.search(r"k0-(\d{4})", uk0_member, re.IGNORECASE)
        version = (
            available_year_match.group(1)
            if available_year_match
            else (preferred_version or "unknown")
        )
        report = {
            "source_path": str(source_path),
            "library_version_detected": version,
            "source_files": {
                "uk0": uk0_member,
                "uQ0": q0_member,
                "uT12": t12_member,
                "MDcode": md_member,
                "FCd": fcd_member,
                "LB1": lb1_member,
                "LB2": lb2_member,
                "IRI_MB1": iri_mb1_member,
                "IRI_MB2": iri_mb2_member,
                "IRI_MB3": iri_mb3_member,
            },
            "half_life_scale": {
                "scale_factor": half_life_scale,
                "interpretation": half_life_scale_note,
            },
            "summary": {
                "record_count": len(records),
                "resolved_without_missing_fields": len(records)
                - len(unresolved_records),
                "partial_record_count": len(unresolved_records),
                "opaque_binary_files_present": bool(lb1_member or lb2_member),
                "iri_text_files_present": bool(
                    iri_mb1_member or iri_mb2_member or iri_mb3_member
                ),
            },
            "fallback_counts": dict(sorted(fallback_counts.items())),
            "missing_field_counts": dict(sorted(missing_field_counts.items())),
            "unresolved_records": unresolved_records,
            "notes": [
                "This importer uses text-discoverable Kayzero sidecar files and transparent FluxForge fallbacks.",
                "Where present, companion IRI_MB1/IRI_MB2 text tables are used to surface target-isotope, abundance, sigma0/I0, and gamma-intensity data distributed with the Kayzero library bundle.",
                "IRI_MB1 decay-chain ancestry and IRI_MB3 line tables are used to promote daughter/isomer products onto their governing thermal-capture parent records where possible.",
                "A small supplementary decay-line fallback derived from bundled ENDF/B-VIII decay evaluations is used for a handful of residual primary gamma intensities not surfaced by the Kayzero text tables.",
                "LB1/LB2 remain opaque binary payloads; remaining gaps after IRI/text enrichment are reported explicitly.",
                "Target isotopes are currently inferred as same-element (n,gamma) parents with A-1 mass number.",
            ],
        }
        library = GovernedLibrary(
            library_id=f"fluxforge.k0.kayzero.{version}",
            version=version,
            scope=(
                "Kayzero-derived governed k0 library imported from user-supplied text sidecar files with "
                "explicit unresolved-field reporting."
            ),
            status="imported_partial" if unresolved_records else "imported_complete",
            provenance_note=(
                f"Imported from {source_path}. Current importer uses uk0/uQ0/uT12/MDcode/FCd text files and, when available, "
                "the companion IRI_MB1/IRI_MB2/IRI_MB3 text tables packaged alongside the Kayzero binary libraries, plus a "
                "small bundled ENDF/B-VIII decay-line supplement for residual primary gamma intensities. Opaque LB1/LB2 "
                "payloads are not decoded directly; see the companion import report for any remaining unresolved fields."
            ),
            records=records,
        )
        return KayzeroImportResult(library=library, report=report)
    finally:
        source.close()


def write_governed_library_json(path: str | Path, library: GovernedLibrary) -> None:
    Path(path).write_text(json.dumps(library.to_dict(), indent=2), encoding="utf-8")


def write_import_report_json(path: str | Path, report: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
