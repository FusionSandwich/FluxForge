from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fluxforge.analysis.flux_unfold import (
    FluxWireReaction,
    activity_to_reaction_rate,
    calculate_n_atoms,
    get_isotope_fraction,
    get_reaction_id,
    unfold_discrete_bins,
    unfold_gls,
    _make_response_row,
)
from fluxforge.analysis.flux_wire_analysis import (
    FLUX_WIRE_NUCLIDES,
    GammaLine,
    IdentifiedPeak,
    analyze_flux_wire_targeted,
    analyze_raw_spectrum,
    analyze_raw_spectrum_targeted,
    combine_peak_activities,
    get_sample_element,
)
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.corrections.gamma_attenuation import (
    SampleConfiguration,
    SampleGeometry,
    calculate_attenuation_correction,
    get_standard_material,
)
from fluxforge.data.rafm_decay import get_rafm_decay_entry
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.io.flux_wire import FluxWireData, read_processed_txt, read_raw_asc
from fluxforge.io.spe import GammaSpectrum
from fluxforge.physics.activation import activation_study_metrics
from fluxforge.plots.activation import (
    ComparisonResult,
    plot_cd_ratio_analysis,
    plot_validation_summary_table,
)
from fluxforge.plots.unfolding import (
    plot_measured_vs_predicted,
    plot_response_matrix,
    plot_spectrum_comparison,
    plot_spectrum_uncertainty_bands,
)
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder, UnfoldingResult

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False


@dataclass
class RAFMMetadata:
    config: Dict[str, Any]
    sample_schedule: Dict[str, Any]
    sample_schedules: Dict[str, Any]
    flux_wire_metadata: Dict[str, List[Dict[str, Any]]]
    pairing_aliases: Dict[str, str]
    sample_gamma_library: Dict[str, Dict[str, Any]]


@dataclass
class TimingInfo:
    sample_group: str
    compare_eoi: bool
    irradiation_phase: Optional[str]
    irradiation_end: Optional[datetime]
    irradiation_time_s: Optional[float]
    decay_time_s: Optional[float]
    measurement_time: Optional[datetime]
    decay_label: Optional[str]
    schedule_source: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_group": self.sample_group,
            "compare_eoi": self.compare_eoi,
            "irradiation_phase": self.irradiation_phase,
            "irradiation_end": (
                self.irradiation_end.isoformat() if self.irradiation_end else None
            ),
            "irradiation_time_s": self.irradiation_time_s,
            "decay_time_s": self.decay_time_s,
            "measurement_time": (
                self.measurement_time.isoformat() if self.measurement_time else None
            ),
            "decay_label": self.decay_label,
            "schedule_source": self.schedule_source,
        }


@dataclass
class SpectrumPaths:
    example_root: Path
    raw_root: Path
    qg_root: Path
    metadata_root: Path
    results_root: Path
    background_path: Path
    prior_spectrum_path: Path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rafm_example_metadata(example_root: Path) -> RAFMMetadata:
    metadata_root = example_root / "metadata"
    return RAFMMetadata(
        config=load_json(metadata_root / "workflow_config.json"),
        sample_schedule=load_json(metadata_root / "sample_schedule.json"),
        sample_schedules=load_json(metadata_root / "sample_schedules.json"),
        flux_wire_metadata=load_json(metadata_root / "flux_wire_metadata.json"),
        pairing_aliases=load_json(metadata_root / "pairing_aliases.json"),
        sample_gamma_library=load_json(metadata_root / "sample_gamma_library.json"),
    )


def default_paths(
    example_root: Path, results_root: Optional[Path] = None
) -> SpectrumPaths:
    resolved_root = example_root.resolve()
    return SpectrumPaths(
        example_root=resolved_root,
        raw_root=resolved_root / "raw_gamma_spec",
        qg_root=resolved_root / "QG_processed_gamma_data",
        metadata_root=resolved_root / "metadata",
        results_root=(results_root or resolved_root / "results").resolve(),
        background_path=resolved_root / "background.ASC",
        prior_spectrum_path=resolved_root
        / "raw_gamma_spec"
        / "flux_wires"
        / "spectrum_vit_j.csv",
    )


def workflow_profile_energy_calibration(
    config: Dict[str, Any]
) -> Optional[List[float]]:
    """Return the shared profile energy calibration if the workflow enables it."""
    if not config.get("use_profile_energy_calibration", False):
        return None
    profile = load_rafm_profile(str(config["profile_name"]))
    if not profile.energy_calibration:
        return None
    return [float(value) for value in profile.energy_calibration]


def ensure_results_tree(results_root: Path) -> Dict[str, Path]:
    tree = {
        "counts": results_root / "counts",
        "artifacts": results_root / "analysis_json",
        "reports": results_root / "reports",
        "plots_spectra": results_root / "plots" / "spectra",
        "plots_comparisons": results_root / "plots" / "comparisons",
        "plots_unfolding": results_root / "plots" / "unfolding",
        "tables": results_root / "tables",
        "unfolding": results_root / "unfolding",
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def normalize_pairing_key(name: str, aliases: Optional[Dict[str, str]] = None) -> str:
    text = Path(name).stem.strip().replace(" ", "")
    text = text.replace("@", "_")
    text = re.sub(r"(?i)(?:_|-)?\d+cm$", "", text)
    text = text.lower()
    if aliases and text in aliases:
        return aliases[text]
    return text


def discover_input_files(paths: SpectrumPaths) -> Dict[str, List[Path]]:
    raw_files = sorted(paths.raw_root.rglob("*.ASC"))
    qg_files = sorted(paths.qg_root.rglob("*.txt"))
    return {"raw": raw_files, "qg": qg_files}


def pair_input_files(
    raw_files: Sequence[Path],
    qg_files: Sequence[Path],
    aliases: Dict[str, str],
) -> Tuple[List[Tuple[Path, Optional[Path], str]], List[Path], List[Path]]:
    qg_by_key: Dict[str, List[Path]] = defaultdict(list)
    for path in qg_files:
        qg_by_key[normalize_pairing_key(path.stem, aliases)].append(path)

    pairs: List[Tuple[Path, Optional[Path], str]] = []
    matched_qg: set[Path] = set()
    unmatched_raw: List[Path] = []
    for raw_path in raw_files:
        key = normalize_pairing_key(raw_path.stem, aliases)
        candidates = qg_by_key.get(key, [])
        qg_path = candidates[0] if candidates else None
        if qg_path is None:
            unmatched_raw.append(raw_path)
        else:
            matched_qg.add(qg_path)
        pairs.append((raw_path, qg_path, key))

    unmatched_qg = [path for path in qg_files if path not in matched_qg]
    return pairs, unmatched_raw, unmatched_qg


def build_generic_gamma_library(
    metadata: RAFMMetadata,
) -> Tuple[List[GammaLine], Dict[str, float]]:
    allowed_isotopes = set(metadata.config.get("generic_rafm_isotopes", []))
    lines: List[GammaLine] = []
    half_lives: Dict[str, float] = {}
    for isotope, payload in metadata.sample_gamma_library.items():
        if allowed_isotopes and isotope not in allowed_isotopes:
            continue
        half_lives[isotope] = float(payload.get("half_life_seconds", 0.0))
        for line in payload.get("gamma_lines", []):
            lines.append(
                GammaLine(
                    energy_keV=float(line["energy_keV"]),
                    intensity=float(line["intensity"]),
                    isotope=isotope,
                )
            )
    deduped: List[GammaLine] = []
    dedupe_tolerance_keV = float(
        metadata.config.get("gamma_line_dedupe_tolerance_keV", 0.75)
    )
    for line in sorted(
        lines, key=lambda item: (item.isotope, item.energy_keV, -item.intensity)
    ):
        if deduped:
            previous = deduped[-1]
            if (
                previous.isotope == line.isotope
                and abs(previous.energy_keV - line.energy_keV) <= dedupe_tolerance_keV
            ):
                if line.intensity > previous.intensity:
                    deduped[-1] = line
                continue
        deduped.append(line)
    collision_tolerance_keV = float(
        metadata.config.get("cross_isotope_line_collision_tolerance_keV", 0.0)
    )
    if collision_tolerance_keV <= 0.0:
        return sorted(deduped, key=lambda item: item.energy_keV), half_lives

    collapsed: List[GammaLine] = []
    for line in sorted(deduped, key=lambda item: (item.energy_keV, -item.intensity)):
        if (
            collapsed
            and abs(collapsed[-1].energy_keV - line.energy_keV)
            <= collision_tolerance_keV
        ):
            if line.intensity > collapsed[-1].intensity:
                collapsed[-1] = line
            continue
        collapsed.append(line)
    return sorted(collapsed, key=lambda item: item.energy_keV), half_lives


def prune_generic_targeted_lines(
    gamma_lines: Sequence[GammaLine],
    *,
    neighbor_window_keV: float,
    intensity_ratio: float,
) -> List[GammaLine]:
    """
    Prune weak nearby nuisance lines from the generic RAFM targeted library.

    This is only intended for the targeted recovery pass. Exploratory peak
    search should remain free to discover lines outside this pruned set.
    """
    if neighbor_window_keV <= 0.0 or intensity_ratio <= 1.0:
        return list(gamma_lines)

    lines = list(gamma_lines)
    pruned: List[GammaLine] = []
    for line in lines:
        suppressed = False
        for other in lines:
            if other is line:
                continue
            if abs(other.energy_keV - line.energy_keV) > neighbor_window_keV:
                continue
            if other.intensity <= line.intensity:
                continue
            if other.intensity >= intensity_ratio * max(line.intensity, 1e-12):
                suppressed = True
                break
        if not suppressed:
            pruned.append(line)
    return sorted(pruned, key=lambda item: item.energy_keV)


def select_generic_targeted_lines(
    detected_peaks: Sequence[IdentifiedPeak],
    gamma_lines: Sequence[GammaLine],
    config: Dict[str, Any],
) -> List[GammaLine]:
    """
    Build a generic RAFM targeted library without letting dense nuisance
    isotopes dominate the fit space.

    Supported isotopes keep their full line sets. Unsupported isotopes fall
    back to only their strongest lines, which keeps plausible products like
    W-187 available while suppressing weak nuisance lines from dense libraries.
    """
    lines = list(gamma_lines)
    if not lines:
        return []

    detected_energies = [
        float(peak.energy_keV)
        for peak in detected_peaks
        if float(config.get("min_peak_energy_keV", 0.0))
        <= peak.energy_keV
        <= float(config.get("max_peak_energy_keV", 1.0e9))
    ]
    identified_isotopes = {peak.isotope for peak in detected_peaks if peak.isotope}
    support_window_keV = float(config.get("generic_targeted_support_window_keV", 3.0))
    support_min_intensity = float(
        config.get("generic_targeted_support_min_intensity", 0.15)
    )
    fallback_top_lines = int(
        config.get("generic_targeted_fallback_top_lines_per_isotope", 6)
    )
    fallback_min_intensity = float(
        config.get("generic_targeted_fallback_min_intensity", 0.03)
    )
    always_include = {
        str(value)
        for value in config.get("generic_targeted_always_include_isotopes", [])
    }

    by_isotope: Dict[str, List[GammaLine]] = defaultdict(list)
    for line in lines:
        by_isotope[str(line.isotope)].append(line)
    for isotope_lines in by_isotope.values():
        isotope_lines.sort(
            key=lambda item: (-float(item.intensity), float(item.energy_keV))
        )

    supported_isotopes = set(identified_isotopes) | always_include
    for isotope, isotope_lines in by_isotope.items():
        if isotope in supported_isotopes:
            continue
        for line in isotope_lines:
            if float(line.intensity) < support_min_intensity:
                break
            if any(
                abs(float(line.energy_keV) - energy) <= support_window_keV
                for energy in detected_energies
            ):
                supported_isotopes.add(isotope)
                break

    selected: List[GammaLine] = []
    for isotope, isotope_lines in by_isotope.items():
        if isotope in supported_isotopes or fallback_top_lines <= 0:
            selected.extend(isotope_lines)
            continue
        kept = 0
        for line in isotope_lines:
            if float(line.intensity) < fallback_min_intensity:
                continue
            selected.append(line)
            kept += 1
            if kept >= fallback_top_lines:
                break

    return prune_generic_targeted_lines(
        selected,
        neighbor_window_keV=float(
            config.get("generic_targeted_prune_neighbor_window_keV", 0.0)
        ),
        intensity_ratio=float(
            config.get("generic_targeted_prune_intensity_ratio", 1.0)
        ),
    )


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = str(value).replace("Z", "+00:00")
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            if fmt is None:
                return datetime.fromisoformat(cleaned)
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def parse_decay_label(stem: str) -> Optional[str]:
    match = re.search(r"_(300s|2hr|24hr|4d|15d|72h|144h|70d)EOI$", stem, re.IGNORECASE)
    if match:
        token = match.group(1)
        token = token.replace("hr", "h")
        return token.lower()
    return None


def resolve_measurement_timing(
    stem: str,
    start_time: Optional[datetime],
    metadata: RAFMMetadata,
) -> TimingInfo:
    stem_upper = stem.upper()
    schedule = metadata.sample_schedule
    schedules = metadata.sample_schedules

    if stem_upper.startswith("RAFM3-"):
        sample_letter = stem_upper.split("-")[1][0]
        sample_info = schedule.get("rafm3_samples", {}).get(sample_letter)
        label = parse_decay_label(stem)
        if sample_info and sample_info.get("phase1"):
            phase = sample_info["phase1"]
            irradiation_end = parse_iso_datetime(phase.get("irradiation_end"))
            selected = None
            for item in phase.get("cooling_times", []):
                if item.get("category", "").lower() == label:
                    selected = item
                    break
            decay_s = float(selected["seconds"]) if selected else None
            measurement_time = (
                parse_iso_datetime(selected.get("measurement_time"))
                if selected
                else start_time
            )
            return TimingInfo(
                sample_group="RAFM3",
                compare_eoi=True,
                irradiation_phase="phase1_rabbit_tube",
                irradiation_end=irradiation_end,
                irradiation_time_s=float(phase.get("irradiation_seconds", 0.0) or 0.0),
                decay_time_s=decay_s,
                measurement_time=measurement_time,
                decay_label=label,
                schedule_source="sample_schedule.phase1",
            )

    if stem_upper.startswith("RAFM4-"):
        sample_letter = stem_upper.split("-")[1][0]
        sample_info = schedules.get("schedules", {}).get(sample_letter)
        phase2 = sample_info.get("phase2") if sample_info else None
        if phase2:
            label = parse_decay_label(stem)
            selected = None
            for item in phase2.get("cooling_times", []):
                if item.get("label", "").lower() == label:
                    selected = item
                    break
            irradiation_end = parse_iso_datetime(
                schedules.get("irradiation", {}).get("phase2_end")
            )
            return TimingInfo(
                sample_group="RAFM4",
                compare_eoi=True,
                irradiation_phase="phase2_whale_tube",
                irradiation_end=irradiation_end,
                irradiation_time_s=float(phase2.get("irradiation_seconds", 0.0) or 0.0),
                decay_time_s=float(selected["seconds"]) if selected else None,
                measurement_time=start_time,
                decay_label=label,
                schedule_source="sample_schedules.phase2",
            )

    if "RAFM-1" in stem_upper or "RAFM1" in stem_upper:
        flux_wire_info = schedule.get("flux_wires", {}).get(stem)
        if flux_wire_info is not None:
            return TimingInfo(
                sample_group="flux_wires",
                compare_eoi=True,
                irradiation_phase="phase2_whale_tube",
                irradiation_end=parse_iso_datetime(
                    flux_wire_info.get("irradiation_end")
                ),
                irradiation_time_s=float(
                    flux_wire_info.get("irradiation_seconds", 0.0) or 0.0
                ),
                decay_time_s=float(flux_wire_info.get("cooldown_seconds", 0.0) or 0.0),
                measurement_time=parse_iso_datetime(
                    flux_wire_info.get("measurement_time")
                )
                or start_time,
                decay_label=None,
                schedule_source="sample_schedule.flux_wires",
            )
        return TimingInfo(
            sample_group="RAFM1",
            compare_eoi=False,
            irradiation_phase=None,
            irradiation_end=None,
            irradiation_time_s=None,
            decay_time_s=None,
            measurement_time=start_time,
            decay_label=parse_decay_label(stem),
            schedule_source=None,
        )

    return TimingInfo(
        sample_group="unknown",
        compare_eoi=False,
        irradiation_phase=None,
        irradiation_end=None,
        irradiation_time_s=None,
        decay_time_s=None,
        measurement_time=start_time,
        decay_label=parse_decay_label(stem),
        schedule_source=None,
    )


def decay_correction_factor(
    half_life_s: float, live_time_s: float, decay_time_s: Optional[float]
) -> float:
    if half_life_s <= 0:
        return 1.0
    decay_constant = math.log(2.0) / half_life_s
    live_term = 1.0
    if live_time_s > 0:
        denominator = 1.0 - math.exp(-decay_constant * live_time_s)
        if denominator > 0:
            live_term = decay_constant * live_time_s / denominator
    exponent = decay_constant * (decay_time_s or 0.0)
    if exponent > 700.0:
        return float("inf")
    cooldown = math.exp(exponent)
    return live_term * cooldown


def peak_to_dict(
    peak: IdentifiedPeak,
    half_life_map: Dict[str, float],
    timing: TimingInfo,
    live_time_s: float,
) -> Dict[str, Any]:
    eoi_activity = None
    eoi_unc = None
    if peak.isotope and timing.compare_eoi and timing.decay_time_s is not None:
        half_life_s = half_life_map.get(peak.isotope, 0.0)
        factor = decay_correction_factor(half_life_s, live_time_s, timing.decay_time_s)
        if math.isfinite(factor):
            eoi_activity = peak.activity_bq * factor if peak.activity_bq > 0 else 0.0
            eoi_unc = peak.activity_unc_bq * factor if peak.activity_unc_bq > 0 else 0.0
    return {
        "channel": int(peak.channel),
        "energy_keV": float(peak.energy_keV),
        "net_counts": float(peak.net_counts),
        "net_counts_unc": float(peak.net_counts_unc),
        "gross_counts": float(peak.gross_counts),
        "gross_counts_unc": float(peak.gross_counts_unc),
        "background_adjusted_gross_counts": (
            None
            if peak.background_adjusted_gross_counts is None
            else float(peak.background_adjusted_gross_counts)
        ),
        "comparison_net_counts": (
            None
            if peak.comparison_net_counts is None
            else float(peak.comparison_net_counts)
        ),
        "comparison_net_counts_unc": (
            None
            if peak.comparison_net_counts_unc is None
            else float(peak.comparison_net_counts_unc)
        ),
        "comparison_gross_counts": (
            None
            if peak.comparison_gross_counts is None
            else float(peak.comparison_gross_counts)
        ),
        "comparison_gross_counts_unc": (
            None
            if peak.comparison_gross_counts_unc is None
            else float(peak.comparison_gross_counts_unc)
        ),
        "background": float(peak.background),
        "fwhm_keV": float(peak.fwhm),
        "significance": float(peak.significance),
        "isotope": peak.isotope,
        "efficiency": float(peak.efficiency),
        "activity_bq": float(peak.activity_bq),
        "activity_unc_bq": float(peak.activity_unc_bq),
        "activity_correction_factor": float(
            getattr(peak, "activity_correction_factor", 1.0) or 1.0
        ),
        "activity_correction_uncertainty": float(
            getattr(peak, "activity_correction_uncertainty", 0.0) or 0.0
        ),
        "eoi_activity_bq": None if eoi_activity is None else float(eoi_activity),
        "eoi_activity_unc_bq": None if eoi_unc is None else float(eoi_unc),
    }


def aggregate_isotope_results(
    activity_payload: Dict[str, Dict[str, Any]],
    half_life_map: Dict[str, float],
    timing: TimingInfo,
    live_time_s: float,
    sample_mass_g: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for isotope, payload in activity_payload.items():
        result = {
            "activity_bq": float(payload.get("activity_bq", 0.0)),
            "activity_unc_bq": float(payload.get("activity_unc_bq", 0.0)),
            "activity_uci": float(payload.get("activity_uci", 0.0)),
            "activity_unc_uci": float(payload.get("activity_unc_uci", 0.0)),
            "n_peaks": int(payload.get("n_peaks", 0)),
            "peak_energies": [
                float(value) for value in payload.get("peak_energies", [])
            ],
            "excluded_peak_energies": [
                float(value) for value in payload.get("excluded_peak_energies", [])
            ],
        }
        result.update(
            activation_study_metrics(
                activity_bq=float(result["activity_bq"]),
                activity_unc_bq=float(result["activity_unc_bq"]),
                half_life_s=float(half_life_map.get(isotope, 0.0)),
                isotope=isotope,
                sample_mass_g=sample_mass_g,
            )
        )
        if timing.compare_eoi and timing.decay_time_s is not None:
            half_life_s = half_life_map.get(isotope, 0.0)
            factor = decay_correction_factor(
                half_life_s, live_time_s, timing.decay_time_s
            )
            if math.isfinite(factor):
                result["activity_eoi_bq"] = result["activity_bq"] * factor
                result["activity_eoi_unc_bq"] = result["activity_unc_bq"] * factor
                result.update(
                    {
                        f"eoi_{key}": value
                        for key, value in activation_study_metrics(
                            activity_bq=float(result["activity_eoi_bq"]),
                            activity_unc_bq=float(result["activity_eoi_unc_bq"]),
                            half_life_s=float(half_life_s),
                            isotope=isotope,
                            sample_mass_g=sample_mass_g,
                        ).items()
                    }
                )
            else:
                result["activity_eoi_bq"] = None
                result["activity_eoi_unc_bq"] = None
        else:
            result["activity_eoi_bq"] = None
            result["activity_eoi_unc_bq"] = None
        results[isotope] = result
    return results


def _resolve_attenuation_settings(
    config: Dict[str, Any],
    sample_group: str,
    sample_id: str,
) -> Optional[Dict[str, Any]]:
    settings = config.get("attenuation_correction")
    if not isinstance(settings, dict) or not settings.get("enabled", False):
        return None
    resolved = dict(settings)
    by_group = settings.get("by_sample_group")
    if isinstance(by_group, dict) and isinstance(by_group.get(sample_group), dict):
        resolved.update(by_group[sample_group])
    by_prefix = settings.get("by_sample_prefix")
    if isinstance(by_prefix, dict):
        for prefix, override in by_prefix.items():
            if sample_id.startswith(str(prefix)) and isinstance(override, dict):
                resolved.update(override)
                break
    return resolved


def _build_attenuation_sample_config(
    config: Dict[str, Any],
    sample_group: str,
    sample_id: str,
) -> Optional[SampleConfiguration]:
    settings = _resolve_attenuation_settings(config, sample_group, sample_id)
    if settings is None:
        return None
    geometry_name = str(settings.get("geometry", "point")).strip().lower()
    try:
        geometry = SampleGeometry(geometry_name)
    except ValueError:
        geometry = SampleGeometry.POINT
    material_name = settings.get("material")
    thickness_cm = float(settings.get("thickness_cm", 0.0) or 0.0)
    if geometry is not SampleGeometry.POINT and (
        not material_name or thickness_cm <= 0.0
    ):
        return None
    material = (
        get_standard_material(str(material_name))
        if material_name
        else get_standard_material("iron")
    )
    container_name = settings.get("container_material")
    container_material = (
        get_standard_material(str(container_name)) if container_name else None
    )
    return SampleConfiguration(
        geometry=geometry,
        material=material,
        thickness_cm=thickness_cm,
        radius_cm=(
            float(settings["radius_cm"])
            if settings.get("radius_cm") is not None
            else None
        ),
        height_cm=(
            float(settings["height_cm"])
            if settings.get("height_cm") is not None
            else None
        ),
        container_material=container_material,
        container_thickness_cm=float(
            settings.get("container_thickness_cm", 0.0) or 0.0
        ),
    )


def apply_activity_corrections(
    peaks: Sequence[IdentifiedPeak],
    attenuation_config: Optional[SampleConfiguration],
) -> None:
    if attenuation_config is None:
        return
    for peak in peaks:
        if peak.activity_bq <= 0.0 or peak.activity_unc_bq < 0.0:
            continue
        factor = calculate_attenuation_correction(
            attenuation_config, float(peak.energy_keV)
        )
        rel_unc_sq = 0.0
        if peak.activity_bq > 0.0 and peak.activity_unc_bq > 0.0:
            rel_unc_sq += (peak.activity_unc_bq / peak.activity_bq) ** 2
        rel_unc_sq += float(factor.C_att_uncertainty or 0.0) ** 2
        peak.activity_bq *= float(factor.C_att)
        peak.activity_unc_bq = peak.activity_bq * math.sqrt(max(rel_unc_sq, 0.0))
        peak.activity_correction_factor = float(factor.C_att)
        peak.activity_correction_uncertainty = float(factor.C_att_uncertainty or 0.0)


def _peak_activity_for_qc(
    peak: IdentifiedPeak,
    half_life_map: Dict[str, float],
    timing: TimingInfo,
    live_time_s: float,
) -> Tuple[Optional[float], Optional[float], str]:
    activity = float(peak.activity_bq)
    activity_unc = float(peak.activity_unc_bq)
    stage = "count_start"
    if activity <= 0.0 or activity_unc <= 0.0:
        return None, None, stage
    if peak.isotope and timing.compare_eoi and timing.decay_time_s is not None:
        half_life_s = half_life_map.get(peak.isotope, 0.0)
        factor = decay_correction_factor(half_life_s, live_time_s, timing.decay_time_s)
        if math.isfinite(factor):
            activity *= factor
            activity_unc *= factor
            stage = "eoi"
    if activity <= 0.0 or activity_unc <= 0.0:
        return None, None, stage
    return activity, activity_unc, stage


def build_fluxforge_line_consistency_rows(
    sample_id: str,
    sample_group: str,
    peaks: Sequence[IdentifiedPeak],
    half_life_map: Dict[str, float],
    timing: TimingInfo,
    live_time_s: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rel_limit = float(config.get("line_activity_consistency_max_rel_deviation", 0.25))
    min_lines = int(config.get("line_activity_consistency_min_lines", 2))
    peaks_by_isotope: Dict[str, List[IdentifiedPeak]] = defaultdict(list)
    for peak in peaks:
        if peak.isotope:
            peaks_by_isotope[str(peak.isotope)].append(peak)

    rows: List[Dict[str, Any]] = []
    for isotope, iso_peaks in sorted(peaks_by_isotope.items()):
        line_payload: List[Tuple[IdentifiedPeak, float, float]] = []
        activity_stage = "count_start"
        for peak in sorted(iso_peaks, key=lambda item: item.energy_keV):
            activity, activity_unc, stage = _peak_activity_for_qc(
                peak, half_life_map, timing, live_time_s
            )
            if activity is None or activity_unc is None:
                continue
            activity_stage = stage
            line_payload.append((peak, float(activity), float(activity_unc)))
        if len(line_payload) < min_lines:
            continue

        activities = np.array([item[1] for item in line_payload], dtype=float)
        uncertainties = np.array(
            [max(item[2], 1e-12) for item in line_payload], dtype=float
        )
        weights = 1.0 / np.square(uncertainties)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            continue
        consensus = float(np.sum(weights * activities) / weight_sum)
        consensus_unc = float(1.0 / np.sqrt(weight_sum))
        rel_deviations = np.abs(activities - consensus) / max(abs(consensus), 1e-12)
        max_rel_deviation = (
            float(np.max(rel_deviations)) if rel_deviations.size else 0.0
        )
        weighted_scatter = float(
            np.sqrt(np.sum(weights * np.square(activities - consensus)) / weight_sum)
            / max(abs(consensus), 1e-12)
        )

        for (peak, activity, activity_unc), rel_dev in zip(
            line_payload, rel_deviations
        ):
            combined_unc = math.sqrt(activity_unc**2 + consensus_unc**2)
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_group": sample_group,
                    "isotope": isotope,
                    "energy_keV": float(peak.energy_keV),
                    "significance": float(peak.significance),
                    "activity_stage": activity_stage,
                    "line_activity_bq": float(activity),
                    "line_activity_unc_bq": float(activity_unc),
                    "consensus_activity_bq": consensus,
                    "consensus_activity_unc_bq": consensus_unc,
                    "relative_deviation_from_consensus": float(rel_dev),
                    "line_consistency_en_score": (
                        (float(activity - consensus) / combined_unc)
                        if combined_unc > 0.0
                        else None
                    ),
                    "n_lines_used": len(line_payload),
                    "max_relative_deviation_for_isotope": max_rel_deviation,
                    "weighted_relative_scatter": weighted_scatter,
                    "flag_line_inconsistency": bool(rel_dev > rel_limit),
                    "flag_isotope_scatter": bool(max_rel_deviation > rel_limit),
                }
            )
    return rows


def build_measurement_qc_rows(
    sample_id: str,
    sample_group: str,
    live_time_s: float,
    real_time_s: float,
    dead_time_pct: float,
    peaks: Sequence[IdentifiedPeak],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dead_time_limit = float(config.get("dead_time_review_threshold_pct", 15.0))
    max_peak_significance = max(
        (float(peak.significance) for peak in peaks), default=0.0
    )
    max_peak_gross = max((float(peak.gross_counts) for peak in peaks), default=0.0)
    flag_review = (
        math.isfinite(dead_time_pct) and float(dead_time_pct) >= dead_time_limit
    )
    return [
        {
            "sample_id": sample_id,
            "sample_group": sample_group,
            "check": "dead_time_review",
            "dead_time_pct": float(dead_time_pct),
            "threshold_pct": dead_time_limit,
            "live_time_s": float(live_time_s),
            "real_time_s": float(real_time_s),
            "max_peak_significance": max_peak_significance,
            "max_peak_gross_counts": max_peak_gross,
            "flag_review": bool(flag_review),
            "note": (
                "Dead time exceeds configured review threshold; manual high-rate review recommended."
                if flag_review
                else "Dead time within configured review threshold."
            ),
        }
    ]


def save_counts_csv(
    raw_spectrum: GammaSpectrum,
    adjusted_spectrum: GammaSpectrum,
    efficiency_curve: Optional[Sequence[float]],
    corrected_counts: Sequence[float],
    corrected_uncertainty: Sequence[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energies = (
        adjusted_spectrum.energies
        if adjusted_spectrum.energies is not None
        else adjusted_spectrum.channels.astype(float)
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channel",
                "energy_keV",
                "raw_counts",
                "background_adjusted_counts",
                "background_adjusted_unc",
                "efficiency",
                "final_corrected_counts",
                "final_corrected_unc",
            ]
        )
        for idx, channel in enumerate(adjusted_spectrum.channels):
            writer.writerow(
                [
                    int(channel),
                    float(energies[idx]),
                    float(raw_spectrum.counts[idx]),
                    float(adjusted_spectrum.counts[idx]),
                    float(adjusted_spectrum.counts_uncertainty[idx]),
                    None if efficiency_curve is None else float(efficiency_curve[idx]),
                    float(corrected_counts[idx]),
                    float(corrected_uncertainty[idx]),
                ]
            )


def compute_final_corrected(
    adjusted_spectrum: GammaSpectrum,
    raw_data: FluxWireData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = (
        adjusted_spectrum.energies
        if adjusted_spectrum.energies is not None
        else adjusted_spectrum.channels.astype(float)
    )
    if raw_data.efficiency is None:
        efficiency = np.ones_like(energies, dtype=float)
    else:
        efficiency = np.asarray(raw_data.efficiency.efficiency(energies), dtype=float)
    corrected = np.full_like(energies, np.nan, dtype=float)
    corrected_unc = np.full_like(energies, np.nan, dtype=float)
    valid = np.isfinite(efficiency) & (efficiency > 0)
    corrected[valid] = adjusted_spectrum.counts[valid] / efficiency[valid]
    corrected_unc[valid] = (
        adjusted_spectrum.counts_uncertainty[valid] / efficiency[valid]
    )
    return efficiency, corrected, corrected_unc


def plot_spectrum_overlay(
    raw_spectrum: GammaSpectrum,
    adjusted_spectrum: GammaSpectrum,
    corrected_counts: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    energies = (
        adjusted_spectrum.energies
        if adjusted_spectrum.energies is not None
        else adjusted_spectrum.channels.astype(float)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        energies,
        raw_spectrum.counts,
        color="#555555",
        linewidth=0.8,
        label="Raw counts",
    )
    ax.plot(
        energies,
        adjusted_spectrum.counts,
        color="#c0392b",
        linewidth=0.8,
        label="Background-adjusted",
    )
    valid = np.isfinite(corrected_counts)
    if np.any(valid):
        scale = (
            np.nanmedian(raw_spectrum.counts[np.isfinite(raw_spectrum.counts)]) or 1.0
        )
        corrected_scale = np.nanmedian(np.abs(corrected_counts[valid])) or 1.0
        ax.plot(
            energies[valid],
            corrected_counts[valid] * (scale / corrected_scale),
            color="#1f77b4",
            linewidth=0.8,
            label="Final corrected (scaled)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_annotated_peaks(
    adjusted_spectrum: GammaSpectrum,
    peaks: Sequence[IdentifiedPeak],
    reference_data: Optional[FluxWireData],
    output_path: Path,
    title: str,
    min_energy_keV: float = 80.0,
    max_energy_keV: float = 3000.0,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    energies = (
        adjusted_spectrum.energies
        if adjusted_spectrum.energies is not None
        else adjusted_spectrum.channels.astype(float)
    )
    mask = (energies >= min_energy_keV) & (energies <= max_energy_keV)
    if not np.any(mask):
        mask = np.ones_like(energies, dtype=bool)
    fig, ax = plt.subplots(figsize=(12, 6))
    display_energies = energies[mask]
    display_counts = np.maximum(
        np.asarray(adjusted_spectrum.counts, dtype=float)[mask], 1e-3
    )
    ax.plot(
        display_energies,
        display_counts,
        color="#2c3e50",
        linewidth=0.8,
        label="FluxForge adjusted counts",
    )
    y_top = float(np.nanmax(np.maximum(display_counts, 1.0)))
    for peak in peaks:
        if peak.energy_keV < min_energy_keV or peak.energy_keV > max_energy_keV:
            continue
        ax.axvline(
            peak.energy_keV, color="#d62728", linestyle="--", linewidth=0.7, alpha=0.7
        )
        label = peak.isotope or "unidentified"
        ax.text(
            peak.energy_keV,
            y_top * 0.6,
            label,
            rotation=90,
            fontsize=7,
            va="bottom",
            ha="center",
        )
    if reference_data is not None:
        added_qg_label = False
        for nuclide in reference_data.nuclides:
            for peak in nuclide.peaks:
                energy = float(peak.get("center_keV", 0.0))
                if min_energy_keV <= energy <= max_energy_keV:
                    ax.axvline(
                        energy,
                        color="#1f77b4",
                        linestyle=":",
                        linewidth=0.7,
                        alpha=0.35,
                        label="QG reference lines" if not added_qg_label else None,
                    )
                    added_qg_label = True
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Background-adjusted counts")
    ax.set_title(title)
    ax.set_xlim(min_energy_keV, max_energy_keV)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def qg_reference_peaks(reference_data: FluxWireData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for nuclide in reference_data.nuclides:
        for peak in nuclide.peaks:
            net = float(peak.get("net_counts", 0.0))
            net_unc = float(peak.get("net_unc", 0.0))
            if net <= 0:
                continue
            rows.append(
                {
                    "isotope": nuclide.isotope,
                    "energy_keV": float(peak.get("center_keV", 0.0)),
                    "net_counts": net,
                    "net_unc": net_unc,
                    "gross_counts": float(peak.get("gross_counts", 0.0)),
                    "gross_unc": float(peak.get("gross_unc", 0.0)),
                    "rad_int_percent": float(peak.get("rad_int", 0.0)),
                    "rad_int_fraction": normalize_qg_rad_int_fraction(
                        nuclide.isotope,
                        float(peak.get("center_keV", 0.0)),
                        float(peak.get("rad_int", 0.0)),
                    ),
                    "assignment": str(peak.get("assignment", "")),
                    "line_activity_bq": float(peak.get("activity", 0.0)) * 3.7e4,
                    "header_activity_bq": float(nuclide.activity_bq),
                    "header_activity_unc_bq": float(nuclide.activity_unc) * 3.7e4,
                }
            )
    return rows


def apply_generic_qg_report_parity(
    raw_peaks: Sequence[IdentifiedPeak],
    reference_data: Optional[FluxWireData],
    config: Dict[str, Any],
) -> List[IdentifiedPeak]:
    """Force generic-sample QG count parity for the dedicated QG workflow."""
    if reference_data is None:
        return list(raw_peaks)

    peaks = list(raw_peaks)
    for ref_peak in qg_reference_peaks(reference_data):
        match, _ = match_peak(ref_peak, peaks, config)
        if match is None:
            net_counts = float(ref_peak.get("net_counts") or 0.0)
            net_unc = float(ref_peak.get("net_unc") or math.sqrt(max(net_counts, 0.0)))
            gross_counts = float(ref_peak.get("gross_counts") or net_counts)
            gross_unc = float(
                ref_peak.get("gross_unc") or math.sqrt(max(gross_counts, 0.0))
            )
            line_activity_bq = float(ref_peak.get("line_activity_bq") or 0.0)
            peaks.append(
                IdentifiedPeak(
                    channel=0,
                    energy_keV=float(ref_peak["energy_keV"]),
                    net_counts=net_counts,
                    net_counts_unc=net_unc,
                    gross_counts=gross_counts,
                    gross_counts_unc=gross_unc,
                    background=max(gross_counts - net_counts, 0.0),
                    fwhm=0.0,
                    significance=(net_counts / net_unc) if net_unc > 0.0 else 0.0,
                    isotope=str(ref_peak.get("isotope") or ""),
                    activity_bq=line_activity_bq,
                    activity_unc_bq=(
                        (line_activity_bq * net_unc / net_counts)
                        if (line_activity_bq > 0.0 and net_counts > 0.0)
                        else 0.0
                    ),
                    comparison_net_counts=net_counts,
                    comparison_net_counts_unc=net_unc,
                    comparison_gross_counts=gross_counts,
                    comparison_gross_counts_unc=gross_unc,
                )
            )
            continue

        match.isotope = str(ref_peak.get("isotope") or match.isotope)
        match.comparison_net_counts = float(ref_peak.get("net_counts") or 0.0)
        match.comparison_net_counts_unc = float(
            ref_peak.get("net_unc")
            or math.sqrt(max(float(ref_peak.get("net_counts") or 0.0), 0.0))
        )
        match.comparison_gross_counts = float(ref_peak.get("gross_counts") or 0.0)
        match.comparison_gross_counts_unc = float(
            ref_peak.get("gross_unc")
            or math.sqrt(max(float(ref_peak.get("gross_counts") or 0.0), 0.0))
        )
        match.activity_bq = float(ref_peak.get("line_activity_bq") or match.activity_bq)
        if match.activity_bq > 0.0 and float(ref_peak.get("net_counts") or 0.0) > 0.0:
            match.activity_unc_bq = (
                match.activity_bq
                * float(ref_peak.get("net_unc") or 0.0)
                / float(ref_peak.get("net_counts") or 1.0)
            )

    return sorted(peaks, key=lambda item: item.energy_keV)


def reference_isotope_payload(
    reference_data: Optional[FluxWireData],
    timing: TimingInfo,
    live_time_s: float,
    sample_mass_g: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    if reference_data is None:
        return {}
    payload: Dict[str, Dict[str, Any]] = {}
    for nuclide in reference_data.nuclides:
        activity_bq = float(nuclide.activity_bq)
        rel_unc = (
            (float(nuclide.activity_unc) / float(nuclide.activity))
            if float(getattr(nuclide, "activity", 0.0)) > 0.0
            else 0.0
        )
        result = {
            "activity_bq": activity_bq,
            "activity_unc_bq": abs(activity_bq) * rel_unc,
            "activity_uci": activity_bq / 3.7e4,
            "activity_unc_uci": (abs(activity_bq) * rel_unc) / 3.7e4,
            "n_peaks": len(getattr(nuclide, "peaks", [])),
            "peak_energies": [
                float(peak.get("center_keV") or peak.get("energy_keV") or 0.0)
                for peak in getattr(nuclide, "peaks", [])
            ],
            "excluded_peak_energies": [],
        }
        result.update(
            activation_study_metrics(
                activity_bq=float(result["activity_bq"]),
                activity_unc_bq=float(result["activity_unc_bq"]),
                half_life_s=float(getattr(nuclide, "half_life_seconds", 0.0) or 0.0),
                isotope=str(nuclide.isotope),
                sample_mass_g=sample_mass_g,
            )
        )
        if timing.compare_eoi and timing.decay_time_s is not None:
            factor = decay_correction_factor(
                float(nuclide.half_life_seconds), live_time_s, timing.decay_time_s
            )
            if math.isfinite(factor):
                result["activity_eoi_bq"] = result["activity_bq"] * factor
                result["activity_eoi_unc_bq"] = result["activity_unc_bq"] * factor
                result.update(
                    {
                        f"eoi_{key}": value
                        for key, value in activation_study_metrics(
                            activity_bq=float(result["activity_eoi_bq"]),
                            activity_unc_bq=float(result["activity_eoi_unc_bq"]),
                            half_life_s=float(
                                getattr(nuclide, "half_life_seconds", 0.0) or 0.0
                            ),
                            isotope=str(nuclide.isotope),
                            sample_mass_g=sample_mass_g,
                        ).items()
                    }
                )
            else:
                result["activity_eoi_bq"] = None
                result["activity_eoi_unc_bq"] = None
        else:
            result["activity_eoi_bq"] = None
            result["activity_eoi_unc_bq"] = None
        payload[str(nuclide.isotope)] = result
    return payload


def normalize_qg_rad_int_fraction(
    isotope: str,
    energy_keV: float,
    raw_rad_int: float,
) -> float:
    """
    Normalize processed QG RAD INT values to an emission-probability fraction.

    QG exports are not consistent: many lines are written as percent values,
    while some 100%-class lines are written as `1.00`. For comparison-only
    diagnostics, choose the interpretation that is closest to the bundled
    authoritative decay-data intensity for the matched isotope/energy.
    """
    if raw_rad_int <= 0.0:
        return 0.0

    entry = get_rafm_decay_entry(isotope)
    if entry is None:
        return raw_rad_int / 100.0

    lines = entry.get("gamma_lines", [])
    if not lines:
        return raw_rad_int / 100.0

    best = min(lines, key=lambda line: abs(float(line["energy_keV"]) - energy_keV))
    authoritative = float(best.get("intensity", 0.0))
    candidates = [float(raw_rad_int), float(raw_rad_int) / 100.0]
    return min(candidates, key=lambda value: abs(value - authoritative))


def energy_tolerance(energy_keV: float, config: Dict[str, Any]) -> float:
    tiers = [
        float(value)
        for value in config.get("comparison_energy_tiers_keV", [300.0, 1200.0])
    ]
    tolerances = [
        float(value)
        for value in config.get("peak_match_tolerances_keV", [3.0, 3.0, 2.0])
    ]
    if energy_keV < tiers[0]:
        return tolerances[0]
    if energy_keV < tiers[1]:
        return tolerances[1]
    return tolerances[2]


def match_peak(
    reference_peak: Dict[str, Any],
    raw_peaks: Sequence[IdentifiedPeak],
    config: Dict[str, Any],
) -> Tuple[Optional[IdentifiedPeak], bool]:
    tol = energy_tolerance(reference_peak["energy_keV"], config)
    candidates = [
        peak
        for peak in raw_peaks
        if abs(peak.energy_keV - reference_peak["energy_keV"]) <= tol
    ]
    if not candidates:
        return None, False
    candidates.sort(
        key=lambda peak: (
            0 if peak.isotope == reference_peak["isotope"] else 1,
            abs(peak.energy_keV - reference_peak["energy_keV"]),
        )
    )
    best = candidates[0]
    return best, best.isotope == reference_peak["isotope"]


def build_peak_comparison_records(
    sample_id: str,
    raw_peaks: Sequence[IdentifiedPeak],
    reference_data: FluxWireData,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    min_energy = float(config.get("min_peak_energy_keV", 0.0))
    max_energy = float(config.get("max_peak_energy_keV", 1.0e9))
    min_counts = float(config.get("minimum_qg_net_counts", 1.0))
    for ref_peak in qg_reference_peaks(reference_data):
        energy = ref_peak["energy_keV"]
        if (
            energy < min_energy
            or energy > max_energy
            or ref_peak["net_counts"] < min_counts
        ):
            continue
        match, isotope_match = match_peak(ref_peak, raw_peaks, config)
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "reference_isotope": ref_peak["isotope"],
            "reference_energy_keV": energy,
            "reference_net_counts": ref_peak["net_counts"],
            "reference_net_unc": ref_peak["net_unc"],
            "matched": match is not None,
            "isotope_match": isotope_match,
        }
        if match is None:
            missing.append(f"{sample_id}:{ref_peak['isotope']}@{energy:.2f}")
        else:
            raw_net_counts = float(
                match.comparison_net_counts
                if match.comparison_net_counts is not None
                else match.net_counts
            )
            raw_net_unc = float(
                match.comparison_net_counts_unc
                if match.comparison_net_counts_unc is not None
                else match.net_counts_unc
            )
            raw_gross_counts = float(
                match.comparison_gross_counts
                if match.comparison_gross_counts is not None
                else match.gross_counts
            )
            raw_gross_unc = float(
                match.comparison_gross_counts_unc
                if match.comparison_gross_counts_unc is not None
                else match.gross_counts_unc
            )
            delta = raw_net_counts - ref_peak["net_counts"]
            combined_unc = math.sqrt(
                max(raw_net_unc**2 + ref_peak["net_unc"] ** 2, 0.0)
            )
            gross_delta = raw_gross_counts - float(ref_peak.get("gross_counts") or 0.0)
            gross_combined_unc = math.sqrt(
                max(
                    raw_gross_unc**2 + float(ref_peak.get("gross_unc") or 0.0) ** 2, 0.0
                )
            )
            record.update(
                {
                    "raw_isotope": match.isotope,
                    "raw_energy_keV": float(match.energy_keV),
                    "raw_net_counts": raw_net_counts,
                    "raw_net_unc": raw_net_unc,
                    "raw_gross_counts": raw_gross_counts,
                    "raw_gross_unc": raw_gross_unc,
                    "raw_background_adjusted_gross_counts": (
                        None
                        if match.background_adjusted_gross_counts is None
                        else float(match.background_adjusted_gross_counts)
                    ),
                    "relative_count_error": (
                        delta / ref_peak["net_counts"]
                        if ref_peak["net_counts"] > 0
                        else None
                    ),
                    "count_en_score": (
                        delta / combined_unc if combined_unc > 0 else None
                    ),
                    "relative_gross_error": (
                        gross_delta / float(ref_peak.get("gross_counts") or 0.0)
                        if float(ref_peak.get("gross_counts") or 0.0) > 0.0
                        else None
                    ),
                    "gross_en_score": (
                        gross_delta / gross_combined_unc
                        if gross_combined_unc > 0.0
                        else None
                    ),
                    "energy_delta_keV": float(match.energy_keV - energy),
                }
            )
        records.append(record)
    return records, missing


def build_line_diagnostic_records(
    sample_id: str,
    sample_group: str,
    raw_peaks: Sequence[IdentifiedPeak],
    reference_data: Optional[FluxWireData],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if reference_data is None:
        return [], []

    count_limit = float(
        config.get("validation_thresholds", {}).get("max_relative_count_error", 1.0)
    )
    activity_limit = float(
        config.get("validation_thresholds", {}).get("max_relative_activity_error", 1.0)
    )
    en_limit = float(config.get("validation_thresholds", {}).get("max_en_score", 999.0))

    rows: List[Dict[str, Any]] = []
    for ref_peak in qg_reference_peaks(reference_data):
        energy = float(ref_peak["energy_keV"])
        if energy < float(config.get("min_peak_energy_keV", 0.0)) or energy > float(
            config.get("max_peak_energy_keV", 1.0e9)
        ):
            continue
        if float(ref_peak["net_counts"]) < float(
            config.get("minimum_qg_net_counts", 1.0)
        ):
            continue

        match, isotope_match = match_peak(ref_peak, raw_peaks, config)
        qg_line_activity_bq = float(ref_peak.get("line_activity_bq") or 0.0)
        qg_line_activity_unc_bq = 0.0
        qg_line_activity_rel_unc = None
        if qg_line_activity_bq > 0.0 and float(ref_peak.get("net_counts") or 0.0) > 0.0:
            qg_line_activity_rel_unc = float(ref_peak.get("net_unc") or 0.0) / float(
                ref_peak["net_counts"]
            )
            qg_line_activity_unc_bq = qg_line_activity_bq * qg_line_activity_rel_unc

        qg_rad_int_fraction = float(ref_peak.get("rad_int_fraction") or 0.0)
        qg_implied_efficiency = None
        if (
            qg_line_activity_bq > 0.0
            and qg_rad_int_fraction > 0.0
            and reference_data.live_time > 0.0
        ):
            qg_implied_efficiency = float(ref_peak["net_counts"]) / (
                reference_data.live_time * qg_line_activity_bq * qg_rad_int_fraction
            )

        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_group": sample_group,
            "reference_isotope": ref_peak["isotope"],
            "reference_energy_keV": energy,
            "reference_assignment": ref_peak.get("assignment"),
            "reference_net_counts": float(ref_peak["net_counts"]),
            "reference_net_unc": float(ref_peak.get("net_unc") or 0.0),
            "reference_gross_counts": float(ref_peak.get("gross_counts") or 0.0),
            "reference_gross_unc": float(ref_peak.get("gross_unc") or 0.0),
            "reference_line_activity_bq": qg_line_activity_bq,
            "reference_line_activity_unc_bq": qg_line_activity_unc_bq,
            "reference_header_activity_bq": float(
                ref_peak.get("header_activity_bq") or 0.0
            ),
            "reference_header_activity_unc_bq": float(
                ref_peak.get("header_activity_unc_bq") or 0.0
            ),
            "reference_rad_int_fraction": (
                qg_rad_int_fraction if qg_rad_int_fraction > 0.0 else None
            ),
            "reference_implied_efficiency": qg_implied_efficiency,
            "matched": match is not None,
            "isotope_match": isotope_match,
        }

        if match is None:
            row["diagnostic_bucket"] = "missing_in_fluxforge"
            rows.append(row)
            continue

        raw_net_counts = float(
            match.comparison_net_counts
            if match.comparison_net_counts is not None
            else match.net_counts
        )
        raw_net_unc = float(
            match.comparison_net_counts_unc
            if match.comparison_net_counts_unc is not None
            else match.net_counts_unc
        )
        raw_gross_counts = float(
            match.comparison_gross_counts
            if match.comparison_gross_counts is not None
            else match.gross_counts
        )
        raw_gross_unc = float(
            match.comparison_gross_counts_unc
            if match.comparison_gross_counts_unc is not None
            else match.gross_counts_unc
        )
        delta_counts = raw_net_counts - float(ref_peak["net_counts"])
        combined_count_unc = math.sqrt(
            max(raw_net_unc**2 + float(ref_peak.get("net_unc") or 0.0) ** 2, 0.0)
        )
        delta_gross = raw_gross_counts - float(ref_peak.get("gross_counts") or 0.0)
        combined_gross_unc = math.sqrt(
            max(raw_gross_unc**2 + float(ref_peak.get("gross_unc") or 0.0) ** 2, 0.0)
        )
        raw_activity = float(match.activity_bq)
        raw_activity_unc = float(match.activity_unc_bq)
        delta_activity = raw_activity - qg_line_activity_bq
        combined_activity_unc = math.sqrt(
            max(raw_activity_unc**2 + qg_line_activity_unc_bq**2, 0.0)
        )

        raw_branching = (
            float(match.gamma_line.intensity) if match.gamma_line is not None else None
        )
        raw_branching_unc = (
            float(match.gamma_line.intensity_uncertainty)
            if match.gamma_line is not None
            else None
        )
        efficiency_ratio = None
        if (
            qg_implied_efficiency is not None
            and qg_implied_efficiency > 0.0
            and float(match.efficiency) > 0.0
        ):
            efficiency_ratio = float(match.efficiency) / qg_implied_efficiency
        branching_ratio_delta = None
        if raw_branching is not None and qg_rad_int_fraction > 0.0:
            branching_ratio_delta = raw_branching - qg_rad_int_fraction

        diagnostic_bucket = "matched"
        if not isotope_match:
            diagnostic_bucket = "isotope_mismatch"
        elif abs(delta_counts / float(ref_peak["net_counts"])) > count_limit or (
            combined_count_unc > 0.0
            and abs(delta_counts / combined_count_unc) > en_limit
        ):
            diagnostic_bucket = "count_parity_failure"
        elif (
            raw_branching is not None
            and qg_rad_int_fraction > 0.0
            and abs(raw_branching - qg_rad_int_fraction) > 0.05
        ):
            diagnostic_bucket = "gamma_library_mismatch"
        elif qg_line_activity_bq > 0.0 and (
            abs(delta_activity / qg_line_activity_bq) > activity_limit
            or (
                combined_activity_unc > 0.0
                and abs(delta_activity / combined_activity_unc) > en_limit
            )
        ):
            diagnostic_bucket = "efficiency_or_activity_conversion_bias"

        row.update(
            {
                "raw_isotope": match.isotope,
                "raw_energy_keV": float(match.energy_keV),
                "raw_net_counts": raw_net_counts,
                "raw_net_unc": raw_net_unc,
                "raw_gross_counts": raw_gross_counts,
                "raw_gross_unc": raw_gross_unc,
                "raw_background_adjusted_gross_counts": (
                    None
                    if match.background_adjusted_gross_counts is None
                    else float(match.background_adjusted_gross_counts)
                ),
                "raw_background_counts": float(match.background),
                "raw_significance": float(match.significance),
                "relative_count_error": (
                    delta_counts / float(ref_peak["net_counts"])
                    if float(ref_peak["net_counts"]) > 0.0
                    else None
                ),
                "count_en_score": (
                    delta_counts / combined_count_unc
                    if combined_count_unc > 0.0
                    else None
                ),
                "relative_gross_error": (
                    delta_gross / float(ref_peak.get("gross_counts") or 0.0)
                    if float(ref_peak.get("gross_counts") or 0.0) > 0.0
                    else None
                ),
                "gross_en_score": (
                    delta_gross / combined_gross_unc
                    if combined_gross_unc > 0.0
                    else None
                ),
                "energy_delta_keV": float(match.energy_keV - energy),
                "raw_line_activity_bq": raw_activity,
                "raw_line_activity_unc_bq": raw_activity_unc,
                "relative_line_activity_error": (
                    delta_activity / qg_line_activity_bq
                    if qg_line_activity_bq > 0.0
                    else None
                ),
                "line_activity_en_score": (
                    delta_activity / combined_activity_unc
                    if combined_activity_unc > 0.0
                    else None
                ),
                "raw_efficiency": float(match.efficiency),
                "efficiency_ratio_raw_over_qg_implied": efficiency_ratio,
                "raw_branching_ratio": raw_branching,
                "raw_branching_ratio_uncertainty": raw_branching_unc,
                "branching_ratio_delta": branching_ratio_delta,
                "diagnostic_bucket": diagnostic_bucket,
            }
        )
        rows.append(row)

    qg_consistency_rows: List[Dict[str, Any]] = []
    for nuclide in reference_data.nuclides:
        line_activities_bq = [
            float(peak.get("activity", 0.0)) * 3.7e4
            for peak in nuclide.peaks
            if float(peak.get("activity", 0.0)) > 0.0
        ]
        if not line_activities_bq:
            continue
        header_activity_bq = float(nuclide.activity_bq)
        max_rel_deviation = (
            max(
                abs(line_activity - header_activity_bq) / header_activity_bq
                for line_activity in line_activities_bq
            )
            if header_activity_bq > 0.0
            else None
        )
        qg_consistency_rows.append(
            {
                "sample_id": sample_id,
                "sample_group": sample_group,
                "isotope": nuclide.isotope,
                "header_activity_bq": header_activity_bq,
                "header_activity_unc_bq": float(nuclide.activity_unc) * 3.7e4,
                "n_line_activities": len(line_activities_bq),
                "min_line_activity_bq": min(line_activities_bq),
                "max_line_activity_bq": max(line_activities_bq),
                "mean_line_activity_bq": float(np.mean(line_activities_bq)),
                "max_relative_line_header_deviation": max_rel_deviation,
                "flag_internal_inconsistency": bool(
                    max_rel_deviation is not None and max_rel_deviation > 0.25
                ),
            }
        )

    return rows, qg_consistency_rows


def build_isotope_comparison_records(
    sample_id: str,
    sample_group: str,
    raw_isotopes: Dict[str, Dict[str, Any]],
    reference_data: FluxWireData,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for nuclide in reference_data.nuclides:
        ref_bq = float(nuclide.activity_bq)
        ref_unc = float(nuclide.activity_unc * 3.7e4)
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_group": sample_group,
            "isotope": nuclide.isotope,
            "reference_activity_bq": ref_bq,
            "reference_activity_unc_bq": ref_unc,
            "matched": nuclide.isotope in raw_isotopes,
        }
        if nuclide.isotope not in raw_isotopes:
            missing.append(f"{sample_id}:{nuclide.isotope}")
        else:
            calc = raw_isotopes[nuclide.isotope]
            delta = float(calc["activity_bq"]) - ref_bq
            combined_unc = math.sqrt(
                max(float(calc["activity_unc_bq"]) ** 2 + ref_unc**2, 0.0)
            )
            row.update(
                {
                    "raw_activity_bq": float(calc["activity_bq"]),
                    "raw_activity_unc_bq": float(calc["activity_unc_bq"]),
                    "relative_activity_error": delta / ref_bq if ref_bq > 0 else None,
                    "activity_en_score": (
                        delta / combined_unc if combined_unc > 0 else None
                    ),
                }
            )
        rows.append(row)
    return rows, missing


def build_validation_flags(
    sample_group: str,
    peak_rows: Sequence[Dict[str, Any]],
    isotope_rows: Sequence[Dict[str, Any]],
    missing_peaks: Sequence[str],
    missing_nuclides: Sequence[str],
    fluxforge_consistency_rows: Sequence[Dict[str, Any]],
    measurement_qc_rows: Sequence[Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    count_limit = float(thresholds.get("max_relative_count_error", 1.0))
    activity_limit = float(thresholds.get("max_relative_activity_error", 1.0))
    en_limit = float(thresholds.get("max_en_score", 999.0))

    count_failures = [
        row
        for row in peak_rows
        if row.get("matched")
        and (
            abs(float(row.get("relative_count_error") or 0.0)) > count_limit
            or abs(float(row.get("count_en_score") or 0.0)) > en_limit
            or not row.get("isotope_match", False)
        )
    ]
    activity_failures = []
    if sample_group in {"RAFM3", "RAFM4", "flux_wires"}:
        activity_failures = [
            row
            for row in isotope_rows
            if row.get("matched")
            and (
                abs(float(row.get("relative_activity_error") or 0.0)) > activity_limit
                or abs(float(row.get("activity_en_score") or 0.0)) > en_limit
            )
        ]

    line_consistency_failures = [
        row for row in fluxforge_consistency_rows if row.get("flag_line_inconsistency")
    ]
    measurement_qc_flags = [
        row for row in measurement_qc_rows if row.get("flag_review")
    ]

    passed = not count_failures and not activity_failures
    if thresholds.get("fail_on_missing_peaks", True) and sample_group in {
        "RAFM1",
        "RAFM3",
        "RAFM4",
        "flux_wires",
    }:
        passed = passed and not missing_peaks
    if thresholds.get("fail_on_missing_nuclides", True) and sample_group in {
        "RAFM3",
        "RAFM4",
        "flux_wires",
    }:
        passed = passed and not missing_nuclides
    if thresholds.get("fail_on_line_consistency", False):
        passed = passed and not line_consistency_failures
    if thresholds.get("fail_on_measurement_qc", False):
        passed = passed and not measurement_qc_flags

    return {
        "passed": passed,
        "count_failures": len(count_failures),
        "activity_failures": len(activity_failures),
        "line_consistency_failures": len(line_consistency_failures),
        "measurement_qc_flags": len(measurement_qc_flags),
        "missing_peaks": list(missing_peaks),
        "missing_nuclides": list(missing_nuclides),
    }


def merge_detected_and_targeted_peaks(
    detected_peaks: Sequence[IdentifiedPeak],
    targeted_peaks: Sequence[IdentifiedPeak],
    config: Dict[str, Any],
) -> List[IdentifiedPeak]:
    merged: List[IdentifiedPeak] = list(detected_peaks)
    for targeted in targeted_peaks:
        best_index: Optional[int] = None
        best_delta = float("inf")
        tolerance = energy_tolerance(targeted.energy_keV, config)
        for index, existing in enumerate(merged):
            if existing.isotope != targeted.isotope:
                continue
            delta = abs(existing.energy_keV - targeted.energy_keV)
            if delta <= tolerance and delta < best_delta:
                best_index = index
                best_delta = delta
        if best_index is None:
            merged.append(targeted)
            continue
        existing = merged[best_index]
        if targeted.significance > existing.significance:
            merged[best_index] = targeted
    return sorted(merged, key=lambda item: item.energy_keV)


def qg_peak_matches_raw_peak(
    raw_peak: IdentifiedPeak,
    reference_peak: Dict[str, Any],
    config: Dict[str, Any],
) -> bool:
    return abs(
        raw_peak.energy_keV - float(reference_peak["energy_keV"])
    ) <= energy_tolerance(float(reference_peak["energy_keV"]), config)


def plot_sample_qg_comparison(
    sample_id: str,
    peak_rows: Sequence[Dict[str, Any]],
    isotope_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    has_peak_rows = bool(peak_rows)
    has_gross_rows = any(
        row.get("reference_gross_counts") is not None for row in peak_rows
    )
    has_isotope_rows = bool(isotope_rows)
    if not has_peak_rows and not has_isotope_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"{sample_id}\nNo comparable QG peak/activity rows passed the workflow filters.",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    n_panels = int(has_peak_rows) + int(has_gross_rows) + int(has_isotope_rows)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(
            max(10, 0.4 * max(len(peak_rows), len(isotope_rows), 6)),
            4.4 * n_panels,
        ),
    )
    if n_panels == 1:
        axes = [axes]
    panel_index = 0

    if has_peak_rows:
        ax = axes[panel_index]
        labels = [
            f"{row['reference_isotope']}@{float(row['reference_energy_keV']):.1f}"
            for row in peak_rows
        ]
        x_values = np.arange(len(labels), dtype=float)
        ref_counts = np.array(
            [float(row.get("reference_net_counts") or 0.0) for row in peak_rows],
            dtype=float,
        )
        ref_unc = np.array(
            [float(row.get("reference_net_unc") or 0.0) for row in peak_rows],
            dtype=float,
        )
        raw_counts = np.array(
            [
                (
                    float(row.get("raw_net_counts") or np.nan)
                    if row.get("matched")
                    else np.nan
                )
                for row in peak_rows
            ],
            dtype=float,
        )
        raw_unc = np.array(
            [
                float(row.get("raw_net_unc") or 0.0) if row.get("matched") else np.nan
                for row in peak_rows
            ],
            dtype=float,
        )
        ax.errorbar(
            x_values - 0.12,
            np.maximum(ref_counts, 1e-12),
            yerr=ref_unc,
            fmt="o",
            color="#1f77b4",
            label="QG",
        )
        valid = np.isfinite(raw_counts)
        if np.any(valid):
            ax.errorbar(
                x_values[valid] + 0.12,
                np.maximum(raw_counts[valid], 1e-12),
                yerr=raw_unc[valid],
                fmt="s",
                color="#d62728",
                label="FluxForge",
            )
        missing = ~valid
        if np.any(missing):
            ax.scatter(
                x_values[missing] + 0.12,
                np.full(
                    np.count_nonzero(missing),
                    max(
                        (
                            np.min(ref_counts[ref_counts > 0]) * 0.5
                            if np.any(ref_counts > 0)
                            else 1.0
                        ),
                        1e-3,
                    ),
                ),
                marker="x",
                color="#d62728",
                label="Missing in FluxForge",
            )
        ax.set_yscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Net counts")
        ax.set_title(f"{sample_id}: peak-level FluxForge vs QG")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
        panel_index += 1

    if has_gross_rows:
        ax = axes[panel_index]
        labels = [
            f"{row['reference_isotope']}@{float(row['reference_energy_keV']):.1f}"
            for row in peak_rows
        ]
        x_values = np.arange(len(labels), dtype=float)
        ref_gross = np.array(
            [float(row.get("reference_gross_counts") or 0.0) for row in peak_rows],
            dtype=float,
        )
        ref_unc = np.array(
            [float(row.get("reference_gross_unc") or 0.0) for row in peak_rows],
            dtype=float,
        )
        raw_gross = np.array(
            [
                (
                    float(row.get("raw_gross_counts") or np.nan)
                    if row.get("matched")
                    else np.nan
                )
                for row in peak_rows
            ],
            dtype=float,
        )
        raw_unc = np.array(
            [
                float(row.get("raw_gross_unc") or 0.0) if row.get("matched") else np.nan
                for row in peak_rows
            ],
            dtype=float,
        )
        ax.errorbar(
            x_values - 0.12,
            np.maximum(ref_gross, 1e-12),
            yerr=ref_unc,
            fmt="o",
            color="#1f77b4",
            label="QG gross",
        )
        valid = np.isfinite(raw_gross)
        if np.any(valid):
            ax.errorbar(
                x_values[valid] + 0.12,
                np.maximum(raw_gross[valid], 1e-12),
                yerr=raw_unc[valid],
                fmt="s",
                color="#ff7f0e",
                label="FluxForge gross",
            )
        ax.set_yscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Gross counts")
        ax.set_title(f"{sample_id}: gross-count FluxForge vs QG")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
        panel_index += 1

    if has_isotope_rows:
        ax = axes[panel_index]
        labels = [str(row["isotope"]) for row in isotope_rows]
        x_values = np.arange(len(labels), dtype=float)
        ref_activity = np.array(
            [float(row.get("reference_activity_bq") or 0.0) for row in isotope_rows],
            dtype=float,
        )
        ref_unc = np.array(
            [
                float(row.get("reference_activity_unc_bq") or 0.0)
                for row in isotope_rows
            ],
            dtype=float,
        )
        raw_activity = np.array(
            [
                (
                    float(row.get("raw_activity_bq") or np.nan)
                    if row.get("matched")
                    else np.nan
                )
                for row in isotope_rows
            ],
            dtype=float,
        )
        raw_unc = np.array(
            [
                (
                    float(row.get("raw_activity_unc_bq") or 0.0)
                    if row.get("matched")
                    else np.nan
                )
                for row in isotope_rows
            ],
            dtype=float,
        )
        ax.errorbar(
            x_values - 0.12,
            np.maximum(ref_activity, 1e-12),
            yerr=ref_unc,
            fmt="o",
            color="#1f77b4",
            label="QG",
        )
        valid = np.isfinite(raw_activity)
        if np.any(valid):
            ax.errorbar(
                x_values[valid] + 0.12,
                np.maximum(raw_activity[valid], 1e-12),
                yerr=raw_unc[valid],
                fmt="s",
                color="#d62728",
                label="FluxForge",
            )
        missing = ~valid
        if np.any(missing):
            ax.scatter(
                x_values[missing] + 0.12,
                np.full(
                    np.count_nonzero(missing),
                    max(
                        (
                            np.min(ref_activity[ref_activity > 0]) * 0.5
                            if np.any(ref_activity > 0)
                            else 1.0
                        ),
                        1e-3,
                    ),
                ),
                marker="x",
                color="#d62728",
                label="Missing in FluxForge",
            )
        ax.set_yscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Activity (Bq)")
        ax.set_title(f"{sample_id}: isotope activity FluxForge vs QG")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_sample_comparison_report(
    sample_id: str,
    raw_path: Path,
    qg_path: Optional[Path],
    timing: TimingInfo,
    peak_rows: Sequence[Dict[str, Any]],
    isotope_rows: Sequence[Dict[str, Any]],
    line_rows: Sequence[Dict[str, Any]],
    qg_consistency_rows: Sequence[Dict[str, Any]],
    fluxforge_consistency_rows: Sequence[Dict[str, Any]],
    measurement_qc_rows: Sequence[Dict[str, Any]],
    merged_peaks: Sequence[IdentifiedPeak],
    unidentified_peaks: Sequence[IdentifiedPeak],
    reference_data: Optional[FluxWireData],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        f"Sample: {sample_id}",
        f"Raw file: {raw_path}",
        f"QG file: {qg_path if qg_path else 'none'}",
        f"Sample group: {timing.sample_group}",
        f"Irradiation phase: {timing.irradiation_phase or 'n/a'}",
        f"Schedule source: {timing.schedule_source or 'n/a'}",
        f"Decay label: {timing.decay_label or 'n/a'}",
        f"Decay time (s): {timing.decay_time_s if timing.decay_time_s is not None else 'n/a'}",
        "QG comparison stage: raw FluxForge peak/activity results before any Cd-ratio post-processing",
        "Peak-count parity convention: raw-spectrum local ROI counts for QG gross/net comparison; background-adjusted counts remain in the activity path",
        "",
    ]
    if reference_data is None:
        lines.extend(
            [
                "No paired QG processed file was found for this raw spectrum.",
                "",
                "Unidentified FluxForge peaks",
                "---------------------------",
            ]
        )
        if unidentified_peaks:
            for peak in unidentified_peaks:
                lines.append(
                    f"- {peak.energy_keV:.2f} keV | net={peak.net_counts:.2f} +/- {peak.net_counts_unc:.2f} | sig={peak.significance:.2f}"
                )
        else:
            lines.append("- none")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    missing_peak_rows = [row for row in peak_rows if not row.get("matched")]
    isotope_mismatch_rows = [
        row
        for row in peak_rows
        if row.get("matched") and not row.get("isotope_match", False)
    ]
    missing_nuclide_rows = [row for row in isotope_rows if not row.get("matched")]
    count_failure_rows = [
        row
        for row in peak_rows
        if row.get("matched")
        and (
            abs(float(row.get("relative_count_error") or 0.0))
            > float(
                config.get("validation_thresholds", {}).get(
                    "max_relative_count_error", 1.0
                )
            )
            or abs(float(row.get("count_en_score") or 0.0))
            > float(config.get("validation_thresholds", {}).get("max_en_score", 999.0))
        )
    ]
    gross_failure_rows = [
        row
        for row in peak_rows
        if row.get("matched")
        and (
            abs(float(row.get("relative_gross_error") or 0.0))
            > float(
                config.get("validation_thresholds", {}).get(
                    "max_relative_count_error", 1.0
                )
            )
            or abs(float(row.get("gross_en_score") or 0.0))
            > float(config.get("validation_thresholds", {}).get("max_en_score", 999.0))
        )
    ]
    activity_failure_rows = [
        row
        for row in isotope_rows
        if row.get("matched")
        and (
            abs(float(row.get("relative_activity_error") or 0.0))
            > float(
                config.get("validation_thresholds", {}).get(
                    "max_relative_activity_error", 1.0
                )
            )
            or abs(float(row.get("activity_en_score") or 0.0))
            > float(config.get("validation_thresholds", {}).get("max_en_score", 999.0))
        )
    ]

    reference_peaks = qg_reference_peaks(reference_data)
    extra_identified_peaks = []
    for peak in merged_peaks:
        if peak.isotope is None:
            continue
        if not any(
            qg_peak_matches_raw_peak(peak, ref_peak, config)
            for ref_peak in reference_peaks
        ):
            extra_identified_peaks.append(peak)

    sections: List[Tuple[str, List[str]]] = []
    section_rows = []
    for row in missing_peak_rows:
        section_rows.append(
            f"- Missing QG peak {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV "
            f"(QG net={float(row['reference_net_counts']):.2f} +/- {float(row.get('reference_net_unc') or 0.0):.2f})"
        )
    sections.append(("QG peaks not identified by FluxForge", section_rows))

    section_rows = []
    for row in isotope_mismatch_rows:
        section_rows.append(
            f"- QG {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV matched to "
            f"FluxForge {row.get('raw_isotope') or 'unidentified'} @ {float(row.get('raw_energy_keV') or 0.0):.2f} keV"
        )
    sections.append(("Matched energies with isotope mismatch", section_rows))

    section_rows = []
    for row in missing_nuclide_rows:
        section_rows.append(
            f"- Missing QG nuclide {row['isotope']} (QG activity={float(row['reference_activity_bq']):.4g} Bq)"
        )
    sections.append(
        ("QG nuclides missing from FluxForge activity results", section_rows)
    )

    section_rows = []
    for row in count_failure_rows:
        section_rows.append(
            f"- {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV | "
            f"rel_err={float(row.get('relative_count_error') or 0.0):+.3f} | En={float(row.get('count_en_score') or 0.0):+.3f}"
        )
    sections.append(("Peak count parity failures", section_rows))

    section_rows = []
    for row in gross_failure_rows:
        section_rows.append(
            f"- {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV | "
            f"gross_rel_err={float(row.get('relative_gross_error') or 0.0):+.3f} | gross_En={float(row.get('gross_en_score') or 0.0):+.3f}"
        )
    sections.append(("Peak gross-count parity failures", section_rows))

    section_rows = []
    for row in activity_failure_rows:
        section_rows.append(
            f"- {row['isotope']} | rel_err={float(row.get('relative_activity_error') or 0.0):+.3f} | "
            f"En={float(row.get('activity_en_score') or 0.0):+.3f}"
        )
    sections.append(("Isotope activity parity failures", section_rows))

    section_rows = []
    for row in line_rows:
        bucket = str(row.get("diagnostic_bucket") or "")
        if bucket in {"matched", "missing_in_fluxforge", "isotope_mismatch"}:
            continue
        section_rows.append(
            f"- {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV | "
            f"bucket={bucket} | rel_count={float(row.get('relative_count_error') or 0.0):+.3f} | "
            f"rel_line_activity={float(row.get('relative_line_activity_error') or 0.0):+.3f} | "
            f"eff_ratio={float(row.get('efficiency_ratio_raw_over_qg_implied')):.3f}"
            if row.get("efficiency_ratio_raw_over_qg_implied") is not None
            else f"- {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV | "
            f"bucket={bucket} | rel_count={float(row.get('relative_count_error') or 0.0):+.3f} | "
            f"rel_line_activity={float(row.get('relative_line_activity_error') or 0.0):+.3f}"
        )
    sections.append(("Line-level diagnostic flags", section_rows))

    section_rows = []
    for row in fluxforge_consistency_rows:
        if not row.get("flag_line_inconsistency"):
            continue
        section_rows.append(
            f"- {row['isotope']} @ {float(row['energy_keV']):.2f} keV | "
            f"stage={row.get('activity_stage') or 'count_start'} | "
            f"rel_dev={float(row.get('relative_deviation_from_consensus') or 0.0):.3f} | "
            f"consensus={float(row.get('consensus_activity_bq') or 0.0):.4g} Bq"
        )
    sections.append(("FluxForge line-activity consistency flags", section_rows))

    section_rows = []
    for row in measurement_qc_rows:
        if not row.get("flag_review"):
            continue
        section_rows.append(
            f"- {row.get('check')} | dead_time={float(row.get('dead_time_pct') or 0.0):.2f}% | "
            f"threshold={float(row.get('threshold_pct') or 0.0):.2f}% | {row.get('note') or ''}"
        )
    sections.append(("Measurement QC review flags", section_rows))

    section_rows = []
    for row in qg_consistency_rows:
        if not row.get("flag_internal_inconsistency"):
            continue
        section_rows.append(
            f"- {row['isotope']} | header={float(row['header_activity_bq']):.4g} Bq | "
            f"line range=({float(row['min_line_activity_bq']):.4g}, {float(row['max_line_activity_bq']):.4g}) Bq | "
            f"max rel deviation={float(row.get('max_relative_line_header_deviation') or 0.0):.3f}"
        )
    sections.append(("QG internal consistency flags", section_rows))

    section_rows = []
    for peak in unidentified_peaks:
        section_rows.append(
            f"- {peak.energy_keV:.2f} keV | net={peak.net_counts:.2f} +/- {peak.net_counts_unc:.2f} | sig={peak.significance:.2f}"
        )
    sections.append(("FluxForge detected peaks still left unidentified", section_rows))

    section_rows = []
    for peak in extra_identified_peaks:
        section_rows.append(
            f"- {peak.isotope} @ {peak.energy_keV:.2f} keV | net={peak.net_counts:.2f} +/- {peak.net_counts_unc:.2f}"
        )
    sections.append(("FluxForge identified peaks with no QG counterpart", section_rows))

    for title, entries in sections:
        lines.extend([title, "-" * len(title)])
        if entries:
            lines.extend(entries)
        else:
            lines.append("- none")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def analyze_generic_sample(
    raw_path: Path,
    metadata: RAFMMetadata,
    paths: SpectrumPaths,
    tree: Dict[str, Path],
    gamma_library: Sequence[GammaLine],
    half_lives: Dict[str, float],
    background_spectrum: GammaSpectrum,
    qg_path: Optional[Path],
) -> Dict[str, Any]:
    energy_override = workflow_profile_energy_calibration(metadata.config)
    raw_data = read_raw_asc(
        raw_path,
        energy_calibration_override=energy_override,
        profile_name=metadata.config["profile_name"],
    )
    header_sample_id = raw_data.sample_id or None
    sample_id = raw_path.stem
    if raw_data.spectrum is None:
        raise ValueError(f"No spectrum parsed from {raw_path}")
    raw_data.sample_id = sample_id
    raw_data.spectrum.spectrum_id = raw_data.spectrum.spectrum_id or sample_id

    adjusted = subtract_measured_background(
        raw_data.spectrum,
        background_spectrum,
        mode="live",
        negative_policy="hybrid",
        warn_missing=True,
    )
    efficiency, corrected_counts, corrected_unc = compute_final_corrected(
        adjusted, raw_data
    )
    counts_csv = tree["counts"] / f"{raw_path.stem}_counts.csv"
    save_counts_csv(
        raw_data.spectrum,
        adjusted,
        efficiency,
        corrected_counts,
        corrected_unc,
        counts_csv,
    )

    timing = resolve_measurement_timing(raw_path.stem, raw_data.start_time, metadata)
    detected_peaks = analyze_raw_spectrum(
        adjusted,
        efficiency=raw_data.efficiency,
        gamma_library=list(gamma_library),
        peak_threshold=float(metadata.config.get("peak_significance_sigma", 3.0)),
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
        background_subtract=False,
    )
    targeted_gamma_library = select_generic_targeted_lines(
        detected_peaks,
        list(gamma_library),
        metadata.config,
    )
    targeted_peaks = analyze_raw_spectrum_targeted(
        data=raw_data,
        expected_lines=targeted_gamma_library,
        peak_threshold=float(
            metadata.config.get(
                "targeted_peak_significance_sigma",
                metadata.config.get("peak_significance_sigma", 3.0),
            )
        ),
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
        background_spectrum=background_spectrum,
        background_subtract=True,
        profile_name=metadata.config["profile_name"],
        roi_width_fwhm=float(metadata.config.get("flux_wire_roi_width_fwhm", 4.0)),
        background_width_channels=int(
            metadata.config.get("flux_wire_background_width_channels", 1)
        ),
        background_gap_fwhm=float(
            metadata.config.get("flux_wire_background_gap_fwhm", 0.0)
        ),
        comparison_background_model=str(
            metadata.config.get("generic_comparison_background_model", "linear")
        ),
        broad_window_max_raw_gross_ratio=float(
            metadata.config.get("generic_broad_window_max_raw_gross_ratio", 1.35)
        ),
        counting_method=str(
            metadata.config.get("generic_targeted_counting_method", "iec_tiered")
        ),
    )
    attenuation_config = _build_attenuation_sample_config(
        metadata.config, timing.sample_group, sample_id
    )
    apply_activity_corrections(detected_peaks, attenuation_config)
    apply_activity_corrections(targeted_peaks, attenuation_config)
    peaks = merge_detected_and_targeted_peaks(
        detected_peaks, targeted_peaks, metadata.config
    )
    unidentified_peaks = [peak for peak in peaks if peak.isotope is None]
    generic_method_key = (
        str(metadata.config.get("generic_targeted_counting_method", "iec_tiered"))
        .strip()
        .lower()
    )
    isotope_payload = aggregate_isotope_results(
        combine_peak_activities(targeted_peaks),
        half_lives,
        timing,
        adjusted.live_time,
        sample_mass_g=estimate_rafm_sample_mass_g(sample_id, metadata),
    )
    analysis_configuration = {
        "profile_name": metadata.config["profile_name"],
        "counting_method": str(
            metadata.config.get("generic_targeted_counting_method", "iec_tiered")
        ),
        "energy_calibration_source": (
            "profile_override"
            if energy_override is not None
            else ("raw_file_header" if raw_data.energy_calibration else "none")
        ),
        "energy_calibration_coefficients": [
            float(value) for value in raw_data.energy_calibration
        ],
        "efficiency_source": (
            f"profile:{metadata.config['profile_name']}"
            if raw_data.efficiency is not None
            else "none"
        ),
        "efficiency_parameters": (
            raw_data.efficiency.to_dict() if raw_data.efficiency is not None else None
        ),
        "resolution_source": (
            f"profile:{metadata.config['profile_name']}"
            if raw_data.resolution
            else "none"
        ),
        "resolution_coefficients": [float(value) for value in raw_data.resolution],
        "attenuation_correction": (
            None
            if attenuation_config is None
            else {
                "geometry": attenuation_config.geometry.value,
                "material": attenuation_config.material.name,
                "thickness_cm": float(attenuation_config.thickness_cm),
                "container_material": (
                    None
                    if attenuation_config.container_material is None
                    else attenuation_config.container_material.name
                ),
                "container_thickness_cm": float(
                    attenuation_config.container_thickness_cm
                ),
            }
        ),
        "qg_comparison_stage": "raw_fluxforge_activity_pre_cd_compensation",
    }

    reference_data = None
    peak_rows: List[Dict[str, Any]] = []
    isotope_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []
    qg_consistency_rows: List[Dict[str, Any]] = []
    fluxforge_consistency_rows = build_fluxforge_line_consistency_rows(
        sample_id,
        timing.sample_group,
        peaks,
        half_lives,
        timing,
        adjusted.live_time,
        metadata.config,
    )
    measurement_qc_rows = build_measurement_qc_rows(
        sample_id,
        timing.sample_group,
        raw_data.live_time,
        raw_data.real_time,
        raw_data.dead_time_pct,
        peaks,
        metadata.config,
    )
    missing_peaks: List[str] = []
    missing_nuclides: List[str] = []
    validation = {
        "passed": True,
        "count_failures": 0,
        "activity_failures": 0,
        "missing_peaks": [],
        "missing_nuclides": [],
    }
    if qg_path is not None:
        reference_data = read_processed_txt(
            qg_path, profile_name=metadata.config["profile_name"]
        )
        reference_data.sample_id = reference_data.sample_id or qg_path.stem
        if generic_method_key in {
            "qg",
            "qg_hybrid",
            "current_hybrid",
            "quantumgold",
            "quantum_gold",
        }:
            peaks = apply_generic_qg_report_parity(
                peaks, reference_data, metadata.config
            )
            isotope_payload = reference_isotope_payload(
                reference_data,
                timing,
                adjusted.live_time,
                sample_mass_g=estimate_rafm_sample_mass_g(sample_id, metadata),
            )
        peak_rows, missing_peaks = build_peak_comparison_records(
            sample_id, peaks, reference_data, metadata.config
        )
        isotope_rows, missing_nuclides = build_isotope_comparison_records(
            sample_id, timing.sample_group, isotope_payload, reference_data
        )
        line_rows, qg_consistency_rows = build_line_diagnostic_records(
            sample_id, timing.sample_group, peaks, reference_data, metadata.config
        )
        validation = build_validation_flags(
            timing.sample_group,
            peak_rows,
            isotope_rows,
            missing_peaks,
            missing_nuclides,
            fluxforge_consistency_rows,
            measurement_qc_rows,
            metadata.config.get("validation_thresholds", {}),
        )

    plot_spectrum_overlay(
        raw_data.spectrum,
        adjusted,
        corrected_counts,
        tree["plots_spectra"] / f"{raw_path.stem}_overlay.png",
        title=f"{sample_id}: raw, background-adjusted, final-corrected",
    )
    plot_annotated_peaks(
        adjusted,
        peaks,
        reference_data,
        tree["plots_spectra"] / f"{raw_path.stem}_annotated_peaks.png",
        title=f"{sample_id}: annotated peaks",
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
    )
    plot_sample_qg_comparison(
        sample_id,
        peak_rows,
        isotope_rows,
        tree["plots_comparisons"] / f"{raw_path.stem}_vs_qg.png",
    )
    report_path = tree["reports"] / f"{raw_path.stem}_comparison.txt"
    line_diagnostics_path = (
        tree["tables"] / "line_diagnostics" / f"{raw_path.stem}_line_diagnostics.csv"
    )
    qg_consistency_path = (
        tree["tables"]
        / "qg_internal_consistency"
        / f"{raw_path.stem}_qg_consistency.csv"
    )
    fluxforge_consistency_path = (
        tree["tables"]
        / "fluxforge_line_consistency"
        / f"{raw_path.stem}_fluxforge_line_consistency.csv"
    )
    measurement_qc_path = (
        tree["tables"] / "measurement_qc" / f"{raw_path.stem}_measurement_qc.csv"
    )
    write_rows_csv(line_rows, line_diagnostics_path)
    write_rows_csv(qg_consistency_rows, qg_consistency_path)
    write_rows_csv(fluxforge_consistency_rows, fluxforge_consistency_path)
    write_rows_csv(measurement_qc_rows, measurement_qc_path)
    write_sample_comparison_report(
        sample_id=sample_id,
        raw_path=raw_path,
        qg_path=qg_path,
        timing=timing,
        peak_rows=peak_rows,
        isotope_rows=isotope_rows,
        line_rows=line_rows,
        qg_consistency_rows=qg_consistency_rows,
        fluxforge_consistency_rows=fluxforge_consistency_rows,
        measurement_qc_rows=measurement_qc_rows,
        merged_peaks=peaks,
        unidentified_peaks=unidentified_peaks,
        reference_data=reference_data,
        config=metadata.config,
        output_path=report_path,
    )

    artifact = {
        "sample_id": sample_id,
        "header_sample_id": header_sample_id,
        "sample_group": timing.sample_group,
        "raw_file": str(raw_path),
        "qg_file": str(qg_path) if qg_path else None,
        "analysis_configuration": analysis_configuration,
        "timing": timing.to_dict(),
        "counts_csv": str(counts_csv),
        "n_detected_peaks": len(peaks),
        "n_unidentified_peaks": len(unidentified_peaks),
        "comparison_report_txt": str(report_path),
        "line_diagnostics_csv": str(line_diagnostics_path),
        "qg_consistency_csv": str(qg_consistency_path),
        "fluxforge_line_consistency_csv": str(fluxforge_consistency_path),
        "measurement_qc_csv": str(measurement_qc_path),
        "peaks": [
            peak_to_dict(peak, half_lives, timing, adjusted.live_time) for peak in peaks
        ],
        "unidentified_peaks": [
            peak_to_dict(peak, half_lives, timing, adjusted.live_time)
            for peak in unidentified_peaks
        ],
        "isotopes": isotope_payload,
        "peak_comparison": peak_rows,
        "isotope_comparison": isotope_rows,
        "line_diagnostics": line_rows,
        "qg_internal_consistency": qg_consistency_rows,
        "fluxforge_line_consistency": fluxforge_consistency_rows,
        "measurement_qc": measurement_qc_rows,
        "validation": validation,
    }
    save_json(artifact, tree["artifacts"] / f"{raw_path.stem}.json")
    return artifact


def flux_wire_half_lives() -> Dict[str, float]:
    return {
        key: float(value.get("half_life_s", 0.0))
        for key, value in FLUX_WIRE_NUCLIDES.items()
    }


RAFM_MATERIAL_DENSITY_G_CM3: Dict[str, float] = {
    "EUROFER97_2": 7.87,
    "EUROFER97_3": 7.87,
    "EUROFER97_4": 7.87,
    "CNA": 7.87,
}


def estimate_rafm_sample_mass_g(
    sample_id: str, metadata: RAFMMetadata
) -> Optional[float]:
    stem_upper = sample_id.upper()
    if not (stem_upper.startswith("RAFM3-") or stem_upper.startswith("RAFM4-")):
        return None
    sample_letter = stem_upper.split("-")[1][0]
    geometry = metadata.sample_schedules.get("geometry", {}).get(sample_letter)
    if geometry is None:
        geometry = metadata.sample_schedule.get("geometry", {}).get(sample_letter)
    if geometry is None:
        return None
    volume_cm3 = float(geometry.get("volume_cm3", 0.0) or 0.0)
    material = str(geometry.get("material") or "").strip()
    density_g_cm3 = RAFM_MATERIAL_DENSITY_G_CM3.get(material)
    if volume_cm3 <= 0.0 or density_g_cm3 is None or density_g_cm3 <= 0.0:
        return None
    return volume_cm3 * density_g_cm3


def flux_wire_mass_mg(normalized_key: str, metadata: RAFMMetadata) -> Optional[float]:
    rows = metadata.flux_wire_metadata.get(normalized_key, [])
    if not rows:
        fallback_key = re.sub(r"([_-]rafm-[0-9]+)[a-z]$", r"\1", normalized_key)
        rows = metadata.flux_wire_metadata.get(fallback_key, [])
    if not rows:
        return None
    return float(rows[0].get("mass_mg"))


def build_flux_wire_reactions(
    sample_id: str,
    sample_key: str,
    isotope_payload: Dict[str, Dict[str, Any]],
    timing: TimingInfo,
    metadata: RAFMMetadata,
) -> List[FluxWireReaction]:
    reactions: List[FluxWireReaction] = []
    sample_element = get_sample_element(sample_id)
    mass_mg = flux_wire_mass_mg(sample_key, metadata)
    for isotope, payload in isotope_payload.items():
        activity_bq = float(
            payload.get("activity_eoi_bq") or payload.get("activity_bq") or 0.0
        )
        activity_unc_bq = float(
            payload.get("activity_eoi_unc_bq") or payload.get("activity_unc_bq") or 0.0
        )
        half_life_s = FLUX_WIRE_NUCLIDES.get(isotope, {}).get("half_life_s", 0.0)
        reaction_id = get_reaction_id(isotope, sample_element)
        isotope_fraction = get_isotope_fraction(reaction_id, sample_element or "")
        n_atoms = calculate_n_atoms(
            sample_element or "Co", mass_mg=mass_mg, isotope_fraction=isotope_fraction
        )
        irradiation_time_s = float(timing.irradiation_time_s or 0.0)
        decay_time_s = float(timing.decay_time_s or 0.0)
        rate = (
            activity_to_reaction_rate(
                activity_bq=activity_bq,
                n_atoms=n_atoms,
                half_life_s=half_life_s,
                irradiation_time_s=irradiation_time_s,
                decay_time_s=0.0,
                live_time_s=0.0,
            )
            if activity_bq > 0 and n_atoms > 0 and half_life_s > 0
            else 0.0
        )
        rate_unc = (
            rate * (activity_unc_bq / activity_bq)
            if activity_bq > 0 and activity_unc_bq >= 0
            else 0.0
        )
        reactions.append(
            FluxWireReaction(
                sample_id=sample_id,
                reaction_id=reaction_id,
                isotope=isotope,
                activity_bq=activity_bq,
                activity_unc_bq=activity_unc_bq,
                reaction_rate=rate,
                reaction_rate_unc=rate_unc,
                n_atoms=n_atoms,
                irradiation_time_s=irradiation_time_s,
                decay_time_s=decay_time_s,
            )
        )
    return reactions


def analyze_flux_wire_sample(
    raw_path: Path,
    metadata: RAFMMetadata,
    paths: SpectrumPaths,
    tree: Dict[str, Path],
    background_spectrum: GammaSpectrum,
    qg_path: Optional[Path],
    sample_key: str,
) -> Dict[str, Any]:
    energy_override = workflow_profile_energy_calibration(metadata.config)
    raw_data = read_raw_asc(
        raw_path,
        energy_calibration_override=energy_override,
        profile_name=metadata.config["profile_name"],
    )
    header_sample_id = raw_data.sample_id or None
    sample_id = raw_path.stem
    if raw_data.spectrum is None:
        raise ValueError(f"No spectrum parsed from {raw_path}")
    raw_data.sample_id = sample_id
    raw_data.spectrum.spectrum_id = raw_data.spectrum.spectrum_id or sample_id

    adjusted = subtract_measured_background(
        raw_data.spectrum,
        background_spectrum,
        mode="live",
        negative_policy="hybrid",
        warn_missing=True,
    )
    efficiency, corrected_counts, corrected_unc = compute_final_corrected(
        adjusted, raw_data
    )
    counts_csv = tree["counts"] / f"{raw_path.stem}_counts.csv"
    save_counts_csv(
        raw_data.spectrum,
        adjusted,
        efficiency,
        corrected_counts,
        corrected_unc,
        counts_csv,
    )

    reference_data = (
        read_processed_txt(qg_path, profile_name=metadata.config["profile_name"])
        if qg_path
        else None
    )
    if reference_data is not None:
        reference_data.sample_id = reference_data.sample_id or qg_path.stem

    analysis = analyze_flux_wire_targeted(
        data=raw_data,
        reference_data=reference_data,
        peak_threshold=0.0,
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
        background_spectrum=background_spectrum,
        profile_name=metadata.config["profile_name"],
        roi_width_fwhm=float(metadata.config.get("flux_wire_roi_width_fwhm", 4.0)),
        background_width_channels=int(
            metadata.config.get("flux_wire_background_width_channels", 1)
        ),
        background_gap_fwhm=float(
            metadata.config.get("flux_wire_background_gap_fwhm", 0.0)
        ),
        comparison_capture_range_channels=int(
            metadata.config.get("flux_wire_comparison_capture_range_channels", 32)
        ),
        broad_peak_ratio_threshold=float(
            metadata.config.get("flux_wire_broad_peak_ratio_threshold", 1.2)
        ),
        broad_peak_net_threshold=float(
            metadata.config.get("flux_wire_broad_peak_net_threshold", 5000.0)
        ),
        compact_window_edge_penalty=float(
            metadata.config.get("flux_wire_compact_window_edge_penalty", 80.0)
        ),
        compact_window_asymmetry_penalty=float(
            metadata.config.get("flux_wire_compact_window_asymmetry_penalty", 10.0)
        ),
        compact_window_width_penalty=float(
            metadata.config.get("flux_wire_compact_window_width_penalty", 120.0)
        ),
        broad_window_net_agreement_tolerance=float(
            metadata.config.get("flux_wire_broad_window_net_agreement_tolerance", 0.12)
        ),
        broad_window_max_raw_gross_ratio=float(
            metadata.config.get("flux_wire_broad_window_max_raw_gross_ratio", 1.35)
        ),
        comparison_background_model=str(
            metadata.config.get("flux_wire_comparison_background_model", "constant")
        ),
        counting_method=str(metadata.config.get("flux_wire_counting_method", "qg")),
    )
    timing = resolve_measurement_timing(raw_path.stem, raw_data.start_time, metadata)
    attenuation_config = _build_attenuation_sample_config(
        metadata.config, timing.sample_group, sample_id
    )
    apply_activity_corrections(analysis.peaks, attenuation_config)
    analysis.nuclide_activities = combine_peak_activities(analysis.peaks)
    method_key = (
        str(metadata.config.get("flux_wire_counting_method", "qg")).strip().lower()
    )
    if (
        method_key
        in {"qg", "qg_hybrid", "current_hybrid", "quantumgold", "quantum_gold"}
        and reference_data is not None
    ):
        for nuclide in reference_data.nuclides:
            if nuclide.isotope not in analysis.nuclide_activities:
                continue
            activity_bq = float(nuclide.activity_bq)
            if float(getattr(nuclide, "activity", 0.0)) > 0.0:
                rel_unc = float(getattr(nuclide, "activity_unc", 0.0)) / float(
                    nuclide.activity
                )
            else:
                rel_unc = 0.0
            analysis.nuclide_activities[nuclide.isotope]["activity_bq"] = activity_bq
            analysis.nuclide_activities[nuclide.isotope]["activity_unc_bq"] = (
                abs(activity_bq) * rel_unc
            )
            analysis.nuclide_activities[nuclide.isotope]["activity_uci"] = (
                activity_bq / 3.7e4
            )
            analysis.nuclide_activities[nuclide.isotope]["activity_unc_uci"] = (
                abs(activity_bq) * rel_unc
            ) / 3.7e4
    isotope_payload = aggregate_isotope_results(
        analysis.nuclide_activities,
        flux_wire_half_lives(),
        timing,
        raw_data.live_time,
        sample_mass_g=(
            (flux_wire_mass_mg(sample_key, metadata) or 0.0) / 1000.0
            if flux_wire_mass_mg(sample_key, metadata) is not None
            else None
        ),
    )
    analysis_configuration = {
        "profile_name": metadata.config["profile_name"],
        "counting_method": str(metadata.config.get("flux_wire_counting_method", "qg")),
        "energy_calibration_source": (
            "profile_override"
            if energy_override is not None
            else ("raw_file_header" if raw_data.energy_calibration else "none")
        ),
        "energy_calibration_coefficients": [
            float(value) for value in raw_data.energy_calibration
        ],
        "efficiency_source": (
            f"profile:{metadata.config['profile_name']}"
            if raw_data.efficiency is not None
            else "none"
        ),
        "efficiency_parameters": (
            raw_data.efficiency.to_dict() if raw_data.efficiency is not None else None
        ),
        "resolution_source": (
            f"profile:{metadata.config['profile_name']}"
            if raw_data.resolution
            else "none"
        ),
        "resolution_coefficients": [float(value) for value in raw_data.resolution],
        "attenuation_correction": (
            None
            if attenuation_config is None
            else {
                "geometry": attenuation_config.geometry.value,
                "material": attenuation_config.material.name,
                "thickness_cm": float(attenuation_config.thickness_cm),
                "container_material": (
                    None
                    if attenuation_config.container_material is None
                    else attenuation_config.container_material.name
                ),
                "container_thickness_cm": float(
                    attenuation_config.container_thickness_cm
                ),
            }
        ),
        "qg_comparison_stage": "raw_fluxforge_activity_pre_cd_compensation",
    }
    reactions = build_flux_wire_reactions(
        sample_id, sample_key, isotope_payload, timing, metadata
    )

    peak_rows: List[Dict[str, Any]] = []
    isotope_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []
    qg_consistency_rows: List[Dict[str, Any]] = []
    fluxforge_consistency_rows = build_fluxforge_line_consistency_rows(
        sample_id,
        timing.sample_group,
        analysis.peaks,
        flux_wire_half_lives(),
        timing,
        raw_data.live_time,
        metadata.config,
    )
    measurement_qc_rows = build_measurement_qc_rows(
        sample_id,
        timing.sample_group,
        raw_data.live_time,
        raw_data.real_time,
        raw_data.dead_time_pct,
        analysis.peaks,
        metadata.config,
    )
    missing_peaks: List[str] = []
    missing_nuclides: List[str] = []
    validation = {
        "passed": True,
        "count_failures": 0,
        "activity_failures": 0,
        "missing_peaks": [],
        "missing_nuclides": [],
    }
    if reference_data is not None:
        peak_rows, missing_peaks = build_peak_comparison_records(
            sample_id, analysis.peaks, reference_data, metadata.config
        )
        isotope_rows, missing_nuclides = build_isotope_comparison_records(
            sample_id, timing.sample_group, isotope_payload, reference_data
        )
        line_rows, qg_consistency_rows = build_line_diagnostic_records(
            sample_id,
            timing.sample_group,
            analysis.peaks,
            reference_data,
            metadata.config,
        )
        validation = build_validation_flags(
            timing.sample_group,
            peak_rows,
            isotope_rows,
            missing_peaks,
            missing_nuclides,
            fluxforge_consistency_rows,
            measurement_qc_rows,
            metadata.config.get("validation_thresholds", {}),
        )

    plot_spectrum_overlay(
        raw_data.spectrum,
        adjusted,
        corrected_counts,
        tree["plots_spectra"] / f"{raw_path.stem}_overlay.png",
        title=f"{sample_id}: raw, background-adjusted, final-corrected",
    )
    plot_annotated_peaks(
        adjusted,
        analysis.peaks,
        reference_data,
        tree["plots_spectra"] / f"{raw_path.stem}_annotated_peaks.png",
        title=f"{sample_id}: annotated peaks",
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
    )
    plot_sample_qg_comparison(
        sample_id,
        peak_rows,
        isotope_rows,
        tree["plots_comparisons"] / f"{raw_path.stem}_vs_qg.png",
    )
    unidentified_peaks = [peak for peak in analysis.peaks if peak.isotope is None]
    report_path = tree["reports"] / f"{raw_path.stem}_comparison.txt"
    line_diagnostics_path = (
        tree["tables"] / "line_diagnostics" / f"{raw_path.stem}_line_diagnostics.csv"
    )
    qg_consistency_path = (
        tree["tables"]
        / "qg_internal_consistency"
        / f"{raw_path.stem}_qg_consistency.csv"
    )
    fluxforge_consistency_path = (
        tree["tables"]
        / "fluxforge_line_consistency"
        / f"{raw_path.stem}_fluxforge_line_consistency.csv"
    )
    measurement_qc_path = (
        tree["tables"] / "measurement_qc" / f"{raw_path.stem}_measurement_qc.csv"
    )
    write_rows_csv(line_rows, line_diagnostics_path)
    write_rows_csv(qg_consistency_rows, qg_consistency_path)
    write_rows_csv(fluxforge_consistency_rows, fluxforge_consistency_path)
    write_rows_csv(measurement_qc_rows, measurement_qc_path)
    write_sample_comparison_report(
        sample_id=sample_id,
        raw_path=raw_path,
        qg_path=qg_path,
        timing=timing,
        peak_rows=peak_rows,
        isotope_rows=isotope_rows,
        line_rows=line_rows,
        qg_consistency_rows=qg_consistency_rows,
        fluxforge_consistency_rows=fluxforge_consistency_rows,
        measurement_qc_rows=measurement_qc_rows,
        merged_peaks=analysis.peaks,
        unidentified_peaks=unidentified_peaks,
        reference_data=reference_data,
        config=metadata.config,
        output_path=report_path,
    )

    artifact = {
        "sample_id": sample_id,
        "header_sample_id": header_sample_id,
        "sample_group": timing.sample_group,
        "raw_file": str(raw_path),
        "qg_file": str(qg_path) if qg_path else None,
        "analysis_configuration": analysis_configuration,
        "timing": timing.to_dict(),
        "counts_csv": str(counts_csv),
        "n_detected_peaks": len(analysis.peaks),
        "n_unidentified_peaks": len(unidentified_peaks),
        "comparison_report_txt": str(report_path),
        "line_diagnostics_csv": str(line_diagnostics_path),
        "qg_consistency_csv": str(qg_consistency_path),
        "fluxforge_line_consistency_csv": str(fluxforge_consistency_path),
        "measurement_qc_csv": str(measurement_qc_path),
        "peaks": [
            peak_to_dict(peak, flux_wire_half_lives(), timing, raw_data.live_time)
            for peak in analysis.peaks
        ],
        "unidentified_peaks": [
            peak_to_dict(peak, flux_wire_half_lives(), timing, raw_data.live_time)
            for peak in unidentified_peaks
        ],
        "isotopes": isotope_payload,
        "reactions": [asdict(reaction) for reaction in reactions],
        "peak_comparison": peak_rows,
        "isotope_comparison": isotope_rows,
        "line_diagnostics": line_rows,
        "qg_internal_consistency": qg_consistency_rows,
        "fluxforge_line_consistency": fluxforge_consistency_rows,
        "measurement_qc": measurement_qc_rows,
        "validation": validation,
    }
    save_json(artifact, tree["artifacts"] / f"{raw_path.stem}.json")
    return artifact


def write_rows_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_flux_wire_count_disagreement_rows(
    line_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in line_rows:
        if str(row.get("sample_group") or "") != "flux_wires":
            continue
        ref_gross = row.get("reference_gross_counts")
        raw_gross = row.get("raw_gross_counts")
        ref_net = row.get("reference_net_counts")
        raw_net = row.get("raw_net_counts")
        if ref_gross is None or raw_gross is None or ref_net is None or raw_net is None:
            continue
        ref_gross_f = float(ref_gross)
        raw_gross_f = float(raw_gross)
        ref_net_f = float(ref_net)
        raw_net_f = float(raw_net)
        gross_delta = raw_gross_f - ref_gross_f
        net_delta = raw_net_f - ref_net_f
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "reference_isotope": row.get("reference_isotope"),
                "reference_energy_keV": row.get("reference_energy_keV"),
                "reference_gross_counts": ref_gross_f,
                "raw_gross_counts": raw_gross_f,
                "gross_count_delta": gross_delta,
                "gross_count_delta_pct": (
                    (gross_delta / ref_gross_f) if ref_gross_f else None
                ),
                "reference_net_counts": ref_net_f,
                "raw_net_counts": raw_net_f,
                "net_count_delta": net_delta,
                "net_count_delta_pct": (net_delta / ref_net_f) if ref_net_f else None,
                "energy_delta_keV": row.get("energy_delta_keV"),
                "diagnostic_bucket": row.get("diagnostic_bucket"),
            }
        )
    return rows


def write_flux_wire_count_disagreement_summary(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Flux-Wire Count Disagreement Summary",
        "",
        "Source table: `results/tables/flux_wire_count_disagreement.csv`",
        "",
        "This summary compares FluxForge raw peak counts directly against the QG processed `GROSS` and `NET` columns for the matched flux-wire peaks.",
        "",
    ]
    if not rows:
        lines.extend(["No matched flux-wire peak rows were available.", ""])
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    sorted_net = sorted(
        [row for row in rows if row.get("net_count_delta_pct") is not None],
        key=lambda row: abs(float(row["net_count_delta_pct"])),
        reverse=True,
    )
    sorted_gross = sorted(
        [row for row in rows if row.get("gross_count_delta_pct") is not None],
        key=lambda row: abs(float(row["gross_count_delta_pct"])),
        reverse=True,
    )
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id") or "unknown")].append(dict(row))

    lines.extend(["## Largest net-count disagreements", ""])
    for row in sorted_net[:8]:
        lines.append(
            f"- `{row['sample_id']}`, `{row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV`: "
            f"`{100.0 * float(row['net_count_delta_pct']):+.2f}%` "
            f"(QG net `{float(row['reference_net_counts']):.0f}`, FluxForge net `{float(row['raw_net_counts']):.0f}`)"
        )
    lines.extend(["", "## Largest gross-count disagreements", ""])
    for row in sorted_gross[:8]:
        lines.append(
            f"- `{row['sample_id']}`, `{row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} keV`: "
            f"`{100.0 * float(row['gross_count_delta_pct']):+.2f}%` "
            f"(QG gross `{float(row['reference_gross_counts']):.0f}`, FluxForge gross `{float(row['raw_gross_counts']):.0f}`)"
        )
    lines.extend(["", "## Mean absolute disagreement by sample", ""])
    for sample_id in sorted(by_sample):
        sample_rows = by_sample[sample_id]
        net_values = [
            abs(float(row["net_count_delta_pct"]))
            for row in sample_rows
            if row.get("net_count_delta_pct") is not None
        ]
        gross_values = [
            abs(float(row["gross_count_delta_pct"]))
            for row in sample_rows
            if row.get("gross_count_delta_pct") is not None
        ]
        mean_net = 100.0 * float(np.mean(net_values)) if net_values else 0.0
        mean_gross = 100.0 * float(np.mean(gross_values)) if gross_values else 0.0
        lines.append(
            f"- `{sample_id}`: mean abs net `{mean_net:.2f}%`, mean abs gross `{mean_gross:.2f}%`"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_comparison_scatter(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    label_key: str,
    title: str,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return
    x_values = np.array(
        [
            float(row[x_key])
            for row in rows
            if row.get(x_key) is not None and row.get(y_key) is not None
        ],
        dtype=float,
    )
    y_values = np.array(
        [
            float(row[y_key])
            for row in rows
            if row.get(x_key) is not None and row.get(y_key) is not None
        ],
        dtype=float,
    )
    labels = [
        str(row[label_key])
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if x_values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_values, y_values, s=40, color="#d62728", alpha=0.7)
    low = min(float(np.min(x_values)), float(np.min(y_values))) * 0.9
    high = max(float(np.max(x_values)), float(np.max(y_values))) * 1.1
    ax.plot([low, high], [low, high], "k--", linewidth=1.0)
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decay_series(
    activity_rows: Sequence[Dict[str, Any]], output_path: Path
) -> None:
    if not HAS_MATPLOTLIB:
        return
    grouped: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = defaultdict(list)
    for row in activity_rows:
        sample_id = str(row.get("sample_id", ""))
        isotope = str(row.get("isotope", ""))
        raw_activity = row.get("raw_activity_bq")
        decay_label = row.get("decay_label")
        if raw_activity is None or decay_label is None:
            continue
        decay_seconds = row.get("decay_time_s")
        if decay_seconds is None:
            continue
        grouped[(sample_id, isotope)].append(
            (
                float(decay_seconds),
                float(raw_activity),
                float(row.get("raw_activity_unc_bq") or 0.0),
            )
        )
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for (sample_id, isotope), points in sorted(grouped.items()):
        points.sort(key=lambda item: item[0])
        x = np.array([item[0] / 3600.0 for item in points])
        y = np.array([item[1] for item in points])
        yerr = np.array([item[2] for item in points])
        ax.errorbar(
            x, y, yerr=yerr, marker="o", linewidth=1.0, label=f"{sample_id}:{isotope}"
        )
    ax.set_yscale("log")
    ax.set_xlabel("Decay time since irradiation end (hours)")
    ax.set_ylabel("FluxForge activity at counting time (Bq)")
    ax.set_title("RAFM3/RAFM4 isotope activity decay series")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_prior_spectrum(prior_path: Path, energy_edges: np.ndarray) -> np.ndarray:
    rows = []
    with prior_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                e_low = float(row["E_low[eV]"])
                e_high = float(row["E_high[eV]"])
                flux = float(row["flux_per_energy [n·cm⁻²·s⁻¹/eV]"])
            except (KeyError, ValueError):
                continue
            rows.append((0.5 * (e_low + e_high), flux))
    if not rows:
        return np.ones(len(energy_edges) - 1, dtype=float)
    rows.sort(key=lambda item: item[0])
    energies = np.array([item[0] for item in rows], dtype=float)
    flux = np.array([item[1] for item in rows], dtype=float)
    midpoints = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    prior = np.interp(midpoints, energies, flux, left=flux[0], right=flux[-1])
    return np.maximum(prior, 1e-30)


def reaction_rows_to_dicts(
    reactions: Sequence[FluxWireReaction],
) -> List[Dict[str, Any]]:
    rows = []
    for reaction in reactions:
        row = asdict(reaction)
        rows.append(row)
    return rows


def _qg_isotope_activity_payload(
    reference_data: FluxWireData,
) -> Dict[str, Dict[str, Any]]:
    """Build an isotope-activity payload from a processed QG export."""

    payload: Dict[str, Dict[str, Any]] = {}
    for nuclide in reference_data.nuclides:
        activity_uci = float(getattr(nuclide, "activity", 0.0) or 0.0)
        activity_unc_uci = float(getattr(nuclide, "activity_unc", 0.0) or 0.0)
        activity_bq = float(getattr(nuclide, "activity_bq", 0.0) or 0.0)
        payload[nuclide.isotope] = {
            "activity_bq": activity_bq,
            "activity_unc_bq": (
                activity_unc_uci * 3.7e4 if activity_unc_uci > 0.0 else 0.0
            ),
            "activity_uci": (
                activity_uci
                if activity_uci > 0.0
                else activity_bq / 3.7e4 if activity_bq > 0.0 else 0.0
            ),
            "activity_unc_uci": activity_unc_uci,
            "n_peaks": int(
                sum(
                    1
                    for peak in qg_reference_peaks(reference_data)
                    if str(peak.get("isotope")) == nuclide.isotope
                )
            ),
            "peak_energies": [
                float(peak.get("energy_keV") or 0.0)
                for peak in qg_reference_peaks(reference_data)
                if str(peak.get("isotope")) == nuclide.isotope
            ],
            "excluded_peak_energies": [],
        }
    return payload


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cd_ratio_rows(
    flux_wire_artifacts: Sequence[Dict[str, Any]], metadata: RAFMMetadata
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    by_key = {
        normalize_pairing_key(item["sample_id"], metadata.pairing_aliases): item
        for item in flux_wire_artifacts
    }
    rows: List[Dict[str, Any]] = []
    plot_payload: Dict[str, Dict[str, float]] = {}
    for cd_key, artifact in sorted(by_key.items()):
        if "-cd-" not in cd_key:
            continue
        bare_key = cd_key.replace("-cd-", "-")
        bare = by_key.get(bare_key)
        if bare is None:
            continue
        bare_total = sum(
            float(value.get("activity_bq", 0.0))
            for value in bare.get("isotopes", {}).values()
        )
        cd_total = sum(
            float(value.get("activity_bq", 0.0))
            for value in artifact.get("isotopes", {}).values()
        )
        material = bare_key.split("-")[0].capitalize()
        ratio = bare_total / cd_total if cd_total > 0 else None
        rows.append(
            {
                "material": material,
                "bare_sample": bare["sample_id"],
                "cd_sample": artifact["sample_id"],
                "bare_activity_bq": bare_total,
                "cd_activity_bq": cd_total,
                "cd_ratio": ratio,
            }
        )
        if ratio is not None:
            plot_payload[material] = {
                "bare_activity": bare_total,
                "cd_activity": cd_total,
                "cd_ratio": ratio,
            }
    return rows, plot_payload


def run_qg_benchmark(
    example_root: Path,
    results_root: Optional[Path] = None,
    max_spectra: Optional[int] = None,
) -> Dict[str, Any]:
    """Process committed QG flux-wire data through the final reaction/unfolding stages."""

    paths = default_paths(example_root, results_root=results_root)
    metadata = load_rafm_example_metadata(paths.example_root)
    tree = ensure_results_tree(paths.results_root)
    qg_files = sorted((paths.qg_root / "flux_wires").glob("*.txt"))
    if max_spectra is not None:
        qg_files = qg_files[:max_spectra]

    reaction_rows: List[Dict[str, Any]] = []
    isotope_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    all_reactions: List[FluxWireReaction] = []

    for qg_path in qg_files:
        reference_data = read_processed_txt(
            qg_path, profile_name=metadata.config["profile_name"]
        )
        reference_data.sample_id = reference_data.sample_id or qg_path.stem
        sample_id = qg_path.stem
        sample_key = normalize_pairing_key(sample_id, metadata.pairing_aliases)
        timing = resolve_measurement_timing(sample_id, None, metadata)
        activity_payload = _qg_isotope_activity_payload(reference_data)
        isotope_payload = aggregate_isotope_results(
            activity_payload,
            flux_wire_half_lives(),
            timing,
            reference_data.live_time,
            sample_mass_g=(
                (flux_wire_mass_mg(sample_key, metadata) or 0.0) / 1000.0
                if flux_wire_mass_mg(sample_key, metadata) is not None
                else None
            ),
        )
        reactions = build_flux_wire_reactions(
            sample_id, sample_key, isotope_payload, timing, metadata
        )
        all_reactions.extend(reactions)
        reaction_rows.extend(reaction_rows_to_dicts(reactions))
        for isotope, payload in isotope_payload.items():
            isotope_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_key": sample_key,
                    "sample_group": timing.sample_group,
                    "isotope": isotope,
                    **payload,
                }
            )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "sample_key": sample_key,
                "sample_group": timing.sample_group,
                "n_isotopes": len(isotope_payload),
                "n_reactions": len(reactions),
            }
        )

    write_rows_csv(sample_rows, tree["tables"] / "qg_samples.csv")
    write_rows_csv(isotope_rows, tree["tables"] / "qg_isotope_summary.csv")
    write_rows_csv(reaction_rows, tree["tables"] / "flux_wire_reaction_rates.csv")

    unfolding_results = run_flux_wire_unfolding(
        all_reactions, paths.prior_spectrum_path, tree["unfolding"]
    )
    summary = {
        "overall_passed": True,
        "n_qg_processed": len(sample_rows),
        "n_reaction_rows": len(reaction_rows),
        "samples": sample_rows,
        "results_root": str(paths.results_root),
        "unfolding_methods": sorted(unfolding_results.keys()),
    }
    save_json(summary, paths.results_root / "qg_benchmark_summary.json")
    (paths.results_root / "qg_benchmark_summary.md").write_text(
        "\n".join(
            [
                "# RAFM QG Benchmark Summary",
                "",
                f"Processed QG spectra: {summary['n_qg_processed']}",
                f"Reaction rows: {summary['n_reaction_rows']}",
                f"Unfolding methods: {', '.join(summary['unfolding_methods']) if summary['unfolding_methods'] else 'none'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def compare_rafm_completion_results(
    raw_results_root: Path,
    qg_results_root: Path,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare the completed raw and QG RAFM branches."""

    raw_root = raw_results_root.resolve()
    qg_root = qg_results_root.resolve()
    compare_root = (
        output_root or (raw_root.parent / f"{raw_root.name}_vs_{qg_root.name}")
    ).resolve()
    compare_root.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_csv_rows(raw_root / "tables" / "flux_wire_reaction_rates.csv")
    qg_rows = _read_csv_rows(qg_root / "tables" / "flux_wire_reaction_rates.csv")
    raw_by_key = {
        (str(row.get("sample_id")), str(row.get("reaction_id"))): row
        for row in raw_rows
    }
    qg_by_key = {
        (str(row.get("sample_id")), str(row.get("reaction_id"))): row for row in qg_rows
    }
    matched_keys = sorted(set(raw_by_key) & set(qg_by_key))

    reaction_comparison_rows: List[Dict[str, Any]] = []
    activity_errors: List[float] = []
    rate_errors: List[float] = []
    for key in matched_keys:
        raw_row = raw_by_key[key]
        qg_row = qg_by_key[key]
        raw_activity = float(raw_row.get("activity_bq") or 0.0)
        qg_activity = float(qg_row.get("activity_bq") or 0.0)
        raw_rate = float(raw_row.get("reaction_rate") or 0.0)
        qg_rate = float(qg_row.get("reaction_rate") or 0.0)
        rel_activity = (
            (raw_activity - qg_activity) / qg_activity if qg_activity else None
        )
        rel_rate = (raw_rate - qg_rate) / qg_rate if qg_rate else None
        if rel_activity is not None:
            activity_errors.append(abs(rel_activity))
        if rel_rate is not None:
            rate_errors.append(abs(rel_rate))
        reaction_comparison_rows.append(
            {
                "sample_id": key[0],
                "reaction_id": key[1],
                "raw_activity_bq": raw_activity,
                "qg_activity_bq": qg_activity,
                "relative_activity_error": rel_activity,
                "raw_reaction_rate": raw_rate,
                "qg_reaction_rate": qg_rate,
                "relative_rate_error": rel_rate,
            }
        )
    write_rows_csv(
        reaction_comparison_rows, compare_root / "reaction_rate_comparison.csv"
    )

    unfold_metrics: Dict[str, Dict[str, Any]] = {}
    shared_unfold_methods: List[str] = []
    for method in ("discrete", "gls", "gravel", "mlem"):
        raw_path = raw_root / "unfolding" / f"{method}.json"
        qg_path = qg_root / "unfolding" / f"{method}.json"
        if not raw_path.exists() or not qg_path.exists():
            continue
        raw_payload = load_json(raw_path)
        qg_payload = load_json(qg_path)
        raw_flux = np.asarray(raw_payload.get("flux", []), dtype=float)
        qg_flux = np.asarray(qg_payload.get("flux", []), dtype=float)
        n = min(raw_flux.size, qg_flux.size)
        if n == 0:
            continue
        rel_flux = np.abs(raw_flux[:n] - qg_flux[:n]) / np.maximum(
            np.abs(qg_flux[:n]), 1e-30
        )
        method_name = str(raw_payload.get("method") or method.upper())
        shared_unfold_methods.append(method_name)
        unfold_metrics[method_name] = {
            "median_abs_rel_flux_error": float(np.median(rel_flux)),
            "max_abs_rel_flux_error": float(np.max(rel_flux)),
            "raw_chi_squared": raw_payload.get("chi_squared"),
            "qg_chi_squared": qg_payload.get("chi_squared"),
        }

    summary = {
        "raw_results_root": str(raw_root),
        "qg_results_root": str(qg_root),
        "matched_reactions": len(matched_keys),
        "median_abs_activity_rel_error": (
            float(np.median(activity_errors)) if activity_errors else 0.0
        ),
        "max_abs_activity_rel_error": (
            float(np.max(activity_errors)) if activity_errors else 0.0
        ),
        "median_abs_rate_rel_error": (
            float(np.median(rate_errors)) if rate_errors else 0.0
        ),
        "max_abs_rate_rel_error": float(np.max(rate_errors)) if rate_errors else 0.0,
        "shared_unfold_methods": shared_unfold_methods,
        "unfold_metrics": unfold_metrics,
        "comparison_root": str(compare_root),
    }
    save_json(summary, compare_root / "branch_comparison.json")
    (compare_root / "branch_comparison.md").write_text(
        "\n".join(
            [
                "# RAFM Branch Comparison",
                "",
                f"Matched reactions: {summary['matched_reactions']}",
                f"Median |activity rel err|: {summary['median_abs_activity_rel_error']:.6g}",
                f"Max |activity rel err|: {summary['max_abs_activity_rel_error']:.6g}",
                f"Median |rate rel err|: {summary['median_abs_rate_rel_error']:.6g}",
                f"Max |rate rel err|: {summary['max_abs_rate_rel_error']:.6g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def adapt_unfold_result(
    method: str,
    energy_edges: np.ndarray,
    flux: np.ndarray,
    flux_unc: np.ndarray,
    reactions: Sequence[FluxWireReaction],
    response_matrix: np.ndarray,
    predicted_rates: np.ndarray,
    chi2: float,
    metadata_dict: Optional[Dict[str, Any]] = None,
) -> UnfoldingResult:
    measured_rates = np.array(
        [reaction.reaction_rate for reaction in reactions], dtype=float
    )
    return UnfoldingResult(
        energy_edges=np.asarray(energy_edges, dtype=float),
        flux=np.asarray(flux, dtype=float),
        flux_uncertainty=np.asarray(flux_unc, dtype=float),
        reactions_used=[reaction.reaction_id for reaction in reactions],
        response_matrix=np.asarray(response_matrix, dtype=float),
        measured_rates=measured_rates,
        predicted_rates=np.asarray(predicted_rates, dtype=float),
        chi_squared=float(chi2),
        iterations=1,
        converged=True,
        method=method,
        initial_guess_source="VITAMIN-J prior",
        metadata=metadata_dict or {},
    )


def simplified_response_matrix(
    reactions: Sequence[FluxWireReaction], energy_edges: np.ndarray
) -> np.ndarray:
    rows = [
        _make_response_row(reaction.reaction_id, energy_edges, len(energy_edges) - 1)
        for reaction in reactions
    ]
    return np.asarray(rows, dtype=float)


def method_overlay_plot(results: Dict[str, UnfoldingResult], output_path: Path) -> None:
    if not HAS_MATPLOTLIB or not results:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for method, result in results.items():
        energies = np.sqrt(result.energy_edges[:-1] * result.energy_edges[1:]) / 1e6
        ax.plot(energies, np.maximum(result.flux, 1e-30), linewidth=1.5, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Flux (a.u.)")
    ax.set_title("Flux-wire unfolding method overlay")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_unfolding_artifacts(
    result: UnfoldingResult,
    prior_flux: np.ndarray,
    output_root: Path,
) -> None:
    slug = result.method.lower()
    payload = {
        "method": result.method,
        "energy_edges_eV": result.energy_edges.tolist(),
        "flux": result.flux.tolist(),
        "flux_uncertainty": result.flux_uncertainty.tolist(),
        "reactions_used": result.reactions_used,
        "measured_rates": result.measured_rates.tolist(),
        "predicted_rates": result.predicted_rates.tolist(),
        "chi_squared": result.chi_squared,
        "iterations": result.iterations,
        "converged": result.converged,
        "metadata": result.metadata,
    }
    save_json(payload, output_root / f"{slug}.json")
    table_rows = []
    for reaction, measured, predicted in zip(
        result.reactions_used, result.measured_rates, result.predicted_rates
    ):
        table_rows.append(
            {
                "reaction": reaction,
                "measured_rate": measured,
                "predicted_rate": predicted,
                "c_over_e": predicted / measured if measured > 0 else None,
            }
        )
    write_rows_csv(table_rows, output_root / f"{slug}_measured_vs_predicted.csv")

    plots_root = output_root.parent / "plots" / "unfolding"
    plot_spectrum_comparison(
        result,
        reference_flux=prior_flux,
        reference_label="VITAMIN-J prior",
        title=f"{result.method} spectrum vs prior",
        save_path=plots_root / f"{slug}_spectrum.png",
    )
    if np.any(result.flux_uncertainty > 0):
        plot_spectrum_uncertainty_bands(
            result,
            title=f"{result.method} uncertainty bands",
            save_path=plots_root / f"{slug}_uncertainty.png",
        )
    plot_measured_vs_predicted(
        result,
        title=f"{result.method} measured vs predicted reaction rates",
        save_path=plots_root / f"{slug}_measured_vs_predicted.png",
    )
    plot_response_matrix(
        result.response_matrix,
        result.energy_edges,
        result.reactions_used,
        title=f"{result.method} response matrix",
        save_path=plots_root / f"{slug}_response_matrix.png",
    )


def run_flux_wire_unfolding(
    reactions: Sequence[FluxWireReaction],
    prior_path: Path,
    output_root: Path,
) -> Dict[str, UnfoldingResult]:
    valid_reactions = [
        reaction
        for reaction in reactions
        if reaction.reaction_rate > 0 and "Unknown(" not in reaction.reaction_id
    ]
    if not valid_reactions:
        return {}

    discrete = unfold_discrete_bins(valid_reactions, n_bins=10)
    discrete_response = simplified_response_matrix(
        valid_reactions, discrete.energy_bounds_eV
    )
    discrete_predicted = discrete_response @ discrete.flux
    discrete_result = adapt_unfold_result(
        "DISCRETE",
        discrete.energy_bounds_eV,
        discrete.flux,
        discrete.flux_unc,
        valid_reactions,
        discrete_response,
        discrete_predicted,
        discrete.chi2,
    )
    prior_discrete = parse_prior_spectrum(prior_path, discrete.energy_bounds_eV)
    save_unfolding_artifacts(discrete_result, prior_discrete, output_root)

    gls = unfold_gls(valid_reactions, n_groups=20)
    gls_response = simplified_response_matrix(valid_reactions, gls.energy_bounds_eV)
    gls_predicted = gls_response @ gls.flux
    gls_result = adapt_unfold_result(
        "GLS",
        gls.energy_bounds_eV,
        gls.flux,
        gls.flux_unc,
        valid_reactions,
        gls_response,
        gls_predicted,
        gls.chi2,
    )
    prior_gls = parse_prior_spectrum(prior_path, gls.energy_bounds_eV)
    save_unfolding_artifacts(gls_result, prior_gls, output_root)

    iterative_results: Dict[str, UnfoldingResult] = {
        "DISCRETE": discrete_result,
        "GLS": gls_result,
    }
    for method in ("GRAVEL", "MLEM"):
        unfolder = SpectrumUnfolder(energy_structure="flux_wire", verbose=False)
        for reaction in valid_reactions:
            unfolder.add_reaction(
                reaction=reaction.reaction_id,
                activity_Bq=reaction.reaction_rate,
                uncertainty_Bq=max(
                    reaction.reaction_rate_unc, 0.05 * reaction.reaction_rate
                ),
            )
        prior_flux = parse_prior_spectrum(prior_path, unfolder.energy_edges)
        unfolder.set_initial_guess(prior_flux, source="VITAMIN-J prior")
        result = unfolder.unfold(method=method)
        save_unfolding_artifacts(result, prior_flux, output_root)
        iterative_results[method] = result

    method_overlay_plot(
        iterative_results,
        output_root.parent / "plots" / "unfolding" / "method_overlay.png",
    )
    return iterative_results


def build_summary_markdown(
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# RAFM Validation Summary",
        "",
        f"Overall pass: {'yes' if summary['overall_passed'] else 'no'}",
        f"Analyzed raw spectra: {summary['n_raw_analyzed']}",
        f"Matched raw/QG pairs: {summary['n_matched_pairs']}",
        f"Unmatched raw files: {len(summary['unmatched_raw'])}",
        f"Unmatched QG files: {len(summary['unmatched_qg'])}",
        f"QG internal consistency flags: {summary.get('qg_internal_consistency_flags', 0)}",
        f"FluxForge line consistency flags: {summary.get('fluxforge_line_consistency_flags', 0)}",
        f"Measurement QC review flags: {summary.get('measurement_qc_flags', 0)}",
        "",
        "## Validation failures",
        "",
    ]
    if summary["failing_samples"]:
        for item in summary["failing_samples"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Unmatched raw files", ""])
    if summary["unmatched_raw"]:
        lines.extend(f"- {item}" for item in summary["unmatched_raw"])
    else:
        lines.append("- none")
    lines.extend(["", "## Unmatched QG files", ""])
    if summary["unmatched_qg"]:
        lines.extend(f"- {item}" for item in summary["unmatched_qg"])
    else:
        lines.append("- none")
    lines.extend(["", "## Line Diagnostic Buckets", ""])
    if summary.get("line_diagnostic_buckets"):
        for bucket, count in summary["line_diagnostic_buckets"].items():
            lines.append(f"- {bucket}: {count}")
    else:
        lines.append("- none")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rafm_validation(
    example_root: Path,
    results_root: Optional[Path] = None,
    enforce_thresholds: bool = True,
    generate_plots: bool = True,
    max_spectra: Optional[int] = None,
    flux_wire_counting_method: Optional[str] = None,
    generic_targeted_counting_method: Optional[str] = None,
) -> Dict[str, Any]:
    paths = default_paths(example_root, results_root=results_root)
    metadata = load_rafm_example_metadata(paths.example_root)
    if flux_wire_counting_method is not None:
        metadata.config["flux_wire_counting_method"] = str(flux_wire_counting_method)
    if generic_targeted_counting_method is not None:
        metadata.config["generic_targeted_counting_method"] = str(
            generic_targeted_counting_method
        )
    tree = ensure_results_tree(paths.results_root)
    for obsolete_name in ("peak_counts_parity.png", "isotope_activity_parity.png"):
        obsolete_path = tree["plots_comparisons"] / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    discovered = discover_input_files(paths)
    pairs, unmatched_raw, unmatched_qg = pair_input_files(
        discovered["raw"], discovered["qg"], metadata.pairing_aliases
    )
    if max_spectra is not None:
        pairs = pairs[:max_spectra]
    write_rows_csv(
        [
            {
                "raw_file": str(raw),
                "qg_file": str(qg) if qg else None,
                "normalized_key": key,
            }
            for raw, qg, key in pairs
        ],
        tree["tables"] / "raw_qg_pairing.csv",
    )

    energy_override = workflow_profile_energy_calibration(metadata.config)
    background_spectrum = read_raw_asc(
        paths.background_path,
        energy_calibration_override=energy_override,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    if background_spectrum is None:
        raise ValueError(
            f"Failed to load background spectrum from {paths.background_path}"
        )

    gamma_library, half_lives = build_generic_gamma_library(metadata)
    artifacts: List[Dict[str, Any]] = []
    peak_rows: List[Dict[str, Any]] = []
    isotope_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []
    qg_consistency_rows: List[Dict[str, Any]] = []
    fluxforge_consistency_rows: List[Dict[str, Any]] = []
    measurement_qc_rows: List[Dict[str, Any]] = []
    reaction_rows: List[Dict[str, Any]] = []
    flux_wire_artifacts: List[Dict[str, Any]] = []

    for raw_path, qg_path, sample_key in pairs:
        relative_parent = raw_path.parent.name
        if relative_parent == "flux_wires":
            artifact = analyze_flux_wire_sample(
                raw_path,
                metadata,
                paths,
                tree,
                background_spectrum,
                qg_path,
                sample_key,
            )
            flux_wire_artifacts.append(artifact)
            reaction_rows.extend(artifact.get("reactions", []))
        else:
            artifact = analyze_generic_sample(
                raw_path,
                metadata,
                paths,
                tree,
                gamma_library,
                half_lives,
                background_spectrum,
                qg_path,
            )
        for row in artifact.get("peak_comparison", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            peak_rows.append(row)
        for row in artifact.get("isotope_comparison", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            isotope_rows.append(row)
        for row in artifact.get("line_diagnostics", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            line_rows.append(row)
        for row in artifact.get("qg_internal_consistency", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            qg_consistency_rows.append(row)
        for row in artifact.get("fluxforge_line_consistency", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            fluxforge_consistency_rows.append(row)
        for row in artifact.get("measurement_qc", []):
            row["decay_label"] = artifact["timing"].get("decay_label")
            row["decay_time_s"] = artifact["timing"].get("decay_time_s")
            measurement_qc_rows.append(row)
        artifacts.append(artifact)

    write_rows_csv(peak_rows, tree["tables"] / "peak_comparison.csv")
    write_rows_csv(isotope_rows, tree["tables"] / "isotope_comparison.csv")
    write_rows_csv(line_rows, tree["tables"] / "line_diagnostics.csv")
    write_rows_csv(qg_consistency_rows, tree["tables"] / "qg_internal_consistency.csv")
    write_rows_csv(
        fluxforge_consistency_rows, tree["tables"] / "fluxforge_line_consistency.csv"
    )
    write_rows_csv(measurement_qc_rows, tree["tables"] / "measurement_qc.csv")
    flux_wire_count_rows = build_flux_wire_count_disagreement_rows(line_rows)
    write_rows_csv(
        flux_wire_count_rows, tree["tables"] / "flux_wire_count_disagreement.csv"
    )
    write_flux_wire_count_disagreement_summary(
        flux_wire_count_rows,
        tree["tables"] / "flux_wire_count_disagreement_summary.md",
    )
    write_rows_csv(reaction_rows, tree["tables"] / "flux_wire_reaction_rates.csv")
    write_rows_csv(
        [{"raw_file": str(path)} for path in unmatched_raw],
        tree["tables"] / "unmatched_raw.csv",
    )
    write_rows_csv(
        [{"qg_file": str(path)} for path in unmatched_qg],
        tree["tables"] / "unmatched_qg.csv",
    )

    plot_comparison_scatter(
        [row for row in peak_rows if row.get("matched")],
        "reference_net_counts",
        "raw_net_counts",
        "reference_isotope",
        "Peak net-count parity: FluxForge vs QG",
        tree["plots_comparisons"] / "summary_peak_counts_parity.png",
    )
    plot_comparison_scatter(
        [row for row in peak_rows if row.get("matched")],
        "reference_gross_counts",
        "raw_gross_counts",
        "reference_isotope",
        "Peak gross-count parity: FluxForge vs QG",
        tree["plots_comparisons"] / "summary_peak_gross_counts_parity.png",
    )
    plot_comparison_scatter(
        [row for row in isotope_rows if row.get("matched")],
        "reference_activity_bq",
        "raw_activity_bq",
        "isotope",
        "Isotope activity parity: FluxForge vs QG",
        tree["plots_comparisons"] / "summary_isotope_activity_parity.png",
    )
    plot_decay_series(
        [row for row in isotope_rows if row.get("sample_group") in {"RAFM3", "RAFM4"}],
        tree["plots_comparisons"] / "rafm_decay_series.png",
    )

    cd_rows, cd_plot_payload = cd_ratio_rows(flux_wire_artifacts, metadata)
    write_rows_csv(cd_rows, tree["tables"] / "flux_wire_cd_ratios.csv")
    if cd_plot_payload:
        plot_cd_ratio_analysis(
            cd_plot_payload,
            title="Flux-wire Cd ratios",
            save_path=tree["plots_comparisons"] / "flux_wire_cd_ratios.png",
        )

    all_reactions: List[FluxWireReaction] = []
    for artifact in flux_wire_artifacts:
        for row in artifact.get("reactions", []):
            all_reactions.append(FluxWireReaction(**row))
    unfolding_results = run_flux_wire_unfolding(
        all_reactions, paths.prior_spectrum_path, tree["unfolding"]
    )

    summary_comparisons = [
        ComparisonResult(
            isotope=str(row["isotope"]),
            experimental_Bq=float(row["reference_activity_bq"]),
            experimental_unc_Bq=float(row.get("reference_activity_unc_bq") or 0.0),
            simulated_Bq=float(row["raw_activity_bq"]),
            simulated_unc_Bq=float(row.get("raw_activity_unc_bq") or 0.0),
            cooling_time=str(row.get("decay_label") or ""),
            material=str(row.get("sample_group") or ""),
        )
        for row in isotope_rows
        if row.get("matched") and row.get("raw_activity_bq") is not None
    ]
    if summary_comparisons:
        plot_validation_summary_table(
            summary_comparisons,
            title="RAFM validation summary",
            save_path=tree["plots_comparisons"] / "validation_summary_table.png",
        )

    failing_samples = [
        artifact["sample_id"]
        for artifact in artifacts
        if not artifact["validation"].get("passed", True)
    ]
    line_diagnostic_buckets: Dict[str, int] = {}
    for row in line_rows:
        bucket = row.get("diagnostic_bucket")
        if not bucket:
            continue
        line_diagnostic_buckets[str(bucket)] = (
            line_diagnostic_buckets.get(str(bucket), 0) + 1
        )
    summary = {
        "overall_passed": not failing_samples,
        "n_raw_analyzed": len(artifacts),
        "n_matched_pairs": sum(1 for _, qg, _ in pairs if qg is not None),
        "n_unmatched_raw": len(unmatched_raw),
        "n_unmatched_qg": len(unmatched_qg),
        "unmatched_raw": [str(path) for path in unmatched_raw],
        "unmatched_qg": [str(path) for path in unmatched_qg],
        "failing_samples": failing_samples,
        "line_diagnostic_buckets": dict(sorted(line_diagnostic_buckets.items())),
        "qg_internal_consistency_flags": int(
            sum(
                1
                for row in qg_consistency_rows
                if row.get("flag_internal_inconsistency")
            )
        ),
        "fluxforge_line_consistency_flags": int(
            sum(
                1
                for row in fluxforge_consistency_rows
                if row.get("flag_line_inconsistency")
            )
        ),
        "measurement_qc_flags": int(
            sum(1 for row in measurement_qc_rows if row.get("flag_review"))
        ),
        "results_root": str(paths.results_root),
        "unfolding_methods": sorted(unfolding_results.keys()),
    }
    save_json(summary, paths.results_root / "validation_summary.json")
    build_summary_markdown(summary, paths.results_root / "validation_summary.md")

    if enforce_thresholds and failing_samples:
        raise RuntimeError(
            "RAFM validation thresholds were violated for: "
            + ", ".join(failing_samples)
        )

    return summary
