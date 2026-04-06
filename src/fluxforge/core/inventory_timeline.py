"""Inventory-at-irradiation and arbitrary-time timeline helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fluxforge.core.activity_review import ActivityReviewResult
from fluxforge.core.analysis_workspace import ActivityCalculationResult
from fluxforge.data.gamma_database import DecayData, GammaDatabase, GammaLine
from fluxforge.data.isotope_names import format_isotope_name, parse_nndc_isotope_name
from fluxforge.data.nuclear_data_sources import (
    load_decay_dataset_source,
    load_gamma_identification_source,
)
from fluxforge.physics.decay_inventory import DecayInventory
from fluxforge.physics.decay_library import DecayDataset, normalize_nuclide_label
from fluxforge.physics.dose import GammaLine as DoseGammaLine
from fluxforge.physics.dose import isotope_dose_rate


DEFAULT_DECAY_SOURCE_ID = "radioactivedecay_icrp107_kayzero_2023"
TIME_ORIGINS = ("eoi", "count_start", "count_end")


@dataclass(frozen=True)
class IrradiationSchedule:
    """Minimal schedule needed for EOI/count-time inventory reconstruction."""

    cooling_time_s: float
    count_live_time_s: float

    @property
    def count_start_time_s(self) -> float:
        return max(float(self.cooling_time_s), 0.0)

    @property
    def count_end_time_s(self) -> float:
        return self.count_start_time_s + max(float(self.count_live_time_s), 0.0)


@dataclass(frozen=True)
class InventorySeed:
    """One isotope seed at end of irradiation."""

    nuclide: str
    activity_eoi_bq: float
    uncertainty_eoi_bq: float
    activity_count_start_bq: float
    uncertainty_count_start_bq: float
    half_life_s: float
    line_count: int = 0


@dataclass(frozen=True)
class InventoryState:
    """EOI inventory reconstructed from one spectrum review."""

    sample_id: str
    gamma_source_id: str
    custom_gamma_path: str | None
    schedule: IrradiationSchedule
    seeds: tuple[InventorySeed, ...]
    decay_source_id: str = DEFAULT_DECAY_SOURCE_ID


@dataclass(frozen=True)
class ObservableTimeSeries:
    """One named observable trajectory."""

    label: str
    observable: str
    unit: str
    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class InventoryTimeEvolutionResult:
    """Arbitrary-time evolution bundle with export- and plot-ready series."""

    inventory_state: InventoryState
    decay_source_id: str
    time_origin: str
    relative_times_s: tuple[float, ...]
    absolute_times_s: tuple[float, ...]
    activity_series: dict[str, ObservableTimeSeries]
    atoms_series: dict[str, ObservableTimeSeries]
    mass_series: dict[str, ObservableTimeSeries]
    dose_series: dict[str, ObservableTimeSeries]
    parents_by_nuclide: dict[str, tuple[str, ...]]
    daughters_by_nuclide: dict[str, tuple[str, ...]]
    reference_rows_by_name: dict[str, tuple[dict[str, Any], ...]]
    notes: tuple[str, ...]

    def series_for(self, observable: str) -> dict[str, ObservableTimeSeries]:
        if observable == "activity":
            return self.activity_series
        if observable == "atoms":
            return self.atoms_series
        if observable == "mass":
            return self.mass_series
        if observable == "dose":
            return self.dose_series
        raise KeyError(f"Unknown observable: {observable}")

    def plot_data(
        self,
        observable: str = "activity",
        *,
        top_n: int = 8,
        include_total: bool = True,
    ) -> dict[str, tuple[tuple[float, float, float], ...]]:
        series = self.series_for(observable)
        if not series:
            return {}

        selected: list[tuple[str, ObservableTimeSeries]] = []
        total = series.get("Total")
        if include_total and total is not None:
            selected.append(("Total", total))

        candidates = [
            (label, item)
            for label, item in series.items()
            if label != "Total"
        ]
        candidates.sort(
            key=lambda entry: max((point[1] for point in entry[1].points), default=0.0),
            reverse=True,
        )
        for label, item in candidates[: max(int(top_n), 0)]:
            selected.append((label, item))
        return {label: item.points for label, item in selected}

    def reference_rows(self, reference_name: str) -> tuple[dict[str, Any], ...]:
        return self.reference_rows_by_name.get(reference_name, ())

    def time_series_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        labels = _ordered_series_labels(self.activity_series)
        for label in labels:
            activity = self.activity_series.get(label)
            atoms = self.atoms_series.get(label)
            mass = self.mass_series.get(label)
            dose = self.dose_series.get(label)
            if activity is None or atoms is None or mass is None or dose is None:
                continue
            for index, rel_time_s in enumerate(self.relative_times_s):
                rows.append(
                    {
                        "sample_id": self.inventory_state.sample_id,
                        "time_origin": self.time_origin,
                        "relative_time_s": rel_time_s,
                        "absolute_time_s": self.absolute_times_s[index],
                        "nuclide": label,
                        "activity_bq": activity.points[index][1],
                        "activity_uncertainty_bq": activity.points[index][2],
                        "atoms": atoms.points[index][1],
                        "atoms_uncertainty": atoms.points[index][2],
                        "mass_g": mass.points[index][1],
                        "mass_uncertainty_g": mass.points[index][2],
                        "dose_rate_uSv_h": dose.points[index][1],
                        "dose_rate_uncertainty_uSv_h": dose.points[index][2],
                    }
                )
        return rows


def build_inventory_state_from_activity_review(
    review: ActivityReviewResult,
    *,
    sample_id: str = "",
    decay_source_id: str = DEFAULT_DECAY_SOURCE_ID,
) -> InventoryState:
    seeds = tuple(
        InventorySeed(
            nuclide=item.nuclide,
            activity_eoi_bq=float(item.irradiation_time_activity_bq),
            uncertainty_eoi_bq=float(item.irradiation_time_uncertainty_bq),
            activity_count_start_bq=float(item.count_time_activity_bq),
            uncertainty_count_start_bq=float(item.count_time_uncertainty_bq),
            half_life_s=float(item.half_life_s),
            line_count=int(item.line_count),
        )
        for item in review.isotope_summaries
        if item.irradiation_time_activity_bq > 0.0
    )
    return InventoryState(
        sample_id=str(sample_id or ""),
        gamma_source_id=str(review.source_id),
        custom_gamma_path=review.custom_gamma_path,
        schedule=IrradiationSchedule(
            cooling_time_s=float(review.cooling_time_s),
            count_live_time_s=float(review.live_time_s),
        ),
        seeds=seeds,
        decay_source_id=str(decay_source_id),
    )


def build_inventory_state_from_activity_results(
    results: Sequence[ActivityCalculationResult],
    *,
    live_time_s: float,
    gamma_source_id: str,
    custom_gamma_path: str | None = None,
    sample_id: str = "",
    decay_source_id: str = DEFAULT_DECAY_SOURCE_ID,
) -> InventoryState:
    cooling_time_s = max(
        (
            float(result.source_age_s)
            for result in results
            if result.source_age_s is not None
        ),
        default=0.0,
    )
    seeds = tuple(
        InventorySeed(
            nuclide=result.nuclide,
            activity_eoi_bq=float(result.age_corrected_activity_bq),
            uncertainty_eoi_bq=float(
                result.age_corrected_uncertainty_bq
                or result.uncertainty_bq
                * math.exp(
                    math.log(2.0)
                    * float(result.source_age_s)
                    / max(float(result.half_life_s), 1e-12)
                )
            ),
            activity_count_start_bq=float(result.activity_bq),
            uncertainty_count_start_bq=float(result.uncertainty_bq),
            half_life_s=float(result.half_life_s),
            line_count=1,
        )
        for result in results
        if result.age_corrected_activity_bq > 0.0
    )
    return InventoryState(
        sample_id=str(sample_id or ""),
        gamma_source_id=str(gamma_source_id),
        custom_gamma_path=custom_gamma_path,
        schedule=IrradiationSchedule(
            cooling_time_s=cooling_time_s,
            count_live_time_s=max(float(live_time_s), 0.0),
        ),
        seeds=seeds,
        decay_source_id=str(decay_source_id),
    )


def build_inventory_state_from_payload(
    payload: dict[str, Any],
    *,
    decay_source_id: str = DEFAULT_DECAY_SOURCE_ID,
) -> InventoryState:
    summaries = payload.get("isotope_summaries", []) or []
    seeds: list[InventorySeed] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        activity_eoi_bq = float(item.get("irradiation_time_activity_Bq", 0.0) or 0.0)
        if activity_eoi_bq <= 0.0:
            continue
        seeds.append(
            InventorySeed(
                nuclide=str(item.get("nuclide", "") or "").strip(),
                activity_eoi_bq=activity_eoi_bq,
                uncertainty_eoi_bq=float(
                    item.get("irradiation_time_activity_unc_Bq", 0.0) or 0.0
                ),
                activity_count_start_bq=float(item.get("count_time_activity_Bq", 0.0) or 0.0),
                uncertainty_count_start_bq=float(
                    item.get("count_time_activity_unc_Bq", 0.0) or 0.0
                ),
                half_life_s=float(item.get("half_life_s", 0.0) or 0.0),
                line_count=int(item.get("line_count", 0) or 0),
            )
        )
    return InventoryState(
        sample_id=str(payload.get("spectrum_id", "") or ""),
        gamma_source_id=str(payload.get("source_id", "fluxforge_bundled_gamma")),
        custom_gamma_path=(
            str(payload["custom_gamma_path"])
            if payload.get("custom_gamma_path")
            else None
        ),
        schedule=IrradiationSchedule(
            cooling_time_s=float(payload.get("cooling_time_s", 0.0) or 0.0),
            count_live_time_s=float(payload.get("live_time_s", 0.0) or 0.0),
        ),
        seeds=tuple(seeds),
        decay_source_id=str(decay_source_id),
    )


def build_time_grid(
    *,
    start_s: float,
    stop_s: float,
    count: int,
) -> tuple[float, ...]:
    resolved_count = max(int(count), 2)
    start = float(start_s)
    stop = float(stop_s)
    if stop < start:
        raise ValueError("Stop time must be greater than or equal to start time.")
    return tuple(float(value) for value in np.linspace(start, stop, resolved_count))


def compute_inventory_time_evolution(
    inventory_state: InventoryState,
    *,
    relative_times_s: Sequence[float],
    time_origin: str = "eoi",
    decay_source_id: str | None = None,
    distance_cm: float = 30.0,
) -> InventoryTimeEvolutionResult:
    if not inventory_state.seeds:
        raise ValueError("No irradiation-time inventory is available to evolve.")
    if time_origin not in TIME_ORIGINS:
        raise ValueError(f"Unsupported time origin: {time_origin}")

    resolved_decay_source_id = str(decay_source_id or inventory_state.decay_source_id)
    dataset = load_decay_dataset_source(resolved_decay_source_id)
    gamma_database = load_gamma_identification_source(
        inventory_state.gamma_source_id,
        custom_path=inventory_state.custom_gamma_path,
    )

    relative = tuple(float(value) for value in relative_times_s)
    absolute = tuple(
        _origin_offset_s(inventory_state.schedule, time_origin) + value
        for value in relative
    )
    if any(value < -1e-9 for value in absolute):
        raise ValueError(
            "Inventory evolution cannot be evaluated before end of irradiation."
        )

    observables = _compute_observables(
        inventory_state.seeds,
        absolute_times_s=absolute,
        dataset=dataset,
        gamma_database=gamma_database,
        distance_cm=max(float(distance_cm), 1e-6),
    )
    parents_by_nuclide, daughters_by_nuclide = _build_family_maps(
        dataset,
        [seed.nuclide for seed in inventory_state.seeds],
    )

    reference_rows_by_name = {
        "eoi": _snapshot_rows(
            inventory_state,
            dataset=dataset,
            observables=_compute_observables(
                inventory_state.seeds,
                absolute_times_s=(0.0,),
                dataset=dataset,
                gamma_database=gamma_database,
                distance_cm=max(float(distance_cm), 1e-6),
            ),
            absolute_time_s=0.0,
            time_origin="eoi",
            parents_by_nuclide=parents_by_nuclide,
            daughters_by_nuclide=daughters_by_nuclide,
            decay_source_id=resolved_decay_source_id,
        ),
        "count_start": _snapshot_rows(
            inventory_state,
            dataset=dataset,
            observables=_compute_observables(
                inventory_state.seeds,
                absolute_times_s=(inventory_state.schedule.count_start_time_s,),
                dataset=dataset,
                gamma_database=gamma_database,
                distance_cm=max(float(distance_cm), 1e-6),
            ),
            absolute_time_s=inventory_state.schedule.count_start_time_s,
            time_origin="count_start",
            parents_by_nuclide=parents_by_nuclide,
            daughters_by_nuclide=daughters_by_nuclide,
            decay_source_id=resolved_decay_source_id,
        ),
        "count_end": _snapshot_rows(
            inventory_state,
            dataset=dataset,
            observables=_compute_observables(
                inventory_state.seeds,
                absolute_times_s=(inventory_state.schedule.count_end_time_s,),
                dataset=dataset,
                gamma_database=gamma_database,
                distance_cm=max(float(distance_cm), 1e-6),
            ),
            absolute_time_s=inventory_state.schedule.count_end_time_s,
            time_origin="count_end",
            parents_by_nuclide=parents_by_nuclide,
            daughters_by_nuclide=daughters_by_nuclide,
            decay_source_id=resolved_decay_source_id,
        ),
    }

    notes = (
        "Inventory seeds are reconstructed from irradiation-time isotope activities.",
        "Time-series uncertainty combines seed-activity uncertainty, available line-intensity dose uncertainty, and direct per-nuclide half-life uncertainty.",
        "Full chain-covariance uncertainty propagation remains planned for a later phase.",
    )

    return InventoryTimeEvolutionResult(
        inventory_state=inventory_state,
        decay_source_id=resolved_decay_source_id,
        time_origin=time_origin,
        relative_times_s=relative,
        absolute_times_s=absolute,
        activity_series=_build_series(
            "activity",
            "Bq",
            relative,
            observables["activity"],
            observables["activity_variance"],
        ),
        atoms_series=_build_series(
            "atoms",
            "atoms",
            relative,
            observables["atoms"],
            observables["atoms_variance"],
        ),
        mass_series=_build_series(
            "mass",
            "g",
            relative,
            observables["mass"],
            observables["mass_variance"],
        ),
        dose_series=_build_series(
            "dose",
            "uSv/h",
            relative,
            observables["dose"],
            observables["dose_variance"],
        ),
        parents_by_nuclide=parents_by_nuclide,
        daughters_by_nuclide=daughters_by_nuclide,
        reference_rows_by_name=reference_rows_by_name,
        notes=notes,
    )


def _origin_offset_s(schedule: IrradiationSchedule, origin: str) -> float:
    if origin == "eoi":
        return 0.0
    if origin == "count_start":
        return schedule.count_start_time_s
    if origin == "count_end":
        return schedule.count_end_time_s
    raise KeyError(origin)


def _compute_observables(
    seeds: Sequence[InventorySeed],
    *,
    absolute_times_s: Sequence[float],
    dataset: DecayDataset,
    gamma_database: GammaDatabase,
    distance_cm: float,
) -> dict[str, dict[str, list[float]]]:
    size = len(tuple(absolute_times_s))
    observables: dict[str, dict[str, list[float]]] = {
        "activity": {},
        "activity_variance": {},
        "atoms": {},
        "atoms_variance": {},
        "mass": {},
        "mass_variance": {},
        "dose": {},
        "dose_variance": {},
    }

    for seed in seeds:
        if seed.activity_eoi_bq <= 0.0:
            continue
        root_rel_unc = seed.uncertainty_eoi_bq / max(seed.activity_eoi_bq, 1e-30)
        inventory = DecayInventory.from_quantities(
            {seed.nuclide: seed.activity_eoi_bq},
            unit="bq",
            dataset=dataset,
        )
        for index, absolute_time_s in enumerate(absolute_times_s):
            evolved = inventory.decay(float(absolute_time_s), units="s")
            activities = evolved.activities("bq")
            atoms = evolved.numbers()
            masses = evolved.masses("g")
            dose_values, dose_variances = _dose_by_nuclide(
                gamma_database,
                activities,
                distance_cm=distance_cm,
            )

            for nuclide, value in activities.items():
                _accumulate_value(
                    observables["activity"],
                    nuclide,
                    index,
                    size,
                    float(value),
                )
                _accumulate_variance(
                    observables["activity_variance"],
                    nuclide,
                    index,
                    size,
                    (float(value) * root_rel_unc) ** 2
                    + _half_life_variance(
                        dataset,
                        nuclide,
                        value=float(value),
                        absolute_time_s=float(absolute_time_s),
                    ),
                )
            for nuclide, value in atoms.items():
                _accumulate_value(
                    observables["atoms"],
                    nuclide,
                    index,
                    size,
                    float(value),
                )
                _accumulate_variance(
                    observables["atoms_variance"],
                    nuclide,
                    index,
                    size,
                    (float(value) * root_rel_unc) ** 2
                    + _half_life_variance(
                        dataset,
                        nuclide,
                        value=float(value),
                        absolute_time_s=float(absolute_time_s),
                    ),
                )
            for nuclide, value in masses.items():
                _accumulate_value(
                    observables["mass"],
                    nuclide,
                    index,
                    size,
                    float(value),
                )
                _accumulate_variance(
                    observables["mass_variance"],
                    nuclide,
                    index,
                    size,
                    (float(value) * root_rel_unc) ** 2
                    + _half_life_variance(
                        dataset,
                        nuclide,
                        value=float(value),
                        absolute_time_s=float(absolute_time_s),
                    ),
                )
            for nuclide, value in dose_values.items():
                _accumulate_value(
                    observables["dose"],
                    nuclide,
                    index,
                    size,
                    float(value),
                )
                _accumulate_variance(
                    observables["dose_variance"],
                    nuclide,
                    index,
                    size,
                    dose_variances.get(nuclide, 0.0)
                    + (float(value) * root_rel_unc) ** 2
                    + _half_life_variance(
                        dataset,
                        nuclide,
                        value=float(value),
                        absolute_time_s=float(absolute_time_s),
                    ),
                )

    for observable_name in ("activity", "atoms", "mass", "dose"):
        total_values = [0.0] * size
        total_variances = [0.0] * size
        for values in observables[observable_name].values():
            for index, value in enumerate(values):
                total_values[index] += float(value)
        for variances in observables[f"{observable_name}_variance"].values():
            for index, value in enumerate(variances):
                total_variances[index] += float(value)
        observables[observable_name]["Total"] = total_values
        observables[f"{observable_name}_variance"]["Total"] = total_variances

    return observables


def _accumulate_value(
    bucket: dict[str, list[float]],
    label: str,
    index: int,
    size: int,
    value: float,
) -> None:
    values = bucket.setdefault(str(label), [0.0] * size)
    values[index] += float(value)


def _accumulate_variance(
    bucket: dict[str, list[float]],
    label: str,
    index: int,
    size: int,
    variance: float,
) -> None:
    values = bucket.setdefault(str(label), [0.0] * size)
    values[index] += max(float(variance), 0.0)


def _half_life_variance(
    dataset: DecayDataset,
    nuclide: str,
    *,
    value: float,
    absolute_time_s: float,
) -> float:
    half_life_s = dataset.half_life_s(nuclide) or 0.0
    half_life_uncertainty_s = dataset.half_life_uncertainty_s(nuclide) or 0.0
    if half_life_s <= 0.0 or half_life_uncertainty_s <= 0.0 or absolute_time_s <= 0.0:
        return 0.0
    rel_uncertainty = (
        math.log(2.0)
        * abs(float(absolute_time_s))
        * half_life_uncertainty_s
        / (half_life_s * half_life_s)
    )
    return (float(value) * rel_uncertainty) ** 2


def _build_series(
    observable: str,
    unit: str,
    relative_times_s: Sequence[float],
    values_by_label: dict[str, list[float]],
    variances_by_label: dict[str, list[float]],
) -> dict[str, ObservableTimeSeries]:
    series: dict[str, ObservableTimeSeries] = {}
    for label in _ordered_value_labels(values_by_label):
        points = tuple(
            (
                float(relative_times_s[index]),
                float(values_by_label[label][index]),
                math.sqrt(max(float(variances_by_label.get(label, [0.0])[index]), 0.0)),
            )
            for index in range(len(relative_times_s))
        )
        series[label] = ObservableTimeSeries(
            label=label,
            observable=observable,
            unit=unit,
            points=points,
        )
    return series


def _ordered_value_labels(values_by_label: dict[str, list[float]]) -> list[str]:
    labels = sorted(label for label in values_by_label if label != "Total")
    if "Total" in values_by_label:
        return ["Total", *labels]
    return labels


def _ordered_series_labels(series: dict[str, ObservableTimeSeries]) -> list[str]:
    labels = sorted(label for label in series if label != "Total")
    if "Total" in series:
        return ["Total", *labels]
    return labels


def _dose_by_nuclide(
    gamma_database: GammaDatabase,
    activities_bq: dict[str, float],
    *,
    distance_cm: float,
) -> tuple[dict[str, float], dict[str, float]]:
    values: dict[str, float] = {}
    variances: dict[str, float] = {}
    for nuclide, activity_bq in activities_bq.items():
        if activity_bq <= 0.0:
            continue
        resolved = _resolve_decay_record(gamma_database, nuclide)
        if resolved is None:
            continue
        _resolved_name, decay = resolved
        lines = _dose_lines_from_decay(decay)
        if not lines:
            continue
        result = isotope_dose_rate(
            lines,
            activity_Bq=float(activity_bq),
            distance_cm=float(distance_cm),
            half_life_s=decay.halflife,
        )
        values[str(nuclide)] = float(result.dose_rate_uSv_h)
        variances[str(nuclide)] = float(result.uncertainty_uSv_h or 0.0) ** 2
    return values, variances


def _dose_lines_from_decay(decay: DecayData) -> list[DoseGammaLine]:
    lines: list[DoseGammaLine] = []
    for line in decay.gamma_lines:
        intensity = float(line.intensity * line.norm)
        if intensity <= 0.0:
            continue
        intensity_unc = 0.0
        if line.intensity > 0.0 and line.intensity_unc > 0.0:
            intensity_unc += (line.intensity_unc * line.norm) ** 2
        if line.norm > 0.0 and line.norm_unc > 0.0:
            intensity_unc += (line.intensity * line.norm_unc) ** 2
        lines.append(
            DoseGammaLine(
                energy_keV=float(line.energy_keV),
                intensity=intensity,
                energy_unc_keV=float(getattr(line, "energy_unc_keV", 0.0))
                if hasattr(line, "energy_unc_keV")
                else float(getattr(line, "energy_unc", 0.0)) / 1000.0,
                intensity_unc=math.sqrt(intensity_unc),
            )
        )
    return lines


def _candidate_nuclide_names(nuclide: str) -> tuple[str, ...]:
    raw = str(nuclide).strip()
    candidates = [raw, raw.replace("-", ""), raw.replace("_", "-")]
    try:
        element, mass, meta = parse_nndc_isotope_name(raw)
        candidates.extend(
            [
                format_isotope_name(element, mass, meta, separator=""),
                format_isotope_name(element, mass, meta, separator="-"),
            ]
        )
    except ValueError:
        pass

    normalized = normalize_nuclide_label(raw)
    candidates.extend([normalized, normalized.replace("-", "")])

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if text and text.lower() not in seen:
            ordered.append(text)
            seen.add(text.lower())
    return tuple(ordered)


def _resolve_decay_record(
    database: GammaDatabase,
    nuclide: str,
) -> tuple[str, DecayData] | None:
    lookup = {name.lower(): name for name in database.nuclides}
    for candidate in _candidate_nuclide_names(nuclide):
        record = database.get(candidate)
        if record is not None:
            return candidate, record
        matched_name = lookup.get(candidate.lower())
        if matched_name is not None:
            record = database.get(matched_name)
            if record is not None:
                return matched_name, record
    return None


def _build_family_maps(
    dataset: DecayDataset,
    roots: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    reverse_map: dict[str, set[str]] = {}
    for parent, children in dataset.progeny.items():
        for child in children:
            reverse_map.setdefault(child, set()).add(parent)

    tracked: set[str] = {normalize_nuclide_label(item) for item in roots}
    for nuclide in list(tracked):
        tracked.update(dataset.decay_products(nuclide))
        tracked.update(reverse_map.get(nuclide, set()))

    parents_by_nuclide = {
        nuclide: tuple(sorted(reverse_map.get(nuclide, set())))
        for nuclide in sorted(tracked)
    }
    daughters_by_nuclide = {
        nuclide: tuple(sorted(dataset.decay_products(nuclide)))
        for nuclide in sorted(tracked)
    }
    return parents_by_nuclide, daughters_by_nuclide


def _snapshot_rows(
    inventory_state: InventoryState,
    *,
    dataset: DecayDataset,
    observables: dict[str, dict[str, list[float]]],
    absolute_time_s: float,
    time_origin: str,
    parents_by_nuclide: dict[str, tuple[str, ...]],
    daughters_by_nuclide: dict[str, tuple[str, ...]],
    decay_source_id: str,
) -> tuple[dict[str, Any], ...]:
    seed_map = {
        normalize_nuclide_label(seed.nuclide): seed
        for seed in inventory_state.seeds
    }
    rows: list[dict[str, Any]] = []
    labels = _ordered_value_labels(observables["activity"])
    for label in labels:
        value = observables["activity"].get(label, [0.0])[0]
        if label != "Total" and value <= 0.0:
            continue
        normalized = normalize_nuclide_label(label) if label != "Total" else "Total"
        seed = seed_map.get(normalized)
        half_life_s = dataset.half_life_s(normalized) if label != "Total" else None
        half_life_uncertainty_s = (
            dataset.half_life_uncertainty_s(normalized)
            if label != "Total"
            else None
        )
        rows.append(
            {
                "sample_id": inventory_state.sample_id,
                "reference_state": time_origin,
                "absolute_time_s": float(absolute_time_s),
                "count_start_time_s": inventory_state.schedule.count_start_time_s,
                "count_live_time_s": inventory_state.schedule.count_live_time_s,
                "gamma_source_id": inventory_state.gamma_source_id,
                "decay_source_id": decay_source_id,
                "nuclide": label,
                "line_count": seed.line_count if seed is not None else 0,
                "activity_bq": float(value),
                "activity_uncertainty_bq": math.sqrt(
                    max(observables["activity_variance"].get(label, [0.0])[0], 0.0)
                ),
                "atoms": float(observables["atoms"].get(label, [0.0])[0]),
                "atoms_uncertainty": math.sqrt(
                    max(observables["atoms_variance"].get(label, [0.0])[0], 0.0)
                ),
                "mass_g": float(observables["mass"].get(label, [0.0])[0]),
                "mass_uncertainty_g": math.sqrt(
                    max(observables["mass_variance"].get(label, [0.0])[0], 0.0)
                ),
                "dose_rate_uSv_h": float(observables["dose"].get(label, [0.0])[0]),
                "dose_rate_uncertainty_uSv_h": math.sqrt(
                    max(observables["dose_variance"].get(label, [0.0])[0], 0.0)
                ),
                "half_life_s": float(half_life_s) if half_life_s is not None else None,
                "half_life_uncertainty_s": (
                    float(half_life_uncertainty_s)
                    if half_life_uncertainty_s is not None
                    else None
                ),
                "immediate_parents": ", ".join(parents_by_nuclide.get(normalized, ())),
                "immediate_daughters": ", ".join(
                    daughters_by_nuclide.get(normalized, ())
                ),
                "provenance_tag": "inventory_time_evolution",
            }
        )
    return tuple(rows)


__all__ = [
    "DEFAULT_DECAY_SOURCE_ID",
    "IrradiationSchedule",
    "InventorySeed",
    "InventoryState",
    "InventoryTimeEvolutionResult",
    "ObservableTimeSeries",
    "TIME_ORIGINS",
    "build_inventory_state_from_activity_results",
    "build_inventory_state_from_activity_review",
    "build_inventory_state_from_payload",
    "build_time_grid",
    "compute_inventory_time_evolution",
]
