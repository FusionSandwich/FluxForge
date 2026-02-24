"""
Inventory-based decay calculations with unit conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg

from fluxforge.physics.decay_library import DecayDataset, normalize_nuclide_label


AVOGADRO = 6.02214076e23

_ACTIVITY_UNITS = {
    "bq": 1.0,
    "ci": 3.7e10,
    "dpm": 1.0 / 60.0,
}

_MASS_UNITS = {
    "g": 1.0,
    "kg": 1000.0,
    "mg": 1e-3,
}

_MOLE_UNITS = {
    "mol": 1.0,
    "kmol": 1e3,
}


def _unit_category(unit: str) -> str:
    unit = unit.lower()
    if unit in _ACTIVITY_UNITS:
        return "activity"
    if unit in _MASS_UNITS:
        return "mass"
    if unit in _MOLE_UNITS:
        return "mole"
    if unit in ("num", "atoms"):
        return "atoms"
    raise ValueError(f"Unknown unit: {unit}")


def _convert_activity_to_atoms(activity: float, half_life_s: float, unit: str) -> float:
    if half_life_s <= 0:
        return 0.0
    activity_bq = activity * _ACTIVITY_UNITS[unit.lower()]
    return activity_bq / (np.log(2) / half_life_s)


def _convert_mass_to_atoms(mass: float, atomic_mass: float, unit: str) -> float:
    mass_g = mass * _MASS_UNITS[unit.lower()]
    if atomic_mass <= 0:
        return 0.0
    return (mass_g / atomic_mass) * AVOGADRO


def _convert_moles_to_atoms(moles: float, unit: str) -> float:
    return moles * _MOLE_UNITS[unit.lower()] * AVOGADRO


@dataclass
class CountObservation:
    """Observed decays for a nuclide over a time interval."""

    nuclide: str
    start: float
    stop: float
    decays: float
    uncertainty: float


@dataclass
class ProductionSegment:
    """Constant production segment for multiple nuclides."""

    duration: float
    rates: Dict[str, float]


@dataclass
class DecayInventory:
    """
    Nuclide inventory stored as atoms with unit conversion helpers.
    """

    atoms: Dict[str, float] = field(default_factory=dict)
    dataset: Optional[DecayDataset] = None

    @classmethod
    def from_quantities(
        cls,
        quantities: Dict[str, float],
        unit: str = "bq",
        dataset: Optional[DecayDataset] = None,
    ) -> "DecayInventory":
        unit = unit.lower()
        category = _unit_category(unit)
        if dataset is None:
            raise ValueError("DecayDataset is required for unit conversions.")

        atoms: Dict[str, float] = {}
        for nuclide, value in quantities.items():
            canon = normalize_nuclide_label(nuclide)
            if category == "activity":
                hl = dataset.half_life_s(canon) or 0.0
                atoms[canon] = _convert_activity_to_atoms(value, hl, unit)
            elif category == "mass":
                atomic_mass = dataset.atomic_mass(canon) or 0.0
                atoms[canon] = _convert_mass_to_atoms(value, atomic_mass, unit)
            elif category == "mole":
                atoms[canon] = _convert_moles_to_atoms(value, unit)
            elif category == "atoms":
                atoms[canon] = float(value)

        return cls(atoms=atoms, dataset=dataset)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Tuple[str, float, Optional[str]]],
        default_unit: str = "bq",
        dataset: Optional[DecayDataset] = None,
    ) -> "DecayInventory":
        if dataset is None:
            raise ValueError("DecayDataset is required for unit conversions.")

        atoms: Dict[str, float] = {}
        for nuclide, value, unit in rows:
            unit_final = (unit or default_unit).lower()
            category = _unit_category(unit_final)
            canon = normalize_nuclide_label(nuclide)
            if category == "activity":
                hl = dataset.half_life_s(canon) or 0.0
                atoms[canon] = atoms.get(canon, 0.0) + _convert_activity_to_atoms(value, hl, unit_final)
            elif category == "mass":
                atomic_mass = dataset.atomic_mass(canon) or 0.0
                atoms[canon] = atoms.get(canon, 0.0) + _convert_mass_to_atoms(value, atomic_mass, unit_final)
            elif category == "mole":
                atoms[canon] = atoms.get(canon, 0.0) + _convert_moles_to_atoms(value, unit_final)
            elif category == "atoms":
                atoms[canon] = atoms.get(canon, 0.0) + float(value)

        return cls(atoms=atoms, dataset=dataset)

    def activities(self, unit: str = "bq") -> Dict[str, float]:
        unit = unit.lower()
        factor = _ACTIVITY_UNITS.get(unit, 1.0)
        activities: Dict[str, float] = {}
        if self.dataset is None:
            return activities
        for nuclide, atoms in self.atoms.items():
            hl = self.dataset.half_life_s(nuclide) or 0.0
            if hl <= 0:
                activities[nuclide] = 0.0
            else:
                lam = np.log(2) / hl
                activities[nuclide] = atoms * lam / factor
        return activities

    def masses(self, unit: str = "g") -> Dict[str, float]:
        unit = unit.lower()
        factor = _MASS_UNITS.get(unit, 1.0)
        masses: Dict[str, float] = {}
        if self.dataset is None:
            return masses
        for nuclide, atoms in self.atoms.items():
            atomic_mass = self.dataset.atomic_mass(nuclide) or 0.0
            masses[nuclide] = (atoms / AVOGADRO) * atomic_mass / factor
        return masses

    def moles(self, unit: str = "mol") -> Dict[str, float]:
        unit = unit.lower()
        factor = _MOLE_UNITS.get(unit, 1.0)
        return {nuclide: (atoms / AVOGADRO) / factor for nuclide, atoms in self.atoms.items()}

    def numbers(self) -> Dict[str, float]:
        return dict(self.atoms)

    def activity_fractions(self) -> Dict[str, float]:
        activities = self.activities("bq")
        total = sum(activities.values())
        if total == 0:
            return {k: 0.0 for k in activities}
        return {k: v / total for k, v in activities.items()}

    def mass_fractions(self) -> Dict[str, float]:
        masses = self.masses("g")
        total = sum(masses.values())
        if total == 0:
            return {k: 0.0 for k in masses}
        return {k: v / total for k, v in masses.items()}

    def mole_fractions(self) -> Dict[str, float]:
        moles = self.moles("mol")
        total = sum(moles.values())
        if total == 0:
            return {k: 0.0 for k in moles}
        return {k: v / total for k, v in moles.items()}

    def decay(
        self,
        time: float,
        units: str = "s",
    ) -> "DecayInventory":
        if self.dataset is None:
            raise ValueError("DecayDataset is required for decay calculations.")
        network = DecayNetwork.from_roots(self.atoms.keys(), self.dataset)
        atoms_t = network.evolve(self.atoms, time, units=units)
        return DecayInventory(atoms=atoms_t, dataset=self.dataset)

    def cumulative_decays(
        self,
        time: float,
        units: str = "s",
        n_points: int = 200,
    ) -> Dict[str, float]:
        if self.dataset is None:
            raise ValueError("DecayDataset is required for decay calculations.")
        network = DecayNetwork.from_roots(self.atoms.keys(), self.dataset)
        unit_factor = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "y": 365.2422 * 86400.0,
        }.get(units, 1.0)
        times = np.linspace(0.0, time, n_points)
        activities = network.activity_series(self.atoms, times, units=units)
        times_s = times * unit_factor
        totals: Dict[str, float] = {}
        for nuclide, values in activities.items():
            totals[nuclide] = float(np.trapz(values, times_s))
        return totals


@dataclass
class DecayNetwork:
    """Decay network built from a subset of nuclides."""

    nuclides: List[str]
    dataset: DecayDataset

    def _transition_matrix(self) -> np.ndarray:
        n = len(self.nuclides)
        M = np.zeros((n, n))
        for i, parent in enumerate(self.nuclides):
            hl = self.dataset.half_life_s(parent) or 0.0
            lam = np.log(2) / hl if hl > 0 else 0.0
            M[i, i] = -lam
            products = self.dataset.decay_products(parent)
            branches = self.dataset.branching_fractions(parent)
            for product, br in zip(products, branches):
                if product in self.nuclides:
                    j = self.nuclides.index(product)
                    M[j, i] += lam * br
        return M

    def _vector_from_atoms(self, atoms: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.nuclides))
        for i, nuclide in enumerate(self.nuclides):
            vec[i] = atoms.get(nuclide, 0.0)
        return vec

    def _atoms_from_vector(self, vec: np.ndarray) -> Dict[str, float]:
        return {nuclide: float(vec[i]) for i, nuclide in enumerate(self.nuclides)}

    @classmethod
    def from_roots(
        cls,
        roots: Iterable[str],
        dataset: DecayDataset,
        max_depth: int = 20,
    ) -> "DecayNetwork":
        seen: List[str] = []
        queue: List[Tuple[str, int]] = [(normalize_nuclide_label(r), 0) for r in roots]
        while queue:
            nuclide, depth = queue.pop(0)
            if nuclide in seen:
                continue
            seen.append(nuclide)
            if depth >= max_depth:
                continue
            for child in dataset.decay_products(nuclide):
                queue.append((child, depth + 1))
        return cls(nuclides=seen, dataset=dataset)

    def evolve(
        self,
        atoms: Dict[str, float],
        time: float,
        units: str = "s",
    ) -> Dict[str, float]:
        unit_factor = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "y": 365.2422 * 86400.0,
        }.get(units, 1.0)
        t_sec = time * unit_factor
        M = self._transition_matrix()
        N0 = self._vector_from_atoms(atoms)
        Nt = linalg.expm(M * t_sec) @ N0
        return self._atoms_from_vector(np.maximum(Nt, 0.0))

    def activity_series(
        self,
        atoms: Dict[str, float],
        times: np.ndarray,
        units: str = "s",
    ) -> Dict[str, np.ndarray]:
        unit_factor = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "y": 365.2422 * 86400.0,
        }.get(units, 1.0)
        M = self._transition_matrix()
        N0 = self._vector_from_atoms(atoms)

        activities: Dict[str, np.ndarray] = {nuclide: np.zeros_like(times, dtype=float) for nuclide in self.nuclides}

        for idx, t in enumerate(times):
            Nt = linalg.expm(M * (t * unit_factor)) @ N0
            for i, nuclide in enumerate(self.nuclides):
                hl = self.dataset.half_life_s(nuclide) or 0.0
                lam = np.log(2) / hl if hl > 0 else 0.0
                activities[nuclide][idx] = Nt[i] * lam

        return activities


def schedule_from_rates(
    rates: Dict[str, List[Tuple[float, float]]],
    timestamp: bool = True,
) -> List[ProductionSegment]:
    """
    Build production segments from per-nuclide rate tables.

    rates: {nuclide: [(rate, time), ...]} where time is end time if timestamp=True.
    """
    nuclide_keys = [normalize_nuclide_label(k) for k in rates]
    num_segments = len(next(iter(rates.values())))
    segments: List[ProductionSegment] = []

    last_time = 0.0
    for idx in range(num_segments):
        segment_rates: Dict[str, float] = {}
        current_time = None
        for nuclide in rates:
            rate, t_val = rates[nuclide][idx]
            segment_rates[normalize_nuclide_label(nuclide)] = float(rate)
            if current_time is None:
                current_time = float(t_val)
        if current_time is None:
            continue
        duration = current_time - last_time if timestamp else current_time
        segments.append(ProductionSegment(duration=duration, rates=segment_rates))
        last_time = current_time if timestamp else last_time + duration

    return segments


def evolve_with_schedule(
    network: DecayNetwork,
    atoms: Dict[str, float],
    schedule: List[ProductionSegment],
    units: str = "s",
) -> Dict[str, float]:
    """
    Evolve inventory through a piecewise-constant production schedule.
    """
    unit_factor = {
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
        "y": 365.2422 * 86400.0,
    }.get(units, 1.0)

    current_atoms = dict(atoms)
    M = network._transition_matrix()
    n = len(network.nuclides)
    I = np.eye(n)

    for segment in schedule:
        duration_s = segment.duration * unit_factor
        N0 = network._vector_from_atoms(current_atoms)
        P = np.zeros(n)
        for nuclide, rate in segment.rates.items():
            if nuclide in network.nuclides:
                idx = network.nuclides.index(nuclide)
                P[idx] = rate

        expMt = linalg.expm(M * duration_s)
        Nt = expMt @ N0
        if np.any(P > 0):
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            Nt += M_inv @ (expMt - I) @ P
        current_atoms = network._atoms_from_vector(np.maximum(Nt, 0.0))

    return current_atoms


def fit_schedule_scale(
    network: DecayNetwork,
    atoms: Dict[str, float],
    schedule: List[ProductionSegment],
    observations: List[CountObservation],
    units: str = "s",
) -> Dict[str, float]:
    """
    Fit scale factors per nuclide to match observed decays.
    """
    from scipy.optimize import least_squares

    nuclides = sorted({normalize_nuclide_label(o.nuclide) for o in observations})
    nuclide_index = {n: i for i, n in enumerate(nuclides)}

    def _scaled_schedule(scales: np.ndarray) -> List[ProductionSegment]:
        scaled_segments: List[ProductionSegment] = []
        for segment in schedule:
            rates = {}
            for nuclide, rate in segment.rates.items():
                if nuclide in nuclide_index:
                    rates[nuclide] = rate * scales[nuclide_index[nuclide]]
                else:
                    rates[nuclide] = rate
            scaled_segments.append(ProductionSegment(duration=segment.duration, rates=rates))
        return scaled_segments

    def _predict_decays(scales: np.ndarray) -> np.ndarray:
        scaled = _scaled_schedule(scales)
        atoms_end = evolve_with_schedule(network, atoms, scaled, units=units)
        unit_factor = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "y": 365.2422 * 86400.0,
        }.get(units, 1.0)
        residuals = []
        for obs in observations:
            canon = normalize_nuclide_label(obs.nuclide)
            hl = network.dataset.half_life_s(canon) or 0.0
            if hl <= 0:
                predicted = 0.0
            else:
                lam = np.log(2) / hl
                n0 = atoms_end.get(canon, 0.0)
                start_s = obs.start * unit_factor
                stop_s = obs.stop * unit_factor
                predicted = n0 * (np.exp(-lam * start_s) - np.exp(-lam * stop_s))
            residuals.append((predicted - obs.decays) / max(obs.uncertainty, 1.0))
        return np.array(residuals)

    result = least_squares(_predict_decays, x0=np.ones(len(nuclides)))
    return {nuclide: float(result.x[idx]) for nuclide, idx in nuclide_index.items()}
