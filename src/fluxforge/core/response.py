"""Response matrix construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from fluxforge.core.linalg import Matrix, Vector, matmul


@dataclass
class EnergyGroupStructure:
    """Defines the energy bin boundaries for groupwise calculations."""

    boundaries_eV: List[float]

    def __post_init__(self) -> None:
        if any(b2 <= b1 for b1, b2 in zip(self.boundaries_eV, self.boundaries_eV[1:])):
            raise ValueError("Energy boundaries must be strictly increasing.")

    @property
    def group_count(self) -> int:
        return len(self.boundaries_eV) - 1


@dataclass
class ReactionCrossSection:
    """Stores groupwise cross sections for a reaction."""

    reaction_id: str
    sigma_g: Vector


@dataclass
class ResponseMatrix:
    """Container for the response matrix and metadata."""

    matrix: Matrix
    reactions: List[str]
    energy_groups: EnergyGroupStructure


def build_response_matrix(
    reactions: Iterable[ReactionCrossSection],
    groups: EnergyGroupStructure,
    target_number_densities: Iterable[float],
) -> ResponseMatrix:
    reaction_list = list(reactions)
    number_densities = list(target_number_densities)
    if len(reaction_list) != len(number_densities):
        raise ValueError("Cross section list and number densities must align.")

    rows: Matrix = []
    reaction_ids: List[str] = []
    for rx, n_density in zip(reaction_list, number_densities):
        if len(rx.sigma_g) != groups.group_count:
            raise ValueError("Cross section group structure mismatch.")
        rows.append([n_density * sigma for sigma in rx.sigma_g])
        reaction_ids.append(rx.reaction_id)
    return ResponseMatrix(matrix=rows, reactions=reaction_ids, energy_groups=groups)
