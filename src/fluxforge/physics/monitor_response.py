"""Physical per-monitor response rows: reaction x cover transmission x self-shielding.

A monitor observation's response in group ``g`` is the 1/E-weighted group
average of

    sigma_eff(E) = sigma_reaction(E) * T_cover(E) * G_self(E)

where ``T_cover`` is the scalar-flux transmission of an absorbing cover and
``G_self`` is the first-flight self-shielding factor of the monitor body. The
pointwise product is formed before group collapse so resonance structure in
the reaction, the cover and the monitor total cross section interacts
correctly.

Assumptions (recorded in each row's metadata):

* Covers are thin slabs in an isotropic field: ``T = E2(N sigma d)``
  (``exp(-N sigma d)`` for a normal beam). Scattering in the cover is either
  neglected (``attenuation="disap"``, the default) or treated as removal
  (``"tot"``).
* Self-shielding uses first-flight collision probabilities for an isotropic
  incident field: ``G = (1 - <T_chord>) / (Sigma_t * l_mean)``, with exact chord
  averages for an infinite slab, infinite cylinder or sphere and
  ``l_mean = 4V/S``. Neutrons scattered inside the monitor are assumed to
  escape. The caller supplies the monitor's *total* cross section; resonance
  scattering matters (e.g. Co-59 at 132 eV).
* No flux depression of the surrounding medium is modelled.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expn

from fluxforge.data.irdff import IRDFFCrossSection, IRDFFDatabase

try:
    _trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2
    _trapezoid = np.trapz

AVOGADRO = 6.02214076e23
BARN_TO_CM2 = 1e-24

# Metal properties for common cover elements (density g/cm3, atomic mass g/mol)
COVER_ELEMENT_PROPERTIES: Dict[str, Tuple[float, float]] = {
    "Cd": (8.65, 112.414),
    "Gd": (7.90, 157.25),
    "B": (2.34, 10.811),
    "B-10": (2.16, 10.0129),
}


@dataclass(frozen=True)
class CoverLayer:
    """An absorbing cover around a monitor (e.g. Cd)."""

    material: str
    thickness_cm: float
    thickness_unc_cm: float = 0.0
    density_g_cm3: Optional[float] = None
    atomic_mass: Optional[float] = None
    attenuation: str = "disap"
    angular_model: str = "isotropic"

    def __post_init__(self) -> None:
        if self.thickness_cm <= 0 or self.thickness_unc_cm < 0:
            raise ValueError("Cover thickness must be positive; uncertainty non-negative")
        if self.attenuation not in {"disap", "tot"}:
            raise ValueError("attenuation must be 'disap' or 'tot'")
        if self.angular_model not in {"isotropic", "beam"}:
            raise ValueError("angular_model must be 'isotropic' or 'beam'")
        if (self.density_g_cm3 is None or self.atomic_mass is None) and (
            self.material not in COVER_ELEMENT_PROPERTIES
        ):
            raise ValueError(
                f"Provide density_g_cm3 and atomic_mass for cover material {self.material!r}"
            )

    @property
    def number_density_per_cm3(self) -> float:
        default_density, default_mass = COVER_ELEMENT_PROPERTIES.get(self.material, (0.0, 0.0))
        density = self.density_g_cm3 if self.density_g_cm3 is not None else default_density
        mass = self.atomic_mass if self.atomic_mass is not None else default_mass
        return density * AVOGADRO / mass

    @property
    def key(self) -> str:
        return f"{self.material}:{self.thickness_cm:.6g}cm:{self.attenuation}:{self.angular_model}"


@dataclass(frozen=True)
class MonitorShielding:
    """Monitor body geometry and total cross section for self-shielding."""

    geometry: str
    dimension_cm: float
    number_density_per_cm3: float
    total_energies_eV: Tuple[float, ...]
    total_cross_section_barn: Tuple[float, ...]
    dimension_unc_cm: float = 0.0
    total_source: str = ""

    def __post_init__(self) -> None:
        if self.geometry not in {"slab", "cylinder", "sphere"}:
            raise ValueError("geometry must be 'slab', 'cylinder' or 'sphere'")
        if self.dimension_cm <= 0 or self.number_density_per_cm3 <= 0:
            raise ValueError("Monitor dimension and number density must be positive")
        if len(self.total_energies_eV) != len(self.total_cross_section_barn) or len(
            self.total_energies_eV
        ) < 2:
            raise ValueError("Total cross section needs matching energy/value arrays")
        if not self.total_source:
            raise ValueError("Record the source of the monitor total cross section")

    @property
    def mean_chord_cm(self) -> float:
        """4V/S: slab thickness t -> 2t, cylinder diameter d -> d, sphere d -> 2d/3."""
        return {"slab": 2.0, "cylinder": 1.0, "sphere": 2.0 / 3.0}[self.geometry] * self.dimension_cm

    @property
    def key(self) -> str:
        digest = hashlib.sha256(
            np.asarray(self.total_cross_section_barn, dtype="<f8").tobytes()
        ).hexdigest()[:12]
        return f"{self.geometry}:{self.dimension_cm:.6g}cm:{digest}"


@dataclass(frozen=True)
class MonitorResponseSpec:
    """Identity and physics of one monitor observation's response row."""

    observation_id: str
    sample_id: str
    reaction: str
    cover: Optional[CoverLayer] = None
    shielding: Optional[MonitorShielding] = None

    @property
    def physics_key(self) -> Tuple[str, str, str]:
        return (
            self.reaction,
            self.cover.key if self.cover else "bare",
            self.shielding.key if self.shielding else "unshielded",
        )


@dataclass
class MonitorResponse:
    spec: MonitorResponseSpec
    group_cross_section_barn: np.ndarray
    group_uncertainty_barn: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Transmission and self-shielding kernels
# -----------------------------------------------------------------------------


def cover_transmission(optical_thickness: np.ndarray, angular_model: str = "isotropic") -> np.ndarray:
    """Scalar-flux transmission through a thin slab cover."""
    tau = np.maximum(np.asarray(optical_thickness, dtype=float), 0.0)
    if angular_model == "beam":
        return np.exp(-tau)
    out = np.ones_like(tau)
    positive = tau > 0
    out[positive] = expn(2, np.minimum(tau[positive], 700.0))
    return out


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(96)


@lru_cache(maxsize=None)
def _chord_table(geometry: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tabulate <exp(-Sigma*chord)> versus x = Sigma * l_mean for a unit body."""
    x_grid = np.concatenate([[0.0], np.geomspace(1e-7, 1e3, 1200)])
    # Map Gauss-Legendre nodes on [-1, 1] to [0, 1]
    u = 0.5 * (_GL_NODES + 1.0)
    wu = 0.5 * _GL_WEIGHTS
    if geometry == "cylinder":
        # Unit diameter (R = 0.5, l_mean = 1). Impact parameter uniform in b/R,
        # polar angle weighted by sin^2(theta); chord = 2R sqrt(1-u^2)/sin(theta).
        theta = 0.5 * np.pi * u
        wtheta = 0.5 * np.pi * wu * np.sin(theta) ** 2
        wtheta = wtheta / wtheta.sum()
        chord2d = np.sqrt(1.0 - u**2)  # 2R sqrt(1-u^2) with 2R = 1
        wb = wu / wu.sum()
        chords = chord2d[:, None] / np.sin(theta)[None, :]
        weights = wb[:, None] * wtheta[None, :]
        mean = float((weights * chords).sum())
    elif geometry == "sphere":
        # Unit diameter (l_mean = 2/3); r/R uniform in projected area.
        chords = np.sqrt(1.0 - u**2)
        weights = 2.0 * u * wu
        weights = weights / weights.sum()
        mean = float((weights * chords).sum())
    else:
        raise ValueError(geometry)
    transmission = np.array(
        [float((weights * np.exp(-(x / mean) * chords)).sum()) for x in x_grid]
    )
    return x_grid, transmission


def self_shielding_factor(macroscopic_total: np.ndarray, shielding: MonitorShielding) -> np.ndarray:
    """First-flight self-shielding factor G(E) for the monitor body."""
    sigma = np.maximum(np.asarray(macroscopic_total, dtype=float), 0.0)
    x = sigma * shielding.mean_chord_cm
    g = np.ones_like(x)
    positive = x > 1e-10
    if shielding.geometry == "slab":
        tau = 0.5 * x[positive]  # x = 2 * Sigma * t
        g[positive] = (1.0 - 2.0 * expn(3, np.minimum(tau, 700.0))) / (2.0 * tau)
        return g
    x_grid, transmission = _chord_table(shielding.geometry)
    xp = x[positive]
    inside = xp <= x_grid[-1]
    t_bar = np.empty_like(xp)
    t_bar[inside] = np.interp(xp[inside], x_grid, transmission)
    t_bar[~inside] = 0.0
    g[positive] = (1.0 - t_bar) / xp
    return g


# -----------------------------------------------------------------------------
# Row construction
# -----------------------------------------------------------------------------


def _cover_sigma(db: IRDFFDatabase, cover: CoverLayer) -> IRDFFCrossSection:
    xs = db.get_cover_cross_section(cover.material, cover.attenuation)
    if xs is None:
        raise ValueError(
            f"No evaluated {cover.material}(n,{cover.attenuation}) data; "
            "the IRDFF-II abs archive is required for cover transmission"
        )
    return xs


def _pointwise_factors(
    energies: np.ndarray,
    cover: Optional[CoverLayer],
    cover_xs: Optional[IRDFFCrossSection],
    shielding: Optional[MonitorShielding],
    cover_thickness_cm: Optional[float] = None,
    shielding_dimension_cm: Optional[float] = None,
) -> np.ndarray:
    factor = np.ones_like(energies)
    if cover is not None:
        thickness = cover.thickness_cm if cover_thickness_cm is None else cover_thickness_cm
        tau = cover.number_density_per_cm3 * cover_xs.evaluate(energies) * BARN_TO_CM2 * thickness
        factor *= cover_transmission(tau, cover.angular_model)
    if shielding is not None:
        if shielding_dimension_cm is not None and shielding_dimension_cm != shielding.dimension_cm:
            shielding = MonitorShielding(
                geometry=shielding.geometry,
                dimension_cm=shielding_dimension_cm,
                number_density_per_cm3=shielding.number_density_per_cm3,
                total_energies_eV=shielding.total_energies_eV,
                total_cross_section_barn=shielding.total_cross_section_barn,
                total_source=shielding.total_source,
            )
        sigma_t = np.interp(
            energies,
            np.asarray(shielding.total_energies_eV, dtype=float),
            np.asarray(shielding.total_cross_section_barn, dtype=float),
        )
        factor *= self_shielding_factor(
            sigma_t * BARN_TO_CM2 * shielding.number_density_per_cm3, shielding
        )
    return factor


def _collapse(
    edges: np.ndarray,
    reaction_xs: IRDFFCrossSection,
    extra_points: Sequence[np.ndarray],
    factor_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(edges) - 1
    group = np.zeros(n)
    group_unc = np.zeros(n)
    points = np.concatenate([reaction_xs.energies, *extra_points])
    for g in range(n):
        lo, hi = max(edges[g], 1e-5), edges[g + 1]
        inside = points[(points > lo) & (points < hi)]
        e = np.unique(np.concatenate([np.geomspace(lo, hi, 100), inside]))
        w = 1.0 / e
        sigma = reaction_xs.evaluate(e)
        unc = np.interp(e, reaction_xs.energies, reaction_xs.uncertainties)
        unc[sigma == 0.0] = 0.0
        f = factor_fn(e)
        denominator = _trapezoid(w, e)
        if denominator > 0:
            group[g] = _trapezoid(sigma * f * w, e) / denominator
            # Fully correlated within group (see IRDFFCrossSection.to_group_structure)
            group_unc[g] = _trapezoid(unc * f * w, e) / denominator
    return group, group_unc


def build_monitor_response(
    spec: MonitorResponseSpec,
    energy_edges_eV: np.ndarray,
    db: Optional[IRDFFDatabase] = None,
) -> MonitorResponse:
    """
    Build one physical response row (group-averaged effective cross section, b).

    Multiply by group width (for differential flux per eV) or use directly
    with group-integral flux, times 1e-24, to obtain reactions/atom/s.
    """
    db = db or IRDFFDatabase()
    edges = np.asarray(energy_edges_eV, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges_eV must be strictly increasing")
    reaction_xs = db.get_cross_section(spec.reaction)
    if reaction_xs is None:
        raise ValueError(f"No evaluated cross section for {spec.reaction}")
    if reaction_xs.is_approximation:
        raise ValueError(
            f"{spec.reaction} resolves to a built-in approximation; physical "
            "response rows require evaluated data"
        )
    cover_xs = _cover_sigma(db, spec.cover) if spec.cover else None
    extra = []
    if cover_xs is not None:
        extra.append(cover_xs.energies)
    if spec.shielding is not None:
        extra.append(np.asarray(spec.shielding.total_energies_eV, dtype=float))

    def nominal(e):
        return _pointwise_factors(e, spec.cover, cover_xs, spec.shielding)

    row, row_unc = _collapse(edges, reaction_xs, extra, nominal)

    variance = row_unc**2
    components = {"reaction_cross_section": row_unc.copy()}
    if spec.cover is not None and spec.cover.thickness_unc_cm > 0:
        shifted, _ = _collapse(
            edges, reaction_xs, extra,
            lambda e: _pointwise_factors(
                e, spec.cover, cover_xs, spec.shielding,
                cover_thickness_cm=spec.cover.thickness_cm + spec.cover.thickness_unc_cm,
            ),
        )
        components["cover_thickness"] = np.abs(shifted - row)
        variance = variance + components["cover_thickness"] ** 2
    if spec.shielding is not None and spec.shielding.dimension_unc_cm > 0:
        shifted, _ = _collapse(
            edges, reaction_xs, extra,
            lambda e: _pointwise_factors(
                e, spec.cover, cover_xs, spec.shielding,
                shielding_dimension_cm=spec.shielding.dimension_cm + spec.shielding.dimension_unc_cm,
            ),
        )
        components["monitor_dimension"] = np.abs(shifted - row)
        variance = variance + components["monitor_dimension"] ** 2

    metadata: Dict[str, object] = {
        "observation_id": spec.observation_id,
        "sample_id": spec.sample_id,
        "reaction": spec.reaction,
        "reaction_source": reaction_xs.source,
        "reaction_evaluation_key": reaction_xs.evaluation_key,
        "reaction_source_sha256": reaction_xs.source_sha256,
        "cover": None if spec.cover is None else {
            "key": spec.cover.key,
            "thickness_cm": spec.cover.thickness_cm,
            "thickness_unc_cm": spec.cover.thickness_unc_cm,
            "number_density_per_cm3": spec.cover.number_density_per_cm3,
            "data": cover_xs.evaluation_key,
            "data_sha256": cover_xs.source_sha256,
            "model": f"{'E2' if spec.cover.angular_model == 'isotropic' else 'exp'} slab transmission",
        },
        "self_shielding": None if spec.shielding is None else {
            "key": spec.shielding.key,
            "geometry": spec.shielding.geometry,
            "dimension_cm": spec.shielding.dimension_cm,
            "dimension_unc_cm": spec.shielding.dimension_unc_cm,
            "total_source": spec.shielding.total_source,
            "model": "first-flight chord average, isotropic incidence",
        },
        "uncertainty_components_barn": {k: v.tolist() for k, v in components.items()},
        "weighting": "1/E within group",
    }
    return MonitorResponse(spec, row, np.sqrt(variance), metadata)


def build_monitor_response_matrix(
    specs: Sequence[MonitorResponseSpec],
    energy_edges_eV: np.ndarray,
    db: Optional[IRDFFDatabase] = None,
) -> Tuple[np.ndarray, np.ndarray, List[MonitorResponse]]:
    """Build one row per observation; rows are never merged by reaction label."""
    ids = [spec.observation_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Observation IDs must be unique")
    db = db or IRDFFDatabase()
    rows = [build_monitor_response(spec, energy_edges_eV, db) for spec in specs]
    return (
        np.array([r.group_cross_section_barn for r in rows]),
        np.array([r.group_uncertainty_barn for r in rows]),
        rows,
    )


__all__ = [
    "COVER_ELEMENT_PROPERTIES",
    "CoverLayer",
    "MonitorResponse",
    "MonitorResponseSpec",
    "MonitorShielding",
    "build_monitor_response",
    "build_monitor_response_matrix",
    "cover_transmission",
    "self_shielding_factor",
]
