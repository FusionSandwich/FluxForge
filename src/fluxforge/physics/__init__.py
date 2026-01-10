"""FluxForge physics module."""

from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    ReactionRateEstimate,
    weighted_activity,
    irradiation_buildup_factor,
    reaction_rate_from_activity,
)
from fluxforge.physics.dose import (
    GammaLine,
    DoseRateResult,
    decay_constant,
    decay_activity,
    gamma_dose_rate,
    isotope_dose_rate,
    fluence_from_activity,
    interpolate_coefficient,
)
from fluxforge.physics.nuclides import (
    NuclideData,
    GammaLineData,
    NuclideDatabase,
    get_nuclide_database,
    get_half_life,
    get_gamma_lines,
)
from fluxforge.physics.neutron_corrections import (
    SelfShieldingResult,
    CdCoverResult,
    NeutronCorrections,
    calculate_thermal_self_shielding_factor,
    calculate_epithermal_self_shielding_factor,
    calculate_self_shielding,
    calculate_cd_ratio_correction,
    calculate_cd_ratio,
    extract_thermal_epithermal_components,
    apply_cd_cover_correction,
    calculate_all_corrections,
    THERMAL_CROSS_SECTIONS,
    RESONANCE_INTEGRALS,
    ATOMIC_WEIGHTS,
    CD_CUTOFF_ENERGY,
)

__all__ = [
    # activation
    "GammaLineMeasurement",
    "IrradiationSegment", 
    "ReactionRateEstimate",
    "weighted_activity",
    "irradiation_buildup_factor",
    "reaction_rate_from_activity",
    # dose
    "GammaLine",
    "DoseRateResult",
    "decay_constant",
    "decay_activity",
    "gamma_dose_rate",
    "isotope_dose_rate",
    "fluence_from_activity",
    "interpolate_coefficient",
    # nuclides
    "NuclideData",
    "GammaLineData",
    "NuclideDatabase",
    "get_nuclide_database",
    "get_half_life",
    "get_gamma_lines",
    # neutron corrections
    "SelfShieldingResult",
    "CdCoverResult",
    "NeutronCorrections",
    "calculate_thermal_self_shielding_factor",
    "calculate_epithermal_self_shielding_factor",
    "calculate_self_shielding",
    "calculate_cd_ratio_correction",
    "calculate_cd_ratio",
    "extract_thermal_epithermal_components",
    "apply_cd_cover_correction",
    "calculate_all_corrections",
    "THERMAL_CROSS_SECTIONS",
    "RESONANCE_INTEGRALS",
    "ATOMIC_WEIGHTS",
    "CD_CUTOFF_ENERGY",
]
