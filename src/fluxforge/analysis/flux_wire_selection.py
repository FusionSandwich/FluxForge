"""
Flux Wire Selection Advisor

Provides guidance for selecting optimal flux wire combinations based on:
- IRDFF-II cross section library
- Threshold energies for neutron reactions
- Sensitivity to a priori spectrum uncertainty
- Recommendations from INL/EXT-21-64191 (Holschuh et al., 2021)

References:
    T. Holschuh et al., "Impact of Flux Wire Selection on Neutron Spectrum 
    Adjustment", INL/EXT-21-64191, August 2021.
    
    ASTM E720, "Standard Guide for Selection and Use of Neutron Sensors"
    
    A. Trkov et al., "IRDFF-II: A New Neutron Metrology Library", 
    Nuclear Data Sheets 163, 1-108 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Flux Wire Database (from IRDFF-II and INL recommendations)
# =============================================================================

class WireCategory(Enum):
    """Categories for flux wire applications."""
    THERMAL = auto()
    EPITHERMAL = auto()
    FAST_LOW = auto()     # 0.1 - 1 MeV
    FAST_MID = auto()     # 1 - 5 MeV
    FAST_HIGH = auto()    # 5 - 20 MeV
    FISSION = auto()


@dataclass
class FluxWireReaction:
    """
    Complete metadata for a flux wire reaction.
    
    Attributes
    ----------
    target : str
        Target nuclide (e.g., 'Ti-46')
    product : str
        Product nuclide (e.g., 'Sc-46')
    reaction_type : str
        Reaction type (e.g., 'n,p')
    threshold_MeV : float
        Threshold energy in MeV (0 for thermal reactions)
    half_life_s : float
        Product half-life in seconds
    gamma_energy_keV : float
        Primary gamma energy for detection
    gamma_intensity : float
        Gamma intensity (0-1)
    thermal_xs_barn : float
        Thermal cross section at 0.0253 eV (barns)
    category : WireCategory
        Energy sensitivity category
    recommended_combos : List[str]
        Recommended combination partners (from INL study)
    notes : str
        Special considerations
    """
    target: str
    product: str
    reaction_type: str
    threshold_MeV: float = 0.0
    half_life_s: float = 0.0
    gamma_energy_keV: float = 0.0
    gamma_intensity: float = 1.0
    thermal_xs_barn: float = 0.0
    category: WireCategory = WireCategory.THERMAL
    recommended_combos: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def reaction_str(self) -> str:
        """Full reaction string (e.g., 'Ti-46(n,p)Sc-46')."""
        return f"{self.target}({self.reaction_type}){self.product}"
    
    @property
    def element(self) -> str:
        """Element name from target."""
        return ''.join(c for c in self.target if c.isalpha())
    
    @property
    def is_thermal(self) -> bool:
        """True if this is a thermal/epithermal reaction."""
        return self.threshold_MeV == 0.0 or self.category in [
            WireCategory.THERMAL, WireCategory.EPITHERMAL
        ]
    
    @property
    def is_threshold(self) -> bool:
        """True if this is a threshold reaction."""
        return self.threshold_MeV > 0.0


# =============================================================================
# IRDFF-II Flux Wire Database
# =============================================================================

# INL "Good Wire Combinations" from Table 7 and Table 8 (INL/EXT-21-64191)
# Format: (Wire #2, Wire #3, Wire #4) with Ti-Fe-Co as baseline

INL_ROBUST_COMBOS = [
    # From Table 7: Good for potentially large a priori deviations
    ("Ti", "Fe", "Co", "Au"),
    ("Ti", "Fe", "Co", "As"),
    ("Ti", "Fe", "Co", "I"),
    ("Ti", "Fe", "Co", "La"),
    ("Ti", "Fe", "Co", "Nb"),
    ("Ti", "Fe", "Co", "Y"),
    ("Ti", "Fe", "Co", "Zr"),
    ("Ti", "Mn", "Fe", "Co"),
]

INL_WELL_CHARACTERIZED_COMBOS = [
    # From Table 8: Good for well-characterized a priori spectra
    # Includes all of Table 7 plus additional combinations
    ("Co", "Au"),
    ("Co", "Ag"),
    ("Co", "Mn"),
    ("Co", "Sc"),
]


FLUX_WIRE_DATABASE: Dict[str, FluxWireReaction] = {
    # ==========================================================================
    # Thermal/Epithermal Reactions (n,γ)
    # ==========================================================================
    "Co-59(n,g)Co-60": FluxWireReaction(
        target="Co-59",
        product="Co-60",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=1.664e8,  # 5.27 years
        gamma_energy_keV=1332.5,
        gamma_intensity=0.9998,
        thermal_xs_barn=37.18,
        category=WireCategory.THERMAL,
        recommended_combos=["Ti", "Fe", "Ni", "Au"],
        notes="Primary thermal monitor. Strong thermal cross section.",
    ),
    "Au-197(n,g)Au-198": FluxWireReaction(
        target="Au-197",
        product="Au-198",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=2.695 * 86400,  # 2.695 days
        gamma_energy_keV=411.8,
        gamma_intensity=0.9562,
        thermal_xs_barn=98.65,
        category=WireCategory.THERMAL,
        recommended_combos=["Co", "Ti", "Fe"],
        notes="Strong epithermal resonances at 4.9 eV.",
    ),
    "Sc-45(n,g)Sc-46": FluxWireReaction(
        target="Sc-45",
        product="Sc-46",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=83.79 * 86400,  # 83.79 days
        gamma_energy_keV=889.3,
        gamma_intensity=0.9998,
        thermal_xs_barn=27.2,
        category=WireCategory.THERMAL,
        recommended_combos=["Co", "Cu"],
        notes="Good for long irradiations due to long half-life.",
    ),
    "In-115(n,g)In-116m": FluxWireReaction(
        target="In-115",
        product="In-116m",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=54.29 * 60,  # 54.29 min
        gamma_energy_keV=1293.6,
        gamma_intensity=0.848,
        thermal_xs_barn=162.3,
        category=WireCategory.EPITHERMAL,
        recommended_combos=["Au", "Co"],
        notes="Strong 1.457 eV resonance. Best for epithermal flux.",
    ),
    "Mn-55(n,g)Mn-56": FluxWireReaction(
        target="Mn-55",
        product="Mn-56",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=2.5789 * 3600,  # 2.58 hours
        gamma_energy_keV=846.8,
        gamma_intensity=0.989,
        thermal_xs_barn=13.3,
        category=WireCategory.THERMAL,
        recommended_combos=["Ti", "Fe", "Co"],
        notes="Short half-life enables quick turnaround.",
    ),
    "Cu-63(n,g)Cu-64": FluxWireReaction(
        target="Cu-63",
        product="Cu-64",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=12.7 * 3600,  # 12.7 hours
        gamma_energy_keV=1345.8,
        gamma_intensity=0.0047,  # Low but usable
        thermal_xs_barn=4.50,
        category=WireCategory.THERMAL,
        recommended_combos=["Co", "Sc"],
        notes="Often used with Cd cover for Cd-ratio.",
    ),
    "Fe-58(n,g)Fe-59": FluxWireReaction(
        target="Fe-58",
        product="Fe-59",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=44.49 * 86400,  # 44.49 days
        gamma_energy_keV=1099.2,
        gamma_intensity=0.565,
        thermal_xs_barn=1.28,
        category=WireCategory.THERMAL,
        recommended_combos=["Ti", "Co", "Ni"],
        notes="Low abundance (0.28%) but good for long irradiations.",
    ),
    "Na-23(n,g)Na-24": FluxWireReaction(
        target="Na-23",
        product="Na-24",
        reaction_type="n,g",
        threshold_MeV=0.0,
        half_life_s=14.96 * 3600,  # 14.96 hours
        gamma_energy_keV=1368.6,
        gamma_intensity=0.9999,
        thermal_xs_barn=0.530,
        category=WireCategory.THERMAL,
        recommended_combos=["Co", "Mn"],
        notes="Very clean gamma signature.",
    ),
    
    # ==========================================================================
    # Fast Threshold Reactions - (n,p)
    # ==========================================================================
    "Ti-46(n,p)Sc-46": FluxWireReaction(
        target="Ti-46",
        product="Sc-46",
        reaction_type="n,p",
        threshold_MeV=1.62,
        half_life_s=83.79 * 86400,  # 83.79 days
        gamma_energy_keV=889.3,
        gamma_intensity=0.9998,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_MID,
        recommended_combos=["Fe", "Co", "Ni"],
        notes="Primary fast flux monitor. INL 'standard' wire.",
    ),
    "Ti-47(n,p)Sc-47": FluxWireReaction(
        target="Ti-47",
        product="Sc-47",
        reaction_type="n,p",
        threshold_MeV=0.22,
        half_life_s=3.349 * 86400,  # 3.349 days
        gamma_energy_keV=159.4,
        gamma_intensity=0.683,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Ti-46", "Ti-48"],
        notes="Lower threshold than Ti-46, complements Ti package.",
    ),
    "Ti-48(n,p)Sc-48": FluxWireReaction(
        target="Ti-48",
        product="Sc-48",
        reaction_type="n,p",
        threshold_MeV=3.35,
        half_life_s=43.67 * 3600,  # 43.67 hours
        gamma_energy_keV=983.5,
        gamma_intensity=1.0,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_MID,
        recommended_combos=["Ti-46", "Ti-47"],
        notes="Higher threshold, sensitive to >3 MeV spectrum.",
    ),
    "Fe-54(n,p)Mn-54": FluxWireReaction(
        target="Fe-54",
        product="Mn-54",
        reaction_type="n,p",
        threshold_MeV=0.09,
        half_life_s=312.2 * 86400,  # 312.2 days
        gamma_energy_keV=834.8,
        gamma_intensity=0.9998,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Ti", "Co", "Ni"],
        notes="Very low threshold. Long half-life good for archival.",
    ),
    "Fe-56(n,p)Mn-56": FluxWireReaction(
        target="Fe-56",
        product="Mn-56",
        reaction_type="n,p",
        threshold_MeV=2.97,
        half_life_s=2.5789 * 3600,  # 2.58 hours
        gamma_energy_keV=846.8,
        gamma_intensity=0.989,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_MID,
        recommended_combos=["Ti", "Co"],
        notes="Same product as Mn-55(n,g), requires Cd cover separation.",
    ),
    "Ni-58(n,p)Co-58": FluxWireReaction(
        target="Ni-58",
        product="Co-58",
        reaction_type="n,p",
        threshold_MeV=0.40,
        half_life_s=70.86 * 86400,  # 70.86 days
        gamma_energy_keV=810.8,
        gamma_intensity=0.994,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Ti", "Fe", "Co"],
        notes="Common fast flux monitor. Good sensitivity.",
    ),
    "S-32(n,p)P-32": FluxWireReaction(
        target="S-32",
        product="P-32",
        reaction_type="n,p",
        threshold_MeV=0.96,
        half_life_s=14.26 * 86400,  # 14.26 days
        gamma_energy_keV=0.0,  # Pure beta emitter!
        gamma_intensity=0.0,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Ti", "Ni"],
        notes="Pure beta emitter - requires liquid scintillation counting.",
    ),
    
    # ==========================================================================
    # Fast Threshold Reactions - (n,α)
    # ==========================================================================
    "Al-27(n,a)Na-24": FluxWireReaction(
        target="Al-27",
        product="Na-24",
        reaction_type="n,a",
        threshold_MeV=3.25,
        half_life_s=14.96 * 3600,  # 14.96 hours
        gamma_energy_keV=1368.6,
        gamma_intensity=0.9999,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_MID,
        recommended_combos=["Ti", "Ni", "Fe"],
        notes="Clean gamma. Same product as Na-23(n,g).",
    ),
    "Fe-54(n,a)Cr-51": FluxWireReaction(
        target="Fe-54",
        product="Cr-51",
        reaction_type="n,a",
        threshold_MeV=0.84,
        half_life_s=27.70 * 86400,  # 27.70 days
        gamma_energy_keV=320.1,
        gamma_intensity=0.0983,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Ti", "Ni"],
        notes="Low gamma intensity, requires longer counting.",
    ),
    "Co-59(n,a)Mn-56": FluxWireReaction(
        target="Co-59",
        product="Mn-56",
        reaction_type="n,a",
        threshold_MeV=5.17,
        half_life_s=2.5789 * 3600,  # 2.58 hours
        gamma_energy_keV=846.8,
        gamma_intensity=0.989,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_HIGH,
        recommended_combos=["Ti", "Al"],
        notes="High threshold. Good for >5 MeV spectrum.",
    ),
    
    # ==========================================================================
    # (n,n') and (n,2n) Reactions - High Energy
    # ==========================================================================
    "In-115(n,n')In-115m": FluxWireReaction(
        target="In-115",
        product="In-115m",
        reaction_type="n,n'",
        threshold_MeV=0.34,
        half_life_s=4.486 * 3600,  # 4.486 hours
        gamma_energy_keV=336.2,
        gamma_intensity=0.458,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_LOW,
        recommended_combos=["Au", "Co", "Ti"],
        notes="Low threshold inelastic. Good intermediate energy probe.",
    ),
    "Nb-93(n,2n)Nb-92m": FluxWireReaction(
        target="Nb-93",
        product="Nb-92m",
        reaction_type="n,2n",
        threshold_MeV=9.0,
        half_life_s=10.15 * 86400,  # 10.15 days
        gamma_energy_keV=934.4,
        gamma_intensity=0.992,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_HIGH,
        recommended_combos=["Ti", "Ni"],
        notes="Very high threshold. For fusion-like spectra.",
    ),
    "Ni-58(n,2n)Ni-57": FluxWireReaction(
        target="Ni-58",
        product="Ni-57",
        reaction_type="n,2n",
        threshold_MeV=12.4,
        half_life_s=35.6 * 3600,  # 35.6 hours
        gamma_energy_keV=1377.6,
        gamma_intensity=0.817,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_HIGH,
        recommended_combos=["Ti", "Fe"],
        notes="Very high threshold. For 14 MeV D-T neutrons.",
    ),
    "Zr-90(n,2n)Zr-89": FluxWireReaction(
        target="Zr-90",
        product="Zr-89",
        reaction_type="n,2n",
        threshold_MeV=12.0,
        half_life_s=78.41 * 3600,  # 78.41 hours
        gamma_energy_keV=909.2,
        gamma_intensity=0.9904,
        thermal_xs_barn=0.0,
        category=WireCategory.FAST_HIGH,
        recommended_combos=["Ti", "Nb", "Ni"],
        notes="For 14 MeV D-T neutron characterization.",
    ),
}


# =============================================================================
# Flux Wire Selection Advisor
# =============================================================================

@dataclass
class WireCombinationScore:
    """
    Score for a flux wire combination.
    
    Attributes
    ----------
    wires : Tuple[str, ...]
        Wire elements in combination
    energy_coverage : float
        Score for energy range coverage (0-1)
    threshold_spacing : float
        Score for threshold energy distribution (0-1)
    has_thermal : bool
        Whether combination includes thermal reaction
    has_fast : bool
        Whether combination includes fast reaction
    overall_score : float
        Combined quality score (0-1)
    recommendations : List[str]
        Specific recommendations
    """
    wires: Tuple[str, ...]
    energy_coverage: float
    threshold_spacing: float
    has_thermal: bool
    has_fast: bool
    overall_score: float
    recommendations: List[str] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)


def get_wire_reactions(element: str) -> List[FluxWireReaction]:
    """Get all reactions for a given element/wire."""
    results = []
    element_upper = element.upper()
    
    for reaction_str, rxn in FLUX_WIRE_DATABASE.items():
        if rxn.element.upper() == element_upper:
            results.append(rxn)
    
    return results


def analyze_wire_combination(
    wires: List[str],
    verbose: bool = True,
) -> WireCombinationScore:
    """
    Analyze a flux wire combination for spectrum unfolding suitability.
    
    Parameters
    ----------
    wires : List[str]
        Wire elements (e.g., ['Ti', 'Fe', 'Co'])
    verbose : bool
        Print analysis details
    
    Returns
    -------
    WireCombinationScore
        Detailed scoring and recommendations
    """
    reactions_found: List[FluxWireReaction] = []
    thresholds: List[float] = []
    categories: Set[WireCategory] = set()
    
    for wire in wires:
        wire_rxns = get_wire_reactions(wire)
        reactions_found.extend(wire_rxns)
        for rxn in wire_rxns:
            if rxn.threshold_MeV > 0:
                thresholds.append(rxn.threshold_MeV)
            categories.add(rxn.category)
    
    # Check thermal coverage
    has_thermal = WireCategory.THERMAL in categories or WireCategory.EPITHERMAL in categories
    has_fast = any(cat in categories for cat in [
        WireCategory.FAST_LOW, WireCategory.FAST_MID, WireCategory.FAST_HIGH
    ])
    
    # Energy coverage score (0-1)
    n_categories = len(categories)
    energy_coverage = min(n_categories / 5.0, 1.0)  # Max 5 categories
    
    # Threshold spacing score
    if len(thresholds) >= 2:
        thresholds_sorted = sorted(thresholds)
        log_ratios = []
        for i in range(1, len(thresholds_sorted)):
            if thresholds_sorted[i-1] > 0:
                log_ratios.append(
                    np.log10(thresholds_sorted[i] / thresholds_sorted[i-1])
                )
        # Good spacing = uniform log distribution
        if log_ratios:
            mean_ratio = np.mean(log_ratios)
            std_ratio = np.std(log_ratios)
            threshold_spacing = max(0, 1 - std_ratio / max(mean_ratio, 0.1))
        else:
            threshold_spacing = 0.5
    else:
        threshold_spacing = 0.3  # Poor threshold coverage
    
    # Overall score
    thermal_bonus = 0.2 if has_thermal else 0.0
    fast_bonus = 0.2 if has_fast else 0.0
    overall_score = 0.4 * energy_coverage + 0.2 * threshold_spacing + thermal_bonus + fast_bonus
    
    # Generate recommendations
    recommendations = []
    
    if not has_thermal:
        recommendations.append(
            "⚠️ No thermal reaction. Consider adding Co, Au, or In for thermal flux monitoring."
        )
    
    if not has_fast:
        recommendations.append(
            "⚠️ No fast threshold reactions. Consider adding Ti, Ni, or Fe for fast flux."
        )
    
    if len(wires) < 3:
        recommendations.append(
            "⚠️ Fewer than 3 wires. INL recommends at least Ti-Fe-Co as baseline."
        )
    
    if len(thresholds) < 3:
        recommendations.append(
            "Consider adding reactions with different thresholds for better energy resolution."
        )
    
    # Check if this matches INL robust combinations
    wire_set = set(w.upper() for w in wires)
    inl_baseline = {"TI", "FE", "CO"}
    if inl_baseline.issubset(wire_set):
        recommendations.append(
            "✓ Includes INL baseline {Ti, Fe, Co} - robust against a priori uncertainty."
        )
    
    return WireCombinationScore(
        wires=tuple(wires),
        energy_coverage=energy_coverage,
        threshold_spacing=threshold_spacing,
        has_thermal=has_thermal,
        has_fast=has_fast,
        overall_score=overall_score,
        recommendations=recommendations,
        reactions=[rxn.reaction_str for rxn in reactions_found],
    )


def suggest_wire_combinations(
    spectrum_type: str = "reactor",
    must_include: Optional[List[str]] = None,
    max_wires: int = 4,
    min_wires: int = 2,
) -> List[WireCombinationScore]:
    """
    Suggest optimal flux wire combinations.
    
    Parameters
    ----------
    spectrum_type : str
        'reactor' (thermal+fast), 'fast' (fast only), 'fusion' (14 MeV)
    must_include : List[str], optional
        Wires that must be included
    max_wires : int
        Maximum number of wires
    min_wires : int
        Minimum number of wires
    
    Returns
    -------
    List[WireCombinationScore]
        Ranked wire combinations
    """
    from itertools import combinations
    
    # Define wire sets based on spectrum type
    if spectrum_type == "fusion":
        available = ["Nb", "Ni", "Ti", "Fe", "Zr", "Al"]
    elif spectrum_type == "fast":
        available = ["Ti", "Ni", "Fe", "Al", "In", "S"]
    else:  # reactor
        available = ["Ti", "Fe", "Co", "Ni", "Au", "Sc", "In", "Mn", "Cu"]
    
    if must_include:
        for wire in must_include:
            if wire not in available:
                available.append(wire)
    
    # Generate combinations
    results = []
    for n in range(min_wires, min(max_wires + 1, len(available) + 1)):
        for combo in combinations(available, n):
            combo_list = list(combo)
            if must_include and not all(w in combo_list for w in must_include):
                continue
            
            score = analyze_wire_combination(combo_list, verbose=False)
            results.append(score)
    
    # Sort by overall score
    results.sort(key=lambda x: x.overall_score, reverse=True)
    
    return results[:10]  # Top 10


def recommend_wire_additions(
    current_wires: List[str],
    spectrum_type: str = "reactor",
) -> List[Tuple[str, str, float]]:
    """
    Recommend additional wires to improve a combination.
    
    Parameters
    ----------
    current_wires : List[str]
        Current wire selection
    spectrum_type : str
        Type of spectrum being measured
    
    Returns
    -------
    List[Tuple[str, str, float]]
        List of (wire, reason, score_improvement)
    """
    current_score = analyze_wire_combination(current_wires, verbose=False)
    
    # Candidate wires
    if spectrum_type == "fusion":
        candidates = ["Nb", "Ni", "Ti", "Zr", "Al", "Fe"]
    else:
        candidates = ["Ti", "Fe", "Co", "Ni", "Au", "Sc", "In", "Mn", "Cu", "Al"]
    
    recommendations = []
    
    for wire in candidates:
        if wire in current_wires:
            continue
        
        new_combo = current_wires + [wire]
        new_score = analyze_wire_combination(new_combo, verbose=False)
        
        improvement = new_score.overall_score - current_score.overall_score
        
        if improvement > 0.01:
            # Determine reason
            rxns = get_wire_reactions(wire)
            if rxns:
                primary = rxns[0]
                if primary.is_thermal:
                    reason = f"Adds thermal sensitivity ({primary.reaction_str})"
                elif primary.threshold_MeV > 5:
                    reason = f"Adds high-energy coverage ({primary.threshold_MeV:.1f} MeV threshold)"
                else:
                    reason = f"Adds {primary.threshold_MeV:.2f} MeV threshold reaction"
            else:
                reason = "Improves energy coverage"
            
            recommendations.append((wire, reason, improvement))
    
    # Sort by improvement
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    return recommendations[:5]


def print_wire_summary(reaction_str: str) -> None:
    """Print detailed information about a flux wire reaction."""
    if reaction_str not in FLUX_WIRE_DATABASE:
        print(f"Reaction {reaction_str} not found in database.")
        print("Available reactions:")
        for rxn in list(FLUX_WIRE_DATABASE.keys())[:10]:
            print(f"  - {rxn}")
        return
    
    rxn = FLUX_WIRE_DATABASE[reaction_str]
    
    print(f"\n{'='*60}")
    print(f"Reaction: {rxn.reaction_str}")
    print(f"{'='*60}")
    print(f"Target:           {rxn.target}")
    print(f"Product:          {rxn.product}")
    print(f"Type:             ({rxn.reaction_type})")
    print(f"Category:         {rxn.category.name}")
    print(f"Threshold:        {rxn.threshold_MeV:.3f} MeV")
    print(f"Half-life:        {rxn.half_life_s/3600:.2f} hours ({rxn.half_life_s/86400:.2f} days)")
    print(f"Gamma energy:     {rxn.gamma_energy_keV:.1f} keV")
    print(f"Gamma intensity:  {rxn.gamma_intensity*100:.2f}%")
    if rxn.thermal_xs_barn > 0:
        print(f"Thermal σ:        {rxn.thermal_xs_barn:.2f} barn")
    print(f"Recommended with: {', '.join(rxn.recommended_combos)}")
    if rxn.notes:
        print(f"Notes:            {rxn.notes}")


# =============================================================================
# ASTM E722 1-MeV Silicon Equivalent Fluence
# =============================================================================

def calculate_1mev_equivalent_fluence(
    energy_edges_MeV: np.ndarray,
    flux: np.ndarray,
    damage_function: str = "kerma_si",
) -> Tuple[float, float]:
    """
    Calculate 1-MeV silicon equivalent fluence per ASTM E722.
    
    The 1-MeV equivalent fluence is a single-number metric that represents
    the fluence of 1-MeV neutrons that would cause the same damage as the
    actual neutron spectrum.
    
    Parameters
    ----------
    energy_edges_MeV : np.ndarray
        Energy group boundaries in MeV
    flux : np.ndarray
        Flux per energy group (n/cm²/s or n/cm² for fluence)
    damage_function : str
        Damage function to use: 'kerma_si', 'niel_si', 'dpa_fe'
    
    Returns
    -------
    fluence_1mev : float
        1-MeV equivalent fluence
    hardness_parameter : float
        Spectral hardness H = φ(>1MeV) / φ(total)
    
    References
    ----------
    ASTM E722-19: Standard Practice for Characterizing Neutron Fluence 
    Spectra in Terms of an Equivalent Monoenergetic Neutron Fluence
    
    K. R. DePriest, "Historical Examination of the ASTM Standard E722 
    1-MeV Silicon Equivalent Fluence Metric", SAND2019-15194
    """
    # Simplified damage function (normalized to 1 at 1 MeV)
    # Real implementation would use tabulated NIEL or KERMA data
    
    energy_centers_MeV = np.sqrt(energy_edges_MeV[:-1] * energy_edges_MeV[1:])
    energy_widths_MeV = energy_edges_MeV[1:] - energy_edges_MeV[:-1]
    
    if damage_function == "kerma_si":
        # Silicon KERMA damage function (approximate)
        # Normalized to 1.0 at 1 MeV
        damage = _kerma_silicon(energy_centers_MeV)
    elif damage_function == "niel_si":
        # Non-Ionizing Energy Loss for silicon
        damage = _niel_silicon(energy_centers_MeV)
    elif damage_function == "dpa_fe":
        # DPA for iron (for structural materials)
        damage = _dpa_iron(energy_centers_MeV)
    else:
        raise ValueError(f"Unknown damage function: {damage_function}")
    
    # Calculate weighted fluence
    # φ_eq = Σ φ(E) × D(E) / D(1 MeV)
    damage_at_1mev = np.interp(1.0, energy_centers_MeV, damage)
    if damage_at_1mev == 0:
        damage_at_1mev = 1.0
    
    damage_normalized = damage / damage_at_1mev
    
    # Integrate: sum of flux × damage function
    fluence_1mev = np.sum(flux * energy_widths_MeV * damage_normalized)
    
    # Calculate spectral hardness
    total_flux = np.sum(flux * energy_widths_MeV)
    fast_mask = energy_centers_MeV > 1.0
    fast_flux = np.sum(flux[fast_mask] * energy_widths_MeV[fast_mask])
    
    hardness = fast_flux / total_flux if total_flux > 0 else 0.0
    
    return fluence_1mev, hardness


def _kerma_silicon(energy_MeV: np.ndarray) -> np.ndarray:
    """
    Approximate silicon KERMA damage function.
    
    Based on ASTM E722 and IRDF data.
    """
    # Simplified piece-wise approximation
    # Real data from IRDFF or NJOY processing
    damage = np.zeros_like(energy_MeV)
    
    # Thermal region (< 0.1 eV in MeV = 1e-7)
    thermal = energy_MeV < 1e-7
    damage[thermal] = 0.005 * (energy_MeV[thermal] / 1e-7)
    
    # Epithermal/intermediate (1e-7 to 0.1 MeV)
    intermediate = (energy_MeV >= 1e-7) & (energy_MeV < 0.1)
    damage[intermediate] = 0.005 + 0.5 * np.log10(energy_MeV[intermediate] / 1e-7) / 6
    
    # Fast region (0.1 to 20 MeV) - roughly linear in log scale
    fast = (energy_MeV >= 0.1) & (energy_MeV <= 20)
    damage[fast] = 0.5 + 0.5 * np.log10(energy_MeV[fast] / 0.1) / np.log10(10)
    # Normalize so damage(1 MeV) = 1
    damage[fast] = (energy_MeV[fast] / 1.0) ** 0.3
    
    # Very high energy (> 20 MeV)
    high = energy_MeV > 20
    damage[high] = (20.0 / 1.0) ** 0.3 * (energy_MeV[high] / 20.0) ** 0.1
    
    # Ensure damage(1 MeV) = 1.0
    return damage


def _niel_silicon(energy_MeV: np.ndarray) -> np.ndarray:
    """
    Approximate silicon NIEL (Non-Ionizing Energy Loss) damage function.
    
    NIEL is more relevant for semiconductor damage.
    """
    # Similar shape to KERMA but different scaling
    # Based on Messenger & Burke data
    damage = np.zeros_like(energy_MeV)
    
    # Low energy (threshold effects)
    low = energy_MeV < 0.185  # Si displacement threshold ~ 185 keV
    damage[low] = 0.0
    
    # Above threshold
    above = energy_MeV >= 0.185
    damage[above] = (energy_MeV[above] / 1.0) ** 0.5
    
    return damage


def _dpa_iron(energy_MeV: np.ndarray) -> np.ndarray:
    """
    Approximate iron DPA (displacements per atom) damage function.
    
    For structural steel damage calculations.
    """
    # Simplified NRT-based approximation
    damage = np.zeros_like(energy_MeV)
    
    # Iron displacement energy ~ 40 eV, so threshold ~ 0.3 keV PKA
    threshold = 0.0003  # MeV
    
    low = energy_MeV < threshold
    damage[low] = 0.0
    
    above = energy_MeV >= threshold
    # Lindhard partition function approximation
    damage[above] = (energy_MeV[above] / 1.0) ** 0.4
    
    return damage


def calculate_dpa(
    energy_edges_MeV: np.ndarray,
    fluence: np.ndarray,
    material: str = "Fe",
) -> float:
    """
    Calculate displacements per atom (DPA) for a material.
    
    Parameters
    ----------
    energy_edges_MeV : np.ndarray
        Energy group boundaries in MeV
    fluence : np.ndarray
        Neutron fluence per energy group (n/cm²)
    material : str
        Target material ('Fe', 'Si', 'Graphite')
    
    Returns
    -------
    float
        DPA value
    
    References
    ----------
    ASTM E693: Standard Practice for Characterizing Neutron Exposures
    in Iron and Low Alloy Steels in Terms of DPA
    """
    energy_centers = np.sqrt(energy_edges_MeV[:-1] * energy_edges_MeV[1:])
    energy_widths = energy_edges_MeV[1:] - energy_edges_MeV[:-1]
    
    # DPA cross sections (simplified - should use IRDFF or NJOY data)
    if material.upper() == "FE":
        dpa_xs = _dpa_iron(energy_centers) * 500  # ~500 barn-equivalent at 1 MeV
    elif material.upper() == "SI":
        dpa_xs = _niel_silicon(energy_centers) * 300
    else:
        raise ValueError(f"Unknown material: {material}")
    
    # Convert to barn-cm² units and integrate
    # DPA = Σ σ_DPA(E) × φ(E) × ΔE
    dpa = np.sum(dpa_xs * 1e-24 * fluence * energy_widths)
    
    return dpa


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "WireCategory",
    "FluxWireReaction",
    "WireCombinationScore",
    "FLUX_WIRE_DATABASE",
    "INL_ROBUST_COMBOS",
    "INL_WELL_CHARACTERIZED_COMBOS",
    "get_wire_reactions",
    "analyze_wire_combination",
    "suggest_wire_combinations",
    "recommend_wire_additions",
    "print_wire_summary",
    "calculate_1mev_equivalent_fluence",
    "calculate_dpa",
]
