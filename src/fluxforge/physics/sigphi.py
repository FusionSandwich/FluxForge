"""
SigPhi-equivalent saturation reaction rate engine for STAYSL parity.

This module provides saturation activity and reaction rate calculations that
match STAYSL PNNL's SigPhi workflow, including:
- BCF-like flux history correction factor computation
- Saturation Activity correction type
- Saturation Activity with Sampling Decay correction type
- Multi-segment irradiation history support

Reference: PNNL-22253, STAYSL PNNL User Guide
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np


class CorrectionType(Enum):
    """Types of saturation correction pathways supported."""
    
    BCF = "bcf"  # BCF-like computation
    SATURATION_ACTIVITY = "saturation_activity"  # Standard saturation activity
    SATURATION_WITH_SAMPLING_DECAY = "saturation_with_sampling_decay"  # With sampling decay


@dataclass
class NeutronBurnupModel:
        """Parameters for neutron burnup / transmutation during irradiation.

        This models depletion of target atoms during irradiation due to neutron reactions.
        It provides a correction factor to map the measured activity (which is reduced
        by depletion) back to the equivalent saturation reaction rate per *initial*
        target atom.

        Notes:
        - This model is intended to be driven by an *estimated* flux level (e.g. prior
            spectrum or transport prediction). In a full STAYSL-like workflow this can be
            re-evaluated after unfolding.
        - Product destruction can be approximated via `sigma_product_removal_barns`.
            If omitted, only product radioactive decay is modeled.
        """

        flux_cm2_s_at_rel_power_1: float
        sigma_target_destruction_barns: float
        sigma_product_removal_barns: float = 0.0
        max_correction_factor: float = 10.0

        @property
        def sigma_target_destruction_cm2(self) -> float:
                return self.sigma_target_destruction_barns * 1e-24

        @property
        def sigma_product_removal_cm2(self) -> float:
                return self.sigma_product_removal_barns * 1e-24



@dataclass
class IrradiationHistory:
    """
    Complete irradiation history with power levels and timing.
    
    Attributes:
        segments: List of (duration_s, relative_power) tuples
        total_duration_s: Total irradiation time
        cooling_time_s: Time from end of irradiation to start of counting
        counting_time_s: Duration of gamma counting
    """
    
    segments: List[tuple[float, float]]  # (duration_s, relative_power)
    cooling_time_s: float = 0.0
    counting_time_s: float = 0.0
    
    @property
    def total_duration_s(self) -> float:
        """Total irradiation duration in seconds."""
        return sum(seg[0] for seg in self.segments)
    
    def validate(self) -> None:
        """Validate irradiation history parameters."""
        if not self.segments:
            raise ValueError("At least one irradiation segment is required")
        for i, (duration, power) in enumerate(self.segments):
            if duration <= 0:
                raise ValueError(f"Segment {i}: duration must be positive")
            if power < 0:
                raise ValueError(f"Segment {i}: power cannot be negative")


@dataclass
class SaturationRateResult:
    """
    Result of saturation reaction rate calculation.
    
    Attributes:
        R_sat: Saturated reaction rate (reactions/s at infinite irradiation)
        R_eoi: End-of-irradiation reaction rate (reactions/s)
        A_sat: Saturation activity (Bq)
        A_eoi: End-of-irradiation activity (Bq)
        A_count: Activity at counting reference time (Bq)
        flux_history_factor: BCF-like flux history correction factor
        saturation_factor: S = 1 - exp(-λ*t_irr)
        decay_factor: D = exp(-λ*t_cool)
        counting_factor: C = (1 - exp(-λ*t_count)) / (λ*t_count)
        burnup_correction: Target depletion correction factor (if applied)
        uncertainty: Propagated uncertainty in R_sat
        correction_type: Which correction pathway was used
    """
    
    R_sat: float
    R_eoi: float
    A_sat: float
    A_eoi: float
    A_count: float
    flux_history_factor: float
    saturation_factor: float
    decay_factor: float
    counting_factor: float
    burnup_correction: float = 1.0
    uncertainty: float = 0.0
    correction_type: CorrectionType = CorrectionType.SATURATION_ACTIVITY
    burnup_warning: Optional[str] = None


@dataclass
class MonitorMeasurement:
    """
    Measurement data from a flux monitor (activation foil/wire).
    
    Attributes:
        reaction_id: Unique identifier (e.g., "fe58_ng_fe59")
        activity_bq: Measured activity in Bq at count time
        activity_uncertainty: Uncertainty in activity (Bq)
        half_life_s: Product half-life in seconds
        target_atoms: Number of target atoms (N_target)
        irradiation_history: Complete irradiation/cooling/counting history
    """
    
    reaction_id: str
    activity_bq: float
    activity_uncertainty: float
    half_life_s: float
    target_atoms: float
    irradiation_history: IrradiationHistory


def decay_constant(half_life_s: float) -> float:
    """Calculate decay constant from half-life."""
    return math.log(2.0) / half_life_s


def saturation_factor(decay_const: float, irradiation_time_s: float) -> float:
    """
    Calculate saturation factor S = 1 - exp(-λ*t_irr).
    
    For long irradiations, S → 1 (saturation).
    For short irradiations, S ≈ λ*t_irr.
    """
    x = decay_const * irradiation_time_s
    if x > 50:  # Effectively saturated
        return 1.0
    elif x < 1e-10:  # Very short, use linear approximation
        return x
    else:
        return 1.0 - math.exp(-x)


def decay_factor(decay_const: float, cooling_time_s: float) -> float:
    """
    Calculate decay factor D = exp(-λ*t_cool).
    
    Accounts for decay during cooling from EOI to count start.
    """
    x = decay_const * cooling_time_s
    if x > 50:  # Decayed away
        return 0.0
    return math.exp(-x)


def counting_factor(decay_const: float, counting_time_s: float) -> float:
    """
    Calculate counting factor C = (1 - exp(-λ*t_count)) / (λ*t_count).
    
    Accounts for decay during counting interval.
    For short counts relative to half-life, C → 1.
    """
    x = decay_const * counting_time_s
    if x < 1e-10:  # Very short count
        return 1.0
    elif x > 50:  # Very long count
        return 1.0 / x
    else:
        return (1.0 - math.exp(-x)) / x


def flux_history_correction_factor(
    history: IrradiationHistory,
    decay_const: float,
) -> float:
    """
    Calculate BCF-like flux history correction factor.
    
    For multi-segment irradiation with varying power levels, this
    computes the equivalent factor that relates measured activity
    to the saturation rate.
    
    The factor accounts for:
    - Build-up during each irradiation segment
    - Decay during gaps between segments (if any)
    - Decay from segment end to EOI
    
    Reference: STAYSL PNNL, BCF methodology
    
    Args:
        history: Complete irradiation history
        decay_const: Decay constant λ = ln(2)/T_1/2
        
    Returns:
        Flux history correction factor F_BCF
    """
    total_duration = history.total_duration_s
    elapsed = 0.0
    factor = 0.0
    
    for duration_s, rel_power in history.segments:
        elapsed += duration_s
        # Build-up during this segment
        segment_buildup = rel_power * (1.0 - math.exp(-decay_const * duration_s))
        # Decay from end of this segment to EOI
        time_to_eoi = total_duration - elapsed
        decay_to_eoi = math.exp(-decay_const * time_to_eoi)
        factor += segment_buildup * decay_to_eoi
    
    return factor


def burnup_adjusted_buildup_factor(
    history: IrradiationHistory,
    decay_const: float,
    burnup: NeutronBurnupModel,
) -> float:
    """Compute a burnup-adjusted build-up factor at EOI.

    Defines a dimensionless factor F such that:
        A_eoi = N0 * R_sat * F

    where N0 is the *initial* target atoms, and R_sat is the saturation reaction rate
    per initial target atom under rel_power=1.

    When target depletion is zero, this reduces to the standard flux history
    correction factor for a multi-segment irradiation.
    """
    history.validate()

    total_duration = history.total_duration_s
    elapsed = 0.0

    # Target survival fraction relative to initial N0 at the start of each segment
    target_frac_start = 1.0
    F = 0.0

    for duration_s, rel_power in history.segments:
        if duration_s <= 0:
            continue

        elapsed += duration_s
        time_to_eoi_after_segment = total_duration - elapsed

        phi = burnup.flux_cm2_s_at_rel_power_1 * max(rel_power, 0.0)
        k_t = phi * burnup.sigma_target_destruction_cm2  # target removal rate [1/s]

        # Optional product removal (neutron destruction), modeled as additional removal
        k_p = phi * burnup.sigma_product_removal_cm2
        lambda_eff = decay_const + k_p

        # Contribution to *activity* at EOI from production during this segment.
        # Includes an explicit factor of decay_const (not lambda_eff) because activity
        # is A=decay_const*N_product. The survival of product nuclei uses lambda_eff.
        #
        # dA_eoi = decay_const * (R_sat * rel_power * target_frac(t)) * exp(-lambda_eff * (t_irr - t)) dt
        # with target_frac(t)=target_frac_start*exp(-k_t*t)
        #
        # This yields:
        # F_seg = rel_power * target_frac_start * exp(-lambda_eff*(time_to_eoi_after_segment+duration_s))
        #         * decay_const * integral_0^duration exp(-(k_t - lambda_eff)*t) dt
        if rel_power <= 0:
            # Still deplete target if phi>0? If rel_power==0, phi==0 so no depletion.
            pass

        exp_tail = math.exp(-lambda_eff * (time_to_eoi_after_segment + duration_s))

        delta = k_t - lambda_eff
        if abs(delta) < 1e-14:
            integ = duration_s
        else:
            # integral exp(-delta*t) dt
            integ = (1.0 - math.exp(-delta * duration_s)) / delta

        F += max(rel_power, 0.0) * target_frac_start * exp_tail * decay_const * integ

        # Advance target survival fraction to next segment
        if k_t > 0:
            target_frac_start *= math.exp(-k_t * duration_s)

    return F


def calculate_saturation_rate(
    measurement: MonitorMeasurement,
    correction_type: CorrectionType = CorrectionType.SATURATION_ACTIVITY,
    apply_burnup: bool = False,
    burnup_threshold: float = 0.01,  # 1% depletion threshold
    burnup_model: Optional[NeutronBurnupModel] = None,
) -> SaturationRateResult:
    """
    Calculate saturation reaction rate from monitor measurement.
    
    This is the core SigPhi-equivalent calculation that converts measured
    activity to saturation reaction rate suitable for GLS spectral adjustment.
    
    The relationship is:
        A_count = R_sat * S * D * C * N_target * (burnup correction)
        
    Where:
        R_sat = saturation reaction rate (reactions/target-atom/s at saturation)
        S = saturation factor = 1 - exp(-λ*t_irr)
        D = decay factor = exp(-λ*t_cool)
        C = counting factor = (1 - exp(-λ*t_count)) / (λ*t_count)
        N_target = number of target atoms
        
    For multi-segment histories, S is replaced by the BCF-like factor.
    
    Args:
        measurement: Monitor measurement data
        correction_type: Which correction pathway to use
        apply_burnup: Whether to apply burnup correction automatically
        burnup_threshold: Depletion threshold for auto-enabling burnup
        
    Returns:
        SaturationRateResult with all computed quantities
    """
    history = measurement.irradiation_history
    history.validate()
    
    λ = decay_constant(measurement.half_life_s)
    t_irr = history.total_duration_s
    
    # Calculate correction factors
    S = saturation_factor(λ, t_irr)
    D = decay_factor(λ, history.cooling_time_s)
    C = counting_factor(λ, history.counting_time_s)
    
    # For multi-segment, use BCF-like factor instead of simple S
    if len(history.segments) > 1 or correction_type == CorrectionType.BCF:
        F_BCF = flux_history_correction_factor(history, λ)
    else:
        F_BCF = S
    
    # Burnup correction (target depletion during irradiation)
    burnup_corr = 1.0
    burnup_warning: Optional[str] = None
    if apply_burnup:
        if burnup_model is not None:
            # Compute depletion-aware build-up factor and correct the inferred R accordingly.
            F_burn = burnup_adjusted_buildup_factor(history, λ, burnup_model)
            if F_burn > 1e-30 and F_BCF > 1e-30:
                burnup_corr = F_BCF / F_burn

            if burnup_corr > burnup_model.max_correction_factor:
                burnup_warning = (
                    f"Burnup correction factor {burnup_corr:.3g} exceeds guardrail "
                    f"({burnup_model.max_correction_factor:.3g})."
                )
        else:
            # Backward-compatible heuristic (kept only as a fallback when no burnup model is supplied).
            estimated_depletion = max(0.0, 1.0 - S)
            if estimated_depletion > burnup_threshold:
                burnup_corr = (1.0 - math.exp(-estimated_depletion)) / max(estimated_depletion, 1e-30)
    
    # Calculate activities
    A_count = measurement.activity_bq
    
    # Back-calculate to EOI
    if D > 1e-12:
        if correction_type == CorrectionType.SATURATION_WITH_SAMPLING_DECAY:
            # Account for sampling decay during counting
            A_eoi = A_count / (D * C)
        else:
            A_eoi = A_count / D
    else:
        A_eoi = 0.0  # Decayed too much
    
    # Calculate saturation activity
    if F_BCF > 1e-12:
        A_sat = A_eoi / F_BCF
    else:
        A_sat = 0.0
    
    # Calculate reaction rates per target atom
    if measurement.target_atoms > 0:
        R_sat = A_sat / measurement.target_atoms
        R_eoi = A_eoi / measurement.target_atoms
    else:
        raise ValueError("Target atoms must be positive")
    
    # Apply burnup correction to rate
    R_sat = R_sat * burnup_corr
    
    # Propagate uncertainty
    rel_unc = measurement.activity_uncertainty / max(measurement.activity_bq, 1e-12)
    unc_R_sat = R_sat * rel_unc
    
    return SaturationRateResult(
        R_sat=R_sat,
        R_eoi=R_eoi,
        A_sat=A_sat,
        A_eoi=A_eoi,
        A_count=A_count,
        flux_history_factor=F_BCF,
        saturation_factor=S,
        decay_factor=D,
        counting_factor=C,
        burnup_correction=burnup_corr,
        uncertainty=unc_R_sat,
        correction_type=correction_type,
        burnup_warning=burnup_warning,
    )


def calculate_saturation_rates_batch(
    measurements: Sequence[MonitorMeasurement],
    correction_type: CorrectionType = CorrectionType.SATURATION_ACTIVITY,
    apply_burnup: bool = False,
    burnup_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, List[SaturationRateResult]]:
    """
    Calculate saturation rates for multiple monitors.
    
    Returns the y vector and diagonal of V_y for GLS adjustment.
    
    Args:
        measurements: List of monitor measurements
        correction_type: Correction pathway for all monitors
        apply_burnup: Whether to apply burnup corrections
        burnup_threshold: Depletion threshold for burnup
        
    Returns:
        Tuple of (y_vector, y_variances, full_results)
        - y_vector: Saturation reaction rates for GLS input
        - y_variances: Diagonal variances for V_y
        - full_results: Complete SaturationRateResult for each monitor
    """
    results = []
    y_values = []
    y_vars = []
    
    for m in measurements:
        result = calculate_saturation_rate(
            m, correction_type, apply_burnup, burnup_threshold
        )
        results.append(result)
        y_values.append(result.R_sat)
        y_vars.append(result.uncertainty ** 2)
    
    return np.array(y_values), np.array(y_vars), results


@dataclass
class SaturationRatesArtifact:
    """
    Artifact containing saturated reaction rates for GLS input.
    
    This is the canonical output format that feeds into STAYSL-like
    spectral adjustment.
    """
    
    reaction_ids: List[str]
    y: np.ndarray  # Saturation rates
    V_y_diag: np.ndarray  # Diagonal of measurement covariance
    V_y: Optional[np.ndarray] = None  # Full covariance matrix (if available)
    results: List[SaturationRateResult] = field(default_factory=list)
    correction_type: CorrectionType = CorrectionType.SATURATION_ACTIVITY
    
    def to_dict(self) -> dict:
        """Export artifact to dictionary for JSON serialization."""
        return {
            "schema": "fluxforge.saturation_rates.v1",
            "reaction_ids": self.reaction_ids,
            "y": self.y.tolist(),
            "V_y_diag": self.V_y_diag.tolist(),
            "V_y": self.V_y.tolist() if self.V_y is not None else None,
            "correction_type": self.correction_type.value,
            "n_monitors": len(self.reaction_ids),
            "factors": [
                {
                    "reaction_id": rid,
                    "R_sat": r.R_sat,
                    "R_eoi": r.R_eoi,
                    "saturation_factor": r.saturation_factor,
                    "decay_factor": r.decay_factor,
                    "counting_factor": r.counting_factor,
                    "flux_history_factor": r.flux_history_factor,
                    "burnup_correction": r.burnup_correction,
                    "uncertainty": r.uncertainty,
                }
                for rid, r in zip(self.reaction_ids, self.results)
            ],
        }
    
    @classmethod
    def from_measurements(
        cls,
        measurements: Sequence[MonitorMeasurement],
        correction_type: CorrectionType = CorrectionType.SATURATION_ACTIVITY,
        apply_burnup: bool = False,
        V_y_full: Optional[np.ndarray] = None,
    ) -> "SaturationRatesArtifact":
        """Create artifact from monitor measurements."""
        y, V_y_diag, results = calculate_saturation_rates_batch(
            measurements, correction_type, apply_burnup
        )
        
        return cls(
            reaction_ids=[m.reaction_id for m in measurements],
            y=y,
            V_y_diag=V_y_diag,
            V_y=V_y_full,
            results=results,
            correction_type=correction_type,
        )
