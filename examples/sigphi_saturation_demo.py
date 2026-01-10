"""Minimal SigPhi-equivalent saturation-rate demo.

Shows how to compute SigPhi-style saturation reaction rates (R_sat) from a
measured activity at count time, including optional burnup correction.

Run:
  python examples/sigphi_saturation_demo.py
"""

from __future__ import annotations

from fluxforge.physics.sigphi import (
    IrradiationHistory,
    MonitorMeasurement,
    NeutronBurnupModel,
    calculate_saturation_rate,
)


def main() -> None:
    # Example: Au-197(n,g)Au-198 (half-life ~ 2.6947 d)
    half_life_s = 2.6947 * 24 * 3600

    history = IrradiationHistory(
        segments=[
            (2.0 * 3600, 1.0),  # 2h at full power
            (0.5 * 3600, 0.0),  # 0.5h outage
            (1.0 * 3600, 0.8),  # 1h at 80%
        ],
        cooling_time_s=30 * 60,
        counting_time_s=60 * 60,
    )

    meas = MonitorMeasurement(
        reaction_id="Au-197(n,g)Au-198",
        activity_bq=5.0e4,
        activity_uncertainty=2.0e2,
        half_life_s=half_life_s,
        target_atoms=1.0e17,
        irradiation_history=history,
    )

    no_burn = calculate_saturation_rate(measurement=meas, apply_burnup=False)

    burn = NeutronBurnupModel(
        # Rough flux estimate at rel_power=1
        flux_cm2_s_at_rel_power_1=1.0e12,
        # "Target destruction" is workflow-dependent; this is just a demo number.
        sigma_target_destruction_barns=50.0,
        max_correction_factor=10.0,
    )

    with_burn = calculate_saturation_rate(
        measurement=meas,
        apply_burnup=True,
        burnup_model=burn,
    )

    print("=== SigPhi saturation-rate demo ===")
    print(f"R_sat (no burnup):   {no_burn.R_sat:.6e} 1/s")
    print(f"R_sat (with burnup): {with_burn.R_sat:.6e} 1/s")
    print(f"Burnup correction:   {with_burn.burnup_correction:.6f}")
    if with_burn.burnup_warning:
        print(f"Burnup warning:      {with_burn.burnup_warning}")


if __name__ == "__main__":
    main()
