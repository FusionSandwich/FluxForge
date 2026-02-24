#!/usr/bin/env python3
"""
Attenuation demo using XCOM-backed materials.
"""

from fluxforge.physics.attenuation import get_material, mixture_attenuation_factor


def main() -> None:
    iron = get_material("Fe")
    print("Iron mu @ 511 keV:", float(iron.mu(511.0)))
    print("Iron transmission @ 511 keV, 0.3 cm:", float(iron.transmission(511.0, 0.3)))

    mix = {"Iron": 0.5, "Lead": 0.5}
    transmission = mixture_attenuation_factor(mix, 1000.0, 0.2, density=8.0)
    print("Mixture transmission @ 1 MeV, 0.2 cm:", float(transmission))


if __name__ == "__main__":
    main()
