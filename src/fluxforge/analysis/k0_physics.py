"""Compatibility wrapper for shared k0 physics helpers.

Prefer importing from ``fluxforge.k0_physics`` in new code.
"""

from fluxforge.k0_physics import (  # noqa: F401
    calculate_q0_alpha,
    counting_factor,
    decay_factor,
    saturation_factor,
    sdc_factor,
)
