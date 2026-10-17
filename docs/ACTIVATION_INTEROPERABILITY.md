# Target monitors, FISPACT and cross-energy calibration uncertainty

## Target activation/current normalization

`physics.target_monitor.target_monitor_calibration` covers the lithium-target
Be-7 activation/current-calibration use case with explicit, replaceable physics
inputs. It requires isotope areal density (atoms/cm²), effective production cross
section (cm²), total intercepted current (µA), positive charge-state magnitude,
piecewise durations/currents, half-life and activity reference/cooling time.
No natural-lithium abundance is hard-coded. It accounts for production decay,
shutdown intervals and cooling instead of equating produced atoms with surviving
atoms when irradiation decay matters. Returned particle rates are particles/s;
divide by justified beam area before interpreting them as flux per cm²/s.

The target must have effectively uniform isotope areal density and the same
effective response across irradiation segments. Thick-target slowing-down,
energy/angle-dependent neutron yield, beam interception, feeding and burnup
require additional physics. An effective production cross section is not a
universal neutron-source yield. Its energy uncertainty must be propagated through
the response, rather than adding an unrelated fractional beam-energy error.

The optional full joint covariance has order `[activity, areal_density,
cross_section, currents...]`; propagated uncertainty is first-order and
conditional on timing/half-life. A common current error remains common across
segments. Missing covariance is unavailable. `target_monitor_scale_draws`
transforms caller-supplied joint input draws without destroying correlations.
Neither route grants physical qualification of the experiment or source spectrum.

## FISPACT interchange

`io.fispact.read_fispact_activities` reads the inventory_data/nuclides JSON layout,
preserving exact ZAI ground/metastable identity, explicit zero and absent entries.
The required interval index is zero-based. Missing requested nuclides, duplicate
identities and invalid activities raise errors; no error strings become numeric
activity values.

`write_fispact_history` exports an inventory-phase history fragment after initial
setup/ATOMS. Inputs are scalar flux in particles/cm²/s and **interval durations**
in seconds. Every TIME interval is followed by ATOMS so it is solved. Shutdowns
remain zero-flux intervals without an implicit clock reset. Existing output files
are never overwritten. This is not a FISPACT execution or full-deck validator.

These conventions follow the official FISPACT-II keyword documentation:
[FLUX](https://fispact.ukaea.uk/wiki/Keyword%3AFLUX),
[TIME](https://fispact.ukaea.uk/wiki/Keyword%3ATIME), and
[STEP](https://fispact.ukaea.uk/wiki/Keyword%3ASTEP).

`io.spe.read_multiple_spe(..., sum_spectra=True)` now rejects mismatched native
grids/detectors, repeated paths and duplicate byte copies instead of silently
dropping mismatched spectra. It preserves counting covariance and each source's
hash and acquisition interval. Aggregated live/real times are totals; the result
does not claim one contiguous decay clock. Use the original intervals for
activation/count prediction, especially when gaps or substantial decay matter.

## Detector covariance across gamma lines

`analysis.efficiency_covariance.efficiency_evaluation_covariance` propagates
log-polynomial/Gray fit parameter covariance to a full cross-energy efficiency
covariance using `J C Jᵀ`. Nonzero detector systematic uncertainty requires an
explicit shared-versus-independent correlation convention. Missing covariance,
unsupported models and extrapolation beyond the calibration interval raise;
the legacy 5% default is not treated as measured uncertainty. Source activity,
geometry transfer, emission and certificate applicability remain experimental
responsibilities. This is a first-order propagation, not automatic coverage.

The common covariance validator checks PSD in correlation coordinates. A matrix
containing Bq², cm⁻⁴, cm⁴ and µA² can have an enormous dynamic range: a dimensional
global tolerance must not hide an invalid small-variance block. Zero-variance
inputs cannot carry nonzero covariance. No uncertainty floor is introduced.

```sh
PYTHONPATH=src pytest -q tests/test_activation_interoperability.py
```

Tests compare target activation with an independent decay-time integral and
finite-difference Jacobian, verify shared normalization cancellation and mixed-unit
covariance rejection, preserve metastable inventory identities, and verify full
cross-energy efficiency covariance. External FISPACT execution is not claimed.
