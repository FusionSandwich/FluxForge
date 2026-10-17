# Optional SANDY/NJOY response processing

`data.sandy_response.process_neutron_endf` processes a local neutron ENDF
evaluation with a caller-supplied local NJOY executable via SANDY. It requests
MF33 ERRORR output, using the same explicit energy grid and weighting for its
mean responses and covariance. ERRORR is requested in absolute barn² units;
temperature, tolerance, weighting, SANDY version and source/executable hashes are
recorded. No evaluated data are downloaded or guessed. Engine errors, missing
reactions, missing covariance, ambiguous units and mismatched labels/grids fail.

`response_from_errorr` also imports already-processed SANDY ERRORR objects.
The caller must declare `relative` or `absolute_barn2` covariance. MAT/MT/energy
labels align both covariance axes before conversion; flattening an unselected
matrix is not sufficient. Full energy and cross-reaction covariance is retained.
Relative covariance is converted using the outer product of mean cross sections;
absolute covariance is never multiplied by the mean twice.

`ResponseBundle` stores energy in eV, group-averaged response in cm² per target
atom and absolute response covariance in cm⁴. Its operator multiplies integrated
group flux. Save/load use compressed NPZ with `allow_pickle=False`, explicit
reaction/particle identities and no overwrite. The representation can hold other
incident particles when their applicable evaluated/processed data are supplied;
the SANDY processing function in this PR is explicitly **neutron-only**.

```python
from fluxforge.data.sandy_response import process_neutron_endf
from fluxforge.data.response_bundle import write_response_bundle

bundle = process_neutron_endf(
    "local_evaluation.endf", [("monitor_capture", 125, 102)],
    [1e-5, 1.0, 1e3, 2e7], njoy_executable="/absolute/path/to/njoy",
    temperature_K=300, weighting_iwt=3,
)
write_response_bundle("new_response.npz", bundle)
```

Supply the resulting response/covariance to `GaussianResponseModel` with matching
per-target-atom rates. Multiply by target atoms exactly once if using total foil
reaction rates. A fixed within-group weighting can bias coarse-group results;
establish response closure with finer grids or source-appropriate weighting.
Covariance availability does not establish complete uncertainty, experimental
calibration applicability or physical qualification.

The historical `njoy.parse_gendf` empty-output placeholder is replaced with
optional SANDY labelled GENDF parsing. Missing SANDY now fails clearly rather
than allowing an empty parse to count as NJOY processing success. Its legacy
return contract supports one MAT and mean cross sections in barns; use response
bundles for covariance-bearing inference. Old NJOY deck-generation paths are
not newly qualified by the adapter.

API inspected against official SANDY revision
[`2bb1da4`](https://github.com/luca-fiorito-11/sandy/tree/2bb1da40b436c8d88ba83e5a11a1aae0d7a78d2c):
[ERRORR extraction](https://github.com/luca-fiorito-11/sandy/blob/2bb1da40b436c8d88ba83e5a11a1aae0d7a78d2c/sandy/errorr.py),
[processing](https://github.com/luca-fiorito-11/sandy/blob/2bb1da40b436c8d88ba83e5a11a1aae0d7a78d2c/sandy/endf6.py),
and [relative/absolute ERRORR options](https://github.com/luca-fiorito-11/sandy/blob/2bb1da40b436c8d88ba83e5a11a1aae0d7a78d2c/sandy/njoy.py).
Install a compatible SANDY version and NJOY separately according to their
documentation. They remain optional dependencies.

Validation here uses independently constructed labelled pandas outputs,
permuted reaction/energy axes, known full covariance, unit conversions, missing
data controls, safe roundtrips and mocked external-call contracts. **A real
SANDY/NJOY run on evaluated ENDF was not performed.** That external integration
gate remains open; the adapter is not experimentally validated or a certified
replacement for every SANDY/NJOY version.
