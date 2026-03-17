# Spectrum IO Test Bundle

This directory contains sample spectrum files used by FluxForge tests and
examples for SPE, CHN, CNF, and IEC readers.

- `examples/`: small reference example inputs.
- `samples/`: regression/parity sample files used by automated tests.

These files are treated as data-only test inputs; FluxForge does not import or
execute external reference program code during pytest runs.
