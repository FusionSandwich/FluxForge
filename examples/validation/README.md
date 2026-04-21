# Validation Demos

This directory contains small validation-oriented example scripts that exercise
specific capability areas rather than full end-to-end workflows.

## Scripts

| Script | Purpose | Command |
|---|---|---|
| `attenuation_demo.py` | attenuation material and transmission demo | `python examples/validation/attenuation_demo.py` |
| `calibration_fit_demo.py` | detector efficiency and resolution fitting demo | `python examples/validation/calibration_fit_demo.py` |
| `cross_section_demo.py` | reaction-library lookup and evaluation demo | `python examples/validation/cross_section_demo.py` |
| `iec_read_demo.py` | IEC spectrum reader demo | `python examples/validation/iec_read_demo.py /path/to/file.iec` |

## Shared Inputs

The first three demos are self-contained or use built-in data. The IEC reader
demo requires a user-supplied IEC file. The broader small-file spectroscopy
assets live in:

- [examples/spectroscopy_data/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/spectroscopy_data/README.md:1)
