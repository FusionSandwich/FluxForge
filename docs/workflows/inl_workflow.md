# INL Dosimetry Workflow

## Overview
The INL Dosimetry workflow in FluxForge mimics the specific metrology sequence used by the Idaho National Laboratory (INL) for interpreting flux wire spectra.

## Key Concepts
- Relies on raw `.ASC` files as primary inputs.
- Employs a strict two-stream calculation: Unsmoothed data for area calculations, Savitzky-Golay smoothed data for peak identification.
- Integrates FWHM-scaled Regions of Interest (ROI) (typically varying from 1.5 to 2.5 times the FWHM).
- Utilizes Covell's linear continuum subtraction method.

## Usage
Run the following script to see the INL workflow in action against test files:
```bash
python examples/RAFM_irradiation/compare_inl_workflow_to_qg.py
```
