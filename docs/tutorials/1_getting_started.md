# Tutorial 1: Getting Started with FluxForge

## Introduction
FluxForge lets you run either through the command line or our `tkinter` driven GUI. 

## Python GUI
Using the GUI is excellent for those looking to visualize the FWHM bounds on a spectrum or analyze multiplet structures in real time.
```bash
# Must be run from an active virtual environment
python -m fluxforge.gui.app
```
You can import your own raw `.ASC` files natively.

## Python CLI
Running the CLI is highly recommended for building batch automation scripts.

1. **Spectrum Interrogations**
   Generate activities by building a strict plan definition (usually a JSON file) linking target counts, reaction products, and timing geometries.
   ```bash
   python -m fluxforge.cli.app build-response ...
   ```
2. **ASTM E261 Runner**
   You can also launch prebuilt configurations using the `--plan` argument. Check the repository `examples/astm_e261_plan.json` for structure definitions explicitly targeting standard Reactor Dosimetry setups.
