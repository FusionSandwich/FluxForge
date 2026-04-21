# FluxForge Quick Start

This is the shortest path from clone to a real FluxForge workflow.

## 1. Install FluxForge

CLI-only:

```bash
pip install -e .
```

CLI plus GUI:

```bash
pip install -e '.[native-gui,reporting]'
```

## 2. Verify the Entry Points

```bash
fluxforge --help
fluxforge commands
fluxforge gui --help
```

## 3. Pick One First Workflow

### Option A: Launch the GUI

```bash
fluxforge gui --project-dir .
```

Recommended first files:

- foreground: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background: `examples/RAFM_irradiation/background.ASC`

### Option B: Run a Real CLI Workflow

```bash
fluxforge rafm-validate \
  --example-root examples/RAFM_irradiation \
  --results-root /tmp/rafm_validation \
  --no-fail
```

### Option C: Run the Shortest Artifact Chain

```bash
fluxforge ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output /tmp/rafm4_b_ingest.json

fluxforge peaks \
  --spectrum-file /tmp/rafm4_b_ingest.json \
  --output /tmp/rafm4_b_peaks.json
```

## 4. Where to Go Next

- Setup guide:
  [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
- Full CLI reference:
  [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
- Example cookbook:
  [docs/EXAMPLE_WORKFLOWS.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/EXAMPLE_WORKFLOWS.md:1)
- Full example inventory:
  [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
