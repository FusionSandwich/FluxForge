# Contributing to FluxForge

FluxForge is developed under an issue-first, additive roadmap. The repository contains
the project-management source of truth needed to apply the current roadmap on GitHub,
including milestones, labels, issue templates, ADRs, and the initial epic/issue seed set.

## Governing Rules

### 1. Issue-First Execution

No significant implementation work should begin without a corresponding GitHub issue.
The canonical issue seed data lives under `.github/project-management/`.

### 2. Additive Capability Policy

FluxForge does not remove a valid analytical capability simply because a newer capability
is being added. Existing methods remain available unless they are incorrect, unsafe, or
unmaintainable beyond repair.

Examples:

- Add RMLE without removing GRAVEL or MAXED.
- Add Bayesian identification without removing manual assignment.
- Add a PySide6 GUI without deleting the existing Tk workflow before parity exists.
- Add a Vispy renderer without removing PyQtGraph support.
- Add Standards mode without restricting Expert mode.

### 3. User-Choice Policy

When multiple scientifically defensible methods exist:

- expose all supported methods,
- mark one as the recommended default,
- document when each method should be used,
- persist the user's choice where the workflow expects repetition, and
- record the chosen method and parameters in exported artifacts.

### 4. Standards-Locked Workflows

When a workflow is advertised as standards-compliant, the standards-constrained variant
must lock equations, thresholds, reporting fields, and validation logic to the active
standard. Expert mode remains available for free-form analysis.

## Roadmap Assets

The Stage 0 project-management assets live here:

- `.github/project-management/milestones.json`
- `.github/project-management/labels.json`
- `.github/project-management/board.json`
- `.github/project-management/issues.json`
- `.github/workflows/sync-project-planning.yml`

The architecture decision records live in `docs/adr/`.

## ADR Process

- Create or update an ADR for architectural decisions that materially affect the roadmap.
- Do not delete superseded ADRs; mark them superseded.
- If a PR conflicts with an accepted ADR, update the ADR or revise the PR.

## Local Verification

Before merging planning or scaffolding changes, run the targeted tests that validate the
project tracker and the new package scaffolding:

```bash
PYTHONPATH=src pytest -q tests/test_project_tracker_assets.py tests/test_plugin_registry.py
```
