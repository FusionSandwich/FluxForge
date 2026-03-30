# FluxForge — Additions & Changes to the GUI Implementation Guide
### Merged from: GUI Implementation Guide (base) + Synthesis Plan + v2 Review
### Branch: `chore/folder-audit-native-gui-review-20260317`

---

## Verdict: Which Plan is the Best Base?

Three documents were compared:

| Document | Strengths | Weaknesses |
|---|---|---|
| **GUI Implementation Guide** | Most granular, developer-ready: Qt architecture, dockable zones, interaction bindings, color tokens, 23-item action list, packaging strategy | Missing: plugin registry, Standards mode, GitHub setup, HAL in Phase 1, RMLE, QA drift monitoring, N42 writing |
| **Technical Synthesis Plan** | Strong on theory: RMLE, RMLE math, HAL-first, hierarchical rendering, ASTM depth, anti-automation bias | Too conceptual; no widget names, no code snippets, no interaction specs |
| **v2 Merged Review** (uploaded) | Strong on governance: plugin registries, three-mode system, ADRs, comprehensive GitHub setup, additive policy formalized | High-level only; defers all implementation detail to the other two documents |

**Base: GUI Implementation Guide.** It is the only document that can be handed to a developer and coded from directly.

**This document** captures every addition and change needed after merging the best of the Synthesis Plan and the v2 Review into that base. Nothing is removed. Everything is additive.

---

## Four Non-Negotiable Governing Rules

These rules apply to every decision in this document and in the project at large. They supersede any individual feature preference.

### Rule 1 — Issue-First Execution
No major implementation work begins until the GitHub infrastructure (milestones, labels, epics, issue templates, project board) is in place. The roadmap must be executable through GitHub issues, not left as a static document. **Stage 0 must be completed first.**

### Rule 2 — Additive Capability Policy
FluxForge never removes a valid analytical capability to make room for a new one. Concretely:
- Add RMLE **without removing** GRAVEL or MAXED.
- Add Bayesian ID **without removing** manual assignment.
- Add ML assistance **without forcing** ML as a requirement.
- Add Vispy backend **without deleting** PyQtGraph support.
- Add quick calibration mode **without removing** the full calibration workflow.
- Add the Standards mode **without restricting** the Expert mode.

The only exception is genuinely broken or unmaintainable code that has no users.

### Rule 3 — User-Choice Policy
Where multiple scientifically defensible methods exist for the same task, FluxForge must:
- Expose all of them as selectable options.
- Mark one as the **recommended default** with a clear rationale tooltip.
- Persist the user's last-used choice per-task in `QSettings`.
- Record the chosen method in every exported report and result file (provenance).
- Provide batch-mode consistency controls so batch runs use a single declared method.

### Rule 4 — Standards-Locked Workflows
When a workflow is labelled as ASTM-compliant (E181, E261, E1297, E1218, C1232, C1030, or future modules), FluxForge must provide a **Standards mode** that locks the required equations, thresholds, reporting fields, and sequence of operations to exactly what the written standard specifies. Outside Standards mode, any alternative method may be used freely.

---

## Table of Contents

- [Stage 0 — GitHub Infrastructure (must happen first)](#stage-0--github-infrastructure-must-happen-first)
- [1. Architecture Decisions and Module Structure](#1-architecture-decisions-and-module-structure)
- [2. PyQtGraph-First Rendering (not Vispy-first)](#2-pyqtgraph-first-rendering-not-vispy-first)
- [3. Plugin/Registry Layer in Phase 1](#3-pluginregistry-layer-in-phase-1)
- [4. HAL Promoted to Phase 1](#4-hal-promoted-to-phase-1)
- [5. Hierarchical Visual Representation for Legacy Hardware](#5-hierarchical-visual-representation-for-legacy-hardware)
- [6. Three-Mode GUI: Simple / Expert / Standards](#6-three-mode-gui-simple--expert--standards)
- [7. Coordinated Multiple Views — SelectionBus](#7-coordinated-multiple-views--selectionbus)
- [8. Three-Tier Defaults vs Requirements](#8-three-tier-defaults-vs-requirements)
- [9. ANSI N42.42 Compliant Output](#9-ansi-n4242-compliant-output)
- [10. ASTM C1030 — Plutonium Isotopic Analysis](#10-astm-c1030--plutonium-isotopic-analysis)
- [11. ASTM E181 QA Monitoring — FWHM Drift Detection](#11-astm-e181-qa-monitoring--fwhm-drift-detection)
- [12. Spectrum Unfolding: Add RMLE Alongside GRAVEL/MAXED/ML](#12-spectrum-unfolding-add-rmle-alongside-gravelmaxedml)
- [13. Nuclide Aging and Daughter Product Evolution](#13-nuclide-aging-and-daughter-product-evolution)
- [14. GPS Data Extraction and Field Survey Mapping](#14-gps-data-extraction-and-field-survey-mapping)
- [15. Anti-Automation Bias — Residuals First](#15-anti-automation-bias--residuals-first)
- [16. Digital Twin Hardware Dashboard](#16-digital-twin-hardware-dashboard)
- [17. Bayesian Nuclide ID — "Guess" Mode](#17-bayesian-nuclide-id--guess-mode)
- [18. GUI Panel Positioning from the Synthesis Plan](#18-gui-panel-positioning-from-the-synthesis-plan)
- [19. Consolidated Priority Action List](#19-consolidated-priority-action-list)
- [Quick Reference: Category Winners](#quick-reference-category-winners)

---

## Stage 0 — GitHub Infrastructure (must happen first)

> **This entire stage must be completed before any Phase 1 implementation work begins.** The point is to make the roadmap executable through the tracker, not just a document. Every issue created in 0.6 maps directly to a section of this additions document.

### Repository Status — 2026-03-30

- This document remains a secondary detail source. Where it conflicts with
  `docs/FluxForge_Final_Additions.md`, the final document controls execution.
- Completed in-repo: tracker manifests, issue templates, accepted ADRs, the sync
  workflow, and `tests/spectra/`.
- Ordered execution tracker: `.github/project-management/implementation_steps.json`.
- Canonical status tracker: `docs/ROADMAP_EXECUTION_STATUS.md`.
- Stage 0 live tracker application was verified complete on GitHub on 2026-03-30.
- Phase 1 implementation is repo-complete and formally complete in sequence.
- Phase 2.1 is now implemented in the redesigned Qt path, including the unified
  calibration dialog, residual-first plots, ASTM E181 order locking, and the native
  review probe used for Playwright inspection.
- Phase 2.2 through 2.6 are now implemented in the redesigned Qt path as well,
  including quick-slider anchors, deviation-pair fine tuning, draggable ROI peak
  fitting, the registered skew fitter, and the registry-driven method selector.
- Phase 2.7 through 2.20 are now implemented in the redesigned Qt path too,
  including undoable peak-table operations, auto peak review, Bayesian library
  matching, efficiency fitting, activity calculation, source-age correction,
  background subtraction, mini residual strips, survey-map rendering,
  multi-spectrum tabs, pinned nuclides, cascade-sum overlays, and the Bayesian
  Gaussian fitter.
- Additional audit evidence now exists for the redesigned Qt GUI: mouse-driven peak
  picking, governed data-library selectors, library-assisted calibration-line assignment,
  both Manual and Standards calibration workflows, quick-slider mode, deviation-pair
  tuning, and ROI fitter interaction were exercised successfully.
- Additional audit evidence now exists for the completed Phase 2 shell as well:
  auto peak review, undo/redo, Bayesian matching, efficiency fitting, activity
  calculation, background subtraction, pinned nuclides, survey-map rendering,
  multi-spectrum switching, and Playwright-reviewed completion screenshots.
- Phase 2 as a whole is now complete; roadmap items `2.1` through `2.20` are now
  complete and the next required implementation step is `3.1`.
- The current roadmap next step is Phase 3.1, GRAVEL.
- GUI redesign status in-repo: the Qt shell now covers the repo-side deliverables for
  roadmap items `1.3` through `1.18`, including the `.ffs` session path, reader
  factory, validated N42 export, SQLite nuclide database, and instant overlay search;
  the older Tk GUI remains in place as a legacy/archive fallback while parity work continues.
- Legacy GUI planning references are now archived as `docs/GUI_PLAN_old.md` and
  `docs/GUI_CAPABILITY_PROGRAM_old.md` so the active GUI plan set remains this document,
  `docs/FluxForge_Final_Additions.md`, and `docs/FluxForge_Improvement_Guide.docx`.

### 0.1 Create Milestones

| Milestone | Scope |
|---|---|
| `M0 — Planning & Backlog` | All Stage 0 items; ADRs; issue scaffolding |
| `M1 — GUI Shell & Architecture` | Phases 1.1–1.7 of the merged roadmap |
| `M2 — Core Analysis Parity` | Phase 2: calibration, peak fitting, ID, activity |
| `M3 — Advanced Analysis` | Phase 3: unfolding, ML, Standards framework |
| `M4 — Standards & QA` | ASTM modules, QA monitor, C1030, E261 |
| `M5 — MCA Foundations` | Phase 4: HAL drivers, live acquisition, dashboard |
| `M6 — Packaging & Release` | AppImage, Windows .exe, CI, regression baselines |

### 0.2 Create Labels

**Area labels** — applied to every issue to route it to the right contributor:
```
area/gui          area/core         area/io           area/ml
area/unfolding    area/standards    area/hal          area/reporting
area/performance  area/docs         area/testing      area/packaging
```

**Type labels:**
```
type/epic         type/feature      type/refactor     type/bug
type/design       type/adr          type/spike
```

**Priority labels:**
```
priority/p0       priority/p1       priority/p2       priority/p3
```

**Platform labels:**
```
platform/linux    platform/windows
```

**Status labels:**
```
good-first-issue  blocked           needs-design      needs-spec
```

### 0.3 Create Issue Templates

Create these seven templates under `.github/ISSUE_TEMPLATE/`:

```
feature_request.md      — Title, problem statement, proposed solution,
                          acceptance criteria, labels to apply
epic.md                 — Epic title, child issues list, done definition,
                          milestone, blocking issues
bug_report.md           — Description, reproduction steps, expected vs
                          actual, platform, FluxForge version, spectra/logs
performance_issue.md    — Description, profiling data, target metric,
                          hardware spec
standards_issue.md      — Standard name and section, specific requirement,
                          current behavior, required behavior, test data
gui_ux_issue.md         — Component affected, current UX, proposed UX,
                          mockup or ASCII diagram, mode (Simple/Expert/Standards)
research_spike.md       — Question to answer, time-box, deliverable
                          (ADR, proof-of-concept, benchmark, or decision)
```

### 0.4 Create Project Board

Create a GitHub Project with these columns:
```
Backlog → Ready → In Progress → Review → Blocked → Done
```

Link all milestones and labels. Enable the "Iteration" field for sprint planning.

### 0.5 Create Architecture Decision Records (ADRs)

Create an `docs/adr/` directory. The first three ADRs should be created as issues (`type/adr`) before any code is written:

**ADR-001: GUI stack and rendering architecture**
- Decision: PySide6 as the GUI framework.
- Decision: PyQtGraph as the initial production rendering backend.
- Decision: `SpectrumCanvas` abstract class isolates the rendering backend.
- Decision: Vispy is a supported optional backend, not the default.
- Decision: C++/Vulkan is the documented future migration path if Python rendering becomes insufficient, not a current requirement.
- Rationale: contributor accessibility, ML/Python ecosystem integration, delivery speed.

**ADR-002: Additive capability policy**
- Decision: FluxForge never removes a valid analytical capability to add a new one.
- Decision: Every task with multiple valid methods must expose all of them with documented defaults.
- Decision: Method choice is persisted per-user in `QSettings` and recorded in all reports.
- Rationale: scientific reproducibility; user trust; open-source sustainability.

**ADR-003: Standards mode and user-choice policy**
- Decision: Three GUI modes — Simple, Expert, Standards.
- Decision: Standards mode locks equations, thresholds, and report fields to the exact requirements of the active standard.
- Decision: Expert mode provides full user control over all methods.
- Decision: Simple mode reduces visible complexity without removing capabilities.
- Rationale: satisfies both routine lab use and strict compliance simultaneously.

### 0.6 Create the Initial Epic Issues

Open these eight epics on GitHub immediately, each as a `type/epic` issue with child issue stubs listed:

```
Epic 1: GUI Shell and Docking Framework
  Children: MainWindow scaffold, ModeManager, QSS themes, status bar,
            SelectionBus, persistent layout, keyboard shortcut map

Epic 2: Spectrum Canvas and Rendering Abstraction
  Children: SpectrumCanvas interface, PyQtGraph backend, HierarchicalBuffer,
            interaction model (zoom/pan/context menus), overlay layers,
            mini-residuals pane, Vispy backend stub

Epic 3: File I/O and Session Model
  Children: Reader factory, N42/CHN/SPC/CNF/SPE/CSV readers, N42.42 writer,
            GPS metadata extraction, .ffs session file, drag-and-drop,
            recent files, metadata panel

Epic 4: Core Spectroscopy Workflows
  Children: Calibration workspace, peak fitting algorithms, peak table,
            SelectionBus synchronization, Bayesian ID, auto-search,
            nuclide library DB, activity calculation, background subtraction,
            source age correction, undo/redo

Epic 5: Standards and QA Framework
  Children: Standards module interface, ASTM E181, E1297, E1218, C1232,
            C1030, E261 stubs, QAMonitor class, QA history panel,
            Standards mode GUI locking, ASTM report template

Epic 6: Unfolding and Advanced Analysis
  Children: Unfolding registry, GRAVEL, MAXED, RMLE, ML-seed method,
            response matrix loader, convergence plot, uncertainty bands,
            unfolding dialog with comparison mode, algorithm provenance

Epic 7: HAL and Live Acquisition Foundation
  Children: MCADevice ABC, device registry, mock device, DeviceStatus model,
            Devices panel stub, Dashboard tab stub, status bar LED,
            HAL plugin interface

Epic 8: Reporting, Packaging, and Release Engineering
  Children: Jinja2 report engine, standard lab template, ASTM template,
            batch summary template, residuals grid in reports, PDF export,
            PyInstaller Windows .exe, AppImage Linux, CI pipeline,
            regression test harness
```

### 0.7 First 20 Issues to Open Immediately (Ordered)

These are the concrete first issues after the epics exist. They should be opened in this order so dependencies are visible:

```
 1. [ADR] ADR-001: GUI stack, PySide6, PyQtGraph-first, SpectrumCanvas ABC
 2. [ADR] ADR-002: Additive capability policy
 3. [ADR] ADR-003: Standards mode and three-tier defaults
 4. Scaffold MainWindow with all six docking zones (A–F per base plan layout)
 5. Implement ModeManager: Simple / Expert / Standards with QSettings persistence
 6. Implement SelectionBus singleton for coordinated multiple views
 7. Implement SpectrumCanvas ABC + PyQtGraph backend + HierarchicalSpectrumBuffer
 8. Implement Spectrum dataclass + .ffs session file save/restore
 9. Implement reader factory + N42 / CHN / SPC / CNF / SPE / CSV readers
10. Implement N42.42 (2012) writer with XSD schema validation
11. GPS metadata extraction for N42, CHN, CNF + Spectrum.metadata dict
12. Build nuclide SQLite DB from ENDF/B-VIII.0 including decay_chains table
13. Implement nuclide search panel with search-as-you-type + instant line overlay
14. Implement peak table with all columns + SelectionBus synchronization
15. Implement calibration workspace shell with embedded live canvas
16. Define MCADevice ABC + DeviceStatus model + mock device + device registry
17. Reserve Dashboard tab in Zone C + hardware LED in status bar
18. Implement Jinja2 report engine shell + standard lab template
19. Implement plugin registries: PeakFitter, Unfolder, CalibrationModel, NuclideID,
    StandardsModule, RenderBackend
20. Add test harness with sample spectra (HPGe, NaI) and CI regression baselines
```

---

## 1. Architecture Decisions and Module Structure

### What needs to change in the base plan

The base plan's directory structure (`fluxforge/gui/`, `fluxforge/core/`, `fluxforge/io/`, `fluxforge/ml/`, `fluxforge/standards/`) is correct but needs three additions from the v2 review:

**Addition 1 — Separate `fluxforge/unfolding/` module**

The v2 review correctly separates unfolding into its own top-level module rather than burying it in `core`. Unfolding has its own registry, response matrices, method implementations, and convergence logic. Keeping it distinct makes it easier for physics specialists to contribute to just the unfolding layer without understanding the full GUI codebase.

```
fluxforge/
├── gui/                          # All Qt/PySide6 UI code — no analysis logic
│   ├── main_window.py
│   ├── spectrum_canvas.py        # SpectrumCanvas ABC + backend dispatch
│   ├── panels/                   # QDockWidget panels
│   ├── dialogs/                  # Modal dialogs
│   ├── themes/                   # QSS theme files + color tokens
│   └── widgets/                  # Reusable custom Qt widgets
├── core/                         # Analysis engine — zero Qt imports
│   ├── spectrum.py               # Spectrum dataclass
│   ├── calibration.py
│   ├── peak_fitting.py
│   ├── activity.py
│   └── nuclide_library.py
├── unfolding/                    # NEW — unfolding as a top-level module
│   ├── __init__.py
│   ├── registry.py               # Unfolder plugin registry
│   ├── gravel.py                 # GRAVEL implementation
│   ├── maxed.py                  # MAXED implementation
│   ├── rmle.py                   # RMLE implementation (new)
│   ├── ml_seed.py                # U-Net fast approximation
│   └── response_matrix.py       # Response matrix loader and validator
├── io/                           # File readers/writers
│   ├── reader_factory.py
│   ├── n42.py                    # Reader + writer (N42.42 compliant)
│   ├── chn.py
│   ├── spc.py
│   ├── cnf.py
│   ├── spe.py
│   └── csv_reader.py
├── ml/                           # ML inference engine — zero Qt imports
│   ├── peak_cnn.py
│   ├── nuclide_classifier.py
│   └── onnx_runner.py
├── standards/                    # ASTM and other standards modules
│   ├── base.py                   # StandardsModule ABC
│   ├── registry.py               # Standards plugin registry
│   ├── e181.py
│   ├── e261.py                   # stub
│   ├── e1297.py
│   ├── e1218.py
│   ├── c1232.py
│   ├── c1030.py
│   └── qa_monitor.py
├── hal/                          # Hardware Abstraction Layer
│   ├── base.py                   # MCADevice ABC
│   ├── device_registry.py
│   ├── mock_device.py
│   └── protocols/
│       ├── usb_hid.py            # skeleton
│       ├── ethernet.py           # skeleton
│       └── serial_rs232.py       # skeleton
├── plugins/                      # NEW — plugin registry infrastructure
│   ├── __init__.py
│   └── registry.py               # Generic PluginRegistry[T] class
└── resources/
    ├── db/                       # nuclides.db (ENDF/B-VIII.0 SQLite)
    ├── schemas/                  # N42.42 XSD, ASTM report schemas
    ├── themes/                   # dark.qss, light.qss
    ├── templates/                # Jinja2 report templates
    └── icons/                    # SVG icons
```

**Addition 2 — Strict core/GUI separation rule**

Add to the project's `CONTRIBUTING.md`:
> **Core/GUI boundary rule:** Files in `fluxforge/core/`, `fluxforge/unfolding/`, `fluxforge/ml/`, `fluxforge/standards/`, `fluxforge/hal/`, and `fluxforge/io/` must contain **zero Qt imports**. All communication from these modules to the GUI must go through Python signals emitted from thin worker QThread wrapper classes defined in `fluxforge/gui/workers/`. CI will enforce this with an import-check lint rule.

---

## 2. PyQtGraph-First Rendering (not Vispy-first)

### What to change from the previous additions document

The previous additions document (v1) leaned toward replacing PyQtGraph with Vispy as the primary renderer. The v2 review correctly pushes back on this. The right approach is:

**PyQtGraph is the production backend.** Vispy is an optional high-performance backend that can be selected by the user. The C++/Vulkan path (Datoviz) remains a documented future migration target but is not a current requirement.

### Change to Phase 1.3 — Spectrum Canvas

Instead of "replace PyQtGraph with Vispy", implement a `SpectrumCanvas` **abstract base class** that makes the backend swappable:

```python
# fluxforge/gui/spectrum_canvas.py
from abc import ABC, abstractmethod
import numpy as np
from PySide6.QtWidgets import QWidget

class SpectrumCanvas(QWidget, ABC):
    """
    Abstract base class for all spectrum rendering backends.
    Concrete implementations: PyQtGraphCanvas, VispyCanvas (optional).
    The GUI code only speaks to this interface — never to PyQtGraph directly.
    """

    # Signals (declared in concrete subclasses via PySide6 Signal)
    cursor_moved     = None   # emits (channel: int, energy_keV: float, counts: float)
    roi_defined      = None   # emits (start_ch: int, end_ch: int)
    peak_right_clicked = None # emits (peak_id: int, global_pos: QPoint)

    @abstractmethod
    def set_spectrum(self, counts: np.ndarray, calibration=None) -> None: ...
    @abstractmethod
    def set_background(self, counts: np.ndarray) -> None: ...
    @abstractmethod
    def set_secondary(self, counts: np.ndarray) -> None: ...
    @abstractmethod
    def add_roi(self, start_ch: int, end_ch: int, color: str) -> int: ...
    @abstractmethod
    def remove_roi(self, roi_id: int) -> None: ...
    @abstractmethod
    def add_reference_lines(self, nuclide: str, energies: list, intensities: list) -> None: ...
    @abstractmethod
    def clear_reference_lines(self, nuclide: str = None) -> None: ...
    @abstractmethod
    def add_peak_fit(self, roi_id: int, x: np.ndarray, y: np.ndarray, color: str) -> None: ...
    @abstractmethod
    def set_y_mode(self, mode: str) -> None: ...  # "log" | "linear" | "sqrt"
    @abstractmethod
    def set_x_mode(self, mode: str) -> None: ...  # "channel" | "keV" | "MeV"
    @abstractmethod
    def zoom_to_range(self, x_min: float, x_max: float) -> None: ...
    @abstractmethod
    def set_residuals_visible(self, visible: bool) -> None: ...
    @abstractmethod
    def screenshot_png(self) -> bytes: ...        # for embedding in reports
```

**Register backends in the plugin registry** (see Section 3):
```python
# On application startup in main.py:
from fluxforge.plugins.registry import PluginRegistry
from fluxforge.gui.backends.pyqtgraph_canvas import PyQtGraphCanvas

PluginRegistry.renderers.register("pyqtgraph", PyQtGraphCanvas, default=True)

# Optional: register Vispy if installed
try:
    from fluxforge.gui.backends.vispy_canvas import VispyCanvas
    PluginRegistry.renderers.register("vispy", VispyCanvas, default=False)
except ImportError:
    pass  # Vispy not installed — silently skip
```

**Add to Settings dialog** (`View → Preferences → Rendering`):
- Backend selector: `PyQtGraph (default)` / `Vispy (if installed)`
- Explanation tooltip: "PyQtGraph is recommended for most users. Vispy provides higher frame rates for spectra with >32,768 channels or GPU-accelerated rendering."
- Restart required notice.

---

## 3. Plugin/Registry Layer in Phase 1

### What is missing from the base plan

The base plan adds features one by one. By Phase 3 there are three unfolding methods, four peak fitting algorithms, five nuclide ID methods, and six standards modules — but no architectural mechanism to manage them. Without a registry, each new method requires scattered if/elif chains in the GUI code and batch runners. The v2 review correctly identifies this as a Phase 1 need.

### Addition to Phase 1.1 — Project Structure

Add `fluxforge/plugins/registry.py`:

```python
# fluxforge/plugins/registry.py
from typing import TypeVar, Generic, Type, Dict, Optional

T = TypeVar("T")

class PluginRegistry(Generic[T]):
    """
    Generic registry for pluggable analytical components.
    Used for: peak fitters, unfolding methods, calibration models,
              nuclide ID engines, standards modules, render backends.
    """
    def __init__(self, label: str):
        self._label = label
        self._registry: Dict[str, Type[T]] = {}
        self._default: Optional[str] = None

    def register(self, name: str, cls: Type[T], default: bool = False) -> None:
        self._registry[name] = cls
        if default or self._default is None:
            self._default = name

    def get(self, name: str) -> Type[T]:
        if name not in self._registry:
            raise KeyError(f"{self._label}: unknown plugin '{name}'")
        return self._registry[name]

    def get_default(self) -> Type[T]:
        return self._registry[self._default]

    def list_all(self) -> Dict[str, Type[T]]:
        return dict(self._registry)

    @property
    def default_name(self) -> str:
        return self._default


# Singleton registries — instantiated once, imported everywhere
class PluginRegistries:
    renderers      = PluginRegistry("RenderBackend")
    peak_fitters   = PluginRegistry("PeakFitter")
    unfolders      = PluginRegistry("Unfolder")
    cal_models     = PluginRegistry("CalibrationModel")
    id_engines     = PluginRegistry("NuclideIDEngine")
    standards      = PluginRegistry("StandardsModule")
```

**Populate registries at startup** in `fluxforge/__init__.py` or a dedicated `bootstrap.py`:

```python
# bootstrap.py — called once at application startup
def register_all_plugins():
    from fluxforge.plugins.registry import PluginRegistries as R

    # Render backends
    from fluxforge.gui.backends.pyqtgraph_canvas import PyQtGraphCanvas
    R.renderers.register("PyQtGraph", PyQtGraphCanvas, default=True)

    # Peak fitters
    from fluxforge.core.peak_fitting import GaussianFitter, SkewedGaussianFitter, BayesianFitter
    R.peak_fitters.register("Gaussian",         GaussianFitter,        default=True)
    R.peak_fitters.register("Skewed Gaussian",  SkewedGaussianFitter)
    R.peak_fitters.register("Bayesian",         BayesianFitter)

    # Unfolding methods
    from fluxforge.unfolding.gravel import GravelUnfolder
    from fluxforge.unfolding.maxed  import MaxedUnfolder
    from fluxforge.unfolding.rmle   import RMLEUnfolder
    from fluxforge.unfolding.ml_seed import MLSeedUnfolder
    R.unfolders.register("GRAVEL",    GravelUnfolder)
    R.unfolders.register("MAXED",     MaxedUnfolder)
    R.unfolders.register("RMLE",      RMLEUnfolder,    default=True)
    R.unfolders.register("ML Seed",   MLSeedUnfolder)

    # Calibration models
    from fluxforge.core.calibration import (LinearCalModel, QuadraticCalModel,
                                             CubicCalModel, SplineCalModel)
    R.cal_models.register("Linear",    LinearCalModel)
    R.cal_models.register("Quadratic", QuadraticCalModel, default=True)
    R.cal_models.register("Cubic",     CubicCalModel)
    R.cal_models.register("Spline",    SplineCalModel)

    # Nuclide ID engines
    from fluxforge.core.nuclide_library import ManualIDEngine, BayesianIDEngine
    R.id_engines.register("Manual",   ManualIDEngine,   default=True)
    R.id_engines.register("Bayesian", BayesianIDEngine)
    # ML engine registered only if model file is present:
    try:
        from fluxforge.ml.nuclide_classifier import MLIDEngine
        R.id_engines.register("ML Assisted", MLIDEngine)
    except (ImportError, FileNotFoundError):
        pass

    # Standards modules
    from fluxforge.standards.e181  import E181Module
    from fluxforge.standards.e1297 import E1297Module
    from fluxforge.standards.e1218 import E1218Module
    from fluxforge.standards.c1232 import C1232Module
    from fluxforge.standards.c1030 import C1030Module
    for mod in [E181Module, E1297Module, E1218Module, C1232Module, C1030Module]:
        R.standards.register(mod.standard_id, mod)
```

**Benefits:**
- Adding a new peak fitter = create the class, call `R.peak_fitters.register()`. Zero changes to GUI code.
- Settings dialog auto-generates a dropdown from `R.peak_fitters.list_all()`.
- Batch runner automatically uses the registered default unless overridden.
- Reports automatically record the registry key used for each analytical step.

---

## 4. HAL Promoted to Phase 1

### What needs to change in the base plan

The base plan defers MCA acquisition to Phase 4 and reserves only minimal placeholder architecture. The v2 review and Synthesis Plan both argue correctly that the **interface** must be defined in Phase 1 — otherwise every Phase 1–3 data pipeline will be written for file-only sources and Phase 4 will require a full refactor.

### Addition to Phase 1.1 — Project Structure

Add `fluxforge/hal/` with a full interface even though no hardware drivers are written until Phase 4:

```python
# fluxforge/hal/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np

@dataclass
class DeviceStatus:
    state: str            # "idle" | "acquiring" | "busy" | "fault" | "disconnected"
    count_rate_cps: float = 0.0
    input_rate_cps: float = 0.0
    dead_time_pct:  float = 0.0
    elapsed_real_s: float = 0.0
    elapsed_live_s: float = 0.0
    high_voltage_v: float = 0.0
    temperature_k:  float = 0.0

@dataclass
class DeviceInfo:
    device_id:    str
    model:        str
    serial:       str
    n_channels:   int
    connection:   str   # "usb" | "ethernet" | "serial" | "mock"
    address:      str   # IP, COM port, or USB path

class MCADevice(ABC):
    """Abstract base for all MCA hardware drivers. Phase 1: define interface.
    Phase 4: implement concrete drivers."""

    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def start_acquisition(self, preset_seconds: Optional[float] = None,
                          preset_counts: Optional[int] = None) -> None: ...
    @abstractmethod
    def stop_acquisition(self) -> None: ...
    @abstractmethod
    def clear_spectrum(self) -> None: ...
    @abstractmethod
    def get_spectrum(self) -> np.ndarray: ...
    @abstractmethod
    def get_status(self) -> DeviceStatus: ...
    @abstractmethod
    def get_info(self) -> DeviceInfo: ...

    def register_update_callback(
        self,
        callback: Callable[[np.ndarray, DeviceStatus], None],
        interval_ms: int = 500
    ) -> None:
        """Register a callback for live spectrum updates.
        Default implementation: polling at interval_ms. Override for push."""
        ...
```

**Also add to the `Spectrum` dataclass in `core/spectrum.py`:**
```python
@dataclass
class Spectrum:
    channels:    np.ndarray
    counts:      np.ndarray
    live_time_s: float
    real_time_s: float
    energy_cal:  Optional[list] = None   # polynomial coefficients
    file_path:   Optional[str]  = None
    metadata:    dict           = field(default_factory=dict)
    # HAL fields — populated when source is a live device:
    source_type: str            = "file"   # "file" | "live_mca" | "batch_result"
    device_ref:  Optional[object] = None   # MCADevice instance or None
```

**Reserve the Devices panel in the left sidebar (Phase 1):**
The `area/hal` section in Zone B left sidebar is created in Phase 1 with a `QLabel("MCA hardware support — Phase 4")` placeholder and a greyed icon. The QDockWidget layout must accommodate it from the start so Phase 4 is not a layout refactor.

---

## 5. Hierarchical Visual Representation for Legacy Hardware

### What to add to Phase 1.3 — Spectrum Canvas

The base plan uses PyQtGraph's `auto_downsample=True` which is a simple pixel-density downsampler. The Synthesis Plan's "Hierarchical Visual Representation" is superior: pre-computed multi-resolution levels with semantic awareness (peak markers and reference lines always at full resolution regardless of zoom level).

### Addition to Phase 1.3 — Implement `HierarchicalSpectrumBuffer`

```python
# fluxforge/gui/spectrum_canvas.py (add alongside SpectrumCanvas ABC)

class HierarchicalSpectrumBuffer:
    """
    Pre-computes 5 resolution levels for a spectrum array.
    Canvas queries the appropriate level based on current zoom ratio.
    Construction time: < 5ms for 16,384-channel spectrum.
    """
    LEVELS = [1, 4, 16, 64, 256]   # downsampling factors

    def __init__(self, counts: np.ndarray):
        self._raw = counts.astype(np.float64)
        self._levels: dict[int, np.ndarray] = {}
        for factor in self.LEVELS:
            if factor == 1:
                self._levels[1] = self._raw
            else:
                # Bin-sum downsampling: preserves total counts (not average)
                n = len(self._raw) // factor
                self._levels[factor] = self._raw[:n * factor].reshape(n, factor).sum(axis=1)

    def get_for_zoom(self, visible_channels: int) -> tuple[np.ndarray, int]:
        """
        Returns (downsampled_array, factor) appropriate for the current
        number of visible channels. Factor 1 = full resolution.
        """
        if visible_channels <= 200:
            return self._levels[1], 1
        elif visible_channels <= 800:
            return self._levels[4], 4
        elif visible_channels <= 3200:
            return self._levels[16], 16
        elif visible_channels <= 12800:
            return self._levels[64], 64
        else:
            return self._levels[256], 256

    def get_raw(self) -> np.ndarray:
        return self._levels[1]
```

**Rule:** Peak markers, fitted curves, ROI boundaries, reference lines, calibration markers, and annotations are **always rendered at full channel resolution** regardless of the active display level. These are vector/overlay elements, not raster data — they must not be downsampled.

**Benefit:** On a 10-year-old GPU (Intel HD 4000 class), a 65,536-channel HPGe spectrum pans and zooms at 60 FPS because the displayed array never exceeds ~256 data points at the furthest zoom level.

---

## 6. Three-Mode GUI: Simple / Expert / Standards

### What to change in the base plan

The base plan implements a **two-mode system**: Simple and Expert (from the Synthesis Plan). The v2 review adds a critical third mode: **Standards**, which locks workflows when a standards-compliant analysis is required. This is the cleanest way to satisfy both scientific flexibility and regulatory compliance simultaneously.

### Addition to Phase 1.2 — MainWindow Skeleton

**Implement a `ModeManager` with three modes:**

```python
# fluxforge/gui/mode_manager.py
from enum import Enum
from PySide6.QtCore import QObject, Signal, QSettings

class AnalysisMode(Enum):
    SIMPLE    = "simple"
    EXPERT    = "expert"
    STANDARDS = "standards"

class ModeManager(QObject):
    mode_changed = Signal(AnalysisMode)

    _instance = None

    @classmethod
    def instance(cls) -> "ModeManager":
        if cls._instance is None:
            cls._instance = ModeManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        settings = QSettings()
        saved = settings.value("analysis_mode", AnalysisMode.EXPERT.value)
        self._mode = AnalysisMode(saved)
        self._active_standard: str | None = None

    @property
    def mode(self) -> AnalysisMode:
        return self._mode

    @property
    def active_standard(self) -> str | None:
        """Name of the locked standard when mode == STANDARDS, else None."""
        return self._active_standard

    def set_mode(self, mode: AnalysisMode, standard: str | None = None) -> None:
        if mode == AnalysisMode.STANDARDS and standard is None:
            raise ValueError("Standards mode requires an active standard name")
        self._mode = mode
        self._active_standard = standard if mode == AnalysisMode.STANDARDS else None
        QSettings().setValue("analysis_mode", mode.value)
        self.mode_changed.emit(mode)
```

**Mode Switcher widget in the top toolbar** (always visible, top-right):

```
[FluxForge]  File  Edit  View  Analysis  Calibration  Tools  Help
                                               [ EXPERT ▾ ]  [🌙]
```

Clicking the mode switcher shows a dropdown:
```
  ● Simple Mode
  ● Expert Mode  ✓ (current)
  ─────────────────────────
  Standards Mode ▶
    → ASTM E181
    → ASTM E1297
    → ASTM C1030
    → ASTM E261
    → (more from registry...)
```

**What each mode shows/hides:**

| Panel / Feature | Simple | Expert | Standards |
|---|---|---|---|
| Right sidebar (ASTM panel, ML settings) | Hidden | Visible | Visible (ASTM mandatory) |
| Batch Queue tab | Hidden | Visible | Visible |
| Efficiency calibration full dialog | "Load file" button only | Full dialog | Full dialog (locked equation) |
| Spectrum unfolding menu | Greyed | Active | Active (locked method) |
| Advanced peak fitting algorithm selector | Gaussian only | All algorithms | Standard-specified algorithm |
| Non-linear deviation pairs | Hidden | Visible | Visible |
| QA & Standards sidebar section | Summary only | Full | Full + compliance indicators |
| Algorithm selector dropdowns | Default only | All options | Locked to standard |
| Report template selector | Simple template | All templates | ASTM-format template (required) |

**Key rule:** Modes control **visibility and locking**, never **data deletion**. Switching from Standards to Expert mode preserves every analysis result. The underlying analysis capabilities are always present — modes are a UI lens, not a capability gate.

**Standards mode locking mechanism:**

When Standards mode is active, any parameter controlled by the active standard is replaced by a `QLabel` showing the locked value with a lock icon `🔒` and a tooltip explaining which standard section mandates it. The underlying `QDoubleSpinBox` or `QComboBox` still exists in the widget hierarchy (for programmatic access) but is hidden. This ensures that if the user switches back to Expert mode, their previous values are restored.

---

## 7. Coordinated Multiple Views — SelectionBus

### What to add

The base plan implies this (e.g., "clicking a row highlights the corresponding peak on canvas") but never formalizes it as a first-class design rule applied consistently across all views.

### Addition — Formal CMV Implementation

**Implement a `SelectionBus` singleton in Phase 1:**

```python
# fluxforge/gui/selection_bus.py
from PySide6.QtCore import QObject, Signal

class SelectionBus(QObject):
    """
    Central event bus for coordinated multiple views.
    All views emit to this bus; all views receive from this bus.
    No view holds a direct reference to any other view.
    """
    _instance = None

    @classmethod
    def instance(cls) -> "SelectionBus":
        if cls._instance is None:
            cls._instance = SelectionBus()
        return cls._instance

    # ── Spectrum / peak selections ──────────────────────────────────
    peak_selected          = Signal(int)    # peak_id
    roi_selected           = Signal(int)    # roi_id
    energy_range_selected  = Signal(float, float)   # keV_min, keV_max

    # ── Nuclide selections ───────────────────────────────────────────
    nuclide_selected       = Signal(str)    # nuclide symbol e.g. "Cs-137"
    nuclide_deselected     = Signal(str)

    # ── Calibration selections ───────────────────────────────────────
    cal_point_selected     = Signal(int)    # calibration point index
    cal_updated            = Signal()       # any calibration change

    # ── Spectrum / file selections ───────────────────────────────────
    spectrum_activated     = Signal(str)    # file_path of the now-active spectrum
    batch_item_selected    = Signal(str)    # file_path from batch queue
```

**All three core views connect to SelectionBus in their constructors:**

```python
# In spectrum canvas, peak table, nuclide browser, calibration dialog,
# activity results, and batch queue — all wired from Phase 1:

bus = SelectionBus.instance()
bus.peak_selected.connect(self._on_peak_selected)
bus.nuclide_selected.connect(self._on_nuclide_selected)
# ... etc.
```

**Bidirectional link table (mandatory for Phase 1, not Phase 3):**

| Source Action | SelectionBus Signal | All Connected Views Respond |
|---|---|---|
| Click peak row in Peak Table | `peak_selected(id)` | Canvas scrolls to peak, flashes ROI; right sidebar shows fit params |
| Click nuclide in Nuclide Browser | `nuclide_selected("Cs-137")` | All Cs-137 assigned peaks flash on canvas; their rows highlight in Peak Table |
| Click energy in Activity Results | `energy_range_selected(keV-5, keV+5)` | Canvas zooms to that energy range |
| Drag ROI edge on canvas | `roi_selected(id)` + `cal_updated()` | Peak Table updates net counts live (50ms debounce) |
| Change calibration point | `cal_updated()` | All reference lines on canvas redraw; Peak Table energies update |
| Click batch queue row | `batch_item_selected(path)` | That spectrum loads in main canvas; its Peak Table populates |

---

## 8. Three-Tier Defaults vs Requirements

### What is missing from both previous documents

Neither the base plan nor the first additions document clearly separates three fundamentally different categories of method choices. The v2 review identifies this distinction and it must be formalized:

**Tier 1 — Recommended Default:** FluxForge's suggested method for general use. Can be changed freely by the user. Persisted in `QSettings`. Default is chosen based on accuracy, robustness, and HPGe best practice.

**Tier 2 — User-Selectable Alternative:** A valid alternative method that FluxForge fully supports. Selectable via a dropdown or settings panel. Choosing it does not produce warnings or restrict functionality.

**Tier 3 — Standards-Mandated Fixed Setting:** Required by a specific standard (e.g., ASTM E181 mandates a least-squares calibration fit; ASTM E261 mandates a specific efficiency formula structure). These settings are locked when Standards mode is active for that standard. Outside Standards mode, the user may use any alternative.

### Implementation — Apply the Three-Tier Classification to Every Method Selector

Every method selector in the GUI (calibration model, peak fitting algorithm, efficiency formula, ID engine, unfolding method) must display its tier visually:

```
Peak Fitting Algorithm:
  ● Gaussian (Levenberg-Marquardt)    [⭐ Default]
  ○ Skewed Gaussian (Exp. left tail)  [User choice]
  ○ Bayesian (emcee MCMC)            [User choice]
  ○ ML-Assisted (CNN proposals)       [User choice]
  ── Standards ──
  ○ ASTM E181 / Gaussian             [🔒 Required in Standards mode]
```

The `[⭐ Default]` badge changes to `[Your default]` once the user has explicitly chosen a non-default option, to make it clear they have deviated from the recommendation.

In batch mode settings, a "Consistency" section shows:
```
[ ] Force all spectra to use the same peak fitting algorithm
    Currently: Gaussian (Levenberg-Marquardt)  [⭐ Default]
```

In every exported report and `.ffs` session file, the provenance section records:
```json
{
  "peak_fitting": {
    "algorithm": "Gaussian",
    "tier": "default",
    "parameters": {"max_iterations": 200, "tolerance": 1e-6}
  },
  "calibration": {
    "model": "Quadratic",
    "tier": "user_choice",
    "parameters": {"polynomial_order": 2}
  }
}
```

---

## 9. ANSI N42.42 Compliant Output

### What to add to Phase 1.4 — File I/O

The base plan lists N42 as a read format only. FluxForge must also **write** ANSI N42.42 (2012) compliant files to ensure interoperability with national laboratories, GADRAS, FRAM, and regulatory agency tools.

**Add N42.42 writer to `fluxforge/io/n42.py`:**

```
Required N42.42 elements for FluxForge output:
  <RadInstrumentData>
    <RadInstrumentInformation>
      — detector model, manufacturer ("FluxForge"), software version
    <RadDetectorInformation>
      — detector type ("HPGe" | "NaI" | "LaBr3" | etc.), crystal dimensions
    <RadMeasurement>
      <MeasurementClassCode>   — "Foreground" | "Background" | "Calibration"
      <StartDateTime>          — ISO 8601
      <RealTimeDuration>       — PT%.3fS format
      <Spectrum>
        <LiveTimeDuration>
        <ChannelData>          — space-separated integer channel counts
        <EnergyCalibration>    — polynomial coefficients
    <AnalysisResults>          — peak fits, nuclide IDs, activities
      <NuclideAnalysisResults>
        <Nuclide>
          <NuclideName>
          <NuclideActivityValue> + <NuclideActivityUncertaintyValue>
          <NuclideIdentifiedIndicator>
      <MethodDescription>      — provenance: algorithm names, version
```

**Validation:** Bundle the N42.42 XSD schema in `resources/schemas/n42_2012.xsd`. Validate every generated file at write time using `lxml`. If validation fails, show a non-dismissable warning dialog listing each failed element and the XSD rule that rejected it. Never silently write a non-compliant file.

**Menu location:** `File → Export → ANSI N42.42 (2012)...`

---

## 10. ASTM C1030 — Plutonium Isotopic Analysis

### What to add to Phase 3 — Standards Framework

ASTM C1030 covers isotopic analysis of plutonium for NDA in nuclear safeguards. This is Expert/Standards-mode only and requires the ASTM standard's specific peak-ratio methodology.

**Add `fluxforge/standards/c1030.py`:**

```
ASTM C1030 — Plutonium Isotopic Analysis Module

Key gamma lines (HPGe only — requires FWHM < 0.8 keV at 661 keV):
  Pu-239:  51.6, 129.3, 143.4, 375.0, 413.7 keV
  Pu-240:  45.2, 160.3 keV
  Pu-241:  148.6, 164.6 keV
  Am-241:  59.5, 125.3 keV

Algorithm (Standards-locked when active):
  1. Identify and fit Pu gamma lines in peak table
  2. Correct for efficiency, branching ratio, and decay (locked to ASTM C1030
     table values — not user-modifiable in Standards mode)
  3. Calculate isotopic ratios: Pu240/Pu239, Pu241/Pu239, Am241/Pu239
  4. Apply age correction: Am-241 ingrowth from Pu-241 beta decay
     (half-life 14.29 years) if source age is known or estimated
  5. Report mass fractions with full uncertainty propagation
  6. Classify: weapons-grade / fuel-grade / reactor-grade

ASTM C1030 Compliance Checks:
  ● FWHM at 413.7 keV > 1.2 keV → RED (resolution insufficient)
  ● Net counts in Pu-240 160.3 keV peak < 1000 → RED (insufficient statistics)
  ● Efficiency calibration uncertainty > 3% → RED (C1030 §8 requirement)

GUI — Location:
  Analysis menu → Pu Isotopics Wizard... (visible in Expert and Standards mode;
  hidden in Simple mode)
  Opens a 4-step wizard dialog (not a panel — this is a modal workflow):
    Step 1: Confirm Pu lines are fitted and assigned in peak table
    Step 2: Enter source age or select "Estimate from Am-241/Pu-241 ratio"
    Step 3: Review isotopic ratios with uncertainty propagation table
    Step 4: Generate C1030-compliant report
```

> **Regulatory note:** Pu isotopic analysis data may be subject to export control (10 CFR Part 810, EAR). FluxForge does not restrict access to this feature — compliance is the user's responsibility. Add this note to the documentation and as a one-time acknowledgment dialog on first use.

---

## 11. ASTM E181 QA Monitoring — FWHM Drift Detection

### What to add to Phase 3.3 — ASTM Compliance Framework

ASTM E181 requires not only per-measurement checks but also **monitoring of FWHM and centroid of check sources over time** to detect electronic drift or resolution degradation. This is a continuous QA function absent from the base plan.

**Add `fluxforge/standards/qa_monitor.py`:**

```python
# Data model (SQLite-backed, stored in ~/.fluxforge/qa_history.db)

QARecord fields:
  timestamp:             datetime
  nuclide:               str        # e.g. "Cs-137"
  energy_keV:            float      # e.g. 661.66
  measured_centroid_keV: float
  measured_fwhm_keV:     float
  measured_fwhm_channels: float
  net_counts:            int
  efficiency:            float
  spectrum_file:         str        # path to the source file

# QAMonitor auto-records when a spectrum is analyzed and contains:
# Cs-137 661 keV | Co-60 1173/1332 keV | Eu-152 lines | Am-241 59.5 keV
# User can also manually tag any measurement as a QA check measurement.

Drift alert thresholds (ASTM E181 §6 informed):
  Centroid drift > 0.5 keV from baseline  → AMBER
  Centroid drift > 1.0 keV               → RED — recalibration required
  FWHM degradation > 10% from baseline   → AMBER
  FWHM degradation > 20%                 → RED — detector/electronics fault
  Efficiency deviation > 5%              → AMBER
  Efficiency deviation > 10%             → RED
```

**GUI — QA History panel (location: `Tools → QA History`):**
- Three sub-plots stacked vertically: FWHM vs. time | Centroid vs. time | Efficiency vs. time
- Each trace = one check source/energy combination; colored differently
- Horizontal reference bands = ASTM E181 acceptance limits
- Points outside limits shown as red diamonds
- User can add text annotations to any point (e.g., "replaced preamp")
- Export: PNG image or CSV of the full QA history

**Integration with the QA & Standards sidebar panel:**
```
QA & STANDARDS (left sidebar section):

  ASTM Status:   E181 [●]  E1297 [●]  E1218 [●]  C1232 [●]  C1030 [●]
  QA Monitor:    FWHM @ 661: 1.82 keV ✅    Drift: +0.02 keV
                 Last check: 2026-03-15 14:22
  [View QA History →]    [Run ASTM Check →]
```

---

## 12. Spectrum Unfolding: Add RMLE Alongside GRAVEL/MAXED/ML

### What to add — additive only

All four unfolding methods are fully implemented and user-selectable. RMLE is the recommended default for new users. GRAVEL and MAXED are not deprecated, downgraded, or flagged as "legacy" — they are first-class methods with documented characteristics. The plugin registry manages them all.

**All four methods in `fluxforge/unfolding/`:**

```
1. GRAVEL  [fully supported — no change]
   Algorithm: Gold iterative deconvolution with background correction
   When preferred: reproducibility with prior results; workflows where GRAVEL
   is a documented method requirement; existing literature comparisons
   Known behaviour: may produce small negative oscillations in low-count
   bins — these are flagged transparently with an amber badge ("N negative bins")
   in the dialog, but are never automatically clamped. The analyst decides.
   User options: max iterations (50–2000), convergence threshold

2. MAXED  [fully supported — no change]
   Algorithm: Maximum Entropy Deconvolution
   When preferred: very low statistics where a physics-motivated prior exists;
   maximum-entropy reconstructions for nuclear physics research
   Known behaviour: strongly prior-dependent — document in UI
   User options: prior spectrum (file or flat default), smoothing factor

3. RMLE  [NEW — recommended default]
   Algorithm: Regularized Maximum-Likelihood Estimation
   Objective: Maximize L(φ) = Σ[d_i · ln(Σ R_ij·φ_j) − Σ R_ij·φ_j] − λ·Ω(φ)
   where d_i = measured counts in channel i
         R_ij = detector response matrix element
         φ_j = true flux in energy bin j (the solution)
         λ·Ω = regularization term (Tikhonov or Total Variation, user-selectable)
   Properties:
     — Non-negativity guaranteed by construction (Poisson MLE)
     — Provides ±1σ uncertainty bands via Fisher information matrix
     — GPU-parallelizable: each iteration is a matrix-vector multiply (CuPy optional)
     — Convergence criterion: relative change in log-likelihood < 1e-6
   Implementation: NumPy + SciPy (CPU) / CuPy if available (GPU)
   User options: λ regularization strength (auto via L-curve or manual),
   regularization type (Tikhonov / Total Variation),
   max iterations (100–5000)

4. ML Seed  [fast approximation — unchanged]
   When preferred: rapid batch unfolding; result used as initial seed for
   RMLE or GRAVEL to reduce iterations needed for convergence (~60% fewer)
   User options: confidence threshold for seed quality acceptance
```

**Unfolding Dialog additions (beyond base plan):**

```
Algorithm selector shows all four methods from the registry.
Tier badges: RMLE [⭐ Default]  GRAVEL [User choice]  MAXED [User choice]  ML Seed [User choice]

New controls added to the dialog:
  ● Regularization Strength slider (λ, RMLE only):
    [Smooth ────────────────── Follows Data]
    Auto (L-curve) is the default; manual override available

  ● Convergence sub-plot (below main unfolding canvas):
    Shows log-likelihood (RMLE) or chi-squared (GRAVEL) vs. iteration number
    Updated in real time during computation

  ● Uncertainty Bands toggle (RMLE only):
    Shows ±1σ shaded region on unfolded spectrum derived from Fisher matrix

  ● Negative Bin Indicator (GRAVEL / MAXED):
    Amber badge: "3 negative bins — hover for details"
    Tooltip explains what causes them and why they are not auto-clamped
    The analyst is shown the raw output, not a silently corrected one

  ● Algorithm Comparison Mode:
    [ ] Run two algorithms simultaneously
    Method A: [RMLE ▾]   Method B: [GRAVEL ▾]
    Both results overlaid on the same canvas with different colors
    Useful for validating that RMLE and GRAVEL agree on main features
```

**Response Matrix panel** — reserved in the unfolding dialog from Phase 1 as a color-map `ImageItem`. Shows R_ij as a 2D heatmap. In Phase 3, load from MCNP/GEANT4 tab-delimited output, analytical HPGe model, or user CSV.

---

## 13. Nuclide Aging and Daughter Product Evolution

### What to add to Phase 2.5 — Activity Calculation

Implement on-the-fly source age correction using Bateman equations. Critical for spent nuclear fuel, aged industrial sources, and NORM.

**Database addition (Phase 1.5 — add to nuclide SQLite schema now):**

```sql
CREATE TABLE decay_chains (
  id             INTEGER PRIMARY KEY,
  parent_id      INTEGER REFERENCES nuclides(id),
  daughter_id    INTEGER REFERENCES nuclides(id),
  branching_ratio REAL,    -- fraction of parent decays producing this daughter
  decay_mode     TEXT      -- "alpha" | "beta-" | "beta+" | "EC" | "IT"
);
-- Populated from ENDF/B-VIII.0 decay data at build time.
-- Bateman solver queries this table to construct decay chains of arbitrary depth.
```

**Source Age sub-panel in Activity Results tab (collapsible):**

```
SOURCE AGE CORRECTION
─────────────────────────────────────────────────────
[ ] Enable age correction

  Source age:  [______] years   [or]  [ Estimate from spectrum ▾ ]

  Estimation methods:
    ● Am-241 / Pu-241 ratio           (for Pu sources)
    ● Ba-137m / Cs-137 ratio          (secular equilibrium check)
    ● Pb-210 / Ra-226 ratio           (NORM)
    ● Custom pair:  [nuclide 1 ▾]  /  [nuclide 2 ▾]

  [Show decay chain diagram]
─────────────────────────────────────────────────────
```

When age correction is enabled:
- Bateman equations solved numerically via `scipy.integrate.odeint` for chains with >2 members.
- The "Line Intensity" column in Peak Table updates to reflect age-corrected intensities.
- All Bq activities and uncertainties in the Activity Results tab recalculate immediately (live, < 200ms for a typical chain).
- A decay chain visualization panel opens below: parent → daughter → granddaughter shown as a flow diagram with colored bars proportional to current calculated activities.

---

## 14. GPS Data Extraction and Field Survey Mapping

### What to add to Phase 1.4 and Phase 2

**Phase 1.4 — GPS extraction in file readers:**
- N42.42: parse `<MeasurementLocationDescription>` and `<GeographicPoint>` elements.
- Ortec CHN / Canberra CNF: check for embedded NMEA GPS strings in file header.
- Store in `Spectrum.metadata`: `{"gps_lat": float, "gps_lon": float, "gps_alt": float, "gps_timestamp": datetime}`.

**Phase 2 — Survey Map panel (optional bottom panel tab):**

```
Tab: "Survey Map"
Visibility: only shown when ≥ 2 loaded spectra contain GPS coordinates

Implementation:
  folium (Python) generates an HTML map with OpenStreetMap tiles.
  Panel renders it in a QWebEngineView widget.

Offline operation:
  Bundle low-resolution world tiles (~50 MB) for offline use.
  Auto-detect internet availability: if online, fetch higher-resolution tiles.
  Map is fully functional offline at regional scale.

Map markers:
  One pin per spectrum with GPS data.
  Pin color = ASTM compliance status (green/amber/red) OR dominant nuclide color.
  Click pin → that spectrum loads in main canvas.
  Marker clustering (Leaflet.markercluster) at low zoom levels.

Export:
  "Export Survey Report" → HTML file with embedded map + activity table
  per GPS point, sorted by activity level. Fully self-contained HTML (no internet).
```

---

## 15. Anti-Automation Bias — Residuals First

### Design principle to apply globally

> **FluxForge Design Rule:** Every fitting operation (energy calibration, efficiency calibration, peak fitting, spectrum unfolding) must display its residuals before displaying its primary result. No fit result is final without an adjacent goodness-of-fit indicator the analyst can visually inspect.

This follows the InterSpec "not a magic button" philosophy and the bGamma residuals plots. Apply it consistently as follows:

**Peak Fitting (Phase 2.2):**
- Every ROI has a **mini residuals sub-plot** directly beneath it on the canvas (a stick plot of `(data − model) / σ` per channel within the ROI).
- Visible by default in Expert and Standards modes; hidden in Simple mode.
- Toggle: `View → Canvas → Peak Residuals → [Off | Compact | Full]`.
- Channels where |residual| > 2σ → amber; > 3σ → red.
- This immediately reveals systematic background mismodeling, unresolved sub-peaks, and pile-up artifacts.

**Energy Calibration (Phase 2.1):**
- Residual plot in the calibration dialog shows measured centroid − fitted polynomial in keV.
- ASTM E181 limit band: ±0.5 keV drawn as horizontal reference region.
- Points outside band highlighted in red; the ASTM E181 checker flags these as RED.

**Efficiency Calibration:**
- Relative residuals as percentages: `(measured − fitted) / fitted × 100%`.
- ASTM E181 ±3% limit band drawn as horizontal reference region.

**Unfolding:**
- Convergence sub-plot always shown below the unfolding canvas (see Section 12).
- Negative bin indicator for GRAVEL/MAXED.

**Reports (Phase 3.5):**
Every generated report must include by default:
```
{{ spectrum_image }}        — canvas screenshot
{{ calibration_curve }}     — energy calibration polynomial plot (NEW)
{{ calibration_residuals }} — energy calibration residuals plot  (NEW)
{{ efficiency_curve }}      — efficiency vs. energy
{{ efficiency_residuals }}  — efficiency residuals plot          (NEW)
{{ residuals_grid }}        — grid of per-ROI peak residuals     (NEW)
{{ peak_table }}
{{ activity_table }}
{{ astm_status_table }}
{{ qa_status_snapshot }}    — FWHM/centroid at time of measurement (NEW)
{{ provenance }}            — algorithm choices, versions, parameters (NEW)
```

The `{{ residuals_grid }}` is a grid of PNG thumbnails, one per fitted ROI, captured by the canvas renderer. This makes reports self-contained for peer review without reopening the data file.

---

## 16. Digital Twin Hardware Dashboard

### What to reserve in Phase 1, implement in Phase 4

**Phase 1 actions (reserve the space):**

Add a **Dashboard tab** alongside the Spectrum tab in the main canvas area (Zone C):
```
Zone C (main canvas area) — tab structure from Phase 1:

  [ 📊 Spectrum ] [ ⚙ Dashboard ]
  ──────────────────────────────────────────────────────
  (Phase 1–3: Dashboard tab shows placeholder panel with a description
   of what it will contain in Phase 4)
```

Add a **hardware status LED** to the status bar (Zone F) that is always visible regardless of which tab is active:
```
Status bar:   [filename.n42]  [ch:1247 | 661.7 keV | 8,432 cts]  [●●● 62%]  [● NO DEVICE]
                                ↑ cursor readout                   ↑ progress  ↑ always visible
```
Clicking the LED opens the Dashboard tab.

**Phase 4 full implementation:**

```
Dashboard tab contents:
┌──────────────────────────────────────────────────────────────┐
│  DETECTOR: [device name]  [model]  [serial]        ● LIVE   │
├───────────────────────┬──────────────────────────────────────┤
│  REAL-TIME METRICS    │  PULSE SHAPE MONITOR                 │
│  Input:   45,231 cps  │  (digital oscilloscope waveform      │
│  Output:  44,890 cps  │   from HAL device, if supported)     │
│  Dead:    0.75%       │                                      │
│  Live:    00:14:32    │                                      │
│  HV:      3500 V ████ │                                      │
│  Temp:    77.2 K ████ │                                      │
├───────────────────────┴──────────────────────────────────────┤
│  COUNT RATE HISTORY  (sparkline — last 30 min)               │
├──────────────────────────────────────────────────────────────┤
│  QA  FWHM @ 661: 1.82 keV ✅   Centroid drift: +0.02 keV    │
│      Efficiency @ 661: 0.0312 ✅                             │
└──────────────────────────────────────────────────────────────┘

Predictive features (Phase 4+):
  ● Count rate trend → estimated time to target counts for current ROI
  ● Dead time trend  → saturation warning if dead time is rising
  ● FWHM trend       → predicted recalibration date from QA history slope
```

---

## 17. Bayesian Nuclide ID — "Guess" Mode

### What to add to Phase 2.4 — Automatic Peak Search

After the second-difference auto-search, add a **Bayesian Library Matching** pass as a second, independent identification engine in the plugin registry.

**Algorithm:**

```
Prior:      P(nuclide) ∝ uniform across active library
            OR P(nuclide) ∝ user_defined_prior
            User sets prior via: Analysis → Set Nuclide Prior

Likelihood: For measured peaks {E_1,...,E_n} and candidate nuclide C
            with library lines {L_1,...,L_m}:

  P(data | C) = Π P(E_i matches best L_j) × P(unmatched lines | C)
  where:
  P(E_i matches L_j) = Gaussian(E_i; μ=L_j, σ=σ_cal)
                        weighted by library line intensity

Posterior:  P(C | data) ∝ P(data | C) × P(C), normalized over all nuclides

Output:
  ● Top-5 candidate nuclides ranked by posterior probability (%)
  ● Confidence score: posterior / sum(all posteriors)
  ● "Unexplained peaks" list (peaks not accounted for by top-5)
  ● "Missing expected peaks" list (strong lines of top candidate absent)
    — this is critical for distinguishing a true ID from a coincidental match

Computational cost: < 200ms for 4000-nuclide library + 50 peaks (pure NumPy)
```

**GUI — Location:**
```
Toolbar:  [🔍 Auto-Find Peaks]  [⚛ Bayesian ID]  [🤖 ML Analysis]
                                    ↑ NEW button
```

Clicking Bayesian ID:
1. Runs the algorithm in a `QThread` worker (< 200ms — no visible loading state needed for most spectra).
2. Opens a **Bayesian ID Results panel** in the right sidebar (not a modal dialog — the analyst needs to see the canvas at the same time):
```
BAYESIAN NUCLIDE ID
──────────────────────────────
Top candidates:
  1. Cs-137     ████████████  72.4%
  2. Ba-137m    ████          18.1%
  3. Co-60      ██             6.2%
  4. Eu-152     █              2.1%
  5. Am-241     ░              1.2%

[Click to assign all peaks for a nuclide]

⚠ Unexplained peaks: 485.2 keV, 1028.4 keV
💡 Missing Cs-137 lines: 283.5 keV (intensity 0.6%)
──────────────────────────────
Prior: [Uniform ▾]  [Set custom prior]
```

Clicking a nuclide in the results list:
- Assigns it to all matching peaks in the Peak Table.
- Draws reference lines for all its gammas on the canvas.
- The assignment is undoable (Ctrl+Z via QUndoStack).

---

## 18. GUI Panel Positioning from the Synthesis Plan

### Specific changes the Synthesis Plan's component table implies

The Synthesis Plan's component priority table contains three concrete panel hierarchy deltas worth formalizing. The base plan's Zone A–F layout is kept unchanged; these are additions within that structure.

### 18.1 — QA & Standards as a Primary Left-Sidebar Section

The Synthesis Plan names "library, calibration, and QA" as the three primary docking widgets, placing QA at the same tier as the Nuclide Library. The base plan buries ASTM/QA in the right sidebar. Promote it.

**Change to Phase 1.2 — Left Sidebar (Zone B), add a fourth section:**

```
Zone B — Left Sidebar, four sections:

  ┌─────────────────────────────────┐
  │ 📁  FILE BROWSER    [─]         │  open spectra tree with status icons
  ├─────────────────────────────────┤
  │ ⚛   NUCLIDE BROWSER [─]         │  search-as-you-type + instant overlay
  ├─────────────────────────────────┤
  │ 📊  ANALYSIS RESULTS[─]         │  live peak table summary
  ├─────────────────────────────────┤
  │ ✅  QA & STANDARDS  [─]   NEW   │  ASTM status + QA monitor summary
  └─────────────────────────────────┘
```

**QA & Standards section contents:**
```
ASTM Status:   E181 [●]  E1297 [●]  E1218 [●]  C1232 [●]  C1030 [●]
               (● = green/amber/red — click any dot to expand in right sidebar)
QA Monitor:    FWHM @ 661: 1.82 keV ✅    Centroid drift: +0.02 keV
               Last check: 2026-03-15 14:22
[View QA History →]     [Run ASTM Check →]
```

The **right sidebar (Zone E)** continues to show the *full expanded detail* when an ASTM status dot is clicked. The left sidebar section is the summary; the right sidebar is the detail. These are complementary, not duplicates.

### 18.2 — Dashboard Tab in the Main Canvas Area

The hardware dashboard occupies the **main canvas area** (Zone C) as a peer tab of the Spectrum view, not a subordinate panel. This matches the Synthesis Plan's "Digital Twin" as a first-class workspace component.

```
Zone C — Main canvas area tab structure:

  [ 📊 Spectrum ] [ ⚙ Dashboard ]
```

See Section 16 for full Dashboard implementation.

### 18.3 — Residuals Plots Required in Every Generated Report

The Synthesis Plan specifies "rich HTML-based reports with **embedded** residuals plots." The `{{ residuals_grid }}` template variable is mandatory in the default report template, not optional.

See Section 15 for full report template specification.

---

## 19. Consolidated Priority Action List

The following replaces and supersedes the base plan's 23-item list. Items are grouped by phase. **Stage 0 items must be completed before any Phase 1 coding begins.**

### Stage 0 (Before any coding)

| # | Action | Area | Priority |
|---|---|---|---|
| S0-1 | Create GitHub milestones M0–M6 | docs | p0 |
| S0-2 | Create all labels (area, type, priority, platform, status) | docs | p0 |
| S0-3 | Create 7 issue templates in `.github/ISSUE_TEMPLATE/` | docs | p0 |
| S0-4 | Create GitHub Project with 6 board columns | docs | p0 |
| S0-5 | Open ADR-001: GUI stack and rendering architecture | type/adr | p0 |
| S0-6 | Open ADR-002: Additive capability policy | type/adr | p0 |
| S0-7 | Open ADR-003: Standards mode and three-tier defaults | type/adr | p0 |
| S0-8 | Open 8 epic issues with child issue stubs | type/epic | p0 |
| S0-9 | Open first 20 concrete issues (§0.7) | type/feature | p0 |
| S0-10 | Create `docs/adr/` directory and write ADR-001 through ADR-003 | docs | p0 |

### Phase 1 — Foundation

| # | Action | Area | Priority |
|---|---|---|---|
| 1-1 | Establish project structure with all 9 top-level modules including `unfolding/` and `plugins/` | area/core | p0 |
| 1-2 | Implement `PluginRegistry[T]` generic class and `PluginRegistries` singletons. Wire bootstrap. | area/core | p0 |
| 1-3 | Scaffold `MainWindow` with all 6 docking zones (A–F). QSS dark/light themes. QSettings layout save/restore. | area/gui | p0 |
| 1-4 | Implement `ModeManager` (Simple/Expert/Standards) with mode switcher toolbar widget. | area/gui | p0 |
| 1-5 | Implement `SelectionBus` singleton. Wire Canvas, Peak Table, Nuclide Browser to it. | area/gui | p0 |
| 1-6 | Implement `SpectrumCanvas` ABC + PyQtGraph backend + `HierarchicalSpectrumBuffer`. Full interaction model (zoom/pan/crosshair/context menus). | area/gui | p0 |
| 1-7 | Implement `Spectrum` dataclass with HAL fields. `.ffs` session file save/restore. | area/core | p0 |
| 1-8 | Implement reader factory + N42/CHN/SPC/CNF/SPE/CSV readers. Drag-and-drop. Recent files. GPS metadata extraction. | area/io | p0 |
| 1-9 | Implement N42.42 (2012) writer with XSD schema validation. | area/io | p0 |
| 1-10 | Build nuclide SQLite DB (ENDF/B-VIII.0) with `decay_chains` table. Nuclide search panel. | area/core | p0 |
| 1-11 | Define `MCADevice` ABC + `DeviceStatus` + mock device + device registry. Wire HAL fields into `Spectrum`. | area/hal | p0 |
| 1-12 | Add QA & Standards fourth section to left sidebar (Zone B) — stub content. | area/gui | p1 |
| 1-13 | Add Dashboard tab to Zone C + hardware LED to status bar. Both are placeholders. | area/gui | p1 |
| 1-14 | Add Vispy backend stub (registered but disabled by default). | area/gui | p2 |

### Phase 2 — Core Analysis

| # | Action | Area | Priority |
|---|---|---|---|
| 2-1 | Implement unified calibration dialog: live embedded canvas + energy/FWHM calibration + residuals + Chi-squared. | area/gui | p0 - complete in repo |
| 2-2 | Implement slider-bar quick calibration mode (PeakEasy-inspired) alongside full calibration. | area/gui | p1 |
| 2-3 | Implement non-linear deviation pairs in calibration fine-tuning sub-tab. | area/core | p1 |
| 2-4 | Implement ROI drag + real-time Gaussian peak fitting (Levenberg-Marquardt). Register in `PluginRegistries.peak_fitters`. | area/core | p0 |
| 2-5 | Implement skewed Gaussian fitter. Register in peak_fitters registry. | area/core | p1 |
| 2-6 | Implement Bayesian peak fitter (emcee/MCMC). Register in peak_fitters registry. | area/core | p2 |
| 2-7 | Implement Peak Table with all columns. SelectionBus full sync. Three-tier method badge display. | area/gui | p0 |
| 2-8 | Implement auto-peak search (second-difference, Ctrl+A). Review dialog. | area/core | p0 |
| 2-9 | Implement Bayesian Library Matching ID engine. "Bayesian ID" toolbar button. Results panel. Register in id_engines registry. | area/core | p1 |
| 2-10 | Implement undo/redo via `QUndoStack` for all peak operations. | area/gui | p0 |
| 2-11 | Implement efficiency calibration dialog with log-polynomial + power law + spline. Residuals plot. Monte Carlo import. | area/gui | p0 |
| 2-12 | Implement activity calculation with uncertainty propagation. Activity Results tab. | area/core | p0 |
| 2-13 | Implement source age correction (Bateman equations). Source Age sub-panel in Activity Results. Decay chain visualization. | area/core | p1 |
| 2-14 | Implement background subtraction (three methods). | area/core | p0 |
| 2-15 | Implement multi-spectrum tabs (QTabBar above canvas). Foreground/background/secondary overlay. | area/gui | p1 |
| 2-16 | Implement mini residuals sub-plot beneath every ROI on canvas (Expert/Standards mode). | area/gui | p1 |
| 2-17 | Implement Survey Map panel using folium + offline tile bundle. GPS extraction wired in. | area/gui | p2 |
| 2-18 | Implement three-tier method badge display in all method selectors. Provenance in `.ffs` and reports. | area/gui | p1 |

### Phase 3 — Advanced Analysis and Standards

| # | Action | Area | Priority |
|---|---|---|---|
| 3-1 | Implement `StandardsModule` ABC + standards registry. Standards mode GUI locking mechanism (QLabel + lock icon). | area/standards | p0 |
| 3-2 | Implement ASTM E181 module: calibration checks + counting statistics checks + residuals-based checks. | area/standards | p0 |
| 3-3 | Implement ASTM E1297 module: MDA calculation (Currie method). Flag activities below MDA. | area/standards | p0 |
| 3-4 | Implement ASTM E1218 + C1232 modules. | area/standards | p1 |
| 3-5 | Implement `QAMonitor` class + SQLite history. Auto-record check-source lines. QA History panel. Wire drift alerts to QA & Standards sidebar. | area/standards | p0 |
| 3-6 | Implement ASTM C1030 Pu isotopics wizard (4-step modal, Expert/Standards only). | area/standards | p2 |
| 3-7 | Implement ASTM E261 module stub (neutron fluence — if in scope). | area/standards | p3 |
| 3-8 | Implement unfolding registry + GRAVEL method. Full unfolding dialog with convergence sub-plot. | area/unfolding | p0 |
| 3-9 | Add MAXED method. Register. | area/unfolding | p1 |
| 3-10 | Add RMLE method (NumPy/SciPy CPU; CuPy GPU optional). Register as default. Add λ slider, uncertainty bands, comparison mode. | area/unfolding | p1 |
| 3-11 | Add ML Seed method. Wire as RMLE/GRAVEL initializer. Register. | area/unfolding | p2 |
| 3-12 | Implement response matrix loader (MCNP/GEANT4 tab-delimited, analytical HPGe model, user CSV). | area/unfolding | p1 |
| 3-13 | Implement ML peak analysis engine (PyTorch → ONNX). Optional GPU inference via CuPy/CUDA. Register in id_engines. | area/ml | p2 |
| 3-14 | Implement Jinja2 report engine. Standard lab template with all 9 required sections (including residuals_grid, provenance). | area/reporting | p0 |
| 3-15 | Implement ASTM-compliant report template. Batch summary template. PDF export (WeasyPrint). | area/reporting | p1 |
| 3-16 | Implement full batch analysis queue panel: ProcessPoolExecutor, progress bars, per-spectrum JSON output. | area/gui | p1 |

### Phase 4 (Future) — MCA Acquisition

| # | Action | Area | Priority |
|---|---|---|---|
| 4-1 | Implement Dashboard tab full content: Digital Twin panel layout (metrics, sparkline, QA status row). | area/hal | p1 |
| 4-2 | Implement Devices panel (replaces placeholder in Zone B). Discovery dialog. Device thumbnail cards. | area/hal | p1 |
| 4-3 | Implement at least one real HAL driver (e.g., generic USB HID MCA). | area/hal | p2 |
| 4-4 | Implement Spectrogram tab (2D time-energy color map). Feed from live MSS acquisition. | area/gui | p2 |
| 4-5 | Implement Windows .exe (PyInstaller) + Linux AppImage packaging. GitHub Actions CI for releases. | area/packaging | p0 |

---

## Quick Reference: Category Winners

| Category | Winner | Rationale |
|---|---|---|
| GUI widget-level implementation detail | **Base Plan** | Only plan with exact widget names, color tokens, interaction bindings |
| GitHub project infrastructure | **v2 Review** | Milestones, labels, epic structure, issue ordering |
| Plugin/registry architecture | **v2 Review** | Entirely absent from base plan |
| Three-mode GUI (Standards mode) | **v2 Review** | Base plan has only two modes |
| Architecture Decision Records | **v2 Review** | Not present in either other document |
| PyQtGraph-first rendering stance | **v2 Review** | Base plan additions doc was Vispy-first — v2 corrects this |
| Three-tier defaults formalization | **v2 Review** | Neither other plan distinguishes default / user-choice / standards-locked |
| HAL interface design | **Synthesis Plan** | Fullest abstract base class spec |
| RMLE algorithm spec | **Synthesis Plan** | Mathematical detail |
| QA drift monitoring thresholds | **Synthesis Plan** | Specific AMBER/RED thresholds |
| Bayesian ID algorithm | **Synthesis Plan** | Statistical detail including "missing lines" diagnostic |
| ASTM C1030 Pu isotopics | **Synthesis Plan** | Not in base plan at all |
| Unfolding additive policy | **This document** | Treats GRAVEL/MAXED as first-class, not fallbacks |
| Hierarchical spectrum buffer | **Synthesis Plan** | 5-level pre-computation strategy |
| GPS survey map implementation | **Synthesis Plan + additions** | folium + offline tile bundle spec |
| Distribution and packaging | **Base Plan** | AppImage + .exe + pip install strategy |
| Priority action list | **This document** | Supersedes base plan's 23-item list with phase-structured table |

---

*FluxForge — `chore/folder-audit-native-gui-review-20260317` — March 2026*
*Base plan: GUI Implementation Guide | Merged from: Technical Synthesis Plan, v2 Review*
