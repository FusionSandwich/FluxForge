# FluxForge: Final Additions & Changes Document
## Base Plan — GUI Implementation Guide + Previous Additions Doc
## Additions Source — Uploaded v2 Merged Plan

> **Which plan is the better base?**
>
> The **GUI Implementation Guide + Additions Doc** (my prior two documents) remains the stronger
> execution base. It specifies exact widget types, color tokens, interaction bindings, canvas
> rendering layers, SQLite schemas, performance targets (60 FPS, <50 ms response), Jinja2 template
> variables, packaging commands, and a 23-item prioritized action list. The uploaded v2 merged plan
> is better written at the **governance and policy level**, contributing seven structural items that
> the base plan genuinely missed. Those seven items are merged in here.
>
> **The seven genuine additions from the uploaded v2 plan:**
> 1. Stage 0 — GitHub project setup as an explicit, mandatory first phase
> 2. A third GUI mode: `Standards` (locked workflows), not just Simple/Expert
> 3. Plugin/registry layer defined in Phase 1 as a structural property of the codebase
> 4. Architecture Decision Records (ADRs) as a formal process
> 5. Explicit separation of: recommended defaults / user-selectable alternatives / standards-mandated fixed settings
> 6. `fluxforge/unfolding/` as a dedicated top-level package (not buried in `core/`)
> 7. The additive capability policy and user-choice policy formalized as written project rules, not just applied case by case

---

## Table of Contents

1. [Four Governing Rules — Project-Wide Policy](#1-four-governing-rules--project-wide-policy)
2. [Stage 0 — GitHub Project Setup (Must Happen First)](#2-stage-0--github-project-setup-must-happen-first)
3. [Standards Mode — Third GUI Mode Added to ModeManager](#3-standards-mode--third-gui-mode-added-to-modemanager)
4. [Plugin Registry Layer in Phase 1](#4-plugin-registry-layer-in-phase-1)
5. [Architecture Decision Records (ADRs)](#5-architecture-decision-records-adrs)
6. [Separating Defaults, Alternatives, and Mandated Settings](#6-separating-defaults-alternatives-and-mandated-settings)
7. [fluxforge/unfolding/ as a Top-Level Package](#7-fluxforgeunfolding-as-a-top-level-package)
8. [PyQtGraph-First Rendering Strategy (Soften the Vispy Swap)](#8-pyqtgraph-first-rendering-strategy-soften-the-vispy-swap)
9. [Confirmed Carryovers from the Previous Additions Doc](#9-confirmed-carryovers-from-the-previous-additions-doc)
10. [Full Revised Project Structure](#10-full-revised-project-structure)
11. [Full Revised Priority Action List](#11-full-revised-priority-action-list)

---

## 1. Four Governing Rules — Project-Wide Policy

These four rules are the most important addition from the uploaded plan. They are not feature
additions — they are **constitutional rules** that govern every decision in the rest of the
roadmap. They must be written into `CONTRIBUTING.md` and into the first ADR before Phase 1
code is written.

### Rule 1 — Issue-First Execution

No significant code work begins without a corresponding GitHub issue. The roadmap must be
executable through issues, not left as a static design document. Stage 0 (Section 2 of this
document) defines exactly how to set this up.

### Rule 2 — Additive Capability Policy

FluxForge **never removes a valid analytical capability** when adding a new one, unless the old
one is broken beyond repair or creates a security/correctness risk. This applies universally:

| Addition | What Must Be Kept |
|---|---|
| Add RMLE unfolding | Keep GRAVEL and MAXED as fully supported options |
| Add Bayesian ID | Keep manual assignment and library search |
| Add ML peak proposals | Keep second-difference search and manual ROI |
| Add Vispy render backend | Keep PyQtGraph; make both selectable |
| Add quick slider calibration | Keep full calibration workflow |
| Add Expert mode | Keep Simple mode — do not merge them |
| Add Standards mode | Keep Expert mode — do not replace free-form analysis |
| Add log-polynomial efficiency | Keep power-law, polynomial, exponential models |

**This rule applies to every PR review.** A PR that deletes a prior method as part of adding a
new one must be rejected unless a written justification explains why the old method cannot
coexist (e.g., mutually exclusive data models, not just "the new one is better").

### Rule 3 — User-Choice Policy

Where multiple scientifically defensible methods exist for the same task, FluxForge must:

1. Expose all methods in the UI — never hide a supported method
2. Mark one as the **recommended default** with a label such as `[Recommended]` or a filled
   star icon — do not leave users without guidance
3. Show a one-line **explanation tooltip** for each method describing: when to use it, known
   limitations, and relevant citation or standard
4. **Remember the user's last choice** per workflow via `QSettings`, so power users do not
   re-select their preferred method on every session
5. **Batch-mode consistency controls** — if the user runs batch analysis, they must be able to
   lock a method for the whole batch rather than having each spectrum inherit a different default
6. **Report and session provenance** — every exported report and every `.ffs` session file must
   record which method and parameter values were used for each analytical step, so results are
   reproducible

### Rule 4 — Standards-Locked Workflows

When a workflow must comply with a specific standard (ASTM E181, E261, E1297, E1218, C1232,
C1030, or future additions), FluxForge must provide a **Standards Mode** variant of that
workflow that:

- Locks the equations and parameters to exactly what the standard specifies
- Prevents the user from selecting alternate methods that would violate the standard
- Displays a banner identifying which standard is active and its current year/edition
- Validates inputs and outputs against the standard's acceptance criteria
- Generates a report that includes all fields required by the standard
- Logs any deviation from the standard's prescribed sequence as a compliance warning

Outside Standards Mode, users are free to use any combination of methods. The two modes
must coexist cleanly — switching from Expert Mode to Standards Mode never destroys the
user's current analysis state, it only adds constraints on top of it.

---

## 2. Stage 0 — GitHub Project Setup (Must Happen First)

This is the single largest structural gap in the prior documents. The uploaded v2 plan is
correct that issue creation must be a formal first stage, not an afterthought. Stage 0 must
be completed before any Phase 1 code is committed on the branch.

### Repository Status — 2026-03-30

- Completed in-repo: `CONTRIBUTING.md`, `docs/adr/ADR-001` through `ADR-007`,
  `.github/ISSUE_TEMPLATE/`, `.github/project-management/`, the sync workflow, and the
  `tests/spectra/` scaffold.
- Ordered execution tracker: `.github/project-management/implementation_steps.json`.
- Canonical status tracker: `docs/ROADMAP_EXECUTION_STATUS.md`.
- Stage 0 live tracker application was verified complete on GitHub on 2026-03-30.
- Phase 1 implementation is repo-complete and formally complete in sequence.
- Phase 2.1 is now implemented in the redesigned Qt path under `src/fluxforge/core/`
  and `src/fluxforge/gui/`, including the unified calibration dialog, residual-first
  plots, ASTM E181 order locking, and the native review probe.
- The current roadmap next step is Phase 2.2, the additive quick slider calibration mode.
- GUI redesign status in-repo: Phase 1 items `1.3` through `1.18` now exist in the
  repository under `src/fluxforge/gui/`, including the `.ffs` session path, reader
  factory, validated N42 export, SQLite nuclide search, and overlay wiring, while
  `src/fluxforge_gui/` is retained as the legacy/archive fallback rather than removed.
- Legacy GUI planning references are now explicitly archived in `docs/GUI_PLAN_old.md`
  and `docs/GUI_CAPABILITY_PROGRAM_old.md` so the controlling GUI plan set stays limited
  to this document, `docs/FluxForge_Additions_v3_Final.md`, and
  `docs/FluxForge_Improvement_Guide.docx`.

### 2.1 Create Milestones

Create these milestones in the GitHub repository. Each phase of the roadmap maps to one
milestone.

| Milestone | Covers |
|---|---|
| `M0 — Planning & Backlog` | Stage 0: project setup, ADRs, architecture scaffolding |
| `M1 — GUI Shell & Architecture` | Phase 1: main window, canvas, HAL interface, file I/O, nuclide DB |
| `M2 — Core Analysis Parity` | Phase 2: calibration, peak fitting, nuclide ID, activity calculation |
| `M3 — Advanced Analysis` | Phase 3: unfolding, ML engine, batch mode, reporting |
| `M4 — Standards & QA` | Phase 3 continuation: ASTM modules, QA monitor, C1030 |
| `M5 — MCA Foundations` | Phase 4: HAL drivers, live acquisition, spectrogram |
| `M6 — Packaging & Release` | Distribution: AppImage, Windows .exe, CI/CD, documentation |

### 2.2 Create Labels

Apply this label taxonomy. Every issue must have exactly one `area/` label, one `type/` label,
and one `priority/` label. Platform labels are added when the issue is platform-specific.

**Area labels** (what part of the codebase):
```
area/gui          area/core         area/io
area/ml           area/unfolding    area/standards
area/hal          area/reporting    area/performance
area/docs         area/testing      area/packaging
area/registry     area/calibration  area/nuclide-db
```

**Type labels** (what kind of work):
```
type/epic         type/feature      type/refactor
type/bug          type/design       type/research
type/validation   type/adr
```

**Priority labels** (urgency/blocking):
```
priority/p0-blocking    priority/p1-critical
priority/p2-normal      priority/p3-nice-to-have
```

**Platform labels** (when platform-specific):
```
platform/linux    platform/windows  platform/both
```

**Special labels**:
```
good-first-issue    blocked    needs-design-review    standards-locked
```

### 2.3 Create Issue Templates

Create these templates in `.github/ISSUE_TEMPLATE/`:

**`feature.yml`** — for new capabilities:
```yaml
Fields: Title | Area | Type | Priority | Platform
        | Description | Motivation/Use Case
        | Proposed Implementation | Alternatives Considered
        | Additive Policy Check (does this remove any existing capability?)
        | Standards Impact (does this affect any ASTM module?)
        | Acceptance Criteria | Related Issues
```

**`epic.yml`** — for multi-issue work streams:
```yaml
Fields: Title | Area | Milestone | Description
        | Child Issues (checklist) | Definition of Done
        | Notes on additive policy across the epic
```

**`bug.yml`** — for defects:
```yaml
Fields: Title | Area | Priority | Platform | Severity
        | Steps to Reproduce | Expected vs Actual | Stack Trace
        | Spectrum File (attach if relevant) | Regression Test Required?
```

**`gui_ux.yml`** — for interface-specific work:
```yaml
Fields: Title | Screen/Panel affected | Mode (Simple/Expert/Standards)
        | Current Behavior | Desired Behavior | Mockup/Screenshot
        | Accessibility Impact | Performance Impact
```

**`standards_validation.yml`** — for ASTM/compliance work:
```yaml
Fields: Title | Standard (E181/E261/E1297/...) | Edition/Year
        | Section of Standard | Required Change | Validation Test Plan
        | Locked Parameters (list equations/thresholds that must not be user-editable)
```

**`performance.yml`** — for speed/memory issues:
```yaml
Fields: Title | Area | Platform | Hardware Spec | Measured vs Target
        | Profile Data | Proposed Fix | Regression Benchmark Required?
```

**`adr.yml`** — for architecture decisions:
```yaml
Fields: Title | Decision | Status (Proposed/Accepted/Superseded)
        | Context | Decision Rationale | Consequences | Alternatives Rejected
```

### 2.4 Create Project Board

Set up a GitHub Project (table/board view) with these columns:

```
Backlog → Ready → In Progress → In Review → Blocked → Done
```

Configure automation:
- Issue opened → auto-add to `Backlog`
- PR opened and linked to issue → move issue to `In Review`
- PR merged → move issue to `Done`
- Issue labeled `blocked` → move to `Blocked`

### 2.5 Create Initial Epics

Open these eight issues immediately as epics (`type/epic`):

| # | Epic Title | Milestone | Priority |
|---|---|---|---|
| E1 | GUI shell and docking framework | M1 | P0 |
| E2 | Spectrum canvas and renderer abstraction | M1 | P0 |
| E3 | File I/O, session model, and N42.42 export | M1 | P0 |
| E4 | Nuclide library database | M1 | P1 |
| E5 | Core spectroscopy workflows (calibration, fitting, ID, activity) | M2 | P0 |
| E6 | Standards and QA framework | M4 | P1 |
| E7 | Unfolding workspace (GRAVEL + MAXED + RMLE + ML) | M3 | P1 |
| E8 | HAL, live acquisition, and dashboard foundation | M5 | P2 |
| E9 | Reporting engine and packaging | M6 | P1 |
| E10 | Plugin/registry architecture | M1 | P0 |

### 2.6 First Issue Set — Open Immediately

These are the concrete issues to open as the first development cycle begins.
Each issue references its parent epic in brackets.

| # | Issue Title | Epic | Area | Priority |
|---|---|---|---|---|
| 1 | ADR: adopt PySide6 and additive capability policy | — | area/adr | P0 |
| 2 | ADR: renderer abstraction — PyQtGraph first, Vispy optional | — | area/adr | P0 |
| 3 | ADR: Standards-mode design and user-choice policy | — | area/adr | P0 |
| 4 | ADR: plugin/registry architecture — fitters, unfolding, calibration, ID | — | area/adr | P0 |
| 5 | Scaffold `MainWindow` with dockable zones A–F | E1 | area/gui | P0 |
| 6 | Implement `ModeManager` (Simple / Expert / Standards) | E1 | area/gui | P0 |
| 7 | Implement `SelectionBus` for coordinated multiple views | E1 | area/gui | P0 |
| 8 | Implement plugin registry base classes (all six registries) | E10 | area/registry | P0 |
| 9 | Implement `SpectrumCanvas` abstraction interface | E2 | area/gui | P0 |
| 10 | Implement PyQtGraph backend for `SpectrumCanvas` | E2 | area/gui | P0 |
| 11 | Add Vispy backend stub and renderer capability flags | E2 | area/gui | P1 |
| 12 | Define `Spectrum` session model and `.ffs` session file | E3 | area/io | P0 |
| 13 | Define `MCADevice` abstract base class and mock device | E8 | area/hal | P0 |
| 14 | Implement reader factory and N42 / CHN / SPC / CNF / SPE / CSV readers | E3 | area/io | P0 |
| 15 | Implement N42.42 (2012) export writer with XSD validation | E3 | area/io | P1 |
| 16 | Build nuclide library SQLite schema with decay-chain support | E4 | area/nuclide-db | P0 |
| 17 | Implement nuclide search panel with FTS5 and instant overlay | E4 | area/gui | P0 |
| 18 | Implement peak table and canvas SelectionBus synchronization | E5 | area/gui | P1 |
| 19 | Implement calibration workspace shell with shared main canvas | E5 | area/calibration | P1 |
| 20 | Implement Jinja2 report engine skeleton | E9 | area/reporting | P1 |
| 21 | Scaffold ASTM standards module interface | E6 | area/standards | P1 |
| 22 | Scaffold QA monitor storage and history plot | E6 | area/standards | P1 |
| 23 | Scaffold unfolding workspace with algorithm registry | E7 | area/unfolding | P1 |
| 24 | Add test harness: sample spectra, regression snapshots, CI baseline | — | area/testing | P0 |
| 25 | Add dark/light theme QSS files and ModeManager theme hook | E1 | area/gui | P1 |

---

## 3. Standards Mode — Third GUI Mode Added to ModeManager

### What the uploaded v2 plan adds

The uploaded plan introduces a **third GUI mode**: `Standards`. The prior documents had
`Simple` and `Expert`. The uploaded plan correctly identifies that these two modes handle
*visual complexity* but not *workflow locking*. A third mode is needed that restricts
parameters and sequences to what a specific standard requires.

### Change to Phase 1.2 / Phase 4 — ModeManager

**Replace the two-mode system with a three-mode system throughout the plan.**

```python
# gui/mode_manager.py — revised

class GUIMode(Enum):
    SIMPLE    = "simple"    # Reduced panels, beginner-friendly
    EXPERT    = "expert"    # All panels visible, full user choice
    STANDARDS = "standards" # Panels visible as Expert, plus workflow locking

class ModeManager(QObject):
    mode_changed = Signal(GUIMode)
    standard_changed = Signal(str)  # e.g., "ASTM E181-23"

    def set_mode(self, mode: GUIMode, standard: str = None):
        """
        mode=STANDARDS requires a standard string, e.g. "ASTM_E181".
        The standards module registry looks up which parameters to lock.
        """
```

**Toolbar mode switcher — revised three-button layout:**

```
[FluxForge]  File  Edit  View  Analysis  Calibration  Tools  Help
                                              [ SIMPLE | EXPERT | STANDARDS ▾ ]  [🌙]
                                                                    ↑
                                           dropdown lists active standards modules:
                                           ● ASTM E181-23
                                           ● ASTM E1297-23
                                           ● ASTM C1030-22
                                           (check to activate; multiple allowed)
```

**Behavioral difference between Expert and Standards modes:**

| Aspect | Expert Mode | Standards Mode |
|---|---|---|
| Panels visible | All panels | Same as Expert |
| Method selector | All methods selectable | Only methods permitted by the active standard |
| Parameter inputs | Fully editable | Locked inputs shown with a padlock icon 🔒 |
| Polynomial order | Any 1–5 | Locked to standard requirement (e.g., ≤2 for E181 energy cal) |
| Report fields | Customizable template | Required fields enforced; missing fields flagged |
| Mode banner | None | Amber banner at top of canvas: "Standards Mode — ASTM E181-23 Active" |
| Deviation from standard | User's problem | Flagged as compliance warning in ASTM panel |

**Switching between modes must never destroy analysis state.** Switching Expert → Standards
adds constraints. Switching Standards → Expert removes constraints. The underlying data,
fitted peaks, and calibration coefficients are not altered by the switch.

**In the ASTM compliance panel (left sidebar QA & Standards section),** when Standards Mode
is active, the status dots change from advisory to mandatory:
- Green = required check passed
- Amber = check passed with a minor deviation the standard permits
- Red = required check failed — report generation is blocked until resolved

---

## 4. Plugin Registry Layer in Phase 1

### What the uploaded v2 plan adds

The uploaded plan correctly identifies that exposing multiple methods per task (Rule 2 and
Rule 3) should be a **structural property of the codebase from Phase 1**, not a UI pattern
bolted on later. The mechanism for this is a plugin registry for each analytical dimension.

### Addition to Phase 1.1 — Project Structure

Add `fluxforge/registry/` as a new top-level package, created in Phase 1 before any
analytical implementations:

```
fluxforge/
└── registry/
    ├── __init__.py
    ├── base.py              # PluginRegistry base class
    ├── fitter_registry.py   # Peak fitting algorithms
    ├── unfold_registry.py   # Unfolding methods
    ├── cal_registry.py      # Calibration models
    ├── id_registry.py       # Nuclide identification engines
    ├── standards_registry.py # ASTM/standards modules
    └── render_registry.py   # Spectrum render backends
```

**`PluginRegistry` base class:**

```python
# registry/base.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

@dataclass
class PluginMeta:
    key: str                    # Unique identifier, e.g. "gravel"
    display_name: str           # Shown in UI, e.g. "GRAVEL"
    short_description: str      # One-line tooltip
    long_description: str       # Full help text (shown in ? dialog)
    reference: str              # Citation or standard reference
    is_default: bool = False    # Only one plugin per registry should have True
    requires_gpu: bool = False
    requires_calibration: bool = False
    standards_locked_by: List[str] = field(default_factory=list)
                                # If non-empty, only available in listed standards modes

class PluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, tuple[PluginMeta, Any]] = {}

    def register(self, meta: PluginMeta, factory: Callable) -> None:
        self._plugins[meta.key] = (meta, factory)

    def get(self, key: str) -> Any:
        return self._plugins[key][1]()

    def get_default(self) -> Any:
        for meta, factory in self._plugins.values():
            if meta.is_default:
                return factory()
        raise RuntimeError("No default plugin registered")

    def list_available(self, standards_mode: Optional[str] = None) -> List[PluginMeta]:
        """
        If standards_mode is set, returns only plugins not locked out by that standard.
        If standards_locked_by is empty, plugin is always available.
        If standards_locked_by is non-empty, plugin is ONLY available in those standards modes.
        """
        result = []
        for meta, _ in self._plugins.values():
            if standards_mode is None or not meta.standards_locked_by:
                result.append(meta)
            elif standards_mode in meta.standards_locked_by:
                result.append(meta)
        return result
```

**`standards_locked_by` is the mechanism that connects Rule 3 (user choice) to Rule 4
(standards-locked workflows) without any if/else logic scattered through the GUI.** When
Standards Mode is active, the method selector widget calls `list_available(standards_mode=...)`
and the dropdown automatically shows only the permitted methods. No changes needed to the
individual method implementations.

### Six Registry Instances — What Each Covers

```
fitter_registry:    gaussian | gaussian_skew | bayesian | ml_assisted
                    default: gaussian

unfold_registry:    gravel | maxed | rmle | ml_seed
                    default: rmle
                    (rmle recommended for new users; gravel/maxed for reproducibility)

cal_registry:       linear | quadratic | cubic | log_polynomial | cubic_spline
                    + efficiency sub-registry:
                        log_polynomial_eff | power_law | polynomial | exponential | spline
                    (GSA-v4's four efficiency models are all registered here)

id_registry:        manual | library_search | bayesian_library | ml_classifier
                    default: bayesian_library (fast, no model required)

standards_registry: astm_e181 | astm_e261 | astm_e1297 | astm_e1218 |
                    astm_c1232 | astm_c1030
                    (each module is a registered standards plugin)

render_registry:    pyqtgraph | vispy
                    default: pyqtgraph
```

### GUI Integration — Method Selector Widget

Create a reusable `MethodSelectorWidget` that is used by every analytical panel:

```
MethodSelectorWidget behavior:
  - Shows a QComboBox of available methods from the relevant registry
  - The default method has a ★ star prefix in the display name
  - A [?] button next to the combo opens a dialog showing short + long description
    and the reference/citation for each method
  - A [⚙] button opens per-method parameter settings
  - Last selection is remembered in QSettings keyed by workflow name
  - In Standards Mode, methods that are not permitted by the active standard
    are shown in the list but grayed out with a tooltip:
    "Not available in ASTM E181 Standards Mode"
    (shown grayed, not hidden — user can see what options exist and why they
    cannot be selected, maintaining transparency)
```

---

## 5. Architecture Decision Records (ADRs)

### What the uploaded v2 plan adds

The uploaded plan lists ADR creation as the first four issues to open. ADRs are short
structured documents that record *why* an architectural decision was made, what
alternatives were rejected, and what the consequences are. They live in the repository
at `docs/adr/` and are referenced from the relevant code.

### Addition to Phase 1.1 — Project Structure

```
docs/
└── adr/
    ├── README.md                      # How to write an ADR for this project
    ├── ADR-001-gui-stack.md           # PySide6 chosen; Electron/JVM rejected
    ├── ADR-002-renderer-abstraction.md # PyQtGraph first; Vispy optional; Vulkan future
    ├── ADR-003-additive-capability.md  # The additive policy (Rule 2) written here
    ├── ADR-004-plugin-registry.md      # Registry architecture decision
    ├── ADR-005-standards-mode.md       # Three-mode system and locking mechanism
    ├── ADR-006-hal-first.md           # HAL defined Phase 1; drivers Phase 4
    └── ADR-007-offline-first.md       # No internet dependency; bundled data only
```

**ADR template** (stored at `docs/adr/README.md`):

```markdown
# ADR-NNN: [Short Title]

**Status:** Proposed | Accepted | Superseded by ADR-NNN

## Context
[What situation or problem led to this decision? What forces are at play?]

## Decision
[What was decided, stated as a clear positive assertion.]

## Rationale
[Why this option over the alternatives? What evidence or reasoning?]

## Alternatives Rejected
[What other options were considered and why each was rejected.]

## Consequences
[What becomes easier or harder as a result? What must be done to implement this?]

## Compliance with Project Rules
[Which of the four governing rules does this uphold or constrain?]
```

**ADRs that must exist before any Phase 1 code is merged:**
- ADR-001 through ADR-007 listed above must all be in `Accepted` status
- Any future PR that contradicts an accepted ADR must either update the ADR or be rejected
- New ADRs can supersede old ones — they are never deleted

---

## 6. Separating Defaults, Alternatives, and Mandated Settings

### What the uploaded v2 plan adds

The uploaded plan correctly identifies that the prior documents conflated three distinct
categories of settings. The distinction matters for both UI design and for correctness:

| Category | Definition | Who sets it | Can user change? |
|---|---|---|---|
| **Recommended Default** | Best choice for most users in most situations | Project team (documented in ADR) | Yes, always |
| **User-Selectable Alternative** | Valid method with different trade-offs | User selects from registry | Yes, always |
| **Standards-Mandated Fixed** | Required exactly as the standard specifies | Standard author | No, only in Standards Mode |

### How This Changes the UI

**Every parameter and method selector must be annotated with which category it belongs to.**

In the UI this means:

```
─ Peak Fitting Method ─────────────────────────────────────────────
  ★ Gaussian  [default]         ← recommended default, star prefix
    Gaussian + Skew             ← user-selectable, no marker
    Bayesian                    ← user-selectable, no marker
    ML-Assisted                 ← user-selectable, requires ML module
  [?]  [⚙ Parameters]          ← help and per-method settings

─ When Standards Mode (ASTM E181) is active: ─────────────────────
  🔒 Gaussian  [required by E181 §6.3]   ← locked, padlock icon
     Gaussian + Skew  [not permitted]    ← grayed, tooltip explains
     Bayesian          [not permitted]   ← grayed
  [?]  [⚙ Parameters — some locked 🔒]
```

**For parameter inputs in calibration dialogs:**

```
  Polynomial order:  [ 2 ▾ ]   ← free choice in Expert Mode

  In Standards Mode (ASTM E181):
  Polynomial order:  [ 2 🔒 ]  ← locked; hover shows "ASTM E181-23 §5.4.2
                                   requires ≤ 2nd order for energy calibration
                                   in the 60 keV – 2 MeV range"
```

### How This Changes Reports and Session Files

Every `.ffs` session file and every generated report must record, per analytical step:

```json
{
  "step": "energy_calibration",
  "method": "quadratic",
  "method_category": "recommended_default",
  "standards_mode": null,
  "coefficients": [0.045, 0.2981, -0.00000312],
  "polynomial_order": 2,
  "polynomial_order_locked_by": null
}
```

```json
{
  "step": "energy_calibration",
  "method": "quadratic",
  "method_category": "standards_mandated",
  "standards_mode": "ASTM_E181_2023",
  "coefficients": [0.045, 0.2981, -0.00000312],
  "polynomial_order": 2,
  "polynomial_order_locked_by": "ASTM E181-23 §5.4.2"
}
```

This provenance trail is what makes FluxForge results **reproducible and auditable**. A
reviewer looking at a FluxForge report can see exactly which method was used, whether it was
the user's free choice or a standard requirement, and what the parameters were.

---

## 7. fluxforge/unfolding/ as a Top-Level Package

### What the uploaded v2 plan adds

The uploaded plan implicitly separates `fluxforge/unfolding/` from `fluxforge/core/`. This
is architecturally correct: unfolding is a large, distinct analytical domain with its own
registry, algorithm implementations, response matrix I/O, convergence diagnostics, and GPU
backend. Embedding it in `core/` would make `core/` a catch-all and complicate the registry
architecture.

### Change to Phase 1.1 — Project Structure

**Move** unfolding to its own top-level package. Remove references to it from `core/`.

```
fluxforge/
├── gui/                   # PySide6 GUI code only — no core imports except via signals
├── core/                  # Peak fitting, calibration, activity, nuclide ID
│   ├── spectrum.py        # Spectrum data model
│   ├── calibration.py     # Energy + FWHM + efficiency calibration engines
│   ├── peak_fitting.py    # Gaussian, skew, Bayesian fitters
│   ├── nuclide_id.py      # Manual, library, Bayesian, ML identification
│   ├── activity.py        # Activity calculation + uncertainty + source aging
│   └── background.py      # Background subtraction models
├── unfolding/             # NEW top-level package — all unfolding work here
│   ├── __init__.py
│   ├── base.py            # UnfoldingMethod abstract base class
│   ├── gravel.py          # GRAVEL implementation
│   ├── maxed.py           # MAXED implementation
│   ├── rmle.py            # RMLE implementation (new addition)
│   ├── ml_seed.py         # ML U-Net seed/accelerator
│   ├── response_matrix.py # Response matrix I/O and validation
│   ├── convergence.py     # Convergence diagnostics, L-curve, chi-squared tracking
│   └── gpu_backend.py     # Optional CuPy / CUDA acceleration for RMLE
├── io/                    # File format readers and writers
├── ml/                    # ML inference engine (ONNX runtime wrapper)
├── standards/             # ASTM compliance modules
├── hal/                   # Hardware abstraction layer
├── registry/              # Plugin registries (Phase 1 new package)
└── utils/                 # Shared utilities (logging, unit conversion, etc.)
```

**The `UnfoldingMethod` abstract base class:**

```python
# unfolding/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class UnfoldingResult:
    flux: np.ndarray               # Unfolded flux spectrum
    uncertainties: Optional[np.ndarray]  # 1-sigma, None if method doesn't provide
    convergence_history: list      # chi-squared or log-likelihood per iteration
    method_used: str
    parameters_used: dict          # All hyperparameters (lambda, iterations, etc.)
    negative_bin_count: int        # Number of bins with flux < 0 (should be 0 for RMLE)
    method_category: str           # "recommended_default" | "user_selected"
    standards_locked_by: Optional[str]

class UnfoldingMethod(ABC):
    @abstractmethod
    def unfold(
        self,
        measured: np.ndarray,
        response_matrix: np.ndarray,
        **kwargs
    ) -> UnfoldingResult: ...

    @abstractmethod
    def get_meta(self) -> "PluginMeta": ...
```

**GRAVEL, MAXED, RMLE, and ML-seed each implement `UnfoldingMethod` and register themselves
with `unfold_registry` at module import time.** No manual registration step is needed in GUI
code. The registry discovers them when `fluxforge/unfolding/` is imported.

---

## 8. PyQtGraph-First Rendering Strategy (Soften the Vispy Swap)

### What the uploaded v2 plan adds / corrects

The uploaded plan correctly pushes back on the previous additions document's lean toward
replacing PyQtGraph with Vispy as the primary renderer. The better policy is:

- **PyQtGraph first** — it works today, is well-documented, and handles 16K-channel HPGe
  spectra at 60 FPS on hardware that is 8+ years old with `auto_downsample=True`
- **Vispy optional** — higher-performance alternative, registered as a second backend
- **Renderer abstraction mandatory** — the `SpectrumCanvas` class must never call PyQtGraph
  APIs directly; it always goes through the abstraction so backends are swappable
- **Vulkan/C++ future option** — documented in ADR-002 but not on the active roadmap

### Change to Section 1 of the Previous Additions Doc

**Revise the Vispy recommendation** from "replace PyQtGraph" to "add alongside":

```
render_registry.register(
    PluginMeta(
        key="pyqtgraph",
        display_name="PyQtGraph (Default)",
        short_description="Stable OpenGL-backed renderer. Works on all supported hardware.",
        long_description="PyQtGraph uses Qt's OpenGL paint engine. Sufficient for "
                         "all standard HPGe spectroscopy up to 65K channels at 60 FPS "
                         "with hierarchical downsampling. Recommended for most users.",
        reference="pyqtgraph.org",
        is_default=True,
        requires_gpu=False,
    ),
    PyQtGraphCanvas
)

render_registry.register(
    PluginMeta(
        key="vispy",
        display_name="Vispy (High Performance)",
        short_description="OpenGL/Vulkan backend via Vispy. Best for very large spectra "
                          "or GPU-accelerated live rendering.",
        long_description="Vispy uses native OpenGL calls and optionally Vulkan via Qt's "
                         "RHI. Provides substantially higher frame rates for 65K+ channel "
                         "spectra and live MCA feeds. Requires vispy to be installed.",
        reference="vispy.org",
        is_default=False,
        requires_gpu=False,  # CPU OpenGL is sufficient; GPU improves performance
    ),
    VispyCanvas
)
```

**Backend capability flags** — the `SpectrumCanvas` abstraction must expose capability
flags so the GUI can adapt gracefully if a backend does not support a specific feature:

```python
class RendererCapabilities:
    supports_webgl: bool = False
    supports_touch: bool = False
    max_overlay_count: int = 10     # Some backends limit simultaneous overlays
    supports_real_time_update: bool = True
    supports_residual_subplots: bool = True
```

**Switching backends** is available at `View → Renderer → [PyQtGraph ★ | Vispy]`. The
switch triggers a canvas rebuild (< 1 second) and restores all current overlays, ROIs,
and reference lines. No analysis state is lost.

---

## 9. Confirmed Carryovers from the Previous Additions Doc

The following items from the previous additions document are **unchanged and confirmed**.
They are listed here so the complete merged picture is clear, with a brief note on any
interaction with the new material above.

| Section | Item | Interaction with New Material |
|---|---|---|
| §2 (HAL in Phase 1) | `MCADevice` base class defined Phase 1 | Now also issues #6 and #13 in §2.6 |
| §3 (Hierarchical Visual Rep.) | `HierarchicalSpectrumBuffer` with 5 resolution levels | Implemented in PyQtGraph backend first; Vispy backend inherits same buffer |
| §4 (Simple/Expert modes) | Two-mode `ModeManager` | Extended to three modes (Simple/Expert/Standards) by §3 above |
| §5 (CMV / SelectionBus) | `SelectionBus` singleton | Now also issue #7 in §2.6 |
| §6 (N42.42 write compliance) | N42.42 writer with XSD validation | Now also issue #15 in §2.6 |
| §7 (ASTM C1030) | Pu isotopics wizard, `standards/c1030.py` | Now registered in `standards_registry`; locked in Standards Mode |
| §8 (QA Monitoring) | `QAMonitor`, SQLite QA history, drift alerts | Now also issue #22 in §2.6 |
| §9 (RMLE added) | RMLE as fourth unfolding method, GRAVEL/MAXED kept | Now lives in `unfolding/rmle.py`; registered in `unfold_registry` |
| §10 (Source aging) | Bateman equation solver, `decay_chains` SQL table | Core stays in `core/activity.py`; decay chain table in nuclide DB |
| §11 (GPS + Survey Map) | GPS extraction, folium offline map panel | Map panel uses QWebEngineView; offline tile bundle |
| §12 (Residuals-first) | Mini residuals sub-plots on canvas, residuals in reports | `{{ residuals_grid }}` required in base Jinja2 template |
| §13 (Digital Twin) | Dashboard panel reserved in Phase 1 | Now the Dashboard tab in Zone C (§3 of positions doc) |
| §14 (Bayesian ID) | Bayesian library matching after auto-search | Now registered in `id_registry` as `bayesian_library` |
| §15 (GUI positioning) | QA in sidebar; Dashboard tab in Zone C; residuals in reports | All confirmed; Standards Mode banner added to Zone C canvas area |

---

## 10. Full Revised Project Structure

This is the complete directory tree reflecting all additions. Annotated with which Phase
each directory is populated.

```
fluxforge/
│
├── gui/                            # Phase 1 — all Qt/PySide6 UI code
│   ├── main_window.py              # MainWindow — assembles all panels
│   ├── mode_manager.py             # ModeManager (Simple/Expert/Standards)
│   ├── selection_bus.py            # SelectionBus singleton (CMV)
│   ├── spectrum_canvas.py          # SpectrumCanvas abstraction class
│   ├── canvas_backends/
│   │   ├── pyqtgraph_backend.py    # Phase 1 — default production backend
│   │   └── vispy_backend.py        # Phase 1 — optional high-performance backend stub
│   ├── panels/
│   │   ├── left_sidebar.py         # File Browser + Nuclide Browser + Analysis Results + QA & Standards
│   │   ├── right_sidebar.py        # Context-sensitive tool parameters
│   │   ├── bottom_tabs.py          # Peak Table + Calibration + Activity + Batch + Log + Spectrogram
│   │   └── dashboard_tab.py        # Hardware Dashboard (Phase 1: placeholder; Phase 4: Digital Twin)
│   ├── dialogs/
│   │   ├── calibration_dialog.py   # Unified energy + FWHM calibration
│   │   ├── efficiency_dialog.py    # Efficiency calibration
│   │   ├── unfolding_dialog.py     # Unfolding workspace
│   │   ├── batch_dialog.py         # Batch analysis settings
│   │   ├── pu_isotopics_dialog.py  # ASTM C1030 wizard (Phase 3)
│   │   └── settings_dialog.py      # Application settings
│   ├── widgets/
│   │   ├── method_selector.py      # Reusable MethodSelectorWidget
│   │   ├── nuclide_search.py       # Search bar + FTS5 completer
│   │   ├── led_status.py           # Hardware state LED
│   │   ├── astm_status_dots.py     # ASTM pass/fail dot indicators
│   │   ├── hover_tooltip.py        # Customizable peak tooltip
│   │   └── qa_summary.py           # QA drift summary row
│   └── themes/
│       ├── dark.qss                # Dark mode styles
│       ├── light.qss               # Light mode styles
│       └── color_tokens.py         # Shared color constants (both themes)
│
├── core/                           # Phase 2 — no Qt imports
│   ├── spectrum.py                 # Spectrum dataclass + session model
│   ├── calibration.py              # Energy + FWHM calibration engines
│   ├── peak_fitting.py             # Gaussian + skew + Bayesian fitters
│   ├── nuclide_id.py               # Manual + library + Bayesian + ML ID
│   ├── activity.py                 # Activity calculation + source aging + Bateman
│   └── background.py               # Background subtraction models
│
├── unfolding/                      # Phase 3 — separate top-level package
│   ├── base.py                     # UnfoldingMethod ABC + UnfoldingResult dataclass
│   ├── gravel.py                   # GRAVEL (fully supported, original method)
│   ├── maxed.py                    # MAXED (fully supported, maximum entropy)
│   ├── rmle.py                     # RMLE (new recommended default)
│   ├── ml_seed.py                  # ML U-Net accelerator / seed
│   ├── response_matrix.py          # R matrix I/O + validation + built-in HPGe models
│   ├── convergence.py              # Convergence diagnostics
│   └── gpu_backend.py              # CuPy/CUDA optional acceleration
│
├── io/                             # Phase 1
│   ├── reader_factory.py           # Format detection + reader dispatch
│   ├── session.py                  # .ffs session file read/write
│   ├── readers/
│   │   ├── n42.py                  # N42.42 (2006 and 2012)
│   │   ├── chn.py                  # Ortec .CHN
│   │   ├── spc.py                  # Canberra .SPC
│   │   ├── cnf.py                  # Canberra .CNF (pure Python binary parser)
│   │   ├── spe.py                  # IAEA .SPE
│   │   └── csv_reader.py           # CSV (counts-only or energy+counts)
│   └── writers/
│       └── n42_writer.py           # N42.42 (2012) writer + XSD validation
│
├── ml/                             # Phase 3 — no Qt imports
│   ├── inference.py                # ONNX runtime wrapper (CPU + optional CUDA)
│   ├── peak_detector.py            # CNN peak detection
│   ├── nuclide_classifier.py       # Nuclide classification head
│   └── models/                     # Bundled .onnx model files
│
├── standards/                      # Phase 3 / Phase 4
│   ├── base.py                     # StandardsModule ABC
│   ├── e181.py                     # ASTM E181-23
│   ├── e261.py                     # ASTM E261 (if in scope)
│   ├── e1297.py                    # ASTM E1297
│   ├── e1218.py                    # ASTM E1218
│   ├── c1232.py                    # ASTM C1232
│   ├── c1030.py                    # ASTM C1030 (Pu isotopics)
│   └── qa_monitor.py               # QA drift monitoring (FWHM/centroid history)
│
├── hal/                            # Phase 1 (interfaces); Phase 4 (drivers)
│   ├── base.py                     # MCADevice ABC + DeviceStatus
│   ├── device_registry.py          # Runtime device discovery
│   ├── mock_device.py              # Simulator for testing without hardware
│   └── protocols/                  # Phase 4 driver implementations
│       ├── usb_hid.py
│       ├── ethernet.py
│       └── serial_rs232.py
│
├── registry/                       # Phase 1 — plugin architecture
│   ├── base.py                     # PluginRegistry + PluginMeta
│   ├── fitter_registry.py
│   ├── unfold_registry.py
│   ├── cal_registry.py
│   ├── id_registry.py
│   ├── standards_registry.py
│   └── render_registry.py
│
└── utils/                          # Phase 1 — shared utilities
    ├── logging.py
    ├── units.py                    # keV ↔ MeV ↔ channel conversions
    └── uncertainty.py              # Quadrature uncertainty propagation

docs/
├── adr/                            # Architecture Decision Records (Phase 0)
│   ├── README.md
│   ├── ADR-001 through ADR-007
│   └── ...
└── user_guide/

resources/
├── nuclide_db/
│   └── endf_b8_nudat.sqlite        # Bundled offline nuclide database
├── schemas/
│   └── n42_2012.xsd                # N42.42 schema for export validation
├── report_templates/
│   ├── standard_lab_report.html
│   ├── astm_compliance_report.html
│   └── batch_summary_report.html
├── themes/
│   └── icons/                      # SVG icons
└── map_tiles/                      # Offline OSM tile bundle (~50 MB)

tests/
├── spectra/                        # Test spectrum files (N42, CHN, SPC, ...)
├── regression/                     # Regression snapshots
└── ...

.github/
├── ISSUE_TEMPLATE/                 # Issue templates from §2.3
├── workflows/
│   ├── ci.yml                      # Test + lint on every PR
│   └── release.yml                 # Build AppImage + .exe on release tag
└── PROJECT/                        # GitHub Project board config
```

---

## 11. Full Revised Priority Action List

This replaces the 23-item list from the original GUI Implementation Guide. Items are ordered
by dependency and impact. All Stage 0 items must complete before Stage 1 begins.

### Stage 0 — Project Setup (Before Any Code)

| # | Action | Area | Phase |
|---|---|---|---|
| S0.1 | Create milestones, labels, issue templates, project board (§2.1–2.4) | area/docs | M0 |
| S0.2 | Open all 10 epics (§2.5) | area/docs | M0 |
| S0.3 | Open first 25 issues (§2.6) | area/docs | M0 |
| S0.4 | Write and merge ADR-001 through ADR-007 (§5) | area/adr | M0 |
| S0.5 | Scaffold `docs/adr/`, `tests/spectra/`, `.github/ISSUE_TEMPLATE/` | area/docs | M0 |

### Phase 1 — Foundation (Sprints 1–3)

| # | Action | Area | Notes |
|---|---|---|---|
| 1.1 | Create full `fluxforge/` package structure (§10) | area/core | All `__init__.py` files; no logic yet |
| 1.2 | Implement `PluginRegistry` base + all 6 registry instances (§4) | area/registry | Must exist before any plugin is registered |
| 1.3 | Implement `MainWindow` with dockable zones A–F | area/gui | QDockWidget layout; dark theme; layout save/restore |
| 1.4 | Implement `ModeManager` with Simple/Expert/Standards (§3) | area/gui | Signal: `mode_changed(GUIMode, str)` |
| 1.5 | Implement `SelectionBus` singleton (§5 of additions doc) | area/gui | Wire to canvas, peak table, nuclide browser |
| 1.6 | Implement `SpectrumCanvas` abstraction + PyQtGraph backend (§8) | area/gui | All 11 canvas layers; full interaction model |
| 1.7 | Add Vispy backend stub; register both backends in `render_registry` | area/gui | Stub returns RendererCapabilities |
| 1.8 | Implement `HierarchicalSpectrumBuffer` 5-level downsampling (§3 additions) | area/performance | Connected to canvas zoom ratio |
| 1.9 | Implement `MCADevice` base class + mock device + `hal/base.py` (§2 additions) | area/hal | Wire into `Spectrum` source_type field |
| 1.10 | Implement `Spectrum` dataclass + `.ffs` session file read/write | area/io | Includes GPS metadata fields |
| 1.11 | Implement reader factory + N42 / CHN / SPC / CNF / SPE / CSV readers | area/io | Drag-and-drop file opening |
| 1.12 | Implement N42.42 (2012) export writer with XSD schema validation | area/io | `File → Export → ANSI N42.42 (2012)` |
| 1.13 | Build nuclide DB SQLite schema including decay_chains table (§10 additions) | area/nuclide-db | ENDF/B-VIII.0 + NuDat; FTS5 full-text search |
| 1.14 | Implement Nuclide Search panel with instant reference line overlay | area/gui | < 50ms from keystroke to canvas update |
| 1.15 | Implement QA & Standards section in left sidebar (§15.1 additions) | area/gui | ASTM status dots always visible |
| 1.16 | Reserve Dashboard tab in Zone C (§15.2 additions) | area/gui | Placeholder with Phase 4 description |
| 1.17 | Add `● HARDWARE` LED to status bar (§15.2 additions) | area/gui | Shows "NO DEVICE" until Phase 4 |
| 1.18 | Implement dark/light theme + system-theme detection | area/gui | QSS files + color_tokens.py |
| 1.19 | Add test harness with sample spectra, CI baseline, regression snapshots | area/testing | GitHub Actions CI on every PR |

### Phase 2 — Core Analysis (Sprints 4–8)

| # | Action | Area | Notes |
|---|---|---|---|
| 2.1 | Implement unified energy + FWHM calibration dialog (bGamma-style) | area/calibration | Complete in repo: `src/fluxforge/gui/dialogs/calibration_dialog.py`; canvas embedded in dialog; live updates |
| 2.2 | Add quick slider calibration mode (PeakEasy-inspired) | area/calibration | Additive — does not replace full workflow |
| 2.3 | Add non-linear deviation pairs (InterSpec-style) | area/calibration | Fine calibration sub-tab |
| 2.4 | Implement ROI drag + real-time Gaussian peak fitting | area/core | LM algorithm; < 5ms per ROI |
| 2.5 | Implement skew model fitter; register all fitters in `fitter_registry` | area/core | Gaussian + skew both registered |
| 2.6 | Implement `MethodSelectorWidget` and wire to `fitter_registry` | area/gui | Used in calibration, fitting, ID, unfolding |
| 2.7 | Implement undo/redo via `QUndoStack` for all peak operations | area/gui | Ctrl+Z / Ctrl+Shift+Z |
| 2.8 | Implement Peak Table with CMV synchronization via SelectionBus | area/gui | Candidates column; color-coded status dots |
| 2.9 | Implement auto peak search (Ctrl+A) with review dialog | area/core | Second-difference; review dialog |
| 2.10 | Implement Bayesian library matching (§14 additions); register in `id_registry` | area/core | < 200ms CPU for 4K nuclides + 50 peaks |
| 2.11 | Implement efficiency calibration dialog (log-polynomial + other models from cal_registry) | area/calibration | All four GSA-v4 efficiency models registered |
| 2.12 | Implement activity calculation + uncertainty propagation | area/core | MDA via ASTM E1297 Currie method |
| 2.13 | Implement source age correction + Bateman equation solver (§10 additions) | area/core | Decay chain visualization in Activity tab |
| 2.14 | Implement background subtraction (3 modes: simple / scaled / statistical) | area/core | Toggle background overlay on canvas |
| 2.15 | Implement mini residuals sub-plots on canvas (Expert + Standards modes) | area/gui | Per-ROI; amber/red threshold thresholds |
| 2.16 | Implement GPS extraction for N42 / CHN / CNF; Survey Map panel | area/io + area/gui | folium + offline tile bundle (~50 MB) |
| 2.17 | Add multi-spectrum tabs (QTabBar above canvas) | area/gui | Foreground + background + secondary overlay |
| 2.18 | Implement pinned nuclides + nuclide tagging (bGamma-style) | area/gui | Right-click → Pin in nuclide browser |
| 2.19 | Implement cascade sum line display (InterSpec-style) | area/gui | Dotted lines at sum energies |
| 2.20 | Implement Bayesian peak fitting; register in `fitter_registry` | area/core | Prior from FWHM calibration |

### Phase 3 — Advanced Analysis (Sprints 9–14)

| # | Action | Area | Notes |
|---|---|---|---|
| 3.1 | Implement GRAVEL in `unfolding/gravel.py`; register in `unfold_registry` | area/unfolding | Original method; fully supported |
| 3.2 | Implement MAXED in `unfolding/maxed.py`; register in `unfold_registry` | area/unfolding | Maximum entropy; fully supported |
| 3.3 | Implement RMLE in `unfolding/rmle.py`; register as default in `unfold_registry` | area/unfolding | Regularization slider; uncertainty bands |
| 3.4 | Implement ML seed in `unfolding/ml_seed.py`; register in `unfold_registry` | area/unfolding | Reduces RMLE/GRAVEL iterations by ~60% |
| 3.5 | Build unfolding dialog with algorithm comparison mode | area/gui | Side-by-side method overlay |
| 3.6 | Implement ML peak analysis engine (PyTorch → ONNX) | area/ml | GPU optional; ONNX CPU fallback |
| 3.7 | Implement ASTM E181 compliance module; register in `standards_registry` | area/standards | Locks: poly order ≤2; calibration sequence |
| 3.8 | Implement ASTM E1297 MDA module; register in `standards_registry` | area/standards | Currie method; MDA in activity table |
| 3.9 | Implement ASTM E1218 + C1232 modules | area/standards | Calibration source bracketing; lab QA |
| 3.10 | Implement ASTM C1030 Pu isotopics wizard (§7 additions) | area/standards | Expert/Standards mode only |
| 3.11 | Implement QA monitor (§8 additions); `Tools → QA History` view | area/standards | SQLite QA history; drift alerts |
| 3.12 | Implement Standards Mode workflow locking via registry (§3, §4 above) | area/gui | Padlock icons; mode banner on canvas |
| 3.13 | Implement Jinja2 report engine + 3 bundled templates | area/reporting | `{{ residuals_grid }}` required |
| 3.14 | Implement batch analysis queue + ProcessPoolExecutor workers | area/gui | Per-spectrum JSON + aggregate CSV |
| 3.15 | Add GPU-accelerated batch inference via optional CuPy (§7 unfolding/gpu_backend.py) | area/performance | Falls back to CPU if CuPy unavailable |

### Phase 4 — MCA / Live Acquisition (Future)

| # | Action | Area | Notes |
|---|---|---|---|
| 4.1 | Implement USB HID + Ethernet HAL drivers | area/hal | First real hardware drivers |
| 4.2 | Implement device discovery + thumbnails (ProSpect-inspired) | area/gui | Discovered Devices + Detectors panels |
| 4.3 | Implement Digital Twin dashboard (§13 additions; §15.2 positions) | area/gui | Activates the reserved Dashboard tab |
| 4.4 | Implement Spectrogram panel (time–energy color map) | area/gui | Activates the reserved Spectrogram tab |
| 4.5 | Implement live spectrum canvas update path (incremental counts append) | area/gui | Uses HAL callback; SelectionBus aware |

### Packaging (Continuous — Target M6)

| # | Action | Notes |
|---|---|---|
| P.1 | PyInstaller Windows .exe bundle | Bundle: Python, PySide6, PyQtGraph, Vispy stub, NumPy, SciPy, SQLite, ONNX runtime |
| P.2 | AppImage for Linux | Universal; glibc ≥ 2.17; no external dependencies |
| P.3 | GitHub Actions release workflow | Triggers on git tag; builds both artifacts automatically |
| P.4 | pip install fluxforge for developers | uv sync for full dev environment |

---

*Document reflects all three plan iterations — March 2026*
*Branch: chore/folder-audit-native-gui-review-20260317*
