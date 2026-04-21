# FluxForge Installation and Environment Setup

This guide is the canonical setup reference for FluxForge users. All commands
assume you are starting from the repository root.

## 1. Choose Your Environment Manager

### Option A: `venv` on Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Option B: `venv` on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Option C: Conda

```bash
conda env create -f environment.yml
conda activate fluxforge
python -m pip install --upgrade pip
```

## 2. Pick an Install Profile

FluxForge supports three common install profiles.

### CLI-only

Use this when you want command-line workflows and bundled examples:

```bash
pip install -e .
```

You get:

- `fluxforge`
- all CLI analysis workflows
- bundled examples that do not require the Qt GUI

### Full user install

Use this when you want the CLI plus the modern Qt GUI and reporting support:

```bash
pip install -e '.[native-gui,reporting]'
```

You get:

- `fluxforge`
- `fluxforge gui`
- `fluxforge-gui`
- reporting extras used by richer export workflows

### Developer and test extras

Use this only if you are extending FluxForge or running the heavier GUI/test surface:

```bash
pip install -e '.[dev,gui-test]'
```

You can combine profiles when needed:

```bash
pip install -e '.[dev,native-gui,gui-test,reporting]'
```

## 3. Verify the Installation

After installation, confirm that the entrypoints are available.

### Required CLI checks

```bash
fluxforge --help
fluxforge commands
fluxforge gui --help
```

What to expect:

- `fluxforge --help` shows grouped command families
- `fluxforge commands` shows the full grouped command catalog
- `fluxforge gui --help` confirms the GUI command surface is installed

### Optional direct GUI entrypoint check

Only for installs that included `native-gui`:

```bash
fluxforge-gui --help
```

### First real command

Run one maintained starter workflow:

```bash
fluxforge phase6-ldrd-worked-example \
  --sample-id RAFM4-C_15dEOI \
  --output-root /tmp/phase6_ldrd_worked_example
```

## 4. Understand the Entry Points

| Entrypoint | What it is for | When to use it |
|---|---|---|
| `fluxforge` | Main CLI entrypoint | Always use this first for command discovery and scripted workflows |
| `fluxforge commands` | Grouped command catalog | Use this when you want to see all workflow families quickly |
| `fluxforge <command> --help` | Detailed command flags | Use this when you already know the command name |
| `fluxforge gui` | GUI launch via the main CLI | Recommended GUI launch path in user docs |
| `fluxforge-gui` | Direct modern GUI launcher | Use if you want a GUI-only entrypoint after installing `native-gui` |

## 5. Common Setup Pitfalls

### `fluxforge: command not found`

Usually means one of these:

- the environment is not activated
- `pip install -e .` was not run in the active environment
- your shell session predates the install

Fix:

1. activate the environment again
2. rerun the install command
3. run `which fluxforge` or `where fluxforge` to confirm the entrypoint path

### `fluxforge gui` says the modern GUI is unavailable

Usually means the `native-gui` extra was not installed.

Fix:

```bash
pip install -e '.[native-gui,reporting]'
```

Then re-run:

```bash
fluxforge gui --help
fluxforge-gui --help
```

### `python -m fluxforge.gui.app` fails with `No module named 'fluxforge.gui'`

This usually means one of these:

- FluxForge was not installed into the active environment
- the active interpreter does not satisfy the project requirement of Python 3.11+
- the user is trying to run a source-module path instead of the installed entrypoint

Recommended fix:

```bash
python -m pip install --upgrade pip
pip install -e '.[native-gui,reporting]'
fluxforge gui --help
fluxforge-gui --help
```

For user workflows, prefer:

```bash
fluxforge gui --project-dir .
```

or:

```bash
fluxforge-gui --project-dir .
```

Do not document `python -m fluxforge.gui.app` as the normal user path.

### Examples fail because they cannot find files

Run all documented example commands from the repository root. The example docs
assume paths like `examples/...` are resolved relative to the repo root.

### You are not sure which install profile you need

Start with:

```bash
pip install -e '.[native-gui,reporting]'
```

That profile is the best default for end users because it supports both the CLI
and the GUI.

## 6. Where to Go Next

- To run `pytest`, install the developer/test extras first:

```bash
pip install -e '.[dev,gui-test]'
pytest -q
```

- Command catalog: `fluxforge commands`
- Full CLI reference: [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
- User guide: [docs/USER_GUIDE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/USER_GUIDE.md:1)
- Example workflows: [docs/EXAMPLE_WORKFLOWS.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/EXAMPLE_WORKFLOWS.md:1)
- Full example inventory: [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
