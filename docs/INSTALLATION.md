# Install the FluxForge GUI on Windows or Linux

This is the canonical source-install guide for the modern FluxForge Qt GUI.
The supported Python versions are **3.11 and 3.12 (64-bit)**. Python 3.13 is
not currently supported because FluxForge pins NumPy below 2.0.

The commands below install into a repository-local `.venv`; they do not alter
your system Python. Run them from a terminal, not from inside a Python prompt.

## Windows: copy-and-paste setup

### Prerequisites

Install these once:

- [Git for Windows](https://git-scm.com/download/win)
- 64-bit Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/)
  with the Python launcher enabled

Open **PowerShell**, then confirm the tools are visible:

```powershell
git --version
py -3.11 --version
```

Use `py -3.12` in the commands below if that is the version you installed.

### Clone, install, and verify

```powershell
git clone https://github.com/FusionSandwich/FluxForge.git
Set-Location FluxForge
git switch optimization-workflows

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[native-gui,reporting]"

.\.venv\Scripts\fluxforge.exe --help
.\.venv\Scripts\fluxforge.exe commands
.\.venv\Scripts\fluxforge-gui.exe --help
.\.venv\Scripts\python.exe -c "from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR; assert QT_AVAILABLE, QT_IMPORT_ERROR; print('Qt GUI ready')"
```

Using the executables inside `.venv` avoids PowerShell execution-policy and
PATH problems. If you prefer activation, run:

```powershell
.\.venv\Scripts\Activate.ps1
fluxforge gui --project-dir .
```

If activation is blocked, do not change the machine policy just for
FluxForge. Launch directly instead:

```powershell
.\.venv\Scripts\fluxforge.exe gui --project-dir .
```

## Linux: copy-and-paste setup

### Prerequisites

First check the interpreter:

```bash
python3 --version
```

Continue only with Python 3.11 or 3.12. On Debian 12, the default is normally
3.11; on Ubuntu 24.04, it is normally 3.12. If your distribution defaults to
3.13, install or select a 3.11/3.12 interpreter before creating the venv.

On Debian/Ubuntu, install Git, venv support, and the Qt/XCB runtime libraries:

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3-venv \
  libegl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-shape0 libxcb-xkb1
```

These libraries prevent errors such as `libEGL.so.1: cannot open shared object
file` and `Could not load the Qt platform plugin "xcb"` on minimal desktops
and WSL.

### Clone, install, and verify

```bash
git clone https://github.com/FusionSandwich/FluxForge.git
cd FluxForge
git switch optimization-workflows

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[native-gui,reporting]'

fluxforge --help
fluxforge commands
fluxforge-gui --help
python -c "from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR; assert QT_AVAILABLE, QT_IMPORT_ERROR; print('Qt GUI ready')"
```

Launch the GUI:

```bash
fluxforge gui --project-dir .
```

### Windows Subsystem for Linux (WSL)

FluxForge works through WSLg on current Windows 11 installations. From
PowerShell, update WSL once if GUI windows do not appear:

```powershell
wsl --update
wsl --shutdown
```

Then reopen the Linux distro, confirm `echo $DISPLAY` prints a value, and use
the Linux instructions above. Do not set `QT_QPA_PLATFORM=offscreen` for normal
use; that setting is for automated tests and intentionally hides the window.

## First launch and bundled-data check

The recommended launcher is:

```text
fluxforge gui --project-dir .
```

The direct modern-GUI launcher is equivalent:

```text
fluxforge-gui --project-dir .
```

For a first GUI workflow, load the committed files below; no external data
download is required:

- foreground: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background: `examples/RAFM_irradiation/background.ASC`

Then verify that you can:

1. assign foreground and background roles;
2. inspect the calibrated spectrum and find peaks;
3. open the efficiency workspace and edit the HPGe detector ID, C1-C4,
   geometry, window thickness, detector dimensions, dead layer, angle,
   source distance, and relative uncertainty;
4. fit and accept the efficiency curve;
5. run activity, inventory, masking, and optimization views.

The maintained CLI replay checks the same committed RAFM assets:

```bash
fluxforge phase6-ldrd-worked-example \
  --sample-id RAFM4-C_15dEOI \
  --output-root /tmp/phase6_ldrd_worked_example
```

In PowerShell, use a Windows output directory:

```powershell
.\.venv\Scripts\fluxforge.exe phase6-ldrd-worked-example `
  --sample-id RAFM4-C_15dEOI `
  --output-root "$env:TEMP\phase6_ldrd_worked_example"
```

## Install profiles and entry points

| Profile | Command | Use it for |
|---|---|---|
| CLI only | `python -m pip install -e .` | Scripted workflows and bundled examples |
| Full user | `python -m pip install -e '.[native-gui,reporting]'` | Modern Qt GUI, CLI, and reports |
| Developer/QA | `python -m pip install -e '.[dev,native-gui,gui-test,reporting]'` | Tests, formatting, and desktop automation |

| Entrypoint | Purpose |
|---|---|
| `fluxforge` | Main CLI and command discovery |
| `fluxforge gui` | Recommended modern Qt GUI launch path |
| `fluxforge-gui` | Direct modern Qt GUI launcher |
| `fluxforge-gui-legacy` | Legacy Tk GUI for compatibility testing |

The modern Qt GUI is the supported default. The legacy launcher remains
available for regression and compatibility workflows.

## Troubleshooting

### `fluxforge` or `fluxforge-gui` is not found

The venv is not active, or a different Python performed the install. Check:

```bash
python -m pip show fluxforge
python -c "import sys; print(sys.executable)"
```

On Windows, the no-activation launch always works when installation succeeded:

```powershell
.\.venv\Scripts\fluxforge.exe gui --project-dir .
```

### Pip tries to build NumPy or spends a long time resolving it

Check `python --version`. Use Python 3.11 or 3.12 and recreate `.venv`. Do not
force a NumPy 1.26 source build under Python 3.13.

### The modern GUI is reported as unavailable

Install the GUI extra with the same interpreter used to launch FluxForge:

```bash
python -m pip install -e '.[native-gui,reporting]'
python -c "from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR; print(QT_AVAILABLE, QT_IMPORT_ERROR)"
```

The second command prints the exact missing Qt library if loading still fails.
The same steps fix `No module named 'fluxforge.gui'`, which means FluxForge was
not installed into the interpreter that launched the command.

### Linux reports an XCB or EGL plugin error

Install the Debian/Ubuntu prerequisite command from the Linux section, then
start a fresh terminal. To inspect unresolved libraries directly:

```bash
ldd .venv/lib/python*/site-packages/PySide6/Qt/plugins/platforms/libqxcb.so | grep 'not found'
```

### Examples cannot find their input files

Run the documented commands from the repository root. Paths under `examples/`
are intentionally relative to that directory.

## Developer verification

Install the combined QA profile:

```bash
python -m pip install -e '.[dev,native-gui,gui-test,reporting]'
```

Run the focused cross-platform GUI checks:

```bash
python -m pytest -q \
  tests/test_gui_native_app.py \
  tests/test_calibration_workspace_qt.py \
  tests/test_gui_desktop_native.py
```

More examples and command references:

- [User guide](USER_GUIDE.md)
- [Example workflows](EXAMPLE_WORKFLOWS.md)
- [CLI reference](CLI_REFERENCE.md)
- [Example inventory](../examples/README.md)
