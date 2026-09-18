# Legacy GUI recovery

The maintained desktop implementation is `src/fluxforge/gui/`. The older Tk implementation is retained in `src/fluxforge_gui/` for explicit source-level regression and recovery. It must not be part of the normal installed distribution or a fallback for Qt failures.

## Preserved version and restoration

Source version: `c297542da781291c3d1461af181b79349bda3087`.

Local recovery archive: `C:\Users\Josh\projects\FluxForge-recovery-20260916\legacy\legacy-source-c297542.zip`.

The adjacent `manifest.json` records the archive hash and every member's SHA-256. All 16 members were extracted to a separate `restored` directory and verified byte-for-byte. The archive contains all 11 legacy modules plus original packaging, launcher, README and license. The complete repository history is separately retained in `D:\FluxForge-recovery-20260916\archives\pre-fetch.bundle`. The bundle and full working-content archive were moved to D: with matching before/after hashes when C: filled; `archive-relocation-receipt.json` in the original recovery directory records both locations.

To restore the historical application independently:

1. Clone `pre-fetch.bundle` to a new empty directory and check out the pinned commit on an unprefixed recovery branch.
2. Extract the legacy archive into another empty directory and verify its files against `manifest.json` before applying any locally preserved overlay.
3. Create a separate Python 3.12 environment and install the pinned source with its documented dependencies. Keep this environment separate from the maintained installation.
4. Run the legacy launch entrypoint only from that historical environment. Do not reinstall it into the current production environment.

The local `verify_restore.py` and `restore-launch.log` record an automated native Tk 8.6.15 instantiation and clean close from the extracted source. This test used the maintained backend dependencies and does not claim a clean historical dependency installation or manual workflow qualification. Archive restoration, launch and functional validation are separate checks.

## Source disposition

| Legacy files | Disposition and maintained equivalent |
|---|---|
| `__init__.py`, `app.py`, `ui_builder.py` | Archive-only desktop assembly; current entrypoint and widgets live under `fluxforge.gui`. |
| `commands.py` | Archive-only CLI dispatch; scientific command handlers remain in `fluxforge.cli.app` and shared backend modules. |
| `models.py`, `constants.py`, `presets.py` | Archive-only view models and presets; retain for migration tests. Current state uses the versioned workspace and Qt panels. |
| `mpl_helpers.py` | Archive-only Tk/offscreen drawing support; modern renderers are under `fluxforge.gui.backends`. |
| `parity_registry.py` | Historical registry retained for regression comparison; current production capability exposure is tested separately. |
| `reporting.py` | Historical textual/plot previews retained; reusable report generation remains under `fluxforge.reporting`, CLI and current Qt report surfaces. |
| `spectrum_ops.py` | Historical calibration/counting/plot adapters retained. Shared maintained calibration, peak fit, ROI and activity operations live under `fluxforge.analysis` and `fluxforge.core`. |

The legacy free-form amplitude-constraint multiplet fitter is a distinct implementation. Its source is preserved exactly; this pass does not claim a modern GUI equivalent or numerical qualification for it. The current provisional INL raw-area reduction uses independent single-Gaussian and sideband estimates and does not invoke this function. Any future workflow requiring that constraint fitter needs a scoped backend port and independent tests before retiring its recovery source.

No experimental data, prior result archive or original checkout is removed by this cleanup. Existing older environments remain preserved but must not be used as the current installation; always launch using the verified environment's full executable path.
