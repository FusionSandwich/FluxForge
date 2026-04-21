from __future__ import annotations

import json
import os
from pathlib import Path
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402


def _plot_click_point(dialog, channel: float):
    counts = dialog._spectrum.counts
    y_value = float(counts[int(round(channel))])
    scene_point = dialog.spectrum_plot.plotItem.vb.mapViewToScene(
        pg.Point(float(channel), y_value)
    )
    return dialog.spectrum_plot.mapFromScene(scene_point)


def _write_review_gallery(
    output_dir: Path,
    screenshots: list[Path],
    summary: dict[str, object],
) -> Path:
    review_path = output_dir / "index.html"
    items = []
    for image in screenshots:
        slug = image.stem
        items.append(
            f"""
            <article class="card" data-shot="{slug}">
              <h2>{image.name}</h2>
              <img src="{image.name}" alt="{image.name}" />
            </article>
            """
        )

    filters = ['<button class="filter active" data-shot="all">All States</button>']
    for image in screenshots:
        filters.append(
            f'<button class="filter" data-shot="{image.stem}">{image.stem}</button>'
        )

    details = []
    for key, value in summary.items():
        details.append(f"<dt>{key}</dt><dd>{value}</dd>")

    review_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FluxForge Calibration Workspace Review</title>
    <style>
      :root {{
        --ink: #102036;
        --subtle: #607086;
        --bg: #edf3f8;
        --card: rgba(255, 255, 255, 0.94);
        --edge: rgba(96, 112, 134, 0.18);
        --accent: #1a6cc0;
        --warm: #b7791f;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(26, 108, 192, 0.1), transparent 28%),
          linear-gradient(135deg, #f7fafc, var(--bg));
        color: var(--ink);
      }}
      main {{
        max-width: 1240px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }}
      header {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 24px 60px rgba(16, 32, 54, 0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      p {{
        margin: 0;
        color: var(--subtle);
        line-height: 1.6;
      }}
      dl {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px 18px;
        margin-top: 20px;
      }}
      dt {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--warm);
      }}
      dd {{
        margin: 4px 0 0;
        font-size: 1rem;
        font-weight: 700;
      }}
      section {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin-top: 24px;
      }}
      .filters {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
      }}
      .filter {{
        border: 1px solid rgba(96, 112, 134, 0.16);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.8);
        color: var(--ink);
        padding: 10px 14px;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
      }}
      .filter.active {{
        background: linear-gradient(90deg, var(--accent), var(--warm));
        color: #fff;
        border-color: transparent;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 22px;
        padding: 16px;
        box-shadow: 0 20px 50px rgba(16, 32, 54, 0.08);
      }}
      .card h2 {{
        margin: 0 0 12px;
        font-size: 1rem;
      }}
      img {{
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(96, 112, 134, 0.12);
        display: block;
      }}
      @media (max-width: 960px) {{
        main {{ padding: 24px 16px 44px; }}
        section {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
      }}
      @media (max-width: 640px) {{
        header {{ padding: 18px; border-radius: 18px; }}
        dl {{ grid-template-columns: 1fr; }}
        section {{ grid-template-columns: 1fr; gap: 14px; }}
        .card {{ border-radius: 16px; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Calibration Workspace Review</h1>
        <p>
          Native Qt screenshots captured from the redesigned GUI path. Use this page for browser-based review and Playwright inspection of the quick-slider, deviation-pair, and ROI fitting tools.
        </p>
        <dl>
          {''.join(details)}
        </dl>
      </header>
      <div class="filters">
        {''.join(filters)}
      </div>
      <section>
        {''.join(items)}
      </section>
    </main>
    <script>
      const filters = Array.from(document.querySelectorAll('.filter'));
      const cards = Array.from(document.querySelectorAll('.card'));
      filters.forEach((button) => {{
        button.addEventListener('click', () => {{
          const target = button.dataset.shot;
          filters.forEach((item) => item.classList.toggle('active', item === button));
          cards.forEach((card) => {{
            const visible = target === 'all' || card.dataset.shot === target;
            card.style.display = visible ? 'block' : 'none';
          }});
        }});
      }});
    </script>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return review_path


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: gui_calibration_workspace_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    manager = ModeManager()
    bus = SelectionBus()
    window = FluxForgeMainWindow(mode_manager=manager, selection_bus=bus)
    window.show()
    app.processEvents()

    screenshots: list[Path] = []

    main_window_shot = output_dir / "01-main-window.png"
    window.grab().save(str(main_window_shot))
    screenshots.append(main_window_shot)

    sidebar = window.left_dock.widget()
    sidebar.gamma_source_combo.setCurrentIndex(
        max(sidebar.gamma_source_combo.findData("nndc_offline_activation"), 0)
    )
    app.processEvents()
    libraries_shot = output_dir / "02-library-selectors.png"
    window.grab().save(str(libraries_shot))
    screenshots.append(libraries_shot)

    bottom_tabs = window.bottom_dock.widget()
    bottom_tabs.setCurrentIndex(1)
    app.processEvents()

    quick_button = bottom_tabs.findChild(
        QPushButton,
        "QuickSliderCalibrationWorkflowButton",
    )
    if quick_button is None:
        raise RuntimeError("Quick slider workflow button was not found.")
    QTest.mouseClick(quick_button, Qt.LeftButton)
    app.processEvents()

    dialog = window._calibration_dialog
    if dialog is None:
      # Fallback for cases where the quick workflow button wiring changes.
      window._open_quick_slider_calibration_mode()
      app.processEvents()
      dialog = window._calibration_dialog
    if dialog is None:
      raise RuntimeError("Calibration workspace dialog was not created.")
    dialog.raise_()
    dialog.activateWindow()
    app.processEvents()

    quick_slider_shot = output_dir / "03-quick-slider.png"
    dialog.grab().save(str(quick_slider_shot))
    screenshots.append(quick_slider_shot)

    manual_button = bottom_tabs.findChild(
        QPushButton,
        "ManualCalibrationWorkflowButton",
    )
    if manual_button is None:
        raise RuntimeError("Manual workflow button was not found.")
    QTest.mouseClick(manual_button, Qt.LeftButton)
    app.processEvents()
    if window._calibration_dialog is None:
      window._open_manual_calibration_workflow()
      app.processEvents()

    expert_shot = output_dir / "04-calibration-manual-workflow.png"
    dialog.grab().save(str(expert_shot))
    screenshots.append(expert_shot)

    dialog.energy_table.selectRow(0)
    QTest.mouseClick(
        dialog.spectrum_plot.viewport(),
        Qt.LeftButton,
        Qt.NoModifier,
        _plot_click_point(dialog, 1173.0),
    )
    app.processEvents()
    mouse_pick_shot = output_dir / "05-mouse-peak-pick.png"
    dialog.grab().save(str(mouse_pick_shot))
    screenshots.append(mouse_pick_shot)

    dialog.library_search.setText("co")
    app.processEvents()
    dialog.library_results.setCurrentRow(0)
    app.processEvents()
    dialog.library_lines.setCurrentRow(0)
    dialog._assign_selected_library_line()
    app.processEvents()
    library_assign_shot = output_dir / "06-library-assignment.png"
    dialog.grab().save(str(library_assign_shot))
    screenshots.append(library_assign_shot)

    dialog.energy_table.item(2, 3).setText("1515.0")
    app.processEvents()
    dialog.advanced_tabs.setCurrentWidget(dialog.deviation_pairs_tab)
    dialog.seed_deviation_pairs_button.click()
    app.processEvents()
    deviation_shot = output_dir / "07-deviation-pairs.png"
    dialog.grab().save(str(deviation_shot))
    screenshots.append(deviation_shot)

    dialog.advanced_tabs.setCurrentWidget(dialog.roi_fit_tab)
    dialog.roi_region.setRegion((1160.0, 1190.0))
    app.processEvents()
    roi_gaussian_shot = output_dir / "08-roi-gaussian-fit.png"
    dialog.grab().save(str(roi_gaussian_shot))
    screenshots.append(roi_gaussian_shot)

    dialog.roi_method_selector.set_current_key("gaussian_skew")
    app.processEvents()
    roi_skew_shot = output_dir / "09-roi-skew-fit.png"
    dialog.grab().save(str(roi_skew_shot))
    screenshots.append(roi_skew_shot)

    standards_button = bottom_tabs.findChild(
        QPushButton,
        "StandardsCalibrationWorkflowButton",
    )
    if standards_button is None:
        raise RuntimeError("Standards workflow button was not found.")
    QTest.mouseClick(standards_button, Qt.LeftButton)
    app.processEvents()
    standards_shot = output_dir / "10-calibration-standards-workflow.png"
    dialog.grab().save(str(standards_shot))
    screenshots.append(standards_shot)

    summary = {
        "mode": manager.state.mode.value,
        "standard": manager.state.standard or "none",
        "gamma_source": window.library_manager.record_for_category("gamma_identification").label,
        "calibration_source": window.library_manager.record_for_category("calibration").label,
        "naa_monitor_source": window.library_manager.record_for_category("naa_monitor").label,
        "dosimetry_source": window.library_manager.record_for_category("dosimetry").label,
        "activation_source": window.library_manager.record_for_category("activation").label,
        "energy_rows": dialog.energy_table.rowCount(),
        "fwhm_rows": dialog.fwhm_table.rowCount(),
        "energy_order_enabled": dialog.energy_order.isEnabled(),
        "library_results": dialog.library_results.count(),
        "library_lines": dialog.library_lines.count(),
        "deviation_pairs": dialog.deviation_table.rowCount(),
        "roi_method": dialog.roi_method_selector.current_key(),
        "selected_peak_energy_keV": bus.state.peak_energy_keV,
        "energy_summary": dialog.energy_summary.text().split("\n")[0],
        "fwhm_summary": dialog.fwhm_summary.text().split("\n")[0],
        "roi_summary": dialog.roi_fit_summary.text().split("\n")[0],
    }
    review_gallery = _write_review_gallery(output_dir, screenshots, summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dialog.close()
    window.close()
    app.processEvents()

    print(
        json.dumps(
            {
                "status": "ok",
                "review_gallery": str(review_gallery),
                "summary": summary,
                "screenshots": [str(path) for path in screenshots],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
