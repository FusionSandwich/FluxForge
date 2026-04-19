from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QApplication, Qt  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from tests._phase6_real_data import (  # noqa: E402
    DEFAULT_PHASE6_SAMPLE_ID,
    load_phase6_real_activity_results,
    load_phase6_real_activity_review,
    load_phase6_real_data_paths,
    load_phase6_real_optimization_grids,
    load_phase6_real_spectrum,
    load_phase6_real_target_weights_text,
)


def _write_review_gallery(
    output_dir: Path,
    screenshots: list[Path],
    summary: dict[str, object],
) -> Path:
    review_path = output_dir / "index.html"
    filters = ['<button class="filter active" data-shot="all">All States</button>']
    cards = []
    for image in screenshots:
        filters.append(
            f'<button class="filter" data-shot="{image.stem}">{image.stem}</button>'
        )
        cards.append(
            f"""
            <article class="card" data-shot="{image.stem}">
              <h2>{image.stem}</h2>
              <img src="{image.name}" alt="{image.name}" />
            </article>
            """
        )

    details = "".join(f"<dt>{key}</dt><dd>{value}</dd>" for key, value in summary.items())
    review_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FluxForge Phase 6 Review</title>
    <style>
      :root {{
        --ink: #10253d;
        --muted: #5e7289;
        --bg: #eef4f8;
        --card: rgba(255,255,255,0.96);
        --edge: rgba(16,37,61,0.12);
        --accent: #0f7b6c;
        --warm: #c5751d;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(15,123,108,0.12), transparent 28%),
          linear-gradient(135deg, #fbfdfe, var(--bg));
      }}
      main {{
        max-width: 1320px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }}
      header {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 20px 50px rgba(16,37,61,0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }}
      dl {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px 18px;
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
        font-weight: 700;
      }}
      .filters {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
      }}
      .filter {{
        border: 1px solid rgba(16,37,61,0.14);
        border-radius: 999px;
        background: rgba(255,255,255,0.82);
        padding: 10px 14px;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
      }}
      .filter.active {{
        color: white;
        border-color: transparent;
        background: linear-gradient(90deg, var(--accent), var(--warm));
      }}
      section {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin-top: 24px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 18px 42px rgba(16,37,61,0.08);
      }}
      img {{
        width: 100%;
        display: block;
        border-radius: 16px;
        border: 1px solid rgba(16,37,61,0.1);
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Phase 6 Review</h1>
        <p>Native Qt screenshots captured from the phase-6 masking, optimization, and second-irradiation workflows.</p>
        <dl>{details}</dl>
      </header>
      <div class="filters">{''.join(filters)}</div>
      <section>{''.join(cards)}</section>
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
        print("usage: gui_phase6_optimization_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
      stale.unlink(missing_ok=True)
    for stale_name in ("index.html", "phase6_probe.ffexp"):
      (output_dir / stale_name).unlink(missing_ok=True)
    stale_worked_example_dir = output_dir / "ldrd_worked_example_probe"
    if stale_worked_example_dir.exists():
      shutil.rmtree(stale_worked_example_dir)

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.library_manager.set_gamma_identification_source("nasa_common_lab_sources")
    window.analysis_workspace.replace_spectrum_slot(
        "foreground",
        load_phase6_real_spectrum(DEFAULT_PHASE6_SAMPLE_ID),
        source_label=DEFAULT_PHASE6_SAMPLE_ID,
    )
    window.analysis_workspace.select_spectrum("foreground")
    window.analysis_workspace.set_activity_results(
        load_phase6_real_activity_results(DEFAULT_PHASE6_SAMPLE_ID)
    )
    window.bottom_dock.widget().activity_results_panel._last_activity_review = (  # noqa: SLF001
        load_phase6_real_activity_review(DEFAULT_PHASE6_SAMPLE_ID)
    )
    window.show()
    app.processEvents()

    screenshots: list[Path] = []
    grids = load_phase6_real_optimization_grids(DEFAULT_PHASE6_SAMPLE_ID)
    bottom_tabs = window.bottom_dock.widget()

    def click(widget) -> None:
      QTest.mouseClick(widget, Qt.LeftButton)
      app.processEvents()

    def capture(name: str) -> None:
        path = output_dir / f"{name}.png"
        window.grab().save(str(path))
        screenshots.append(path)

    inventory = bottom_tabs.inventory_timeline_panel
    inventory._refresh_inventory()  # noqa: SLF001
    bottom_tabs.setCurrentWidget(inventory)
    app.processEvents()
    capture("01-inventory-time-evolution")

    masking = bottom_tabs.masking_review_panel
    masking.energy_window_spin.setValue(10.0)
    bottom_tabs.setCurrentWidget(masking)
    click(masking.refresh_button)
    app.processEvents()
    capture("02-line-interference-masking")

    optimization = bottom_tabs.optimization_workspace_panel
    optimization.irradiation_grid_edit.setText(grids["irradiation_grid_s"])
    optimization.cooldown_grid_edit.setText(grids["cooldown_grid_s"])
    optimization.count_grid_edit.setText(grids["count_grid_s"])
    bottom_tabs.setCurrentWidget(optimization)
    click(optimization.run_button)
    app.processEvents()
    capture("03-irradiation-optimizer")

    optimization.tabs.setCurrentWidget(optimization.recommendation_browser)
    app.processEvents()
    capture("04-optimizer-recommendation")

    worked_example_root = output_dir / "ldrd_worked_example_probe"
    optimization.ldrd_sample_id_edit.setText(DEFAULT_PHASE6_SAMPLE_ID)
    optimization.ldrd_output_root_edit.setText(str(worked_example_root))
    click(optimization.ldrd_worked_example_button)
    app.processEvents()
    capture("05-ldrd-worked-example")

    second = bottom_tabs.second_irradiation_panel
    second.flux_scales_edit.setText("1.0")
    second.duration_factors_edit.setText("1.0")
    second.cooling_grid_edit.setText(grids["cooldown_grid_s"])
    second.target_weights_edit.setText(
        load_phase6_real_target_weights_text(DEFAULT_PHASE6_SAMPLE_ID)
    )
    bottom_tabs.setCurrentWidget(second)
    click(second.run_button)
    app.processEvents()
    capture("06-second-irradiation")

    ffexp_path = output_dir / "phase6_probe.ffexp"
    optimization.export_ffexp(ffexp_path)
    worked_example_summary = worked_example_root / "WORKED_EXAMPLE_SUMMARY.md"

    source_paths = load_phase6_real_data_paths(DEFAULT_PHASE6_SAMPLE_ID)
    summary = {
        "sample_id": DEFAULT_PHASE6_SAMPLE_ID,
        "screenshots": len(screenshots),
        "inventory_rows": inventory.table.rowCount(),
        "masking_rows": masking.line_table.rowCount(),
        "optimization_rows": optimization.heatmap_table.rowCount(),
        "second_irradiation_rows": second.table.rowCount(),
        "ffexp_bundle": ffexp_path.name,
        "worked_example_summary": worked_example_summary.name,
        "worked_example_complete": worked_example_summary.exists(),
        "analysis_json": source_paths["analysis_json"].name,
        "schedule_metadata": source_paths["sample_schedules"].name,
        "neutron_unfold": source_paths["unfold_result"].name,
    }
    review_path = _write_review_gallery(output_dir, screenshots, summary)
    print(json.dumps({"review_gallery": str(review_path), **summary}))
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
