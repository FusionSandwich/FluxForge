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
    <title>FluxForge Phase 2.1 Review</title>
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
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Phase 2.1 Calibration Workspace</h1>
        <p>
          Native Qt screenshots captured from the redesigned GUI path. Use this page for browser-based review and Playwright inspection.
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
        print("usage: gui_phase2_calibration_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    manager = ModeManager()
    bus = SelectionBus()
    window = FluxForgeMainWindow(mode_manager=manager, selection_bus=bus)
    window.show()
    app.processEvents()

    window._open_energy_fwhm_workspace()
    app.processEvents()
    dialog = window._calibration_dialog
    dialog.raise_()
    dialog.activateWindow()
    app.processEvents()

    screenshots: list[Path] = []

    main_window_shot = output_dir / "01-main-window.png"
    window.grab().save(str(main_window_shot))
    screenshots.append(main_window_shot)

    expert_shot = output_dir / "02-calibration-expert.png"
    dialog.grab().save(str(expert_shot))
    screenshots.append(expert_shot)

    manager.set_standard("ASTM E181")
    app.processEvents()
    standards_shot = output_dir / "03-calibration-standards.png"
    dialog.grab().save(str(standards_shot))
    screenshots.append(standards_shot)

    dialog.energy_table.item(2, 3).setText("1515.0")
    app.processEvents()
    residual_shot = output_dir / "04-calibration-outlier.png"
    dialog.grab().save(str(residual_shot))
    screenshots.append(residual_shot)

    summary = {
        "mode": manager.state.mode.value,
        "standard": manager.state.standard or "none",
        "energy_rows": dialog.energy_table.rowCount(),
        "fwhm_rows": dialog.fwhm_table.rowCount(),
        "energy_order_enabled": dialog.energy_order.isEnabled(),
        "energy_summary": dialog.energy_summary.text().split("\n")[0],
        "fwhm_summary": dialog.fwhm_summary.text().split("\n")[0],
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
