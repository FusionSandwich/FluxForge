from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fluxforge.core.analysis_workspace import PeakCandidate  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402


def _write_review_gallery(output_dir: Path, screenshots: list[Path]) -> Path:
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
    review_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FluxForge Predictive Review</title>
    <style>
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: #10253c;
        background: linear-gradient(135deg, #f8fcfd, #eef4f7);
      }}
      main {{ max-width: 1260px; margin: 0 auto; padding: 32px 24px 64px; }}
      header {{ background: rgba(255,255,255,0.96); border: 1px solid rgba(16,37,60,0.12); border-radius: 24px; padding: 24px 28px; }}
      .filters {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }}
      .filter {{ border: 1px solid rgba(16,37,60,0.14); border-radius: 999px; background: rgba(255,255,255,0.82); padding: 10px 14px; font: inherit; font-weight: 700; cursor: pointer; }}
      .filter.active {{ color: white; border-color: transparent; background: linear-gradient(90deg, #0f766e, #c05621); }}
      section {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-top: 24px; }}
      .card {{ background: rgba(255,255,255,0.96); border: 1px solid rgba(16,37,60,0.12); border-radius: 20px; padding: 16px; box-shadow: 0 18px 42px rgba(16,37,60,0.08); }}
      img {{ width: 100%; display: block; border-radius: 16px; border: 1px solid rgba(16,37,60,0.1); }}
      @media (max-width: 960px) {{
        main {{ padding: 24px 16px 44px; }}
        section {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
      }}
      @media (max-width: 640px) {{
        header {{ padding: 18px; border-radius: 18px; }}
        section {{ grid-template-columns: 1fr; gap: 14px; }}
        .card {{ border-radius: 16px; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Predictive Review</h1>
        <p>Native Qt screenshots captured from the offline predictive dashboard and QA sidebar.</p>
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
        print("usage: gui_predictive_dashboard_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    selection_bus = SelectionBus()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=selection_bus)
    window.analysis_workspace.replace_peaks(
        (
            PeakCandidate(
                peak_id="peak-1",
                channel=662.0,
                energy_keV=661.66,
                significance=9.8,
                roi_bounds_keV=(655.0, 668.0),
                net_counts=4200.0,
                fit_quality=0.98,
            ),
        )
    )
    window.analysis_workspace.select_peak("peak-1")
    selection_bus.publish_roi(655.0, 668.0)
    window.show()
    app.processEvents()

    screenshots: list[Path] = []

    def capture(widget, name: str) -> None:
        path = output_dir / f"{name}.png"
        widget.grab().save(str(path))
        screenshots.append(path)

    capture(window.left_dock.widget(), "01-qa-sidebar-predictive")

    tab_bar = window.central_tabs.tabBar()
    QTest.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(1).center())
    app.processEvents()
    window.central_tabs.predictive_dashboard.target_counts_spin.setValue(12000.0)
    app.processEvents()
    capture(window.central_tabs.predictive_dashboard, "02-predictive-dashboard")

    review_path = _write_review_gallery(output_dir, screenshots)
    window.close()
    print(review_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
