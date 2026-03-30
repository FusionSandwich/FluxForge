from __future__ import annotations

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
    <title>FluxForge Module 3 Review</title>
    <style>
      :root {{
        --ink: #10253c;
        --bg: #eef4f7;
        --card: rgba(255,255,255,0.96);
        --edge: rgba(16,37,60,0.12);
        --accent: #0f766e;
        --warm: #c05621;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        background: linear-gradient(135deg, #f8fcfd, var(--bg));
      }}
      main {{ max-width: 1260px; margin: 0 auto; padding: 32px 24px 64px; }}
      header {{ background: var(--card); border: 1px solid var(--edge); border-radius: 24px; padding: 24px 28px; }}
      .filters {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }}
      .filter {{ border: 1px solid rgba(16,37,60,0.14); border-radius: 999px; background: rgba(255,255,255,0.82); padding: 10px 14px; font: inherit; font-weight: 700; cursor: pointer; }}
      .filter.active {{ color: white; border-color: transparent; background: linear-gradient(90deg, var(--accent), var(--warm)); }}
      section {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-top: 24px; }}
      .card {{ background: var(--card); border: 1px solid var(--edge); border-radius: 20px; padding: 16px; box-shadow: 0 18px 42px rgba(16,37,60,0.08); }}
      img {{ width: 100%; display: block; border-radius: 16px; border: 1px solid rgba(16,37,60,0.1); }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Module 3 Review</h1>
        <p>Native Qt screenshots captured from the modern standards, reporting, and batch-analysis workflows.</p>
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
        print("usage: gui_module3_workflows_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    window.show()
    app.processEvents()

    screenshots: list[Path] = []

    def capture(widget, name: str) -> None:
        path = output_dir / f"{name}.png"
        widget.grab().save(str(path))
        screenshots.append(path)

    capture(window, "01-main-shell-qa")

    window._open_qa_history()
    app.processEvents()
    capture(window._qa_history_dialog, "02-qa-history")

    window._open_report_export()
    app.processEvents()
    capture(window._report_dialog, "03-report-export")

    window._open_pu_isotopics_wizard()
    app.processEvents()
    capture(window._pu_isotopics_dialog, "04-pu-isotopics")

    batch_panel = window.bottom_dock.widget().batch_queue_panel
    QTest.mouseClick(batch_panel.queue_button, Qt.LeftButton)
    app.processEvents()
    QTest.mouseClick(batch_panel.run_button, Qt.LeftButton)
    app.processEvents()
    capture(batch_panel, "05-batch-queue")

    review_path = _write_review_gallery(output_dir, screenshots)
    window.close()
    print(review_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
