from __future__ import annotations

import json
import os
from pathlib import Path
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
from fluxforge.gui.qt_compat import QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402


def _write_review_gallery(
    output_dir: Path,
    screenshots: list[Path],
    summary: dict[str, object],
) -> Path:
    review_path = output_dir / "index.html"
    cards = []
    for image in screenshots:
        cards.append(
            f"""
            <article class=\"card\"> 
              <h2>{image.stem}</h2>
              <img src=\"{image.name}\" alt=\"{image.name}\" />
            </article>
            """
        )

    details = "".join(f"<dt>{key}</dt><dd>{value}</dd>" for key, value in summary.items())
    review_path.write_text(
        f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>FluxForge Phase 5 Parity Review</title>
    <style>
      :root {{
        --ink: #10253d;
        --muted: #5d7188;
        --bg: #eef3f7;
        --card: rgba(255,255,255,0.96);
        --edge: rgba(16,37,61,0.12);
        --accent: #186d8d;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        background: linear-gradient(130deg, #f9fcff, var(--bg));
      }}
      main {{
        max-width: 1280px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }}
      header {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 20px 52px rgba(16,37,61,0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      p {{
        margin: 0;
        color: var(--muted);
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
        color: var(--accent);
      }}
      dd {{
        margin: 4px 0 0;
        font-weight: 700;
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
        <h1>FluxForge Phase 5 Parity Review</h1>
        <p>Native Qt screenshots for crosswalk loading, replay filtering, and parity execution state.</p>
        <dl>{details}</dl>
      </header>
      <section>{''.join(cards)}</section>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return review_path


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: gui_phase5_parity_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink(missing_ok=True)
    (output_dir / "index.html").unlink(missing_ok=True)
    (output_dir / "phase5_probe_report.json").unlink(missing_ok=True)

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    app.processEvents()

    bottom_tabs = window.bottom_dock.widget()
    panel = bottom_tabs.phase5_parity_panel
    bottom_tabs.setCurrentWidget(panel)
    app.processEvents()

    screenshots: list[Path] = []

    def capture(name: str) -> None:
        path = output_dir / f"{name}.png"
        window.grab().save(str(path))
        screenshots.append(path)

    panel.refresh_crosswalk()
    capture("01-crosswalk-loaded")

    replay_index = panel.replay_filter_combo.findText("adapter-required")
    if replay_index >= 0:
        panel.replay_filter_combo.setCurrentIndex(replay_index)
        app.processEvents()
    capture("02-adapter-required-filter")

    parity_scope_index = panel.parity_scope_combo.findText("workflow")
    if parity_scope_index >= 0:
        panel.parity_scope_combo.setCurrentIndex(parity_scope_index)
        app.processEvents()
    panel.fixture_filter_edit.setText("roi_statistics_workflow_case")
    app.processEvents()
    capture("03-workflow-fixture-filter")

    parity_payload = panel.run_parity_suite() or {}
    app.processEvents()
    capture("04-parity-suite-summary")

    summary = {
        "screenshots": len(screenshots),
        "crosswalk_rows": panel.table.rowCount(),
        "parity_total": (parity_payload.get("summary") or {}).get("total", 0),
        "parity_failed": (parity_payload.get("summary") or {}).get("failed", 0),
        "parity_scope": panel.parity_scope_combo.currentText(),
        "fixture_filter": panel.fixture_filter_edit.text(),
    }

    gallery = _write_review_gallery(output_dir, screenshots, summary)
    report = {
        "schema": "fluxforge.gui_phase5_probe.v1",
        "output_dir": str(output_dir),
        "gallery": str(gallery),
        "summary": summary,
    }
    (output_dir / "phase5_probe_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report))
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
