from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fluxforge.analysis.detector_calibration import EfficiencyPoint  # noqa: E402
from fluxforge.core.analysis_workspace import (  # noqa: E402
    detect_peak_candidates,
    fit_efficiency_model,
)
from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from fluxforge.gui.panels.modern_shell import (  # noqa: E402
    build_demo_background_spectrum,
    build_demo_overlay_spectrum,
    build_demo_spectrum,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QDialog, QInputDialog  # noqa: E402


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
    <title>FluxForge Analysis Workspace Review</title>
    <style>
      :root {{
        --ink: #15263d;
        --muted: #63748b;
        --bg: #edf2f7;
        --card: rgba(255,255,255,0.95);
        --edge: rgba(21,38,61,0.1);
        --accent: #1a6cc0;
        --warm: #b7791f;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(26,108,192,0.12), transparent 28%),
          linear-gradient(135deg, #f8fbfd, var(--bg));
      }}
      main {{
        max-width: 1260px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }}
      header {{
        background: var(--card);
        border: 1px solid var(--edge);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 20px 50px rgba(21,38,61,0.08);
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
        border: 1px solid rgba(21,38,61,0.14);
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
        box-shadow: 0 18px 42px rgba(21,38,61,0.08);
      }}
      img {{
        width: 100%;
        display: block;
        border-radius: 16px;
        border: 1px solid rgba(21,38,61,0.1);
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>FluxForge Analysis Workspace Review</h1>
        <p>Native Qt screenshots captured from the redesigned GUI after completing the analysis workspace workflows.</p>
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


def _efficiency_points():
    return (
        EfficiencyPoint(121.78, 82000.0, 100.0, 1e5, 1.0, count_uncertainty=286.0),
        EfficiencyPoint(356.01, 41000.0, 100.0, 1e5, 1.0, count_uncertainty=203.0),
        EfficiencyPoint(661.657, 19000.0, 100.0, 1e5, 1.0, count_uncertainty=138.0),
        EfficiencyPoint(1173.228, 12500.0, 100.0, 1e5, 1.0, count_uncertainty=112.0),
        EfficiencyPoint(1332.492, 10300.0, 100.0, 1e5, 1.0, count_uncertainty=101.0),
    )


def _write_spectrum_csv(path: Path, spectrum) -> None:
    lines = ["channel,counts"]
    for channel, count in zip(np.asarray(spectrum.channels, dtype=float), np.asarray(spectrum.counts, dtype=float)):
        lines.append(f"{int(round(float(channel)))},{float(count):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: gui_analysis_workspace_probe.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    app.processEvents()

    screenshots: list[Path] = []

    def capture(name: str) -> None:
        path = output_dir / f"{name}.png"
        window.grab().save(str(path))
        screenshots.append(path)

    capture("01-main-window")

    peaks = detect_peak_candidates(build_demo_spectrum())
    original_exec = AutoPeakReviewDialog.exec
    original_accepted = AutoPeakReviewDialog.accepted_peaks
    original_get_text = QInputDialog.getText
    try:
        AutoPeakReviewDialog.exec = lambda self: QDialog.Accepted  # type: ignore[method-assign]
        AutoPeakReviewDialog.accepted_peaks = lambda self: peaks  # type: ignore[method-assign]
        peak_panel = window.bottom_dock.widget().peak_table_panel
        QTest.mouseClick(peak_panel.auto_find_button, Qt.LeftButton)
        app.processEvents()
        QTest.mouseClick(peak_panel.match_button, Qt.LeftButton)
        app.processEvents()
        capture("02-peak-review")

        co60_row = 0
        for row in range(peak_panel.table.rowCount()):
            if "co60" in peak_panel.table.item(row, 4).text().lower():
                co60_row = row
                break
        peak_panel.table.selectRow(co60_row)
        QInputDialog.getText = lambda *args, **kwargs: ("demo-tag", True)  # type: ignore[method-assign]
        QTest.mouseClick(peak_panel.pin_button, Qt.LeftButton)
        QTest.mouseClick(peak_panel.tag_button, Qt.LeftButton)
        app.processEvents()
        capture("03-pinned-tagged")

        fit = fit_efficiency_model(_efficiency_points(), model_key="log_poly_2")
        window.analysis_workspace.set_efficiency_fit(fit)
        activity_panel = window.bottom_dock.widget().activity_results_panel
        peak_panel.table.selectRow(co60_row)
        window.analysis_workspace.select_peak(window.analysis_workspace.state.peaks[co60_row].peak_id)
        app.processEvents()
        activity_panel.source_age_hours.setValue(24.0)
        activity_panel.background_mode_combo.setCurrentIndex(
            activity_panel.background_mode_combo.findData("statistical")
        )
        QTest.mouseClick(activity_panel.compute_activity_button, Qt.LeftButton)
        app.processEvents()
        window.bottom_dock.widget().setCurrentWidget(activity_panel)
        capture("04-activity-results")

        survey_panel = window.bottom_dock.widget().survey_map_panel
        window.bottom_dock.widget().setCurrentWidget(survey_panel)
        app.processEvents()
        capture("05-survey-map")

        window.bottom_dock.widget().setCurrentWidget(peak_panel)
        table_index = peak_panel.table.model().index(0, 0)
        table_rect = peak_panel.table.visualRect(table_index)
        QTest.mouseClick(
            peak_panel.table.viewport(),
            Qt.LeftButton,
            Qt.NoModifier,
            table_rect.center(),
        )
        QTest.mouseClick(peak_panel.use_selected_peak_button, Qt.LeftButton)
        peak_panel.peak_id_tolerance.setValue(2.0)
        peak_panel.peak_id_filter.setText("")
        app.processEvents()
        if peak_panel.peak_id_matches.count():
            match_item = peak_panel.peak_id_matches.item(0)
            match_rect = peak_panel.peak_id_matches.visualItemRect(match_item)
            QTest.mouseClick(
                peak_panel.peak_id_matches.viewport(),
                Qt.LeftButton,
                Qt.NoModifier,
                match_rect.center(),
            )
            app.processEvents()
        capture("06-peak-id-browser")

        review_input_dir = output_dir / "probe_inputs"
        review_input_dir.mkdir(exist_ok=True)
        sample_path = review_input_dir / "sample.csv"
        background_path = review_input_dir / "background.csv"
        overlay_path = review_input_dir / "overlay.csv"
        _write_spectrum_csv(sample_path, build_demo_spectrum())
        _write_spectrum_csv(background_path, build_demo_background_spectrum())
        _write_spectrum_csv(overlay_path, build_demo_overlay_spectrum())
        window.open_path(sample_path)
        window.open_path(background_path)
        window.open_path(overlay_path)
        sidebar = window.left_dock.widget()
        sidebar.foreground_spectrum_combo.setCurrentIndex(
            sidebar.foreground_spectrum_combo.findData("sample-csv")
        )
        sidebar.background_spectrum_combo.setCurrentIndex(
            sidebar.background_spectrum_combo.findData("background-csv")
        )
        sidebar.overlay_spectrum_combo.setCurrentIndex(
            sidebar.overlay_spectrum_combo.findData("overlay-csv")
        )
        app.processEvents()
        capture("07-background-selector-overlay")

        window.central_tabs.spectrum_slot_tabs.setCurrentIndex(1)
        app.processEvents()
        capture("08-background-slot")
    finally:
        AutoPeakReviewDialog.exec = original_exec  # type: ignore[method-assign]
        AutoPeakReviewDialog.accepted_peaks = original_accepted  # type: ignore[method-assign]
        QInputDialog.getText = original_get_text  # type: ignore[method-assign]

    summary = {
        "screenshots": len(screenshots),
        "peak_count": len(window.analysis_workspace.state.peaks),
        "pinned": ", ".join(window.analysis_workspace.state.pinned_nuclides) or "none",
        "activity_results": len(window.analysis_workspace.state.activity_results),
        "survey_points": len(window.analysis_workspace.state.survey_points),
        "background_source": (
            window.analysis_workspace.slot("background").source_label
            if window.analysis_workspace.slot("background") is not None
            else "none"
        ),
        "overlay_source": (
            window.analysis_workspace.slot("overlay").source_label
            if window.analysis_workspace.slot("overlay") is not None
            else "none"
        ),
    }
    review_path = _write_review_gallery(output_dir, screenshots, summary)
    print(json.dumps({"review_gallery": str(review_path), **summary}))
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
