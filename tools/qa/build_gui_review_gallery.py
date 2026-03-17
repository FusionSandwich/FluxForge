#!/usr/bin/env python3
"""Build a static review gallery for native FluxForge GUI screenshots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_ROOT = REPO_ROOT / "tests" / "data" / "gui_review_baselines"
DEFAULT_MANIFEST = DEFAULT_BASELINE_ROOT / "manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _diff_images(
    current_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> float | None:
    with Image.open(current_path) as current_image, Image.open(
        baseline_path
    ) as baseline_image:
        if current_image.size != baseline_image.size:
            return None
        diff = ImageChops.difference(
            current_image.convert("RGBA"),
            baseline_image.convert("RGBA"),
        )
        stat = ImageStat.Stat(diff.convert("RGB"))
        diff_ratio = sum(stat.mean[:3]) / (255.0 * 3.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diff.save(output_path)
        return diff_ratio


def build_gallery(
    *,
    current_dir: Path,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    run_payload = load_json(current_dir / "run.json")
    manifest = load_json(manifest_path)
    platform = str(run_payload.get("platform") or "unknown")
    platform_manifest = manifest.get("platforms", {}).get(platform, {})
    checkpoints = platform_manifest.get("checkpoints", [])

    current_output = output_dir / "current"
    baseline_output = output_dir / "baseline"
    diff_output = output_dir / "diff"
    records: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        name = checkpoint["name"]
        current_source = current_dir / name
        baseline_source = baseline_root / platform / name
        current_target = current_output / name
        baseline_target = baseline_output / name
        diff_target = diff_output / name

        current_present = _copy_if_exists(current_source, current_target)
        baseline_present = _copy_if_exists(baseline_source, baseline_target)
        diff_ratio = None
        status = "missing-current"
        if current_present and baseline_present:
            diff_ratio = _diff_images(current_source, baseline_source, diff_target)
            status = "changed" if diff_ratio not in (None, 0.0) else "match"
            if diff_ratio is None:
                status = "size-mismatch"
        elif current_present:
            status = "missing-baseline"

        records.append(
            {
                "name": name,
                "label": checkpoint["label"],
                "scenario": checkpoint["scenario"],
                "review_note": checkpoint["review_note"],
                "status": status,
                "current_path": (
                    current_target.relative_to(output_dir).as_posix()
                    if current_present
                    else None
                ),
                "baseline_path": (
                    baseline_target.relative_to(output_dir).as_posix()
                    if baseline_present
                    else None
                ),
                "diff_path": (
                    diff_target.relative_to(output_dir).as_posix()
                    if diff_target.exists()
                    else None
                ),
                "diff_ratio": diff_ratio,
            }
        )

    summary = {
        "platform": platform,
        "window_geometry": platform_manifest.get("window_geometry"),
        "records": records,
        "run_payload": run_payload,
    }
    save_json(output_dir / "summary.json", summary)
    write_gallery_html(summary, output_dir / "index.html")
    return summary


def write_gallery_html(summary: dict[str, Any], output_path: Path) -> None:
    cards = []
    for record in summary["records"]:
        current_html = (
            f'<img src="{record["current_path"]}" alt="{record["label"]} current">'
            if record["current_path"]
            else "<div class='missing'>Current image missing</div>"
        )
        baseline_html = (
            f'<img src="{record["baseline_path"]}" alt="{record["label"]} baseline">'
            if record["baseline_path"]
            else "<div class='missing'>Baseline image missing</div>"
        )
        diff_html = (
            f'<img src="{record["diff_path"]}" alt="{record["label"]} diff">'
            if record["diff_path"]
            else "<div class='missing'>Diff unavailable</div>"
        )
        diff_ratio = (
            (
                f"{record['diff_ratio']:.4f}"
                if isinstance(record["diff_ratio"], float)
                else "n/a"
            )
        )
        cards.append(
            f"""
            <section class="card status-{record['status']}">
              <h2>{record['label']}</h2>
              <p><strong>Scenario:</strong> {record['scenario']}<br>
                 <strong>Status:</strong> {record['status']}<br>
                 <strong>Diff ratio:</strong> {diff_ratio}</p>
              <p>{record['review_note']}</p>
              <div class="grid">
                <figure><figcaption>Current</figcaption>{current_html}</figure>
                <figure><figcaption>Baseline</figcaption>{baseline_html}</figure>
                <figure><figcaption>Diff</figcaption>{diff_html}</figure>
              </div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>FluxForge GUI Review Gallery</title>
    <style>
      body {{
        font-family: sans-serif;
        margin: 0;
        padding: 24px;
        background: #f3f5f8;
        color: #17212b;
      }}
      h1 {{ margin-top: 0; }}
      .meta {{ margin-bottom: 24px; }}
      .card {{
        background: white;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
      }}
      figure {{ margin: 0; background: #eef2f6; padding: 8px; border-radius: 8px; }}
      figcaption {{ font-weight: 600; margin-bottom: 8px; }}
      img {{ width: 100%; height: auto; border: 1px solid #cad3dd; background: white; }}
      .missing {{
        min-height: 120px;
        display: grid;
        place-items: center;
        color: #7a8693;
        border: 1px dashed #a8b3bf;
        background: #fbfcfd;
      }}
      .status-match {{ border-left: 6px solid #2d8a4f; }}
      .status-changed {{ border-left: 6px solid #d08b22; }}
      .status-missing-baseline,
      .status-size-mismatch {{ border-left: 6px solid #2a6fcf; }}
      .status-missing-current {{ border-left: 6px solid #c23b22; }}
    </style>
  </head>
  <body>
    <h1>FluxForge GUI Review Gallery</h1>
    <div class="meta">
      <strong>Platform:</strong> {summary['platform']}<br>
      <strong>Window geometry:</strong> {summary.get('window_geometry') or 'n/a'}
    </div>
    {''.join(cards)}
  </body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a static review gallery from native FluxForge GUI screenshots."
        )
    )
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root directory containing per-platform baseline screenshots.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest JSON describing expected screenshot checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_gallery(
        current_dir=args.current_dir.resolve(),
        baseline_root=args.baseline_root.resolve(),
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(f"Built GUI review gallery at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
