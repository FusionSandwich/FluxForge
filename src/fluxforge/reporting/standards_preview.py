"""Toolkit-independent ASTM result previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_gui_astm_e261_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI ASTM E261 preview pane."""

    lines = ["FluxForge ASTM E261 Preview", "============================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(f"Standard: {payload.get('standard', 'ASTM E261')}")
    lines.append("")

    summary = payload.get("summary") or {}
    if summary:
        lines.append("Summary")
        lines.append("-------")
        for key in sorted(summary):
            lines.append(f"{key}: {summary[key]}")
        lines.append("")

    measurements = payload.get("measurements") or []
    if measurements:
        lines.append("Measurements")
        lines.append("------------")
        for row in measurements:
            lines.append(
                f"{row.get('reaction_id', 'unknown')}: phi={float(row.get('fluence_cm2', 0.0) or 0.0):.6g} cm^-2, "
                f"phi_dot={float(row.get('fluence_rate_cm2_s', 0.0) or 0.0):.6g} cm^-2 s^-1"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_gui_astm_e262_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI ASTM E262 preview pane."""

    lines = ["FluxForge ASTM E262 Preview", "============================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(f"Standard: {payload.get('standard', 'ASTM E262')}")
    lines.append(f"Convention: {payload.get('convention', 'Stoughton and Halperin')}")
    lines.append("")

    summary = payload.get("summary") or {}
    if summary:
        lines.append("Summary")
        lines.append("-------")
        for key in sorted(summary):
            lines.append(f"{key}: {summary[key]}")
        lines.append("")

    measurements = payload.get("measurements") or []
    if measurements:
        lines.append("Measurements")
        lines.append("------------")
        for row in measurements:
            lines.append(
                f"{row.get('reaction_id', 'unknown')} [{row.get('mode', 'radiometric')}]: "
                f"phi0={float(row.get('equivalent_2200ms_fluence_cm2', 0.0) or 0.0):.6g} cm^-2, "
                f"phi0_dot={float(row.get('equivalent_2200ms_fluence_rate_cm2_s', 0.0) or 0.0):.6g} cm^-2 s^-1"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
