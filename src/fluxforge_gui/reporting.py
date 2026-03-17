"""Report/result summary and plotting helpers for FluxForge GUI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    Figure = None

from fluxforge.cli import app as cli_app


def summarize_gui_unfold_result(payload: dict[str, Any]) -> str:
    """Build a short human-readable summary for an unfold artifact."""

    flux = np.asarray(payload.get("flux", []), dtype=float)
    boundaries = np.asarray(payload.get("boundaries_eV", []), dtype=float)
    method = str(payload.get("method") or "unknown").upper()
    chi2 = payload.get("chi2")
    diagnostics = payload.get("diagnostics") or {}
    parts = [f"Method: {method}"]
    if chi2 is not None:
        parts.append(f"χ²: {float(chi2):.4g}")
    if flux.size:
        peak_index = int(np.argmax(flux))
        if boundaries.size == flux.size + 1:
            left = float(boundaries[peak_index])
            right = float(boundaries[peak_index + 1])
            parts.append(f"Peak group: {left:.3g}–{right:.3g} eV")
        parts.append(f"Integral flux: {float(np.sum(flux)):.4g}")
    flux_uncertainty = np.asarray(diagnostics.get("flux_uncertainty", []), dtype=float)
    if flux_uncertainty.size == flux.size and flux.size:
        valid = flux > 0.0
        if np.any(valid):
            rel_unc = np.median(
                flux_uncertainty[valid] / np.maximum(flux[valid], 1e-30)
            )
            parts.append(f"Median flux σ/φ: {float(rel_unc):.3g}")
    if "iterations" in diagnostics:
        parts.append(f"Iterations: {int(diagnostics['iterations'])}")
    if "converged" in diagnostics:
        parts.append("Converged" if diagnostics["converged"] else "Not converged")
    if diagnostics.get("convergence_mode"):
        parts.append(f"Mode: {diagnostics['convergence_mode']}")
    return " | ".join(parts)


def summarize_gui_activity_result(payload: dict[str, Any]) -> str:
    """Build a concise activity-artifact summary with uncertainties."""

    lines = payload.get("lines", []) or []
    if not lines:
        return "No line activities loaded."
    activities = np.asarray(
        [float(item.get("activity_Bq", 0.0) or 0.0) for item in lines], dtype=float
    )
    uncertainties = np.asarray(
        [float(item.get("activity_unc_Bq", 0.0) or 0.0) for item in lines], dtype=float
    )
    rel_unc = np.divide(
        uncertainties,
        np.maximum(np.abs(activities), 1e-30),
        out=np.zeros_like(uncertainties),
        where=np.abs(activities) > 0.0,
    )
    radioactive_mass = np.asarray(
        [float(item.get("radioactive_mass_g", 0.0) or 0.0) for item in lines],
        dtype=float,
    )
    specific_activity = np.asarray(
        [float(item.get("specific_activity_Bq_g", 0.0) or 0.0) for item in lines],
        dtype=float,
    )
    strongest_index = int(np.argmax(np.abs(activities)))
    strongest = lines[strongest_index]
    parts = [
        f"Lines: {len(lines)} | Total activity: {float(np.sum(activities)):.4g} Bq | "
        f"Median relative σ: {float(np.median(rel_unc)):.3g} | "
        f"Strongest: {strongest.get('isotope', 'unknown')} {float(strongest.get('energy_keV', 0.0) or 0.0):.1f} keV"
    ]
    if np.any(radioactive_mass > 0.0):
        parts.append(f"Total radioactive mass: {float(np.sum(radioactive_mass)):.4g} g")
    if np.any(specific_activity > 0.0):
        parts.append(
            f"Max specific activity: {float(np.max(specific_activity)):.4g} Bq/g"
        )
    return " | ".join(parts)


def summarize_gui_rate_result(payload: dict[str, Any]) -> str:
    """Build a concise rate-artifact summary with uncertainties."""

    rates = payload.get("rates", []) or []
    if not rates:
        return "No reaction rates loaded."
    rate_values = np.asarray(
        [float(item.get("rate", 0.0) or 0.0) for item in rates], dtype=float
    )
    uncertainties = np.asarray(
        [float(item.get("uncertainty", 0.0) or 0.0) for item in rates], dtype=float
    )
    rel_unc = np.divide(
        uncertainties,
        np.maximum(np.abs(rate_values), 1e-30),
        out=np.zeros_like(uncertainties),
        where=np.abs(rate_values) > 0.0,
    )
    strongest_index = int(np.argmax(np.abs(rate_values)))
    strongest = rates[strongest_index]
    return (
        f"Reactions: {len(rates)} | Sum rate: {float(np.sum(rate_values)):.4g} reactions/s | "
        f"Median relative σ: {float(np.median(rel_unc)):.3g} | "
        f"Largest: {strongest.get('reaction_id', 'reaction')}"
    )


def summarize_gui_validation_result(payload: dict[str, Any]) -> str:
    """Build a concise validation summary for the final compare stage."""

    metrics = payload.get("metrics", {}) or {}
    truth_flux = np.asarray(payload.get("truth_flux", []), dtype=float)
    predicted_flux = np.asarray(payload.get("predicted_flux", []), dtype=float)
    residuals = np.asarray(payload.get("residuals", []), dtype=float)
    parts = []
    if metrics:
        for key in ("mae", "rmse", "mape", "chi2"):
            value = metrics.get(key)
            if value is not None:
                parts.append(f"{key.upper()}: {float(value):.4g}")
    if truth_flux.size and predicted_flux.size:
        truth_norm = float(np.sum(np.abs(truth_flux)))
        parts.append(
            f"Flux norm ratio: {float(np.sum(np.abs(predicted_flux))) / max(truth_norm, 1e-30):.4g}"
        )
    if residuals.size:
        parts.append(f"Max |residual|: {float(np.max(np.abs(residuals))):.4g}")
    return " | ".join(parts) if parts else "No validation metrics loaded."


def build_gui_report_preview(payload: dict[str, Any], report_path: str | Path) -> str:
    """Return the text shown in the GUI report preview pane."""

    text_report = payload.get("text_report") or {}
    text_path = text_report.get("path")
    if text_path:
        resolved = (Path(report_path).resolve().parent / str(text_path)).resolve()
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")

    lines = ["FluxForge Report Preview", "========================", ""]
    summary = payload.get("summary") or {}
    if summary:
        lines.append("Summary")
        lines.append("-------")
        for key in sorted(summary):
            lines.append(f"{key}: {summary[key]}")
    else:
        lines.append("No report summary available.")
    return "\n".join(lines) + "\n"


def build_gui_k0_preview(payload: dict[str, Any], bundle_path: str | Path) -> str:
    """Return the text shown in the GUI k0 workflow preview pane."""

    if str(payload.get("schema") or "") == "fluxforge.report_bundle.v1":
        return build_gui_report_preview(payload, bundle_path)

    lines = ["FluxForge k0-NAA Preview", "========================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append("")

    summary = payload.get("summary") or {}
    if summary:
        lines.append("Summary")
        lines.append("-------")
        for key in sorted(summary):
            lines.append(f"{key}: {summary[key]}")
        lines.append("")

    element_results = payload.get("element_results") or []
    if element_results:
        lines.append("Element Results")
        lines.append("---------------")
        for row in element_results:
            lines.append(
                f"{row.get('element', 'unknown')}: {float(row.get('concentration_ug_g', 0.0) or 0.0):.6g} ± "
                f"{float(row.get('concentration_unc_ug_g', 0.0) or 0.0):.3g} ug/g"
            )
        lines.append("")

    recognized = payload.get("recognized_but_not_applied") or []
    if recognized:
        lines.append("Recognized but not applied")
        lines.append("--------------------------")
        for item in recognized:
            lines.append(f"- {item}")
        lines.append("")

    capability_flags = payload.get("capability_flags") or {}
    if capability_flags:
        lines.append("Capability flags")
        lines.append("----------------")
        for key in sorted(capability_flags):
            lines.append(f"{key}: {capability_flags[key]}")
        lines.append("")

    libraries = payload.get("libraries") or {}
    standard = libraries.get("standard_k0_library") or {}
    if standard:
        lines.append("Library")
        lines.append("-------")
        lines.append(
            f"standard_k0_library: {standard.get('library_id', 'unknown')} @ {standard.get('version', 'unknown')}"
        )
    return "\n".join(lines) + "\n"


def build_gui_astm_e2005_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI ASTM E2005 preview pane."""

    lines = ["FluxForge ASTM E2005 Preview", "============================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append("Standard: ASTM E2005")
    lines.append("")
    for ft in payload.get("fluence_transfers", []):
        lines.append(f"Transfer: {ft.get('transfer_id')}")
        lines.append(
            f"  Fluence: {ft.get('fluence_rate_cm2_s')} +- {ft.get('fluence_rate_unc_cm2_s')}"
        )
    for si in payload.get("spectral_indices", []):
        lines.append(f"Index: {si.get('index_id')}")
        lines.append(
            f"  Measured: {si.get('measured_index')} +- {si.get('measured_index_unc')}"
        )
        lines.append(
            f"  Calculated: {si.get('calculated_index')} +- {si.get('calculated_index_unc')}"
        )
        lines.append(f"  C/E Ratio: {si.get('c_e_ratio')} +- {si.get('c_e_ratio_unc')}")
    return "\n".join(lines) + "\n"


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


def build_gui_astm_e3376_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI ASTM E3376 preview pane."""

    lines = ["FluxForge ASTM E3376 Preview", "=============================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(f"Standard: {payload.get('standard', 'ASTM E3376')}")
    lines.append(f"Mode: {payload.get('mode', 'measurement')}")
    lines.append("")

    summary = payload.get("summary") or {}
    if summary:
        lines.append("Summary")
        lines.append("-------")
        for key in sorted(summary):
            lines.append(f"{key}: {summary[key]}")
        lines.append("")

    outputs = payload.get("outputs") or []
    if outputs:
        lines.append("Outputs")
        lines.append("-------")
        for row in outputs:
            label = row.get("output_id") or row.get("kind") or "output"
            value = row.get("value")
            uncertainty = row.get("uncertainty")
            units = row.get("units") or ""
            if uncertainty is not None:
                lines.append(f"{label}: {value} ± {uncertainty} {units}".rstrip())
            else:
                lines.append(f"{label}: {value} {units}".rstrip())
        lines.append("")

    return "\n".join(lines) + "\n"


def discover_gui_validation_report_inputs(results_root: str | Path) -> dict[str, str]:
    """Discover report defaults from an RAFM validation results directory."""

    discovered = cli_app.discover_validation_report_inputs(Path(results_root))
    root = Path(results_root)
    output_path = root / "validation_report.json"
    figure_dir = root / "validation_report_figures"
    return {
        "validation_results_root": str(root),
        "unfold_file": (
            ""
            if discovered.get("preferred_unfold_file") is None
            else str(discovered["preferred_unfold_file"])
        ),
        "validation_file": "",
        "output": str(output_path),
        "figure_dir": str(figure_dir),
    }


def render_gui_activity_result(
    payload: dict[str, Any],
    *,
    figure: Figure | None = None,
) -> Figure | None:
    """Render line-activity uncertainties for report exports."""

    if Figure is None:
        return None
    fig = figure or Figure(figsize=(8.2, 4.4), dpi=100)
    fig.clf()
    ax = fig.add_subplot(111)
    lines = payload.get("lines", []) or []
    if not lines:
        ax.set_title("Line activities")
        ax.text(
            0.5,
            0.5,
            "No line activities loaded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    activities = np.asarray(
        [float(item.get("activity_Bq", 0.0) or 0.0) for item in lines], dtype=float
    )
    uncertainties = np.asarray(
        [float(item.get("activity_unc_Bq", 0.0) or 0.0) for item in lines], dtype=float
    )
    radioactive_mass = np.asarray(
        [float(item.get("radioactive_mass_g", 0.0) or 0.0) for item in lines],
        dtype=float,
    )
    labels = [
        f"{item.get('isotope', 'line')}\n{float(item.get('energy_keV', 0.0) or 0.0):.1f} keV"
        for item in lines
    ]
    x = np.arange(len(lines), dtype=float)
    ax.errorbar(
        x,
        activities,
        yerr=uncertainties if uncertainties.size == activities.size else None,
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        elinewidth=1.1,
        capsize=3,
        label="Activity",
    )
    ax.set_title("Line activity summary")
    ax.set_xlabel("Gamma line")
    ax.set_ylabel("Activity (Bq)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    if np.all(activities > 0.0):
        ax.set_yscale("log")
    if np.any(radioactive_mass > 0.0):
        mass_ax = ax.twinx()
        mass_ax.plot(
            x,
            radioactive_mass,
            color="#d62728",
            marker="s",
            linewidth=1.0,
            label="Radioactive mass",
        )
        mass_ax.set_ylabel("Radioactive mass (g)")
        if np.all(radioactive_mass > 0.0):
            mass_ax.set_yscale("log")
    fig.tight_layout()
    return fig


def render_gui_rate_result(
    payload: dict[str, Any],
    *,
    figure: Figure | None = None,
) -> Figure | None:
    """Render reaction-rate uncertainties for report exports."""

    if Figure is None:
        return None
    fig = figure or Figure(figsize=(8.4, 4.4), dpi=100)
    fig.clf()
    ax = fig.add_subplot(111)
    rates = payload.get("rates", []) or []
    if not rates:
        ax.set_title("Reaction rates")
        ax.text(
            0.5,
            0.5,
            "No reaction rates loaded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    values = np.asarray(
        [float(item.get("rate", 0.0) or 0.0) for item in rates], dtype=float
    )
    uncertainties = np.asarray(
        [float(item.get("uncertainty", 0.0) or 0.0) for item in rates], dtype=float
    )
    labels = [str(item.get("reaction_id", f"R{i + 1}")) for i, item in enumerate(rates)]
    x = np.arange(len(rates), dtype=float)
    ax.errorbar(
        x,
        values,
        yerr=uncertainties if uncertainties.size == values.size else None,
        fmt="o",
        color="#2ca02c",
        ecolor="#98df8a",
        elinewidth=1.1,
        capsize=3,
    )
    ax.set_title("Reaction rate summary")
    ax.set_xlabel("Reaction")
    ax.set_ylabel("Rate (reactions/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    if np.all(values > 0.0):
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def render_gui_validation_result(
    payload: dict[str, Any],
    *,
    figure: Figure | None = None,
) -> Figure | None:
    """Render end-to-end compare diagnostics for report exports."""

    if Figure is None:
        return None
    fig = figure or Figure(figsize=(8.4, 4.8), dpi=100)
    fig.clf()
    truth_flux = np.asarray(payload.get("truth_flux", []), dtype=float)
    predicted_flux = np.asarray(payload.get("predicted_flux", []), dtype=float)
    residuals = np.asarray(payload.get("residuals", []), dtype=float)

    top_ax = fig.add_subplot(211)
    bottom_ax = fig.add_subplot(212)
    if truth_flux.size == 0 or predicted_flux.size == 0:
        top_ax.set_title("Validation comparison")
        top_ax.text(
            0.5,
            0.5,
            "No validation spectra loaded.",
            ha="center",
            va="center",
            transform=top_ax.transAxes,
        )
        bottom_ax.axis("off")
        fig.tight_layout()
        return fig

    x = np.arange(1, max(truth_flux.size, predicted_flux.size) + 1, dtype=float)
    top_ax.step(
        x[: truth_flux.size],
        truth_flux,
        where="mid",
        color="#1f77b4",
        linewidth=1.6,
        label="Truth",
    )
    top_ax.step(
        x[: predicted_flux.size],
        predicted_flux,
        where="mid",
        color="#d62728",
        linewidth=1.4,
        label="Predicted",
    )
    top_ax.set_title("Validation flux comparison")
    top_ax.set_xlabel("Group")
    top_ax.set_ylabel("Flux (a.u.)")
    top_ax.grid(True, alpha=0.25)
    if np.all(truth_flux > 0.0) and np.all(predicted_flux > 0.0):
        top_ax.set_yscale("log")
    top_ax.legend(loc="best", fontsize=8)

    if residuals.size:
        bottom_ax.bar(
            x[: residuals.size],
            residuals,
            color="#ff7f0e",
            alpha=0.8,
            edgecolor="black",
        )
        bottom_ax.axhline(0.0, color="black", linewidth=1.0)
        bottom_ax.set_ylabel("Residual")
        bottom_ax.set_xlabel("Group")
        bottom_ax.set_title("Validation residuals")
        bottom_ax.grid(True, axis="y", alpha=0.25)
    else:
        bottom_ax.axis("off")
    fig.tight_layout()
    return fig


def render_gui_unfold_result(
    payload: dict[str, Any],
    *,
    figure: Figure | None = None,
) -> Figure | None:
    """Render an unfolded spectrum preview into a Matplotlib figure."""

    if Figure is None:
        return None
    fig = figure or Figure(figsize=(8.4, 4.6), dpi=100)
    fig.clf()

    boundaries = np.asarray(payload.get("boundaries_eV", []), dtype=float)
    flux = np.asarray(payload.get("flux", []), dtype=float)
    covariance = np.asarray(payload.get("covariance", []), dtype=float)
    diagnostics = payload.get("diagnostics") or {}
    flux_uncertainty = np.asarray(diagnostics.get("flux_uncertainty", []), dtype=float)
    chi2_history = np.asarray(diagnostics.get("chi2_history", []), dtype=float)
    measured_rates = np.asarray(diagnostics.get("measured_rates", []), dtype=float)
    predicted_rates = np.asarray(diagnostics.get("predicted_rates", []), dtype=float)
    measured_rate_unc = np.asarray(
        diagnostics.get("measured_rate_uncertainties", []), dtype=float
    )
    predicted_rate_unc = np.asarray(
        diagnostics.get("predicted_rate_uncertainties", []), dtype=float
    )
    residuals = np.asarray(diagnostics.get("rate_residuals", []), dtype=float)
    pulls = np.asarray(diagnostics.get("rate_pulls", []), dtype=float)
    reaction_labels = list(
        diagnostics.get("reactions", []) or payload.get("reactions", []) or []
    )

    has_rate_panel = (
        measured_rates.size
        and predicted_rates.size
        and measured_rates.shape == predicted_rates.shape
    )
    has_covariance = (
        covariance.ndim == 2
        and covariance.shape[0] == flux.size
        and covariance.shape[1] == flux.size
        and np.any(np.abs(covariance) > 0.0)
    )
    has_residual_panel = (
        residuals.size and pulls.size and residuals.shape == pulls.shape
    )
    needs_grid = (
        has_rate_panel or has_covariance or chi2_history.size or has_residual_panel
    )

    if needs_grid:
        flux_ax = fig.add_subplot(221)
        parity_ax = fig.add_subplot(222)
        residual_ax = fig.add_subplot(223)
        diagnostics_ax = fig.add_subplot(224)
    else:
        flux_ax = fig.add_subplot(111)
        parity_ax = None
        residual_ax = None
        diagnostics_ax = None

    if flux.size == 0:
        flux_ax.set_title("Unfolded spectrum")
        flux_ax.set_xlabel("Energy (eV)")
        flux_ax.set_ylabel("Flux (a.u.)")
        flux_ax.text(
            0.5,
            0.5,
            "No unfolded flux loaded.",
            ha="center",
            va="center",
            transform=flux_ax.transAxes,
        )
        fig.tight_layout()
        return fig

    if boundaries.size == flux.size + 1:
        x_values = boundaries[:-1]
        flux_ax.step(
            x_values, flux, where="post", color="#1f77b4", linewidth=1.8, label="Flux"
        )
        if np.all(boundaries > 0):
            flux_ax.set_xscale("log")
            flux_ax.set_xlim(float(boundaries[0]), float(boundaries[-1]))
        centers = (
            np.sqrt(boundaries[:-1] * boundaries[1:])
            if np.all(boundaries[:-1] > 0) and np.all(boundaries[1:] > 0)
            else 0.5 * (boundaries[:-1] + boundaries[1:])
        )
    else:
        centers = np.arange(1, flux.size + 1, dtype=float)
        flux_ax.plot(centers, flux, color="#1f77b4", linewidth=1.8, label="Flux")

    if (
        flux_uncertainty.size != flux.size
        and covariance.ndim == 2
        and covariance.shape[0] == flux.size
        and covariance.shape[1] == flux.size
    ):
        flux_uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if flux_uncertainty.size == flux.size and np.any(flux_uncertainty > 0.0):
        flux_ax.errorbar(
            centers,
            flux,
            yerr=flux_uncertainty,
            fmt="none",
            ecolor="#6baed6",
            elinewidth=1.0,
            alpha=0.9,
            label="1σ",
        )

    if np.all(flux > 0):
        flux_ax.set_yscale("log")
    flux_ax.set_title(
        f"Unfolded spectrum ({str(payload.get('method') or 'unknown').upper()})"
    )
    flux_ax.set_xlabel("Energy (eV)")
    flux_ax.set_ylabel("Flux (a.u.)")
    flux_ax.grid(True, which="both", alpha=0.25)
    flux_ax.legend(loc="best", fontsize=8)

    if parity_ax is not None:
        if has_rate_panel:
            min_val = float(
                max(min(np.min(measured_rates), np.min(predicted_rates)), 1e-30)
            )
            max_val = float(
                max(np.max(measured_rates), np.max(predicted_rates), min_val * 10.0)
            )
            parity_ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=1.0,
                label="1:1",
            )
            parity_ax.errorbar(
                measured_rates,
                predicted_rates,
                xerr=(
                    measured_rate_unc
                    if measured_rate_unc.size == measured_rates.size
                    else None
                ),
                yerr=(
                    predicted_rate_unc
                    if predicted_rate_unc.size == predicted_rates.size
                    else None
                ),
                fmt="o",
                color="#2ca02c",
                ecolor="#98df8a",
                elinewidth=1.0,
                capsize=2,
                alpha=0.85,
            )
            for index, label in enumerate(reaction_labels[: measured_rates.size]):
                parity_ax.annotate(
                    label,
                    (measured_rates[index], predicted_rates[index]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
            parity_ax.set_xscale("log")
            parity_ax.set_yscale("log")
            parity_ax.set_xlabel("Measured rate")
            parity_ax.set_ylabel("Predicted rate")
            parity_ax.set_title("Measured vs predicted")
            parity_ax.grid(True, which="both", alpha=0.25)
        elif chi2_history.size:
            parity_ax.plot(
                np.arange(1, chi2_history.size + 1, dtype=float),
                chi2_history,
                color="#d62728",
                linewidth=1.6,
            )
            parity_ax.set_title("Convergence history")
            parity_ax.set_xlabel("Iteration")
            parity_ax.set_ylabel("χ² / dof")
            parity_ax.grid(True, alpha=0.25)
        else:
            parity_ax.axis("off")

    if residual_ax is not None:
        if has_residual_panel:
            x = np.arange(residuals.size, dtype=float)
            labels = (
                reaction_labels[: residuals.size]
                if reaction_labels
                else [f"R{i+1}" for i in range(residuals.size)]
            )
            residual_ax.bar(x, residuals, color="#ff7f0e", alpha=0.8, edgecolor="black")
            residual_ax.axhline(0.0, color="black", linewidth=1.0)
            residual_ax.set_xticks(x)
            residual_ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            residual_ax.set_ylabel("Predicted - Measured")
            residual_ax.set_title("Residuals / pulls")
            residual_ax.grid(True, axis="y", alpha=0.2)
            pull_ax = residual_ax.twinx()
            pull_ax.plot(x, pulls, color="#9467bd", marker="o", linewidth=1.2)
            pull_ax.axhline(
                2.0, color="#9467bd", linestyle="--", linewidth=0.9, alpha=0.6
            )
            pull_ax.axhline(
                -2.0, color="#9467bd", linestyle="--", linewidth=0.9, alpha=0.6
            )
            pull_ax.set_ylabel("Pull")
        else:
            residual_ax.axis("off")

    if diagnostics_ax is not None:
        if has_covariance:
            sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            denom = np.outer(sigma, sigma)
            corr = np.divide(
                covariance, denom, out=np.zeros_like(covariance), where=denom > 0.0
            )
            np.fill_diagonal(corr, 1.0)
            image = diagnostics_ax.imshow(
                corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
            )
            diagnostics_ax.set_title("Flux correlation")
            diagnostics_ax.set_xlabel("Group")
            diagnostics_ax.set_ylabel("Group")
            fig.colorbar(image, ax=diagnostics_ax, fraction=0.046, pad=0.04)
        elif chi2_history.size:
            diagnostics_ax.plot(
                np.arange(1, chi2_history.size + 1, dtype=float),
                chi2_history,
                color="#d62728",
                linewidth=1.6,
            )
            diagnostics_ax.set_title("Convergence history")
            diagnostics_ax.set_xlabel("Iteration")
            diagnostics_ax.set_ylabel("χ² / dof")
            diagnostics_ax.grid(True, alpha=0.25)
        else:
            diagnostics_ax.axis("off")

    fig.tight_layout()
    return fig


def build_gui_rafm_validation_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI RAFM validation preview pane."""

    lines = [
        "FluxForge RAFM Validation Preview",
        "================================",
        "",
    ]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(f"Overall pass: {'yes' if payload.get('overall_passed') else 'no'}")
    lines.append(f"Analyzed raw spectra: {payload.get('n_raw_analyzed', 0)}")
    lines.append(f"Matched raw/QG pairs: {payload.get('n_matched_pairs', 0)}")
    lines.append(f"Unmatched raw files: {payload.get('n_unmatched_raw', 0)}")
    lines.append(f"Unmatched QG files: {payload.get('n_unmatched_qg', 0)}")
    lines.append(
        f"QG consistency flags: {payload.get('qg_internal_consistency_flags', 0)}"
    )
    lines.append(
        f"FluxForge line consistency flags: {payload.get('fluxforge_line_consistency_flags', 0)}"
    )
    lines.append(f"Measurement QC flags: {payload.get('measurement_qc_flags', 0)}")
    lines.append("")

    line_buckets = payload.get("line_diagnostic_buckets") or {}
    if line_buckets:
        lines.append("Line diagnostic buckets")
        lines.append("-----------------------")
        for key in sorted(line_buckets):
            lines.append(f"{key}: {line_buckets[key]}")
        lines.append("")

    failing = payload.get("failing_samples") or []
    lines.append("Failing samples")
    lines.append("---------------")
    if failing:
        for item in failing:
            lines.append(f"- {item}")
    else:
        lines.append("None")
    lines.append("")

    methods = payload.get("unfolding_methods") or []
    if methods:
        lines.append("Unfolding methods")
        lines.append("-----------------")
        lines.append(", ".join(str(item) for item in methods))
        lines.append("")

    return "\n".join(lines) + "\n"


def build_gui_rafm_qg_benchmark_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI RAFM QG benchmark preview pane."""

    lines = [
        "FluxForge RAFM QG Benchmark Preview",
        "==================================",
        "",
    ]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(
        f"Overall pass: {'yes' if payload.get('overall_passed', True) else 'no'}"
    )
    lines.append(f"Processed QG spectra: {payload.get('n_qg_processed', 0)}")
    lines.append(f"Reaction rows: {payload.get('n_reaction_rows', 0)}")
    lines.append(f"Output root: {payload.get('results_root', '')}")
    lines.append("")

    methods = payload.get("unfolding_methods") or []
    if methods:
        lines.append("Unfolding methods")
        lines.append("-----------------")
        lines.append(", ".join(str(item) for item in methods))
        lines.append("")

    samples = payload.get("samples") or []
    if samples:
        lines.append("Samples")
        lines.append("-------")
        for row in samples[:12]:
            lines.append(
                f"{row.get('sample_id', 'sample')}: "
                f"{row.get('n_isotopes', 0)} isotopes, "
                f"{row.get('n_reactions', 0)} reactions"
            )
        if len(samples) > 12:
            lines.append(f"... {len(samples) - 12} more")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_gui_rafm_branch_comparison_preview(
    payload: dict[str, Any], bundle_path: str | Path
) -> str:
    """Return the text shown in the GUI RAFM branch-comparison preview pane."""

    lines = ["FluxForge RAFM Branch Comparison", "===============================", ""]
    lines.append(f"Bundle: {Path(bundle_path)}")
    lines.append(f"Matched reactions: {payload.get('matched_reactions', 0)}")
    lines.append(
        f"Median |activity rel err|: {float(payload.get('median_abs_activity_rel_error', 0.0) or 0.0):.4g}"
    )
    lines.append(
        f"Max |activity rel err|: {float(payload.get('max_abs_activity_rel_error', 0.0) or 0.0):.4g}"
    )
    lines.append(
        f"Median |rate rel err|: {float(payload.get('median_abs_rate_rel_error', 0.0) or 0.0):.4g}"
    )
    lines.append(
        f"Max |rate rel err|: {float(payload.get('max_abs_rate_rel_error', 0.0) or 0.0):.4g}"
    )
    lines.append("")

    methods = payload.get("shared_unfold_methods") or []
    if methods:
        lines.append("Shared unfold methods")
        lines.append("---------------------")
        for method in methods:
            metrics = (payload.get("unfold_metrics") or {}).get(str(method), {})
            lines.append(
                f"{method}: median rel flux err={float(metrics.get('median_abs_rel_flux_error', 0.0) or 0.0):.4g}, "
                f"max rel flux err={float(metrics.get('max_abs_rel_flux_error', 0.0) or 0.0):.4g}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_gui_table_preview(path: str | Path, max_rows: int = 20) -> str:
    """Return a compact human-readable preview for CSV, JSON, text, or markdown artifacts."""

    target = Path(path)
    if not target.exists():
        return f"File not found: {target}\n"

    suffix = target.suffix.lower()
    if suffix == ".csv":
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        preview_rows = rows[: max_rows + 1]
        body = "\n".join(",".join(item for item in row) for row in preview_rows)
        if len(rows) > len(preview_rows):
            body += f"\n... {len(rows) - len(preview_rows)} more row(s)"
        return body + "\n"
    if suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return target.read_text(encoding="utf-8")
