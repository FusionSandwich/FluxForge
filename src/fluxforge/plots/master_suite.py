"""Headless master-plan plotting helpers.

This module provides reproducible, non-interactive generation of the core
master-plan plot set (G1.1 - G1.5) from FluxForge-only inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    build_response_matrix,
)
from fluxforge.core.schemas import validate_or_raise
from fluxforge.io.artifacts import (
    read_reaction_rates,
    read_response_bundle,
    read_unfold_result,
)
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    reaction_rate_from_activity,
    weighted_activity,
)
from fluxforge.plots.unfolding import (
    plot_covariance_correlation_heatmaps,
    plot_measured_vs_predicted,
    plot_residuals_pulls,
    plot_response_matrix,
    plot_spectrum_comparison,
    plot_spectrum_uncertainty_bands,
)
from fluxforge.solvers.gls import gls_adjust
from fluxforge.workflows.spectrum_unfolding import UnfoldingResult


@dataclass(frozen=True)
class MasterPlotInputs:
    """Inputs required to generate the master plotting suite."""

    boundaries_eV: np.ndarray
    prior_flux: np.ndarray
    posterior_flux: np.ndarray
    covariance: np.ndarray
    response_matrix: np.ndarray
    reactions: List[str]
    measured_rates: np.ndarray
    measured_uncertainties: np.ndarray
    chi2: float
    method: str = "gls"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_vector(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _as_square_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return arr


def _parse_prior_flux(prior_flux_file: Path) -> np.ndarray:
    payload = _read_json(prior_flux_file)
    if isinstance(payload, list):
        return _as_vector(payload, "prior_flux")
    if isinstance(payload, dict):
        for key in ("prior_flux", "flux"):
            if key in payload:
                return _as_vector(payload[key], f"prior_flux[{key}]")
    raise ValueError(
        "Prior flux file must contain either a JSON list, or a JSON object "
        "with 'prior_flux' or 'flux' key."
    )


def _build_unfolding_result(inputs: MasterPlotInputs) -> UnfoldingResult:
    predicted_rates = np.asarray(
        inputs.response_matrix @ inputs.posterior_flux, dtype=float
    )
    predicted_rates = np.maximum(predicted_rates, 1e-30)
    flux_unc = np.sqrt(np.clip(np.diag(inputs.covariance), a_min=0.0, a_max=None))

    result = UnfoldingResult(
        energy_edges=np.asarray(inputs.boundaries_eV, dtype=float),
        flux=np.asarray(inputs.posterior_flux, dtype=float),
        flux_uncertainty=flux_unc,
        reactions_used=list(inputs.reactions),
        response_matrix=np.asarray(inputs.response_matrix, dtype=float),
        measured_rates=np.asarray(inputs.measured_rates, dtype=float),
        predicted_rates=predicted_rates,
        chi_squared=float(inputs.chi2),
        iterations=0,
        converged=True,
        method=inputs.method.upper(),
        initial_guess_source="artifact",
        metadata={},
    )
    return result


def _save_figure(
    fig, stem: str, output_dir: Path, formats: Sequence[str]
) -> List[Path]:
    from matplotlib import pyplot as plt

    saved: List[Path] = []
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def normalize_plot_formats(plot_format: str) -> Tuple[str, ...]:
    """Normalize CLI plot format selection to explicit file extensions."""
    if plot_format == "both":
        return ("png", "pdf")
    if plot_format in {"png", "pdf"}:
        return (plot_format,)
    raise ValueError(f"Unsupported plot format: {plot_format}")


def generate_master_plan_plots(
    inputs: MasterPlotInputs,
    output_dir: Path,
    *,
    formats: Sequence[str] = ("png",),
    include_response_plot: bool = True,
) -> Dict[str, List[Path]]:
    """
    Generate the core master-plan plot set in a headless-safe way.

    Returns a mapping of plot identifiers to written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _build_unfolding_result(inputs)
    produced: Dict[str, List[Path]] = {}

    fig, _ = plot_spectrum_uncertainty_bands(
        result,
        title="G1.1 Spectrum with Uncertainty Bands",
        save_path=None,
    )
    produced["g1_spectrum_uncertainty"] = _save_figure(
        fig, "g1_spectrum_uncertainty", output_dir, formats
    )

    fig, _ = plot_spectrum_comparison(
        result,
        reference_flux=np.asarray(inputs.prior_flux, dtype=float),
        reference_label="Prior spectrum",
        title="G1.2 Prior vs Posterior Overlay",
        save_path=None,
    )
    produced["g1_prior_posterior_overlay"] = _save_figure(
        fig,
        "g1_prior_posterior_overlay",
        output_dir,
        formats,
    )

    fig, _ = plot_residuals_pulls(
        result,
        rate_uncertainties=np.asarray(inputs.measured_uncertainties, dtype=float),
        title="G1.3 Residual/Pull Diagnostics",
        save_path=None,
    )
    produced["g1_residuals_pulls"] = _save_figure(
        fig, "g1_residuals_pulls", output_dir, formats
    )

    fig, _ = plot_covariance_correlation_heatmaps(
        np.asarray(inputs.covariance, dtype=float),
        np.asarray(inputs.boundaries_eV, dtype=float),
        title="G1.4 Covariance/Correlation Heatmaps",
        save_path=None,
    )
    produced["g1_covariance_correlation"] = _save_figure(
        fig,
        "g1_covariance_correlation",
        output_dir,
        formats,
    )

    fig, _ = plot_measured_vs_predicted(
        result,
        title="G1.5 Predicted vs Measured (Parity)",
        save_path=None,
    )
    produced["g1_parity"] = _save_figure(fig, "g1_parity", output_dir, formats)

    if include_response_plot:
        fig, _ = plot_response_matrix(
            np.asarray(inputs.response_matrix, dtype=float),
            np.asarray(inputs.boundaries_eV, dtype=float),
            list(inputs.reactions),
            title="Response Matrix",
            save_path=None,
        )
        produced["response_matrix"] = _save_figure(
            fig, "response_matrix", output_dir, formats
        )

    return produced


def load_plot_inputs_from_artifacts(
    *,
    unfold_file: Path,
    response_file: Path,
    rates_file: Path,
    prior_flux_file: Path,
    validate: bool = True,
) -> MasterPlotInputs:
    """Load master plotting inputs from FluxForge artifact files."""
    unfold_payload = read_unfold_result(unfold_file)
    response_payload = read_response_bundle(response_file)
    rates_payload = read_reaction_rates(rates_file)

    if validate:
        validate_or_raise(unfold_payload)
        validate_or_raise(response_payload)
        validate_or_raise(rates_payload)

    boundaries = _as_vector(unfold_payload["boundaries_eV"], "boundaries_eV")
    posterior_flux = _as_vector(unfold_payload["flux"], "flux")
    covariance = _as_square_matrix(unfold_payload["covariance"], "covariance")
    prior_flux = _parse_prior_flux(prior_flux_file)

    response_matrix = np.asarray(response_payload["matrix"], dtype=float)
    if response_matrix.ndim != 2:
        raise ValueError("Response matrix must be a 2D array")
    reactions = [str(r) for r in response_payload.get("reactions", [])]
    if len(reactions) != response_matrix.shape[0]:
        raise ValueError("Response reaction labels must match number of response rows")

    rates = rates_payload.get("rates", [])
    if len(rates) != response_matrix.shape[0]:
        raise ValueError(
            "Number of reaction rates must match response matrix reaction count "
            f"({len(rates)} != {response_matrix.shape[0]})"
        )

    measured_rates = _as_vector([float(r["rate"]) for r in rates], "measured_rates")
    measured_unc = _as_vector(
        [float(r.get("uncertainty", 0.0)) for r in rates], "measured_uncertainties"
    )

    if boundaries.size != posterior_flux.size + 1:
        raise ValueError("Boundaries must have one more element than posterior flux")
    if covariance.shape[0] != posterior_flux.size:
        raise ValueError("Covariance dimension must match posterior flux length")
    if prior_flux.size != posterior_flux.size:
        raise ValueError("Prior flux length must match posterior flux length")
    if response_matrix.shape[1] != posterior_flux.size:
        raise ValueError(
            "Response matrix group dimension must match posterior flux length"
        )

    return MasterPlotInputs(
        boundaries_eV=boundaries,
        prior_flux=prior_flux,
        posterior_flux=posterior_flux,
        covariance=covariance,
        response_matrix=response_matrix,
        reactions=reactions,
        measured_rates=measured_rates,
        measured_uncertainties=measured_unc,
        chi2=float(unfold_payload.get("chi2", 0.0)),
        method=str(unfold_payload.get("method", "gls")),
    )


def load_example_plot_inputs() -> MasterPlotInputs:
    """Create plot inputs from bundled FluxForge example data only."""
    data_dir = Path(__file__).resolve().parents[1] / "examples" / "fe_cd_rafm_1"

    boundaries = _as_vector(_read_json(data_dir / "boundaries.json"), "boundaries")
    cross_sections = _read_json(data_dir / "cross_sections.json")
    number_densities = _read_json(data_dir / "number_densities.json")
    measurements = _read_json(data_dir / "measurements.json")
    prior_flux = _as_vector(_read_json(data_dir / "prior_flux.json"), "prior_flux")

    groups = EnergyGroupStructure([float(v) for v in boundaries])
    reactions = [
        ReactionCrossSection(reaction_id=r_id, sigma_g=[float(v) for v in sigma_g])
        for r_id, sigma_g in dict(cross_sections).items()
    ]
    nd_values = [float(dict(number_densities)[rx.reaction_id]) for rx in reactions]
    response = build_response_matrix(reactions, groups, nd_values)
    response_matrix = np.asarray(response.matrix, dtype=float)

    segments = [IrradiationSegment(**seg) for seg in dict(measurements)["segments"]]
    measured_rates: List[float] = []
    measured_uncertainties: List[float] = []
    for reaction in dict(measurements)["reactions"]:
        gamma_lines = [GammaLineMeasurement(**line) for line in reaction["gamma_lines"]]
        activity, _ = weighted_activity(gamma_lines)
        rate = reaction_rate_from_activity(activity, segments, reaction["half_life_s"])
        measured_rates.append(float(rate.rate))
        measured_uncertainties.append(float(rate.uncertainty))

    measurement_cov = np.diag(
        np.asarray(measured_uncertainties, dtype=float) ** 2
    ).tolist()
    prior_cov = np.diag((0.25 * prior_flux) ** 2).tolist()

    gls = gls_adjust(
        response_matrix.tolist(),
        list(measured_rates),
        measurement_cov,
        prior_flux.tolist(),
        prior_cov,
    )

    return MasterPlotInputs(
        boundaries_eV=boundaries,
        prior_flux=prior_flux,
        posterior_flux=np.asarray(gls.flux, dtype=float),
        covariance=np.asarray(gls.covariance, dtype=float),
        response_matrix=response_matrix,
        reactions=[rx.reaction_id for rx in reactions],
        measured_rates=np.asarray(measured_rates, dtype=float),
        measured_uncertainties=np.asarray(measured_uncertainties, dtype=float),
        chi2=float(gls.chi2),
        method="gls",
    )
