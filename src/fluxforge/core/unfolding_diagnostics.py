"""Shared diagnostics helpers for unfolding solver outputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def summarize_flux_bins(
    flux: Any,
    *,
    negative_tolerance: float = 0.0,
    negative_policy: str | None = None,
    nonnegativity_enforced: bool | None = None,
    preview_limit: int = 8,
) -> dict[str, Any]:
    """Return a JSON-friendly summary of unfolded flux bin behavior."""

    flux_array = np.asarray(flux, dtype=float).reshape(-1)
    if flux_array.size == 0:
        summary: dict[str, Any] = {
            "flux_bin_count": 0,
            "finite_bin_count": 0,
            "nonfinite_bin_count": 0,
            "negative_bin_count": 0,
            "negative_bin_fraction": 0.0,
            "has_negative_bins": False,
            "negative_bin_index_preview": [],
            "negative_bin_min_flux": 0.0,
            "min_flux": 0.0,
            "max_flux": 0.0,
            "negative_tolerance": float(abs(negative_tolerance)),
        }
    else:
        tolerance = float(abs(negative_tolerance))
        finite_mask = np.isfinite(flux_array)
        finite_flux = flux_array[finite_mask]
        negative_mask = finite_mask & (flux_array < -tolerance)
        negative_indices = np.flatnonzero(negative_mask)

        summary = {
            "flux_bin_count": int(flux_array.size),
            "finite_bin_count": int(np.count_nonzero(finite_mask)),
            "nonfinite_bin_count": int(np.count_nonzero(~finite_mask)),
            "negative_bin_count": int(negative_indices.size),
            "negative_bin_fraction": float(negative_indices.size / flux_array.size),
            "has_negative_bins": bool(negative_indices.size > 0),
            "negative_bin_index_preview": [
                int(index) for index in negative_indices[:preview_limit]
            ],
            "negative_bin_min_flux": (
                float(np.min(flux_array[negative_mask]))
                if negative_indices.size > 0
                else 0.0
            ),
            "min_flux": float(np.min(finite_flux)) if finite_flux.size > 0 else 0.0,
            "max_flux": float(np.max(finite_flux)) if finite_flux.size > 0 else 0.0,
            "negative_tolerance": tolerance,
        }

    if negative_policy is not None:
        summary["negative_policy"] = str(negative_policy)
    if nonnegativity_enforced is not None:
        summary["nonnegativity_enforced"] = bool(nonnegativity_enforced)
    return summary


def merge_flux_diagnostics(
    base: Mapping[str, Any] | None,
    flux: Any,
    *,
    negative_tolerance: float = 0.0,
    negative_policy: str | None = None,
    nonnegativity_enforced: bool | None = None,
    preview_limit: int = 8,
) -> dict[str, Any]:
    """Merge shared unfolding flux diagnostics into an existing payload."""

    merged = dict(base or {})
    merged.update(
        summarize_flux_bins(
            flux,
            negative_tolerance=negative_tolerance,
            negative_policy=negative_policy,
            nonnegativity_enforced=nonnegativity_enforced,
            preview_limit=preview_limit,
        )
    )
    return merged


__all__ = ["merge_flux_diagnostics", "summarize_flux_bins"]
