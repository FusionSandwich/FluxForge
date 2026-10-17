"""Optional SANDY/NJOY neutron response adapter with labelled covariance.

API reviewed against SANDY develop 2bb1da40b436c8d88ba83e5a11a1aae0d7a78d2c.
SANDY/NJOY are optional and imported/executed only by the explicit processing API.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from fluxforge.data.response_bundle import ResponseBundle


def _requests(requests):
    requests = tuple(requests)
    if not requests:
        raise ValueError("at least one (reaction_id, MAT, MT) request is required")
    for item in requests:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("requests must be (reaction_id, MAT, MT) tuples")
        name, mat, mt = item
        if (
            not isinstance(name, str)
            or not name.strip()
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x <= 0
                for x in (mat, mt)
            )
        ):
            raise ValueError(
                "requests require a nonempty ID and positive integer MAT/MT"
            )
    requests = tuple(tuple(item) for item in requests)
    if len({r[0] for r in requests}) != len(requests):
        raise ValueError("reaction IDs must be unique")
    if len({r[1:] for r in requests}) != len(requests):
        raise ValueError("MAT/MT requests must be unique")
    return requests


def response_from_errorr(errorr, requests, *, covariance_kind, metadata=None):
    """Convert SANDY ERRORR output without dropping/reordering correlations.

    `covariance_kind` is explicitly 'relative' or 'absolute_barn2'; it is never
    inferred from magnitudes. Requested MAT/MT and interval labels must exist
    in both mean and covariance outputs. Missing covariance never becomes zero.
    """
    requests = _requests(requests)
    if covariance_kind not in ("relative", "absolute_barn2"):
        raise ValueError("declare covariance_kind as relative or absolute_barn2")
    edges = np.asarray(errorr.get_energy_grid(), dtype=float)
    intervals = pd.IntervalIndex.from_breaks(edges)
    mts = sorted({r[2] for r in requests})
    xs = errorr.get_xs(mts=mts).data
    cov = errorr.get_cov(mts=mts).data
    if (
        list(xs.columns.names) != ["MAT", "MT"]
        or list(cov.index.names) != ["MAT", "MT", "E"]
        or list(cov.columns.names) != ["MAT", "MT", "E"]
    ):
        raise ValueError(
            "unsupported SANDY label layout; require MAT/MT/interval labels"
        )
    if (
        not xs.index.is_unique
        or not xs.columns.is_unique
        or not cov.index.is_unique
        or not cov.columns.is_unique
    ):
        raise ValueError("SANDY mean/covariance labels must be unique")
    columns = pd.MultiIndex.from_tuples([r[1:] for r in requests], names=["MAT", "MT"])
    ordered_xs = xs.reindex(index=intervals, columns=columns)
    labels = pd.MultiIndex.from_tuples(
        [(mat, mt, e) for _, mat, mt in requests for e in intervals],
        names=["MAT", "MT", "E"],
    )
    ordered_cov = cov.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    response_barn = ordered_xs.to_numpy(dtype=float).T
    if not np.isfinite(response_barn).all() or not np.isfinite(ordered_cov).all():
        raise ValueError(
            "requested response/covariance labels are missing or nonfinite"
        )
    if covariance_kind == "relative":
        flat = response_barn.flatten(order="C")
        ordered_cov = ordered_cov * flat[:, None] * flat[None, :]
    provenance = dict(metadata or {})
    provenance.update(
        {
            "adapter": "sandy_ERRORR_labelled_v1",
            "covariance_kind_input": covariance_kind,
            "units": {
                "energy": "eV",
                "response": "cm2",
                "covariance": "cm4",
                "flux": "group_integrated_particles/cm2/s",
            },
            "reaction_requests": [list(r) for r in requests],
            "physical_qualified": False,
        }
    )
    return ResponseBundle(
        tuple(r[0] for r in requests),
        edges,
        response_barn * 1e-24,
        ordered_cov * 1e-48,
        "neutron",
        provenance,
    )


def process_neutron_endf(
    endf_path,
    requests,
    energy_edges_eV,
    *,
    njoy_executable,
    temperature_K=300.0,
    weighting_iwt=3,
    tolerance=0.005,
):
    """Explicit local SANDY/NJOY processing; no implicit network downloads.

    Use one evaluation file and an explicitly supplied local NJOY executable.
    ERRORR is requested in absolute barn² units on the same grid/weighting used
    to obtain its mean responses. This function really executes the external
    engine when called; tests of the adapter alone do not verify that execution.
    """
    requests = _requests(requests)
    edges = np.asarray(energy_edges_eV, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or edges[0] < 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("energy_edges_eV must be a finite increasing grid")
    if (
        not np.isfinite([temperature_K, tolerance]).all()
        or temperature_K <= 0
        or tolerance <= 0
    ):
        raise ValueError("temperature and processing tolerance must be positive")
    if (
        isinstance(weighting_iwt, bool)
        or not isinstance(weighting_iwt, int)
        or not 2 <= weighting_iwt <= 12
    ):
        raise ValueError(
            "weighting_iwt must be an explicit built-in NJOY weighting (2..12)"
        )
    source, executable = Path(endf_path).resolve(), Path(njoy_executable).resolve()
    if not source.is_file() or not executable.is_file():
        raise FileNotFoundError(
            "local ENDF evaluation and NJOY executable are required"
        )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    try:
        import sandy
    except ImportError as exc:
        raise ImportError(
            "Install optional SANDY and a local NJOY2016 executable"
        ) from exc
    try:
        sandy_version = version("sandy")
    except PackageNotFoundError:
        sandy_version = "unknown"
    tape = sandy.Endf6.from_file(str(source))
    output = tape.get_errorr(
        temperature=temperature_K,
        err=tolerance,
        xs=True,
        nubar=False,
        mubar=False,
        chi=False,
        exe=str(executable),
        groupr_kws={"ek": edges.tolist(), "iwt": weighting_iwt},
        errorr33_kws={
            "ek": edges.tolist(),
            "iwt": weighting_iwt,
            "relative": False,
            "mt": sorted({r[2] for r in requests}),
        },
    )
    if "errorr33" not in output:
        raise ValueError("SANDY/NJOY did not produce requested MF33 covariance")
    bundle = response_from_errorr(
        output["errorr33"],
        requests,
        covariance_kind="absolute_barn2",
        metadata={
            "sandy_version": sandy_version,
            "endf_source_sha256": source_hash,
            "njoy_executable_sha256": executable_hash,
            "temperature_K": temperature_K,
            "weighting_iwt": weighting_iwt,
            "processing_tolerance": tolerance,
            "external_engine_executed": True,
        },
    )
    if bundle.energy_edges_eV.shape != edges.shape or not np.allclose(
        bundle.energy_edges_eV, edges, rtol=1e-12, atol=0
    ):
        raise ValueError("processed energy grid differs from requested grid")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
        or hashlib.sha256(executable.read_bytes()).hexdigest() != executable_hash
    ):
        raise ValueError("ENDF source or NJOY executable changed during processing")
    return bundle


def read_sandy_gendf(path):
    """Read one-material GENDF mean XS; preserves legacy barn units."""
    try:
        import sandy
    except ImportError as exc:
        raise ImportError(
            "GENDF parsing requires optional SANDY; empty output is not success"
        ) from exc
    data = sandy.Gendf.from_file(str(path)).get_xs().data
    if (
        not isinstance(data.index, pd.IntervalIndex)
        or list(data.columns.names) != ["MAT", "MT"]
        or data.empty
        or not data.index.is_unique
        or not data.columns.is_unique
    ):
        raise ValueError(
            "GENDF must contain uniquely labelled multigroup cross sections"
        )
    if len(data.columns.get_level_values("MAT").unique()) != 1:
        raise ValueError("legacy GENDF return type requires exactly one MAT")
    intervals = data.index
    if not np.array_equal(intervals.left[1:], intervals.right[:-1]):
        raise ValueError("GENDF energy intervals must be contiguous and ordered")
    edges = np.r_[intervals.left[0], intervals.right.to_numpy()]
    values = data.to_numpy(dtype=float)
    if (
        np.any(np.diff(edges) <= 0)
        or not np.isfinite(edges).all()
        or edges[0] < 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("GENDF energies/cross sections are invalid")
    return edges, {
        int(mt): values[:, i].copy() for i, (_, mt) in enumerate(data.columns)
    }
