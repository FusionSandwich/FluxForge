"""Unit-explicit, covariance-preserving multigroup response interchange."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from fluxforge.core.probability import covariance_matrix


@dataclass(frozen=True)
class ResponseBundle:
    """Group-averaged cm² response and absolute cm⁴ covariance.

    Flux multiplying this response is group-integrated particles/cm²/s.
    Covariance order is reaction-major then energy group, matching parametric
    inference. Different particles may use this interchange; their data are never
    interchangeable. Source qualification is not granted by parsing or PSD checks.
    """

    reaction_ids: tuple[str, ...]
    energy_edges_eV: np.ndarray
    response_cm2: np.ndarray
    covariance_cm4: np.ndarray
    incident_particle: str
    metadata: dict

    def __post_init__(self):
        ids = tuple(self.reaction_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(v, str) or not v.strip() for v in ids)
        ):
            raise ValueError("reaction IDs must be nonempty and unique")
        edges = np.array(self.energy_edges_eV, dtype=float, copy=True)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or edges[0] < 0
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("energy edges must be finite, nonnegative and increasing")
        response = np.array(self.response_cm2, dtype=float, copy=True)
        if (
            response.shape != (len(ids), edges.size - 1)
            or not np.isfinite(response).all()
            or np.any(response < 0)
        ):
            raise ValueError(
                "response must match reaction/group dimensions and be nonnegative"
            )
        covariance = covariance_matrix(
            self.covariance_cm4, response.size, "covariance_cm4"
        )
        if (
            not isinstance(self.incident_particle, str)
            or not self.incident_particle.strip()
        ):
            raise ValueError("incident_particle must be explicitly identified")
        metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        for name, value in (
            ("reaction_ids", ids),
            ("energy_edges_eV", edges),
            ("response_cm2", response),
            ("covariance_cm4", covariance),
            ("metadata", metadata),
        ):
            object.__setattr__(self, name, value)
        edges.setflags(write=False)
        response.setflags(write=False)


def write_response_bundle(path, bundle):
    """Write without pickle or overwriting an existing artifact."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(
            stream,
            schema="fluxforge.response_bundle.v1",
            reaction_ids=np.array(bundle.reaction_ids),
            energy_edges_eV=bundle.energy_edges_eV,
            response_cm2=bundle.response_cm2,
            covariance_cm4=bundle.covariance_cm4,
            incident_particle=bundle.incident_particle,
            metadata_json=json.dumps(bundle.metadata, sort_keys=True, allow_nan=False),
        )
    return Path(path)


def read_response_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != "fluxforge.response_bundle.v1":
            raise ValueError("unsupported response bundle schema")
        return ResponseBundle(
            tuple(data["reaction_ids"].tolist()),
            data["energy_edges_eV"],
            data["response_cm2"],
            data["covariance_cm4"],
            str(data["incident_particle"]),
            json.loads(str(data["metadata_json"])),
        )
