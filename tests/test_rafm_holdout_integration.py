"""#203 computational integration on saved UWNR RAFM rates.

The saved rates predate the current admission and uncertainty reviews. The
nominal Cd thickness, simple synthetic prior, and assumed 3% shared detector
term make this a code-path test, not an experimental spectrum validation.
The evaluated IRDFF-II archive must already be local; this test never downloads.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.analysis.physical_gls import MonitorRow, SourceBinding, unfold_gls_physical
from fluxforge.data.group_structures import get_group_structure
from fluxforge.data.irdff import DEFAULT_CACHE_DIR, IRDFF_TAB_ARCHIVE_NAME, IRDFFDatabase
from fluxforge.physics.monitor_response import CoverLayer, MonitorResponseSpec, build_monitor_response_matrix


ROOT = Path(__file__).resolve().parents[1]
RATES = ROOT / "artifacts/manual_review/rafm_validation_full/tables/flux_wire_reaction_rates.csv"
CONFIG = ROOT / "examples/RAFM_irradiation/metadata/workflow_config.json"
ARCHIVE = DEFAULT_CACHE_DIR / "tab" / IRDFF_TAB_ARCHIVE_NAME


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_saved_rafm_rates_correlated_holdout_is_sealed() -> None:
    if not ARCHIVE.is_file():
        pytest.skip("local evaluated IRDFF-II archive unavailable; no download attempted")

    wanted = [
        ("Co-RAFM-1_25cm", "Co-59(n,g)Co-60", "Co-60", False),
        ("Co-Cd-RAFM-1_25cm", "Co-59(n,g)Co-60", "Co-60", True),
        ("Sc-RAFM-1_25cm", "Sc-45(n,g)Sc-46", "Sc-46", False),
        ("Sc-Cd-RAFM-1_25cm", "Sc-45(n,g)Sc-46", "Sc-46", True),
    ]
    with RATES.open(newline="", encoding="utf-8") as stream:
        rate_rows = {(r["sample_id"], r["reaction_id"]): r for r in csv.DictReader(stream)}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cover = CoverLayer("Cd", float(config["cd_cover_thickness_cm"]))
    edges = get_group_structure("VITAMIN-J")
    specs = []
    identities = []
    rates = []
    uncertainty = []
    for sample, reaction, product, covered in wanted:
        observation_id = f"{sample}|{reaction}"
        row = rate_rows[(sample, reaction)]
        specs.append(MonitorResponseSpec(
            observation_id, sample, reaction, cover=cover if covered else None,
        ))
        identities.append(MonitorRow(
            observation_id, sample, "Cd" if covered else "bare", reaction, product,
        ))
        rates.append(float(row["reaction_rate"]))
        uncertainty.append(float(row["reaction_rate_unc"]))

    db = IRDFFDatabase(auto_download=False, archive_path=ARCHIVE)
    response_barn, _, response_rows = build_monitor_response_matrix(specs, edges, db)
    assert all(not r.metadata.get("reaction_source", "").startswith("approx") for r in response_rows)
    response = response_barn * 1e-24  # cm2 per group-integral flux
    rates = np.asarray(rates)
    uncertainty = np.asarray(uncertainty)
    # Diagnostic covariance: historical marginal uncertainty plus an assumed
    # detector-wide term. It is not a reconstructed calibration certificate.
    observation_covariance = np.diag(uncertainty**2) + np.outer(0.03 * rates, 0.03 * rates)
    prior = np.full(len(edges) - 1, 1e11)  # synthetic; not the saved plot reference
    prior_covariance = np.diag((5.0 * prior) ** 2)
    bindings = {
        name: SourceBinding(uri, digest, units)
        for name, uri, digest, units in [
            ("row_identities", "fixture://four-rafm-rows", _sha(str(wanted).encode()), "identity"),
            ("energy_edges", "repo://group_structures.json", _sha(edges.tobytes()), "eV"),
            ("rates", str(RATES), _file_sha(RATES), "reactions/target_atom/s"),
            ("response", f"derived://IRDFF-II/{_file_sha(ARCHIVE)}", _sha(response.tobytes()), "cm2"),
            ("prior", "synthetic://flat-group-integrals", _sha(prior.tobytes()), "n/cm2/s"),
            ("prior_covariance", "synthetic://broad-diagonal", _sha(prior_covariance.tobytes()), "(n/cm2/s)^2"),
            ("observation_covariance", "diagnostic://marginal-plus-shared-3-percent", _sha(observation_covariance.tobytes()), "(reactions/target_atom/s)^2"),
        ]
    }
    kwargs = dict(
        rows=identities, energy_edges_eV=edges, measured_rates=rates,
        response_matrix=response, prior_flux=prior,
        prior_covariance=prior_covariance,
        observation_covariance=observation_covariance,
        sources=bindings, source_commit="historical-rafm-integration-fixture",
        holdout_ids=[identities[-1].observation_id],
    )
    result = unfold_gls_physical(**kwargs)
    assert result.fit_ids == tuple(r.observation_id for r in identities[:3])
    assert result.holdout_ids == (identities[-1].observation_id,)
    assert result.response_rank >= 2
    assert np.all(np.isfinite(result.holdout_predictions))
    assert np.all(np.isfinite(result.holdout_predictive_covariance))
    assert result.holdout_predictive_covariance[0, 0] > 0
    assert np.isfinite(result.holdout_standardized_chi2)
    assert result.receipt()["scientific_admission"] is False

    changed = dict(kwargs)
    changed["measured_rates"] = rates.copy()
    changed["measured_rates"][-1] *= 2
    second = unfold_gls_physical(**changed)
    np.testing.assert_array_equal(result.flux, second.flux)
    np.testing.assert_array_equal(result.holdout_predictions, second.holdout_predictions)
