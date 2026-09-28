"""Cd ratios use per-atom EOI reaction rates (issue #201)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from fluxforge.examples.rafm_workflow import cd_ratio_rows, load_rafm_example_metadata

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "RAFM_irradiation"


def _artifact(sample_id, activity, rate, rate_unc):
    return {
        "sample_id": sample_id,
        # Raw summed activity deliberately disagrees with the rate ratio
        "isotopes": {"Sc46": {"activity_bq": activity}},
        "reactions": [
            {"reaction_id": "Sc-45(n,g)Sc-46", "reaction_rate": rate, "reaction_rate_unc": rate_unc}
        ],
    }


def test_cd_ratio_from_rates_not_activities() -> None:
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    # Values from the saved RAFM run: masses 0.9645 mg (bare) vs 3.8285 mg (Cd)
    bare = _artifact("Sc-RAFM-1_25cm", 1_607_650.0, 2.1852e-10, 0.02 * 2.1852e-10)
    cd = _artifact("Sc-Cd-RAFM-1_25cm", 159_063.0, 5.4433e-12, 0.25 * 5.4433e-12)
    rows, payload = cd_ratio_rows([bare, cd], metadata)
    assert len(rows) == 1
    row = rows[0]
    assert row["reaction_id"] == "Sc-45(n,g)Sc-46"
    assert row["cd_ratio"] == pytest.approx(2.1852e-10 / 5.4433e-12)  # ~40.1, not 10.1
    assert row["cd_ratio_unc"] == pytest.approx(row["cd_ratio"] * math.hypot(0.02, 0.25))
    assert "Sc Sc-45(n,g)Sc-46" in payload


def test_missing_rate_is_flagged() -> None:
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    rows, _ = cd_ratio_rows(
        [_artifact("Co-RAFM-1_25cm", 1.0, 1e-12, 1e-13), _artifact("Co-Cd-RAFM-1_25cm", 1.0, 0.0, 0.0)],
        metadata,
    )
    assert rows[0]["cd_ratio"] is None and rows[0]["flag_cd_ratio_review"]
