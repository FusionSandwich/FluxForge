from __future__ import annotations

from pathlib import Path

import pytest

from fluxforge.validation.reference_parity import run_reference_parity_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "tests" / "spectra" / "reference_parity"
ACTIVATION_ROOT = REPO_ROOT / "tests" / "activation_inventory" / "fixtures"


def test_reference_parity_suite_runs_all_fixture_families() -> None:
    payload = run_reference_parity_suite(
        reference_root=REFERENCE_ROOT,
        activation_root=ACTIVATION_ROOT,
        scope="all",
        include_activation=True,
    )

    assert payload["schema"] == "fluxforge.reference_parity.run.v1"
    assert payload["summary"]["total"] >= 8
    assert payload["summary"]["failed"] == 0


def test_reference_parity_suite_supports_scope_and_fixture_filters() -> None:
    payload = run_reference_parity_suite(
        reference_root=REFERENCE_ROOT,
        activation_root=ACTIVATION_ROOT,
        scope="algorithm",
        fixture_id="minimal_peak_search_algorithm_case",
        include_activation=False,
    )

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["passed"] == 1
    assert payload["results"][0]["fixture_id"] == "minimal_peak_search_algorithm_case"


def test_reference_parity_suite_runs_phase5_spectrum_algorithm_bundle() -> None:
    payload = run_reference_parity_suite(
        reference_root=REFERENCE_ROOT,
        activation_root=ACTIVATION_ROOT,
        scope="algorithm",
        fixture_id="spectrum_io_normalization_algorithm_case",
        include_activation=False,
    )

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 0
    result = payload["results"][0]
    assert result["fixture_id"] == "spectrum_io_normalization_algorithm_case"
    assert "io_parity_expected.json" in result["compared_outputs"]


def test_reference_parity_suite_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError):
        run_reference_parity_suite(
            reference_root=REFERENCE_ROOT,
            activation_root=ACTIVATION_ROOT,
            scope="invalid",
        )
