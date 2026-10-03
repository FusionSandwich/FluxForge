"""Known-truth contracts for supplied activity uncertainty (#199/#25)."""

import json
import math
from argparse import Namespace

import pytest

from fluxforge.physics.activation import IrradiationSegment, reaction_rate_from_activity
from fluxforge.cli.app import cmd_rates, _build_reaction_rate_rows_from_activity_review
from fluxforge.io.artifacts import make_reaction_rates, write_line_activities
from fluxforge.core.schemas import validate_artifact


@pytest.mark.parametrize(
    "activity,sigma,expected_rate,expected_sigma",
    [(100, 3, 200, 6), (0, 3, 0, 6), (100, 0, 200, 0), (0, 0, 0, 0)],
)
def test_half_life_truth(activity, sigma, expected_rate, expected_sigma):
    result = reaction_rate_from_activity(
        activity, [IrradiationSegment(60)], 60, activity_uncertainty_bq=sigma
    )
    assert result.rate == pytest.approx(expected_rate)
    assert result.uncertainty == pytest.approx(expected_sigma)
    assert result.uncertainty_scope == "activity_only_conditional"
    assert not result.uncertainty_unavailable_reason


def test_missing_sigma_remains_unavailable():
    result = reaction_rate_from_activity(100, [IrradiationSegment(60)], 60)
    assert result.uncertainty is None
    assert result.uncertainty_scope == "unavailable"
    assert result.uncertainty_unavailable_reason


@pytest.mark.parametrize("sigma", [-1, float("nan"), float("inf")])
def test_invalid_activity_sigma_rejected(sigma):
    with pytest.raises(ValueError):
        reaction_rate_from_activity(
            100, [IrradiationSegment(60)], 60, activity_uncertainty_bq=sigma
        )


@pytest.mark.parametrize("activity", [-1, float("nan"), float("inf")])
def test_invalid_activity_rejected(activity):
    with pytest.raises(ValueError):
        reaction_rate_from_activity(activity, [IrradiationSegment(60)], 60)


@pytest.mark.parametrize(
    "segments,half_life",
    [
        ([IrradiationSegment(-1)], 60),
        ([IrradiationSegment(60, -1)], 60),
        ([IrradiationSegment(float("nan"))], 60),
        ([IrradiationSegment(60, float("inf"))], 60),
        ([IrradiationSegment(60)], 0),
    ],
)
def test_invalid_irradiation_rejected(segments, half_life):
    with pytest.raises(ValueError):
        reaction_rate_from_activity(100, segments, half_life)


def test_review_export_does_not_floor_long_lived_buildup():
    row = {
        "nuclide": "truth",
        "irradiation_time_activity_Bq": 2.0,
        "irradiation_time_activity_unc_Bq": 0.1,
        "half_life_s": 1e20,
    }
    result = _build_reaction_rate_rows_from_activity_review(
        isotope_rows=[row], segments=[IrradiationSegment(1)]
    )[0]
    factor = -math.expm1(-math.log(2) / 1e20)
    assert result["reaction_rate_s"] == pytest.approx(2 / factor, rel=1e-14)
    assert result["reaction_rate_unc_s"] == pytest.approx(0.1 / factor, rel=1e-14)


@pytest.mark.parametrize("sigma", [None, 3, 0])
def test_review_export_keeps_zero_activity_and_sigma_state(sigma):
    row = {"nuclide": "truth", "irradiation_time_activity_Bq": 0.0, "half_life_s": 60}
    if sigma is not None:
        row["irradiation_time_activity_unc_Bq"] = sigma
    result = _build_reaction_rate_rows_from_activity_review(
        isotope_rows=[row], segments=[IrradiationSegment(60)]
    )[0]
    assert result["reaction_rate_s"] == 0
    assert result["reaction_rate_unc_s"] == (sigma * 2 if sigma is not None else None)
    assert result["relative_uncertainty"] is None
    assert not result["scientific_admission"]


@pytest.mark.parametrize("sigma", [None, 3, 0])
def test_rates_cli_propagates_and_serializes_sigma(tmp_path, sigma):
    source = tmp_path / "lines.json"
    line = {
        "reaction_id": "truth",
        "energy_keV": 1,
        "net_counts": 1,
        "activity_Bq": 100,
        "activity_unc_Bq": sigma,
        "half_life_s": 60,
    }
    write_line_activities(source, spectrum_id="truth", lines=[line])
    out = tmp_path / "rates.json"
    cmd_rates(
        Namespace(
            lines_file=source,
            segments_file=None,
            duration_s=60,
            half_life_s=60,
            output=out,
            validate=True,
        )
    )
    payload = json.loads(out.read_text())
    assert not validate_artifact(payload)
    rate = payload["rates"][0]
    assert rate["rate"] == pytest.approx(200)
    assert rate["uncertainty"] == (2 * sigma if sigma is not None else None)
    assert not rate["scientific_admission"]
    assert "NaN" not in out.read_text()


def test_unavailable_schema_requires_reason():
    payload = make_reaction_rates(
        rates=[{"reaction_id": "truth", "rate": 1, "uncertainty": None}]
    )
    assert validate_artifact(payload)


def test_existing_numeric_rate_artifact_stays_compatible():
    payload = make_reaction_rates(
        rates=[{"reaction_id": "truth", "rate": 1, "uncertainty": 0.1}]
    )
    assert not validate_artifact(payload)


def test_cli_unfold_rejects_unavailable_sigma(tmp_path, monkeypatch):
    import fluxforge.cli.app as app

    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {"matrix": [[1]], "boundaries_eV": [1, 2], "reactions": ["truth"]},
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {"rates": [{"rate": 1, "uncertainty": None}]},
    )
    with pytest.raises(ValueError, match="uncertainties.*unavailable"):
        app.cmd_unfold(
            Namespace(
                response_file=tmp_path / "response",
                rates_file=tmp_path / "rates",
                validate=False,
            )
        )


def test_plot_loader_rejects_unavailable_sigma(tmp_path, monkeypatch):
    import fluxforge.plots.master_suite as plots

    monkeypatch.setattr(
        plots,
        "read_unfold_result",
        lambda _: {"boundaries_eV": [1, 2], "flux": [1], "covariance": [[1]]},
    )
    monkeypatch.setattr(
        plots,
        "read_response_bundle",
        lambda _: {"matrix": [[1]], "reactions": ["truth"]},
    )
    monkeypatch.setattr(
        plots,
        "read_reaction_rates",
        lambda _: {"rates": [{"rate": 1, "uncertainty": None}]},
    )
    monkeypatch.setattr(plots, "_parse_prior_flux", lambda _: [1])
    with pytest.raises(ValueError, match="uncertainties.*unavailable"):
        plots.load_plot_inputs_from_artifacts(
            unfold_file=tmp_path / "u",
            response_file=tmp_path / "r",
            rates_file=tmp_path / "m",
            prior_flux_file=tmp_path / "p",
            validate=False,
        )


@pytest.mark.parametrize("sigma", ["", "NaN", "-1"])
def test_csv_import_rejects_missing_or_invalid_sigma(tmp_path, sigma):
    from fluxforge.io.interop import read_saturation_rates_csv

    csv = tmp_path / "rates.csv"
    csv.write_text(f"reaction,rate,uncertainty\ntruth,1,{sigma}\n")
    with pytest.raises(ValueError, match="uncertainty"):
        read_saturation_rates_csv(csv)
