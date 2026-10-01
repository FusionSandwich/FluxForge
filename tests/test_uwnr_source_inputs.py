"""Source covariance, operating-log and untouched vendor-table failure paths."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fluxforge.analysis.qg_calibration import QGEfficiencyTable
from fluxforge.analysis.flux_unfold import irradiation_history_factor
from fluxforge.examples.rafm_workflow import (
    RAFMMetadata,
    TimingInfo,
    build_flux_wire_reactions,
    reaction_rows_to_dicts,
    reaction_from_row,
)
from fluxforge.physics.operating_history import (
    history_rate_jacobian,
    load_operating_history,
)
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    rate_covariance,
)

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "examples/RAFM_irradiation/calibration/South Small Vial 25cm.csv"
PROVENANCE = CURVE.with_suffix(".provenance.json")


def test_calibration_covariance_and_ratio_cancellation():
    # Synthetic two-parameter log efficiency; this is not UWNR calibration data.
    cov = [[0.0004, -0.00006], [-0.00006, 0.0001]]
    components = [
        UncertaintyComponent.from_covariance(
            "detector_efficiency",
            cov,
            [-1, -x],
            input_names=["a", "b"],
            input_units=["log fraction", "log fraction"],
            source="synthetic acceptance",
            correlation_group="synthetic-calibration",
            assumed=True,
        )
        for x in [-1, 1]
    ]
    budgets = [
        RateUncertaintyBudget(str(i), rate, [c])
        for i, (rate, c) in enumerate(zip([100, 200], components))
    ]
    absolute = rate_covariance(budgets)
    np.testing.assert_allclose(absolute, [[6.2, 6], [6, 15.2]], rtol=1e-13)
    ratio_jacobian = np.array([1 / 100, -1 / 200])
    assert ratio_jacobian @ absolute @ ratio_jacobian == pytest.approx(0.0004)
    assert budgets[0].missing  # covariance never supplies unknown other terms
    with pytest.raises(ValueError, match="incomplete"):
        rate_covariance(budgets, require_complete=True)


def _component(cov=((1, 0), (0, 2)), names=("a", "b"), units=("s", "relative power")):
    return UncertaintyComponent.from_covariance(
        "irradiation_history",
        cov,
        [-1, 1],
        input_names=names,
        input_units=units,
        source="synthetic",
        correlation_group="one-source",
    )


@pytest.mark.parametrize(
    "other",
    [
        lambda: _component(((2, 0), (0, 2))),
        lambda: _component(names=("b", "a")),
        lambda: _component(units=("h", "relative power")),
    ],
)
def test_shared_covariance_binding_cannot_be_swapped(other):
    with pytest.raises(ValueError, match="binding disagrees"):
        rate_covariance(
            [
                RateUncertaintyBudget("a", 1, [_component()]),
                RateUncertaintyBudget("b", 1, [other()]),
            ]
        )


@pytest.mark.parametrize(
    "cov", [[[1, 2], [0, 1]], [[1, 2], [2, 1]], [[1, float("nan")], [0, 1]]]
)
def test_invalid_input_covariance_rejected(cov):
    with pytest.raises(ValueError):
        _component(cov)


def test_history_jacobian_matches_independent_perturbation():
    segments = [(1800.0, 1.0), (300.0, 0.0), (2700.0, 0.7)]
    gradient, names, units = history_rate_jacobian(43.7 * 3600, segments)
    for i, expected in enumerate(gradient):
        j, k = divmod(i, 2)
        step = 0.01 if k == 0 else 1e-6
        hi, lo = [list(x) for x in segments], [list(x) for x in segments]
        # At zero power use a forward derivative rather than negative power.
        hi[j][k] += step
        if lo[j][k] == 0:
            observed = (
                -(
                    np.log(
                        irradiation_history_factor(43.7 * 3600, irradiation_history=hi)
                    )
                    - np.log(
                        irradiation_history_factor(43.7 * 3600, irradiation_history=lo)
                    )
                )
                / step
            )
        else:
            lo[j][k] -= step
            observed = -(
                np.log(irradiation_history_factor(43.7 * 3600, irradiation_history=hi))
                - np.log(
                    irradiation_history_factor(43.7 * 3600, irradiation_history=lo)
                )
            ) / (2 * step)
        assert observed == pytest.approx(expected, rel=2e-6)
    assert names[0] == "duration_s[0]" and units[0] == "s"


def _metadata(components=None):
    return RAFMMetadata(
        {"rate_uncertainty_components": components or {}},
        {},
        {},
        {"co-rafm-1": [{"mass_mg": 10.0}]},
        {},
        {},
    )


def _timing():
    return TimingInfo(
        "flux_wires", True, "synthetic", None, 7200.0, 0, None, None, "synthetic"
    )


PAYLOAD = {"Co60": {"activity_eoi_bq": 500.0, "activity_eoi_unc_bq": 10.0}}


def _binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _log(tmp_path, **changes):
    power, rods = tmp_path / "power.csv", tmp_path / "rods.csv"
    power.write_text("synthetic power evidence\n")
    rods.write_text("synthetic rod evidence\n")
    shape = tmp_path / "shape.txt"
    shape.write_text("synthetic spectrum-shape evidence\n")
    data = dict(
        schema="fluxforge-operating-history-v1",
        coverage_complete=True,
        source_id="synthetic-log",
        history_model="separable_local_spectrum",
        spectrum_shape_evidence=_binding(shape),
        monitor_ids=["Co-RAFM-1"],
        start="2025-08-05T10:28:00-05:00",
        end="2025-08-05T12:28:00-05:00",
        power_basis="synthetic reference power",
        reference_power={"value": 1, "units": "MW"},
        segments=[{"duration_s": 7200, "relative_power": 1}],
        evidence={
            "reactor_power": dict(_binding(power), units="relative power"),
            "control_rods": dict(_binding(rods), units="inches"),
        },
    )
    data.update(changes)
    p = tmp_path / "log.json"
    p.write_text(json.dumps(data))
    return _binding(p)


def test_physical_requires_log_even_with_named_components():
    declared = {
        name: {"relative": 0.01, "source": "synthetic declared"}
        for name in (
            "activity",
            "detector_efficiency",
            "gamma_yield",
            "half_life",
            "target_mass",
            "isotopic_abundance",
            "irradiation_history",
        )
    }
    with pytest.raises(ValueError, match="complete irradiation operating log"):
        build_flux_wire_reactions(
            "Co-RAFM-1",
            "co-rafm-1",
            PAYLOAD,
            _timing(),
            _metadata(declared),
            mode="physical",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"coverage_complete": False},
        {"end": "2025-08-05T12:28:00"},
        {"evidence": {}},
        {"segments": [{"duration_s": 3600, "relative_power": 1}]},
    ],
)
def test_incomplete_or_unbound_operating_log_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_operating_history(_log(tmp_path, **change))


def test_log_hash_and_segment_join_rejected(tmp_path):
    binding = _log(tmp_path)
    with pytest.raises(ValueError, match="history"):
        load_operating_history(binding, expected_segments=[(7200, 0.5)])
    (tmp_path / "rods.csv").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_operating_history(binding)


def test_workflow_covariance_roundtrip_and_history_integration():
    gradient, names, units = history_rate_jacobian(5.2714 * 365.25 * 86400, [(7200, 1)])
    declared = {
        "irradiation_history": {
            "kind": "irradiation_history",
            "input_covariance": [[4, 0], [0, 0.0001]],
            "input_names": names,
            "input_units": units,
            "source": "synthetic history covariance",
            "correlation_group": "synthetic-log",
        }
    }
    reaction = build_flux_wire_reactions(
        "Co-RAFM-1", "co-rafm-1", PAYLOAD, _timing(), _metadata(declared)
    )[0]
    roundtrip = reaction_from_row(
        json.loads(json.dumps(reaction_rows_to_dicts([reaction])[0]))
    )
    a = next(
        c
        for c in reaction.uncertainty_budget.components
        if c.name == "irradiation_history"
    )
    b = next(
        c
        for c in roundtrip.uncertainty_budget.components
        if c.name == "irradiation_history"
    )
    assert a == b and a.assumed
    assert a.log_sensitivities == pytest.approx(gradient, rel=0.01)
    assert roundtrip.uncertainty_budget.diagnostic_assumptions
    np.testing.assert_allclose(
        rate_covariance([reaction.uncertainty_budget]),
        rate_covariance([roundtrip.uncertainty_budget]),
    )


def test_log_does_not_supply_missing_uncertainty(tmp_path):
    timing = replace(
        _timing(),
        irradiation_operating_log=_log(tmp_path),
        irradiation_end=datetime.fromisoformat("2025-08-05T12:28:00-05:00"),
    )
    with pytest.raises(ValueError, match="incomplete"):
        build_flux_wire_reactions(
            "Co-RAFM-1", "co-rafm-1", PAYLOAD, timing, _metadata(), mode="physical"
        )


@pytest.mark.parametrize("segments", [[(float("nan"), 1)], [(1, float("inf"))]])
def test_nonfinite_irradiation_segments_rejected(segments):
    with pytest.raises(ValueError):
        irradiation_history_factor(3600, irradiation_history=segments)


def test_vendor_curve_exact_bytes_and_invalid_rows_not_clipped():
    table = QGEfficiencyTable(CURVE, PROVENANCE)
    assert (
        table.sha256
        == "03d01e0acbb4da14c3775b00ed239b86187e6b9c16b8d7ed4c2a6be32d3a0000"
    )
    assert table.vendor_coefficients["Error"] == 0.00458934
    assert table.rows[0]["efficiency_reported"] == -0.0151349
    assert (
        table.diagnostic_at(40, unit_assumption="percent")["status"]
        == "excluded_nonpositive_efficiency"
    )
    assert (
        table.diagnostic_at(40.5, unit_assumption="percent")["efficiency_fraction"]
        is None
    )
    assert (
        table.diagnostic_at(983.5, unit_assumption="percent")["efficiency_fraction"] > 0
    )
    assert (
        table.diagnostic_at(0, unit_assumption="percent")["status"]
        == "excluded_outside_table"
    )
    with pytest.raises(ValueError, match="covariance"):
        table.require_physical()


def test_wrong_curve_hash_cannot_be_used(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="SHA256"):
        QGEfficiencyTable(CURVE, bad)


@pytest.mark.parametrize("declared", ["false", "true", 0, 1])
def test_count_decay_declaration_requires_actual_boolean(declared):
    from fluxforge.io.flux_wire import FluxWireData
    from fluxforge.examples.rafm_workflow import report_count_real_time_s
    from fluxforge.analysis.flux_unfold import _report_count_time

    data = FluxWireData(real_time=7200)
    with pytest.raises(ValueError):
        report_count_real_time_s(
            {"qg_report_activity_includes_count_decay": declared}, data
        )
    with pytest.raises(ValueError):
        _report_count_time(declared, data)


def test_complete_synthetic_input_gate_passes_but_not_scientific_admission(tmp_path):
    names = (
        "activity",
        "detector_efficiency",
        "gamma_yield",
        "half_life",
        "target_mass",
        "isotopic_abundance",
        "irradiation_history",
    )
    declared = {
        name: {
            "relative": 0.01,
            "source": "synthetic acceptance input",
            "uncertainty_scope": "itemized",
        }
        for name in names
    }
    timing = replace(
        _timing(),
        irradiation_operating_log=_log(tmp_path),
        irradiation_end=datetime.fromisoformat("2025-08-05T12:28:00-05:00"),
    )
    reaction = build_flux_wire_reactions(
        "Co-RAFM-1", "co-rafm-1", PAYLOAD, timing, _metadata(declared), mode="physical"
    )[0]
    assert reaction.uncertainty_budget.complete
    assert (
        reaction.uncertainty_budget.irradiation_log_binding["source_id"]
        == "synthetic-log"
    )
    assert reaction.uncertainty_budget.as_row()["scientific_admission"] is False
    declared["detector_efficiency"]["assumed"] = True
    with pytest.raises(ValueError, match="assumed"):
        build_flux_wire_reactions(
            "Co-RAFM-1",
            "co-rafm-1",
            PAYLOAD,
            timing,
            _metadata(declared),
            mode="physical",
        )


@pytest.mark.parametrize("series", ["RAFM3", "RAFM4"])
def test_segmented_history_plumbed_for_material_series(series):
    from fluxforge.examples.rafm_workflow import resolve_measurement_timing

    phase = {
        "irradiation_seconds": 7200,
        "irradiation_history": [
            {"duration_s": 3600, "relative_power": 1},
            {"duration_s": 3600, "relative_power": 0.5},
        ],
        "irradiation_operating_log": {"path": "bound-by-later-gate"},
        "cooling_times": [],
    }
    meta = _metadata()
    if series == "RAFM3":
        meta.sample_schedule = {"rafm3_samples": {"A": {"phase1": phase}}}
    else:
        meta.sample_schedules = {
            "schedules": {"A": {"phase2": phase}},
            "irradiation": {},
        }
    timing = resolve_measurement_timing(series + "-A_24hrEOI", None, meta)
    assert timing.irradiation_history == [(3600, 1), (3600, 0.5)]
    assert timing.irradiation_operating_log == {"path": "bound-by-later-gate"}


def test_missing_history_power_is_not_assumed_one():
    from fluxforge.examples.rafm_workflow import _parse_irradiation_history

    with pytest.raises(ValueError, match="explicit"):
        _parse_irradiation_history([{"duration_s": 7200}])


def test_independent_covariance_blocks_do_not_cross_on_diagonal():
    first = _component()
    second = UncertaintyComponent.from_covariance(
        "calibration",
        [[4]],
        [-1],
        input_names=["gain"],
        input_units=["log fraction"],
        source="synthetic separate source",
        correlation_group="separate-source",
    )
    budget = RateUncertaintyBudget("a", 2, [first, second])
    assert rate_covariance([budget])[0, 0] == pytest.approx(4 * (3 + 4))
    assert budget.total_absolute**2 == pytest.approx(28)


def test_shared_covariance_source_does_not_depend_on_display_name():
    first = _component()
    second = replace(first, name="same-source-other-label")
    c = rate_covariance(
        [
            RateUncertaintyBudget("a", 1, [first]),
            RateUncertaintyBudget("b", 2, [second]),
        ]
    )
    assert c[0, 1] == pytest.approx(6)


def test_duplicate_counts_preserve_shared_calibration_rank():
    component = _component()
    c = rate_covariance(
        [RateUncertaintyBudget(str(i), 100, [component]) for i in range(3)]
    )
    np.testing.assert_allclose(c, np.full((3, 3), 30000.0), rtol=1e-14, atol=0)
    assert np.linalg.matrix_rank(c) == 1
    assert float(np.ones(3) @ c @ np.ones(3) / 9) == pytest.approx(
        30000
    )  # no 1/sqrt(3) gain


def test_physical_power_log_does_not_prove_spectrum_shape(tmp_path):
    timing = replace(
        _timing(),
        irradiation_operating_log=_log(
            tmp_path, history_model="unknown", spectrum_shape_evidence=None
        ),
        irradiation_end=datetime.fromisoformat("2025-08-05T12:28:00-05:00"),
    )
    with pytest.raises(ValueError, match="separability"):
        build_flux_wire_reactions(
            "Co-RAFM-1", "co-rafm-1", PAYLOAD, timing, _metadata(), mode="physical"
        )


def test_joint_calibration_yield_coverage_does_not_add_vendor_total():
    spec = {
        "activity": {
            "input_covariance": [[0.0004, 0.0001], [0.0001, 0.0009]],
            "log_sensitivities": [-1, -1],
            "input_names": ["calibration", "gamma_yield"],
            "input_units": ["log fraction", "log fraction"],
            "source": "synthetic joint primitive inputs",
            "correlation_group": "synthetic-joint",
            "covers": ["detector_efficiency", "gamma_yield"],
            "uncertainty_scope": "marginalized",
            "assumed": True,
        }
    }
    reaction = build_flux_wire_reactions(
        "Co-RAFM-1", "co-rafm-1", PAYLOAD, _timing(), _metadata(spec)
    )[0]
    budget = reaction.uncertainty_budget
    assert len(budget.components) == 1
    assert budget.components[0].uncertainty_scope == "marginalized"
    assert budget.total_relative**2 == pytest.approx(0.0015)
    spec["gamma_yield"] = {"relative": 0.03, "source": "duplicate synthetic yield term"}
    with pytest.raises(ValueError, match="overlap"):
        build_flux_wire_reactions(
            "Co-RAFM-1", "co-rafm-1", PAYLOAD, _timing(), _metadata(spec)
        )


@pytest.mark.parametrize(
    "reference", [None, {"value": 0, "units": "MW"}, {"value": 1, "units": ""}]
)
def test_physical_history_requires_reference_power(tmp_path, reference):
    with pytest.raises(ValueError, match="reference power"):
        load_operating_history(
            _log(tmp_path, reference_power=reference), require_separability=True
        )


def test_mixed_scalar_and_vector_shared_source_is_rejected():
    vector = _component()
    scalar = UncertaintyComponent(
        "different-label", 0.1, vector.correlation_group, "same source"
    )
    with pytest.raises(ValueError, match="mix scalar"):
        rate_covariance(
            [
                RateUncertaintyBudget("a", 1, [vector]),
                RateUncertaintyBudget("b", 2, [scalar]),
            ]
        )


def test_source_replay_checks_complete_profile_and_original_windows_bytes():
    import runpy

    replay = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/validate_uwnr_data_use.py")
    )
    count = {
        "QG_report_header": {"detector_id": "South", "source_distance_cm_printed": 25},
        "candidate_C1_C4_A_within_QG_printed_rounding": {},
    }
    assert replay["nominal_profile_matches"](count) is False
    count["candidate_C1_C4_A_within_QG_printed_rounding"] = dict.fromkeys(
        ["C1", "C2", "C3", "C4", "A"], True
    )
    assert replay["nominal_profile_matches"](count) is True
    line = b"counts 12 \xb1 2"
    assert (
        replay["corroborated_source_line"](line + b"\r\n", 1, "counts 12 ± 2")
        == hashlib.sha256(line).hexdigest()
    )
    with pytest.raises(ValueError, match="disagrees"):
        replay["corroborated_source_line"](line, 1, "counts 13 ± 2")


@pytest.mark.parametrize(
    "direction,sensitivity", [([0.14, 0.23], [0.23, -0.14]), ([1, 3], [3, -1])]
)
def test_singular_source_cancellation_remains_psd_and_repeatable(
    direction, sensitivity
):
    from fluxforge.uncertainty.covariance import covariance_matrix

    c = np.outer(direction, direction)
    component = UncertaintyComponent.from_covariance(
        "calibration",
        c,
        sensitivity,
        input_names=["a", "b"],
        input_units=["log fraction", "log fraction"],
        source="synthetic rank one",
        correlation_group="g",
    )
    budget = RateUncertaintyBudget("a", 1, [component])
    for _ in range(3):
        result = rate_covariance([budget])
        assert result[0, 0] >= 0
        covariance_matrix(result, 1)
        assert result[0, 0] == pytest.approx(
            component.relative**2, rel=1e-12, abs=1e-30
        )
    assert result[0, 0] < 1e-14


def test_curve_snapshot_cannot_change_values_under_source_hash():
    directory = (
        Path(__file__).resolve().parents[1] / "examples/RAFM_irradiation/calibration"
    )
    curve = QGEfficiencyTable(
        directory / "South Small Vial 25cm.csv",
        directory / "South Small Vial 25cm.provenance.json",
    )
    before = curve.diagnostic_at(983.5, unit_assumption="percent")
    with pytest.raises(ValueError):
        curve.values[:] *= 2
    outward = curve.values
    outward.setflags(write=True)
    outward *= 2
    curve.rows[0]["efficiency_reported"] = 1
    curve.provenance["sha256"] = "wrong"
    with pytest.raises(AttributeError):
        curve.sha256 = "wrong"
    assert curve.diagnostic_at(983.5, unit_assumption="percent") == before


@pytest.mark.parametrize(
    "ids", ["Co-RAFM-1a", [], [""], ["Co-RAFM-1", "Co-RAFM-1"], [1]]
)
def test_operating_log_monitor_identity_cannot_use_substrings(tmp_path, ids):
    with pytest.raises(ValueError, match="monitor_ids"):
        load_operating_history(
            _log(tmp_path, monitor_ids=ids),
            expected_sample="Co-RAFM-1",
            require_separability=True,
        )
