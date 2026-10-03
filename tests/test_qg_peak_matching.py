from itertools import permutations
import pytest
from fluxforge.analysis.flux_wire_analysis import IdentifiedPeak
from fluxforge.examples.rafm_workflow import match_peak_set


def peak(channel, energy, isotope="Fe59"):
    return IdentifiedPeak(
        channel=channel,
        energy_keV=energy,
        net_counts=100,
        net_counts_unc=10,
        gross_counts=150,
        background=50,
        fwhm=2,
        significance=10,
        isotope=isotope,
    )


def ref(energy, isotope="Fe59"):
    return {"energy_keV": energy, "isotope": isotope}


def test_global_assignment_maximizes_recovery_before_isotope_agreement():
    refs = [ref(100), ref(102, "Ta182")]
    observed = [peak(200, 101, "Fe59"), peak(196, 98, "Ta182")]
    config = {"peak_match_tolerances_keV": [2, 2, 2]}
    result = match_peak_set(refs, observed, config)
    assert [p.channel for p, _ in result] == [196, 200]
    assert not any(agreement for _, agreement in result)


def test_global_assignment_prefers_identity_before_energy():
    refs = [ref(100), ref(101, "Ta182")]
    observed = [peak(200, 100, "Ta182"), peak(202, 101, "Fe59")]
    result = match_peak_set(refs, observed, {})
    assert [p.channel for p, _ in result] == [202, 200]
    assert all(agreement for _, agreement in result)


def test_energy_minimization_and_input_order_invariance():
    refs = [ref(100), ref(102)]
    observed = [peak(201, 100.5), peak(203, 102.5)]
    for source_order in permutations(refs):
        for peak_order in permutations(observed):
            result = match_peak_set(source_order, peak_order, {})
            assert {
                r["energy_keV"]: p.channel for r, (p, _) in zip(source_order, result)
            } == {100: 201, 102: 203}


def test_one_channel_with_two_isotope_labels_is_ambiguous_and_used_once():
    result = match_peak_set(
        [ref(100), ref(100.1, "Ta182")], [peak(200, 100), peak(200, 100.1, "Ta182")], {}
    )
    assert sum(p is not None for p, _ in result) == 1
    assert not any(agreement for _, agreement in result)


def test_duplicate_same_isotope_entries_at_one_channel_do_not_double_recover():
    result = match_peak_set(
        [ref(100), ref(100.1)], [peak(200, 100), peak(200, 100.1)], {}
    )
    assert sum(p is not None for p, _ in result) == 1


def test_tolerance_boundary_and_missing():
    result = match_peak_set([ref(100), ref(110)], [peak(200, 103)], {})
    assert result[0][0].channel == 200
    assert result[1] == (None, False)
    assert match_peak_set([ref(100)], [], {}) == [(None, False)]
    assert match_peak_set([], [], {}) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_reference_and_detected_energies(value):
    with pytest.raises(ValueError, match="finite"):
        match_peak_set([ref(value)], [peak(200, 100)], {})
    with pytest.raises(ValueError, match="finite"):
        match_peak_set([ref(100)], [peak(200, value)], {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_rejects_invalid_tolerances(value):
    with pytest.raises(ValueError, match="tolerances"):
        match_peak_set([ref(100)], [], {"peak_match_tolerances_keV": [value] * 3})


def test_comparison_records_expose_physical_channel_ambiguity(monkeypatch):
    from fluxforge.examples import rafm_workflow

    rows = [dict(ref(100), net_counts=100, net_unc=10)]
    monkeypatch.setattr(rafm_workflow, "qg_reference_peaks", lambda _: rows)
    records, missing = rafm_workflow.build_peak_comparison_records(
        "synthetic", [peak(200, 100), peak(200, 100, "Ta182")], None, {}
    )
    assert not missing
    assert records[0]["matched"]
    assert records[0]["raw_channel"] == 200
    assert records[0]["raw_assignment_ambiguous"]
    assert not records[0]["isotope_match"]


def test_exact_ties_are_stable_under_peak_order_reversal():
    refs = [ref(100), ref(102)]
    observed = [peak(200, 101), peak(201, 101)]
    first = match_peak_set(refs, observed, {})
    second = match_peak_set(list(reversed(refs)), list(reversed(observed)), {})
    assert {r["energy_keV"]: p.channel for r, (p, _) in zip(refs, first)} == {
        r["energy_keV"]: p.channel for r, (p, _) in zip(reversed(refs), second)
    }


def test_assignment_objective_matches_exhaustive_small_cases():
    import itertools
    import random

    rng = random.Random(82)
    for _ in range(12):
        refs = [
            ref(rng.uniform(99, 105), rng.choice(["Fe59", "Ta182"])) for _ in range(3)
        ]
        observed = [
            peak(i + 200, rng.uniform(99, 105), rng.choice(["Fe59", "Ta182"]))
            for i in range(3)
        ]

        def objective(assignment):
            chosen = [j for j in assignment if j is not None]
            if len(chosen) != len(set(chosen)):
                return None
            matched = same = 0
            displacement = 0.0
            for i, j in enumerate(assignment):
                if j is None:
                    continue
                delta = abs(refs[i]["energy_keV"] - observed[j].energy_keV)
                if delta > 3:
                    return None
                matched += 1
                same += refs[i]["isotope"] == observed[j].isotope
                displacement += delta
            return (-matched, -same, displacement)

        expected = min(
            o
            for a in itertools.product([None, 0, 1, 2], repeat=3)
            if (o := objective(a)) is not None
        )
        result = match_peak_set(refs, observed, {})
        actual = objective(
            tuple(None if p is None else p.channel - 200 for p, _ in result)
        )
        assert actual[:2] == expected[:2]
        assert actual[2] == pytest.approx(expected[2])
