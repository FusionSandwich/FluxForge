"""Report-only QC must expose hypotheses, preserve originals, and avoid repair."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxforge.analysis.qg_report_qc import qg_yield_diagnostic
from fluxforge.examples.rafm_workflow import (
    build_line_diagnostic_records,
    qg_reference_peaks,
)
from fluxforge.io.flux_wire import read_processed_txt

FIXTURES = Path(__file__).parent / "data/flux_wires/processed"


@pytest.mark.parametrize("energy", [983.36, 1311.79])
def test_sc48_keeps_both_conventions_and_explicit_bundled_source(energy):
    row = qg_yield_diagnostic("Sc48", energy, 1.0)
    assert row["reference_rad_int_reported_value"] == 1.0
    assert row["reference_rad_int_reported_unit"] == "unspecified"
    assert row["reference_rad_int_percent_assumption_fraction"] == 0.01
    assert row["reference_rad_int_fraction_assumption"] == 1.0
    assert row["yield_qc_status"] == "yield_convention_discrepancy"
    assert row["bundled_emission_probability"] == 1.0
    assert row["bundled_decay_data_origin"] == "decay_2012 via actigamma"
    assert len(row["bundled_decay_data_sha256"]) == 64
    assert row["scientific_admission"] is False


def test_legitimate_cu64_low_percent_is_not_promoted_to_fraction():
    row = qg_yield_diagnostic("Cu64", 1345.48, 0.47)
    assert row["reference_rad_int_percent_assumption_fraction"] == pytest.approx(0.0047)
    assert row["yield_qc_status"] == "consistent_with_percent_assumption"
    assert not row["fraction_assumption_matches_bundled"]


@pytest.mark.parametrize(
    "isotope,energy", [("Unknown", 983.5), ("Sc48", 900.0), ("Sc48", 10000.0)]
)
def test_distant_or_absent_line_cannot_qualify_a_unit(isotope, energy):
    row = qg_yield_diagnostic(isotope, energy, 1.0)
    assert row["yield_qc_status"] == "reference_unavailable"
    assert "bundled_emission_probability" not in row


@pytest.mark.parametrize("raw", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_yield_stays_unqualified(raw):
    assert (
        qg_yield_diagnostic("Sc48", 983.5, raw)["yield_qc_status"]
        == "invalid_reported_yield_or_energy"
    )


def test_import_preserves_printed_units_and_exact_source_location():
    path = FIXTURES / "Ti-RAFM-1_25cm.txt"
    data = read_processed_txt(path)
    nuclide = next(n for n in data.nuclides if n.isotope == "Sc48")
    peak = nuclide.peaks[1]
    assert peak["reported_rad_int_text"] == "1.00"
    assert peak["reported_activity_text"] == "0.646"
    assert peak["activity_unit"] == "uCi"
    assert peak["source_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "0.646" in path.read_bytes().splitlines()[
        peak["source_line_number"] - 1
    ].decode("utf-8", errors="replace")
    assert nuclide.activity == 0.424
    assert nuclide.report_provenance["reported_activity_text"] == "0.424"


def test_line_column_unit_is_separate_from_summary_unit(tmp_path):
    source = (FIXTURES / "Cu-RAFM-1_25cm.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    source = source.replace("11.079 uCi", "11.079 Bq")
    path = tmp_path / "mixed-activity-units.txt"
    path.write_text(source, encoding="utf-8")
    data = read_processed_txt(path)
    row = qg_reference_peaks(data)[0]
    assert row["header_activity_bq"] == 85.74
    assert row["line_activity_bq"] == pytest.approx(85.74 * 3.7e4)
    assert row["reported_line_activity_unit"] == "uCi"
    assert row["header_activity_unc_bq"] == 11.079


def test_unknown_line_unit_remains_unknown_not_assumed_uci(tmp_path):
    source = (
        (FIXTURES / "Cu-RAFM-1_25cm.txt")
        .read_text(encoding="utf-8", errors="replace")
        .replace("(uCi)", "(widgets)")
    )
    path = tmp_path / "unknown-units.txt"
    path.write_text(source, encoding="utf-8")
    data = read_processed_txt(path)
    assert qg_reference_peaks(data)[0]["line_activity_bq"] is None
    rows, _ = build_line_diagnostic_records("control", "flux_wires", [], data, {})
    assert "line_activity_unit_unqualified" in rows[0]["source_qc_bucket"]


def test_missing_column_header_does_not_inherit_summary_unit(tmp_path):
    source = (FIXTURES / "Cu-RAFM-1_25cm.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    path = tmp_path / "no-column-unit.txt"
    path.write_text(
        "\n".join(
            line for line in source.splitlines() if not line.startswith("CENTROID")
        ),
        encoding="utf-8",
    )
    assert qg_reference_peaks(read_processed_txt(path))[0]["line_activity_bq"] is None


def test_source_flags_survive_missing_raw_match_without_mutation():
    data = read_processed_txt(FIXTURES / "Ti-RAFM-1_25cm.txt")
    before = deepcopy([n.to_dict() for n in data.nuclides])
    rows, consistency = build_line_diagnostic_records("Ti", "flux_wires", [], data, {})
    flagged = [
        r for r in rows if r["yield_qc_status"] == "yield_convention_discrepancy"
    ]
    assert len(flagged) == 2
    assert all(r["diagnostic_bucket"] == "missing_in_fluxforge" for r in flagged)
    assert all(r["flag_line_summary_inconsistency"] for r in flagged)
    assert all(
        r["reference_implied_efficiency_percent_assumption"]
        / r["reference_implied_efficiency_fraction_assumption"]
        == pytest.approx(100.0)
        for r in flagged
    )
    assert next(r for r in consistency if r["isotope"] == "Sc48")[
        "flag_internal_inconsistency"
    ]
    assert before == [n.to_dict() for n in data.nuclides]


def test_source_flags_do_not_replace_raw_parity_bucket_or_correct_activity():
    data = read_processed_txt(FIXTURES / "Ti-RAFM-1_25cm.txt")
    ref = next(
        r
        for r in qg_reference_peaks(data)
        if r["isotope"] == "Sc48" and r["energy_keV"] == 983.36
    )
    raw = SimpleNamespace(
        isotope="Sc48",
        energy_keV=983.36,
        net_counts=ref["net_counts"],
        net_counts_unc=ref["net_unc"],
        gross_counts=ref["gross_counts"],
        gross_counts_unc=ref["gross_unc"],
        comparison_net_counts=None,
        comparison_net_counts_unc=None,
        comparison_gross_counts=None,
        comparison_gross_counts_unc=None,
        activity_bq=ref["line_activity_bq"],
        activity_unc_bq=1.0,
        gamma_line=SimpleNamespace(intensity=1.0, intensity_uncertainty=0.03),
        efficiency=0.0001,
        background_adjusted_gross_counts=None,
        background=0.0,
        significance=10.0,
        assignment_ambiguous=False,
        activity_estimation_state="estimated",
    )
    before = deepcopy(raw.__dict__)
    rows, _ = build_line_diagnostic_records("Ti", "flux_wires", [raw], data, {})
    row = next(r for r in rows if r["reference_energy_keV"] == 983.36)
    assert row["diagnostic_bucket"] == "matched"
    assert "yield_convention_discrepancy" in row["source_qc_bucket"]
    assert row["raw_line_activity_bq"] == ref["line_activity_bq"] == 0.646 * 3.7e4
    assert raw.__dict__ == before


@pytest.mark.parametrize("sample", ["Ti-RAFM-1_25cm", "Cu-RAFM-1_25cm"])
def test_report_and_csv_reference_same_source_flags(sample, tmp_path):
    import csv
    from fluxforge.examples.rafm_workflow import (
        TimingInfo,
        write_sample_comparison_report,
        write_rows_csv,
    )

    path = FIXTURES / (sample + ".txt")
    data = read_processed_txt(path)
    rows, consistency = build_line_diagnostic_records(
        sample, "flux_wires", [], data, {}
    )
    csv_path = tmp_path / "line_diagnostics.csv"
    write_rows_csv(rows, csv_path)
    read_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert [r["source_qc_bucket"] for r in read_rows] == [
        r["source_qc_bucket"] for r in rows
    ]
    timing = TimingInfo(
        "flux_wires", False, None, None, None, None, None, None, "unqualified"
    )
    report = tmp_path / "comparison.txt"
    write_sample_comparison_report(
        sample,
        Path("not_reanalysed.ASC"),
        path,
        timing,
        [],
        [],
        rows,
        consistency,
        [],
        [],
        [],
        [],
        data,
        {},
        report,
    )
    text = report.read_text(encoding="utf-8")
    if sample.startswith("Ti"):
        assert "yield_convention_discrepancy" in text
        assert "percent hypothesis=0.01" in text
        assert "printed RAD INT=1.0" in text
        assert "QG internal consistency flags" in text
    else:
        assert "yield_convention_discrepancy" not in text
    assert "QG source QC (no activity or rate correction)" in text


@pytest.mark.parametrize("unit", [None, "widgets"])
def test_matched_unknown_line_unit_has_no_zero_reference_score(unit):
    data = read_processed_txt(FIXTURES / "Cu-RAFM-1_25cm.txt")
    ref = qg_reference_peaks(data)[0]
    data.nuclides[0].peaks[0]["activity_unit"] = unit
    raw = SimpleNamespace(
        isotope="Cu64",
        energy_keV=ref["energy_keV"],
        net_counts=ref["net_counts"],
        net_counts_unc=ref["net_unc"],
        gross_counts=ref["gross_counts"],
        gross_counts_unc=ref["gross_unc"],
        comparison_net_counts=None,
        comparison_net_counts_unc=None,
        comparison_gross_counts=None,
        comparison_gross_counts_unc=None,
        activity_bq=ref["line_activity_bq"],
        activity_unc_bq=1.0,
        gamma_line=SimpleNamespace(intensity=0.004749, intensity_uncertainty=0.0001),
        efficiency=0.0001,
        background_adjusted_gross_counts=None,
        background=0.0,
        significance=10.0,
        assignment_ambiguous=False,
        activity_estimation_state="estimated",
    )
    rows, _ = build_line_diagnostic_records("Cu", "flux_wires", [raw], data, {})
    assert rows[0]["reference_line_activity_bq"] is None
    assert rows[0]["reference_line_activity_unc_bq"] is None
    assert rows[0]["relative_line_activity_error"] is None
    assert rows[0]["line_activity_en_score"] is None
    assert "line_activity_unit_unqualified" in rows[0]["source_qc_bucket"]
    assert rows[0]["raw_line_activity_bq"] == ref["line_activity_bq"]


def test_reference_hash_binds_file_values_not_mutated_cached_entry(monkeypatch):
    from fluxforge.data.rafm_decay import get_rafm_decay_entry

    cached = get_rafm_decay_entry("Sc48")["gamma_lines"][1]
    monkeypatch.setitem(cached, "intensity", 0.01)
    row = qg_yield_diagnostic("Sc48", 983.5, 1.0)
    assert row["bundled_emission_probability"] == 1.0
    assert row["yield_qc_status"] == "yield_convention_discrepancy"
