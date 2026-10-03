"""The Tb correction must never exempt unrelated source rows or native IDs."""

import json
from pathlib import Path

import pytest

from fluxforge.analysis.qg_peak_ids import corrected_reference_ids
from fluxforge.examples.rafm_workflow import qg_reference_peaks
from fluxforge.io.flux_wire import read_processed_txt


ROOT = Path(__file__).resolve().parents[1] / "examples/RAFM_irradiation"


def correction_fixture():
    corrections = json.loads(
        (ROOT / "metadata/qg_peak_id_corrections.json").read_text()
    )["corrections"]
    assert len(corrections) == 1
    item = corrections[0]
    rows = qg_reference_peaks(
        read_processed_txt(ROOT / "QG_processed_gamma_data" / item["report"])
    )
    return item, rows


def test_single_tb_correction_is_source_bound_and_preserves_original():
    item, rows = correction_fixture()
    corrected = corrected_reference_ids(item["report"], rows, [item])
    differences = [
        (before, after) for before, after in zip(rows, corrected) if before != after
    ]
    assert len(differences) == 1
    before, after = differences[0]
    assert before["isotope"] == "Tb154m"
    assert after["isotope"] == "Ta182"
    assert after["reported_isotope"] == "Tb154m"
    assert after["report_source_line_number"] == 92
    assert after["energy_keV"] == 264.28
    assert "identity_correction" not in before
    library = json.loads((ROOT / "metadata/sample_gamma_library.json").read_text())
    assert any(
        abs(line["energy_keV"] - item["library_energy_keV"]) < 1e-6
        for line in library["Ta182"]["gamma_lines"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "changed-report"),
        ("source_line_number", 93),
        ("reported_isotope", "Ta182"),
        ("reported_energy_keV", 265.0),
        ("corrected_isotope", ""),
    ],
)
def test_altered_source_or_invalid_correction_rejected(field, value):
    item, rows = correction_fixture()
    item = dict(item, **{field: value})
    with pytest.raises(ValueError):
        corrected_reference_ids(item["report"], rows, [item])


def test_exception_does_not_apply_to_other_report():
    item, rows = correction_fixture()
    assert corrected_reference_ids("different-report.txt", rows, [item]) == rows


def test_exception_does_not_exempt_all_tb_peaks():
    item, rows = correction_fixture()
    other = dict(next(row for row in rows if row["isotope"] == "Tb154m"))
    other["report_source_line_number"] = 999
    corrected = corrected_reference_ids(item["report"], rows + [other], [item])
    assert corrected[-1]["isotope"] == "Tb154m"
    assert "identity_correction" not in corrected[-1]


def test_duplicate_exception_rejected():
    item, rows = correction_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        corrected_reference_ids(item["report"], rows, [item, item])


def test_corrected_row_never_injects_peak_or_reuses_wrong_activity():
    from fluxforge.analysis.flux_wire_analysis import IdentifiedPeak
    from fluxforge.examples.rafm_workflow import (
        apply_generic_qg_report_parity,
        build_peak_comparison_records,
    )

    item, rows = correction_fixture()
    corrected = corrected_reference_ids(item["report"], rows, [item])
    row = next(row for row in corrected if "identity_correction" in row)
    report = read_processed_txt(ROOT / "QG_processed_gamma_data" / item["report"])
    assert (
        apply_generic_qg_report_parity([], report, {}, reference_identity_rows=[row])
        == []
    )
    native = IdentifiedPeak(
        channel=528,
        energy_keV=264.076,
        net_counts=123,
        net_counts_unc=12,
        gross_counts=150,
        background=27,
        fwhm=2,
        significance=10.25,
        isotope="Ta182",
        activity_bq=456,
        activity_unc_bq=45,
    )
    result = apply_generic_qg_report_parity(
        [native], report, {}, reference_identity_rows=[row]
    )
    assert result == [native]
    assert (native.net_counts, native.activity_bq, native.activity_unc_bq) == (
        123,
        456,
        45,
    )
    assert native.comparison_net_counts is None
    records, missing = build_peak_comparison_records(
        "test", [native], report, {}, reference_identity_rows=[row]
    )
    assert not missing
    assert records[0]["reference_isotope"] == "Ta182"
    assert records[0]["reported_reference_isotope"] == "Tb154m"
    assert records[0]["isotope_match"] is True


def test_all_qg_identity_library_coverage_and_off_energy_reference_are_explicit():
    from fluxforge.analysis.flux_wire_analysis import build_gamma_library
    from fluxforge.examples.rafm_workflow import (
        build_generic_gamma_library,
        load_rafm_example_metadata,
        energy_tolerance,
    )

    metadata = load_rafm_example_metadata(ROOT)
    material, _ = build_generic_gamma_library(metadata)
    wire = build_gamma_library()
    corrections = json.loads(
        (ROOT / "metadata/qg_peak_id_corrections.json").read_text(encoding="utf-8")
    )["corrections"]
    checked = 0
    off_energy = []
    for path in (ROOT / "QG_processed_gamma_data").rglob("*.txt"):
        report = read_processed_txt(path)
        rows = corrected_reference_ids(
            path.relative_to(ROOT / "QG_processed_gamma_data").as_posix(),
            qg_reference_peaks(report),
            corrections,
        )
        library = wire if path.parent.name == "flux_wires" else material
        for row in rows:
            lines = [line for line in library if line.isotope == row["isotope"]]
            assert lines, (path.name, row)
            delta = min(abs(line.energy_keV - row["energy_keV"]) for line in lines)
            if delta > energy_tolerance(row["energy_keV"], metadata.config):
                off_energy.append((path.name, row["isotope"], row["energy_keV"], delta))
            checked += 1
    assert checked == 294
    # This unpaired report needs its original raw acquisition; do not invent
    # a 2529.1-keV Mn56 library line or widen physical matching tolerances.
    assert len(off_energy) == 1
    assert off_energy[0][:3] == ("RAFM3-A_300sEOI.txt", "Mn56", 2529.1)
    assert off_energy[0][3] == pytest.approx(6.04)
