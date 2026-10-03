"""Report inventory must preserve nondetections without inventing activities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fluxforge.io.flux_wire import read_processed_txt


REPORT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "RAFM_irradiation"
    / "QG_processed_gamma_data"
)
REPORTS = sorted(REPORT_ROOT.rglob("*.txt"))


def _assignment_lines(path):
    # Independent inventory: identify source rows by their assignment column,
    # without using FluxForge's ROI regex or requiring a final activity value.
    return {
        index: line
        for index, line in enumerate(
            path.read_bytes().decode("utf-8", errors="replace").splitlines(), 1
        )
        if "@" in line and line.lstrip()[:1].isdigit()
    }


@pytest.mark.parametrize("report", REPORTS, ids=lambda path: path.stem)
def test_every_report_assignment_has_exact_source_provenance(report):
    reference = _assignment_lines(report)
    data = read_processed_txt(report)
    peaks = [peak for nuclide in data.nuclides for peak in nuclide.peaks]
    assert len(peaks) == len(reference)
    assert {peak["source_line_number"] for peak in peaks} == set(reference)
    source_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    for peak in peaks:
        assert peak["source_line_text"] == reference[peak["source_line_number"]]
        assert peak["source_file_sha256"] == source_hash
        assert peak["source_file"] == str(report.resolve())
        assert peak["assignment"] in peak["source_line_text"].replace(" ", "")
        assert peak["activity_available"] == (peak["activity"] is not None)
        if peak["activity_available"]:
            assert float(peak["reported_activity_text"]) == peak["activity"]
        else:
            assert peak["reported_activity_text"] is None


def test_all_bundled_roi_counts_and_blank_activity_nondetections():
    assert len(REPORTS) == 33
    rows = [
        (report.stem, peak)
        for report in REPORTS
        for nuclide in read_processed_txt(report).nuclides
        for peak in nuclide.peaks
    ]
    assert len(rows) == 300
    assert sum(peak["net_counts"] > 0 for _, peak in rows) == 294
    blanks = {
        (name, peak["assignment"], peak["net_unc"])
        for name, peak in rows
        if not peak["activity_available"]
    }
    assert blanks == {
        ("RAFM3-A_4dEOI", "Fe59@1099.2", 82),
        ("RAFM3-B_300sEOI", "Mn56@2523.1", 162),
        ("RAFM3-C_300sEOI", "W187@551.6", 240),
        ("RAFM3-C_300sEOI", "Mn56@2523.1", 123),
        ("RAFM3-N_300sEOI", "W187@551.6", 262),
        ("RAFM3-N_300sEOI", "Mn56@2523.1", 134),
    }
    for _, peak in rows:
        if not peak["activity_available"]:
            assert peak["activity"] is None
            assert peak["net_counts"] == 0
            assert peak["net_unc"] > 0


def test_blank_activity_differs_from_reported_zero(tmp_path):
    report = tmp_path / "nondetection.txt"
    report.write_text(
        "NUCLIDES ANALYZED\n"
        "Co60 5.271 a B Activity = 0 ± 1 uCi\n"
        "CENTROID INT (keV) (cnts) (cnts) ASSIGNMENT (uCi)\n"
        "1173.2 99.85 1173.16 450 ± 21 0 ± 82 Co60@1173.2\n"
        "1332.4 99.98 1332.41 500 ± 22 0 ± 93 Co60@1332.5 0.0\n",
        encoding="utf-8",
    )
    blank, zero = read_processed_txt(report).nuclides[0].peaks
    assert blank["activity"] is None
    assert blank["activity_available"] is False
    assert blank["reported_activity_text"] is None
    assert zero["activity"] == 0.0
    assert zero["activity_available"] is True
    assert zero["reported_activity_text"] == "0.0"
