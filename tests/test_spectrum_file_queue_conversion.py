"""Folder conversion preserves bins, covariance and original files."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from fluxforge.core.spectrum_file_queue import (
    convert_spectrum_file_queue,
    discover_spectrum_files,
    sum_queued_spectra,
)
from fluxforge.io.session import read_ffs_session
from fluxforge.io.spe import GammaSpectrum


def _csv(path, counts=(2, 5), sigma=(1, 2)):
    path.write_text(
        "channel,counts,uncertainty\n"
        + "".join(
            f"{index},{count},{error}\n"
            for index, (count, error) in enumerate(zip(counts, sigma))
        )
    )
    return path


def test_discovery_and_append_keep_unique_sources_and_uncertainties(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = _csv(tmp_path / "a.CSV")
    second = _csv(nested / "b.csv", (4, 8), (3, 4))
    (tmp_path / "notes.txt").write_text("not a spectrum")
    assert discover_spectrum_files(tmp_path) == (first,)
    assert discover_spectrum_files(tmp_path, recursive=True) == (first, second)
    target = tmp_path / "joined.ffs"
    result = convert_spectrum_file_queue([first, second, first], target)
    assert result["input_count"] == 2 and result["output_spectra"] == 2
    session = read_ffs_session(target)
    assert len(session.spectra) == 2
    np.testing.assert_array_equal(session.spectra[1].counts_uncertainty, [3, 4])
    assert (
        session.metadata["sources"][0]["sha256"]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )
    before = target.read_bytes()
    with pytest.raises(ValueError, match="existing"):
        convert_spectrum_file_queue([first], target)
    assert target.read_bytes() == before


def test_sum_retains_full_covariance_and_adds_known_times():
    first = GammaSpectrum(
        counts=np.array([2, 5]),
        counts_covariance=np.array([[4, 1], [1, 9]]),
        live_time=2,
        real_time=3,
    )
    second = GammaSpectrum(
        counts=np.array([4, 8]),
        counts_covariance=np.array([[1, -0.5], [-0.5, 4]]),
        live_time=3,
        real_time=5,
    )
    with pytest.raises(ValueError, match="independent"):
        sum_queued_spectra([first, second])
    result = sum_queued_spectra([first, second], independent_acquisitions=True)
    np.testing.assert_array_equal(result.counts, [6, 13])
    np.testing.assert_array_equal(
        result.counts_covariance.toarray(), [[5, 0.5], [0.5, 13]]
    )
    np.testing.assert_allclose(result.counts_uncertainty, np.sqrt([5, 13]))
    assert (result.live_time, result.real_time) == (5, 8)
    assert result.start_time is None
    np.testing.assert_array_equal(first.counts, [2, 5])
    missing = replace(second, live_time=0)
    result = sum_queued_spectra([first, missing], independent_acquisitions=True)
    assert result.live_time == 0 and result.real_time == 8


@pytest.mark.parametrize(
    "change",
    [
        {"channels": np.array([1, 2])},
        {"energies": np.array([1, 2])},
        {"detector_id": "another"},
        {"calibration": {"energy": [0, 2]}},
        {"metadata": {"count_decay_corrected": True}},
    ],
)
def test_sum_rejects_incompatible_inputs_without_silently_dropping_them(change):
    first = GammaSpectrum(counts=np.array([2, 5]))
    second = replace(first, **change)
    with pytest.raises(ValueError):
        sum_queued_spectra([first, second], independent_acquisitions=True)


def test_bad_source_aborts_conversion_and_sum_roundtrip_preserves_sigma(tmp_path):
    first = _csv(tmp_path / "first.csv")
    bad = tmp_path / "broken.csv"
    bad.write_text("invalid\nnotcounts\n")
    target = tmp_path / "combined.ffs"
    with pytest.raises(ValueError):
        convert_spectrum_file_queue([first, bad], target)
    assert not target.exists()
    second = _csv(tmp_path / "second.csv", (4, 8), (3, 4))
    convert_spectrum_file_queue(
        [first, second], target, mode="sum", independent_acquisitions=True
    )
    result = read_ffs_session(target).spectra[0]
    np.testing.assert_array_equal(result.counts, [6, 13])
    np.testing.assert_allclose(result.counts_uncertainty, np.sqrt([10, 20]))
    assert result.live_time == 0
    assert not list(tmp_path.glob(".fluxforge-queue-*"))
