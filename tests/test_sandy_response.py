"""SANDY labelled-output contract tests, not an external NJOY execution claim."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluxforge.data.njoy import parse_gendf
from fluxforge.data.response_bundle import read_response_bundle, write_response_bundle
from fluxforge.data.sandy_response import (
    process_neutron_endf,
    response_from_errorr,
)
from fluxforge.solvers.parametric import GaussianResponseModel


class ProcessedErrorr:
    def __init__(self):
        self.edges = [0.1, 1, 10]
        intervals = pd.IntervalIndex.from_breaks(self.edges)
        columns = pd.MultiIndex.from_tuples([(125, 102), (125, 2)], names=["MAT", "MT"])
        self.xs = pd.DataFrame(
            [[2.0, 4.0], [3.0, 5.0]], index=intervals, columns=columns
        )
        self.labels = pd.MultiIndex.from_tuples(
            [(mat, mt, e) for mat, mt in columns for e in intervals],
            names=["MAT", "MT", "E"],
        )
        factor = np.array([[1, 0.2], [2, -0.1], [3, 0.4], [4, 0.3]]) * 0.01
        self.cov = pd.DataFrame(
            factor @ factor.T, index=self.labels, columns=self.labels
        )

    def get_energy_grid(self):
        return np.array(self.edges)

    def get_xs(self, mts=None):
        # Deliberate independent permutation of all output axes.
        return SimpleNamespace(data=self.xs.iloc[::-1, ::-1])

    def get_cov(self, mts=None):
        return SimpleNamespace(data=self.cov.iloc[[3, 0, 2, 1], [1, 2, 0, 3]])


REQUESTS = [("elastic", 125, 2), ("capture", 125, 102)]


def test_label_alignment_and_relative_to_absolute_covariance_conversion():
    errorr = ProcessedErrorr()
    result = response_from_errorr(errorr, REQUESTS, covariance_kind="relative")
    expected_response = np.array([[4, 5], [2, 3]]) * 1e-24
    permutation = [2, 3, 0, 1]
    relative = errorr.cov.to_numpy()[np.ix_(permutation, permutation)]
    expected_covariance = relative * np.outer(
        expected_response.ravel(), expected_response.ravel()
    )
    np.testing.assert_allclose(
        result.response_cm2, expected_response, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        result.covariance_cm4, expected_covariance, rtol=1e-14, atol=0
    )
    assert result.covariance_cm4[0, 3] != 0  # cross-reaction covariance retained
    assert result.metadata["physical_qualified"] is False
    # Directly consumable by parametric inference with matching per-atom rates.
    likelihood = GaussianResponseModel(
        result.response_cm2, [1e-13, 2e-13], np.eye(2) * 1e-28, result.covariance_cm4
    )
    assert np.isfinite(likelihood.log_likelihood([1e10, 2e10]))


def test_absolute_covariance_not_scaled_by_mean_twice():
    errorr = ProcessedErrorr()
    result = response_from_errorr(errorr, REQUESTS, covariance_kind="absolute_barn2")
    expected = errorr.cov.to_numpy()[np.ix_([2, 3, 0, 1], [2, 3, 0, 1])] * 1e-48
    np.testing.assert_allclose(result.covariance_cm4, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("kind", ["unknown", None, "percent"])
def test_covariance_units_never_inferred(kind):
    with pytest.raises(ValueError, match="covariance_kind"):
        response_from_errorr(ProcessedErrorr(), REQUESTS, covariance_kind=kind)


def test_missing_covariance_label_does_not_become_zero():
    errorr = ProcessedErrorr()
    errorr.get_cov = lambda mts=None: SimpleNamespace(data=errorr.cov.iloc[:-1, :-1])
    with pytest.raises(ValueError, match="missing"):
        response_from_errorr(errorr, REQUESTS, covariance_kind="relative")


def test_bundle_roundtrip_is_pickle_free_and_refuses_overwrite(tmp_path):
    original = response_from_errorr(
        ProcessedErrorr(), REQUESTS, covariance_kind="relative"
    )
    path = write_response_bundle(tmp_path / "response.npz", original)
    loaded = read_response_bundle(path)
    assert loaded.reaction_ids == original.reaction_ids
    assert loaded.incident_particle == "neutron"
    np.testing.assert_array_equal(loaded.covariance_cm4, original.covariance_cm4)
    assert loaded.metadata == original.metadata
    with pytest.raises(FileExistsError):
        write_response_bundle(path, original)


def test_processing_call_sets_absolute_covariance_consistent_grid_and_binds_sources(
    tmp_path, monkeypatch
):
    source, executable = tmp_path / "input.endf", tmp_path / "njoy"
    source.write_text("fixture evaluation, not physical ENDF")
    executable.write_text("fixture executable, never run")
    seen = {}

    class Endf:
        @staticmethod
        def from_file(path):
            seen["source"] = path
            return Endf()

        def get_errorr(self, **kwargs):
            seen.update(kwargs)
            return {"errorr33": ProcessedErrorr()}

    monkeypatch.setitem(sys.modules, "sandy", SimpleNamespace(Endf6=Endf))
    bundle = process_neutron_endf(
        source, REQUESTS, [0.1, 1, 10], njoy_executable=executable
    )
    assert seen["errorr33_kws"]["relative"] is False
    assert seen["errorr33_kws"]["iwt"] == seen["groupr_kws"]["iwt"]
    assert seen["errorr33_kws"]["ek"] == seen["groupr_kws"]["ek"]
    assert seen["xs"] is True and seen["chi"] is False
    assert len(bundle.metadata["endf_source_sha256"]) == 64


def test_gendf_placeholder_replaced_with_labelled_parser(monkeypatch):
    output = ProcessedErrorr().xs.sort_index()

    class Gendf:
        @staticmethod
        def from_file(path):
            return SimpleNamespace(get_xs=lambda: SimpleNamespace(data=output))

    monkeypatch.setitem(sys.modules, "sandy", SimpleNamespace(Gendf=Gendf))
    edges, sections = parse_gendf("fixture.gendf")
    np.testing.assert_array_equal(edges, [0.1, 1, 10])
    np.testing.assert_array_equal(sections[102], [2, 3])
    assert set(sections) == {2, 102}


def test_missing_optional_sandy_fails_instead_of_empty_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "sandy", None)
    with pytest.raises(ImportError, match="empty output is not success"):
        parse_gendf("unused.gendf")
