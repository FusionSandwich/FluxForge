"""Batch unfolding exports own their temporary pyplot figures."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from fluxforge.examples.rafm_workflow import save_unfolding_artifacts
from fluxforge.workflows.spectrum_unfolding import UnfoldingResult


def _result():
    return UnfoldingResult(
        energy_edges=np.array([1.0, 10.0, 100.0]),
        flux=np.array([2.0, 3.0]),
        flux_uncertainty=np.array([0.2, 0.3]),
        reactions_used=["A", "B"],
        response_matrix=np.eye(2),
        measured_rates=np.array([2.0, 3.0]),
        predicted_rates=np.array([2.0, 3.0]),
        method="TEST",
        metadata={"diagnostic_only": True},
    )


def test_batch_export_closes_only_its_figures_even_if_save_fails(tmp_path, monkeypatch):
    owner, _ = plt.subplots()
    before = set(plt.get_fignums())
    result = _result()
    original_flux = result.flux.copy()
    output = tmp_path / "unfolding"
    output.mkdir()
    (tmp_path / "plots" / "unfolding").mkdir(parents=True)
    try:
        for _ in range(2):
            save_unfolding_artifacts(result, np.array([2.0, 3.0]), output)
            assert set(plt.get_fignums()) == before
            assert plt.fignum_exists(owner.number)
            paths = sorted((tmp_path / "plots" / "unfolding").glob("*.png"))
            assert len(paths) == 4
            for path in paths:
                with Image.open(path) as png:
                    png.verify()

        savefig = matplotlib.figure.Figure.savefig

        def fail_uncertainty(self, filename, *args, **kwargs):
            if Path(filename).name == "test_uncertainty.png":
                raise OSError("simulated save failure")
            return savefig(self, filename, *args, **kwargs)

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_uncertainty)
        with pytest.raises(OSError, match="simulated save failure"):
            save_unfolding_artifacts(result, np.array([2.0, 3.0]), output)
        assert set(plt.get_fignums()) == before
        assert plt.fignum_exists(owner.number)
        np.testing.assert_array_equal(result.flux, original_flux)
    finally:
        plt.close(owner)
