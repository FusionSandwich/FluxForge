"""Mandatory synthetic format contracts; these are not reactor transport acceptance."""

import numpy as np
import h5py
import pytest

from fluxforge.io.mcnp import parse_mcnp_input, read_meshtal_hdf5


@pytest.mark.parametrize("layout", ["flux", "results", "mesh_mean"])
def test_public_mesh_reader_retains_known_values_and_coordinate_semantics(
    tmp_path, layout
):
    path = tmp_path / "synthetic_mesh.h5"
    flux = np.arange(1, 13, dtype=float).reshape(3, 1, 2, 2, 1)
    error = np.full_like(flux, 0.02)
    edges = np.array([0.001, 0.5, 1e3, 2e7])
    with h5py.File(path, "w") as output:
        output.attrs["validation_scope"] = "synthetic_format_contract"
        prefix = (
            "results/mesh_tally/mesh_tally_85114"
            if layout == "mesh_mean"
            else "tallies/tally_85114"
        )
        group = output.create_group(prefix)
        if layout == "results":
            group["results"] = np.stack([flux, error], axis=-1)
        else:
            group["mean" if layout == "mesh_mean" else "flux"] = flux
            group[
                "relative_standard_error" if layout == "mesh_mean" else "relative_error"
            ] = error
        group["energy_bins"] = edges
    read = read_meshtal_hdf5(str(path), 85114)
    np.testing.assert_array_equal(read["flux"], flux)
    np.testing.assert_array_equal(read["error"], error)
    np.testing.assert_array_equal(read["energy_boundaries"], edges)
    with pytest.raises(ValueError, match="not found"):
        read_meshtal_hdf5(str(path), 999)


def test_public_parser_retains_material_fractions_and_continuations(tmp_path):
    path = tmp_path / "synthetic_model.i"
    path.write_text(
        "Synthetic parser contract\n1 1 -7.8 -1 imp:n=1\n\n"
        "m1 26056.80c -0.9 $ iron\n     24052.80c -0.1\n"
        "m2 1001.80c 2 8016.80c 1\n"
    )
    parsed = parse_mcnp_input(str(path))
    assert parsed["materials"][1]["components"] == [
        ("26056.80c", "-0.9"),
        ("24052.80c", "-0.1"),
    ]
    assert parsed["materials"][2]["components"] == [
        ("1001.80c", "2"),
        ("8016.80c", "1"),
    ]
