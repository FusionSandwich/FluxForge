import numpy as np
import pytest
from fluxforge.io.spectrum_csv import read_spectrum_csv


@pytest.mark.parametrize("header", ["energy_keV", "ENERGY_KEV", "Energy_KeV"])
def test_energy_and_supplied_uncertainty_survive_read(tmp_path, header):
    p = tmp_path / "example.csv"
    p.write_text(f"channel,{header},counts,uncertainty\n0,1173.2,-5,8\n1,1173.7,9,6\n")
    s = read_spectrum_csv(p)
    np.testing.assert_array_equal(s.energies, [1173.2, 1173.7])
    np.testing.assert_array_equal(s.counts, [-5, 9])
    np.testing.assert_array_equal(s.counts_uncertainty, [8, 6])


@pytest.mark.parametrize(
    "row",
    [
        "0,,5,2",
        "0,1,5,",
        "0,1,5,-2",
        "0,1,5,nan",
        "0,nan,5,2",
        "0,1,inf,2",
        "0,1,,2",
        ",1,5,2",
    ],
)
def test_missing_or_invalid_optional_values_do_not_become_unqualified_defaults(
    tmp_path, row
):
    p = tmp_path / "example.csv"
    p.write_text("channel,energy_keV,counts,uncertainty\n" + row + "\n")
    with pytest.raises(ValueError):
        read_spectrum_csv(p)
