"""Declared peak energy tolerances apply to scalar and array JSON outputs."""

import pytest

from fluxforge.validation.reference_parity import _values_close


@pytest.mark.parametrize("field", ["energies_keV[1]", "first_peak_keV"])
def test_peak_energy_tolerance_applies_and_preserves_boundary(field):
    options = dict(
        path=f"detected_peaks_expected.json.{field}",
        tolerances={"energy_keV_abs": 8.0, "peak_count_abs": 2.0},
    )
    assert _values_close(200, 200.000001, **options)
    assert _values_close(200, 208, **options)
    assert not _values_close(200, 208.001, **options)


def test_peak_energy_tolerance_does_not_leak_to_channels_or_count():
    tolerances = {"energy_keV_abs": 8.0, "peak_count_abs": 2.0}
    assert not _values_close(
        20, 20.0001, path="output.channels[0]", tolerances=tolerances
    )
    assert _values_close(3, 5, path="output.peak_count", tolerances=tolerances)
    assert not _values_close(3, 6, path="output.peak_count", tolerances=tolerances)


def test_explicit_energy_field_tolerance_takes_precedence():
    assert not _values_close(
        200,
        201,
        path="output.energies_keV[0]",
        tolerances={"energy_keV_abs": 8.0, "energies_keV_abs": 0.01},
    )
