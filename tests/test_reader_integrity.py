"""Reader data-integrity regressions; synthetic fixtures never coerce bad payloads."""

import base64
import struct
from pathlib import Path
import numpy as np
import pytest
from fluxforge.io.spe import (
    GammaSpectrum,
    read_spe_file,
    write_spe_file,
    read_multiple_spe,
)
from fluxforge.io.hpge import read_chn_file
from fluxforge.io.n42 import N42Measurement, read_n42_file, write_n42_file


def dollar(tmp_path, data="0 2\n1 2 3", extra="", name="test.spe"):
    path = tmp_path / name
    path.write_text("$MEAS_TIM:\n100 110\n$DATA:\n" + data + "\n" + extra)
    return path


def chn(tmp_path, counts=(1, 2, 3), declared=None, marker=-1):
    header = bytearray(32)
    struct.pack_into("<h", header, 0, marker)
    struct.pack_into("<II", header, 8, 5500, 5000)
    struct.pack_into(
        "<HH", header, 28, 100, len(counts) if declared is None else declared
    )
    path = tmp_path / "test.chn"
    path.write_bytes(header + struct.pack("<" + "I" * len(counts), *counts))
    return path


def n42(tmp_path, body):
    path = tmp_path / "test.n42"
    path.write_text(
        '<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42">'
        + body
        + "</RadInstrumentData>"
    )
    return path


@pytest.mark.parametrize(
    "data",
    [
        "0 2\n1 2",
        "0 1\n1 2 3",
        "0 2\n1 broken 3",
        "0 2\n1 nan 3",
        "0 2\n1 inf 3",
        "2 0\n1 2 3",
        "0.5 2\n1 2 3",
        "0 2",
    ],
)
def test_spe_rejects_corrupt_payload(tmp_path, data):
    with pytest.raises(ValueError):
        read_spe_file(dollar(tmp_path, data))


def test_shape_header_is_not_a_coefficient(tmp_path):
    result = read_spe_file(dollar(tmp_path, extra="$SHAPE_CAL:\n3\n1.1 0.2 0.03\n"))
    assert result.calibration["shape"] == [1.1, 0.2, 0.03]


@pytest.mark.parametrize("section", ["$MCA_CAL", "$SHAPE_CAL"])
def test_spe_rejects_wrong_coefficient_count(tmp_path, section):
    with pytest.raises(ValueError):
        read_spe_file(dollar(tmp_path, extra=section + ":\n3\n1 .2\n"))


@pytest.mark.parametrize(
    "counts", [[-0.75, 1.5, 3.25], [0, 1.5, 3], [-1, 1, 3], [1, float("nan"), 3]]
)
@pytest.mark.parametrize("exists", [False, True])
def test_spe_unsupported_export_does_not_touch_destination(tmp_path, counts, exists):
    path = tmp_path / "export.spe"
    if exists:
        path.write_bytes(b"original")
    with pytest.raises(ValueError):
        write_spe_file(GammaSpectrum(counts=counts), path)
    assert path.read_bytes() == b"original" if exists else not path.exists()


def test_spe_valid_precision_and_offset_roundtrip(tmp_path):
    before = GammaSpectrum(
        counts=[1, 4, 9],
        channels=[100, 101, 102],
        live_time=100,
        real_time=110,
        calibration={
            "energy": [1.23456789012345, 0.512345678901234, 0],
            "shape": [1.2, 0.03, 0.0001],
        },
    )
    path = tmp_path / "export.spe"
    write_spe_file(before, path)
    after = read_spe_file(path)
    np.testing.assert_array_equal(after.counts, before.counts)
    np.testing.assert_array_equal(after.channels, before.channels)
    assert after.calibration == before.calibration
    assert after.start_time is None


@pytest.mark.parametrize("coeffs", [[1, 0.5], [1, 0.5, 0], [1, 0.5, 0, 0]])
def test_linear_inversion_trailing_zero_and_extrapolation(coeffs):
    spectrum = GammaSpectrum(counts=np.ones(10), calibration={"energy": coeffs})
    np.testing.assert_array_equal(
        spectrum.energy_to_channel(np.array([[-1, 1], [2, 51]])), [[-4, 0], [2, 100]]
    )


@pytest.mark.parametrize("coeffs", [[0, 0, 1], [10, -1, -0.001], [0, 1, 0, 0.001]])
def test_nonlinear_inverse_uses_recorded_channel_domain(coeffs):
    spectrum = GammaSpectrum(
        counts=np.ones(10), channels=np.arange(100, 110), calibration={"energy": coeffs}
    )
    np.testing.assert_array_equal(
        spectrum.energy_to_channel(
            spectrum.channel_to_energy(np.array([100, 104, 109]))
        ),
        [100, 104, 109],
    )


def test_deviation_inverse_uses_channel_offset():
    spectrum = GammaSpectrum(
        counts=np.ones(10),
        channels=np.arange(100, 110),
        calibration={
            "energy": [0, 0.5],
            "deviation_pairs": [
                {"energy_keV": 50, "correction_keV": 0.25},
                {"energy_keV": 55, "correction_keV": 0.5},
            ],
        },
    )
    assert spectrum.energy_to_channel(spectrum.channel_to_energy(105)) == 105


@pytest.mark.parametrize("coeffs", [[5], [1, 0, 0], [1, float("nan")], [25, -10, 1]])
def test_ambiguous_or_invalid_inverse_rejected(coeffs):
    spectrum = GammaSpectrum(counts=np.ones(11), calibration={"energy": coeffs})
    with pytest.raises(ValueError):
        spectrum.energy_to_channel(5)


@pytest.mark.parametrize("kind", ["length", "offset", "energy"])
def test_sum_rejects_misalignment(tmp_path, kind):
    first = dollar(tmp_path, extra="$ENER_FIT:\n0 .5", name="a.spe")
    data = {"length": "0 1\n1 2", "offset": "1 3\n1 2 3", "energy": "0 2\n1 2 3"}[kind]
    second = dollar(
        tmp_path,
        data=data,
        extra="$ENER_FIT:\n0 " + ("1" if kind == "energy" else ".5"),
        name="b.spe",
    )
    with pytest.raises(ValueError):
        read_multiple_spe([first, second], sum_spectra=True)


def test_sum_keeps_counts_exposure_and_uncertainty_together(tmp_path):
    files = [dollar(tmp_path, name=f"{i}.spe") for i in range(2)]
    result = read_multiple_spe(files, sum_spectra=True)
    np.testing.assert_array_equal(result.counts, [2, 4, 6])
    np.testing.assert_allclose(result.counts_uncertainty, np.sqrt([2, 4, 6]))
    assert (result.live_time, result.real_time) == (200, 220)


def test_chn_full_unsigned_range_and_offset(tmp_path):
    result = read_chn_file(chn(tmp_path, [0, 2**31, 2**32 - 1]))
    np.testing.assert_array_equal(result.counts, [0, 2**31, 2**32 - 1])
    np.testing.assert_array_equal(result.channels, [100, 101, 102])


@pytest.mark.parametrize("declared", [0, 4, 32769])
def test_chn_rejects_declared_length_corruption(tmp_path, declared):
    with pytest.raises(ValueError):
        read_chn_file(chn(tmp_path, declared=declared))


def test_chn_rejects_partial_last_uint32(tmp_path):
    path = chn(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        read_chn_file(path)


def test_chn_unknown_header_is_not_guessed(tmp_path):
    with pytest.raises(ValueError):
        read_chn_file(chn(tmp_path, marker=7))


@pytest.mark.parametrize("attribute", ["", ' compressionCode="None"'])
def test_n42_uncompressed_numeric_counts(tmp_path, attribute):
    path = n42(
        tmp_path,
        "<RadMeasurement><Spectrum><ChannelData"
        + attribute
        + ">1234 5678 9012</ChannelData></Spectrum></RadMeasurement>",
    )
    np.testing.assert_array_equal(
        read_n42_file(path).get_spectrum().counts, [1234, 5678, 9012]
    )


def test_n42_counted_zeroes_nist_example(tmp_path):
    path = n42(
        tmp_path,
        '<Spectrum><ChannelData compressionCode="CountedZeroes">22 5 0 1 2 1 0 2 3 4 0 8 1</ChannelData></Spectrum>',
    )
    np.testing.assert_array_equal(
        read_n42_file(path).get_spectrum().counts,
        [22, 5, 0, 2, 1, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    )


@pytest.mark.parametrize(
    "attribute,data",
    [
        ("", "1 bad 3"),
        ("", ""),
        ("", "1 nan 3"),
        (' compressionCode="gzip"', "1 2 3"),
        (' compressionCode="CountedZeroes"', "1 0"),
        (' compressionCode="CountedZeroes"', "0 1.5"),
        (' compressionCode="Base64"', "YWJj"),
    ],
)
def test_n42_corrupt_or_unsupported_data_is_rejected(tmp_path, attribute, data):
    path = n42(
        tmp_path,
        "<Spectrum><ChannelData" + attribute + ">" + data + "</ChannelData></Spectrum>",
    )
    with pytest.raises(ValueError):
        read_n42_file(path)


def test_n42_explicit_legacy_base64_remains_supported(tmp_path):
    text = base64.b64encode(struct.pack("<III", 1, 2**31, 2**32 - 1)).decode()
    path = n42(
        tmp_path,
        '<Spectrum><ChannelData compressionCode="Base64">'
        + text
        + "</ChannelData></Spectrum>",
    )
    np.testing.assert_array_equal(
        read_n42_file(path).get_spectrum().counts, [1, 2**31, 2**32 - 1]
    )


def test_n42_existing_export_roundtrip_keeps_calibration_and_times(tmp_path):
    before = N42Measurement(
        counts=np.array([1, 2, 3]),
        energy_calibration=(1, 0.5, 0.001),
        live_time=10,
        real_time=11,
    )
    path = tmp_path / "roundtrip.n42"
    write_n42_file(path, before)
    after = read_n42_file(path).get_spectrum()
    assert after.energy_calibration == before.energy_calibration
    assert (after.live_time, after.real_time) == (10, 11)


def test_n42_reference_lookup_keeps_multiple_spectra(tmp_path):
    path = n42(
        tmp_path,
        '<EnergyCalibration id="cal"><CoefficientValues>1 .5 0</CoefficientValues></EnergyCalibration><RadMeasurement id="m"><RealTimeDuration>PT11S</RealTimeDuration><Spectrum id="a" energyCalibrationReference="cal"><LiveTimeDuration>PT10S</LiveTimeDuration><ChannelData>1 2</ChannelData></Spectrum><Spectrum id="b" energyCalibrationReference="cal"><ChannelData>3 4</ChannelData></Spectrum></RadMeasurement>',
    )
    doc = read_n42_file(path)
    assert doc.n_spectra == 2
    assert [m.spectrum_id for m in doc.measurements] == ["a", "b"]
    assert all(
        m.energy_calibration == (1, 0.5, 0) and m.real_time == 11
        for m in doc.measurements
    )
    np.testing.assert_array_equal(doc.measurements[1].counts, [3, 4])


def test_n42_missing_reference_rejected(tmp_path):
    path = n42(
        tmp_path,
        '<Spectrum energyCalibrationReference="missing"><ChannelData>1 2 3</ChannelData></Spectrum>',
    )
    with pytest.raises(ValueError):
        read_n42_file(path)


@pytest.mark.parametrize("suffix", [".n42", ".xml", ".chn", ".cnf", ".csv", ".spc"])
def test_cli_delegates_supported_readers_and_preserves_overrides(
    tmp_path, monkeypatch, suffix
):
    from fluxforge.cli.app import _load_spectrum_from_path
    from fluxforge.io import reader_factory

    seen = []

    def reader(path):
        seen.append(path)
        return GammaSpectrum(counts=[1, 2, 3])

    monkeypatch.setattr(reader_factory, "read_spectrum_any", reader)
    path = tmp_path / ("input" + suffix)
    result = _load_spectrum_from_path(path, validate=True, energy_override=[1, 0.5])
    assert seen == [path]
    assert result.calibration["energy"] == [1, 0.5]


def test_cli_real_n42_load(tmp_path):
    from fluxforge.cli.app import _load_spectrum_from_path

    path = n42(
        tmp_path,
        '<Spectrum><ChannelData compressionCode="None">1234 5678 9012</ChannelData></Spectrum>',
    )
    result = _load_spectrum_from_path(path, validate=True)
    np.testing.assert_array_equal(result.counts, [1234, 5678, 9012])
