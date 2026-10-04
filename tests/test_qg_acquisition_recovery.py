"""Recovered spectra must retain native channels, clocks and report provenance."""

import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from fluxforge.io.flux_wire import read_raw_asc


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "examples/RAFM_irradiation/recovered_qg_sources"
spec = importlib.util.spec_from_file_location(
    "qg_recovery", ROOT / "tools/recover_qg_acquisitions.py"
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
MANIFEST = json.loads((SOURCES / "source_manifest.json").read_text(encoding="utf-8"))
MEASUREMENTS = {row["measurement_id"]: row for row in MANIFEST["measurements"]}


@pytest.mark.parametrize("mid", list(recovery.RECOVER))
def test_recovered_counts_and_metadata_are_native_not_report_values(mid):
    measurement = MEASUREMENTS[mid]
    blob = (SOURCES / (mid + ".ANS")).read_bytes()
    decoded = recovery.decode_study_ans(blob, measurement)
    asc = (
        ROOT
        / "examples/RAFM_irradiation/raw_gamma_spec"
        / (recovery.RECOVER[mid] + ".ASC")
    )
    data = read_raw_asc(asc)
    assert np.array_equal(data.spectrum.counts, decoded["counts"])
    assert asc.read_bytes() == recovery.render_asc(decoded)
    assert data.live_time == pytest.approx(decoded["live"])
    assert data.real_time == pytest.approx(decoded["real"])
    assert (
        decoded["start"].isoformat() == measurement["QG_header"]["clock_local_unzoned"]
    )
    native_array = struct.pack("<8192I", *data.spectrum.counts.astype(int))
    assert recovery.sha(native_array) == measurement["channel_array_sha256"]


@pytest.mark.parametrize(
    "kind",
    ["revision", "length", "channels", "clock", "duration", "array", "calibration"],
)
def test_corrupt_or_mismatched_native_input_is_rejected(kind):
    mid = "RAFM-A-300s"
    blob = bytearray((SOURCES / (mid + ".ANS")).read_bytes())
    if kind == "revision":
        struct.pack_into("<h", blob, 0, 3)
    elif kind == "length":
        blob.pop()
    elif kind == "channels":
        struct.pack_into("<h", blob, 1022, 4095)
    elif kind == "clock":
        struct.pack_into("<d", blob, 80, struct.unpack_from("<d", blob, 80)[0] + 1)
    elif kind == "duration":
        struct.pack_into("<d", blob, 104, 299)
    elif kind == "array":
        blob[1600] ^= 1
    elif kind == "calibration":
        struct.pack_into("<f", blob, 424, 123.0)
    with pytest.raises(ValueError):
        recovery.decode_study_ans(blob, MEASUREMENTS[mid])


def test_report_encoding_exception_cannot_change_counts_or_identity():
    original = b"Cu64 6601 \xb1 117\r\n"
    canonical = "Cu64 6601 \ufffd 117\r\n".encode("utf-8")
    assert recovery.report_equivalence(original, original) == "identical_bytes"
    assert recovery.report_equivalence(original, canonical).startswith(
        "only_plus_minus"
    )
    for changed in (
        canonical.replace(b"6601", b"6602"),
        canonical.replace(b"Cu64", b"Fe59"),
    ):
        with pytest.raises(ValueError):
            recovery.report_equivalence(original, changed)
