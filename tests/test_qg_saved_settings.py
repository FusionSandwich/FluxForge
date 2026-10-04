"""Source witnesses and malformed-layout checks for the bounded ANS diagnostic."""

import importlib.util
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ROOT / "examples/RAFM_irradiation/quantumgold_reference/originals"
spec = importlib.util.spec_from_file_location(
    "saved_settings", ROOT / "tools/audit_qg_saved_settings.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_all_original_saved_headers_have_structural_and_report_witnesses():
    rows = [
        audit.inspect_file(path, ORIGINALS / "QG_report")
        for path in sorted((ORIGINALS / "ANS").glob("*.ANS"))
    ]
    assert len(rows) == 32
    assert sum(bool(row["report_anchors"]) for row in rows) == 31
    assert {row["values"]["analysis_ctrl"] for row in rows} == {1}
    assert all(
        not row["saved_setting_inference"]["ambient_background_correction_enabled"]
        for row in rows
    )
    assert all(
        row["saved_setting_inference"]["continuum_correction_enabled"] for row in rows
    )
    # The published byte offsets do not describe these source files. This
    # independent counterexample prevents silently adopting them as a decoder.
    assert all(
        row["wrong_printed_offset_counterexample"]["analysis_ctrl_at_printed_852"] == 0
        for row in rows
    )
    assert all(
        row["wrong_printed_offset_counterexample"]["use_lib_eff_at_printed_1280"]
        == 8224
        for row in rows
    )


@pytest.mark.parametrize(
    "offset,fmt,value",
    [(0, "<h", 3), (1022, "<h", 4095), (1026, "<h", 100), (860, "<H", 4)],
)
def test_saved_layout_rejects_incompatible_files(tmp_path, offset, fmt, value):
    data = bytearray(next((ORIGINALS / "ANS").glob("*.ANS")).read_bytes())
    struct.pack_into(fmt, data, offset, value)
    target = tmp_path / "mutated.ANS"
    target.write_bytes(data)
    with pytest.raises(ValueError):
        audit.inspect_file(target, tmp_path)


def test_saved_flag_changes_the_inference_without_establishing_report_state(tmp_path):
    data = bytearray(next((ORIGINALS / "ANS").glob("*.ANS")).read_bytes())
    struct.pack_into("<H", data, 860, 0)
    target = tmp_path / "ambient_enabled.ANS"
    target.write_bytes(data)
    result = audit.inspect_file(target, tmp_path)
    assert result["saved_setting_inference"]["ambient_background_correction_enabled"]
    assert "SAVED_HEADER_ONLY" in result["saved_setting_inference"]["qualification"]
    assert result["report_anchors"] == {}
