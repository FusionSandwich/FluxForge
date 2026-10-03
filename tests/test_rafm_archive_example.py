from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.examples.rafm_feature_example import (
    compare_conversion,
    verify_assets,
    verify_runtime_input,
)
from fluxforge.examples.rafm_workflow import (
    load_rafm_example_metadata,
    resolve_measurement_timing,
)
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.reader_factory import read_spectrum_any
from fluxforge.io.spe import GammaSpectrum

ROOT = Path(__file__).resolve().parents[1] / "examples/RAFM_irradiation"


def test_archive_inventory_and_known_limits():
    manifest = verify_assets(ROOT)
    assert manifest["canonical_raw_count"] == 32
    assert len(manifest["header_variants"]) == 16
    assert not manifest["accuracy_qualified"]
    assert "rafm_profiles.json" in manifest["runtime_data_sha256"]
    bound = {a["path"] for a in manifest["assets"]}
    assert {
        "background.ASC",
        "metadata/sample_schedule.json",
        "metadata/sample_schedules.json",
        "metadata/workflow_config.json",
    }.issubset(bound)
    assert (
        len([a for a in manifest["assets"] if a["role"] == "unsupported_binary_spc"])
        == 29
    )
    statuses = {a["name"]: a["status"] for a in manifest["archives"]}
    assert statuses["OneDrive_2_10-1-2026.zip"] == "download_errors_only"
    assert statuses["tables-20261001T171053Z-1-001.zip"] == "empty_tables_only"


@pytest.mark.parametrize("count", [None, 0, 28])
def test_example_requires_converted_count_guard(monkeypatch, tmp_path, count):
    import fluxforge.examples.rafm_feature_example as example

    manifest = verify_assets(ROOT)
    manifest.pop("converted_sample_count")
    if count is not None:
        manifest["converted_sample_count"] = count
    monkeypatch.setattr(example, "verify_assets", lambda root: manifest)
    with pytest.raises(ValueError, match="29 converted"):
        example.run_example(ROOT, tmp_path)


@pytest.mark.parametrize(
    "change", ["hash", "missing", "escape", "duplicate", "unrecorded"]
)
def test_manifest_rejects_invalid_evidence(tmp_path, change):
    import hashlib

    raw = tmp_path / "raw_gamma_spec/example.ASC"
    raw.parent.mkdir()
    raw.write_bytes(b"data")
    asset = {
        "path": "raw_gamma_spec/example.ASC",
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "role": "canonical_raw",
    }
    payload = {"assets": [asset], "canonical_raw_count": 1}
    if change == "hash":
        raw.write_bytes(b"wrong")
    if change == "missing":
        raw.unlink()
    if change == "escape":
        asset["path"] = "../outside.ASC"
    if change == "duplicate":
        payload["assets"].append(dict(asset))
    if change == "unrecorded":
        (raw.parent / "extra.ASC").write_bytes(b"extra")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/archive_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_assets(tmp_path)


def test_runtime_profile_hash_rejects_changed_efficiency(tmp_path):
    from fluxforge.examples.rafm_feature_example import digest

    p = tmp_path / "rafm_profiles.json"
    p.write_text('{"efficiency":1}')
    expected = digest(p)
    p.write_text('{"efficiency":2}')
    with pytest.raises(ValueError, match="runtime input"):
        verify_runtime_input(p, expected)


@pytest.mark.parametrize("change", ["missing_format", "extra", "missing_and_unlisted"])
def test_converted_inventory_must_match_manifest_and_format_count(tmp_path, change):
    import hashlib

    asset = {
        "path": "raw_gamma_spec/a.ASC",
        "sha256": hashlib.sha256(b"a").hexdigest(),
        "role": "canonical_raw",
    }
    (tmp_path / "raw_gamma_spec").mkdir()
    (tmp_path / asset["path"]).write_bytes(b"a")
    data = {"assets": [asset], "canonical_raw_count": 1, "converted_sample_count": 1}
    for ext in ["CHN", "SPE", "SPC"]:
        p = tmp_path / f"converted_gamma_spec/{ext}/a.{ext}"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"data")
        data["assets"].append(
            {
                "path": p.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(b"data").hexdigest(),
                "role": "converted_counts_only",
            }
        )
    p = tmp_path / "converted_gamma_spec/CHN/a.CHN"
    if change == "missing_format":
        p.unlink()
    if change == "extra":
        (p.parent / "extra.CHN").write_bytes(b"extra")
    if change == "missing_and_unlisted":
        p.unlink()
        data["assets"] = [
            a for a in data["assets"] if a["path"] != p.relative_to(tmp_path).as_posix()
        ]
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/archive_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_assets(tmp_path)


def test_redistributed_counts_with_same_total_are_rejected():
    a = GammaSpectrum(counts=np.array([2.0, 8.0]))
    b = GammaSpectrum(counts=np.array([8.0, 2.0]))
    assert a.counts.sum() == b.counts.sum()
    with pytest.raises(ValueError, match="counts or channel"):
        compare_conversion(a, b)


def test_shifted_channel_positions_are_rejected():
    a = GammaSpectrum(counts=np.array([2.0, 8.0]))
    b = GammaSpectrum(counts=a.counts.copy(), channels=np.array([1, 2]))
    with pytest.raises(ValueError):
        compare_conversion(a, b)


@pytest.mark.parametrize("extension", ["SPE", "CHN"])
def test_real_converted_counts_do_not_qualify_activity(extension):
    stem = "Co-RAFM-1_25cm"
    a = read_raw_asc(ROOT / f"raw_gamma_spec/flux_wires/{stem}.ASC").spectrum
    b = read_spectrum_any(
        ROOT / f"converted_gamma_spec/{extension}/flux_wires/{stem}.{extension}"
    )
    result = compare_conversion(a, b)
    assert result["state"] == "counts_only"
    assert not result["acquisition_times_equal"]
    assert not result["energy_calibration_equal"]
    assert not result["absolute_activity_qualified"]


def test_real_binary_spc_is_rejected():
    with pytest.raises(ValueError, match="supported ASCII"):
        read_spectrum_any(
            ROOT / "converted_gamma_spec/SPC/flux_wires/Co-RAFM-1_25cm.SPC"
        )


@pytest.mark.parametrize("label", ["72h", "144h", "70d"])
def test_new_rafm2_does_not_invent_eoi_schedule(label):
    stem = f"RAFM2_Long_{label}_EOI"
    raw = read_raw_asc(ROOT / f"raw_gamma_spec/RAFM2/{stem}.ASC").spectrum
    timing = resolve_measurement_timing(
        stem, raw.start_time, load_rafm_example_metadata(ROOT)
    )
    assert not timing.compare_eoi
    assert timing.irradiation_time_s is None
    assert timing.decay_time_s is None
    assert timing.sample_group == "unknown"


def test_composition_missing_values_and_element_units_are_preserved():
    data = json.loads((ROOT / "metadata/material_compositions.json").read_text())
    assert data["units"] == "weight_percent"
    ef = data["materials"]["EUROFER97_2"]["elements_wt_percent"]
    assert ef["Si"] is None
    assert ef["Mn"] == pytest.approx(0.529)
    assert "Mn54" not in ef
    assert sum(v for v in ef.values() if v is not None) == pytest.approx(100.0)
