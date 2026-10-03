"""Independent validator tests: input provenance and tentative/confirmed separation."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def validator():
    spec = importlib.util.spec_from_file_location(
        "qg_validator_test_target", ROOT / "tools/validate_qg_peak_identifications.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def validation_case(tmp_path, monkeypatch, validator):
    root = tmp_path / "isolated"
    qg_root = root / "reports"
    metadata_root = root / "metadata"
    qg_root.mkdir(parents=True)
    metadata_root.mkdir()
    (root / "src/fluxforge").mkdir(parents=True)
    raw = [root / f"raw{i}.ASC" for i in range(2)]
    reports = [qg_root / f"report{i}.txt" for i in range(2)]
    for i, path in enumerate(raw + reports):
        path.write_text(f"Independent source fixture {i}", encoding="utf8")
    manifest = metadata_root / "qg_peak_id_corrections.json"
    manifest.write_text(json.dumps({"corrections": []}), encoding="utf8")
    paths = SimpleNamespace(qg_root=qg_root, metadata_root=metadata_root)
    metadata = SimpleNamespace(pairing_aliases={}, config={"profile_name": "fixture"})
    monkeypatch.setattr(
        validator,
        "fluxforge",
        SimpleNamespace(__file__=str(root / "src/fluxforge/__init__.py")),
    )
    monkeypatch.setattr(validator, "default_paths", lambda _: paths)
    monkeypatch.setattr(validator, "load_rafm_example_metadata", lambda _: metadata)
    monkeypatch.setattr(
        validator, "discover_input_files", lambda _: {"raw": raw, "qg": reports}
    )
    monkeypatch.setattr(
        validator,
        "pair_input_files",
        lambda *a: (
            [(r, q, str(i)) for i, (r, q) in enumerate(zip(raw, reports))],
            [],
            [],
        ),
    )

    def read_report(path, **kwargs):
        peak = {
            "source_line_number": 4,
            "center_keV": 100.0,
            "net_counts": 100.0,
            "net_unc": 10.0,
            "activity_available": True,
        }
        return SimpleNamespace(
            nuclides=[SimpleNamespace(isotope="Fe59", peaks=[peak])], path=path
        )

    monkeypatch.setattr(validator, "read_processed_txt", read_report)
    monkeypatch.setattr(
        validator,
        "qg_reference_peaks",
        lambda report: [
            {
                "isotope": "Fe59",
                "energy_keV": 100.0,
                "net_counts": 100.0,
                "net_unc": 10.0,
                "report_source_line_number": 4,
                "report_source_sha256": validator.sha(report.path),
            }
        ],
    )
    native_peak = {
        "energy_keV": 100.1,
        "channel": 200,
        "isotope": "Fe59",
        "net": 100,
        "sigma": 10,
        "significance": 10,
    }
    cache = {
        "complete": True,
        "stage": "native independent detection before reference overlay",
        "input_sha256": {
            p.relative_to(root).as_posix(): validator.sha(p) for p in raw + reports
        },
        "samples": [
            {
                "qg_file": q.relative_to(root).as_posix(),
                "raw_file": r.relative_to(root).as_posix(),
                "peaks": [dict(native_peak)],
            }
            for r, q in zip(raw, reports)
        ],
    }

    def run(changed=None):
        native = root / "native.json"
        out = root / "result.json"
        native.write_text(
            json.dumps(cache if changed is None else changed), encoding="utf8"
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "validator",
                "--root",
                str(root),
                "--native",
                str(native),
                "--out",
                str(out),
            ],
        )
        validator.main()
        return json.loads(out.read_text(encoding="utf8"))

    return SimpleNamespace(cache=cache, run=run, manifest=manifest)


def test_complete_bound_sources_and_alias_normalization_pass(validation_case):
    result = validation_case.run()
    assert result["passed"] is True
    assert result["metrics"]["status_counts"] == {"same_id": 2}
    assert all(row["confirmed_detection"] for row in result["rows"])
    assert result["scientific_admission"] is False


@pytest.mark.parametrize(
    "defect, message",
    [
        ("partial_hashes", "hash every paired"),
        ("changed_hash", "Native input changed"),
        ("incomplete", "incomplete"),
        ("missing_report", "cover every paired report"),
        ("duplicate_report", "cover every paired report"),
        ("swapped_raw", "pairing changed"),
        ("reference_overlay", "precede all reference overlays"),
    ],
)
def test_rejects_invalid_cache_provenance(validation_case, defect, message):
    cache = deepcopy(validation_case.cache)
    if defect == "partial_hashes":
        cache["input_sha256"].pop(next(iter(cache["input_sha256"])))
    elif defect == "changed_hash":
        cache["input_sha256"][next(iter(cache["input_sha256"]))] = "incorrect"
    elif defect == "incomplete":
        cache["complete"] = False
    elif defect == "missing_report":
        cache["samples"].pop()
    elif defect == "duplicate_report":
        cache["samples"].append(deepcopy(cache["samples"][0]))
    elif defect == "swapped_raw":
        a, b = cache["samples"]
        a["raw_file"], b["raw_file"] = b["raw_file"], a["raw_file"]
    else:
        cache["stage"] = "QG imported reference overlay"
    with pytest.raises(ValueError, match=message):
        validation_case.run(cache)


def test_tentative_same_identity_does_not_pass_as_confirmed(validation_case):
    cache = deepcopy(validation_case.cache)
    for sample in cache["samples"]:
        sample["candidates"] = sample.pop("peaks")
        sample["peaks"] = []
        sample["candidates"][0]["significance"] = 1.5
    result = validation_case.run(cache)
    assert result["passed"] is False
    assert result["metrics"]["status_counts"] == {"tentative_same_id": 2}
    assert result["metrics"]["tentative_candidates_are_confirmed_detections"] is False
    assert not any(row["confirmed_detection"] for row in result["rows"])
    assert all(row["native_isotope"] is None for row in result["rows"])
    assert all(row["tentative_isotope"] == "Fe59" for row in result["rows"])


def test_wrong_tentative_identity_remains_missing(validation_case):
    cache = deepcopy(validation_case.cache)
    sample = cache["samples"][0]
    sample["candidates"] = sample.pop("peaks")
    sample["peaks"] = []
    sample["candidates"][0]["isotope"] = "Ta182"
    result = validation_case.run(cache)
    assert result["passed"] is False
    assert result["metrics"]["status_counts"]["missing_peak"] == 1


def test_unknown_correction_report_rejected(validation_case):
    validation_case.manifest.write_text(
        json.dumps({"corrections": [{"report": "absent.txt"}]}), encoding="utf8"
    )
    with pytest.raises(ValueError, match="unknown report"):
        validation_case.run()


@pytest.mark.parametrize("field", ["net", "sigma"])
@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf")])
def test_native_normalization_rejects_missing_negative_or_nonfinite_counts(
    validator, field, value
):
    row = {
        "energy_keV": 100,
        "channel": 200,
        "isotope": "Fe59",
        "net": 100,
        "sigma": 10,
    }
    row[field] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        validator.native_peaks([row])


def test_native_normalization_preserves_zero_uncertainty_without_inventing_floor(
    validator,
):
    row = {"energy_keV": 100, "channel": 200, "isotope": "Fe59", "net": 0, "sigma": 0}
    normalized = validator.native_peaks([row])[0]
    assert normalized.net_counts == normalized.net_counts_unc == 0
    assert "net_counts" not in row


@pytest.mark.parametrize("field", ["net_counts", "net_counts_unc"])
def test_explicit_invalid_canonical_count_cannot_be_hidden_by_valid_alias(
    validator, field
):
    row = {
        "energy_keV": 100,
        "channel": 200,
        "isotope": "Fe59",
        "net": 100,
        "sigma": 10,
    }
    row[field] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative"):
        validator.native_peaks([row])


def test_tentative_peak_cannot_reuse_confirmed_physical_channel(validation_case):
    cache = deepcopy(validation_case.cache)
    # Each report has only one reference here: a confirmed wrong identity still
    # reserves the observed physical channel, preventing its candidate alias
    # from being presented as a separate tentative recovery.
    sample = cache["samples"][0]
    sample["candidates"] = deepcopy(sample["peaks"])
    sample["peaks"][0]["isotope"] = "Ta182"
    result = validation_case.run(cache)
    row = result["rows"][0]
    assert row["status"] == "different_or_ambiguous_id"
    assert row["tentative_isotope"] is None
    assert row["confirmed_detection"] is False


def test_below_threshold_returned_native_fit_is_explicitly_tentative(validation_case):
    cache = deepcopy(validation_case.cache)
    for sample in cache["samples"]:
        sample["peaks"][0]["significance"] = 1.5
    result = validation_case.run(cache)
    assert result["passed"] is False
    assert result["metrics"]["status_counts"] == {"tentative_native_same_id": 2}
    assert result["metrics"]["matched_native_fits"] == 2
    assert result["metrics"]["confirmed_detections"] == 0
    assert result["metrics"]["paired_identity_labels_agree"] is True
    assert not any(row["confirmed_detection"] for row in result["rows"])
