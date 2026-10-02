"""The all-raw audit must use actual wire pairing keys, including withheld QG."""

import importlib.util
import copy
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "raw_audit", Path(__file__).parents[1] / "tools/audit_rafm_raw_recovery.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("qg", [None, Path("QG.txt")])
@pytest.mark.parametrize("group", ["flux_wires", "RAFM3"])
def test_raw_replay_dispatch_retains_pairing_identity(monkeypatch, qg, group):
    calls = []
    monkeypatch.setattr(
        audit.workflow,
        "analyze_flux_wire_sample",
        lambda *args: calls.append(("wire", args)),
    )
    monkeypatch.setattr(
        audit.workflow,
        "analyze_generic_sample",
        lambda *args: calls.append(("generic", args)),
    )
    raw = Path(group) / "Co-Cd.ASC"
    audit.replay_sample(
        raw,
        qg,
        "metadata",
        "paths",
        "tree",
        "background",
        "library",
        "half_lives",
        "co-cd-pairing-key",
    )
    kind, args = calls[0]
    if group == "flux_wires":
        assert kind == "wire"
        assert args[-2:] == (qg, "co-cd-pairing-key")
    else:
        assert kind == "generic"
        assert args[-1] == qg


@pytest.fixture
def checkpoint(tmp_path):
    raw = tmp_path / "sample.ASC"
    raw.write_text("counts")
    artifact = {
        "peaks": [],
        "isotopes": {},
        "targeted_fit_diagnostics": [],
        "validation": {"passed": None, "reference_used_for_analysis": False},
        "measurement_time_audit": {"accuracy_qualified": False},
        "n_detected_peaks": 0,
        "n_unidentified_peaks": 0,
        "n_ambiguous_peaks": 0,
        "comparison_report_txt": "report.txt",
    }
    for directory in ("raw_iec", "withheld_report"):
        path = tmp_path / directory / "analysis_json/sample.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(artifact))
    current = {
        "source_sha256": {
            "science.py": "science-hash",
            "tools/audit_rafm_raw_recovery.py": "new-driver",
        },
        "metadata_sha256": {"config": "same"},
        "background_sha256": "same",
        "effective_configuration": {"method": "iec_tiered"},
    }
    previous = copy.deepcopy(current)
    previous["source_sha256"]["tools/audit_rafm_raw_recovery.py"] = "reviewed-driver"
    previous["samples"] = [
        {
            "sample": "sample",
            "raw_sha256": audit.digest(raw),
            "qg_sha256": None,
            "artifact": str(tmp_path / "raw_iec/analysis_json/sample.json"),
            "prediction_sha256": audit.prediction_digest(artifact),
            "validation": artifact["validation"],
            "measurement_time_audit": artifact["measurement_time_audit"],
            "withheld_report_prediction_identical": True,
            "n_detected_peaks": 0,
            "n_unidentified_peaks": 0,
            "n_ambiguous_peaks": 0,
            "isotopes": [],
            "targeted_fit_diagnostics": [],
            "report": "report.txt",
        }
    ]
    return previous, current, [(raw, None)], tmp_path, "reviewed-driver"


def test_verified_checkpoint_can_resume(checkpoint):
    assert len(audit.validate_checkpoint(*checkpoint)) == 1


@pytest.mark.parametrize(
    "changed", ["science", "configuration", "raw", "prediction", "driver", "withheld"]
)
def test_changed_checkpoint_cannot_resume(checkpoint, changed):
    previous, current, selected, output, driver = checkpoint
    if changed == "science":
        current["source_sha256"]["science.py"] = "different"
    elif changed == "configuration":
        current["effective_configuration"]["method"] = "qg"
    elif changed == "raw":
        selected[0][0].write_text("different counts")
    elif changed == "prediction":
        previous["samples"][0]["prediction_sha256"] = "different"
    elif changed == "driver":
        driver = "unreviewed-driver"
    else:
        path = output / "withheld_report/analysis_json/sample.json"
        data = json.loads(path.read_text())
        data["peaks"] = ["copied report peak"]
        path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        audit.validate_checkpoint(previous, current, selected, output, driver)


@pytest.mark.parametrize(
    "field",
    [
        "n_detected_peaks",
        "n_unidentified_peaks",
        "n_ambiguous_peaks",
        "isotopes",
        "targeted_fit_diagnostics",
        "report",
    ],
)
def test_cached_receipt_fields_must_agree_with_artifact(checkpoint, field):
    previous, current, selected, output, driver = checkpoint
    previous["samples"][0][field] = "stale-field"
    with pytest.raises(AssertionError):
        audit.validate_checkpoint(previous, current, selected, output, driver)
