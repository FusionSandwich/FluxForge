"""The all-raw audit must use actual wire pairing keys, including withheld QG."""

import importlib.util
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
