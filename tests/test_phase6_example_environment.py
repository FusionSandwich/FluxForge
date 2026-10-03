import os
from types import SimpleNamespace
from fluxforge.workflows import phase6_ldrd_worked_example as example


def test_cli_pythonpath_uses_host_separator(monkeypatch):
    captured = {}
    monkeypatch.setenv("PYTHONPATH", "existing_modules")

    def run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(example.subprocess, "run", run)
    example._run_cli(["--help"])
    assert captured["env"]["PYTHONPATH"].split(os.pathsep) == [
        str(example.REPO_ROOT / "src"),
        "existing_modules",
    ]


def test_bundled_activity_review_does_not_claim_raw_qualification():
    payload = example._build_activity_review_payload("RAFM4-C_15dEOI")
    assert payload["accuracy_qualified"] is False
    assert (
        payload["example_basis"] == "historical_bundled_analysis_planning_demonstration"
    )
