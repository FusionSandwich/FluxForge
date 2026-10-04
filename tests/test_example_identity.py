from __future__ import annotations

import subprocess

import pytest

from fluxforge.validation import example_identity


def test_no_git_checkout_reports_content_and_unknown_revision(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "method.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(example_identity.shutil, "which", lambda _name: None)

    identity = example_identity.source_identity(tmp_path, ["src/method.py"])

    assert identity["files_sha256"]["src/method.py"]
    assert identity["revision"] is None
    assert identity["revision_status"] == "unknown_no_git_checkout"
    assert identity["ancestor_verified"] is None
    assert identity["ancestor_status"] == "unknown_no_git_checkout"


def test_expected_canonical_text_hash_mismatch_is_rejected(tmp_path):
    (tmp_path / "module.py").write_bytes(b"value = 1\r\n")

    with pytest.raises(ValueError, match="canonical-LF SHA-256 mismatch"):
        example_identity.source_identity(
            tmp_path,
            ["module.py"],
            expected_sha256={"module.py": "0" * 64},
            canonical_lf=True,
        )


def test_missing_declared_source_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Declared source file is missing"):
        example_identity.source_identity(tmp_path, ["missing.py"])


def _commit(repository, filename: str) -> str:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    (repository / filename).write_text("identity fixture\n", encoding="utf-8")
    git("add", filename)
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return git("rev-parse", "HEAD")


def test_unrelated_required_ancestor_is_rejected_in_git_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "source.py").write_text("current\n", encoding="utf-8")
    first_commit = _commit(checkout, "source.py")
    subprocess.run(
        ["git", "-C", str(checkout), "switch", "-c", "unrelated"],
        check=True,
        capture_output=True,
        text=True,
    )
    absent_ancestor = _commit(checkout, "other.py")
    subprocess.run(
        ["git", "-C", str(checkout), "switch", "--detach", first_commit],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match="not an ancestor"):
        example_identity.source_identity(
            checkout, ["source.py"], required_ancestor=absent_ancestor
        )


def _binding_fixture(tmp_path):
    import json
    import hashlib

    source = tmp_path / "method.py"
    source.write_bytes(b"value = 1\r\n")
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "profile": "test_epoch",
                "source_revision": "1" * 40,
                "hash_basis": "canonical_lf_sha256",
                "files_sha256": {
                    "method.py": hashlib.sha256(b"value = 1\n").hexdigest()
                },
            }
        )
    )
    return source, binding


def test_published_binding_accepts_portable_crlf_and_retains_epoch(tmp_path):
    source, binding = _binding_fixture(tmp_path)
    result = example_identity.bound_example_engine(tmp_path, binding, [source.name])
    assert result["revision"] is None
    assert result["ancestor_verified"] is None
    assert result["profile"] == "test_epoch"
    assert result["declared_source_revision"] == "1" * 40
    assert result["binding_sha256"]


def test_published_binding_rejects_modified_source(tmp_path):
    source, binding = _binding_fixture(tmp_path)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        example_identity.bound_example_engine(tmp_path, binding, [source.name])


def test_published_binding_rejects_uncovered_file(tmp_path):
    _, binding = _binding_fixture(tmp_path)
    with pytest.raises(ValueError, match="cover all"):
        example_identity.bound_example_engine(tmp_path, binding, ["unbound.py"])


def test_published_binding_checks_other_declared_files(tmp_path):
    import json

    source, binding = _binding_fixture(tmp_path)
    payload = json.loads(binding.read_text())
    payload["files_sha256"]["unrequested.py"] = "0" * 64
    binding.write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError, match="unrequested.py"):
        example_identity.bound_example_engine(tmp_path, binding, [source.name])


def test_current_examples_keep_historical_source_rejection():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for relative, function in [
        ("examples/qg_protocol/run_example.py", "build_evidence"),
        ("examples/RAFM_irradiation/joint_poisson_pilot.py", "run_pilot"),
    ]:
        spec = importlib.util.spec_from_file_location("profile_test", root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            getattr(module, function)(engine_profile="historical")
        with pytest.raises(ValueError, match="Unknown engine profile"):
            getattr(module, function)(engine_profile="unknown")


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "g" * 64])
def test_published_binding_cannot_disable_source_pin(tmp_path, digest):
    import json

    source, binding = _binding_fixture(tmp_path)
    payload = json.loads(binding.read_text())
    payload["files_sha256"][source.name] = digest
    binding.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit SHA-256 pins"):
        example_identity.bound_example_engine(tmp_path, binding, [source.name])


def test_binding_cannot_relabel_changed_code_as_old_git_revision(tmp_path):
    import json
    import hashlib

    revision = _commit(tmp_path, "method.py")
    source = tmp_path / "method.py"
    source.write_bytes(b"modified engine\n")
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "profile": "test_epoch",
                "source_revision": revision,
                "hash_basis": "canonical_lf_sha256",
                "files_sha256": {
                    source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
        )
    )
    with pytest.raises(ValueError, match="differs from declared Git revision"):
        example_identity.bound_example_engine(tmp_path, binding, [source.name])
