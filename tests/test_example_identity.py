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
