"""Content and optional Git identity for portable method examples."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


def _git_marker(root: Path) -> bool:
    return (root / ".git").exists()


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Git identity verification failed: git {' '.join(args)}"
        ) from exc
    return result.stdout.strip()


def source_identity(
    root: str | Path,
    files: Iterable[str],
    *,
    required_ancestor: str | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    canonical_lf: bool = False,
) -> dict:
    """Hash declared files and report Git facts only when actually verified.

    ``canonical_lf`` hashes UTF-8/text source after CRLF-to-LF normalization.
    Expected hashes are checked in the selected byte mode in every environment.
    """
    base = Path(root).resolve()
    hashes: dict[str, str] = {}
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Declared source path must be relative to root: {name}")
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(f"Declared source file is missing: {name}")
        content = path.read_bytes()
        if canonical_lf:
            content = content.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content).hexdigest()
        hashes[relative.as_posix()] = digest
        expected = (expected_sha256 or {}).get(relative.as_posix())
        if expected is not None and digest != expected:
            mode = "canonical-LF " if canonical_lf else ""
            raise ValueError(f"{mode}SHA-256 mismatch: {name}")

    git_marker = _git_marker(base)
    git_executable = shutil.which("git")
    if not git_marker:
        revision = None
        ancestor_verified = None
        revision_status = "unknown_no_git_checkout"
        ancestor_status = "unknown_no_git_checkout"
    else:
        if not git_executable:
            raise RuntimeError("Git checkout detected but git executable is unavailable")
        revision = _git(base, "rev-parse", "HEAD")
        revision_status = "verified"
        if required_ancestor is None:
            ancestor_verified = None
            ancestor_status = "not_requested"
        else:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", required_ancestor, "HEAD"],
                cwd=base,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                ancestor_verified = True
                ancestor_status = "verified"
            elif result.returncode == 1:
                raise ValueError(
                    f"Required Git ancestor is not an ancestor: {required_ancestor}"
                )
            else:
                raise RuntimeError(
                    f"Git ancestry verification failed for {required_ancestor}: "
                    f"{result.stderr.strip()}"
                )
    return {
        "files_sha256": hashes,
        "hash_basis": "canonical_lf_sha256" if canonical_lf else "byte_sha256",
        "revision": revision,
        "revision_status": revision_status,
        "required_ancestor": required_ancestor,
        "ancestor_verified": ancestor_verified,
        "ancestor_status": ancestor_status,
    }
