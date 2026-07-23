from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fluxforge.io import atomic


def _temporary_files(directory: Path, target_name: str) -> list[Path]:
    return list(directory.glob(f".{target_name}.*.tmp"))


def test_atomic_write_json_replaces_once_with_valid_utf8_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "workspace.json"
    target.write_bytes(b'{"old": true}\n')
    replacements: list[tuple[Path, Path]] = []
    real_replace = atomic.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", recording_replace)

    result = atomic.atomic_write_json(
        target,
        {"name": "UWNR γ analysis", "values": [1, 2, 3]},
        sort_keys=True,
    )

    assert result == target
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "name": "UWNR γ analysis",
        "values": [1, 2, 3],
    }
    assert len(replacements) == 1
    assert replacements[0][0].parent == target.parent
    assert replacements[0][1] == target
    assert not _temporary_files(tmp_path, target.name)


def test_serialization_failure_occurs_before_any_filesystem_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "workspace.json"
    original = b"original bytes\r\n"
    target.write_bytes(original)
    temporary_creation_attempted = False

    def unexpected_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal temporary_creation_attempted
        temporary_creation_attempted = True
        raise AssertionError("serialization must precede temporary-file creation")

    monkeypatch.setattr(atomic.tempfile, "mkstemp", unexpected_mkstemp)

    with pytest.raises(ValueError, match="Out of range float values"):
        atomic.atomic_write_json(target, {"not_json": float("nan")})

    assert not temporary_creation_attempted
    assert target.read_bytes() == original


def test_injected_write_failure_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "workspace.json"
    original = b"existing workspace bytes\n"
    target.write_bytes(original)

    def fail_write(stream: object, serialized: str) -> None:
        stream.write(serialized[:5])
        raise OSError("simulated disk-full write failure")

    monkeypatch.setattr(atomic, "_write_serialized", fail_write)

    with pytest.raises(atomic.AtomicWriteError) as error:
        atomic.atomic_write_json(target, {"replacement": True})

    assert error.value.phase == "write"
    assert error.value.replacement_completed is False
    assert "original file remains unchanged" in str(error.value)
    assert target.read_bytes() == original
    assert not _temporary_files(tmp_path, target.name)


def test_injected_replace_failure_is_clear_and_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "workspace.json"
    original = b"byte-identical original\x00content"
    target.write_bytes(original)

    def fail_replace(source: Path, destination: Path) -> None:
        raise PermissionError("simulated sharing violation")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)

    with pytest.raises(atomic.AtomicWriteError) as error:
        atomic.atomic_write_json(target, {"replacement": True})

    assert error.value.phase == "replace"
    assert error.value.replacement_completed is False
    assert "original file remains unchanged" in str(error.value)
    assert "OneDrive" in str(error.value)
    assert "Windows file lock" in str(error.value)
    assert target.read_bytes() == original
    assert not _temporary_files(tmp_path, target.name)


@pytest.mark.skipif(os.name != "posix", reason="parent fsync is POSIX-specific")
def test_parent_directory_sync_failure_reports_that_replace_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "workspace.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_directory_sync(directory: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(atomic, "_fsync_parent_directory", fail_directory_sync)

    with pytest.raises(atomic.AtomicWriteError) as error:
        atomic.atomic_write_json(target, {"new": True})

    assert error.value.phase == "directory-sync"
    assert error.value.replacement_completed is True
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    assert not _temporary_files(tmp_path, target.name)
