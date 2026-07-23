"""Crash-safe atomic JSON persistence helpers.

The target file is never opened for writing.  JSON is serialized in memory, then
written and synced through a temporary file in the target directory before one
atomic replacement.  This keeps an existing document intact if serialization,
temporary-file writing, or replacement fails.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TextIO


class AtomicWriteError(OSError):
    """An I/O failure during an atomic persistence operation."""

    def __init__(
        self,
        path: Path,
        *,
        phase: str,
        cause: BaseException,
        replacement_completed: bool = False,
    ) -> None:
        self.path = path
        self.phase = phase
        self.replacement_completed = replacement_completed

        if phase == "create":
            guidance = (
                "Could not create a temporary file beside the target. Check that "
                "the folder exists and is writable."
            )
        elif phase == "write":
            guidance = (
                "Could not completely write and sync the temporary file. The "
                "original file remains unchanged."
            )
        elif phase == "replace":
            guidance = (
                "Could not atomically replace the target; the original file "
                "remains unchanged. Close applications that may have the file "
                "open and allow OneDrive or other sync software to release any "
                "Windows file lock before retrying."
            )
        elif phase == "directory-sync":
            guidance = (
                "The target was replaced, but its parent directory could not be "
                "synced. The new file is visible, although crash durability could "
                "not be confirmed."
            )
        else:
            guidance = "Atomic persistence failed."

        super().__init__(f"{guidance} Target: {path}. Cause: {cause}")


def _write_serialized(stream: TextIO, serialized: str) -> None:
    """Write and durably flush serialized JSON to an open temporary stream."""

    stream.write(serialized)
    stream.flush()
    os.fsync(stream.fileno())


def _fsync_parent_directory(directory: Path) -> None:
    """Persist replacement metadata on POSIX filesystems."""

    if os.name != "posix":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> Path:
    """Serialize *payload* and atomically replace the JSON file at *path*.

    Serialization happens before any filesystem operation. Non-finite floats are
    rejected so a successful write is always standards-compliant JSON. The
    destination directory must already exist.

    Returns
    -------
    pathlib.Path
        The destination path after a successful, durable replacement.
    """

    target = Path(path)
    serialized = json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=False,
        allow_nan=False,
    )
    if not serialized.endswith("\n"):
        serialized += "\n"

    directory = target.parent
    temporary: Path | None = None

    try:
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=directory,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
        except OSError as exc:
            raise AtomicWriteError(target, phase="create", cause=exc) from exc

        try:
            try:
                stream = os.fdopen(
                    file_descriptor,
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                )
            except Exception:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
                raise
            with stream:
                _write_serialized(stream, serialized)
        except Exception as exc:
            raise AtomicWriteError(target, phase="write", cause=exc) from exc

        try:
            os.replace(temporary, target)
        except OSError as exc:
            raise AtomicWriteError(target, phase="replace", cause=exc) from exc

        temporary = None
        try:
            _fsync_parent_directory(directory)
        except OSError as exc:
            raise AtomicWriteError(
                target,
                phase="directory-sync",
                cause=exc,
                replacement_completed=True,
            ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Preserve the primary error. Cleanup is best-effort when an
                # external Windows/OneDrive lock also prevents temp deletion.
                pass

    return target


__all__ = ["AtomicWriteError", "atomic_write_json"]
