"""Versioned FluxForge ``.ffs`` session persistence and v1 migration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from fluxforge.core.workspace_document import (
    SpectrumRoleAssignment,
    WorkspaceDocument,
    WorkspaceSpectrum,
)
from fluxforge.hal import DeviceRegistry
from fluxforge.io.atomic import atomic_write_json
from fluxforge.io.spe import GammaSpectrum


SESSION_FORMAT = "fluxforge_session"
SESSION_FORMAT_VERSION = 2

_V1_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "created_at",
        "active_spectrum_index",
        "source_files",
        "recent_files",
        "device_snapshot",
        "metadata",
        "spectra",
    }
)
_V2_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "document",
        "recent_files",
        "device_snapshot",
        "metadata",
        "created_at",
    }
)


class SessionFormatError(ValueError):
    """A malformed or unsupported FluxForge session envelope."""


class FluxForgeSession:
    """Canonical v2 session with legacy attribute compatibility.

    ``document`` is the persisted scientific state.  The ``spectra``,
    ``active_spectrum_index`` and ``source_files`` properties retain the v1 API
    used by existing GUI code while new callers can work directly with the
    :class:`~fluxforge.core.workspace_document.WorkspaceDocument`.
    """

    def __init__(
        self,
        spectra: Iterable[GammaSpectrum] = (),
        active_spectrum_index: int = 0,
        source_files: Iterable[str | Path] = (),
        recent_files: Iterable[str | Path] = (),
        device_snapshot: Iterable[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        *,
        document: WorkspaceDocument | None = None,
    ) -> None:
        if document is None:
            document = _document_from_spectra(
                spectra,
                active_spectrum_index=active_spectrum_index,
                source_files=source_files,
                document_id="workspace",
            )
        elif list(spectra) or list(source_files) or active_spectrum_index != 0:
            raise ValueError(
                "Pass either document or legacy spectra/source/index fields, not both."
            )
        document.validate()
        self.document = document
        self.recent_files = [str(Path(value)) for value in recent_files]
        self.device_snapshot = [deepcopy(dict(item)) for item in device_snapshot]
        self.metadata = deepcopy(dict(metadata or {}))
        self.created_at = created_at if created_at is not None else _utc_timestamp()

    @property
    def spectra(self) -> list[GammaSpectrum]:
        """Return spectra in their persisted document order (v1 facade)."""

        return [item.spectrum for item in self.document.spectra]

    @property
    def active_spectrum_index(self) -> int:
        """Return the active spectrum's document index (v1 facade)."""

        active_id = self.document.active_spectrum_id
        if active_id is None:
            return 0
        for index, item in enumerate(self.document.spectra):
            if item.spectrum_id == active_id:
                return index
        raise SessionFormatError(
            f"Active spectrum {active_id!r} is not present in the workspace."
        )

    @property
    def source_files(self) -> list[str]:
        """Return one source path per spectrum where available (v1 facade)."""

        paths = [str(item.source_path or "") for item in self.document.spectra]
        for extension_key in (
            "legacy_unmatched_source_files",
            "unmatched_source_files",
        ):
            unmatched = self.document.extensions.get(extension_key, [])
            if isinstance(unmatched, (list, tuple)):
                paths.extend(str(value) for value in unmatched)
        return paths

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical v2 JSON envelope."""

        self.document.validate()
        _require_string(self.created_at, "created_at", allow_empty=True)
        return {
            "format": SESSION_FORMAT,
            "format_version": SESSION_FORMAT_VERSION,
            "document": self.document.to_dict(),
            "recent_files": list(self.recent_files),
            "device_snapshot": deepcopy(self.device_snapshot),
            "metadata": deepcopy(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FluxForgeSession":
        """Build a session from a v1, unversioned, or v2 payload."""

        migrated = migrate_session_payload(payload)
        return cls(
            document=WorkspaceDocument.from_dict(migrated["document"]),
            recent_files=migrated["recent_files"],
            device_snapshot=migrated["device_snapshot"],
            metadata=migrated["metadata"],
            created_at=migrated["created_at"],
        )


def migrate_session_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Purely migrate a legacy session payload to the canonical v2 envelope.

    The input mapping is never mutated. Calling this function again on its v2
    result produces an equal value. Unknown legacy top-level data and source
    paths that cannot be matched by index are retained in document extensions.
    """

    if not isinstance(payload, Mapping):
        raise SessionFormatError("FluxForge session payload must be a JSON object.")
    source = deepcopy(dict(payload))
    declared_format = source.get("format")
    if declared_format not in (None, SESSION_FORMAT):
        raise SessionFormatError(
            f"Unsupported session format {declared_format!r}; "
            f"expected {SESSION_FORMAT!r}."
        )

    version = source.get("format_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise SessionFormatError("Session format_version must be an integer.")
    if version > SESSION_FORMAT_VERSION:
        raise SessionFormatError(
            f"Session format version {version} is newer than supported version "
            f"{SESSION_FORMAT_VERSION}."
        )
    if version < 1:
        raise SessionFormatError(f"Unsupported session format version {version}.")
    if version == SESSION_FORMAT_VERSION:
        return _canonicalize_v2(source)
    return _migrate_v1(source)


def session_from_spectra(
    spectra: Iterable[GammaSpectrum],
    *,
    active_spectrum_index: int = 0,
    source_files: Iterable[str | Path] = (),
    recent_files: Iterable[str | Path] = (),
    device_registry: DeviceRegistry | None = None,
    metadata: dict[str, Any] | None = None,
) -> FluxForgeSession:
    """Create a new canonical session from spectra and workflow metadata."""

    return FluxForgeSession(
        spectra=list(spectra),
        active_spectrum_index=active_spectrum_index,
        source_files=source_files,
        recent_files=recent_files,
        device_snapshot=(
            device_registry.snapshot() if device_registry is not None else []
        ),
        metadata=metadata,
    )


def write_ffs_session(path: str | Path, session: FluxForgeSession) -> dict[str, Any]:
    """Atomically write a canonical v2 ``.ffs`` session file."""

    target = Path(path)
    if target.suffix.lower() != ".ffs":
        raise ValueError("FluxForge sessions must use the `.ffs` extension.")
    if not isinstance(session, FluxForgeSession):
        raise TypeError("session must be a FluxForgeSession instance.")
    payload = session.to_dict()
    atomic_write_json(target, payload)
    return payload


def read_ffs_session(path: str | Path) -> FluxForgeSession:
    """Read and validate a v1, unversioned, or v2 ``.ffs`` session file."""

    import json

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionFormatError(
            f"Could not read FluxForge session {source}: {exc}"
        ) from exc
    try:
        return FluxForgeSession.from_dict(payload)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SessionFormatError):
            raise
        raise SessionFormatError(f"Invalid FluxForge session {source}: {exc}") from exc


def _canonicalize_v2(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload).difference(_V2_FIELDS))
    if unknown:
        raise SessionFormatError(
            "Version 2 session contains unknown top-level fields: " + ", ".join(unknown)
        )
    if payload.get("format") != SESSION_FORMAT:
        raise SessionFormatError(
            f"Version 2 session format must be {SESSION_FORMAT!r}."
        )
    if not isinstance(payload.get("document"), Mapping):
        raise SessionFormatError("Version 2 session document must be an object.")
    try:
        document = WorkspaceDocument.from_dict(payload["document"])
        document.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionFormatError(
            f"Version 2 session document is invalid: {exc}"
        ) from exc
    recent_files = _string_list(payload.get("recent_files"), "recent_files")
    device_snapshot = _mapping_list(payload.get("device_snapshot"), "device_snapshot")
    metadata = _mapping(payload.get("metadata"), "metadata")
    created_at = _require_string(
        payload.get("created_at"), "created_at", allow_empty=True
    )
    return {
        "format": SESSION_FORMAT,
        "format_version": SESSION_FORMAT_VERSION,
        "document": document.to_dict(),
        "recent_files": recent_files,
        "device_snapshot": device_snapshot,
        "metadata": metadata,
        "created_at": created_at,
    }


def _migrate_v1(payload: dict[str, Any]) -> dict[str, Any]:
    if "spectra" not in payload:
        raise SessionFormatError(
            "Legacy session payload is missing the required spectra array."
        )
    raw_spectra = payload.get("spectra", [])
    if not isinstance(raw_spectra, list):
        raise SessionFormatError("Legacy session spectra must be an array.")
    source_files = _string_list(payload.get("source_files", []), "source_files")
    active_index = _active_index(
        payload.get("active_spectrum_index", 0), len(raw_spectra)
    )

    workspace_spectra: list[WorkspaceSpectrum] = []
    for index, raw_spectrum in enumerate(raw_spectra):
        if not isinstance(raw_spectrum, Mapping):
            raise SessionFormatError(
                f"Legacy spectrum at index {index} must be an object."
            )
        try:
            spectrum = GammaSpectrum.from_dict(dict(raw_spectrum))
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionFormatError(
                f"Legacy spectrum at index {index} is invalid: {exc}"
            ) from exc
        spectrum_id = f"legacy-spectrum-{index + 1}"
        source_path = source_files[index] if index < len(source_files) else None
        label = spectrum.spectrum_id or (
            Path(source_path).name if source_path else f"Spectrum {index + 1}"
        )
        workspace_spectra.append(
            WorkspaceSpectrum(
                spectrum_id=spectrum_id,
                spectrum=spectrum,
                label=label,
                source_path=source_path or None,
                provenance={"migrated_from_session_version": 1},
            )
        )

    unknown = {
        str(key): deepcopy(value)
        for key, value in payload.items()
        if key not in _V1_FIELDS
    }
    extensions: dict[str, Any] = {}
    if unknown:
        extensions["legacy_unknown_top_level"] = unknown
    if len(source_files) > len(workspace_spectra):
        extensions["legacy_unmatched_source_files"] = source_files[
            len(workspace_spectra) :
        ]

    active_id = (
        workspace_spectra[active_index].spectrum_id if workspace_spectra else None
    )
    roles = (
        (SpectrumRoleAssignment(role="foreground", spectrum_ids=(active_id,)),)
        if active_id is not None
        else ()
    )
    legacy_created_at = _require_string(
        payload.get("created_at", ""), "created_at", allow_empty=True
    )
    document_timestamp = legacy_created_at or "1970-01-01T00:00:00+00:00"
    document = WorkspaceDocument(
        document_id="legacy-session",
        title=str(_mapping(payload.get("metadata", {}), "metadata").get("title", "")),
        created_at=document_timestamp,
        updated_at=document_timestamp,
        spectra=tuple(workspace_spectra),
        spectrum_roles=roles,
        active_spectrum_id=active_id,
        provenance={"migrated_from_session_version": 1},
        extensions=extensions,
    )
    document.validate()
    return {
        "format": SESSION_FORMAT,
        "format_version": SESSION_FORMAT_VERSION,
        "document": document.to_dict(),
        "recent_files": _string_list(payload.get("recent_files", []), "recent_files"),
        "device_snapshot": _mapping_list(
            payload.get("device_snapshot", []), "device_snapshot"
        ),
        "metadata": _mapping(payload.get("metadata", {}), "metadata"),
        "created_at": legacy_created_at,
    }


def _document_from_spectra(
    spectra: Iterable[GammaSpectrum],
    *,
    active_spectrum_index: int,
    source_files: Iterable[str | Path],
    document_id: str,
) -> WorkspaceDocument:
    spectrum_values = list(spectra)
    path_values = [str(Path(value)) for value in source_files]
    index = _active_index(active_spectrum_index, len(spectrum_values))
    records = tuple(
        WorkspaceSpectrum(
            spectrum_id=f"spectrum-{position + 1}",
            spectrum=spectrum,
            label=spectrum.spectrum_id
            or (Path(path_values[position]).name if position < len(path_values) else "")
            or f"Spectrum {position + 1}",
            source_path=(
                path_values[position]
                if position < len(path_values) and path_values[position]
                else None
            ),
        )
        for position, spectrum in enumerate(spectrum_values)
    )
    active_id = records[index].spectrum_id if records else None
    roles = (
        (SpectrumRoleAssignment(role="foreground", spectrum_ids=(active_id,)),)
        if active_id is not None
        else ()
    )
    extensions: dict[str, Any] = {}
    if len(path_values) > len(records):
        extensions["unmatched_source_files"] = path_values[len(records) :]
    return WorkspaceDocument(
        document_id=document_id,
        spectra=records,
        spectrum_roles=roles,
        active_spectrum_id=active_id,
        extensions=extensions,
    )


def _active_index(value: Any, spectrum_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionFormatError("active_spectrum_index must be an integer.")
    if spectrum_count == 0:
        if value != 0:
            raise SessionFormatError(
                "An empty session must use active_spectrum_index 0."
            )
        return 0
    if value < 0 or value >= spectrum_count:
        raise SessionFormatError(
            f"active_spectrum_index {value} is outside 0..{spectrum_count - 1}."
        )
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"{name} must be an object.")
    return deepcopy(dict(value))


def _mapping_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SessionFormatError(f"{name} must be an array.")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SessionFormatError(f"{name}[{index}] must be an object.")
        result.append(deepcopy(dict(item)))
    return result


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise SessionFormatError(f"{name} must be an array.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise SessionFormatError(f"{name}[{index}] must be a string.")
        result.append(item)
    return result


def _require_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SessionFormatError(f"{name} must be a string.")
    if not allow_empty and not value:
        raise SessionFormatError(f"{name} must not be empty.")
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "FluxForgeSession",
    "SESSION_FORMAT",
    "SESSION_FORMAT_VERSION",
    "SessionFormatError",
    "migrate_session_payload",
    "read_ffs_session",
    "session_from_spectra",
    "write_ffs_session",
]
