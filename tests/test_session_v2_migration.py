from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.core.workspace_document import WorkspaceDocument
from fluxforge.io import session as session_module
from fluxforge.io.session import (
    SESSION_FORMAT_VERSION,
    FluxForgeSession,
    SessionFormatError,
    migrate_session_payload,
    read_ffs_session,
    session_from_spectra,
    write_ffs_session,
)
from fluxforge.io.spe import GammaSpectrum


def _spectrum_payload(name: str, offset: float = 0.0) -> dict[str, object]:
    return GammaSpectrum(
        counts=np.array([2.0 + offset, 5.0 + offset, 9.0 + offset]),
        live_time=120.0,
        real_time=125.0,
        spectrum_id=name,
        detector_id="uwrn-hpge",
        calibration={
            "energy": [0.25, 0.5],
            "fwhm": [0.8, 0.04],
            "efficiency": {"model": "log-polynomial", "coefficients": [-2.0]},
        },
        source_type="hal",
        device_id="mca-1",
        device_label="UWNR MCA",
        gps={"latitude": 43.072, "longitude": -89.412},
        metadata={"sample": "RAFM"},
    ).to_dict()


def _legacy_payload(*, versioned: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "fluxforge_session",
        "created_at": "2026-07-22T12:34:56+00:00",
        "active_spectrum_index": 1,
        "source_files": ["raw/first.ASC", "raw/second.ASC", "raw/unmatched.ASC"],
        "recent_files": ["recent/a.n42"],
        "device_snapshot": [
            {
                "source_type": "hal",
                "device_id": "mca-1",
                "state": "ready",
                "telemetry": {"temperature_c": 19.5},
            }
        ],
        "metadata": {"campaign": "UWNR", "title": "RAFM activation"},
        "spectra": [_spectrum_payload("first"), _spectrum_payload("second", 1.0)],
        "site_extension": {"operator": "test-user"},
    }
    if versioned:
        payload["format_version"] = 1
    return payload


@pytest.mark.parametrize("versioned", [False, True])
def test_v1_and_unversioned_payloads_migrate_without_data_loss(
    versioned: bool,
) -> None:
    legacy = _legacy_payload(versioned=versioned)
    original = deepcopy(legacy)

    migrated = migrate_session_payload(legacy)

    assert legacy == original
    assert migrated["format"] == "fluxforge_session"
    assert migrated["format_version"] == SESSION_FORMAT_VERSION
    assert migrated["created_at"] == legacy["created_at"]
    assert migrated["recent_files"] == legacy["recent_files"]
    assert migrated["device_snapshot"] == legacy["device_snapshot"]
    assert migrated["metadata"] == legacy["metadata"]

    document = WorkspaceDocument.from_dict(migrated["document"])
    assert [item.spectrum_id for item in document.spectra] == [
        "legacy-spectrum-1",
        "legacy-spectrum-2",
    ]
    assert document.active_spectrum_id == "legacy-spectrum-2"
    assert document.spectrum_roles[0].role == "foreground"
    assert document.spectrum_roles[0].spectrum_ids == ("legacy-spectrum-2",)
    assert [item.source_path for item in document.spectra] == [
        "raw/first.ASC",
        "raw/second.ASC",
    ]
    restored = document.spectra[1].spectrum
    assert restored.calibration == legacy["spectra"][1]["calibration"]
    assert restored.gps == legacy["spectra"][1]["gps"]
    assert restored.source_type == "hal"
    assert restored.device_id == "mca-1"
    assert document.extensions["legacy_unknown_top_level"] == {
        "site_extension": {"operator": "test-user"}
    }
    assert tuple(document.extensions["legacy_unmatched_source_files"]) == (
        "raw/unmatched.ASC",
    )


def test_migration_is_idempotent_and_v2_is_canonical() -> None:
    legacy = _legacy_payload()
    migrated = migrate_session_payload(legacy)

    assert migrate_session_payload(legacy) == migrated
    assert migrate_session_payload(migrated) == migrated
    session = FluxForgeSession.from_dict(migrated)
    assert session.source_files == [
        "raw/first.ASC",
        "raw/second.ASC",
        "raw/unmatched.ASC",
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"format": "other", "format_version": 1}, "Unsupported session format"),
        ({"format": "fluxforge_session", "format_version": "2"}, "integer"),
        ({"format": "fluxforge_session", "format_version": 99}, "newer"),
        ({"format": "fluxforge_session", "format_version": 0}, "Unsupported"),
        ({}, "missing the required spectra"),
        (
            {"format": "fluxforge_session", "format_version": 1},
            "missing the required spectra",
        ),
        (
            {
                "format": "fluxforge_session",
                "format_version": 2,
                "document": {},
                "recent_files": [],
                "device_snapshot": [],
                "metadata": {},
                "created_at": "",
                "unexpected": True,
            },
            "unknown top-level",
        ),
    ],
)
def test_malformed_or_future_versions_fail_explicitly(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(SessionFormatError, match=message):
        migrate_session_payload(payload)


@pytest.mark.parametrize("index", [-1, 2, 7])
def test_invalid_active_index_fails_for_nonempty_legacy_session(index: int) -> None:
    payload = _legacy_payload()
    payload["active_spectrum_index"] = index

    with pytest.raises(SessionFormatError, match="active_spectrum_index"):
        migrate_session_payload(payload)


def test_only_index_zero_is_valid_for_empty_legacy_session() -> None:
    empty = {
        "format": "fluxforge_session",
        "format_version": 1,
        "spectra": [],
        "active_spectrum_index": 0,
    }
    assert migrate_session_payload(empty)["document"]["active_spectrum_id"] is None

    empty["active_spectrum_index"] = 1
    with pytest.raises(SessionFormatError, match="empty session"):
        migrate_session_payload(empty)


def test_new_session_facade_writes_only_v2_and_round_trips(tmp_path: Path) -> None:
    spectrum = GammaSpectrum(
        counts=np.array([4.0, 8.0, 15.0]),
        calibration={"energy": [0.0, 0.5]},
        spectrum_id="sample",
        source_type="hal",
        device_id="mock-mca",
        gps={"latitude": 43.07, "longitude": -89.4},
    )
    session = session_from_spectra(
        [spectrum],
        source_files=["sample.ASC"],
        recent_files=["recent.ASC"],
        metadata={"campaign": "UWNR"},
    )

    assert session.spectra == [spectrum]
    assert session.active_spectrum_index == 0
    assert session.source_files == ["sample.ASC"]

    target = tmp_path / "analysis.ffs"
    payload = write_ffs_session(target, session)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert payload == on_disk
    assert set(on_disk) == {
        "format",
        "format_version",
        "document",
        "recent_files",
        "device_snapshot",
        "metadata",
        "created_at",
    }
    assert "active_spectrum_index" not in on_disk
    assert "spectra" not in on_disk

    restored = read_ffs_session(target)
    assert restored.document.to_dict() == session.document.to_dict()
    assert restored.spectra[0].gps["latitude"] == 43.07
    assert restored.source_files == ["sample.ASC"]
    assert restored.recent_files == ["recent.ASC"]


def test_write_uses_atomic_json_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def record_atomic(path: Path, payload: dict[str, object]) -> Path:
        calls.append((Path(path), deepcopy(payload)))
        return Path(path)

    monkeypatch.setattr(session_module, "atomic_write_json", record_atomic)
    session = FluxForgeSession()
    target = tmp_path / "empty.ffs"

    payload = write_ffs_session(target, session)

    assert calls == [(target, payload)]
    assert payload["format_version"] == 2


def test_read_wraps_invalid_json_with_session_context(tmp_path: Path) -> None:
    target = tmp_path / "broken.ffs"
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(SessionFormatError, match="Could not read FluxForge session"):
        read_ffs_session(target)
