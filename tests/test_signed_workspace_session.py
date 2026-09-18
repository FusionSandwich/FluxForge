from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest

from fluxforge.core.workspace_document import (
    WorkspaceDocument,
    WorkspaceSpectrum,
    WorkspaceValidationError,
)
from fluxforge.io.session import FluxForgeSession, read_ffs_session, write_ffs_session
from fluxforge.io.spe import GammaSpectrum


def _document(counts: list[float], uncertainty: list[float]) -> WorkspaceDocument:
    spectrum = GammaSpectrum(
        counts=counts,
        counts_uncertainty=uncertainty,
        channels=[0.0, 1.5, 3.0],
        energies=np.array([10.25, 11.75, 13.25]),
        live_time=12.5,
        real_time=13.0,
        spectrum_id="background-subtracted",
        detector_id="detector-1",
        calibration={"energy": [10.25, 1.0]},
        metadata={"original": {"kind": "measurement", "run": 7}},
    )
    return WorkspaceDocument(
        document_id="signed-session",
        spectra=(
            WorkspaceSpectrum(
                spectrum_id="background-subtracted",
                spectrum=spectrum,
                label="Background subtracted",
            ),
        ),
        active_spectrum_id="background-subtracted",
    )


def _schema() -> dict:
    resource = files("fluxforge").joinpath(
        "resources/schemas/workspace_document_v2.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def test_signed_fractional_spectrum_ffs_round_trip(tmp_path) -> None:
    document = _document([4.25, -1.75, 0.5], [2.1, 1.4, 0.9])
    target = tmp_path / "signed.ffs"

    written = write_ffs_session(target, FluxForgeSession(document=document))
    restored = read_ffs_session(target).document.spectra[0].spectrum

    np.testing.assert_array_equal(restored.counts, [4.25, -1.75, 0.5])
    np.testing.assert_array_equal(restored.counts_uncertainty, [2.1, 1.4, 0.9])
    np.testing.assert_array_equal(restored.energies, [10.25, 11.75, 13.25])
    assert restored.metadata == {"original": {"kind": "measurement", "run": 7}}


def test_signed_document_agrees_with_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = _document([4.25, -1.75, 0.5], [2.1, 1.4, 0.9]).to_dict()

    jsonschema.Draft202012Validator(_schema()).validate(payload)


def test_json_schema_rejects_signed_counts_with_null_uncertainty() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = _document([4.25, -1.75, 0.5], [2.1, 1.4, 0.9]).to_dict()
    payload["spectra"][0]["spectrum"]["counts_uncertainty"] = None

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema()).validate(payload)


@pytest.mark.parametrize("uncertainty", [None, "missing"])
def test_signed_serialized_payload_requires_explicit_uncertainty(uncertainty) -> None:
    payload = _document([4.25, 1.75, 0.5], [2.1, 1.4, 0.9]).to_dict()
    payload["spectra"][0]["spectrum"]["counts"][1] = -1.75
    spectrum_payload = payload["spectra"][0]["spectrum"]
    if uncertainty == "missing":
        spectrum_payload.pop("counts_uncertainty")
    else:
        spectrum_payload["counts_uncertainty"] = None

    with pytest.raises(WorkspaceValidationError, match="counts_uncertainty"):
        WorkspaceDocument.from_dict(payload)


@pytest.mark.parametrize("uncertainty", [None, "missing"])
def test_positive_legacy_payload_can_default_uncertainty(uncertainty) -> None:
    payload = _document([4.25, 1.75, 0.5], [2.1, 1.4, 0.9]).to_dict()
    spectrum_payload = payload["spectra"][0]["spectrum"]
    if uncertainty == "missing":
        spectrum_payload.pop("counts_uncertainty")
    else:
        spectrum_payload["counts_uncertainty"] = None

    restored = WorkspaceDocument.from_dict(payload).spectra[0].spectrum

    np.testing.assert_allclose(restored.counts_uncertainty, np.sqrt(restored.counts))
