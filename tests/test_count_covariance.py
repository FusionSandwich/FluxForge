import copy
import json
from importlib.resources import files

import numpy as np
import pytest
from scipy import sparse

from fluxforge.core.schemas import SPECTRUM_FILE_SCHEMA
from fluxforge.core.workspace_document import WorkspaceDocument, WorkspaceSpectrum
from fluxforge.io.artifacts import write_spectrum_file, read_spectrum_file
from fluxforge.io.session import FluxForgeSession, read_ffs_session, write_ffs_session
from fluxforge.io.spe import GammaSpectrum, write_spe_file


def correlated():
    return GammaSpectrum(counts=[4.5, -2.0, 8.0],
                         counts_covariance=sparse.csr_matrix([[4, 2, 0], [2, 9, 1], [0, 1, 3]]),
                         energies=np.array([10., 11., 12.]), live_time=10, real_time=12)


def test_correlated_artifact_and_session_roundtrip(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    spectrum = correlated()
    path = tmp_path/'spectrum.json'
    write_spectrum_file(path, spectrum)
    payload = read_spectrum_file(path)
    jsonschema.validate(payload, SPECTRUM_FILE_SCHEMA)
    restored = GammaSpectrum.from_dict(payload['spectrum'])
    document = WorkspaceDocument(document_id='covariance', spectra=(
        WorkspaceSpectrum(spectrum_id='net', spectrum=restored),), active_spectrum_id='net')
    schema = json.loads(files('fluxforge').joinpath('resources/schemas/workspace_document_v2.schema.json').read_text())
    jsonschema.validate(document.to_dict(), schema)
    session = tmp_path/'covariance.ffs'
    write_ffs_session(session, FluxForgeSession(document=document))
    actual = read_ffs_session(session).document.spectra[0].spectrum
    np.testing.assert_array_equal(actual.counts, spectrum.counts)
    np.testing.assert_array_equal(actual.energies, spectrum.energies)
    np.testing.assert_array_equal(actual.counts_covariance.toarray(), spectrum.counts_covariance.toarray())
    # 4 + 9 + 3 + 2*(2+1) = 22, not the diagonal sum 16.
    assert actual.counts_in_range(0, 2, use_energy=False) == pytest.approx((10.5, np.sqrt(22)))
    actual.counts_covariance[0, 0] = 100
    assert spectrum.counts_covariance[0, 0] == 4


@pytest.mark.parametrize('matrix,reason', [
    ([[1, 2], [2, 1]], 'positive semidefinite'),
    ([[1, 0], [1, 1]], 'symmetric'),
    ([[1, 0], [0, -1]], 'nonnegative'),
    ([[1, 0], [0, np.nan]], 'finite'),
])
def test_invalid_covariance(matrix, reason):
    with pytest.raises(ValueError, match=reason):
        GammaSpectrum(counts=[1, 2], counts_covariance=sparse.csr_matrix(matrix))


def test_singular_covariance_is_valid_storage():
    spec = GammaSpectrum(counts=[-1, 2], counts_covariance=sparse.csr_matrix([[4, 4], [4, 4]]))
    assert spec.linear_variance(np.array([1, -1])) == 0
    np.testing.assert_array_equal(spec.counts_uncertainty, [2, 2])


def test_covariance_only_signed_session_payload():
    import jsonschema
    document = WorkspaceDocument(document_id='covariance', spectra=(
        WorkspaceSpectrum(spectrum_id='net', spectrum=correlated()),), active_spectrum_id='net')
    payload = document.to_dict()
    payload['spectra'][0]['spectrum'].pop('counts_uncertainty')
    schema = json.loads(files('fluxforge').joinpath('resources/schemas/workspace_document_v2.schema.json').read_text())
    jsonschema.validate(payload, schema)
    restored = WorkspaceDocument.from_dict(payload).spectra[0].spectrum
    np.testing.assert_allclose(restored.counts_uncertainty, np.sqrt([4,9,3]))


def test_large_banded_psd_with_zero_modes():
    matrix = sparse.diags([np.ones(1023), np.full(1024, 2.), np.ones(1023)], [-1,0,1], format='csr')
    # A PSD rank-one 2x2 block that is not diagonally dominant.
    matrix[0,0] = 1
    matrix[0,1] = matrix[1,0] = 2
    matrix[1,1] = 4
    matrix[1,2] = matrix[2,1] = 0
    spec = GammaSpectrum(counts=np.zeros(1024), counts_covariance=matrix)
    assert spec.counts_covariance.shape == (1024,1024)
    matrix[0,0] = 0.5
    with pytest.raises(ValueError, match='positive semidefinite'):
        GammaSpectrum(counts=np.zeros(1024), counts_covariance=matrix)


def test_diagonal_consistency_and_shape():
    with pytest.raises(ValueError, match='diagonal'):
        GammaSpectrum(counts=[1, 2], counts_uncertainty=[1, 1], counts_covariance=sparse.diags([1, 4]))
    with pytest.raises(ValueError, match='shape'):
        GammaSpectrum(counts=[1, 2], counts_covariance=sparse.eye(3))


@pytest.mark.parametrize('key,value', [('indices', [99]), ('indptr', [0]), ('format', 'dense'),
                                     ('shape', [4, 4]), ('indices', [0.5])])
def test_malformed_sparse_payload(key, value):
    payload = correlated().to_dict()
    payload['counts_covariance'][key] = value
    with pytest.raises(ValueError, match='counts_covariance'):
        GammaSpectrum.from_dict(payload)


def test_legacy_diagonal_and_unsupported_export(tmp_path):
    legacy = GammaSpectrum.from_dict({'counts': [4, 9], 'channels': [0, 1]})
    assert legacy.counts_covariance is None
    np.testing.assert_array_equal(legacy.counts_uncertainty, [2, 3])
    destination = tmp_path/'no-loss.spe'
    destination.write_text('preserve')
    with pytest.raises(ValueError, match='counts_covariance'):
        write_spe_file(correlated(), destination)
    assert destination.read_text() == 'preserve'


def test_lossless_csv_roundtrip(tmp_path):
    from fluxforge.io.spectrum_csv import read_spectrum_csv, write_spectrum_csv
    path = tmp_path/'adjusted.csv'
    spectrum = correlated()
    write_spectrum_csv(path, spectrum)
    actual = read_spectrum_csv(path)
    assert actual.to_dict() == spectrum.to_dict()


def test_copy_keeps_covariance_and_legacy_arithmetic_rejects():
    from fluxforge.analysis.spectrum_math import subtract_measured_background, add_spectra, moving_average
    spectrum = correlated()
    actual = subtract_measured_background(spectrum, None, warn_missing=False)
    assert actual.to_dict() == spectrum.to_dict()
    assert actual.counts_covariance is not spectrum.counts_covariance
    with pytest.raises(ValueError, match='counts_covariance'):
        add_spectra(spectrum, spectrum)
    with pytest.raises(ValueError, match='counts_covariance'):
        moving_average(spectrum, 1)
