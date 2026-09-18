import hashlib, os
from pathlib import Path
import numpy as np
import pytest
from fluxforge.data.rafm_profile import load_rafm_profile,list_rafm_profiles,RAFMProfile
from fluxforge.io.genie import read_genie_spectrum

@pytest.mark.parametrize('name',list_rafm_profiles())
def test_profile_background_resolves_from_actual_package(name):
    p=load_rafm_profile(name).resolve_background_path()
    assert p is not None and p.is_file(),str(p)
    expected=os.environ.get('FEATURE_EXPECT_BACKGROUND_HASH')
    if expected:assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    if os.environ.get('FEATURE_INSTALLED'):
        import fluxforge
        assert p.is_relative_to(Path(fluxforge.__file__).parent)
        assert not any('FluxForge-validation' in value for value in __import__('sys').path)
    s=read_genie_spectrum(p)
    assert len(s.counts)==8192
    assert np.all(np.isfinite(s.counts))

def test_explicit_repository_override_preserved(tmp_path):
    profile=load_rafm_profile('rafm_25cm')
    assert profile.resolve_background_path(repo_root=tmp_path)==(tmp_path/profile.background_relative_path).resolve()

def test_custom_profile_without_background():
    profile=RAFMProfile(name='custom',description='',background_relative_path=None,energy_calibration=[],efficiency={},resolution=[])
    assert profile.resolve_background_path() is None

@pytest.mark.parametrize('name',['rafm_25cm','astm_inl_dosimetry'])
def test_profile_background_loader_and_cli_resolver(name):
    from fluxforge.analysis.flux_wire_analysis import _profile_background_spectrum
    from fluxforge.cli.app import _profile_background_file
    p=_profile_background_file(name)
    assert p.is_file()
    raw=read_genie_spectrum(p);s=_profile_background_spectrum(name)
    np.testing.assert_array_equal(s.counts,raw.counts)
    assert s.calibration['energy']==load_rafm_profile(name).energy_calibration
