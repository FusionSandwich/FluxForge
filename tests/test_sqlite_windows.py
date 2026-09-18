import sqlite3
from urllib.parse import quote
import pytest
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source,register_user_gamma_source

@pytest.mark.parametrize('style',['native','encoded'])
@pytest.mark.parametrize('registered',[False,True])
def test_windows_sqlite_library_paths(tmp_path,monkeypatch,style,registered):
    monkeypatch.setenv('FLUXFORGE_LIBRARY_REGISTRY',str(tmp_path/'registry.json'))
    path=tmp_path/'lab gamma.sqlite'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE gamma_lines (nuclide TEXT, energy_keV REAL, intensity REAL, half_life_s REAL)')
        c.execute('INSERT INTO gamma_lines VALUES (?,?,?,?)',('Co60',1332.5,.99,166344192))
    locator='sqlite:///'+(str(path) if style=='native' else quote(path.as_posix(),safe='/:'+''))+'?table=gamma_lines'
    before=path.read_bytes()
    if registered:
        record=register_user_gamma_source('SQLite Lab',locator)
        db=load_gamma_identification_source(record.source_id)
    else:db=load_gamma_identification_source('custom_gamma_file',custom_path=locator)
    hits=db.find_matches(1332.5,tolerance_keV=.01)
    assert len(hits)==1 and hits[0][0]=='Co60'
    assert hits[0][1].intensity==pytest.approx(.99)
    assert path.read_bytes()==before

def test_missing_sqlite_library_is_not_created(tmp_path):
    path=tmp_path/'missing.sqlite'
    with pytest.raises((OSError,sqlite3.Error)):
        load_gamma_identification_source('custom_gamma_file',custom_path='sqlite:///'+path.as_posix())
    assert not path.exists()
