import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from fluxforge.io.cnf import read_cnf_file
from fluxforge.io.hpge import read_chn_file
from fluxforge.io.iec import read_iec_file
from fluxforge.io.spe import read_spe_file


def _ensure_curie_db() -> None:
    data_dir = Path("testing/curie/curie/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "ziegler.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS compounds (compound TEXT)")
    conn.commit()
    conn.close()


def _load_curie_spectrum(path: Path):
    _ensure_curie_db()
    sys.path.insert(0, str(Path("testing/curie")))
    from curie.spectrum import Spectrum as CurieSpectrum  # type: ignore
    return CurieSpectrum(str(path))


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    denom = a[:n].sum()
    if denom == 0:
        return 1.0
    return float(b[:n].sum() / denom)


@pytest.mark.skipif(
    not Path("testing/curie/examples/eu_calib_7cm.Spe").exists(),
    reason="Curie SPE example missing.",
)
def test_spe_parity():
    path = Path("testing/curie/examples/eu_calib_7cm.Spe")
    curie = _load_curie_spectrum(path)
    ff = read_spe_file(path)
    ratio = _ratio(curie.hist, ff.counts)
    assert np.isfinite(ratio)
    assert abs(1.0 - ratio) < 0.02


@pytest.mark.xfail(reason="Curie CNF parser does not align with available sample format.")
def test_cnf_parity():
    path = Path("testing/becquerel/tests/samples/01122014152731-GT01122014182338-GA37.4963000N-GO122.4633000W.cnf")
    curie = _load_curie_spectrum(path)
    ff = read_cnf_file(path)
    ratio = _ratio(curie.hist, ff.counts)
    assert np.isfinite(ratio)
    assert abs(1.0 - ratio) < 0.02


@pytest.mark.xfail(reason="Curie IEC parser expects numeric detector ID in header.")
def test_iec_parity():
    path = Path("testing/becquerel/tests/samples/hpge_dummy_test_01.iec")
    curie = _load_curie_spectrum(path)
    ff = read_iec_file(path)
    ratio = _ratio(curie.hist, ff.counts)
    assert np.isfinite(ratio)
    assert abs(1.0 - ratio) < 0.02


@pytest.mark.skipif(
    not Path("testing/curie/examples/eu_calib_7cm.Spe").exists(),
    reason="Curie SPE example missing.",
)
def test_chn_parity(tmp_path):
    path = Path("testing/curie/examples/eu_calib_7cm.Spe")
    curie = _load_curie_spectrum(path)
    chn_path = tmp_path / "eu_calib_7cm.Chn"
    curie.saveas(str(chn_path))
    curie_chn = _load_curie_spectrum(chn_path)
    ff = read_chn_file(chn_path)
    ratio = _ratio(curie_chn.hist, ff.counts)
    assert np.isfinite(ratio)
    assert abs(1.0 - ratio) < 0.02
