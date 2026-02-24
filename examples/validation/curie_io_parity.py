#!/usr/bin/env python3
"""
Curie IO parity check for SPE/CHN/CNF/IEC readers.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

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


def _summarize_counts(curie_counts: np.ndarray, ff_counts: np.ndarray) -> dict:
    curie_counts = np.asarray(curie_counts, dtype=float)
    ff_counts = np.asarray(ff_counts, dtype=float)
    min_len = min(len(curie_counts), len(ff_counts))
    ratio = np.nan
    if curie_counts.sum() > 0:
        ratio = float(ff_counts[:min_len].sum() / curie_counts[:min_len].sum())
    return {
        "curie_len": int(len(curie_counts)),
        "fluxforge_len": int(len(ff_counts)),
        "sum_ratio": ratio,
    }


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/curie_io")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    spe_path = Path("testing/curie/examples/eu_calib_7cm.Spe")
    curie_spe = _load_curie_spectrum(spe_path)
    ff_spe = read_spe_file(spe_path)
    results["spe"] = _summarize_counts(curie_spe.hist, ff_spe.counts)

    chn_path = output_dir / "eu_calib_7cm.chn"
    curie_spe.saveas(str(chn_path))
    curie_chn = _load_curie_spectrum(chn_path)
    ff_chn = read_chn_file(chn_path)
    results["chn"] = _summarize_counts(curie_chn.hist, ff_chn.counts)

    cnf_path = Path(
        "testing/becquerel/tests/samples/01122014152731-GT01122014182338-GA37.4963000N-GO122.4633000W.cnf"
    )
    curie_cnf = _load_curie_spectrum(cnf_path)
    ff_cnf = read_cnf_file(cnf_path)
    results["cnf"] = _summarize_counts(curie_cnf.hist, ff_cnf.counts)

    iec_path = Path("testing/becquerel/tests/samples/hpge_dummy_test_01.iec")
    curie_iec = _load_curie_spectrum(iec_path)
    ff_iec = read_iec_file(iec_path)
    results["iec"] = _summarize_counts(curie_iec.hist, ff_iec.counts)

    (output_dir / "curie_io_parity.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    for key, payload in results.items():
        print(f"{key}: len={payload['fluxforge_len']} ratio={payload['sum_ratio']:.6f}")


if __name__ == "__main__":
    main()
