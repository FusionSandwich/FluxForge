import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge.data.nuclear_data_sources import (
    list_nuclear_data_sources,
    load_gamma_identification_source,
    summarize_nuclear_data_source,
)


def test_list_nuclear_data_sources_includes_builtins():
    records = list_nuclear_data_sources()
    ids = {record.source_id for record in records}

    assert "actigamma_2012" in ids
    assert "nndc_offline_activation" in ids
    assert "k0_naa_monitors" in ids
    assert "irdff_ii_dosimetry" in ids
    assert "custom_gamma_file" in ids


def test_load_nndc_identification_source_matches_known_line():
    database = load_gamma_identification_source("nndc_offline_activation")
    matches = database.find_matches(661.7, tolerance_keV=1.0)

    assert matches
    assert any(nuclide == "Cs137" for nuclide, _line in matches)


def test_summarize_nuclear_data_source_reports_capabilities():
    summary = summarize_nuclear_data_source("k0_naa_monitors")

    assert "k0-NAA" in summary or "k0-naa" in summary.lower()
    assert "capabilities=" in summary


def test_load_sqlite_identification_source(tmp_path):
    import sqlite3

    db_path = tmp_path / "gamma.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE gamma_lines (nuclide TEXT, energy_keV REAL, intensity REAL, half_life_s REAL)"
        )
        connection.execute(
            "INSERT INTO gamma_lines VALUES (?, ?, ?, ?)",
            ("Cs137", 661.7, 0.851, 9.493632e8),
        )
    database = load_gamma_identification_source(
        "custom_gamma_file", custom_path=f"sqlite:///{db_path}?table=gamma_lines"
    )

    matches = database.find_matches(661.7, tolerance_keV=1.0)
    assert matches
    assert matches[0][0] == "Cs137"


def test_load_python_identification_source(tmp_path, monkeypatch):
    module_path = tmp_path / "custom_loader.py"
    module_path.write_text(
        "def load():\n"
        "    return [\n"
        "        {'nuclide': 'Co60', 'energy_keV': 1332.5, 'intensity': 1.0, 'half_life_s': 166344192.0},\n"
        "    ]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    database = load_gamma_identification_source(
        "custom_gamma_file", custom_path="python://custom_loader:load"
    )
    matches = database.find_matches(1332.5, tolerance_keV=1.0)

    assert matches
    assert matches[0][0] == "Co60"


def test_offline_mode_blocks_remote_gamma_source(monkeypatch):
    monkeypatch.setenv("FLUXFORGE_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="FLUXFORGE_OFFLINE=1"):
        load_gamma_identification_source(
            "custom_gamma_file",
            custom_path="https://example.invalid/gamma_lines.json",
        )
