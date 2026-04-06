import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge.data.nuclear_data_sources import (
    list_nuclear_data_sources,
    list_nuclear_data_sources_by_capability,
    list_registered_user_gamma_sources,
    load_decay_dataset_source,
    load_gamma_identification_source,
    register_user_gamma_source,
    remove_user_gamma_source,
    summarize_nuclear_data_source,
)


def test_list_nuclear_data_sources_includes_builtins():
    records = list_nuclear_data_sources()
    ids = {record.source_id for record in records}

    assert "actigamma_2012" in ids
    assert "gsa_v4_edit_library" in ids
    assert "gsa_v4_natural_library" in ids
    assert "nasa_common_lab_sources" in ids
    assert "nasa_capture_iaea" in ids
    assert "nasa_delayed_activation_iaea" in ids
    assert "radioactivedecay_icrp107_kayzero_2023" in ids
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


def test_list_nuclear_data_sources_by_capability_filters_gamma_only():
    gamma_ids = {
        record.source_id
        for record in list_nuclear_data_sources_by_capability("peak-identification")
    }

    assert "nasa_common_lab_sources" in gamma_ids
    assert "nasa_capture_capgam" not in gamma_ids
    assert "nasa_capture_iaea" not in gamma_ids
    assert "radioactivedecay_icrp107_kayzero_2023" not in gamma_ids


def test_load_gsa_identification_source_matches_known_line():
    database = load_gamma_identification_source("gsa_v4_edit_library")
    matches = database.find_matches(1460.8, tolerance_keV=0.5)

    assert matches
    assert any(nuclide == "K40" for nuclide, _line in matches)


def test_load_gsa_natural_identification_source_matches_known_line():
    database = load_gamma_identification_source("gsa_v4_natural_library")
    matches = database.find_matches(661.7, tolerance_keV=0.5)

    assert matches
    assert any(nuclide == "Cs137" for nuclide, _line in matches)


def test_load_nasa_common_lab_source_matches_known_line():
    database = load_gamma_identification_source("nasa_common_lab_sources")
    matches = database.find_matches(477.6035, tolerance_keV=0.5)

    assert matches
    assert any(nuclide == "Be7" for nuclide, _line in matches)


def test_load_nasa_capture_iaea_source_matches_known_line():
    database = load_gamma_identification_source("nasa_capture_iaea")
    matches = database.find_matches(2223.248, tolerance_keV=0.5)

    assert matches
    assert any(nuclide == "H1" for nuclide, _line in matches)


def test_load_decay_dataset_source_exposes_decay_chain_and_uncertainty():
    dataset = load_decay_dataset_source("radioactivedecay_icrp107_kayzero_2023")

    assert "Tc-99m" in dataset.decay_products("Mo-99")
    assert dataset.half_life_uncertainty_s("Ag-108") == pytest.approx(0.6)
    assert dataset.half_life_s("Al-28") == pytest.approx(134.484, rel=1e-6)


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


def test_registered_user_gamma_source_round_trip(monkeypatch, tmp_path):
    registry_path = tmp_path / "library_registry.json"
    gamma_path = tmp_path / "user_gamma.csv"
    gamma_path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\nCo60,1332.5,1.0,166344192.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry_path))

    record = register_user_gamma_source("Lab Ref", gamma_path)
    assert record.source_id == "user_gamma_lab_ref"
    assert any(
        item.source_id == "user_gamma_lab_ref"
        for item in list_registered_user_gamma_sources()
    )

    database = load_gamma_identification_source("user_gamma_lab_ref")
    matches = database.find_matches(1332.5, tolerance_keV=1.0)
    assert matches
    assert matches[0][0] == "Co60"

    assert remove_user_gamma_source("user_gamma_lab_ref") is True
    assert remove_user_gamma_source("user_gamma_lab_ref") is False


def test_register_user_gamma_source_rejects_reserved_builtin_alias(monkeypatch, tmp_path):
    registry_path = tmp_path / "library_registry.json"
    gamma_path = tmp_path / "user_gamma.csv"
    gamma_path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\nCo60,1332.5,1.0,166344192.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry_path))

    with pytest.raises(ValueError, match="reserved built-in"):
        register_user_gamma_source("fluxforge_bundled_gamma", gamma_path)
