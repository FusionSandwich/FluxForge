from pathlib import Path
import sqlite3

import numpy as np

from fluxforge.data.gamma_database import GammaDatabase
from fluxforge.data.nuclide_library import (
    build_nuclide_library,
    reference_lines_for_nuclide,
    search_nuclides,
)
from fluxforge.gui import NuclideSearchController, RecentFilesManager, SelectionBus
from fluxforge.gui.file_workflow import normalize_dropped_paths
from fluxforge.hal import DeviceRegistry, MockMCADevice
from fluxforge.io import (
    N42Measurement,
    create_reader_factory,
    read_ffs_session,
    validate_n42_file,
    write_ffs_session,
    write_n42_file,
)
from fluxforge.io.session import session_from_spectra
from fluxforge.io.spe import GammaSpectrum


class FakeSettings:
    def __init__(self) -> None:
        self._values = {}
        self.sync_count = 0

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


def test_device_registry_snapshots_registered_devices():
    registry = DeviceRegistry()
    device = registry.register(MockMCADevice(), default=True)

    assert registry.default() is device
    snapshot = registry.snapshot()
    assert snapshot[0]["source_type"] == "hal"
    assert snapshot[0]["device_id"] == device.device_id


def test_ffs_session_round_trip_preserves_hal_and_gps_fields(tmp_path):
    spectrum = GammaSpectrum(
        counts=np.array([4.0, 8.0, 15.0]),
        calibration={"energy": [0.0, 0.5]},
        source_type="hal",
        device_id="mock-mca",
        device_label="Mock MCA",
        gps={"latitude": 43.07, "longitude": -89.4},
    )
    registry = DeviceRegistry([MockMCADevice()])

    path = tmp_path / "example.ffs"
    write_ffs_session(path, session_from_spectra([spectrum], device_registry=registry))
    restored = read_ffs_session(path)

    assert restored.spectra[0].source_type == "hal"
    assert restored.spectra[0].gps["latitude"] == 43.07
    assert restored.device_snapshot[0]["device_id"] == "mock-mca"


def test_reader_factory_supports_core_extensions():
    factory = create_reader_factory()
    assert factory.supported_extensions() == (
        ".chn",
        ".cnf",
        ".csv",
        ".n42",
        ".spc",
        ".spe",
        ".xml",
    )


def test_reader_factory_reads_csv_and_spc_samples(tmp_path):
    csv_path = tmp_path / "spectrum.csv"
    csv_path.write_text(
        "channel,energy_keV,counts\n0,0.0,10\n1,0.5,22\n2,1.0,35\n",
        encoding="utf-8",
    )
    spc_path = tmp_path / "field.spc"
    spc_path.write_text(
        "SPECTRUM_ID=field\n"
        "LIVE_TIME=120\n"
        "REAL_TIME=125\n"
        "CALIBRATION=0.0,0.5,0.0\n"
        "COUNTS=\n"
        "5 6 7 8\n",
        encoding="utf-8",
    )

    factory = create_reader_factory()
    csv_spectrum = factory.read(csv_path)
    spc_spectrum = factory.read(spc_path)

    assert csv_spectrum.counts.tolist() == [10.0, 22.0, 35.0]
    assert spc_spectrum.energy_calibration == (0.0, 0.5, 0.0)
    assert spc_spectrum.live_time == 120.0


def test_write_n42_file_validates_against_bundled_schema(tmp_path):
    measurement = N42Measurement(
        counts=np.array([11, 17, 29]),
        live_time=60.0,
        real_time=61.2,
        energy_calibration=(0.0, 0.5, 0.0),
        detector_type="HPGe",
        metadata={"gps": {"latitude": 43.07, "longitude": -89.4}},
    )
    output = tmp_path / "validated.n42"

    write_n42_file(output, measurement)
    valid, errors = validate_n42_file(output)

    assert valid is True
    assert errors == []


def test_nuclide_sqlite_library_supports_search_and_decay_chain_schema(tmp_path):
    db_path = tmp_path / "nuclides.db"
    build_nuclide_library(
        db_path,
        gamma_database=GammaDatabase(),
        decay_chain_rows=[("Co60", "Cs137", 1.0, "beta-")],
    )

    hits = search_nuclides(db_path, "cs")
    assert hits
    assert hits[0].display_name == "Cs-137"
    assert reference_lines_for_nuclide(db_path, "Cs137")

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='virtual table'"
            )
        }
        decay_rows = connection.execute("SELECT COUNT(*) FROM decay_chains").fetchone()[0]

    assert "decay_chains" in tables
    assert "nuclide_search" in tables
    assert decay_rows == 1


def test_nuclide_search_controller_publishes_reference_lines(tmp_path):
    db_path = tmp_path / "nuclides.db"
    build_nuclide_library(db_path, gamma_database=GammaDatabase())

    bus = SelectionBus()
    controller = NuclideSearchController(bus, database_path=db_path)
    hits = controller.search("co")
    state = controller.activate(hits[0])

    assert state.nuclide == "Co60"
    assert state.reference_lines_keV
    assert bus.describe()["reference_lines_keV"] == state.reference_lines_keV


def test_nuclide_search_controller_exposes_detail_snapshots_and_decay_relatives(tmp_path):
    db_path = tmp_path / "nuclides.db"
    build_nuclide_library(
        db_path,
        gamma_database=GammaDatabase(),
        decay_chain_rows=[("Co60", "Cs137", 1.0, "beta-")],
    )

    controller = NuclideSearchController(SelectionBus(), database_path=db_path)
    details = controller.nuclide_details("Cs-137", age_s=86400.0)

    assert details.display_name == "Cs-137"
    assert details.half_life_s > 0.0
    assert details.gamma_lines
    assert details.gamma_lines[0].age_adjusted_intensity <= details.gamma_lines[0].intensity
    assert details.parents
    assert details.parents[0].display_name == "Co-60"
    assert details.specific_activity_bq_g > 0.0


def test_recent_files_manager_and_drop_normalization():
    settings = FakeSettings()
    manager = RecentFilesManager(settings, limit=3)

    manager.record("/tmp/a.n42")
    manager.record("/tmp/b.n42")
    files = manager.record_many(["/tmp/c.n42", "/tmp/a.n42", "/tmp/c.n42"])

    assert files == ("/tmp/c.n42", "/tmp/a.n42", "/tmp/b.n42")
    assert settings.sync_count >= 3
    assert normalize_dropped_paths(["/tmp/a.n42", "/tmp/a.n42", Path("/tmp/b.n42")]) == (
        "/tmp/a.n42",
        "/tmp/b.n42",
    )
