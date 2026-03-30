"""SQLite-backed nuclide library for modern GUI search and overlays."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from fluxforge.data.gamma_database import GammaDatabase
from fluxforge.data.isotope_names import format_isotope_name, parse_gamma_nuclide_name


@dataclass(frozen=True)
class NuclideSearchHit:
    """Search result row for the SQLite-backed nuclide library."""

    nuclide: str
    display_name: str
    half_life_s: float
    strongest_lines_keV: tuple[float, ...]


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS nuclides (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      display_name TEXT NOT NULL,
      element TEXT NOT NULL,
      mass_number INTEGER NOT NULL,
      metastable INTEGER NOT NULL DEFAULT 0,
      zai INTEGER NOT NULL DEFAULT 0,
      half_life_s REAL NOT NULL DEFAULT 0.0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gamma_lines (
      id INTEGER PRIMARY KEY,
      nuclide_id INTEGER NOT NULL REFERENCES nuclides(id),
      energy_keV REAL NOT NULL,
      intensity REAL NOT NULL DEFAULT 0.0,
      energy_unc_keV REAL NOT NULL DEFAULT 0.0,
      intensity_unc REAL NOT NULL DEFAULT 0.0,
      norm REAL NOT NULL DEFAULT 1.0,
      norm_unc REAL NOT NULL DEFAULT 0.0,
      line_type TEXT NOT NULL DEFAULT 'gamma'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decay_chains (
      id INTEGER PRIMARY KEY,
      parent_id INTEGER REFERENCES nuclides(id),
      daughter_id INTEGER REFERENCES nuclides(id),
      branching_ratio REAL,
      decay_mode TEXT
    )
    """,
)


def _search_table_ddl() -> str:
    return (
        "CREATE VIRTUAL TABLE IF NOT EXISTS nuclide_search "
        "USING fts5(name, display_name, aliases)"
    )


def _build_aliases(name: str, display_name: str) -> str:
    return " ".join({name, display_name, name.lower(), display_name.lower()})


def ensure_bundled_nuclide_database(
    path: str | Path | None = None,
    *,
    decay_chain_rows: Sequence[tuple[str, str, float, str]] = (),
) -> Path:
    """Build the bundled SQLite nuclide library when it does not exist yet."""

    if path is None:
        path = Path.home() / ".fluxforge" / "db" / "nuclides.db"
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        return db_path
    build_nuclide_library(
        db_path,
        gamma_database=GammaDatabase(),
        decay_chain_rows=decay_chain_rows,
    )
    return db_path


def build_nuclide_library(
    path: str | Path,
    *,
    gamma_database: GammaDatabase,
    decay_chain_rows: Sequence[tuple[str, str, float, str]] = (),
) -> Path:
    """Build a searchable SQLite library from the bundled gamma database."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)

        fts5_enabled = True
        try:
            connection.execute(_search_table_ddl())
        except sqlite3.OperationalError:
            fts5_enabled = False
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS nuclide_search (
                  name TEXT NOT NULL,
                  display_name TEXT NOT NULL,
                  aliases TEXT NOT NULL
                )
                """
            )

        nuclide_ids: dict[str, int] = {}
        for name in gamma_database.nuclides:
            decay = gamma_database[name]
            element, mass_number, metastable = parse_gamma_nuclide_name(name)
            display_name = format_isotope_name(element, mass_number, metastable)
            cursor = connection.execute(
                """
                INSERT INTO nuclides (
                  name, display_name, element, mass_number, metastable, zai, half_life_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    display_name,
                    element,
                    mass_number,
                    metastable,
                    decay.zai,
                    decay.halflife,
                ),
            )
            nuclide_id = int(cursor.lastrowid)
            nuclide_ids[name] = nuclide_id

            connection.execute(
                "INSERT INTO nuclide_search (name, display_name, aliases) VALUES (?, ?, ?)",
                (name, display_name, _build_aliases(name, display_name)),
            )

            for line in decay.gamma_lines:
                connection.execute(
                    """
                    INSERT INTO gamma_lines (
                      nuclide_id, energy_keV, intensity, energy_unc_keV,
                      intensity_unc, norm, norm_unc, line_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'gamma')
                    """,
                    (
                        nuclide_id,
                        line.energy_keV,
                        line.intensity,
                        line.energy_unc / 1000.0,
                        line.intensity_unc,
                        line.norm,
                        line.norm_unc,
                    ),
                )

        for parent, daughter, branching_ratio, decay_mode in decay_chain_rows:
            if parent not in nuclide_ids or daughter not in nuclide_ids:
                continue
            connection.execute(
                """
                INSERT INTO decay_chains (parent_id, daughter_id, branching_ratio, decay_mode)
                VALUES (?, ?, ?, ?)
                """,
                (nuclide_ids[parent], nuclide_ids[daughter], branching_ratio, decay_mode),
            )

        connection.commit()
        connection.execute(f"PRAGMA user_version = {1 if fts5_enabled else 0}")

    return db_path


def search_nuclides(
    path: str | Path,
    query: str,
    *,
    limit: int = 12,
) -> list[NuclideSearchHit]:
    """Search nuclides by compact or display name."""

    db_path = Path(path)
    text = query.strip()
    if not text:
        return []

    with sqlite3.connect(db_path) as connection:
        try:
            rows = connection.execute(
                """
                SELECT n.name, n.display_name, n.half_life_s
                FROM nuclide_search s
                JOIN nuclides n ON n.name = s.name
                WHERE s.nuclide_search MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (f"{text}*", limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{text.lower()}%"
            rows = connection.execute(
                """
                SELECT n.name, n.display_name, n.half_life_s
                FROM nuclide_search s
                JOIN nuclides n ON n.name = s.name
                WHERE lower(s.aliases) LIKE ?
                ORDER BY n.display_name
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()

        hits = []
        for name, display_name, half_life_s in rows:
            hits.append(
                NuclideSearchHit(
                    nuclide=str(name),
                    display_name=str(display_name),
                    half_life_s=float(half_life_s or 0.0),
                    strongest_lines_keV=reference_lines_for_nuclide(
                        db_path, str(name), limit=3
                    ),
                )
            )
        return hits


def reference_lines_for_nuclide(
    path: str | Path,
    nuclide: str,
    *,
    limit: int = 8,
    min_intensity: float = 0.0,
) -> tuple[float, ...]:
    """Return the strongest reference lines for a nuclide."""

    db_path = Path(path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT g.energy_keV
            FROM gamma_lines g
            JOIN nuclides n ON n.id = g.nuclide_id
            WHERE n.name = ? AND g.intensity >= ?
            ORDER BY g.intensity DESC, g.energy_keV ASC
            LIMIT ?
            """,
            (nuclide, min_intensity, limit),
        ).fetchall()
    return tuple(float(row[0]) for row in rows)


__all__ = [
    "NuclideSearchHit",
    "build_nuclide_library",
    "ensure_bundled_nuclide_database",
    "reference_lines_for_nuclide",
    "search_nuclides",
]
