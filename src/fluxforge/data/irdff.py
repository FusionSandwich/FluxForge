"""
IRDFF-II Cross Section Database Access Module

Provides direct access to the IAEA IRDFF-II (International Reactor Dosimetry 
and Fusion File) database for neutron dosimetry cross sections.

The IRDFF-II library contains 119 metrology reactions with covariance information
for neutron dosimetry applications from thermal to 60 MeV.

References:
    A. Trkov et al., "IRDFF-II: A New Neutron Metrology Library", 
    Nuclear Data Sheets, Vol. 163, pp. 1-108 (2020).
    https://doi.org/10.1016/j.nds.2019.12.001

    IAEA-NDS: https://www-nds.iaea.org/IRDFF/
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
from fluxforge.core.runtime import require_network_access

# NumPy 2.0 renamed trapz to trapezoid - provide compatibility
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

# Try to import CrossSection from local module
try:
    from fluxforge.data.crosssections import CrossSection
except ImportError:
    CrossSection = None  # Will define minimal version below


# =============================================================================
# IRDFF-II Database URLs and Configuration
# =============================================================================

IRDFF_BASE_URL = "https://www-nds.iaea.org/IRDFF/"

# Direct download URLs for different formats
IRDFF_URLS = {
    "tab": IRDFF_BASE_URL
    + "IRDFF-II_TAB.zip",  # 4-column tabulated (E, σ, abs unc, rel unc)
    "endf": IRDFF_BASE_URL + "IRDFF-II_ENDF.zip",  # Pointwise ENDF-6 format
    "group640": IRDFF_BASE_URL + "IRDFF-II_g640.zip",  # 640-group structure
    "group725": IRDFF_BASE_URL + "IRDFF-II_g725.zip",  # 725-group SAND-II structure
    "ace": IRDFF_BASE_URL + "IRDFF-II_ACE.zip",  # ACE format for MCNP
}

# Energy group structure URLs
IRDFF_ENERGY_GRIDS = {
    "mcnp640": IRDFF_BASE_URL + "MCNP_640.egb",
    "sand725": IRDFF_BASE_URL + "SAND_725.egb",
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".fluxforge" / "irdff_cache"


# =============================================================================
# IRDFF-II Reaction Catalog
# =============================================================================

# Key dosimetry reactions from IRDFF-II Table I
# Organized by category for flux wire applications
IRDFF_REACTIONS = {
    # Thermal/epithermal reactions (n,γ)
    "thermal": {
        "In-115(n,g)In-116m": {
            "target": "In-115",
            "product": "In-116m",
            "mt": 102,
            "threshold": 0.0,
        },
        "Au-197(n,g)Au-198": {
            "target": "Au-197",
            "product": "Au-198",
            "mt": 102,
            "threshold": 0.0,
        },
        "Co-59(n,g)Co-60": {
            "target": "Co-59",
            "product": "Co-60",
            "mt": 102,
            "threshold": 0.0,
        },
        "Sc-45(n,g)Sc-46": {
            "target": "Sc-45",
            "product": "Sc-46",
            "mt": 102,
            "threshold": 0.0,
        },
        "Cu-63(n,g)Cu-64": {
            "target": "Cu-63",
            "product": "Cu-64",
            "mt": 102,
            "threshold": 0.0,
        },
        "Mn-55(n,g)Mn-56": {
            "target": "Mn-55",
            "product": "Mn-56",
            "mt": 102,
            "threshold": 0.0,
        },
        "Na-23(n,g)Na-24": {
            "target": "Na-23",
            "product": "Na-24",
            "mt": 102,
            "threshold": 0.0,
        },
        "Fe-58(n,g)Fe-59": {
            "target": "Fe-58",
            "product": "Fe-59",
            "mt": 102,
            "threshold": 0.0,
        },
        "W-186(n,g)W-187": {
            "target": "W-186",
            "product": "W-187",
            "mt": 102,
            "threshold": 0.0,
        },
        "Ta-181(n,g)Ta-182": {
            "target": "Ta-181",
            "product": "Ta-182",
            "mt": 102,
            "threshold": 0.0,
        },
    },
    # Epithermal resonance reactions
    "epithermal": {
        "In-113(n,g)In-114m": {
            "target": "In-113",
            "product": "In-114m",
            "mt": 102,
            "threshold": 0.0,
        },
        "In-115(n,g)In-116m": {
            "target": "In-115",
            "product": "In-116m",
            "mt": 102,
            "threshold": 0.0,
        },
    },
    # Threshold reactions - Fast flux monitors
    "fast": {
        # (n,p) reactions
        "Ti-46(n,p)Sc-46": {
            "target": "Ti-46",
            "product": "Sc-46",
            "mt": 103,
            "threshold": 1.62,
        },
        "Ti-47(n,p)Sc-47": {
            "target": "Ti-47",
            "product": "Sc-47",
            "mt": 103,
            "threshold": 0.22,
        },
        "Ti-48(n,p)Sc-48": {
            "target": "Ti-48",
            "product": "Sc-48",
            "mt": 103,
            "threshold": 3.35,
        },
        "Fe-54(n,p)Mn-54": {
            "target": "Fe-54",
            "product": "Mn-54",
            "mt": 103,
            "threshold": 0.09,
        },
        "Fe-56(n,p)Mn-56": {
            "target": "Fe-56",
            "product": "Mn-56",
            "mt": 103,
            "threshold": 2.97,
        },
        "Ni-58(n,p)Co-58": {
            "target": "Ni-58",
            "product": "Co-58",
            "mt": 103,
            "threshold": 0.40,
        },
        "S-32(n,p)P-32": {
            "target": "S-32",
            "product": "P-32",
            "mt": 103,
            "threshold": 0.96,
        },
        "Al-27(n,p)Mg-27": {
            "target": "Al-27",
            "product": "Mg-27",
            "mt": 103,
            "threshold": 1.90,
        },
        "Cu-63(n,p)Ni-63": {
            "target": "Cu-63",
            "product": "Ni-63",
            "mt": 103,
            "threshold": 0.0,
        },
        "Cr-52(n,p)V-52": {
            "target": "Cr-52",
            "product": "V-52",
            "mt": 103,
            "threshold": 2.97,
        },
        "V-51(n,p)Ti-51": {
            "target": "V-51",
            "product": "Ti-51",
            "mt": 103,
            "threshold": 1.53,
        },
        # (n,α) reactions
        "Fe-54(n,a)Cr-51": {
            "target": "Fe-54",
            "product": "Cr-51",
            "mt": 107,
            "threshold": 0.84,
        },
        "Al-27(n,a)Na-24": {
            "target": "Al-27",
            "product": "Na-24",
            "mt": 107,
            "threshold": 3.25,
        },
        "Co-59(n,a)Mn-56": {
            "target": "Co-59",
            "product": "Mn-56",
            "mt": 107,
            "threshold": 5.17,
        },
        # (n,2n) reactions
        "In-115(n,n')In-115m": {
            "target": "In-115",
            "product": "In-115m",
            "mt": 4,
            "threshold": 0.34,
        },
        "Ni-58(n,2n)Ni-57": {
            "target": "Ni-58",
            "product": "Ni-57",
            "mt": 16,
            "threshold": 12.4,
        },
        "Al-27(n,2n)Al-26": {
            "target": "Al-27",
            "product": "Al-26",
            "mt": 16,
            "threshold": 13.5,
        },
        # High-threshold reactions (for high-energy spectra)
        "Nb-93(n,2n)Nb-92m": {
            "target": "Nb-93",
            "product": "Nb-92m",
            "mt": 16,
            "threshold": 9.0,
        },
        "Zr-90(n,2n)Zr-89": {
            "target": "Zr-90",
            "product": "Zr-89",
            "mt": 16,
            "threshold": 12.0,
        },
        "F-19(n,2n)F-18": {
            "target": "F-19",
            "product": "F-18",
            "mt": 16,
            "threshold": 10.4,
        },
    },
    # Fission reactions
    "fission": {
        "U-235(n,f)": {"target": "U-235", "product": "FP", "mt": 18, "threshold": 0.0},
        "U-238(n,f)": {"target": "U-238", "product": "FP", "mt": 18, "threshold": 1.0},
        "Np-237(n,f)": {
            "target": "Np-237",
            "product": "FP",
            "mt": 18,
            "threshold": 0.5,
        },
        "Pu-239(n,f)": {
            "target": "Pu-239",
            "product": "FP",
            "mt": 18,
            "threshold": 0.0,
        },
    },
}

# Cd cutoff energy (standard)
CADMIUM_CUTOFF_EV = 0.55  # eV

# Concatenated pointwise archive inside the IAEA IRDFF-II_TAB.zip
IRDFF_TAB_ARCHIVE_NAME = "IRDFF-II.tab.txt"
# Cover/absorber total, elastic and disappearance cross sections (B, B-10, Cd, Gd)
IRDFF_ABS_ARCHIVE_NAME = "IRDFF-II.abs.txt"
BUILTIN_APPROXIMATION_SOURCE = "built-in approximation (not IRDFF-II)"


class MissingCrossSectionError(ValueError):
    """Raised when a requested reaction has no cross-section data."""


def _endf_float(token: str) -> float:
    """Parse a float that may use ENDF's exponent form (``1.0745-5``)."""
    try:
        return float(token)
    except ValueError:
        match = re.fullmatch(r"([+-]?\d*\.?\d+)([+-]\d+)", token)
        if match is None:
            raise
        return float(f"{match.group(1)}e{match.group(2)}")


def parse_irdff_tab_archive(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse the concatenated IAEA ``IRDFF-II.tab.txt`` archive.

    Each block starts with a reaction header (e.g. ``Co-59(n,g)``) followed by
    lines of energy (eV), cross section (b), absolute uncertainty (b) and
    relative uncertainty (%). Returns ``{header: array(n, 4)}``.
    """
    blocks: Dict[str, List[List[float]]] = {}
    current: Optional[List[List[float]]] = None
    with open(path, "r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            try:
                values = [_endf_float(token) for token in parts]
            except ValueError:
                header = line.strip()
                if header in blocks:
                    raise ValueError(f"Duplicate IRDFF-II block {header!r} in {path}")
                current = blocks.setdefault(header, [])
                continue
            if current is None:
                raise ValueError(f"Data before the first reaction header in {path}")
            values += [np.nan] * (4 - len(values))
            current.append(values[:4])
    return {name: np.asarray(rows, dtype=float) for name, rows in blocks.items() if rows}


_REACTION_ID_RE = re.compile(
    r"^(?P<el>[A-Z][a-z]?)-(?P<a>\d+)\(n,(?P<ej>[^)]+)\)(?P<pel>[A-Z][a-z]?)-(?P<pa>\d+)(?P<state>[mg]?)$"
)
# (delta Z, delta A) of the residual nucleus for n + target -> ejectile + residual
_EJECTILE_SHIFTS = {
    "g": (0, 1),
    "p": (-1, 0),
    "d": (-1, -1),
    "t": (-1, -2),
    "a": (-2, -3),
    "n'": (0, 0),
    "2n": (0, -1),
    "3n": (0, -2),
    "np": (-1, -1),
}


def match_irdff_archive_key(reaction: str, keys) -> Optional[str]:
    """
    Map a FluxForge reaction ID (``Co-59(n,g)Co-60``) to an IRDFF-II header.

    IRDFF-II omits the product when it is the unique residual
    (``Co-59(n,g)``). Such a header is accepted only when the requested
    product is exactly that residual in its ground/total state; metastable
    products must match a header that names them.
    """
    compact = reaction.replace(" ", "")
    key_set = {key.replace(" ", ""): key for key in keys}
    if compact in key_set:
        return key_set[compact]
    match = _REACTION_ID_RE.match(compact)
    if match is None or match.group("state") == "m":
        return None
    shift = _EJECTILE_SHIFTS.get(match.group("ej"))
    if shift is None:
        return None
    from fluxforge.data.elements import z_from_element

    try:
        z_target = z_from_element(match.group("el"))
        z_product = z_from_element(match.group("pel"))
    except (KeyError, ValueError):
        return None
    if (z_product, int(match.group("pa"))) != (
        z_target + shift[0],
        int(match.group("a")) + shift[1],
    ):
        return None
    short = f"{match.group('el')}-{match.group('a')}(n,{match.group('ej')})"
    return key_set.get(short)


# =============================================================================
# IRDFF-II Data Loading Functions
# =============================================================================


@dataclass
class IRDFFCrossSection:
    """
    Container for IRDFF-II cross section data.

    Attributes
    ----------
    reaction : str
        Reaction identifier (e.g., 'Ti-46(n,p)Sc-46')
    target : str
        Target nuclide
    product : str
        Product nuclide
    mt_number : int
        ENDF MT reaction type
    energies : np.ndarray
        Energy grid in eV
    cross_sections : np.ndarray
        Cross section values in barns
    uncertainties : np.ndarray
        Absolute uncertainties in barns
    relative_unc : np.ndarray
        Relative uncertainties (%)
    threshold_eV : float
        Reaction threshold in eV
    source : str
        Data source (typically 'IRDFF-II')
    interpolation : str
        Interpolation law of the tabulation: ``"lin-lin"`` for the linearized
        IRDFF-II pointwise archive, ``"log-log"`` for sparse approximations.
    evaluation_key : str
        Reaction name exactly as it appears in the evaluated source.
    source_sha256 : str
        SHA256 of the source file the data were read from (empty if none).
    """

    reaction: str
    target: str
    product: str
    mt_number: int
    energies: np.ndarray  # eV
    cross_sections: np.ndarray  # barns
    uncertainties: np.ndarray  # barns (absolute)
    relative_unc: np.ndarray  # %
    threshold_eV: float = 0.0
    source: str = "IRDFF-II"
    interpolation: str = "log-log"
    evaluation_key: str = ""
    source_sha256: str = ""

    @property
    def energies_MeV(self) -> np.ndarray:
        """Energy grid in MeV."""
        return self.energies / 1e6

    @property
    def is_approximation(self) -> bool:
        """True for built-in sparse approximations rather than evaluated data."""
        return self.source == BUILTIN_APPROXIMATION_SOURCE

    def evaluate(
        self, energy_eV: Union[float, np.ndarray], log_interp: Optional[bool] = None
    ) -> np.ndarray:
        """
        Evaluate cross section at given energy.

        Parameters
        ----------
        energy_eV : float or np.ndarray
            Energy in eV
        log_interp : bool, optional
            Force log-log (True) or lin-lin (False) interpolation. By default
            the tabulation's own interpolation law is used.

        Returns
        -------
        np.ndarray
            Cross section in barns
        """
        energy_eV = np.atleast_1d(np.asarray(energy_eV, dtype=float))
        if log_interp is None:
            log_interp = self.interpolation == "log-log"

        if log_interp:
            # Log-log interpolation
            log_e = np.log(np.clip(self.energies, 1e-20, None))
            log_xs = np.log(np.clip(self.cross_sections, 1e-50, None))

            result = np.exp(
                np.interp(
                    np.log(np.clip(energy_eV, 1e-20, None)),
                    log_e,
                    log_xs,
                    left=log_xs[0],
                    right=log_xs[-1],
                )
            )
        else:
            result = np.interp(energy_eV, self.energies, self.cross_sections)

        # Zero below threshold
        result[energy_eV < self.threshold_eV] = 0.0
        # Zero above data range
        result[energy_eV > self.energies.max()] = 0.0

        return result.squeeze()

    def to_group_structure(
        self,
        group_edges_eV: np.ndarray,
        weighting_spectrum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collapse cross section to group structure.

        Parameters
        ----------
        group_edges_eV : np.ndarray
            Energy group boundaries in eV (length n_groups + 1)
        weighting_spectrum : np.ndarray, optional
            Weighting spectrum (1/E if None)

        Returns
        -------
        group_xs : np.ndarray
            Group-averaged cross sections (barns)
        group_unc : np.ndarray
            Group uncertainties (barns). Pointwise absolute uncertainties are
            treated as fully correlated within a group, so the group value is
            the weighted mean absolute uncertainty (an upper bound that has the
            units of barns). Correlations between groups are not provided.
        """
        n_groups = len(group_edges_eV) - 1
        group_xs = np.zeros(n_groups)
        group_unc = np.zeros(n_groups)

        for g in range(n_groups):
            e_lo = group_edges_eV[g]
            e_hi = group_edges_eV[g + 1]

            # Fine grid within the group, plus every tabulated point inside it
            # so narrow resonances are integrated rather than stepped over.
            n_points = 100
            e_start = max(e_lo, 1e-5)
            tabulated = self.energies[(self.energies > e_start) & (self.energies < e_hi)]
            e_fine = np.unique(
                np.concatenate([np.geomspace(e_start, e_hi, n_points), tabulated])
            )

            sigma_fine = self.evaluate(e_fine)

            if weighting_spectrum is not None:
                # Interpolate weighting spectrum
                weight_fine = np.interp(
                    e_fine, group_edges_eV[:-1], weighting_spectrum[:n_groups]
                )
            else:
                # 1/E weighting (default for reactor spectra)
                weight_fine = 1.0 / e_fine

            # Weighted average
            numerator = _trapezoid(sigma_fine * weight_fine, e_fine)
            denominator = _trapezoid(weight_fine, e_fine)

            if denominator > 0:
                group_xs[g] = numerator / denominator
                unc_fine = np.interp(e_fine, self.energies, self.uncertainties)
                unc_fine[sigma_fine == 0.0] = 0.0
                group_unc[g] = _trapezoid(unc_fine * weight_fine, e_fine) / denominator

        return group_xs, group_unc


class IRDFFDatabase:
    """
    Interface to the IRDFF-II cross section database.

    Downloads and caches cross-section data from the IAEA IRDFF-II database.
    Supports both tabulated (4-column) and group-averaged formats.

    Examples
    --------
    >>> db = IRDFFDatabase()
    >>> xs = db.get_cross_section("Ti-46(n,p)Sc-46")
    >>> print(f"Threshold: {xs.threshold_eV/1e6:.2f} MeV")
    >>> sigma = xs.evaluate(14e6)  # At 14 MeV
    >>> print(f"σ(14 MeV) = {sigma:.4f} barn")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        auto_download: bool = True,
        verbose: bool = False,
        archive_path: Optional[Union[str, Path]] = None,
        allow_builtin_approximations: bool = False,
    ):
        """
        Initialize IRDFF-II database interface.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory to cache downloaded data
        auto_download : bool
            Automatically download data if not cached
        verbose : bool
            Print status messages
        archive_path : str or Path, optional
            Explicit path to the IAEA ``IRDFF-II.tab.txt`` archive. Defaults to
            ``$FLUXFORGE_IRDFF_ARCHIVE`` or ``<cache_dir>/tab/IRDFF-II.tab.txt``.
        allow_builtin_approximations : bool
            Permit the sparse built-in cross-section approximations when a
            reaction is absent from the evaluated data. They omit resonance
            structure and must not be used for physical unfolding.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.auto_download = auto_download
        self.verbose = verbose
        self.allow_builtin_approximations = allow_builtin_approximations
        env_archive = os.environ.get("FLUXFORGE_IRDFF_ARCHIVE")
        self.archive_path = (
            Path(archive_path)
            if archive_path
            else Path(env_archive)
            if env_archive
            else self.cache_dir / "tab" / IRDFF_TAB_ARCHIVE_NAME
        )

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Loaded cross sections cache
        self._xs_cache: Dict[str, IRDFFCrossSection] = {}
        self._archive: Optional[Dict[str, np.ndarray]] = None
        self._archive_sha256 = ""
        self._abs_archive: Optional[Dict[str, np.ndarray]] = None
        self._abs_archive_sha256 = ""
        self._tab_data_loaded = False
        self._group_data_loaded = False

    def _download_file(self, url: str, local_path: Path) -> bool:
        """Download a file from URL."""
        if self.verbose:
            print(f"Downloading {url}...")

        require_network_access("IRDFF-II auto-download", url)

        try:
            req = Request(url, headers={"User-Agent": "FluxForge/1.0"})
            with urlopen(req, timeout=60) as response:
                data = response.read()
                local_path.write_bytes(data)
            if self.verbose:
                print(f"  Saved to {local_path}")
            return True
        except (URLError, HTTPError) as e:
            print(f"Error downloading {url}: {e}")
            return False

    def _ensure_tab_data(self) -> bool:
        """Ensure tabulated cross section data is available."""
        zip_path = self.cache_dir / "IRDFF-II_TAB.zip"
        extract_dir = self.cache_dir / "tab"

        if self.archive_path.exists():
            return True
        if extract_dir.exists() and any(extract_dir.glob("*.dat")):
            return True

        if not zip_path.exists():
            if not self.auto_download:
                print(
                    "IRDFF-II tabulated data not found. Set auto_download=True to download."
                )
                return False
            if not self._download_file(IRDFF_URLS["tab"], zip_path):
                return False

        # Extract ZIP
        if self.verbose:
            print(f"Extracting {zip_path}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)
            return True
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False

    def _load_archive(self) -> Optional[Dict[str, np.ndarray]]:
        """Parse the concatenated IAEA IRDFF-II tab archive once, if present."""
        if self._archive is None and self.archive_path.exists():
            self._archive = parse_irdff_tab_archive(self.archive_path)
            self._archive_sha256 = hashlib.sha256(
                self.archive_path.read_bytes()
            ).hexdigest()
        return self._archive

    def _get_archive_xs(self, reaction: str) -> Optional[IRDFFCrossSection]:
        """Look up a reaction in the evaluated archive with product checking."""
        archive = self._load_archive()
        if not archive:
            return None
        key = match_irdff_archive_key(reaction, archive.keys())
        if key is None:
            return None
        block = archive[key]
        energies, xs = block[:, 0], block[:, 1]
        nonzero = np.flatnonzero(xs > 0.0)
        target, _, product = reaction.partition(")")
        return IRDFFCrossSection(
            reaction=reaction,
            target=target.split("(")[0],
            product=product,
            mt_number=0,
            energies=energies,
            cross_sections=xs,
            uncertainties=block[:, 2],
            relative_unc=block[:, 3],
            # Last zero point before the first non-zero value, so the
            # linear ramp just above threshold is preserved.
            threshold_eV=(
                float(energies[nonzero[0] - 1]) if len(nonzero) and nonzero[0] > 0 else 0.0
            ),
            source="IRDFF-II",
            interpolation="lin-lin",
            evaluation_key=key,
            source_sha256=self._archive_sha256,
        )

    def get_cover_cross_section(
        self, material: str, quantity: str = "disap"
    ) -> Optional[IRDFFCrossSection]:
        """
        Evaluated cover-material cross section from the IRDFF-II abs archive.

        ``material`` is an archive element such as ``"Cd"``, ``"Gd"``, ``"B"``
        or ``"B-10"``; ``quantity`` is ``"tot"``, ``"elas"`` or ``"disap"``.
        """
        key = f"{material}(n,{quantity})"
        cache_key = f"abs:{key}"
        if cache_key in self._xs_cache:
            return self._xs_cache[cache_key]
        abs_path = self.archive_path.with_name(IRDFF_ABS_ARCHIVE_NAME)
        if not abs_path.exists() and not self._ensure_tab_data():
            return None
        if not abs_path.exists():
            return None
        if self._abs_archive is None:
            self._abs_archive = parse_irdff_tab_archive(abs_path)
            self._abs_archive_sha256 = hashlib.sha256(abs_path.read_bytes()).hexdigest()
        block = self._abs_archive.get(key)
        if block is None:
            return None
        xs = IRDFFCrossSection(
            reaction=key,
            target=material,
            product="",
            mt_number={"tot": 1, "elas": 2, "disap": 101}.get(quantity, 0),
            energies=block[:, 0],
            cross_sections=block[:, 1],
            uncertainties=np.zeros(len(block)),
            relative_unc=np.zeros(len(block)),
            source="IRDFF-II",
            interpolation="lin-lin",
            evaluation_key=key,
            source_sha256=self._abs_archive_sha256,
        )
        self._xs_cache[cache_key] = xs
        return xs

    def _parse_tab_file(self, filepath: Path) -> Optional[IRDFFCrossSection]:
        """
        Parse IRDFF-II 4-column tabulated file.

        Format: Energy(eV), CrossSection(barn), AbsUnc(barn), RelUnc(%)
        """
        try:
            data = np.loadtxt(filepath, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[1] < 2:
                return None

            energies = data[:, 0]  # eV
            cross_sections = data[:, 1]  # barn

            if data.shape[1] >= 3:
                uncertainties = data[:, 2]  # barn (absolute)
            else:
                uncertainties = cross_sections * 0.1  # Default 10%

            if data.shape[1] >= 4:
                relative_unc = data[:, 3]  # %
            else:
                relative_unc = np.where(
                    cross_sections > 0, 100 * uncertainties / cross_sections, 0.0
                )

            # Extract reaction info from filename
            filename = filepath.stem
            # Try to parse reaction string
            reaction = (
                filename.replace("_", "(").replace("-", ")")
                if "_" in filename
                else filename
            )

            # Find threshold (first non-zero cross section)
            nonzero = np.where(cross_sections > 1e-30)[0]
            threshold = energies[nonzero[0]] if len(nonzero) > 0 else 0.0

            return IRDFFCrossSection(
                reaction=reaction,
                target=reaction.split("(")[0] if "(" in reaction else reaction,
                product=reaction.split(")")[-1] if ")" in reaction else "",
                mt_number=0,  # Not in tabulated format
                energies=energies,
                cross_sections=cross_sections,
                uncertainties=uncertainties,
                relative_unc=relative_unc,
                threshold_eV=threshold,
                source="IRDFF-II",
            )

        except Exception as e:
            if self.verbose:
                print(f"Error parsing {filepath}: {e}")
            return None

    def list_reactions(self, category: Optional[str] = None) -> List[str]:
        """
        List available reactions.

        Parameters
        ----------
        category : str, optional
            Filter by category ('thermal', 'epithermal', 'fast', 'fission')

        Returns
        -------
        List[str]
            List of reaction identifiers
        """
        if category:
            if category in IRDFF_REACTIONS:
                return list(IRDFF_REACTIONS[category].keys())
            return []

        reactions = []
        for cat_reactions in IRDFF_REACTIONS.values():
            reactions.extend(cat_reactions.keys())
        return reactions

    def get_cross_section(
        self, reaction: str, force_reload: bool = False
    ) -> Optional[IRDFFCrossSection]:
        """
        Get cross section data for a reaction.

        Parameters
        ----------
        reaction : str
            Reaction identifier (e.g., 'Ti-46(n,p)Sc-46')
        force_reload : bool
            Force reload from file

        Returns
        -------
        IRDFFCrossSection or None
            Cross section data, or None if not found
        """
        if reaction in self._xs_cache and not force_reload:
            return self._xs_cache[reaction]

        # Evaluated IRDFF-II data first: the IAEA archive, then per-reaction files.
        if self._ensure_tab_data():
            xs = self._get_archive_xs(reaction) or self._search_tab_files(reaction)
            if xs is not None:
                self._xs_cache[reaction] = xs
                return xs

        # Sparse approximations only when explicitly allowed.
        if self.allow_builtin_approximations:
            xs = self._get_builtin_xs(reaction)
            if xs is not None:
                self._xs_cache[reaction] = xs
                return xs

        if self.verbose:
            print(f"Cross section not found: {reaction}")
        return None

    def _get_builtin_xs(self, reaction: str) -> Optional[IRDFFCrossSection]:
        """
        Get built-in cross section data (for common flux wire reactions).

        These are tabulated directly in the code for immediate availability
        without requiring internet access.
        """
        # Sparse hand-entered approximations (roughly 1/v for captures, with no
        # resolved resonances). Not evaluated IRDFF-II data; opt-in only.
        builtin_xs = self._get_builtin_cross_section_data()

        if reaction in builtin_xs:
            data = builtin_xs[reaction]
            return IRDFFCrossSection(
                reaction=reaction,
                target=data["target"],
                product=data["product"],
                mt_number=data["mt"],
                energies=np.array(data["energies"]),
                cross_sections=np.array(data["xs"]),
                uncertainties=np.array(data["xs"]) * 0.05,  # 5% default uncertainty
                relative_unc=np.ones(len(data["xs"])) * 5.0,
                threshold_eV=data["threshold"],
                source=BUILTIN_APPROXIMATION_SOURCE,
            )
        return None

    def _get_builtin_cross_section_data(self) -> Dict[str, Any]:
        """
        Return built-in cross section data for common reactions.

        Energy in eV, cross section in barns.
        Data points from IRDFF-II for key flux wire reactions.
        """
        # Representative energy grid points (eV)
        thermal_grid = [
            1e-5,
            0.0253,
            0.1,
            0.5,
            1,
            10,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
            5e6,
            10e6,
            14e6,
            20e6,
        ]
        fast_grid = [
            1e5,
            5e5,
            1e6,
            2e6,
            3e6,
            4e6,
            5e6,
            6e6,
            8e6,
            10e6,
            12e6,
            14e6,
            16e6,
            18e6,
            20e6,
        ]

        return {
            # Thermal capture reactions
            "Co-59(n,g)Co-60": {
                "target": "Co-59",
                "product": "Co-60",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    2095,
                    37.18,
                    18.9,
                    8.5,
                    6.0,
                    1.9,
                    0.6,
                    0.2,
                    0.05,
                    0.01,
                    0.003,
                    0.001,
                    0.0005,
                ],
            },
            "Sc-45(n,g)Sc-46": {
                "target": "Sc-45",
                "product": "Sc-46",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    140,
                    27.2,
                    13.7,
                    6.1,
                    4.3,
                    1.4,
                    0.44,
                    0.14,
                    0.02,
                    0.005,
                    0.002,
                    0.001,
                    0.0003,
                ],
            },
            "Cu-63(n,g)Cu-64": {
                "target": "Cu-63",
                "product": "Cu-64",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    24,
                    4.5,
                    2.3,
                    1.0,
                    0.7,
                    0.22,
                    0.07,
                    0.02,
                    0.006,
                    0.002,
                    0.0007,
                    0.0002,
                    0.0001,
                ],
            },
            "Fe-58(n,g)Fe-59": {
                "target": "Fe-58",
                "product": "Fe-59",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    6.8,
                    1.28,
                    0.65,
                    0.29,
                    0.2,
                    0.065,
                    0.02,
                    0.006,
                    0.002,
                    0.0006,
                    0.0002,
                    0.0001,
                    0.00005,
                ],
            },
            "In-115(n,g)In-116m": {
                "target": "In-115",
                "product": "In-116m",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    1.457,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                ],
                "xs": [
                    1015,
                    162,
                    81.5,
                    36.4,
                    25.7,
                    30000,
                    6.4,
                    2.0,
                    0.6,
                    0.05,
                    0.01,
                    0.003,
                ],  # 1.457 eV resonance
            },
            "Au-197(n,g)Au-198": {
                "target": "Au-197",
                "product": "Au-198",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    4.9,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    558,
                    98.65,
                    49.5,
                    22.1,
                    15.6,
                    30000,
                    5.0,
                    1.5,
                    0.5,
                    0.08,
                    0.02,
                    0.005,
                    0.002,
                    0.001,
                ],  # 4.9 eV resonance
            },
            "Na-23(n,g)Na-24": {
                "target": "Na-23",
                "product": "Na-24",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    3.0,
                    0.530,
                    0.27,
                    0.12,
                    0.085,
                    0.027,
                    0.008,
                    0.003,
                    0.0006,
                    0.0002,
                    0.00007,
                    0.00003,
                    0.00001,
                ],
            },
            # Fast threshold reactions (n,p)
            "Ti-46(n,p)Sc-46": {
                "target": "Ti-46",
                "product": "Sc-46",
                "mt": 103,
                "threshold": 1.62e6,
                "energies": [
                    1.5e6,
                    2e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.003,
                    0.03,
                    0.1,
                    0.18,
                    0.24,
                    0.31,
                    0.34,
                    0.33,
                    0.30,
                    0.26,
                    0.22,
                    0.19,
                ],
            },
            "Ti-47(n,p)Sc-47": {
                "target": "Ti-47",
                "product": "Sc-47",
                "mt": 103,
                "threshold": 0.22e6,
                "energies": [
                    0.2e6,
                    0.5e6,
                    1e6,
                    2e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.005,
                    0.02,
                    0.05,
                    0.08,
                    0.10,
                    0.11,
                    0.12,
                    0.12,
                    0.11,
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                ],
            },
            "Ti-48(n,p)Sc-48": {
                "target": "Ti-48",
                "product": "Sc-48",
                "mt": 103,
                "threshold": 3.35e6,
                "energies": [
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    7e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.001,
                    0.008,
                    0.022,
                    0.04,
                    0.052,
                    0.065,
                    0.068,
                    0.064,
                    0.058,
                    0.050,
                    0.043,
                ],
            },
            "Fe-56(n,p)Mn-56": {
                "target": "Fe-56",
                "product": "Mn-56",
                "mt": 103,
                "threshold": 2.97e6,
                "energies": [
                    2.5e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    7e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.0005,
                    0.005,
                    0.02,
                    0.045,
                    0.07,
                    0.09,
                    0.11,
                    0.115,
                    0.110,
                    0.10,
                    0.09,
                    0.08,
                ],
            },
            "Ni-58(n,p)Co-58": {
                "target": "Ni-58",
                "product": "Co-58",
                "mt": 103,
                "threshold": 0.40e6,
                "energies": [
                    0.3e6,
                    0.5e6,
                    1e6,
                    2e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.01,
                    0.08,
                    0.22,
                    0.35,
                    0.45,
                    0.52,
                    0.55,
                    0.58,
                    0.55,
                    0.50,
                    0.44,
                    0.38,
                    0.32,
                    0.27,
                ],
            },
            "Al-27(n,a)Na-24": {
                "target": "Al-27",
                "product": "Na-24",
                "mt": 107,
                "threshold": 3.25e6,
                "energies": [
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    7e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.001,
                    0.01,
                    0.04,
                    0.08,
                    0.105,
                    0.12,
                    0.125,
                    0.122,
                    0.115,
                    0.105,
                    0.095,
                ],
            },
            "Fe-54(n,a)Cr-51": {
                "target": "Fe-54",
                "product": "Cr-51",
                "mt": 107,
                "threshold": 0.84e6,
                "energies": [
                    0.8e6,
                    1e6,
                    2e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.0005,
                    0.01,
                    0.03,
                    0.05,
                    0.065,
                    0.075,
                    0.085,
                    0.088,
                    0.085,
                    0.080,
                    0.072,
                    0.064,
                    0.056,
                ],
            },
            "Mn-55(n,g)Mn-56": {
                "target": "Mn-55",
                "product": "Mn-56",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    70,
                    13.3,
                    6.7,
                    3.0,
                    2.1,
                    0.67,
                    0.21,
                    0.07,
                    0.01,
                    0.003,
                    0.001,
                    0.0005,
                    0.0002,
                ],
            },
            "W-186(n,g)W-187": {
                "target": "W-186",
                "product": "W-187",
                "mt": 102,
                "threshold": 0.0,
                "energies": [
                    1e-5,
                    0.0253,
                    0.1,
                    0.5,
                    1,
                    10,
                    100,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    5e6,
                    10e6,
                ],
                "xs": [
                    200,
                    37.9,
                    19.1,
                    8.5,
                    6.0,
                    1.9,
                    0.6,
                    0.19,
                    0.03,
                    0.008,
                    0.003,
                    0.001,
                    0.0004,
                ],
            },
            "In-115(n,n')In-115m": {
                "target": "In-115",
                "product": "In-115m",
                "mt": 4,
                "threshold": 0.34e6,
                "energies": [
                    0.3e6,
                    0.5e6,
                    1e6,
                    2e6,
                    3e6,
                    4e6,
                    5e6,
                    6e6,
                    8e6,
                    10e6,
                    12e6,
                    14e6,
                    16e6,
                    18e6,
                    20e6,
                ],
                "xs": [
                    0,
                    0.08,
                    0.18,
                    0.30,
                    0.34,
                    0.36,
                    0.35,
                    0.33,
                    0.28,
                    0.23,
                    0.19,
                    0.16,
                    0.13,
                    0.11,
                    0.09,
                ],
            },
        }

    def _search_tab_files(self, reaction: str) -> Optional[IRDFFCrossSection]:
        """Search for a reaction in downloaded tabulated files."""
        extract_dir = self.cache_dir / "tab"
        if not extract_dir.exists():
            return None

        # Normalize reaction string for matching
        normalized = reaction.replace(" ", "").lower()

        for filepath in extract_dir.rglob("*.dat"):
            filename_norm = filepath.stem.replace("_", "").replace("-", "").lower()
            if normalized in filename_norm or filename_norm in normalized:
                return self._parse_tab_file(filepath)

        # Also try .txt files (never the concatenated archive or its listings)
        for filepath in extract_dir.rglob("*.txt"):
            if filepath.name.startswith("IRDFF-II."):
                continue
            filename_norm = filepath.stem.replace("_", "").replace("-", "").lower()
            if normalized in filename_norm or filename_norm in normalized:
                return self._parse_tab_file(filepath)

        return None

    def get_energy_grid(self, grid_type: str = "sand725") -> np.ndarray:
        """
        Get IRDFF-II standard energy grid.

        Parameters
        ----------
        grid_type : str
            'sand725' (725-group SAND-II) or 'mcnp640' (640-group MCNP)

        Returns
        -------
        np.ndarray
            Energy group boundaries in eV
        """
        if grid_type == "sand725":
            return self._get_sand725_grid()
        elif grid_type == "mcnp640":
            return self._get_mcnp640_grid()
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")

    def _get_sand725_grid(self) -> np.ndarray:
        """
        Get SAND-II 725-group energy structure.

        Standard lethargy width: Δu = 0.1 from 1e-5 eV to 60 MeV
        """
        # Generate lethargy-based grid
        e_min = 1e-5  # eV
        e_max = 60e6  # eV (60 MeV)
        n_groups = 725

        # Lethargy grid
        u_max = np.log(e_max / e_min)
        du = u_max / n_groups
        u = np.linspace(0, u_max, n_groups + 1)

        return e_min * np.exp(u)  # eV

    def _get_mcnp640_grid(self) -> np.ndarray:
        """
        Get MCNP 640-group energy structure (vitamin-J style).
        """
        # MCNP 640-group structure from thermal to 20 MeV
        # Lethargy-based with thermal fine structure
        e_min = 1e-5  # eV
        e_max = 20e6  # eV (20 MeV)
        n_groups = 640

        u_max = np.log(e_max / e_min)
        u = np.linspace(0, u_max, n_groups + 1)

        return e_min * np.exp(u)  # eV


# =============================================================================
# Energy Group Structures for Flux Wire Analysis
# =============================================================================


def get_flux_wire_energy_groups(
    n_thermal: int = 30,
    n_epithermal: int = 100,
    n_fast: int = 120,
    e_thermal_max: float = 0.55,  # Cd cutoff (eV)
    e_epithermal_max: float = 1e5,  # 100 keV
    e_max: float = 20e6,  # 20 MeV
) -> np.ndarray:
    """
    Create energy group structure optimized for flux wire analysis.

    This creates a multi-region energy grid with:
    - Fine thermal groups below Cd cutoff (for capture reactions)
    - Epithermal groups with resonance structure
    - Fast groups for threshold reactions

    Parameters
    ----------
    n_thermal : int
        Number of thermal groups (E < Cd cutoff)
    n_epithermal : int
        Number of epithermal groups (Cd cutoff to 100 keV)
    n_fast : int
        Number of fast groups (100 keV to 20 MeV)
    e_thermal_max : float
        Thermal/epithermal boundary (eV), default Cd cutoff
    e_epithermal_max : float
        Epithermal/fast boundary (eV)
    e_max : float
        Maximum energy (eV)

    Returns
    -------
    np.ndarray
        Energy group boundaries in eV
    """
    e_min = 1e-5  # 0.01 meV

    # Thermal: linear spacing (for Maxwell-Boltzmann)
    thermal_edges = np.linspace(e_min, e_thermal_max, n_thermal + 1)

    # Epithermal: logarithmic spacing (for 1/E spectrum)
    epithermal_edges = np.geomspace(e_thermal_max, e_epithermal_max, n_epithermal + 1)

    # Fast: logarithmic spacing
    fast_edges = np.geomspace(e_epithermal_max, e_max, n_fast + 1)

    # Combine, removing duplicates at boundaries
    edges = np.concatenate([thermal_edges[:-1], epithermal_edges[:-1], fast_edges])

    return edges


def get_activation_energy_groups() -> np.ndarray:
    """
    Get energy group structure suitable for activation analysis.

    Based on VITAMIN-J 175-group structure extended for dosimetry.
    Includes fine structure in regions important for:
    - Thermal capture
    - Resonance integrals
    - Threshold reactions

    Returns
    -------
    np.ndarray
        Energy group boundaries in eV
    """
    # Key threshold energies to include (eV)
    key_energies = [
        1e-5,  # Lower bound
        0.0253,  # Thermal (25.3 meV)
        0.1,  # 100 meV
        0.55,  # Cd cutoff
        1.0,  # 1 eV
        1.457,  # In-115 resonance
        4.28,  # Au-197 resonance
        10,  # 10 eV
        100,  # 100 eV
        1e3,  # 1 keV
        1e4,  # 10 keV
        1e5,  # 100 keV
        2.2e5,  # Ti-47(n,p) threshold
        4e5,  # Ni-58(n,p) threshold
        8.4e5,  # Fe-54(n,α) threshold
        9.6e5,  # S-32(n,p) threshold
        1.53e6,  # V-51(n,p) threshold
        1.62e6,  # Ti-46(n,p) threshold
        1.9e6,  # Al-27(n,p) threshold
        2.97e6,  # Fe-56(n,p), Cr-52(n,p) threshold
        3.25e6,  # Al-27(n,α) threshold
        3.35e6,  # Ti-48(n,p) threshold
        5e6,  # 5 MeV
        10e6,  # 10 MeV
        14.1e6,  # D-T fusion peak
        20e6,  # 20 MeV
    ]

    # Create fine grid around key energies
    edges = []
    for i in range(len(key_energies) - 1):
        e_lo = key_energies[i]
        e_hi = key_energies[i + 1]

        # More groups in resonance region
        if e_lo < 1e3:
            n_groups = 10
        elif e_lo < 1e5:
            n_groups = 8
        else:
            n_groups = 5

        edges.extend(np.geomspace(e_lo, e_hi, n_groups + 1)[:-1].tolist())

    edges.append(key_energies[-1])

    return np.array(edges)


# =============================================================================
# Response Matrix Construction
# =============================================================================


def build_response_matrix(
    reactions: List[str],
    energy_edges: np.ndarray,
    db: Optional[IRDFFDatabase] = None,
    verbose: bool = False,
    on_missing: str = "raise",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build response matrix for spectrum unfolding.

    The response matrix R has dimensions (n_reactions, n_groups) where:
    R[i,g] = group-averaged cross section for reaction i in group g

    Parameters
    ----------
    reactions : List[str]
        List of reaction identifiers
    energy_edges : np.ndarray
        Energy group boundaries in eV
    db : IRDFFDatabase, optional
        Database instance (will create if None)
    verbose : bool
        Print status messages
    on_missing : {"raise", "skip"}
        ``"raise"`` (default) raises :class:`MissingCrossSectionError` naming
        every reaction without data. ``"skip"`` drops those rows; callers must
        then align measurements with the returned reaction names.

    Returns
    -------
    response_matrix : np.ndarray
        Response matrix (n_reactions x n_groups) in barns
    reaction_names : List[str]
        Names of reactions (may be reordered)
    uncertainties : np.ndarray
        Uncertainty matrix (same shape)
    """
    if on_missing not in {"raise", "skip"}:
        raise ValueError("on_missing must be 'raise' or 'skip'")
    if db is None:
        db = IRDFFDatabase(verbose=verbose)

    n_groups = len(energy_edges) - 1
    n_reactions = len(reactions)

    response = np.zeros((n_reactions, n_groups))
    uncertainties = np.zeros((n_reactions, n_groups))
    valid_reactions = []

    missing = sorted({rxn for rxn in reactions if db.get_cross_section(rxn) is None})
    if missing and on_missing == "raise":
        raise MissingCrossSectionError(
            "No evaluated cross section for: " + ", ".join(missing)
        )

    for i, rxn in enumerate(reactions):
        xs = db.get_cross_section(rxn)
        if xs is None:
            if verbose:
                print(f"Warning: No cross section for {rxn}")
            continue

        # Collapse to group structure
        group_xs, group_unc = xs.to_group_structure(energy_edges)

        response[i, :] = group_xs
        uncertainties[i, :] = group_unc
        valid_reactions.append(rxn)

        if verbose:
            print(f"  {rxn}: max σ = {group_xs.max():.3e} barn")

    # Remove empty rows
    valid_idx = [i for i, rxn in enumerate(reactions) if rxn in valid_reactions]
    response = response[valid_idx, :]
    uncertainties = uncertainties[valid_idx, :]

    return response, valid_reactions, uncertainties


# =============================================================================
# Convenience Functions
# =============================================================================


def get_irdff_database(
    cache_dir: Optional[Path] = None, auto_download: bool = True
) -> IRDFFDatabase:
    """
    Get a configured IRDFF-II database instance.

    Parameters
    ----------
    cache_dir : Path, optional
        Custom cache directory
    auto_download : bool
        Automatically download data if needed

    Returns
    -------
    IRDFFDatabase
        Configured database instance
    """
    return IRDFFDatabase(cache_dir=cache_dir, auto_download=auto_download)


def list_dosimetry_reactions(category: Optional[str] = None) -> List[str]:
    """
    List available dosimetry reactions from IRDFF-II.

    Parameters
    ----------
    category : str, optional
        Filter by 'thermal', 'epithermal', 'fast', or 'fission'

    Returns
    -------
    List[str]
        Reaction identifiers
    """
    db = IRDFFDatabase()
    return db.list_reactions(category)


def get_cross_section(reaction: str) -> Optional[IRDFFCrossSection]:
    """
    Get IRDFF-II cross section for a reaction.

    Parameters
    ----------
    reaction : str
        Reaction identifier (e.g., 'Ti-46(n,p)Sc-46')

    Returns
    -------
    IRDFFCrossSection or None
        Cross section data
    """
    db = IRDFFDatabase()
    return db.get_cross_section(reaction)
