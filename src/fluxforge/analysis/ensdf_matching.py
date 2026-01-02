"""
ENSDF/paceENSDF Integration for Isotope Matching

Provides gamma-line database building and three-tier isotope matching
using the paceENSDF library (if available) or fallback data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Try to import paceENSDF
try:
    import paceensdf as pe
    HAS_PACEENSDF = True
except ImportError:
    pe = None
    HAS_PACEENSDF = False


# ============================================================================
# Isotope Name Normalization
# ============================================================================

_ISO_PATTERNS = (
    re.compile(r"^(?P<el>[A-Za-z]{1,3})[-\s]?(?P<mass>\d{1,3})(?P<meta>m\d*|m)?$", re.IGNORECASE),
    re.compile(r"^(?P<mass>\d{1,3})(?P<meta>m\d*|m)?[-\s]?(?P<el>[A-Za-z]{1,3})$", re.IGNORECASE),
)


def normalize_isotope_label(label: str) -> str:
    """
    Normalize isotope label to canonical format (e.g., 'Fe59', 'Co60M').
    
    Handles various input formats:
    - Fe-59, Fe59, 59Fe, 59-Fe
    - Co-60m, Co60m, Co60M
    
    Parameters
    ----------
    label : str
        Input isotope label
    
    Returns
    -------
    str
        Normalized label like 'Fe59' or 'Co60M'
    """
    s = str(label).strip()
    s = s.replace('_', '').replace('/', '').replace('.', '')
    s = s.replace('(', '').replace(')', '')
    s = s.strip()
    
    for pat in _ISO_PATTERNS:
        m = pat.match(s)
        if m:
            el = m.group('el').capitalize()
            mass = int(m.group('mass'))
            meta = m.group('meta') or ''
            meta_norm = 'M' if meta else ''
            return f"{el}{mass}{meta_norm}"
    
    return s.upper()


def element_from_isotope(label: str) -> Optional[str]:
    """
    Extract element symbol from isotope label.
    
    Parameters
    ----------
    label : str
        Isotope label
    
    Returns
    -------
    str or None
        Element symbol (e.g., 'Fe', 'Co') or None
    """
    canon = normalize_isotope_label(label)
    m = re.match(r"^([A-Z][a-z]?|[A-Z]{2,3})\d", canon)
    if not m:
        return None
    
    el_raw = m.group(1)
    if len(el_raw) == 1:
        return el_raw.upper()
    if len(el_raw) == 2:
        return el_raw[0].upper() + el_raw[1].lower()
    return el_raw[0].upper() + el_raw[1:].lower()


# ============================================================================
# Gamma Line Database
# ============================================================================

@dataclass
class GammaLine:
    """A gamma line from nuclear decay data."""
    
    isotope: str          # Canonical isotope name (e.g., 'Co60')
    isotope_raw: str      # Original label from database
    energy_keV: float     # Gamma energy
    intensity: float      # Relative intensity (0-100%)
    element: Optional[str] = None  # Element symbol
    
    def __repr__(self) -> str:
        return f"GammaLine({self.isotope}, {self.energy_keV:.1f} keV, {self.intensity:.1f}%)"


@dataclass
class GammaDatabase:
    """Database of gamma lines for isotope matching."""
    
    lines: List[GammaLine]
    source: str = ""  # 'paceensdf', 'fallback', etc.
    
    def __len__(self) -> int:
        return len(self.lines)
    
    @property
    def energies(self) -> np.ndarray:
        """Array of all gamma energies."""
        return np.array([line.energy_keV for line in self.lines])
    
    @property
    def isotopes(self) -> List[str]:
        """Unique isotope list."""
        return list(set(line.isotope for line in self.lines))
    
    def filter_by_isotopes(self, isotopes: Sequence[str]) -> 'GammaDatabase':
        """Filter to specific isotopes."""
        isotope_set = {normalize_isotope_label(iso) for iso in isotopes}
        filtered = [line for line in self.lines if line.isotope in isotope_set]
        return GammaDatabase(filtered, self.source)
    
    def filter_by_elements(self, elements: Sequence[str]) -> 'GammaDatabase':
        """Filter to specific elements."""
        element_set = {el.capitalize() for el in elements}
        filtered = [line for line in self.lines if line.element in element_set]
        return GammaDatabase(filtered, self.source)
    
    def filter_by_intensity(self, min_intensity: float) -> 'GammaDatabase':
        """Filter to lines above minimum intensity."""
        filtered = [line for line in self.lines if line.intensity >= min_intensity]
        return GammaDatabase(filtered, self.source)


def _walk_ensdf_node(
    node,
    iso_raw: str,
    iso_canon: str,
    elem: Optional[str],
    rows: List[GammaLine],
) -> None:
    """Recursively extract gamma lines from ENSDF data structure."""
    if isinstance(node, dict):
        keys = {k.lower(): k for k in node}
        e_key = next((k for k in keys if 'energy' in k or k in ('e', 'eg')), None)
        i_key = next((k for k in keys if 'intens' in k or k in ('ri', 'i')), None)
        
        if e_key and i_key:
            try:
                energy = float(node[keys[e_key]])
                intensity = float(node[keys[i_key]])
                if energy > 0 and intensity > 0:
                    rows.append(GammaLine(
                        isotope=iso_canon,
                        isotope_raw=iso_raw,
                        energy_keV=energy,
                        intensity=intensity,
                        element=elem,
                    ))
            except (ValueError, TypeError):
                pass
        
        for v in node.values():
            _walk_ensdf_node(v, iso_raw, iso_canon, elem, rows)
    
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_ensdf_node(v, iso_raw, iso_canon, elem, rows)


def build_gamma_database_paceensdf(
    elements: Optional[Sequence[str]] = None,
) -> Optional[GammaDatabase]:
    """
    Build gamma database from paceENSDF.
    
    Parameters
    ----------
    elements : sequence of str, optional
        Filter to specific elements. If None, include all.
    
    Returns
    -------
    GammaDatabase or None
        Database of gamma lines, or None if paceENSDF unavailable
    """
    if not HAS_PACEENSDF:
        return None
    
    ensdf = pe.ENSDF()
    nuclides = ensdf.load_ensdf()
    
    # Element filter
    keep = None
    if elements is not None:
        keep = {el.strip().lower() for el in elements}
    
    rows: List[GammaLine] = []
    
    for nucl in nuclides:
        iso = nucl.get('parentID', 'Unknown')
        iso_raw = iso.strip() if isinstance(iso, str) else str(iso).strip()
        iso_canon = normalize_isotope_label(iso_raw)
        elem = element_from_isotope(iso_raw)
        
        if keep is not None:
            if elem is None or elem.lower() not in keep:
                continue
        
        _walk_ensdf_node(nucl, iso_raw, iso_canon, elem, rows)
    
    if not rows:
        return None
    
    # Remove duplicates (same energy, isotope, intensity)
    seen = set()
    unique = []
    for line in rows:
        key = (line.isotope, round(line.energy_keV, 1), round(line.intensity, 1))
        if key not in seen:
            seen.add(key)
            unique.append(line)
    
    return GammaDatabase(unique, source='paceensdf')


# ============================================================================
# Fallback Database for Common Activation Products
# ============================================================================

# Major gamma lines for common activation products (from NNDC/ENSDF)
FALLBACK_GAMMA_LINES = [
    # Iron activation products
    GammaLine('Fe59', 'Fe-59', 1099.25, 56.5, 'Fe'),
    GammaLine('Fe59', 'Fe-59', 1291.60, 43.2, 'Fe'),
    GammaLine('Mn54', 'Mn-54', 834.85, 99.98, 'Mn'),
    GammaLine('Mn56', 'Mn-56', 846.76, 98.9, 'Mn'),
    GammaLine('Mn56', 'Mn-56', 1810.73, 27.2, 'Mn'),
    GammaLine('Co60', 'Co-60', 1173.23, 99.85, 'Co'),
    GammaLine('Co60', 'Co-60', 1332.49, 99.98, 'Co'),
    GammaLine('Co58', 'Co-58', 810.76, 99.45, 'Co'),
    GammaLine('Cr51', 'Cr-51', 320.08, 9.91, 'Cr'),
    
    # Tungsten activation
    GammaLine('W187', 'W-187', 479.55, 21.8, 'W'),
    GammaLine('W187', 'W-187', 685.81, 27.3, 'W'),
    GammaLine('W185', 'W-185', 125.36, 0.019, 'W'),
    GammaLine('Re186', 'Re-186', 137.16, 9.47, 'Re'),
    GammaLine('Re188', 'Re-188', 155.04, 15.6, 'Re'),
    
    # Other common products
    GammaLine('Na24', 'Na-24', 1368.63, 99.99, 'Na'),
    GammaLine('Na24', 'Na-24', 2754.03, 99.86, 'Na'),
    GammaLine('Zn65', 'Zn-65', 1115.55, 50.04, 'Zn'),
    GammaLine('Sc46', 'Sc-46', 889.28, 99.98, 'Sc'),
    GammaLine('Sc46', 'Sc-46', 1120.55, 99.99, 'Sc'),
    GammaLine('Ta182', 'Ta-182', 1121.30, 35.2, 'Ta'),
    GammaLine('Ta182', 'Ta-182', 1189.05, 16.5, 'Ta'),
    GammaLine('Ta182', 'Ta-182', 1221.41, 27.2, 'Ta'),
    GammaLine('Tb160', 'Tb-160', 298.58, 26.1, 'Tb'),
    GammaLine('Tb160', 'Tb-160', 879.38, 30.1, 'Tb'),
    
    # Vanadium/Aluminum
    GammaLine('V48', 'V-48', 983.52, 99.98, 'V'),
    GammaLine('V48', 'V-48', 1312.11, 97.5, 'V'),
    GammaLine('Al28', 'Al-28', 1778.99, 100.0, 'Al'),
    
    # Annihilation
    GammaLine('Annihil', 'Annihilation', 511.0, 100.0, None),
]


def build_fallback_database(
    elements: Optional[Sequence[str]] = None,
) -> GammaDatabase:
    """
    Build database from fallback hardcoded data.
    
    Parameters
    ----------
    elements : sequence of str, optional
        Filter to specific elements
    
    Returns
    -------
    GammaDatabase
        Fallback gamma database
    """
    lines = FALLBACK_GAMMA_LINES.copy()
    
    if elements is not None:
        element_set = {el.capitalize() for el in elements}
        lines = [l for l in lines if l.element in element_set or l.element is None]
    
    return GammaDatabase(lines, source='fallback')


def build_gamma_database(
    elements: Optional[Sequence[str]] = None,
    use_fallback: bool = True,
) -> GammaDatabase:
    """
    Build gamma database, preferring paceENSDF if available.
    
    Parameters
    ----------
    elements : sequence of str, optional
        Filter to specific elements
    use_fallback : bool
        If True and paceENSDF unavailable, use fallback data
    
    Returns
    -------
    GammaDatabase
        Gamma line database
    """
    db = build_gamma_database_paceensdf(elements)
    if db is not None and len(db) > 0:
        return db
    
    if use_fallback:
        return build_fallback_database(elements)
    
    return GammaDatabase([], source='empty')


# ============================================================================
# Three-Tier Isotope Matching
# ============================================================================

@dataclass
class IsotopeMatch:
    """Result of matching a peak to gamma database."""
    
    observed_energy: float     # Observed peak energy (keV)
    observed_amplitude: float  # Peak amplitude
    matched_isotope: str       # Best-match isotope
    matched_energy: float      # Gamma line energy (keV)
    matched_intensity: float   # Gamma line intensity (%)
    delta_keV: float           # Energy difference
    match_tier: int            # Which tier matched (1, 2, or 3)
    match_source: str          # 'tier1', 'tier2', 'tier3'
    candidates: str            # Formatted string of top candidates
    
    @property
    def is_matched(self) -> bool:
        return self.matched_isotope != ''


def _match_peak_to_database(
    energy: float,
    database: GammaDatabase,
    tolerance_keV: float,
    top_n: int = 3,
) -> Optional[Dict]:
    """Match a single peak energy to a gamma database."""
    if len(database) == 0:
        return None
    
    energies = database.energies
    diffs = np.abs(energies - energy)
    mask = diffs <= tolerance_keV
    
    if not mask.any():
        return None
    
    # Get candidates sorted by intensity
    candidates = [(database.lines[i], diffs[i]) for i in np.where(mask)[0]]
    candidates.sort(key=lambda x: -x[0].intensity)
    candidates = candidates[:top_n]
    
    best = candidates[0][0]
    
    # Format candidate string
    cand_strs = [
        f"{c.isotope} {c.energy_keV:.1f} keV ({c.intensity:.1f}%)"
        for c, _ in candidates
    ]
    
    return {
        'isotope': best.isotope,
        'isotope_raw': best.isotope_raw,
        'energy': best.energy_keV,
        'intensity': best.intensity,
        'delta': candidates[0][1],
        'candidates': '; '.join(cand_strs),
    }


def match_peaks_three_tier(
    peaks: List,  # DetectedPeak or similar with energy_keV, amplitude
    tier1_db: Optional[GammaDatabase],
    tier2_db: Optional[GammaDatabase],
    tier3_db: Optional[GammaDatabase],
    tol_tier1: float = 3.0,
    tol_tier2: float = 3.0,
    tol_tier3: float = 2.0,
    top_n: int = 3,
) -> List[IsotopeMatch]:
    """
    Match peaks using three-tier priority system.
    
    Tier 1: Specific high-priority isotopes (e.g., known activation products)
    Tier 2: Preferred elements (e.g., sample composition elements)
    Tier 3: Full ENSDF catalog
    
    Parameters
    ----------
    peaks : list
        Detected peaks with energy_keV and amplitude attributes
    tier1_db : GammaDatabase or None
        High-priority isotope database
    tier2_db : GammaDatabase or None
        Preferred element database
    tier3_db : GammaDatabase or None
        Full catalog database
    tol_tier1 : float
        Tier 1 matching tolerance (keV)
    tol_tier2 : float
        Tier 2 matching tolerance (keV)
    tol_tier3 : float
        Tier 3 matching tolerance (keV)
    top_n : int
        Number of candidates to report
    
    Returns
    -------
    list of IsotopeMatch
        Matching results for each peak
    """
    matches = []
    
    for peak in peaks:
        energy = peak.energy_keV
        amplitude = getattr(peak, 'amplitude', 0.0)
        
        hit = None
        tier = 0
        source = ''
        tol_used = 0.0
        
        # Try tier 1
        if tier1_db is not None and len(tier1_db) > 0:
            hit = _match_peak_to_database(energy, tier1_db, tol_tier1, top_n)
            if hit:
                tier = 1
                source = 'tier1'
                tol_used = tol_tier1
        
        # Try tier 2
        if hit is None and tier2_db is not None and len(tier2_db) > 0:
            hit = _match_peak_to_database(energy, tier2_db, tol_tier2, top_n)
            if hit:
                tier = 2
                source = 'tier2'
                tol_used = tol_tier2
        
        # Try tier 3
        if hit is None and tier3_db is not None and len(tier3_db) > 0:
            hit = _match_peak_to_database(energy, tier3_db, tol_tier3, top_n)
            if hit:
                tier = 3
                source = 'tier3'
                tol_used = tol_tier3
        
        if hit is None:
            matches.append(IsotopeMatch(
                observed_energy=energy,
                observed_amplitude=amplitude,
                matched_isotope='',
                matched_energy=0.0,
                matched_intensity=0.0,
                delta_keV=0.0,
                match_tier=0,
                match_source='unmatched',
                candidates='',
            ))
        else:
            matches.append(IsotopeMatch(
                observed_energy=energy,
                observed_amplitude=amplitude,
                matched_isotope=hit['isotope'],
                matched_energy=hit['energy'],
                matched_intensity=hit['intensity'],
                delta_keV=hit['delta'],
                match_tier=tier,
                match_source=source,
                candidates=hit['candidates'],
            ))
    
    return matches


def build_tier1_isotopes(
    elements: Sequence[str],
    min_intensity: float = 10.0,
) -> List[str]:
    """
    Build tier-1 isotope list from high-intensity lines of given elements.
    
    Parameters
    ----------
    elements : sequence of str
        Element symbols
    min_intensity : float
        Minimum intensity threshold (%)
    
    Returns
    -------
    list of str
        Tier-1 isotope names
    """
    db = build_gamma_database(elements)
    high_intensity = db.filter_by_intensity(min_intensity)
    return list(set(line.isotope for line in high_intensity.lines))


# ============================================================================
# Convenience Functions
# ============================================================================

def get_data_source() -> str:
    """Return the name of the available gamma data source."""
    if HAS_PACEENSDF:
        return 'paceENSDF'
    return 'fallback'


def create_matching_databases(
    tier1_isotopes: Optional[Sequence[str]] = None,
    tier2_elements: Optional[Sequence[str]] = None,
    include_tier3: bool = True,
) -> Tuple[Optional[GammaDatabase], Optional[GammaDatabase], Optional[GammaDatabase]]:
    """
    Create the three database tiers for isotope matching.
    
    Parameters
    ----------
    tier1_isotopes : sequence of str, optional
        Specific isotopes for tier 1
    tier2_elements : sequence of str, optional
        Elements for tier 2
    include_tier3 : bool
        Whether to build full tier 3 database
    
    Returns
    -------
    tuple of (GammaDatabase or None, GammaDatabase or None, GammaDatabase or None)
        (tier1_db, tier2_db, tier3_db)
    """
    tier1_db = None
    tier2_db = None
    tier3_db = None
    
    full_db = build_gamma_database()
    
    if tier1_isotopes:
        tier1_db = full_db.filter_by_isotopes(tier1_isotopes)
    
    if tier2_elements:
        tier2_db = full_db.filter_by_elements(tier2_elements)
    
    if include_tier3:
        tier3_db = full_db
    
    return tier1_db, tier2_db, tier3_db
