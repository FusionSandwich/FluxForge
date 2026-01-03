#!/usr/bin/env python3
"""
Batch γ-spectrum processing for RAFM samples.

- Scans a spectra directory for Genie-2000 ASCII exports (*.ASC).
- Extracts calibration (A/B/C), live time, channel counts.
- Applies detector efficiency model from a LabSOCS-style CSV.
- Detects peaks (on ε-corrected counts) and optionally labels with ENSDF lines
  using paceENSDF if available.
- Saves per-file plots and C vs N comparison plots for each time point.
- Emits a combined CSV of detected/matched peaks.

Key features:
- Less-sensitive segmented peak finding (higher height/prominence per region).
- Three-tier isotope matching:
    Tier1: specific isotopes
    Tier2: preferred elements
    Tier3: full ENSDF catalog
- Default tier tolerances: 3 keV / 3 keV / 2 keV.
- Filters out non-physical peaks (e.g., 0 keV artifacts).
- Highlights the 511 keV annihilation peak in green when detected.
- NEW: Reads Genie/LabSOCS analysis TXT files from a report directory,
       extracts all reported centroid peaks + isotopes, matches them to the
       corresponding C/N and timepoint spectra, and seeds the peak list +
       IDs before automated matching.

Example:
    python batch_compare_spectra.py \
        --spectra-dir ../spectra_files \
        --eff-csv eff.csv \
        --out-dir output_plots/batch \
        --elements Fe,Cr,Mn,Mo,V,Si,P,S,C,Co,Ni,W,Sb,As,Ta,Tb,Al
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# Add MCNP_ALARA_Workflow to path for nuclear_data module
_workflow_dir = Path(__file__).parent.parent / "MCNP_ALARA_Workflow"
if str(_workflow_dir) not in sys.path:
    sys.path.insert(0, str(_workflow_dir))

from nuclear_data import build_tier1_isotopes, get_data_source

# ───────────────────────── PEAK-DETECTION DEFAULTS ───────────────────────────
SPLIT_CHANNEL1 = 2100
SPLIT_CHANNEL2 = 3000

# You tuned these to be only mildly smoothed.
SIGMAS = (3.0, 2.0, 2.0)

# Your current “slightly more sensitive but not wild” params.
# Keep these as defaults; refine with CLI if needed.
REGION_PARAMS = (
    {"height": 100, "prominence": 60, "distance": 2},
    {"height": 80, "prominence": 40, "distance": 2},
    {"height": 80, "prominence": 30, "distance": 2},
)

# paceENSDF is optional
try:
    import paceENSDF as pe
    HAS_PACE = True
except ImportError:  # pragma: no cover
    HAS_PACE = False


# ──────────────────────────── I/O HELPERS ────────────────────────────────────
ENERGY_CAL_RE = re.compile(r"\b([ABC])\s*=\s*([+-]?\d+\.\d+E[+-]\d+)")
LIVE_TIME_RE = re.compile(r"Elapsed Live Time:\s*([\d.]+)")

ASC_NAME_RE = re.compile(r"(.+?)-([CN])_(.+)\.ASC$", re.IGNORECASE)

ID_LINE_RE = re.compile(r"^\s*ID:\s*(.+?)\s*$", re.IGNORECASE)

NUCLIDES_HEADER_RE = re.compile(r"^\s*NUCLIDES ANALYZED\s*$", re.IGNORECASE)
NUCLIDE_LINE_RE = re.compile(
    r"^\s*([A-Za-z]{1,3}\s*\d{1,3}(?:m\d*|m)?)\s+",
    re.IGNORECASE,
)
CENTROID_HEADER_RE = re.compile(r"^\s*CENTROID", re.IGNORECASE)
ROI_HEADER_RE = re.compile(r"^\s*ROI\s+", re.IGNORECASE)

# Data rows in the ROI table:
# first column is centroid energy (keV)
CENTROID_ROW_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+")


def parse_energy_calibration(txt_path: str) -> Tuple[float, float, float]:
    vals: Dict[str, float] = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ENERGY_CAL_RE.search(line)
            if m:
                vals[m.group(1).upper()] = float(m.group(2))
                if len(vals) == 3:
                    break
    if set(vals) != {"A", "B", "C"}:
        raise RuntimeError(f"Could not find A, B, C in {txt_path}")
    return vals["A"], vals["B"], vals["C"]


def parse_live_time(txt_path: str) -> Optional[float]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LIVE_TIME_RE.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def load_spectrum(txt_path: str) -> pd.DataFrame:
    """Read Channel/Counts pairs from a Genie-2000 ASCII export."""
    records: List[Tuple[int, int]] = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("Channel"):
                break
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                ch, cnt = int(parts[0]), int(parts[1])
                records.append((ch, cnt))
            except ValueError:
                continue
    if not records:
        raise RuntimeError(f"No channel data found in {txt_path}")
    return pd.DataFrame(records, columns=["Channel", "Counts"])


def parse_asc_key(path: str) -> Optional[Tuple[str, str, str]]:
    """Return (sample, letter, timepart) from an .ASC filename."""
    base = os.path.basename(path)
    m = ASC_NAME_RE.match(base)
    if not m:
        return None
    sample, letter, timepart = m.groups()
    return sample, letter.upper(), timepart


def normalize_timepart(s: str) -> str:
    """Normalize time strings to improve matching (e.g., '4d EOI' -> '4dEOI')."""
    t = str(s).strip()
    t = t.replace(" ", "")
    t = t.replace("-", "")
    t = t.replace("_", "")
    return t


# ────────────────────── CALIBRATION / CORRECTION ─────────────────────────────

def apply_energy_cal(df: pd.DataFrame, A: float, B: float, C: float) -> pd.DataFrame:
    df["Energy_keV"] = A + B * df["Channel"] + C * df["Channel"] ** 2
    return df


def parse_efficiency_model(csv_path: str) -> Dict[str, float | str]:
    """Parse detector efficiency model constants.

    Expected single-row CSV with at least:
        C1, C2, C3, C4
    Optional:
        DetModel

    Uses:
        Efficiency = (DetModel) * (C1 + C2*Log(E) + C3*Log(E)^2 + C4*Log(E)^3)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        values = [v.strip() for v in f.readline().strip().split(",")]

    hmap = {h.lower(): h for h in header}

    def _get(name: str) -> float:
        if name.lower() not in hmap:
            raise RuntimeError(f"Efficiency CSV missing column '{name}' in {csv_path}")
        return float(values[header.index(hmap[name.lower()])])

    c1 = _get("C1")
    c2 = _get("C2")
    c3 = _get("C3")
    c4 = _get("C4")

    det_keys = ["detmodel", "det_model", "det. model", "detector_model"]
    det_model = 1.0
    for k in det_keys:
        if k in hmap:
            det_model = float(values[header.index(hmap[k])])
            break

    return {
        "model": "requested_log_poly",
        "DetModel": det_model,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
    }


def apply_efficiency_model(df: pd.DataFrame, model: Dict[str, float | str]) -> pd.DataFrame:
    """Apply the user-specified detector efficiency equation."""
    E = df["Energy_keV"].to_numpy()

    det = float(model.get("DetModel", 1.0))
    c1 = float(model["C1"])
    c2 = float(model["C2"])
    c3 = float(model["C3"])
    c4 = float(model["C4"])

    # Interpret Log(E) as natural log.
    with np.errstate(divide="ignore", invalid="ignore"):
        logE = np.log(np.where(E > 0, E, np.nan))

    eps = det * (c1 + c2 * logE + c3 * logE**2 + c4 * logE**3)

    df["Efficiency"] = np.where(np.isfinite(eps) & (eps > 0), eps, np.nan)
    df["Corrected_Counts"] = df["Counts"] / df["Efficiency"]
    return df


# ───────────────────────── PEAK DETECTION ────────────────────────────────────

def _gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def detect_peaks_segmented(
    df: pd.DataFrame,
    fit_window: int = 4,
    splits: Tuple[int, int] = (SPLIT_CHANNEL1, SPLIT_CHANNEL2),
    sigmas: Tuple[float, float, float] = SIGMAS,
    region_params: Sequence[dict] = REGION_PARAMS,
) -> pd.DataFrame:
    """Segmented peak detection with region-specific thresholds.

    Smoothing is used only to stabilize the initial find_peaks call.
    If any sigma <= 0, smoothing is disabled for that region.
    Gaussian centroid refinement uses *unsmoothed* corrected counts.
    """
    y = df["Corrected_Counts"].fillna(0).to_numpy()

    split1, split2 = splits
    masks = (
        df["Channel"] <= split1,
        (df["Channel"] > split1) & (df["Channel"] <= split2),
        df["Channel"] > split2,
    )

    peaks: List[Dict[str, float]] = []

    for mask, sigma, params in zip(masks, sigmas, region_params):
        idx_region = df.index[mask]
        if idx_region.empty:
            continue

        sigma_val = float(sigma) if sigma is not None else 0.0
        if sigma_val <= 0.0:
            smoothed = y[idx_region]
        else:
            smoothed = gaussian_filter1d(y[idx_region], sigma=sigma_val, mode="nearest")

        pk_idx, _ = find_peaks(smoothed, **params)

        for local_idx in pk_idx:
            ch_idx = idx_region[local_idx]
            lo = max(0, ch_idx - fit_window)
            hi = min(len(df), ch_idx + fit_window + 1)

            x_keV = df["Energy_keV"].iloc[lo:hi].to_numpy()
            y_loc = y[lo:hi]

            try:
                mu0 = float(df["Energy_keV"].iloc[ch_idx])
                p0 = [float(y[ch_idx]), mu0, 1.2, float(np.median(y_loc))]
                popt, _ = curve_fit(_gaussian, x_keV, y_loc, p0=p0, maxfev=4000)
                a, mu, sigma_fit, _c = popt
                area = a * sigma_fit * np.sqrt(2 * np.pi)
                sigma_keV = sigma_fit
            except RuntimeError:
                a, mu, sigma_keV, area = (
                    float(y[ch_idx]),
                    float(df["Energy_keV"].iloc[ch_idx]),
                    np.nan,
                    np.nan,
                )

            peaks.append(
                {
                    "Channel": int(df["Channel"].iloc[ch_idx]),
                    "Energy_keV": float(mu),
                    "Amplitude": float(a),
                    "Peak_height": float(y[ch_idx]),
                    "Sigma_keV": float(sigma_keV) if np.isfinite(sigma_keV) else np.nan,
                    "Area": float(area) if np.isfinite(area) else np.nan,
                    "Is_Report": False,
                    "Report_Isotope": None,
                    "Report_File": None,
                }
            )

    if not peaks:
        return pd.DataFrame(
            columns=[
                "Channel", "Energy_keV", "Amplitude", "Peak_height", "Sigma_keV", "Area",
                "Is_Report", "Report_Isotope", "Report_File"
            ]
        )

    return pd.DataFrame(peaks).sort_values("Energy_keV").reset_index(drop=True)


def deduplicate_peaks_with_report(peaks: pd.DataFrame, merge_keV: float = 0.6) -> pd.DataFrame:
    """Merge near-duplicate peaks; prefer report peaks if present."""
    if peaks.empty:
        return peaks

    peaks = peaks.sort_values("Energy_keV").reset_index(drop=True)
    keep_rows = []
    cluster = [0]

    def _flush(cluster_idx: List[int]):
        sub = peaks.iloc[cluster_idx]

        if "Is_Report" in sub.columns and sub["Is_Report"].any():
            sub_r = sub[sub["Is_Report"] == True]
            best_i = sub_r["Peak_height"].idxmax() if "Peak_height" in sub_r.columns else sub_r.index[0]
            keep_rows.append(peaks.loc[best_i])
            return

        best_i = sub["Peak_height"].idxmax() if "Peak_height" in sub.columns else sub.index[0]
        keep_rows.append(peaks.loc[best_i])

    for i in range(1, len(peaks)):
        if peaks.loc[i, "Energy_keV"] - peaks.loc[cluster[-1], "Energy_keV"] <= merge_keV:
            cluster.append(i)
        else:
            _flush(cluster)
            cluster = [i]
    _flush(cluster)

    return pd.DataFrame(keep_rows).sort_values("Energy_keV").reset_index(drop=True)


def filter_physical_peaks(peaks: pd.DataFrame, min_energy_keV: float) -> pd.DataFrame:
    """Remove non-physical or low-energy noise peaks."""
    if peaks.empty:
        return peaks
    mask = np.isfinite(peaks["Energy_keV"]) & (peaks["Energy_keV"] >= min_energy_keV)
    return peaks.loc[mask].reset_index(drop=True)


# ─────────────────────── ENSDF CATALOG (OPTIONAL) ────────────────────────────

_ISO_PATTERNS = (
    re.compile(r"^(?P<el>[A-Za-z]{1,3})[-\s]?(?P<mass>\d{1,3})(?P<meta>m\d*|m)?$", re.IGNORECASE),
    re.compile(r"^(?P<mass>\d{1,3})(?P<meta>m\d*|m)?[-\s]?(?P<el>[A-Za-z]{1,3})$", re.IGNORECASE),
)


def normalize_isotope_label(label: str) -> str:
    """Canonical label like 'W187' or 'Tb154M'."""
    s = str(label).strip()
    s = s.replace("_", "").replace("/", "").replace(".", "")
    s = s.replace("(", "").replace(")", "")
    s = s.strip()

    for pat in _ISO_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        el = m.group("el").capitalize()
        mass = int(m.group("mass"))
        meta = m.group("meta") or ""
        meta_norm = "M" if meta else ""
        return f"{el}{mass}{meta_norm}"

    return s.upper()


def element_from_isotope(label: str) -> Optional[str]:
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


def _safe_float(x) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    if not m:
        raise ValueError
    return float(m.group())


def _walk_ensdf(node, iso_raw: str, iso_canon: str, elem: Optional[str], rows: List[dict]) -> None:
    if isinstance(node, dict):
        keys = {k.lower(): k for k in node}
        e_key = next((k for k in keys if "energy" in k or k in ("e", "eg")), None)
        i_key = next((k for k in keys if "intens" in k or k in ("ri", "i")), None)
        if e_key and i_key:
            try:
                e = _safe_float(node[keys[e_key]])
                ri = _safe_float(node[keys[i_key]])
                if 0.0 < e < 10000.0 and ri >= 0.0:
                    rows.append(
                        {
                            "Energy_keV": e,
                            "Isotope": iso_raw,
                            "Isotope_canon": iso_canon,
                            "Element": elem,
                            "RI_%": ri,
                        }
                    )
            except ValueError:
                pass
        for v in node.values():
            _walk_ensdf(v, iso_raw, iso_canon, elem, rows)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_ensdf(v, iso_raw, iso_canon, elem, rows)


def build_gamma_catalog(elements: Sequence[str] | None = None) -> Optional[pd.DataFrame]:
    if not HAS_PACE:
        return None

    ensdf = pe.ENSDF()
    nuclides = ensdf.load_ensdf()

    keep = {el.strip().lower() for el in elements} if elements else None
    rows: List[dict] = []

    for nucl in nuclides:
        iso = nucl.get("parentID", "Unknown")
        iso_raw = iso.strip() if isinstance(iso, str) else str(iso).strip()
        iso_canon = normalize_isotope_label(iso_raw)
        elem = element_from_isotope(iso_raw)

        if keep:
            if elem is None or elem.lower() not in keep:
                continue

        _walk_ensdf(nucl, iso_raw, iso_canon, elem, rows)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["Energy_keV", "Isotope_canon", "RI_%"])


def _match_one_peak(
    energy: float,
    gamma_db: pd.DataFrame,
    tolerance_keV: float,
    top_n: int = 3,
) -> Optional[dict]:
    diff = np.abs(gamma_db["Energy_keV"].to_numpy() - energy)
    mask = diff <= tolerance_keV
    if not mask.any():
        return None

    candidates = gamma_db.loc[mask].sort_values("RI_%", ascending=False).head(top_n)
    best = candidates.iloc[0]

    label_list = [
        f"{row.Isotope_canon} {row.Energy_keV:.1f} keV ({row['RI_%']:.1f}%)"
        for _, row in candidates.iterrows()
    ]

    return {
        "Isotope": best["Isotope_canon"],
        "Isotope_raw": best["Isotope"],
        "Gamma_E_keV": float(best["Energy_keV"]),
        "RI_%": float(best["RI_%"]),
        "Candidates": "; ".join(label_list),
    }


def match_peaks_three_tier(
    peaks: pd.DataFrame,
    tier1_db: Optional[pd.DataFrame],
    tier2_db: Optional[pd.DataFrame],
    tier3_db: Optional[pd.DataFrame],
    tol1: float,
    tol2: float,
    tol3: float,
    top_n: int = 3,
) -> pd.DataFrame:
    """Match peaks in three tiers with per-tier tolerances."""
    if peaks.empty:
        return pd.DataFrame()

    matches: List[Dict[str, float | str]] = []

    for _, p in peaks.iterrows():
        e_obs = float(p["Energy_keV"])
        hit = None
        source = None
        tol_used = None

        if tier1_db is not None and not tier1_db.empty:
            hit = _match_one_peak(e_obs, tier1_db, tol1, top_n)
            if hit:
                source, tol_used = "tier1", tol1

        if hit is None and tier2_db is not None and not tier2_db.empty:
            hit = _match_one_peak(e_obs, tier2_db, tol2, top_n)
            if hit:
                source, tol_used = "tier2", tol2

        if hit is None and tier3_db is not None and not tier3_db.empty:
            hit = _match_one_peak(e_obs, tier3_db, tol3, top_n)
            if hit:
                source, tol_used = "tier3", tol3

        if hit is None:
            continue

        matches.append(
            {
                "Observed_E_keV": e_obs,
                "Amplitude": float(p.get("Amplitude", np.nan)),
                "Isotope": hit["Isotope"],
                "Isotope_raw": hit.get("Isotope_raw"),
                "Gamma_E_keV": hit["Gamma_E_keV"],
                "RI_%": hit["RI_%"],
                "Candidates": hit["Candidates"],
                "Source": source,
                "Tolerance_keV": tol_used,
            }
        )

    return pd.DataFrame(matches)


# ─────────────────────── REPORT TXT PARSING (NEW) ────────────────────────────

def extract_id_line_text(report_path: str, max_lines: int = 80) -> str:
    """Extract the ID line string if present, otherwise empty."""
    try:
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                m = ID_LINE_RE.match(line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return ""


def parse_report_nuclide_centroids(report_path: str) -> List[dict]:
    """Parse a Genie/LabSOCS analysis TXT file.

    Returns list of dict:
        {"Isotope": <raw label>, "Energy_keV": <float>}
    """
    peaks: List[dict] = []
    current_iso: Optional[str] = None
    in_block = False

    try:
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return peaks

    # Find the NUCLIDES ANALYZED header
    start_idx = None
    for i, line in enumerate(lines):
        if NUCLIDES_HEADER_RE.match(line):
            start_idx = i + 1
            break

    if start_idx is None:
        return peaks

    for line in lines[start_idx:]:
        if not line.strip():
            # allow blank lines without resetting state
            continue

        # Nuclide header line like: "W187   23.900 h   B   Activity = ..."
        m_iso = NUCLIDE_LINE_RE.match(line)
        if m_iso:
            current_iso = m_iso.group(1).strip()
            in_block = True
            continue

        if not in_block or current_iso is None:
            continue

        # Skip table headers
        if ROI_HEADER_RE.match(line) or CENTROID_HEADER_RE.match(line):
            continue

        # Data row: first number is centroid energy (keV)
        m_cent = CENTROID_ROW_RE.match(line)
        if m_cent:
            try:
                e = float(m_cent.group(1))
                if 0.0 < e < 10000.0:
                    peaks.append({"Isotope": current_iso, "Energy_keV": e, "Report_File": os.path.basename(report_path)})
            except ValueError:
                pass

    return peaks


def score_report_to_asc(
    report_basename_lower: str,
    id_text_lower: str,
    asc_sample: str,
    asc_letter: str,
    asc_timepart: str,
) -> int:
    """Score how well a report text file matches an ASC key."""
    sample_l = asc_sample.lower()
    letter_l = asc_letter.lower()
    time_norm = normalize_timepart(asc_timepart).lower()

    score = 0

    # Sample match
    if sample_l and (sample_l in report_basename_lower or sample_l in id_text_lower):
        score += 3

    # Time match (normalized)
    if time_norm and (time_norm in normalize_timepart(report_basename_lower) or time_norm in normalize_timepart(id_text_lower)):
        score += 3

    # Letter match: prefer explicit token-ish patterns
    letter_hit = False
    for blob in (report_basename_lower, id_text_lower):
        if re.search(rf"(^|[^a-z0-9]){letter_l}([^a-z0-9]|$)", blob):
            letter_hit = True
            break
        # common filename encodings
        if f"_{letter_l}_" in blob or f"-{letter_l}_" in blob or f"_{letter_l}-" in blob or f"-{letter_l}-" in blob:
            letter_hit = True
            break

    if letter_hit:
        score += 2
    else:
        # still allow matching via strong sample+time
        score += 0

    return score


def build_report_map(
    report_dir: str,
    asc_keys: List[Tuple[str, str, str]],
) -> Dict[Tuple[str, str, str], List[dict]]:
    """Scan report directory and map report peaks to ASC (sample, letter, time)."""
    report_map: Dict[Tuple[str, str, str], List[dict]] = {}

    report_dir_exp = os.path.expanduser(report_dir)
    txt_files = sorted(glob(os.path.join(report_dir_exp, "*.txt"))) + sorted(glob(os.path.join(report_dir_exp, "*.TXT")))

    if not txt_files:
        return report_map

    for rpt in txt_files:
        basename = os.path.basename(rpt)
        base_l = basename.lower()
        id_text = extract_id_line_text(rpt)
        id_l = id_text.lower()

        # Parse peaks from this report
        rpt_peaks = parse_report_nuclide_centroids(rpt)
        if not rpt_peaks:
            continue

        # Choose best ASC key by scoring
        best_key = None
        best_score = -1

        for sample, letter, timepart in asc_keys:
            s = score_report_to_asc(base_l, id_l, sample, letter, timepart)
            if s > best_score:
                best_score = s
                best_key = (sample, letter, timepart)

        # Require at least decent evidence:
        #  - sample+time strongly present OR sample+letter+something
        if best_key is None or best_score < 4:
            continue

        report_map.setdefault(best_key, [])
        report_map[best_key].extend(rpt_peaks)

    return report_map


def build_report_peaks_df(
    spectrum_df: pd.DataFrame,
    report_entries: List[dict],
) -> pd.DataFrame:
    """Convert report centroid energies into the internal peak format.

    We map each report energy to the nearest energy bin in the spectrum,
    so that plots/CSV can include sensible Channel and Peak_height values.
    """
    if not report_entries:
        return pd.DataFrame(
            columns=[
                "Channel", "Energy_keV", "Amplitude", "Peak_height", "Sigma_keV", "Area",
                "Is_Report", "Report_Isotope", "Report_File",
            ]
        )

    E_spec = spectrum_df["Energy_keV"].to_numpy()
    Ch_spec = spectrum_df["Channel"].to_numpy()
    Y_spec = spectrum_df["Corrected_Counts"].fillna(0).to_numpy()

    rows = []
    for ent in report_entries:
        iso_raw = ent.get("Isotope")
        e_rep = float(ent.get("Energy_keV", np.nan))
        rpt_file = ent.get("Report_File")

        if not np.isfinite(e_rep):
            continue

        # nearest bin
        idx = int(np.nanargmin(np.abs(E_spec - e_rep)))
        e_use = float(E_spec[idx])
        ch_use = int(Ch_spec[idx])
        h_use = float(Y_spec[idx])

        rows.append(
            {
                "Channel": ch_use,
                "Energy_keV": e_use,
                "Amplitude": h_use,
                "Peak_height": h_use,
                "Sigma_keV": np.nan,
                "Area": np.nan,
                "Is_Report": True,
                "Report_Isotope": normalize_isotope_label(iso_raw) if iso_raw else None,
                "Report_File": rpt_file,
                "Reported_E_keV_raw": e_rep,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Channel", "Energy_keV", "Amplitude", "Peak_height", "Sigma_keV", "Area",
                "Is_Report", "Report_Isotope", "Report_File",
            ]
        )

    return pd.DataFrame(rows).sort_values("Energy_keV").reset_index(drop=True)


def build_report_matches(report_peaks: pd.DataFrame) -> pd.DataFrame:
    """Create matches DataFrame from report peaks (highest priority)."""
    if report_peaks.empty:
        return pd.DataFrame()

    rows = []
    for _, r in report_peaks.iterrows():
        iso = r.get("Report_Isotope")
        if not iso:
            continue
        e = float(r["Energy_keV"])
        rows.append(
            {
                "Observed_E_keV": e,
                "Amplitude": float(r.get("Amplitude", np.nan)),
                "Isotope": iso,
                "Isotope_raw": iso,
                "Gamma_E_keV": e,
                "RI_%": np.nan,
                "Candidates": f"{iso} {e:.1f} keV (report)",
                "Source": "report",
                "Tolerance_keV": 0.0,
            }
        )
    return pd.DataFrame(rows)


def remove_peaks_covered_by_report(
    peaks: pd.DataFrame,
    report_matches: pd.DataFrame,
    atol_keV: float,
) -> pd.DataFrame:
    """Remove peaks that already have report IDs (avoid duplicate matching)."""
    if peaks.empty or report_matches.empty:
        return peaks

    rep_E = report_matches["Observed_E_keV"].to_numpy()
    keep_mask = []
    for e in peaks["Energy_keV"].to_numpy():
        keep_mask.append(bool(np.all(np.abs(rep_E - e) > atol_keV)))
    return peaks.loc[keep_mask].reset_index(drop=True)


# ─────────────────────── PLOTTING HELPERS ────────────────────────────────────

def _highlight_annihilation(peaks: pd.DataFrame, tol_keV: float) -> pd.DataFrame:
    """Return subset of peaks near 511 keV."""
    if peaks.empty:
        return peaks.iloc[0:0]
    return peaks[np.abs(peaks["Energy_keV"] - 511.0) <= tol_keV]


def plot_single(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    matches: Optional[pd.DataFrame],
    title: str,
    out_path: str,
    annotate_top: int,
    match_atol_keV: float,
    annihil_tol_keV: float,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["Energy_keV"], df["Corrected_Counts"], lw=1.0, label="Corrected")
    plt.yscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts / Efficiency")
    plt.title(title)

    if not peaks.empty:
        # Split report vs auto for visualization clarity
        if "Is_Report" in peaks.columns:
            auto = peaks[peaks["Is_Report"] == False]
            rpt = peaks[peaks["Is_Report"] == True]
        else:
            auto = peaks
            rpt = peaks.iloc[0:0]

        if not auto.empty:
            plt.scatter(auto["Energy_keV"], auto["Peak_height"], s=18, c="red", label="Detected Peaks")
        if not rpt.empty:
            plt.scatter(rpt["Energy_keV"], rpt["Peak_height"], s=26, c="orange", label="Report Peaks")

        ann = _highlight_annihilation(peaks, annihil_tol_keV)
        if not ann.empty:
            plt.scatter(ann["Energy_keV"], ann["Peak_height"], s=40, c="green", label="511 keV")

        # Balanced annotation across energy bands
        bands = [(0, 500), (500, 1200), (1200, np.inf)]
        per_band = max(1, annotate_top // 3) if annotate_top > 0 else 0

        for lo, hi in bands:
            band = peaks[(peaks["Energy_keV"] >= lo) & (peaks["Energy_keV"] < hi)]
            if band.empty or per_band == 0:
                continue
            top = band.nlargest(per_band, "Peak_height")

            for _, r in top.iterrows():
                label = None

                # Prefer report isotope label if present at the peak row
                rep_iso = r.get("Report_Isotope") if isinstance(r, pd.Series) else None
                if rep_iso:
                    label = str(rep_iso)

                if label is None and matches is not None and not matches.empty:
                    hit = matches[
                        np.isclose(matches["Observed_E_keV"], r["Energy_keV"], atol=match_atol_keV)
                    ]
                    if not hit.empty:
                        label = f"{hit.iloc[0]['Isotope']}"

                if label is None:
                    if abs(float(r["Energy_keV"]) - 511.0) <= annihil_tol_keV:
                        label = "511 keV"
                    else:
                        label = f"{r['Energy_keV']:.1f} keV"

                plt.text(r["Energy_keV"], r["Peak_height"] * 1.2, label, rotation=72, fontsize=7, ha="center")

    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_comparison(
    record_a: dict,
    record_b: dict,
    label_a: str,
    label_b: str,
    out_path: str,
    annotate_top: int,
    match_atol_keV: float,
    annihil_tol_keV: float,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(record_a["df"]["Energy_keV"], record_a["df"]["Corrected_Counts"], lw=1.0, label=label_a)
    plt.plot(record_b["df"]["Energy_keV"], record_b["df"]["Corrected_Counts"], lw=1.0, label=label_b)
    plt.yscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts / Efficiency")
    plt.title(f"{label_a} vs {label_b}")

    for record, color in ((record_a, "red"), (record_b, "purple")):
        peaks = record["peaks"]
        matches = record.get("matches")
        if peaks.empty:
            continue

        plt.scatter(peaks["Energy_keV"], peaks["Peak_height"], s=12, color=color, alpha=0.35)

        ann = _highlight_annihilation(peaks, annihil_tol_keV)
        if not ann.empty:
            plt.scatter(ann["Energy_keV"], ann["Peak_height"], s=28, c="green", alpha=0.9)

        if annotate_top > 0:
            top = peaks.nlargest(annotate_top, "Peak_height")
            for _, r in top.iterrows():
                label = None
                rep_iso = r.get("Report_Isotope") if isinstance(r, pd.Series) else None
                if rep_iso:
                    label = str(rep_iso)

                if label is None and matches is not None and not matches.empty:
                    hit = matches[
                        np.isclose(matches["Observed_E_keV"], r["Energy_keV"], atol=match_atol_keV)
                    ]
                    if not hit.empty:
                        label = hit.iloc[0]["Isotope"]

                if label is None:
                    label = f"{r['Energy_keV']:.0f}"

                if abs(float(r["Energy_keV"]) - 511.0) <= annihil_tol_keV:
                    label = "511 keV"

                plt.text(
                    r["Energy_keV"],
                    r["Peak_height"] * 1.15,
                    label,
                    fontsize=6,
                    rotation=68,
                    ha="center",
                    color=color,
                )

    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ────────────────────────── MAIN PIPELINE ────────────────────────────────────

def process_file(
    path: str,
    eff_model: Dict[str, float | str],
    gamma_tier1: Optional[pd.DataFrame],
    gamma_tier2: Optional[pd.DataFrame],
    gamma_tier3: Optional[pd.DataFrame],
    report_map: Dict[Tuple[str, str, str], List[dict]],
    args: argparse.Namespace,
) -> dict:
    A, B, C = parse_energy_calibration(path)
    live_time = parse_live_time(path)

    df = load_spectrum(path)
    df = apply_energy_cal(df, A, B, C)
    df = apply_efficiency_model(df, eff_model)

    asc_key = parse_asc_key(path)
    report_entries = report_map.get(asc_key, []) if asc_key else []
    report_peaks = build_report_peaks_df(df, report_entries)

    auto_peaks = detect_peaks_segmented(
        df,
        fit_window=args.fit_window,
        splits=(args.split_channel1, args.split_channel2),
        sigmas=(args.sigma_low, args.sigma_mid, args.sigma_high),
        region_params=(
            {"height": args.height_low, "prominence": args.prom_low, "distance": args.dist_low},
            {"height": args.height_mid, "prominence": args.prom_mid, "distance": args.dist_mid},
            {"height": args.height_high, "prominence": args.prom_high, "distance": args.dist_high},
        ),
    )

    # Combine report + auto peaks, then deduplicate with report preference
    peaks = pd.concat([report_peaks, auto_peaks], ignore_index=True, sort=False)
    peaks = deduplicate_peaks_with_report(peaks, merge_keV=args.merge_keV)
    peaks = filter_physical_peaks(peaks, min_energy_keV=args.min_peak_energy_keV)

    # Build report matches first
    report_matches = build_report_matches(report_peaks)

    # Run three-tier matching on remaining peaks
    auto_match_peaks = remove_peaks_covered_by_report(
        peaks,
        report_matches,
        atol_keV=min(args.merge_keV, 0.6),
    )

    auto_matches = None
    if not auto_match_peaks.empty:
        auto_matches = match_peaks_three_tier(
            auto_match_peaks,
            gamma_tier1,
            gamma_tier2,
            gamma_tier3,
            tol1=args.tol_tier1,
            tol2=args.tol_tier2,
            tol3=args.tol_tier3,
            top_n=args.top_n,
        )

    if report_matches.empty and (auto_matches is None or auto_matches.empty):
        matches = None
    elif auto_matches is None or auto_matches.empty:
        matches = report_matches
    elif report_matches.empty:
        matches = auto_matches
    else:
        matches = pd.concat([report_matches, auto_matches], ignore_index=True).sort_values("Observed_E_keV")

    return {
        "df": df,
        "peaks": peaks,
        "matches": matches,
        "live_time": live_time,
        "A": A,
        "B": B,
        "C": C,
        "asc_key": asc_key,
        "n_report_peaks": len(report_entries),
    }


def discover_pairs(files: Sequence[str]) -> List[Tuple[str, str, str]]:
    """Return list of (C_path, N_path, label) for matched time points."""
    bucket: Dict[Tuple[str, str], Dict[str, str]] = {}
    for f in files:
        m = re.match(r"(.+?)-([CN])_(.+)\.ASC", os.path.basename(f), re.IGNORECASE)
        if not m:
            continue
        sample, tag, timepart = m.groups()
        bucket.setdefault((sample, timepart), {})
        if tag.upper() == "C":
            bucket[(sample, timepart)]["C"] = f
        else:
            bucket[(sample, timepart)]["N"] = f

    pairs: List[Tuple[str, str, str]] = []
    for (sample, timepart), d in bucket.items():
        if "C" in d and "N" in d:
            label = f"{sample} {timepart}"
            pairs.append((d["C"], d["N"], label))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Batch γ-spectrum processing and comparison")

    p.add_argument("--spectra-dir", default="../spectra_files", help="Directory with *.ASC spectra")
    p.add_argument("--eff-csv", default="eff.csv", help="Efficiency CSV (C1–C4 + optional DetModel)")
    p.add_argument("--out-dir", default="output_plots/batch", help="Output directory for plots")

    # NEW: report directory
    p.add_argument(
        "--report-dir",
        default="~/CAE/projects/ALARA/MCNP_ALARA_Workflow/Experimental_Data/RAFM3",
        help="Directory containing Genie/LabSOCS analysis TXT files",
    )

    # ── Matching controls ───────────────────────────────────────────────
    p.add_argument("--tol-tier1", type=float, default=3.0, help="Tier 1 matching tolerance (keV)")
    p.add_argument("--tol-tier2", type=float, default=3.0, help="Tier 2 matching tolerance (keV)")
    p.add_argument("--tol-tier3", type=float, default=2.0, help="Tier 3 matching tolerance (keV)")
    p.add_argument("--top-n", type=int, default=3, help="Number of top-RI candidates to record")

    # ── Peak post-processing ───────────────────────────────────────────
    p.add_argument("--merge-keV", type=float, default=0.6, help="Merge peaks closer than this (keV)")
    p.add_argument(
        "--min-peak-energy-keV",
        type=float,
        default=25.0,
        help="Discard detected peaks below this energy (keV)",
    )
    p.add_argument("--fit-window", type=int, default=4, help="Half-window (points) for local Gaussian fit")

    # ── Segmentation and smoothing ─────────────────────────────────────
    p.add_argument("--split-channel1", type=int, default=SPLIT_CHANNEL1)
    p.add_argument("--split-channel2", type=int, default=SPLIT_CHANNEL2)
    p.add_argument("--sigma-low", type=float, default=SIGMAS[0])
    p.add_argument("--sigma-mid", type=float, default=SIGMAS[1])
    p.add_argument("--sigma-high", type=float, default=SIGMAS[2])

    # ── Region-specific peak thresholds ────────────────────────────────
    p.add_argument("--height-low", type=float, default=REGION_PARAMS[0]["height"])
    p.add_argument("--prom-low", type=float, default=REGION_PARAMS[0]["prominence"])
    p.add_argument("--dist-low", type=int, default=REGION_PARAMS[0]["distance"])

    p.add_argument("--height-mid", type=float, default=REGION_PARAMS[1]["height"])
    p.add_argument("--prom-mid", type=float, default=REGION_PARAMS[1]["prominence"])
    p.add_argument("--dist-mid", type=int, default=REGION_PARAMS[1]["distance"])

    p.add_argument("--height-high", type=float, default=REGION_PARAMS[2]["height"])
    p.add_argument("--prom-high", type=float, default=REGION_PARAMS[2]["prominence"])
    p.add_argument("--dist-high", type=int, default=REGION_PARAMS[2]["distance"])

    # ── Plot annotation controls ───────────────────────────────────────
    p.add_argument(
        "--annotate-top-single",
        type=int,
        default=18,
        help="Peaks to annotate on single-spectrum plots",
    )
    p.add_argument(
        "--annotate-top-compare",
        type=int,
        default=6,
        help="Peaks to annotate on comparison plots (0 = none)",
    )
    p.add_argument(
        "--annihil-tol",
        type=float,
        default=3.0,
        help="Tolerance for highlighting 511 keV peak (keV)",
    )

    # ── Element preference list ────────────────────────────────────────
    p.add_argument(
        "--elements",
        default="Fe,Cr,Mn,Mo,V,Si,P,S,C,Co,Ni,W,Sb,As,Ta,Tb,Al",
        help="Preferred element symbols for tier-2 matching",
    )

    args = p.parse_args()

    spectra_files = sorted(glob(os.path.join(args.spectra_dir, "*.ASC")))
    if not spectra_files:
        raise SystemExit(f"No .ASC files found in {args.spectra_dir}")

    eff_model = parse_efficiency_model(args.eff_csv)

    element_list = [e.strip() for e in args.elements.split(",") if e.strip()]

    # Build tier-1 isotopes dynamically from paceENSDF (or fallback data)
    # These are high-intensity gamma emitters from RAFM-relevant elements
    tier1_isotopes = build_tier1_isotopes(
        elements=['W', 'Cr', 'Fe', 'Ta', 'Co', 'Mn', 'V', 'Al', 'Tb'],
        min_intensity=0.10,
    )
    print(f"Tier-1 isotopes ({get_data_source()}): {tier1_isotopes}")

    def build_isotope_subset(names: List[str]) -> Optional[pd.DataFrame]:
        db = build_gamma_catalog(None)
        if db is None:
            return None
        canon_names = {normalize_isotope_label(n) for n in names}
        sub = db[db["Isotope_canon"].isin(canon_names)]
        return sub if not sub.empty else None

    gamma_tier1 = build_isotope_subset(tier1_isotopes) if HAS_PACE else None
    gamma_tier2 = build_gamma_catalog(element_list) if (element_list and HAS_PACE) else None
    gamma_tier3 = build_gamma_catalog(None) if HAS_PACE else None

    if not HAS_PACE:
        print("paceENSDF not found; isotope matching disabled beyond report seeding.")

    # Build list of ASC keys for report matching
    asc_keys = []
    for f in spectra_files:
        k = parse_asc_key(f)
        if k:
            asc_keys.append(k)

    # NEW: build report map once
    report_map = build_report_map(args.report_dir, asc_keys)

    per_file_records: Dict[str, dict] = {}
    peak_rows: List[Dict[str, object]] = []

    single_dir = os.path.join(args.out_dir, "single")
    compare_dir = os.path.join(args.out_dir, "compare")

    plot_atol = max(args.tol_tier1, args.tol_tier2, args.tol_tier3)

    for path in spectra_files:
        rec = process_file(
            path,
            eff_model,
            gamma_tier1,
            gamma_tier2,
            gamma_tier3,
            report_map,
            args,
        )
        per_file_records[path] = rec

        peaks = rec["peaks"]
        matches = rec["matches"]

        # Build summary rows
        if not peaks.empty:
            for _, r in peaks.iterrows():
                iso_label = None
                gamma_e = None
                ri_val = None
                candidates = None
                match_source = None
                tol_used = None

                # Prefer report isotope if present on the peak row
                if r.get("Is_Report") and r.get("Report_Isotope"):
                    iso_label = r.get("Report_Isotope")
                    gamma_e = float(r["Energy_keV"])
                    ri_val = np.nan
                    candidates = f"{iso_label} {gamma_e:.1f} keV (report)"
                    match_source = "report"
                    tol_used = 0.0
                else:
                    if matches is not None and not matches.empty:
                        hit = matches[
                            np.isclose(matches["Observed_E_keV"], r["Energy_keV"], atol=plot_atol)
                        ]
                        if not hit.empty:
                            row_hit = hit.iloc[0]
                            iso_label = row_hit.get("Isotope")
                            gamma_e = row_hit.get("Gamma_E_keV")
                            ri_val = row_hit.get("RI_%")
                            candidates = row_hit.get("Candidates")
                            match_source = row_hit.get("Source")
                            tol_used = row_hit.get("Tolerance_keV")

                peak_rows.append(
                    {
                        "file": os.path.basename(path),
                        "Energy_keV": r.get("Energy_keV"),
                        "Amplitude": r.get("Amplitude"),
                        "Peak_height": r.get("Peak_height"),
                        "Sigma_keV": r.get("Sigma_keV"),
                        "Area": r.get("Area"),
                        "Live_Time_s": rec.get("live_time"),
                        "A": rec.get("A"),
                        "B": rec.get("B"),
                        "C": rec.get("C"),
                        "Is_Report": bool(r.get("Is_Report", False)),
                        "Report_Isotope": r.get("Report_Isotope"),
                        "Report_File": r.get("Report_File"),
                        "Isotope": iso_label,
                        "Gamma_E_keV": gamma_e,
                        "RI_%": ri_val,
                        "Candidates": candidates,
                        "Match_Source": match_source,
                        "Match_Tolerance_keV": tol_used,
                    }
                )

        title = os.path.basename(path)
        out_path = os.path.join(single_dir, title.replace(".ASC", ".png"))
        plot_single(
            rec["df"],
            peaks,
            matches,
            title,
            out_path,
            annotate_top=args.annotate_top_single,
            match_atol_keV=plot_atol,
            annihil_tol_keV=args.annihil_tol,
        )

    if peak_rows:
        peak_df = pd.DataFrame(peak_rows)
        os.makedirs(args.out_dir, exist_ok=True)
        peak_df.to_csv(os.path.join(args.out_dir, "identified_peaks_batch.csv"), index=False)

    # paired comparisons (C vs N)
    for c_path, n_path, label in discover_pairs(spectra_files):
        rec_c = per_file_records[c_path]
        rec_n = per_file_records[n_path]
        out_path = os.path.join(compare_dir, f"{label.replace(' ', '_')}_C_vs_N.png")
        plot_comparison(
            rec_c,
            rec_n,
            f"C ({label})",
            f"N ({label})",
            out_path,
            annotate_top=args.annotate_top_compare,
            match_atol_keV=plot_atol,
            annihil_tol_keV=args.annihil_tol,
        )

    # Cross-time comparisons: 300s vs 15d for C and N if available
    def find_by_suffix(tag: str) -> Optional[str]:
        for f in spectra_files:
            if f.upper().endswith(tag.upper() + ".ASC"):
                return f
        return None

    c_2h = find_by_suffix("C_300sEOI")
    c_15d = find_by_suffix("C_15dEOI")
    n_2h = find_by_suffix("N_300sEOI")
    n_15d = find_by_suffix("N_15dEOI")

    if c_2h and c_15d:
        out_path = os.path.join(compare_dir, "C_300s_vs_15d.png")
        plot_comparison(
            per_file_records[c_2h],
            per_file_records[c_15d],
            "C 300s",
            "C 15d",
            out_path,
            annotate_top=args.annotate_top_compare,
            match_atol_keV=plot_atol,
            annihil_tol_keV=args.annihil_tol,
        )

    if n_2h and n_15d:
        out_path = os.path.join(compare_dir, "N_300s_vs_15d.png")
        plot_comparison(
            per_file_records[n_2h],
            per_file_records[n_15d],
            "N 300s",
            "N 15d",
            out_path,
            annotate_top=args.annotate_top_compare,
            match_atol_keV=plot_atol,
            annihil_tol_keV=args.annihil_tol,
        )

    print(f"Processed {len(spectra_files)} spectra → {single_dir}")
    print(f"C/N comparisons saved to {compare_dir}")
    if peak_rows:
        print(f"Peak CSV updated: {os.path.join(args.out_dir, 'identified_peaks_batch.csv')}")

    # Report mapping summary
    if report_map:
        n_keys = len(report_map)
        n_total = sum(len(v) for v in report_map.values())
        print(f"Seeded from reports: {n_keys} matched spectrum keys, {n_total} report centroid entries.")
    else:
        print("No report peaks seeded (no matching TXT files found or matching score too low).")


if __name__ == "__main__":
    main()
