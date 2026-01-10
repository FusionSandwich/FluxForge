"""
Interoperability module for data exchange with external tools.

Provides import/export functions for:
- STAYSL PNNL format bundles
- Spreadsheet-based saturation rate data
- Legacy lower-triangular matrix formats
- Cross-tool validation data

Reference: PNNL-22253, STAYSL PNNL User Guide
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SaturationRateData:
    """
    Saturation rate data from spreadsheet import.
    
    Attributes:
        reaction_id: Reaction identifier
        rate: Saturation reaction rate (reactions/s/target-atom)
        uncertainty: Absolute uncertainty in rate
        half_life_s: Product half-life in seconds
        target_atoms: Number of target atoms
        notes: Optional notes/comments
    """
    
    reaction_id: str
    rate: float
    uncertainty: float
    half_life_s: float = 0.0
    target_atoms: float = 0.0
    notes: str = ""


def read_saturation_rates_csv(
    path: Union[str, Path],
    delimiter: str = ",",
    reaction_col: str = "reaction",
    rate_col: str = "rate",
    unc_col: str = "uncertainty",
    halflife_col: Optional[str] = "half_life_s",
    atoms_col: Optional[str] = "target_atoms",
    notes_col: Optional[str] = "notes",
    skip_rows: int = 0,
) -> List[SaturationRateData]:
    """
    Import saturation rates from spreadsheet/CSV file.
    
    Supports flexible column mapping for various spreadsheet formats.
    
    Args:
        path: Path to CSV file
        delimiter: Column delimiter
        reaction_col: Column name for reaction identifier
        rate_col: Column name for saturation rate
        unc_col: Column name for uncertainty
        halflife_col: Optional column for half-life
        atoms_col: Optional column for target atoms
        notes_col: Optional column for notes
        skip_rows: Number of header rows to skip
        
    Returns:
        List of SaturationRateData objects
    """
    path = Path(path)
    
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        # Skip header rows if needed
        for _ in range(skip_rows):
            next(f)
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        results = []
        for row in reader:
            # Required fields
            if reaction_col not in row or rate_col not in row:
                continue
            
            reaction_id = row[reaction_col].strip()
            if not reaction_id:
                continue
            
            try:
                rate = float(row[rate_col])
            except (ValueError, KeyError):
                continue
            
            # Uncertainty (required)
            try:
                unc = float(row.get(unc_col, 0))
            except (ValueError, TypeError):
                unc = 0.0
            
            # Optional fields
            half_life = 0.0
            if halflife_col and halflife_col in row:
                try:
                    half_life = float(row[halflife_col])
                except (ValueError, TypeError):
                    pass
            
            atoms = 0.0
            if atoms_col and atoms_col in row:
                try:
                    atoms = float(row[atoms_col])
                except (ValueError, TypeError):
                    pass
            
            notes = ""
            if notes_col and notes_col in row:
                notes = row[notes_col].strip()
            
            results.append(SaturationRateData(
                reaction_id=reaction_id,
                rate=rate,
                uncertainty=unc,
                half_life_s=half_life,
                target_atoms=atoms,
                notes=notes,
            ))
    
    return results


def write_saturation_rates_csv(
    data: List[SaturationRateData],
    path: Union[str, Path],
    delimiter: str = ",",
) -> None:
    """
    Export saturation rates to CSV file.
    
    Args:
        data: List of saturation rate data
        path: Output file path
        delimiter: Column delimiter
    """
    path = Path(path)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        
        # Header
        writer.writerow([
            "reaction", "rate", "uncertainty", 
            "half_life_s", "target_atoms", "notes"
        ])
        
        # Data rows
        for d in data:
            writer.writerow([
                d.reaction_id,
                f"{d.rate:.6E}",
                f"{d.uncertainty:.6E}",
                f"{d.half_life_s:.6E}" if d.half_life_s > 0 else "",
                f"{d.target_atoms:.6E}" if d.target_atoms > 0 else "",
                d.notes,
            ])


def read_lower_triangular_matrix(
    path: Union[str, Path],
    n: Optional[int] = None,
    delimiter: Optional[str] = None,
) -> np.ndarray:
    """
    Read a lower-triangular symmetric matrix from legacy format.
    
    Legacy STAYSL format stores symmetric matrices as lower-triangular
    with elements packed row-wise:
    
    ```
    a11
    a21 a22
    a31 a32 a33
    ...
    ```
    
    Args:
        path: Path to matrix file
        n: Matrix dimension (if known). If None, inferred from data.
        delimiter: Column delimiter (whitespace if None)
        
    Returns:
        Full symmetric matrix as 2D numpy array
    """
    path = Path(path)
    
    # Read all values
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()
            
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    pass
    
    # Infer matrix size from number of elements
    # n(n+1)/2 = len(values) => n = (-1 + sqrt(1 + 8*len)) / 2
    if n is None:
        n_elements = len(values)
        n = int((-1 + np.sqrt(1 + 8 * n_elements)) / 2)
        expected = n * (n + 1) // 2
        if expected != n_elements:
            raise ValueError(
                f"Cannot form square matrix from {n_elements} elements. "
                f"Expected {expected} for n={n}."
            )
    
    # Build symmetric matrix
    matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            if idx < len(values):
                matrix[i, j] = values[idx]
                matrix[j, i] = values[idx]  # Symmetric
                idx += 1
    
    return matrix


def write_lower_triangular_matrix(
    matrix: np.ndarray,
    path: Union[str, Path],
    precision: int = 6,
    elements_per_line: int = 6,
) -> None:
    """
    Write a symmetric matrix in lower-triangular format.
    
    Args:
        matrix: Square symmetric matrix
        path: Output file path
        precision: Number of decimal places
        elements_per_line: Max elements per line
    """
    path = Path(path)
    n = matrix.shape[0]
    
    with open(path, 'w') as f:
        elements = []
        for i in range(n):
            for j in range(i + 1):
                elements.append(f"{matrix[i, j]:.{precision}E}")
                
                if len(elements) >= elements_per_line:
                    f.write("  ".join(elements) + "\n")
                    elements = []
        
        # Write remaining elements
        if elements:
            f.write("  ".join(elements) + "\n")


@dataclass
class STAYSLBundle:
    """
    STAYSL-compatible data bundle for cross-tool validation.
    
    Contains all data needed to reproduce a STAYSL analysis.
    """
    
    # Prior spectrum
    prior_flux: np.ndarray
    prior_covariance: np.ndarray
    energy_bounds_eV: np.ndarray
    
    # Measurements
    reaction_ids: List[str]
    measured_rates: np.ndarray
    measurement_covariance: np.ndarray
    
    # Response matrix
    response_matrix: np.ndarray
    
    # Metadata
    title: str = ""
    notes: str = ""
    
    def validate(self) -> List[str]:
        """
        Validate bundle consistency.
        
        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []
        n_groups = len(self.prior_flux)
        n_reactions = len(self.reaction_ids)
        
        # Check dimensions
        if self.prior_covariance.shape != (n_groups, n_groups):
            warnings.append(
                f"Prior covariance shape {self.prior_covariance.shape} "
                f"doesn't match {n_groups} groups"
            )
        
        if len(self.energy_bounds_eV) != n_groups + 1:
            warnings.append(
                f"Energy bounds length {len(self.energy_bounds_eV)} "
                f"should be {n_groups + 1}"
            )
        
        if len(self.measured_rates) != n_reactions:
            warnings.append(
                f"Measured rates length {len(self.measured_rates)} "
                f"doesn't match {n_reactions} reactions"
            )
        
        if self.measurement_covariance.shape != (n_reactions, n_reactions):
            warnings.append(
                f"Measurement covariance shape {self.measurement_covariance.shape} "
                f"doesn't match {n_reactions} reactions"
            )
        
        if self.response_matrix.shape != (n_reactions, n_groups):
            warnings.append(
                f"Response matrix shape {self.response_matrix.shape} "
                f"should be ({n_reactions}, {n_groups})"
            )
        
        return warnings


def export_staysl_bundle(
    bundle: STAYSLBundle,
    output_dir: Union[str, Path],
    prefix: str = "staysl",
) -> Dict[str, Path]:
    """
    Export STAYSL-compatible bundle to directory.
    
    Creates individual files for each component in STAYSL-readable formats.
    
    Args:
        bundle: STAYSLBundle to export
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict mapping component names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    # Prior flux (group-wise)
    flux_file = output_dir / f"{prefix}_prior_flux.txt"
    n_groups = len(bundle.prior_flux)
    with open(flux_file, 'w') as f:
        f.write(f"# Prior flux spectrum ({n_groups} groups)\n")
        f.write("# Group  E_low(eV)  E_high(eV)  Flux\n")
        for g in range(n_groups):
            f.write(
                f"{g+1:4d}  {bundle.energy_bounds_eV[g]:.6E}  "
                f"{bundle.energy_bounds_eV[g+1]:.6E}  {bundle.prior_flux[g]:.6E}\n"
            )
    files["prior_flux"] = flux_file
    
    # Prior covariance (lower triangular)
    prior_cov_file = output_dir / f"{prefix}_prior_cov.txt"
    write_lower_triangular_matrix(bundle.prior_covariance, prior_cov_file)
    files["prior_covariance"] = prior_cov_file
    
    # Measurements
    meas_file = output_dir / f"{prefix}_measurements.csv"
    with open(meas_file, 'w') as f:
        f.write("reaction,rate,uncertainty\n")
        n_reactions = len(bundle.reaction_ids)
        meas_unc = np.sqrt(np.diag(bundle.measurement_covariance))
        for i in range(n_reactions):
            f.write(
                f"{bundle.reaction_ids[i]},{bundle.measured_rates[i]:.6E},"
                f"{meas_unc[i]:.6E}\n"
            )
    files["measurements"] = meas_file
    
    # Measurement covariance
    meas_cov_file = output_dir / f"{prefix}_meas_cov.txt"
    write_lower_triangular_matrix(bundle.measurement_covariance, meas_cov_file)
    files["measurement_covariance"] = meas_cov_file
    
    # Response matrix
    resp_file = output_dir / f"{prefix}_response.csv"
    with open(resp_file, 'w') as f:
        # Header: reaction,G1,G2,...
        header = ["reaction"] + [f"G{g+1}" for g in range(n_groups)]
        f.write(",".join(header) + "\n")
        for i, rxn_id in enumerate(bundle.reaction_ids):
            row = [rxn_id] + [f"{bundle.response_matrix[i,g]:.6E}" for g in range(n_groups)]
            f.write(",".join(row) + "\n")
    files["response_matrix"] = resp_file
    
    # Metadata
    meta_file = output_dir / f"{prefix}_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump({
            "title": bundle.title,
            "notes": bundle.notes,
            "n_groups": n_groups,
            "n_reactions": len(bundle.reaction_ids),
            "energy_min_eV": float(bundle.energy_bounds_eV[0]),
            "energy_max_eV": float(bundle.energy_bounds_eV[-1]),
            "reaction_ids": bundle.reaction_ids,
        }, f, indent=2)
    files["metadata"] = meta_file
    
    return files


def import_staysl_bundle(
    input_dir: Union[str, Path],
    prefix: str = "staysl",
) -> STAYSLBundle:
    """
    Import STAYSL-compatible bundle from directory.
    
    Args:
        input_dir: Input directory containing bundle files
        prefix: Filename prefix
        
    Returns:
        STAYSLBundle instance
    """
    input_dir = Path(input_dir)
    
    # Read metadata first
    meta_file = input_dir / f"{prefix}_metadata.json"
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    n_groups = metadata["n_groups"]
    reaction_ids = metadata["reaction_ids"]
    n_reactions = len(reaction_ids)
    
    # Read prior flux
    flux_file = input_dir / f"{prefix}_prior_flux.txt"
    prior_flux = np.zeros(n_groups)
    energy_bounds = np.zeros(n_groups + 1)
    
    with open(flux_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                g = int(parts[0]) - 1
                if 0 <= g < n_groups:
                    energy_bounds[g] = float(parts[1])
                    energy_bounds[g+1] = float(parts[2])
                    prior_flux[g] = float(parts[3])
    
    # Read prior covariance
    prior_cov_file = input_dir / f"{prefix}_prior_cov.txt"
    prior_cov = read_lower_triangular_matrix(prior_cov_file, n=n_groups)
    
    # Read measurements
    meas_file = input_dir / f"{prefix}_measurements.csv"
    measured_rates = np.zeros(n_reactions)
    meas_unc = np.zeros(n_reactions)
    
    with open(meas_file, 'r') as f:
        reader = csv.DictReader(f)
        rxn_to_idx = {r: i for i, r in enumerate(reaction_ids)}
        for row in reader:
            rxn = row["reaction"]
            if rxn in rxn_to_idx:
                idx = rxn_to_idx[rxn]
                measured_rates[idx] = float(row["rate"])
                meas_unc[idx] = float(row["uncertainty"])
    
    # Read measurement covariance
    meas_cov_file = input_dir / f"{prefix}_meas_cov.txt"
    meas_cov = read_lower_triangular_matrix(meas_cov_file, n=n_reactions)
    
    # Read response matrix
    resp_file = input_dir / f"{prefix}_response.csv"
    response = np.zeros((n_reactions, n_groups))
    
    with open(resp_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxn = row["reaction"]
            if rxn in rxn_to_idx:
                idx = rxn_to_idx[rxn]
                for g in range(n_groups):
                    response[idx, g] = float(row[f"G{g+1}"])
    
    return STAYSLBundle(
        prior_flux=prior_flux,
        prior_covariance=prior_cov,
        energy_bounds_eV=energy_bounds,
        reaction_ids=reaction_ids,
        measured_rates=measured_rates,
        measurement_covariance=meas_cov,
        response_matrix=response,
        title=metadata.get("title", ""),
        notes=metadata.get("notes", ""),
    )


def convert_excel_to_saturation_data(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **column_mapping,
) -> List[SaturationRateData]:
    """
    Convert Excel spreadsheet to saturation rate data.
    
    Requires pandas and openpyxl.
    
    Args:
        path: Path to Excel file
        sheet_name: Sheet name or index
        **column_mapping: Column name mappings (reaction_col, rate_col, etc.)
        
    Returns:
        List of SaturationRateData objects
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Excel import")
    
    df = pd.read_excel(path, sheet_name=sheet_name)
    
    # Save to temp CSV and use existing reader
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    try:
        return read_saturation_rates_csv(temp_path, **column_mapping)
    finally:
        Path(temp_path).unlink()
