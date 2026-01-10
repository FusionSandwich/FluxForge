"""Activation analysis pipeline for end-to-end workflow.

This module provides the ActivationPipeline class for orchestrating
the complete FluxForge → ALARA → Validation workflow, enabling
automated flux unfolding and activation analysis.

Pipeline Stages:
1. Spectrum unfolding (GLS/GRAVEL)
2. ALARA input generation
3. ALARA execution (subprocess)
4. Result parsing and comparison
5. Validation report generation

References:
    ALARA User Manual
    PNNL-22253 STAYSL PNNL User Manual
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np


@dataclass
class ALARAConfig:
    """Configuration for ALARA execution.
    
    Attributes
    ----------
    executable : str
        Path to ALARA executable (default: "alara")
    lib_path : str
        Path to ALARA data library directory
    timeout_s : float
        Execution timeout in seconds
    verbose : bool
        Enable verbose output
    working_dir : Path, optional
        Working directory for ALARA run
    """
    
    executable: str = "alara"
    lib_path: str = ""
    timeout_s: float = 600.0
    verbose: bool = False
    working_dir: Optional[Path] = None
    
    def find_executable(self) -> Optional[str]:
        """Find ALARA executable on PATH."""
        exe = shutil.which(self.executable)
        return exe


@dataclass
class ActivationResult:
    """Result from activation calculation.
    
    Attributes
    ----------
    isotope : str
        Activated isotope (e.g., "Co60")
    activity_Bq : float
        Calculated activity in Bq
    activity_unc : float
        Activity uncertainty
    half_life_s : float
        Half-life in seconds
    gamma_energies : list[float]
        Characteristic gamma energies (keV)
    source : str
        Source of result ("ALARA", "measured", etc.)
    metadata : dict
        Additional metadata
    """
    
    isotope: str
    activity_Bq: float
    activity_unc: float = 0.0
    half_life_s: float = 0.0
    gamma_energies: list[float] = field(default_factory=list)
    source: str = ""
    metadata: dict = field(default_factory=dict)
    
    @property
    def activity_Ci(self) -> float:
        """Activity in Curies."""
        return self.activity_Bq / 3.7e10
    
    @property
    def specific_activity(self) -> float:
        """Specific activity (Bq/g) if mass in metadata."""
        mass_g = self.metadata.get("mass_g", 0)
        if mass_g > 0:
            return self.activity_Bq / mass_g
        return 0.0


@dataclass
class ActivationComparison:
    """Comparison between calculated and measured activities.
    
    Attributes
    ----------
    isotope : str
        Isotope name
    calculated : ActivationResult
        ALARA calculation result
    measured : ActivationResult
        Measured activity
    c_over_e : float
        Calculated-to-Experimental ratio
    c_over_e_unc : float
        C/E uncertainty
    within_tolerance : bool
        Whether C/E is within acceptable range
    """
    
    isotope: str
    calculated: ActivationResult
    measured: ActivationResult
    c_over_e: float = 0.0
    c_over_e_unc: float = 0.0
    within_tolerance: bool = False
    
    def __post_init__(self):
        """Compute C/E."""
        if self.measured.activity_Bq > 0:
            self.c_over_e = self.calculated.activity_Bq / self.measured.activity_Bq
            
            # Uncertainty propagation
            rel_c = self.calculated.activity_unc / self.calculated.activity_Bq if self.calculated.activity_Bq > 0 else 0
            rel_e = self.measured.activity_unc / self.measured.activity_Bq if self.measured.activity_Bq > 0 else 0
            self.c_over_e_unc = self.c_over_e * np.sqrt(rel_c**2 + rel_e**2)
            
            # Default tolerance: within 2σ of unity
            self.within_tolerance = abs(self.c_over_e - 1.0) <= 2 * self.c_over_e_unc


@dataclass
class PipelineResult:
    """Complete pipeline execution result.
    
    Attributes
    ----------
    success : bool
        Whether pipeline completed successfully
    spectrum_file : Path
        Generated spectrum file
    alara_input : Path
        Generated ALARA input file
    alara_output : Path
        ALARA output file
    activation_results : list[ActivationResult]
        Calculated activation results
    comparisons : list[ActivationComparison]
        Comparison to measurements (if available)
    execution_time_s : float
        Total execution time
    messages : list[str]
        Status/error messages
    provenance : dict
        Provenance information
    """
    
    success: bool = False
    spectrum_file: Optional[Path] = None
    alara_input: Optional[Path] = None
    alara_output: Optional[Path] = None
    activation_results: list[ActivationResult] = field(default_factory=list)
    comparisons: list[ActivationComparison] = field(default_factory=list)
    execution_time_s: float = 0.0
    messages: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    
    @property
    def mean_c_over_e(self) -> float:
        """Mean C/E ratio across all comparisons."""
        if not self.comparisons:
            return float('nan')
        return float(np.mean([c.c_over_e for c in self.comparisons]))
    
    @property
    def all_within_tolerance(self) -> bool:
        """Whether all comparisons are within tolerance."""
        if not self.comparisons:
            return True
        return all(c.within_tolerance for c in self.comparisons)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "ACTIVATION PIPELINE RESULT",
            "=" * 60,
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Execution time: {self.execution_time_s:.2f} s",
            "",
        ]
        
        if self.activation_results:
            lines.append("Calculated Activities:")
            lines.append("-" * 60)
            for ar in self.activation_results:
                lines.append(f"  {ar.isotope}: {ar.activity_Bq:.4e} ± {ar.activity_unc:.4e} Bq")
        
        if self.comparisons:
            lines.append("")
            lines.append("Comparisons (C/E):")
            lines.append("-" * 60)
            for c in self.comparisons:
                status = "OK" if c.within_tolerance else "CHECK"
                lines.append(f"  {c.isotope}: {c.c_over_e:.4f} ± {c.c_over_e_unc:.4f} [{status}]")
            lines.append(f"Mean C/E: {self.mean_c_over_e:.4f}")
        
        if self.messages:
            lines.append("")
            lines.append("Messages:")
            for msg in self.messages:
                lines.append(f"  {msg}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ActivationPipeline:
    """End-to-end activation analysis pipeline.
    
    Orchestrates the complete workflow from flux unfolding through
    ALARA activation calculation and result validation.
    
    Parameters
    ----------
    alara_config : ALARAConfig
        ALARA execution configuration
    working_dir : Path
        Working directory for files
    """
    
    def __init__(
        self,
        alara_config: Optional[ALARAConfig] = None,
        working_dir: Optional[Path] = None,
    ):
        self.alara_config = alara_config or ALARAConfig()
        self.working_dir = working_dir or Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_alara_flux(
        self,
        spectrum: np.ndarray,
        energy_grid: np.ndarray,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate ALARA flux file from spectrum.
        
        Parameters
        ----------
        spectrum : ndarray
            Flux values per energy group
        energy_grid : ndarray
            Energy bin boundaries
        output_file : Path, optional
            Output file path
            
        Returns
        -------
        Path
            Generated flux file path
        """
        if output_file is None:
            output_file = self.working_dir / "flux.flx"
        
        n_groups = len(spectrum)
        
        with open(output_file, 'w') as f:
            f.write(f"# ALARA flux file generated by FluxForge\n")
            f.write(f"# {n_groups} groups\n")
            f.write("flux_1\n")
            
            # Write flux values (ALARA format: one per line)
            for i, flux in enumerate(spectrum):
                f.write(f"  {flux:.6e}\n")
        
        return output_file
    
    def generate_alara_input(
        self,
        template: str,
        flux_file: Path,
        material: str,
        irradiation_schedule: str,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate ALARA input file.
        
        Parameters
        ----------
        template : str
            Base template or 'default'
        flux_file : Path
            Path to flux file
        material : str
            Material specification
        irradiation_schedule : str
            Irradiation history specification
        output_file : Path, optional
            Output file path
            
        Returns
        -------
        Path
            Generated input file path
        """
        if output_file is None:
            output_file = self.working_dir / "alara.inp"
        
        # Generate minimal ALARA input
        content = f"""# ALARA input generated by FluxForge
geometry rectangular

material mat_1
    {material}
end

cooling
    1.0 s
end

flux flux_1 {flux_file} 1.0 0

schedule sched_1
    {irradiation_schedule}
end

pulsehistory hist_1
    sched_1
end

output zone
    units Bq
    specific_activity
end

dump_file dump.out

data_library alaralib {self.alara_config.lib_path}/fendl2bin

mat_loading
    zone_1 mat_1
end

spatial_norm 1.0

zone zone_1
    volume 1.0
end

solve_spatial
end
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return output_file
    
    def run_alara(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
    ) -> tuple[bool, str, Path]:
        """Execute ALARA calculation.
        
        Parameters
        ----------
        input_file : Path
            ALARA input file
        output_file : Path, optional
            Output file path
            
        Returns
        -------
        tuple[bool, str, Path]
            (success, stdout/stderr, output_file)
        """
        if output_file is None:
            output_file = input_file.with_suffix('.output')
        
        exe = self.alara_config.find_executable()
        if exe is None:
            return False, f"ALARA executable not found: {self.alara_config.executable}", output_file
        
        cmd = [exe, str(input_file)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.alara_config.timeout_s,
                cwd=self.working_dir,
            )
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            
            success = result.returncode == 0
            return success, result.stdout + result.stderr, output_file
            
        except subprocess.TimeoutExpired:
            return False, f"ALARA timed out after {self.alara_config.timeout_s}s", output_file
        except Exception as e:
            return False, f"ALARA execution failed: {e}", output_file
    
    def parse_alara_output(
        self,
        output_file: Path,
    ) -> list[ActivationResult]:
        """Parse ALARA output file for activities.
        
        Parameters
        ----------
        output_file : Path
            ALARA output file
            
        Returns
        -------
        list[ActivationResult]
            Parsed activation results
        """
        results = []
        
        if not output_file.exists():
            return results
        
        # Simple parsing - look for activity lines
        # Format varies by ALARA output options
        content = output_file.read_text()
        
        import re
        
        # Pattern: isotope name followed by activity value
        # Example: "Co-60    1.234e+05 Bq"
        pattern = r'([A-Z][a-z]?-\d+)\s+(\d+\.?\d*[eE][+-]?\d+)\s*(?:Bq|Ci)'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            isotope = match.group(1)
            activity = float(match.group(2))
            
            results.append(ActivationResult(
                isotope=isotope,
                activity_Bq=activity,
                source="ALARA",
            ))
        
        return results
    
    def run_pipeline(
        self,
        spectrum: np.ndarray,
        energy_grid: np.ndarray,
        material: str,
        irradiation_schedule: str,
        measurements: Optional[list[ActivationResult]] = None,
    ) -> PipelineResult:
        """Execute complete pipeline.
        
        Parameters
        ----------
        spectrum : ndarray
            Unfolded flux spectrum
        energy_grid : ndarray
            Energy bin boundaries
        material : str
            Material specification for ALARA
        irradiation_schedule : str
            Irradiation history
        measurements : list[ActivationResult], optional
            Measured activities for comparison
            
        Returns
        -------
        PipelineResult
            Complete pipeline result
        """
        import time
        start_time = time.time()
        
        result = PipelineResult()
        
        try:
            # Step 1: Generate flux file
            result.spectrum_file = self.generate_alara_flux(
                spectrum, energy_grid
            )
            result.messages.append(f"Generated flux file: {result.spectrum_file}")
            
            # Step 2: Generate ALARA input
            result.alara_input = self.generate_alara_input(
                template="default",
                flux_file=result.spectrum_file,
                material=material,
                irradiation_schedule=irradiation_schedule,
            )
            result.messages.append(f"Generated ALARA input: {result.alara_input}")
            
            # Step 3: Run ALARA
            success, output, result.alara_output = self.run_alara(
                result.alara_input
            )
            
            if not success:
                result.messages.append(f"ALARA failed: {output[:500]}")
                result.success = False
                return result
            
            result.messages.append("ALARA completed successfully")
            
            # Step 4: Parse results
            result.activation_results = self.parse_alara_output(
                result.alara_output
            )
            result.messages.append(f"Parsed {len(result.activation_results)} isotopes")
            
            # Step 5: Compare to measurements if provided
            if measurements:
                for meas in measurements:
                    # Find matching calculation
                    calc = None
                    for ar in result.activation_results:
                        # Normalize isotope names for comparison
                        if ar.isotope.replace("-", "").lower() == meas.isotope.replace("-", "").lower():
                            calc = ar
                            break
                    
                    if calc:
                        result.comparisons.append(ActivationComparison(
                            isotope=meas.isotope,
                            calculated=calc,
                            measured=meas,
                        ))
            
            result.success = True
            
        except Exception as e:
            result.messages.append(f"Pipeline error: {e}")
            result.success = False
        
        finally:
            result.execution_time_s = time.time() - start_time
        
        return result


__all__ = [
    "ALARAConfig",
    "ActivationResult",
    "ActivationComparison",
    "PipelineResult",
    "ActivationPipeline",
]
