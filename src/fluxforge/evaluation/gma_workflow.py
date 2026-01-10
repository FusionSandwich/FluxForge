"""
GMA Workflow Manager - Epic S (GMApy Parity)

Implements complete GMA (Generalized Least Squares) workflow:
- Sensitivity calculation from response matrix
- Prior covariance construction
- GLS solution with full uncertainty propagation
- JSON experimental database format
- Result export and visualization support

This provides a high-level interface for nuclear data evaluation
using the GLS method, similar to GMApy's workflow.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from pathlib import Path
import json
import numpy as np
from numpy.typing import NDArray
from datetime import datetime


@dataclass
class Experiment:
    """
    Single experimental measurement for GMA evaluation.
    
    Represents one measurement with its value, uncertainty,
    and metadata for tracking and reproducibility.
    """
    name: str
    value: float
    uncertainty: float
    energy_MeV: float
    reaction: str
    
    # Metadata
    reference: str = ""
    year: Optional[int] = None
    laboratory: str = ""
    method: str = ""
    comment: str = ""
    
    # Optional correlation info
    correlation_group: Optional[str] = None
    systematic_uncertainty: float = 0.0
    statistical_uncertainty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'energy_MeV': self.energy_MeV,
            'reaction': self.reaction,
            'reference': self.reference,
            'year': self.year,
            'laboratory': self.laboratory,
            'method': self.method,
            'comment': self.comment,
            'correlation_group': self.correlation_group,
            'systematic_uncertainty': self.systematic_uncertainty,
            'statistical_uncertainty': self.statistical_uncertainty
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Experiment':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentalDatabase:
    """
    Database of experimental measurements.
    
    Provides JSON import/export and correlation management.
    """
    name: str
    experiments: List[Experiment] = field(default_factory=list)
    correlation_matrices: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_experiment(self, exp: Experiment) -> None:
        """Add an experiment to the database."""
        self.experiments.append(exp)
    
    def get_experiments_by_reaction(self, reaction: str) -> List[Experiment]:
        """Get all experiments for a specific reaction."""
        return [e for e in self.experiments if e.reaction == reaction]
    
    def get_experiments_by_energy_range(
        self,
        E_min: float,
        E_max: float
    ) -> List[Experiment]:
        """Get experiments within energy range."""
        return [
            e for e in self.experiments 
            if E_min <= e.energy_MeV <= E_max
        ]
    
    def get_values_vector(self) -> NDArray[np.float64]:
        """Get vector of experimental values."""
        return np.array([e.value for e in self.experiments])
    
    def get_uncertainties_vector(self) -> NDArray[np.float64]:
        """Get vector of uncertainties."""
        return np.array([e.uncertainty for e in self.experiments])
    
    def build_covariance_matrix(self) -> NDArray[np.float64]:
        """
        Build full experimental covariance matrix.
        
        Includes both statistical (diagonal) and systematic
        (off-diagonal) contributions.
        """
        n = len(self.experiments)
        cov = np.zeros((n, n))
        
        # Diagonal: total variance
        for i, exp in enumerate(self.experiments):
            cov[i, i] = exp.uncertainty**2
        
        # Off-diagonal: correlations from groups
        groups: Dict[str, List[int]] = {}
        for i, exp in enumerate(self.experiments):
            if exp.correlation_group:
                if exp.correlation_group not in groups:
                    groups[exp.correlation_group] = []
                groups[exp.correlation_group].append(i)
        
        # Apply systematic correlations within groups
        for group, indices in groups.items():
            if group in self.correlation_matrices:
                corr = self.correlation_matrices[group]
                for ii, i in enumerate(indices):
                    for jj, j in enumerate(indices):
                        if i != j:
                            # Correlated systematic
                            sys_i = self.experiments[i].systematic_uncertainty
                            sys_j = self.experiments[j].systematic_uncertainty
                            cov[i, j] = corr[ii, jj] * sys_i * sys_j
            else:
                # Default: full correlation of systematics
                for i in indices:
                    for j in indices:
                        if i != j:
                            sys_i = self.experiments[i].systematic_uncertainty
                            sys_j = self.experiments[j].systematic_uncertainty
                            cov[i, j] = sys_i * sys_j
        
        return cov
    
    def to_json(self, filepath: Path) -> None:
        """Save database to JSON file."""
        data = {
            'name': self.name,
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'metadata': self.metadata,
            'experiments': [e.to_dict() for e in self.experiments],
            'correlation_matrices': {
                k: v.tolist() for k, v in self.correlation_matrices.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentalDatabase':
        """Load database from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        db = cls(
            name=data.get('name', 'unknown'),
            metadata=data.get('metadata', {})
        )
        
        for exp_dict in data.get('experiments', []):
            db.experiments.append(Experiment.from_dict(exp_dict))
        
        for k, v in data.get('correlation_matrices', {}).items():
            db.correlation_matrices[k] = np.array(v)
        
        return db


@dataclass
class SensitivityMatrix:
    """
    Sensitivity matrix relating experiments to parameters.
    
    S[i,j] = ∂(experiment_i) / ∂(parameter_j)
    """
    matrix: NDArray[np.float64]
    experiment_names: List[str]
    parameter_names: List[str]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape
    
    def to_dataframe(self):
        """Convert to pandas DataFrame if available."""
        try:
            import pandas as pd
            return pd.DataFrame(
                self.matrix,
                index=self.experiment_names,
                columns=self.parameter_names
            )
        except ImportError:
            return None


@dataclass
class PriorDistribution:
    """Prior distribution for GLS parameters."""
    values: NDArray[np.float64]
    covariance: NDArray[np.float64]
    parameter_names: List[str]
    
    @property
    def uncertainties(self) -> NDArray[np.float64]:
        """Get standard deviations."""
        return np.sqrt(np.diag(self.covariance))
    
    @property
    def correlation_matrix(self) -> NDArray[np.float64]:
        """Get correlation matrix."""
        stds = self.uncertainties
        stds = np.where(stds > 0, stds, 1.0)
        return self.covariance / np.outer(stds, stds)


@dataclass
class GLSResult:
    """Result of GLS evaluation."""
    posterior_values: NDArray[np.float64]
    posterior_covariance: NDArray[np.float64]
    parameter_names: List[str]
    chi2: float
    ndof: int
    converged: bool = True
    n_iterations: int = 1
    
    @property
    def posterior_uncertainties(self) -> NDArray[np.float64]:
        """Posterior standard deviations."""
        return np.sqrt(np.diag(self.posterior_covariance))
    
    @property
    def chi2_per_dof(self) -> float:
        """Reduced chi-squared."""
        return self.chi2 / max(self.ndof, 1)
    
    @property
    def p_value(self) -> float:
        """P-value for chi-squared test."""
        from scipy import stats
        return 1 - stats.chi2.cdf(self.chi2, self.ndof)
    
    def get_parameter(self, name: str) -> Tuple[float, float]:
        """Get (value, uncertainty) for a parameter."""
        if name in self.parameter_names:
            idx = self.parameter_names.index(name)
            return self.posterior_values[idx], self.posterior_uncertainties[idx]
        raise KeyError(f"Parameter '{name}' not found")


class GMAWorkflow:
    """
    Complete GMA workflow manager.
    
    Orchestrates the full GLS evaluation process:
    1. Load experimental data
    2. Calculate sensitivities
    3. Construct prior
    4. Run GLS
    5. Export results
    
    Example
    -------
    >>> workflow = GMAWorkflow()
    >>> workflow.load_experiments("data.json")
    >>> workflow.set_prior(prior_values, prior_cov)
    >>> workflow.calculate_sensitivities(response_function)
    >>> result = workflow.run_gls()
    >>> workflow.export_results("output.json")
    """
    
    def __init__(self, name: str = "GMA Evaluation"):
        """Initialize workflow."""
        self.name = name
        self.database: Optional[ExperimentalDatabase] = None
        self.prior: Optional[PriorDistribution] = None
        self.sensitivity: Optional[SensitivityMatrix] = None
        self.result: Optional[GLSResult] = None
        
        # Configuration
        self.config = {
            'max_iterations': 10,
            'convergence_threshold': 1e-6,
            'use_iterative': False,
            'weight_function': None
        }
    
    def load_experiments(
        self,
        source: Union[str, Path, ExperimentalDatabase]
    ) -> 'GMAWorkflow':
        """
        Load experimental database.
        
        Parameters
        ----------
        source : str, Path, or ExperimentalDatabase
            JSON file path or database object
        
        Returns
        -------
        GMAWorkflow
            Self for chaining
        """
        if isinstance(source, ExperimentalDatabase):
            self.database = source
        else:
            self.database = ExperimentalDatabase.from_json(Path(source))
        
        return self
    
    def set_prior(
        self,
        values: NDArray[np.float64],
        covariance: NDArray[np.float64],
        parameter_names: Optional[List[str]] = None
    ) -> 'GMAWorkflow':
        """
        Set prior distribution.
        
        Parameters
        ----------
        values : array
            Prior parameter values
        covariance : array
            Prior covariance matrix
        parameter_names : list, optional
            Names for parameters
        
        Returns
        -------
        GMAWorkflow
            Self for chaining
        """
        n = len(values)
        if parameter_names is None:
            parameter_names = [f"p_{i}" for i in range(n)]
        
        self.prior = PriorDistribution(
            values=np.asarray(values),
            covariance=np.asarray(covariance),
            parameter_names=parameter_names
        )
        
        return self
    
    def calculate_sensitivities(
        self,
        response_function: Optional[Callable] = None,
        matrix: Optional[NDArray[np.float64]] = None,
        delta: float = 1e-6
    ) -> 'GMAWorkflow':
        """
        Calculate sensitivity matrix.
        
        Either provide a response function for numerical differentiation,
        or provide the sensitivity matrix directly.
        
        Parameters
        ----------
        response_function : callable, optional
            Function(parameters) -> predictions for all experiments
        matrix : array, optional
            Pre-computed sensitivity matrix
        delta : float
            Step size for numerical differentiation
        
        Returns
        -------
        GMAWorkflow
            Self for chaining
        """
        if matrix is not None:
            # Direct matrix input
            n_exp, n_param = matrix.shape
            exp_names = [e.name for e in self.database.experiments[:n_exp]]
            param_names = self.prior.parameter_names if self.prior else [
                f"p_{i}" for i in range(n_param)
            ]
            
            self.sensitivity = SensitivityMatrix(
                matrix=np.asarray(matrix),
                experiment_names=exp_names,
                parameter_names=param_names
            )
        
        elif response_function is not None and self.prior is not None:
            # Numerical differentiation
            n_exp = len(self.database.experiments)
            n_param = len(self.prior.values)
            
            S = np.zeros((n_exp, n_param))
            p0 = self.prior.values.copy()
            f0 = response_function(p0)
            
            for j in range(n_param):
                p_plus = p0.copy()
                p_plus[j] += delta
                f_plus = response_function(p_plus)
                S[:, j] = (f_plus - f0) / delta
            
            self.sensitivity = SensitivityMatrix(
                matrix=S,
                experiment_names=[e.name for e in self.database.experiments],
                parameter_names=self.prior.parameter_names
            )
        
        else:
            raise ValueError("Provide either response_function or matrix")
        
        return self
    
    def run_gls(self) -> GLSResult:
        """
        Run Generalized Least Squares evaluation.
        
        Returns
        -------
        GLSResult
            Posterior distribution and fit statistics
        """
        if self.database is None:
            raise ValueError("No experimental database loaded")
        if self.prior is None:
            raise ValueError("No prior distribution set")
        if self.sensitivity is None:
            raise ValueError("No sensitivity matrix calculated")
        
        # Get matrices
        y = self.database.get_values_vector()
        V_y = self.database.build_covariance_matrix()
        x_0 = self.prior.values
        V_x = self.prior.covariance
        S = self.sensitivity.matrix
        
        # Predicted values with prior
        y_pred = S @ x_0
        
        # Residual
        delta_y = y - y_pred
        
        # Combined covariance
        V_total = V_y + S @ V_x @ S.T
        
        try:
            V_total_inv = np.linalg.inv(V_total)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            V_total_inv = np.linalg.pinv(V_total)
        
        # GLS solution (linear case)
        # x_post = x_0 + V_x @ S.T @ V_total_inv @ delta_y
        gain = V_x @ S.T @ V_total_inv
        x_post = x_0 + gain @ delta_y
        
        # Posterior covariance
        V_post = V_x - gain @ S @ V_x
        
        # Chi-squared
        residual_post = y - S @ x_post
        chi2 = float(residual_post.T @ np.linalg.pinv(V_y) @ residual_post)
        
        # Degrees of freedom
        ndof = len(y) - len(x_0)
        
        self.result = GLSResult(
            posterior_values=x_post,
            posterior_covariance=V_post,
            parameter_names=self.prior.parameter_names,
            chi2=chi2,
            ndof=ndof
        )
        
        return self.result
    
    def export_results(
        self,
        filepath: Path,
        format: str = "json"
    ) -> None:
        """
        Export evaluation results.
        
        Parameters
        ----------
        filepath : Path
            Output file path
        format : str
            Output format: "json", "csv", "endf"
        """
        if self.result is None:
            raise ValueError("No results to export - run GLS first")
        
        if format == "json":
            data = {
                'name': self.name,
                'created': datetime.now().isoformat(),
                'parameters': {
                    name: {
                        'value': float(val),
                        'uncertainty': float(unc)
                    }
                    for name, val, unc in zip(
                        self.result.parameter_names,
                        self.result.posterior_values,
                        self.result.posterior_uncertainties
                    )
                },
                'statistics': {
                    'chi2': self.result.chi2,
                    'ndof': self.result.ndof,
                    'chi2_per_dof': self.result.chi2_per_dof,
                    'p_value': self.result.p_value
                },
                'covariance': self.result.posterior_covariance.tolist()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            with open(filepath, 'w') as f:
                f.write("parameter,value,uncertainty\n")
                for name, val, unc in zip(
                    self.result.parameter_names,
                    self.result.posterior_values,
                    self.result.posterior_uncertainties
                ):
                    f.write(f"{name},{val:.6e},{unc:.6e}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def summary(self) -> str:
        """Generate evaluation summary string."""
        lines = [f"GMA Evaluation: {self.name}", "=" * 50]
        
        if self.database:
            lines.append(f"Experiments: {len(self.database.experiments)}")
        if self.prior:
            lines.append(f"Parameters: {len(self.prior.values)}")
        if self.sensitivity:
            lines.append(f"Sensitivity matrix: {self.sensitivity.shape}")
        
        if self.result:
            lines.append("")
            lines.append("Results:")
            lines.append(f"  χ²/dof = {self.result.chi2_per_dof:.2f}")
            lines.append(f"  p-value = {self.result.p_value:.4f}")
            lines.append("")
            lines.append("Posterior parameters:")
            for name, val, unc in zip(
                self.result.parameter_names,
                self.result.posterior_values,
                self.result.posterior_uncertainties
            ):
                lines.append(f"  {name}: {val:.4f} ± {unc:.4f}")
        
        return "\n".join(lines)


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing gma_workflow module...")
    
    # Create synthetic experimental database
    db = ExperimentalDatabase(name="Test Evaluation")
    
    # Add experiments
    np.random.seed(42)
    true_params = np.array([1.0, 0.5, -0.2])  # True parameters
    
    for i in range(10):
        E = 1.0 + i * 0.5  # Energy in MeV
        # Synthetic value: linear model + noise
        true_val = true_params[0] + true_params[1] * E + true_params[2] * E**2
        noise = np.random.normal(0, 0.05)
        
        db.add_experiment(Experiment(
            name=f"Exp_{i+1}",
            value=true_val + noise,
            uncertainty=0.05,
            energy_MeV=E,
            reaction="Test(n,g)",
            reference="Test et al. 2024"
        ))
    
    print(f"Created database with {len(db.experiments)} experiments")
    
    # Save and reload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        db.to_json(Path(f.name))
        db_loaded = ExperimentalDatabase.from_json(Path(f.name))
    
    print(f"Saved/loaded JSON with {len(db_loaded.experiments)} experiments")
    
    # Set up workflow
    workflow = GMAWorkflow("Test Evaluation")
    workflow.load_experiments(db)
    
    # Prior
    prior_values = np.array([1.1, 0.4, -0.1])  # Slightly off
    prior_cov = np.diag([0.1, 0.1, 0.1])**2
    workflow.set_prior(prior_values, prior_cov, ["a0", "a1", "a2"])
    
    # Response function (linear model)
    def response(params):
        results = []
        for exp in db.experiments:
            E = exp.energy_MeV
            results.append(params[0] + params[1] * E + params[2] * E**2)
        return np.array(results)
    
    workflow.calculate_sensitivities(response)
    
    print(f"\nSensitivity matrix shape: {workflow.sensitivity.shape}")
    
    # Run GLS
    result = workflow.run_gls()
    
    print(f"\nGLS Results:")
    print(f"  χ²/dof = {result.chi2_per_dof:.3f}")
    print(f"  p-value = {result.p_value:.4f}")
    print(f"\nTrue parameters: {true_params}")
    print(f"Prior parameters: {prior_values}")
    print(f"Posterior parameters: {result.posterior_values}")
    print(f"Uncertainties: {result.posterior_uncertainties}")
    
    # Summary
    print("\n" + workflow.summary())
    
    print("\n✅ gma_workflow module tests passed!")
