"""Core data structures and utilities."""

from fluxforge.core.prior_covariance import (
	PriorCovarianceConfig,
	PriorCovarianceModel,
	ResponseUncertaintyConfig,
	ResponseUncertaintyPolicy,
)
from fluxforge.core.response import (
	EnergyGroupStructure,
	ReactionCrossSection,
	ResponseMatrix,
	build_response_matrix,
)
from fluxforge.core.sample import Container, Cover, MaterialComponent, Sample
from fluxforge.core.validation import (
	CEEntry,
	CETable,
	ClosureMetrics,
	ValidationBundle,
	ValidationStatus,
	calculate_ce_table,
	calculate_closure_metrics,
	create_validation_bundle,
)

__all__ = [
	"EnergyGroupStructure",
	"ReactionCrossSection",
	"ResponseMatrix",
	"build_response_matrix",
	"PriorCovarianceConfig",
	"PriorCovarianceModel",
	"ResponseUncertaintyConfig",
	"ResponseUncertaintyPolicy",
	"Sample",
	"MaterialComponent",
	"Cover",
	"Container",
	# Validation
	"CEEntry",
	"CETable",
	"ClosureMetrics",
	"ValidationBundle",
	"ValidationStatus",
	"calculate_ce_table",
	"calculate_closure_metrics",
	"create_validation_bundle",
]
