# Source-bound neutron spectrum GLS

`fluxforge.analysis.unfold_gls_physical` accepts a frozen physical observation
operator. It never builds response rows from reaction labels. Its flux vector is
**group-integral** neutron flux in `n/cm2/s`; the exact, increasing energy edges
are in eV. Each response row is in `cm2` and maps that flux to a reaction rate in
`reactions/target_atom/s`. A group's subdivision must conserve the sum of its
flux integrals and use matching response coefficients.

Provide one `MonitorRow` and response row for every physical observation. The
row identity includes sample, cover, reaction, and product. Bare and Cd foils
may have the same reaction string but must have distinct observation IDs and
their own physically calculated response rows. Duplicate response rows do not
create rank. The API reports rank, nullity, conditioning, and unsupported-group
posterior uncertainty. It does not constrain negative adjusted group values;
their indices are returned explicitly.

Supply the actual prior vector and covariance, full observation covariance,
and optional frozen observation-space response-error covariance. The latter
must already be propagated into squared rate units at the supplied prior; it
is added to the observation covariance during adjustment. Correlated
observations are supported. Covariances must be symmetric and positive
semidefinite; the total fit covariance must be positive definite. The solver
scales flux and each rate row before solving and adds no absolute covariance
floor. It returns both the prior innovation statistic and the postfit residual
statistic. The postfit statistic uses the supplied observation plus response
error covariance and is descriptive for a regularized fit; `effective_residual_df`
is `n_fit - trace(R K)`, not an asserted chi-squared calibration.

Every numerical input and row identity needs a `SourceBinding` with URI,
source-file SHA256, and exact units; `source_commit` is also required. The
receipt separately hashes the numerical arrays actually used and includes
them in full. Source bindings record caller provenance; FluxForge does not
certify their experimental validity or verify remote URI bytes. Review the
bindings and their scientific preparation before admitting results.

Choose `holdout_ids` before fitting. Their rates do not enter the adjustment;
the receipt separates fit IDs from held-out predictions and residuals. The
API conditions the holdout predictive mean and covariance on the fit rows,
including shared observation and response-error covariance. Holdout measured
rates do not enter the fitted flux.

The older `flux_unfold.unfold_gls` remains available only as a **historical
placeholder diagnostic**. It uses reaction-label Gaussian rows, an absolute
covariance floor, a generated equal-lethargy prior, and a pre-update statistic
in `chi2`. The RAFM workflow now records those facts in its output metadata and
plots its actual GLS prior. A new Paper 3 experimental result must use qualified
activities, chronology, physical response, covariance, and frozen holdouts
before calling the source-bound API.
