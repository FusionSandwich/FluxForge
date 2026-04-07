# RAFM Unfolding Solver Fixes

This note records the FluxForge-side changes made while debugging the
`rafm_irradiation_ldrd_copy` neutron-spectrum unfolding workflow against the
current FluxForge codebase.

## Files Changed

- `src/fluxforge/solvers/iterative.py`
- `src/fluxforge/workflows/spectrum_unfolding.py`
- `src/fluxforge/unfolding/gravel.py`
- `tests/test_iterative.py`
- `tests/test_irdff.py`

## Confirmed Solver Issues

The RAFM notebook exposed two real problems in the legacy iterative solver path.

1. `GRAVEL` was computing inverse-variance weights from
   `measurement_uncertainty` and then not using them in the update.
   On inconsistent monitor sets, that gave noisy or contradictory reactions
   more influence than intended.
2. Both `GRAVEL` and `MLEM` were pure multiplicative updates with only a
   positivity floor. On the RAFM bare-wire problem, weakly constrained fast
   bins could collapse to near-zero values even after support masking.

## Implemented Fixes

- `GRAVEL` now applies the inverse-variance weights in its update operator.
- `MLEM` now also uses the provided measurement uncertainties as inverse-variance
  weights when they are available.
- Added optional iterative regularization hooks to both solvers:
  - `prior_strength`
  - `smoothing_strength`
- The regularization path works by:
  - blending each updated iterate geometrically toward the initial/reference
    spectrum, and
  - applying a light nearest-neighbour smoothing step in log-space while
    preserving the total flux scale of the iterate.
- Exposed those options through `SpectrumUnfolder.unfold(...)`.
- Exposed the same options through the registry-backed `GravelUnfolder`.
- Added an optional `basis_edges` solve mode to `SpectrumUnfolder.unfold(...)`.
  This performs the inversion on a coarse prior-shaped basis instead of directly
  on every native energy group.
- The basis coefficients are dimensionless scale factors on contiguous prior
  intervals, so the initial coefficient vector is exactly unity and the final
  solution expands back to the native plotting grid.
- Added an optional `aggregate_duplicate_reactions` mode to
  `SpectrumUnfolder.unfold(...)`.
  This inverse-variance aggregates repeated measurements that map to the same
  reaction response row before the inversion is solved.
- The duplicate-aggregation metadata is exposed on `UnfoldingResult.metadata`
  so notebooks can report how many rows were collapsed and which reaction
  families had replicate counts larger than one.
- Added regression tests for:
  - the GRAVEL uncertainty-weighting behavior on inconsistent duplicate rows,
  - the new iterative regularization behavior on sparse-response problems,
  - exact prior reconstruction and reduced-dimension solving through the
    prior-shaped basis path,
  - duplicate-reaction aggregation before unfolding.

## Additional Changes (April 2026)

Two follow-on changes were added to make RAFM troubleshooting more actionable.

1. Multi-photopeak isotope diagnostics
   - The isotope combiner now emits per-line diagnostics for isotopes with
     multiple photopeaks.
   - Outputs now include:
     - weighted all-lines activity,
     - mean/median/variance/std of line activities,
     - per-line single-vs-all relative deltas,
     - leave-one-out (LOO) consensus shifts,
     - robust modified-z outlier scores and flagged outlier energies.
   - RAFM line-consistency tables and reports now surface these metrics so it is
     easy to compare "one peak only" versus "all peaks combined" behavior.

2. Ti-48 and Cd model-mismatch handling
   - The flux-wire reaction builder now applies configurable model-form
     uncertainty inflation for:
     - `Ti-48(n,p)Sc-48` (known RAFM tension point), and
     - Cd-covered samples (known bare-vs-Cd mismatch source).
   - This preserves those reactions in the solve while preventing them from
     being over-weighted as if they were purely counting-statistics limited.
   - Cd-ratio tables now include review flags against configurable expected
     ranges so Cu/Sc Cd anomalies are visible in standard outputs.

## RAFM Validation Impact

On the live RAFM notebook problem:

- Fixing the GRAVEL weighting bug alone reduced the RAFM `GRAVEL` solution from
  the earlier `chi^2/dof ~ 137` range to about `71`.
- A second real problem was the problem formulation itself: the RAFM bare-wire
  set behaves like a few-channel inverse problem, not a fully independent
  native-grid solve.
- With the notebook switched to the new prior-shaped basis solve, the executed
  RAFM notebook now reports approximately:
  - `GRAVEL chi^2/dof = 2.724`
  - `MLEM chi^2/dof = 2.428`
  - `basis_groups = 14`
  - duplicate aggregation reduced the inversion rows from `9` to `5`

These changes materially improve solver stability and remove the worst
native-grid over-parameterization, but they do not by themselves fix the
underlying monitor inconsistency in the RAFM dataset.

## Remaining Limitation

The RAFM workflow still contains a real model/data mismatch independent of the
iterative update bug:

- Refolding the MCNP a priori spectrum through the IRDFF operator still
  overpredicts most bare-wire rates before unfolding, typically by about
  `15x` to `55x`, with `Ti-48(n,p)Sc-48` as the main smaller-mismatch
  exception.
- The RAFM notebook contains three Ti replicate samples per threshold reaction.
  Treating those replicate rows as fully independent measurements
  over-weights one response family and made the solver diagnostics look worse
  than the true reaction-family inconsistency warranted.
- Once the duplicated Ti rows are aggregated, the remaining dominant mismatch
  is a single reaction family:
  `Ti-48(n,p)Sc-48` still lands at about `-16σ` in a weighted prior-shaped NNLS
  prefit while the rest of the replicate-aggregated bare-wire set becomes
  broadly self-consistent.
- The Cd-covered Cu/Sc wires still cannot both be reconciled with one simple Cd
  transmission model; varying the assumed Cd thickness changes which one fits
  better but does not resolve both simultaneously.

So the FluxForge solver path is now behaving materially better and more
defensibly. The remaining RAFM problem is no longer a generic GRAVEL/MLEM
implementation failure; it is a dataset/model-consistency issue centered on the
`Ti-48(n,p)Sc-48` monitor family plus the unresolved Cd-covered diagnostics.
