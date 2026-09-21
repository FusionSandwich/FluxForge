# Measured-background covariance contract

18 September 2026. Implementation contract; experimental acceptance remains open.

## Grid and coverage

Recorded energies are channel centers. For the conservative transform, interior
edges are arithmetic midpoints of adjacent recorded energies; exterior edges
extend half the first/last center spacing. This explicitly adopts a piecewise
linear mapping between energy centers, including for polynomial header data.
It does not replace stored centers or calibration coefficients, or clamp negative
energy edges to zero. At least two finite, strictly increasing centers are
required for rebinning. Counts are uniform within each source energy bin.

Measured subtraction requires strict coverage of every target bin. Cropped
source counts outside the sample interval are recorded. Partial coverage is
available only in the standalone histogram primitive; it is not an accepted
background-subtraction policy. Missing calibration on just one operand,
nonmonotone axes and uncovered target bins raise before output is written.
Identical valid grids retain channelwise subtraction.

## Counting covariance and persistence

`counts_covariance` is optional, full covariance in counts squared, represented
in memory by a SciPy CSR matrix. Absence means diagonal variance given by
`counts_uncertainty ** 2`. Persisted covariance uses a versioned CSR object:
`format: csr`, `shape`, `data`, `indices`, `indptr`. Both triangles are stored.
Validate shape, finite values, canonical indices, symmetry and positive
semidefiniteness; zero/singular covariance is valid storage. When uncertainty is
also supplied, its square must match the diagonal. Uncertainty is derived from
the diagonal when only covariance is supplied. Signed counts require explicit
uncertainty or covariance. Legacy diagonal payloads remain supported.

For independent sample/background measurements and fixed normalization `a`,
`net = sample - a W background` and `Cnet = Cs + a² W Cb Wᵀ`. Scaling-time,
calibration and overlap-weight uncertainty are not included. Originals remain
unchanged. Signed estimates are retained; clipping correlated estimates is
unsupported. JSON spectrum artifacts and FFS sessions retain the full matrix.
Formats without a covariance representation must reject correlated spectra or
carry an explicit, restorable covariance companion; marginal errors alone are
not a lossless export.

## Consumers and acceptance

Linear estimates use `wᵀ C w`, including signed ROI/sideband weights. Gaussian
fits use the selected observation covariance with absolute weighting and retain
parameter covariance in area uncertainty. Singular fit covariance must be
rejected explicitly unless exact-constraint fitting is implemented; adding
jitter or silently using diagonal errors is not supported. Continuum methods
without estimator covariance must reject correlated quantitative inputs.
Peak search and display may use counts alone but cannot claim uncertainty
qualification. Activity and rate scaling must use propagated area uncertainty.

Reusing one measured background induces cross-spectrum covariance
`Cov(net_i, net_j) = a_i a_j W_i Cb W_jᵀ`. Overlapping peak estimates share
`w_iᵀ Cnet w_j`. Per-spectrum storage alone does not supply either cross-result
matrix to a rate/unfolding solver. These remain explicit blockers for accepted
combined INL reductions unless implemented end to end. Background detector,
geometry, acquisition epoch and physical representativeness are unqualified;
header calibration and count-time ratios are software inputs, not physical
certification. No all-method INL or handoff acceptance follows from this fix.
