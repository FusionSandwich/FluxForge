Absolutely — here is a **merged, publication-ready methods section** that keeps the strongest parts of your draft, folds in the useful additions from my earlier framework, and tightens a few places where the mathematics or positioning needed refinement. It preserves your core structure of transparent baselines first, then adaptive / publishable methods, and it keeps the workflow applicable to both first and second irradiations. The merged structure below is based directly on the method families you laid out in your draft, including DI-FOM, FIM, MWDCS, BASS, and STBD-MR. 

## Optimization Methodology for Post-Irradiation Gamma Spectrometry

### RAFM Steel Activation Analysis for INL LDRD and Related UWNR Assay Workflows

### 1. Scope and design philosophy

The purpose of this framework is to optimize neutron irradiation, cooldown, and HPGe counting schedules for post-irradiation gamma spectrometry in activation- and NAA-style workflows focused on RAFM steels and related reactor or fusion materials. The primary design variables are irradiation time, cooldown time, counting time, and, for multi-window protocols, the placement of repeated counting windows. The primary inference targets are isotope activity at end of irradiation, activity at arbitrary later times, daughter buildup, elemental or impurity mass where identifiable, and time-dependent dose-relevant inventory. The primary experimental goals are to identify which activation products matter most, determine which isotopes mask them, and choose schedules that maximize inferential value rather than raw count rate alone. This merged framework retains the transparent baseline structure of the draft methodology while strengthening the links between spectral identifiability, dose relevance, masking, and second-irradiation redesign.

A key principle is that the framework must degrade gracefully on a first irradiation and improve quantitatively on a second irradiation. For a first irradiation, the optimization proceeds using nominal composition, impurity estimates, library reaction data, and uncertainty models. For a second irradiation, the posterior from the first campaign becomes the prior for redesign, so the same formal machinery applies but with tighter uncertainty bounds and more targeted goals. That continuity between first and second campaigns is one of the strongest features of the draft and is retained here.

---

## 2. Common physics scaffold

All methods share the same forward model. The draft wrote this as a peak count-rate expression; for implementation and inference it is cleaner to work directly with **expected total counts** in a count window rather than average count rate, which avoids unit ambiguity and makes the Poisson likelihood natural. The structure remains the same as in your draft: buildup during irradiation, decay during cooldown, decay during counting, detector efficiency, branching ratio, and explicit Bateman treatment for parent-daughter chains. 

For isotope (i), gamma line (p), and design
[
d = (t_{\mathrm{irr}}, t_{\mathrm{cool}}, t_{\mathrm{count}}, g),
]
define the expected full-energy peak counts as
[
S_{i,p}(d)
==========

A_{\mathrm{sat},i}
\left(1-e^{-\lambda_i t_{\mathrm{irr}}}\right)
e^{-\lambda_i t_{\mathrm{cool}}}
\frac{1-e^{-\lambda_i t_{\mathrm{count}}}}{\lambda_i}
,B_{\gamma,i,p},
\varepsilon_p(g),
C_{i,p}(g),
]
where (A_{\mathrm{sat},i}=N_i\sigma_i\bar\phi) is the saturation activity, (B_{\gamma,i,p}) is the branching ratio, (\varepsilon_p) is the full-energy efficiency at the line energy, and (C_{i,p}) collects correction factors such as self-attenuation, coincidence summing, dead-time effects, and geometry-dependent attenuation. For decay chains with non-negligible daughter buildup, the single-isotope term must be replaced by the appropriate Bateman solution, exactly as noted in the draft.

For full-spectrum methods, the expected counts in spectral bin (k) and count window (m) are written
[
\mu_{k}^{(m)}(d)
================

\sum_i a_i^{(m)}(d), r_{ik}
+
b_k^{(m)},
]
with (a_i^{(m)}) the activity or line-intensity scaling factor for isotope (i) in that window, (r_{ik}) the detector response basis for isotope (i) in bin (k), and (b_k^{(m)}) the continuum/background term. Measured counts are modeled as
[
y_k^{(m)} \sim \mathrm{Poisson}!\left(\mu_k^{(m)}\right),
]
or, if needed, as overdispersed Poisson counts after diagnostics. This is the same matrix-view of the spectrum used in the draft FIM and STBD-MR formulations and is the natural bridge between line-based and full-spectrum methods.

---

## 3. Isotope ranking and task weighting

The draft’s dose-importance ranking is the correct starting point and should remain the first quantity computed before any schedule optimization. For isotope (i) at post-shutdown time (\tau), define the dose-importance weight
[
w_i(\tau)
=========

\frac{\dot D_i(\tau)}
{\sum_j \dot D_j(\tau)},
\qquad
\dot D_i(\tau)
==============

A_i(\tau)\sum_p B_{\gamma,i,p},\mathcal F(E_{i,p}),
]
where (\mathcal F(E)) is the dose-conversion kernel appropriate to the geometry and use case. This should be evaluated at several reference times such as 1 h, 8 h, 24 h, 1 week, 1 month, and 1 year so that the ranking evolves with time rather than collapsing the problem to a single “important isotope” list. That multi-time ranking logic is one of the strongest parts of the draft and should be preserved. 

For a more general implementation, especially if activation significance and NAA-style quantification both matter, define a composite isotope priority
[
W_i
===

\alpha_D D_i
+
\alpha_A A_i^{\mathrm{sig}}
+
\alpha_Q Q_i
+
\alpha_N N_i
+
\alpha_F F_i,
]
where (D_i) is dose relevance over the shutdown interval of interest, (A_i^{\mathrm{sig}}) is activation significance, (Q_i) is assay value, (N_i) is NAA value, and (F_i) is feasibility after masking and background. In practice, the baseline implementation can set (\alpha_D) dominant and the other terms optional. This preserves the draft’s strong dose-centered emphasis while allowing the same architecture to support impurity assay and inventory validation.

---

## 4. Masking and interference metrics

The draft already includes two useful masking concepts: a peak-level interference ratio for DI-FOM and a formal identifiability view in the FIM. Those should both remain. The peak-level interference ratio is
[
\mathcal M_{i,p}
================

\frac{\sum_{j\neq i} I_{j\to p}}{S_{i,p}},
]
with peaks flagged as potentially masked when (\mathcal M_{i,p}) exceeds a threshold such as 0.1. This is operationally useful because it gives an immediate rule for line exclusion or for shifting cooldown time to let the masker decay. 

For more complete design and inference work, define a pairwise masking score
[
M_{ij}(d)
=========

\sum_{p\in L_i}\omega_{i,p}
\left[
\eta_1 O_{ij,p}
+
\eta_2 C_{ij,p}
+
\eta_3 B_{ij,p}
+
\eta_4 T_{ij,p}
\right],
]
where (O_{ij,p}) measures photopeak overlap, (C_{ij,p}) the Compton spill contribution, (B_{ij,p}) the branching-weighted count dominance of isotope (j) over line (p) of isotope (i), and (T_{ij,p}) the temporal similarity of the decay signatures across the scheduled count windows. This formulation is useful later for graph-based visualization of masking communities and for deciding whether a second irradiation should be designed to break specific degeneracies.

---

# Part I — State-of-the-art baseline methods

These are the methods that should be implemented first. They are transparent, physics-based, auditable, and immediately useful on a first irradiation.

## 5. Method 1: Dose-Importance Weighted Figure of Merit (DI-FOM)

DI-FOM should remain the first baseline because it is intuitive, fast to evaluate, and directly aligned with shutdown-dose and activation priorities. The draft’s formulation is good and should be retained, but positioned explicitly as a **screening objective**, not the final inferential criterion. Its role is to rank isotopes, identify promising count windows, and eliminate obviously poor timing regions before more expensive FIM or Bayesian optimization is run. 

For reference time (\tau_{\mathrm{ref}}), define
[
\mathrm{DI!-!FOM}(\tau_{\mathrm{ref}},d)
========================================

\sum_i w_i(\tau_{\mathrm{ref}})
\sum_{p\in L_i}
\frac{S_{i,p}(d)^2}
{S_{i,p}(d)+B_{i,p}(d)+\sum_{j\neq i}I_{j\to p}(d)}.
]

This objective rewards schedules that place counts into high-value isotopes and lines while penalizing strong background and interference. Because the sum is over all usable peaks of all isotopes, it already improves substantially on older one-line timing methods. The recommended outputs remain:

* dose-sorted isotope ranking at multiple shutdown times,
* predicted line-level signal, background, and masking,
* candidate optimal ((t_{\mathrm{irr}}, t_{\mathrm{cool}}, t_{\mathrm{count}})) regions,
* line inclusion/exclusion flags.

This method applies to both first and second irradiations. On the first irradiation, (N_i), (\sigma_i), and (\bar\phi) are uncertain inputs with broad priors or nominal values. On the second irradiation, posterior activities and impurity estimates replace those nominal inputs, which makes DI-FOM a much sharper screening tool.

## 6. Method 2: Fisher Information Matrix optimization over the full spectral vector

The FIM method should remain the main baseline for rigorous schedule optimization. It formalizes spectral masking as parameter correlation and quantifies the best-case precision achievable for the activity vector or any linear function of it, including dose-weighted sums. The draft’s matrix model, D-/A-/C-optimal criteria, and use of the response matrix (\mathbf R) are exactly the right structure and should be kept. 

For expected spectrum (\boldsymbol\mu(d)=\mathbf R,\mathbf a(d)+\mathbf b), the Poisson FIM with respect to the activity vector is
[
\mathbf F(d)
============

\mathbf R^\top
\mathrm{diag}!\left(\frac{1}{\mu_k(d)}\right)
\mathbf R.
]

The design criterion may then be chosen according to the task:
[
\max_d \log\det \mathbf F(d)
\quad\text{(D-optimality)},
]
[
\min_d \mathrm{tr}!\left[\mathbf F(d)^{-1}\right]
\quad\text{(A-optimality)},
]
or
[
\min_d \mathbf c^\top \mathbf F(d)^{-1}\mathbf c
\quad\text{(C-optimality)}.
]

For this application, the most relevant ( \mathbf c ) is often a dose-weight vector, so the objective minimizes uncertainty in a dose-relevant linear combination of isotope activities rather than treating all isotopes equally. This is one of the cleanest ways to connect the draft’s dose weighting with a formal identifiability criterion. 

The FIM should also be used to define isotope-level masking. A pair is spectrally problematic when the relevant submatrix becomes ill-conditioned or when posterior covariance is dominated by off-diagonal terms. In practice, the outputs should include:

* Cramér–Rao lower bounds by isotope,
* a spectral interference / covariance matrix,
* condition numbers for masked isotope pairs,
* Pareto surfaces over irradiation, cooldown, and count time.

The main caveat is that the FIM is optimistic if nuisance parameters or model mismatch are omitted. For that reason, the implementation should permit nuisance parameters for efficiency normalization, energy calibration drift, peak width, background scaling, and dead-time correction so that the FIM remains a realistic design surrogate rather than an idealized bound.

## 7. Method 3: Multi-Window Decay Curve Scheduling (MWDCS)

MWDCS should remain the third baseline and is essential for RAFM materials because the relevant isotope set spans minutes to years. A single-window design is fundamentally incapable of serving all objectives well. The draft’s additive-FIM formulation is correct and should be preserved. 

For (M) counting windows, the augmented FIM is
[
\tilde{\mathbf F}
=================

\sum_{m=1}^M \mathbf F^{(m)}!\left(t_{\mathrm{cool},m},t_{\mathrm{count},m}\right).
]

The schedule is chosen to maximize a design objective such as (\log\det \tilde{\mathbf F}) or a dose-weighted C-optimal criterion under beam-time, handling, and detector-availability constraints. This framework naturally reveals that different windows contribute different kinds of information: early windows carry short-lived dose information, intermediate windows often resolve activation products of practical assay value, and late windows are where long-lived impurities and regulatory isotopes become measurable.

The draft’s half-life-group heuristic should remain as the first implementation:

* immediate / very early windows for minute-scale isotopes,
* hour-scale windows for short shutdown isotopes,
* day-scale windows for medium-lived activation products,
* month-scale windows for long-lived dose-relevant species,
* very late windows for persistent impurities or regulatory contributors. 

The key outputs are:

* recommended multi-window schedules,
* marginal information gain by window,
* isotope classes requiring additional windows,
* a comparison between single-window and multi-window designs.

---

# Part II — Physics-rich inference layer

This layer is the bridge between baseline schedule optimization and the advanced adaptive methods. It should be implemented after the three baselines are functioning.

## 8. Method 4: Joint multi-window spectral inference

Before introducing full adaptive scheduling, there should be a stable joint inference engine that fits all relevant windows and all relevant isotopes simultaneously. This can begin as a penalized maximum-likelihood or MAP fit and later become fully Bayesian.

The objective is
[
\mathcal L(\Theta)
==================

\sum_{m,k}
\left[
\mu_k^{(m)}(\Theta)-y_k^{(m)}\log \mu_k^{(m)}(\Theta)
\right]
+
\mathcal P(\Theta),
]
where (\Theta) contains end-of-irradiation activities, optional elemental masses or impurity abundances, and nuisance parameters such as efficiency normalization, background coefficients, and energy calibration shifts. The penalty or prior term (\mathcal P(\Theta)) can be quadratic for a MAP implementation or fully probabilistic in a later Bayesian implementation.

This layer is where the framework first produces:

* posterior or covariance estimates for end-of-irradiation activity,
* activity at arbitrary future times,
* daughter buildup trajectories,
* line-level residuals,
* isotope identifiability diagnostics.

It is also the engine that provides the posterior required by second-irradiation redesign.

---

# Part III — Novel, publishable methods

These are the methods that should come after the baseline stack is validated.

## 9. Novel Method 1: Dose-weighted Bayesian Adaptive Sequential Scheduling (BASS-D)

The draft’s BASS is already strong. The main improvement is to make the utility explicitly **dose-weighted and decision-aware**, rather than purely activity-vector information gain. That gives a cleaner novelty story and aligns the adaptive design directly with shutdown-dose and assay objectives. The base BASS structure from the draft remains intact: after each count window, update the posterior and choose the next window by maximizing expected information gain. 

At stage (m), with current posterior (p(\mathbf a\mid \mathcal D_{1:m-1})), define
[
\mathrm{EIG}(t)
===============

\mathbb E_{\mathcal D_m}
\left[
D_{\mathrm{KL}}
!\left(
p(\mathbf a\mid \mathcal D_{1:m})
;|;
p(\mathbf a\mid \mathcal D_{1:m-1})
\right)
\right].
]

For tractability, the draft uses the Gaussian-prior / FIM approximation
[
\mathrm{EIG}(t)
\approx
\frac{1}{2}
\log\det
\left(
\mathbf I+\boldsymbol\Sigma_{m-1}\mathbf F^{(m)}(t)
\right),
]
which should remain as the practical implementation. 

The merged framework modifies the utility so that the scheduler is rewarded not only for reducing activity uncertainty, but for reducing uncertainty in decision-relevant dose quantities over a set of shutdown times:
[
U(t)
====

\sum_{\tau\in\mathcal T}
\psi(\tau),
\mathbb E_{\mathcal D_m}
\left[
D_{\mathrm{KL}}
!\left(
p(\mathbf z_\tau\mid \mathcal D_{1:m})
;|;
p(\mathbf z_\tau\mid \mathcal D_{1:m-1})
\right)
\right],
]
where (\mathbf z_\tau) may be the vector of dose-relevant isotope contributions or the total shutdown-dose observable at time (\tau). In practice, this can be approximated by replacing the vanilla EIG objective with a weighted version that emphasizes uncertainty reduction in dose-dominant or assay-dominant subspaces.

This method works for both first and second irradiations, but it is especially valuable for the second irradiation. After a first campaign, the posterior reveals which isotopes are already well constrained, which impurities are unexpectedly active, and which masking relationships dominate. The second irradiation can then be redesigned around those uncertainties rather than around nominal expectations. The most important outputs are:

* adaptive count schedule,
* posterior evolution by window,
* expected value of additional counting,
* explicit quantification of how much more was learned on the second campaign than would have been learned under a static schedule.

## 10. Novel Method 2: Spectro-Temporal Basis Decomposition with Mask-Aware Regularization (STBD-MR)

The core idea of STBD-MR should remain: treat the time series of spectra as a single joint inference problem with physics-defined isotope basis functions rather than learned latent bases. That is a strong and publishable idea. The one place that should be revised is the regularizer. In the draft it is asymmetric in the isotope pair, which is hard to justify mathematically. The merged version keeps the spectral-coherence and temporal-separability concepts but makes the regularization symmetric and better aligned with pairwise masking.

Define the time-series model
[
\boldsymbol\mu^{(m)}
====================

\sum_i a_i^{(0)},d_i^{(m)},\mathbf r_i
+
\mathbf b^{(m)},
]
with (d_i^{(m)}) the decay factor for isotope (i) in window (m), including Bateman terms when needed, and (\mathbf r_i) the spectral fingerprint. Define spectral coherence
[
\rho_{ij}
=========

\frac{\mathbf r_i^\top \mathbf r_j}
{|\mathbf r_i|,|\mathbf r_j|},
]
and temporal separability
[
\delta_{ij}^{(m,n)}
===================

\left|
\frac{d_i^{(m)}}{d_i^{(n)}}-\frac{d_j^{(m)}}{d_j^{(n)}}
\right|.
]

Then build a symmetric masking weight
[
\omega_{ij}
===========

\rho_{ij}
\left(
\max_{m,n}\delta_{ij}^{(m,n)}+\epsilon
\right)^{-1},
]
and use it in a graph-Laplacian-style penalty
[
\mathcal R_{\mathrm{mask}}(\mathbf a)
=====================================

\mu_{\mathrm{mask}}
\sum_{i<j}
\omega_{ij}
\left[
\left(a_i-a_i^{\mathrm{prior}}\right)^2
+
\left(a_j-a_j^{\mathrm{prior}}\right)^2
\right].
]

The full objective becomes
[
\mathcal L(\mathbf a)
=====================

\sum_{m,k}
\left[
\mu_k^{(m)}-y_k^{(m)}\log\mu_k^{(m)}
\right]
+
\mathcal R_{\mathrm{mask}}(\mathbf a)
+
\frac12
(\mathbf a-\mathbf a^{\mathrm{prior}})^\top
\boldsymbol\Sigma_{\mathrm{prior}}^{-1}
(\mathbf a-\mathbf a^{\mathrm{prior}}).
]

This preserves the draft’s key insight: when two isotopes are spectrally similar and only weakly separated in time by the chosen schedule, the inference should be stabilized by prior knowledge rather than by allowing unstable, anti-correlated estimates to inflate arbitrarily. It also turns masking into a clearly defined mathematical object that can be visualized as an interference graph and used directly in schedule design.

For experimental design, STBD-MR can be used prospectively: before any measurement, evaluate (\omega_{ij}) across candidate count-window placements and choose schedules that reduce the worst masking weights for the most important isotope pairs. This is one of the cleanest ways to connect spectral structure, temporal design, and second-irradiation complementarity.

The recommended outputs are:

* posterior end-of-irradiation activity vector with covariance,
* spectral coherence matrix,
* temporal separability matrix,
* masking graph and masking communities,
* residuals by energy and by time window,
* flagged isotope pairs that require extra windows or a redesigned second irradiation.

---

# Part IV — First versus second irradiation logic

The merged workflow is explicitly designed for either a first irradiation alone or a first-plus-second irradiation sequence.

For a **first irradiation**, the process is:

1. compute dose-weighted isotope ranking,
2. run DI-FOM screening over candidate timing regions,
3. design a multi-window baseline schedule using MWDCS,
4. validate and refine the schedule with FIM,
5. perform joint multi-window inference after data are acquired.

For a **second irradiation**, the process is:

1. use the first irradiation posterior as the prior,
2. identify under-constrained isotopes and dominant maskers,
3. redesign irradiation, cooldown, and count timing using FIM and BASS-D,
4. optionally select a complementary irradiation length or count-window sequence specifically to break masking communities identified by STBD-MR.

This logic is exactly in line with the draft’s design principle that the same framework should function on the first campaign and then improve on the second, with the value of sequential design quantified explicitly.

---

# Part V — Recommended implementation order

The implementation order should be:

### Phase 1: transparent baseline design

1. DI-FOM isotope ranking and schedule screening
2. MWDCS heuristic multi-window scheduling
3. FIM evaluation and optimization of candidate schedules

### Phase 2: inference backbone

4. Joint multi-window MAP / penalized-likelihood spectral inference
5. Bateman chain support and nuisance-parameter propagation

### Phase 3: publishable adaptive methods

6. BASS-D for adaptive scheduling and second-irradiation redesign
7. STBD-MR for mask-aware spectro-temporal joint inference and schedule redesign

That ordering is very close to the roadmap in your draft, but it inserts the simpler joint inference layer before the full novel methods, which will make both software development and paper validation much cleaner. 

---

# Part VI — Standard outputs across all methods

Every method should emit a consistent family of outputs so the workflow is coherent rather than method-specific:

* isotope ranking tables at multiple shutdown times,
* recommended irradiation, cooldown, and count schedules,
* multi-window schedules and their marginal information contributions,
* peak-level masking flags and pairwise masking matrices,
* predicted or posterior uncertainty on end-of-irradiation activity,
* activity at arbitrary future times with uncertainty,
* daughter buildup trajectories,
* elemental or impurity mass estimates where identifiable,
* Cramér–Rao bounds or posterior covariance summaries,
* comparison of measured inference results against transport / activation predictions.

That output philosophy is already present in your draft and should be kept exactly because it makes the whole framework software-friendly and publication-friendly at the same time. 

---

## Condensed scientific positioning

In one sentence, the merged framework is:

**a dose-aware, multi-peak, multi-window, uncertainty-propagating experimental design and inference framework for post-irradiation gamma spectrometry that begins with transparent physics-based baselines and culminates in adaptive Bayesian scheduling and mask-aware spectro-temporal decomposition.**

That is a stronger positioning statement than either version alone.


