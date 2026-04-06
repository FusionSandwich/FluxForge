# Multi-Line, Dose-Aware Optimization of Post-Irradiation HPGe Gamma Spectrometry Workflows for Activated RAFM Steels

## Scope and modeling foundation

The motivating setting—activation behavior characterization in irradiated reduced-activation ferritic/martensitic (RAFM) steels and related reactor/fusion materials—naturally couples three pieces of physics into one experiment-planning problem: (i) activation/decay inventory evolution during irradiation and cooldown, (ii) gamma emission and detector response during counting, and (iii) a decision layer that prioritizes radionuclides by “value” (dose relevance, activation significance, assay/NAA value) while explicitly penalizing masking/interference and operational constraints (dead time, pile-up, radiological handling limits). This coupling is exactly why one-line-at-a-time timing heuristics tend to underperform when spectra contain many activation products and many overlapping emissions. citeturn29view3turn14view0turn6view4

A practical, implementation-agnostic forward model can be written in two nested blocks:

**Inventory block (irradiation → end-of-irradiation → cooldown).** Modern activation/inventory solvers (e.g., ORIGEN and FISPACT-style solvers) propagate coupled production/depletion/decay for many nuclides simultaneously and can also produce gamma emission spectra and radiological response functions. ORIGEN is explicitly described as solving time-dependent concentrations/activities and generating alpha/beta/neutron/gamma emission spectra. citeturn4view1turn3search4 FISPACT-II is described as an inventory code with activation/transmutation/depletion capability and includes uncertainty/sensitivity methods and pathway analysis. citeturn4view0turn8view3turn8view4

**Measurement block (cooldown → counting interval → spectrum).** Gamma spectrometry practice emphasizes that complex spectra (common in NAA/activation) contain overlapping peaks, Compton continua, escape peaks, and sum peaks, and that simplistic peak integration can be inaccurate in crowded regions. citeturn7view3turn6view4 Corrections for decay during counting and for high-count-rate effects (dead time, pile-up) must be treated as part of the measurement model rather than afterthoughts; guidance documents and laboratory procedures explicitly discuss these effects and how to characterize pile-up loss as a function of count rate and energy. citeturn4view3turn6view4turn3search10

Two “activation-analysis-specific” complications matter for experiment design:

First, multiple measurement modes (short/medium/long cooldown counts) of the same sample generate **correlated** inferences across gamma lines and across measurement times; uncertainty propagation can be challenging because correlations exist not only in nuclear data but also between results derived from different gamma energies and different measurement times. citeturn29view3 Second, coincidence/summing effects can create both “summing in” peaks and “summing out” losses, and for extended sources the correction requires advanced approaches (often Monte Carlo-based), particularly when geometry is close and efficiency is high. citeturn9view1turn28view0turn6view0

This motivates a workflow architecture with two explicit outputs: (a) **a schedule** (irradiation time, cooldown time(s), counting time(s)) optimized under multi-line inference objectives, and (b) **a ranking + masking map** that explains *why* particular nuclides are prioritized and *which* nuclides/lines are expected to interfere, including time-dependence of those interferences.

Entities and context anchors: the application is motivated by work at entity["organization","Idaho National Laboratory","us natl lab idaho"] and experiments in the irradiation/counting ecosystem associated with entity["point_of_interest","University of Wisconsin Nuclear Reactor","madison, wi, us"]; standards and data practices relevant to NAA and gamma spectrometry are centrally documented by the entity["organization","International Atomic Energy Agency","united nations agency"] and the entity["organization","International Organization for Standardization","standards body"]. citeturn4view2turn5view4

## Physics-first baselines to implement early

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["HPGe gamma spectrum activated steel overlapping peaks","high purity germanium detector gamma spectrometry setup"] ,"num_per_query":1}

### Baseline method A: Poisson multi-line forward model with D-optimal (information-based) schedule optimization

This baseline is “state-of-the-art in spirit” because it treats the entire workflow as a **Poisson counting experiment** with a correct likelihood, then optimizes irradiation/cooldown/count variables to maximize expected inferential performance, instead of optimizing each peak or each isotope separately.

**Core inference model (multi-line, multi-time).** Let the unknown parameter vector be  
\[
\theta=\{\text{EOI activities }A_i^{\text{EOI}}\}_{i\in \mathcal{I}}\ \cup\ \{\text{(optional) elemental masses }m_e\}\ \cup\ \{\text{nuisance calibration parameters}\}.
\]
For a measurement \(k\) starting at cooldown \(t_k\) and lasting live time \(T_k\), the expected net peak counts for isotope \(i\) and gamma line \(\ell\) can be modeled as
\[
\mu_{i\ell k}(\theta, d)\;=\;A_i(t_k;\theta,d)\, I_{i\ell}\,\varepsilon(E_\ell)\,\int_{0}^{T_k}\exp(-\lambda_i \tau)\,d\tau \times C^{\text{corr}}_{i\ell k},
\]
where \(I_{i\ell}\) is emission probability, \(\varepsilon(E_\ell)\) is full-energy peak efficiency, and \(C^{\text{corr}}\) is a multiplicative correction bundle (dead time / pile-up, coincidence summing, self-attenuation, geometry transfer) treated as uncertain but parameterized. The importance of explicitly treating dead time/pile-up and coincidence/summing in high-efficiency geometries is emphasized in procedural and reference materials. citeturn4view3turn9view1turn6view4

Observed counts (peak areas or binned channels) are then modeled as Poisson with mean equal to summed contributions of all candidate nuclides plus background. Gamma-spectrum software literature explicitly frames multi-component spectral models with Poisson counting noise and notes that maximum-likelihood methods are optimal under Poisson statistics, while least-squares is most appropriate under Gaussian approximations. citeturn14view0turn17view1

**Design variables.**  
\(d\) includes irradiation time \(t_{\text{irr}}\), one or more cooldown choices \(\{t_k\}\), counting times \(\{T_k\}\), and (optionally) detector geometry choices (distance/shielding/collimation) if available.

**Objective function (transparent and computable).** Use a Poisson Fisher-information-style criterion as an approximation to “expected posterior tightness.” Under Poisson regression design theory, D-optimal design maximizes \(\det(M(d))\) where \(M\) is an information matrix built from the forward sensitivities. citeturn19view0 A directly usable objective is a weighted D-opt criterion on the subset of parameters that matter most:
\[
J_{\text{D}}(d)=\log\det\left( W^{1/2}\,M_{\theta\theta}(d)\,W^{1/2} + \eta I\right),
\]
where \(W\) encodes isotope-of-interest and dose-relevance weights (defined explicitly in the ranking section below), and \(\eta\) is a numerical stabilizer. The design can be made robust (“parameter-robust”) by evaluating \(J_{\text{D}}\) over a distribution of plausible \(\theta\) values rather than a single nominal prediction, which aligns with how Poisson design literature addresses parameter dependence. citeturn19view0

**Uncertainty treatment.**  
This method is strongest when uncertainties are carried explicitly as:  
- Poisson counting uncertainty in peak areas / binned channels;  
- correlated calibration/efficiency uncertainties (energy calibration, FWHM model, efficiency curve coefficients);  
- irradiation/cooldown timing uncertainties (usually small relative to long counts, but still a model term); citeturn4view3  
- nuclear-data and flux uncertainties, where correlations may be significant; the k0-NAA intercomparison report highlights that correlations exist across nuclear data and across results from different gamma energies and measurement modes, and that peak-area counting statistics is often the main *independent* contributor for individual mass fractions. citeturn29view3

A practical baseline implementation can treat calibration/efficiency parameters as Gaussian nuisance parameters with covariance \( \Sigma_{\text{cal}} \), then use a “profile information” or Bayesian Laplace approximation to propagate them into \(M_{\theta\theta}\).

**Masking/interference integration.**  
Instead of a separate “interference check,” incorporate interference into \(M(d)\) through the use of (i) wide ROI/channel likelihoods (so the model must explain overlaps), and/or (ii) explicit coupling of nearby lines. Overlapping peaks degrade conditioning and inflate decision thresholds; this is explicitly discussed in decision-threshold work for gamma spectrometry that accounts for overlapping peaks and conditioning. citeturn7view1

**Recommended outputs.**  
- Optimal schedule \(d^*\) and a Pareto set of near-optimal schedules (e.g., shorter vs more precise).  
- Predicted uncertainties (posterior covariance or Cramér–Rao-style lower bounds) for \(A_i^{\text{EOI}}\), \(A_i(t)\) at user-selected times, and (if modeled) elemental masses.  
- A “what drives the design” explanation: sensitivity heatmaps of peaks/timepoints to each isotope.  
- A dead-time / pile-up risk forecast per schedule, using rate-dependent correction characterization guidance. citeturn4view3turn3search21

**Why it works for first and second irradiations.**  
Second irradiation is handled by including a nonzero initial inventory at the start of the irradiation step (i.e., \(N_i(t=0)\neq 0\)) and letting the same forward model propagate through the second pulse; inventory codes explicitly support multi-step irradiation/decay sequencing. citeturn8view3turn4view1

### Baseline method B: Dose-weighted multi-objective optimization with ISO-style detectability and multi-line decision thresholds

This baseline is more “operations-facing”: it treats schedule choice as a multi-objective decision problem that must satisfy detectability and ALARA-like constraints while prioritizing dose-relevant nuclides.

**Dose-relevance modeling.**  
Inventory/activation tooling supports dose-relevant response functions. FISPACT-II, for example, documents gamma dose rate calculations (including contact dose from a slab and dose at distance from a point source in air) and prints “dominant nuclides” tables sorted by dose rate, activity, heat, etc. citeturn8view1turn8view0 This enables a directly reportable ranking metric such as:
\[
\text{DoseScore}_i = \int_{t\in \mathcal{T}} w(t)\, \frac{D_i(t)}{\sum_j D_j(t)}\,dt,
\]
where \(D_i(t)\) is predicted gamma dose-rate contribution of nuclide \(i\) under a chosen geometric model, and \(w(t)\) emphasizes times of operational interest (e.g., “hours to days” vs “months to years”).

Empirically, time-dependent dominance of specific activation products is strongly material- and timescale-dependent; for iron-based materials, one example study identifies ^56Mn as dominating dose rate immediately after irradiation, while ^54Mn and ^60Co dominate intermediate and long cooldown periods. citeturn4view4 Fusion-material benchmark work similarly notes time windows where particular nuclides dominate decay heat (and, by extension, radiological significance), such as ^58Co dominating between about a week and a few hundred days in one benchmark set. citeturn6view3

**Detectability constraints using characteristic limits.**  
Rather than ad-hoc SNR thresholds, use characteristic limits (decision threshold, detection limit) in the sense of ISO 11929 guidance. IAEA guidance explicitly frames ISO 11929 as the international standard for characteristic limits and discusses correct application and traceability expectations. citeturn5view4

For multi-gamma emitters, decision-threshold methodology can explicitly combine information across multiple gamma rays; a decision-threshold approach for gamma spectrometry notes that, for multi-gamma emitters, a common decision threshold (using multiple peaks) can be smaller than individual thresholds, while overlapping peaks and peaked backgrounds can increase thresholds considerably. citeturn7view1 This directly supports multi-line schedule optimization: you may accept a schedule where no single line is “excellent” if the joint multi-line detectability is strong.

**Objective function family.**  
Define a vector of objectives and solve for Pareto-optimal schedules:
- **Inference objective:** minimize expected variance of \(A_i^{\text{EOI}}\) and/or elemental masses for high-priority nuclides.  
- **Dose objective:** maximize fraction of dose-relevant nuclides measured above detectability within the time window(s) of interest.  
- **Interference objective:** minimize predicted masking metrics (defined explicitly below).  
- **Operational objective:** minimize total reactor + handling + counting time, and enforce dead-time/pile-up limits.

A concrete scalarization (easy to implement first) is:
\[
J(d)=\sum_{i\in \mathcal{I}} \alpha_i \, \mathrm{Var}_d(\hat A_i^{\text{EOI}}) \;+\;\beta \sum_{t\in \mathcal{T}} w(t)\,[D_{\text{residual}}(t;d)]\;+\;\gamma\,\mathrm{MaskPenalty}(d)\;+\;\delta\,\mathrm{TimeCost}(d),
\]
subject to:
- \( \text{DT}(d)\le \text{DT}_{\max}\) and pile-up correction validity (rate regime within characterized domain); citeturn4view3turn3search21  
- detectability constraints \(L_{d,i}\le A_i(t_k)\) where \(L_{d,i}\) is an ISO-11929-style detection limit translated to activity for isotope \(i\) at measurement \(k\). citeturn5view4turn7view1

**Uncertainty treatment and calibration dependence.**  
This method should explicitly include efficiency/geometry transfer uncertainty. The k0-NAA intercomparison emphasizes that detector characterization (including the radionuclides and peak energies used and how the efficiency curve is modeled) can drive dispersion in final results, which argues for designing schedules that avoid relying on a small number of lines or a narrow energy range. citeturn29view1turn4view2

**Recommended outputs.**  
- A ranked isotope list by DoseScore and by “assay leverage” (variance reduction per unit counting time).  
- A schedule Pareto front: e.g., (total count time) vs (expected uncertainty of key nuclides) vs (masking penalty).  
- ISO-aligned reporting: decision thresholds / detection limits reported per nuclide and per measurement configuration in a way consistent with laboratory QA expectations. citeturn5view4turn7view1

**Why it works for first and second irradiations.**  
Dose and detectability constraints automatically adapt if residual inventory exists before the second irradiation (e.g., persistent ^60Co): the predicted dose and predicted line intensities shift, forcing the optimizer to re-balance cooldown and counting choices accordingly. The method is agnostic to whether the inventory arises from first or second irradiation, as long as the inventory model is initialized correctly. citeturn4view1turn8view3

### Baseline method C: Full-spectrum (multi-channel) unmixing across multiple counts with regularization and time-structure

This baseline moves beyond explicit per-peak extraction and uses *all* channels (or wide spectral regions), which is often advantageous for heavily activated materials where Compton continua and many overlapping lines make “clean ROI” assumptions fragile.

**Core idea.** Build a spectral library (or forward response basis) for candidate nuclides (including detector response, scattering continua, escape peaks, and sum-peak structure if modeled). Then infer nuclide activities by fitting the observed spectrum with a weighted/Poisson-consistent objective.

Library least squares (LLS) variants are well established in gamma spectrum analysis; a modern example proposes a weighted library least squares (WLLS) approach that weights by \(\sqrt{\text{counts}}\) to stabilize statistical fluctuations and reports reduced variability relative to unweighted LLS. citeturn5view2turn3search3 IAEA software guidance also describes maximum-likelihood algorithms that treat both full-energy peaks and Compton parts and aim to reduce bias factors; importantly, it explicitly discusses minimizing activity-estimation errors by choosing an optimal set of initial spectra in a multi-component model with Gaussian or Poisson statistics. citeturn14view0

**Time-structured extension (key for experiment planning).** When you have multiple spectra at different cooldown times \(t_k\), couple them through a shared EOI activity vector \(A^{\text{EOI}}\) and known decay/ingrowth physics:
\[
\text{Spectrum}_k \approx \sum_i a_{ik}(t_k,T_k)\,\text{Response}_i + \text{background}_k,
\]
where \(a_{ik}\) is constrained to be consistent with \(A_i^{\text{EOI}}\) (and Bateman chains). This creates a **single inverse problem** over all timepoints, which tends to reduce false positives/negatives because many nuclides exhibit distinct time signatures even if they overlap in energy.

**Objective functions.** Two robust options:
- **Poisson negative log-likelihood (preferred when modeling channels):**  
  \(\min \sum_{k,c}\left[\lambda_{kc} - y_{kc}\log \lambda_{kc}\right] + \rho\,\Omega(A^{\text{EOI}})\), where \(\Omega\) imposes nonnegativity and smoothness/priors.  
- **Weighted least squares (fast baseline):** WLLS-like \(\min \sum_{k,c} w_{kc}(y_{kc}-\lambda_{kc})^2\) with \(w_{kc}\propto 1/\max(y_{kc},1)\) as a Poisson proxy. citeturn5view2turn14view0

**Regularization and calibrated uncertainties.** A 2025 arXiv study on unfolding gamma spectra motivates regularized maximum-likelihood estimation (RMLE) for ill-posed spectral inversion, emphasizing explicit background/contaminant modeling and calibrated confidence intervals. citeturn25view0 While that work targets detector unfolding, the methodological point—regularized ML with uncertainty calibration for an ill-posed inverse problem—maps directly onto full-spectrum nuclide unmixing in activated materials.

**How this becomes a schedule optimizer.** Use a *design-quality metric* derived from the fit geometry:
- minimize the condition number of the joint design matrix across timepoints,
- minimize mutual coherence between nuclide basis spectra,
- maximize expected separation of time signatures for key nuclides.

This is directly aligned with the observation that conditioning worsens for overlapping peaks and that it impacts uncertainty. citeturn7view1turn7view3

**Recommended outputs.**  
- A ranked list of “explainable basis components” with uncertainty intervals.  
- A time-sequenced residual map: which energies cannot be explained under the current nuclide set (useful for discovering missing impurities/activation routes).  
- A schedule recommendation that maximizes identifiability rather than just “signal strength.”

**Why it works for first and second irradiations.**  
Because the inference is parameterized in terms of EOI activities and then mapped to each subsequent spectrum, residual activity from previous irradiations becomes a prior or initial condition rather than a special case; the same coupled time-structured inversion applies. citeturn4view1turn8view3

## Novel, publishable method concepts to implement later

The methods below are intended to be publishable contributions precisely because they unify: (i) multi-line/multi-spectrum inference, (ii) dose-aware nuclide ranking, (iii) explicit masking metrics, and (iv) adaptive or differentiable decision-making—capabilities not typically combined into a single end-to-end planning framework in activation/PIGS/NAA practice.

### Novel method A: Bayesian adaptive sequencing with mutual-information utility on multi-peak spectra

**Publishable claim.** Replace fixed “choose cooldown then count” planning with **sequential Bayesian experimental design** that updates beliefs after each spectrum and decides the next cooldown/count action to maximize expected utility for dose-relevant isotopes under masking/operational constraints.

**Why this is justified by current state-of-the-art components.**  
- Bayesian spectral deconvolution under Poisson noise has been proposed as a probabilistic measurement model that links measurement time directly to estimation limits and supports “virtual measurement” simulation under an explicit noise model. citeturn5view3  
- Mutual-information-based optimal experimental design (MI-OED) explicitly formulates “maximize mutual information between measurements and parameters” as an optimization criterion to reduce parameter uncertainty, and provides a practical blueprint for how to compute and optimize MI in physics-model-based inference. citeturn20view0  
- Poisson GLM design literature provides additional support for Poisson-informed optimal design concepts and local/Bayesian design handling of parameter uncertainty. citeturn19view0

**Model.**  
Let \(\theta\) include EOI activities of candidate activation products, selected elemental masses, and nuisance calibration/efficiency parameters. Let each planned action \(a_k\) be either:  
- “start counting at time \(t_k\) for live time \(T_k\),” or  
- “wait \(\Delta\) then count,” or  
- (if controllable) “extend irradiation by \(\Delta t_{\text{irr}}\)” *before* EOI to target short-lived products (more limited operationally, but included for completeness).

**Utility function.**  
A strong MI-OED utility is:
\[
U(a_k) = I\big(\theta_{\text{ROI}};\, y_k \mid a_k, \mathcal{D}_{<k}\big) - \lambda_{\text{time}}\,\text{Cost}(a_k) - \lambda_{\text{dose}}\,\text{DoseRisk}(a_k),
\]
where \(\theta_{\text{ROI}}\) is the subset of parameters for isotopes that rank highly for dose/assay value (ranking defined below). MI-OED explicitly uses mutual information to quantify “information content” and sets up an optimization problem to maximize it. citeturn20view0

**Masking-aware modification (critical novelty for activation).**  
Include an explicit penalty term:
\[
\text{MaskPenalty}(a_k) = \sum_{i\in \mathcal{I}_{\text{interest}}} w_i \,\Pr\big(\text{interference}_i \text{ dominates in } y_k \mid a_k,\mathcal{D}_{<k}\big),
\]
where the probability is computed from posterior predictive simulations using the Poisson spectral model (a direct extension of “virtual measurement analytics”). citeturn5view3turn6view4

**Uncertainty handling.**  
This method naturally incorporates the correlation structure emphasized in k0-NAA practice (correlations across gamma energies and across short/medium/long counts) because Bayesian inference is performed jointly over all spectra and all lines. citeturn29view3

**Outputs (what a paper would show).**  
- Adaptive schedules that reduce posterior variance for dose-dominant nuclides faster than fixed schedules (time-to-precision curves).  
- Explicit “why did it choose to wait?” decisions tied to predicted masking from decaying high-activity interferers.  
- Demonstrated robustness to second irradiation by initializing the prior for the second experiment from the posterior of the first (or from measured pre-irradiation spectra).

### Novel method B: Differentiable dose-weighted schedule optimization with an interference graph and calibrated inverse-problem backend

**Publishable claim.** Create a **differentiable, end-to-end objective** that maps schedule variables \((t_{\text{irr}},\{t_k,T_k\})\) to a weighted, dose-aware inference score, enabling gradient-based optimization and robust design under uncertainties—while using an interference graph to guarantee interpretability.

**Key enabling facts from existing literature.**  
- FISPACT-style tools explicitly compute gamma dose rates and identify dominant nuclides by dose rate and other radiological quantities. citeturn8view1turn8view0  
- Full-spectrum inversion is an ill-posed problem where regularized maximum likelihood with uncertainty calibration is a contemporary approach; RMLE-style approaches emphasize explicit background/contaminant modeling and calibrated confidence intervals. citeturn25view0  
- Coincidence/summing and high-rate effects create structured “physics artifacts” (sum peaks, summing out) that must be modeled or penalized, and guidance documents note their importance, especially in complex spectra. citeturn9view1turn6view0turn6view4

**Interference graph construction (core novelty and explainability tool).**  
Build a bipartite (or projected) graph:
- nodes: nuclides \(i\) and gamma features \(g\) (peaks/energy windows);
- edges: expected contribution of nuclide \(i\) to feature \(g\) at time \(t\), scaled by overlap likelihood and detector resolution.

Use this to define a **masking centrality** metric:
\[
\text{MaskCentrality}_j(t) = \sum_{g} \sum_{i\in \mathcal{I}_{\text{interest}}} 
\underbrace{\text{Overlap}(E_{jg},E_{ig})}_{\text{resolution-weighted}}
\times
\underbrace{\frac{\mathbb{E}[C_{jg}(t)]}{\mathbb{E}[C_{ig}(t)]+\epsilon}}_{\text{interference-to-signal}}
\]
and then compress over time windows with weights.

This graph formalizes what gamma spectrometry guidance describes qualitatively: prominent peaks, sum peaks, and nearby interfering features can block identification; crowded regions and overlap drive uncertainty. citeturn6view4turn7view1turn7view3

**Differentiable objective.**  
Define:
\[
J(d)=
\underbrace{\sum_{i} w^{\text{dose}}_i\,\mathrm{Var}_d(\hat A_i^{\text{EOI}})}_{\text{dose-weighted inference quality}}
+\underbrace{\kappa \sum_{t\in\mathcal{T}} w(t)\,\text{MaskCentrality}_{\text{interest}}(t)}_{\text{masking penalty}}
+\underbrace{\lambda\,\text{OperationalPenalty}(d)}_{\text{dead time, pile-up, safety}},
\]
where \(\mathrm{Var}_d(\cdot)\) is produced by a calibrated inverse-problem backend (e.g., Laplace approximation on a Poisson full-spectrum likelihood, or RMLE-like intervals). citeturn25view0turn14view0turn17view1

Because the inventory and counting integrals are differentiable (Bateman equations, exponential decay integrals), and because dose ranking can be computed from gamma emission spectra and dose formulas (point-source or slab approximations), gradients can be computed either analytically or with automatic differentiation in a modern scientific stack. FISPACT explicitly documents dose-rate formulas and the ability to output dose and dominant nuclides, which provides a strong basis for dose-weighted objectives. citeturn8view1turn8view0

**Robustness to second irradiation.**  
Differentiability plus explicit initial inventories makes re-irradiation a “first-class” input: initialize the state with measured inventory from pre-irradiation counting (or the posterior from the last campaign), propagate through the next irradiation pulse, and re-optimize. Inventory tools explicitly support sequences of irradiation steps (including in sensitivity runs), aligning with this structure. citeturn8view3turn4view1

**Outputs (what a paper would show).**  
- A schedule optimizer that produces interpretable tradeoffs: “wait 2 days because ^56Mn dominates early dose and masks low-energy lines; count at 10–30 days because ^51Cr and ^58Co are information-rich and less masked,” etc.—with the interference graph as evidence. citeturn4view4turn6view3turn7view1  
- Demonstrated improvement over classic “optimum cooling time” or single-line heuristics by showing lower uncertainty on EOI activities and better recovery of dose-dominant nuclides under realistic overlaps and sum peaks.

## Explicit masking metrics and isotope ranking logic

A credible optimization workflow must be explicit about (i) isotope importance ranking and (ii) masking/interference quantification, and it must show time dependence.

### Isotope ranking logic

A scientifically defensible ranking should combine at least four components:

**Dose relevance over time.** Use either:  
- direct gamma dose-rate contribution by nuclide from activation calculations (ideal when available), or  
- a proxy proportional to gamma emission energy rate (when detailed dose modeling is unavailable), or  
- contact dose / point-source dose approximations using gamma emission spectra + attenuation (for small samples).

FISPACT-II explicitly supports gamma dose rate outputs and “dominant nuclides” sorting by dose rate. citeturn8view1turn8view0 This is the most direct way to build a **dose-centric importance list** for shutdown-dose-relevant isotopes.

**Activation significance / assay value.** Use predicted EOI activity and pathways: nuclides that are produced along high-percentage pathways (or are sensitive to specific impurities) are high priority for activation mechanism studies. FISPACT’s pathway analysis output supports this style of attribution. citeturn8view4turn6view3

**NAA-style quantitative leverage.** If the goal includes inferring elemental masses (including impurities), adopt k0/relative-NAA logic: peak area relates linearly to element amount under modeled irradiation and detection conditions, and multi-element inference relies on careful detector characterization and wide energy coverage. citeturn29view0turn29view1turn13view0 The IAEA k0 report also emphasizes that results from different gamma energies and different measurement modes can be correlated, which is a strong argument to design schedules and inference to use multiple lines and times jointly. citeturn29view3

**Feasibility and uniqueness.** Prefer nuclides with multiple usable gamma lines spread across energies, because this improves robustness to individual interferences and to efficiency-curve uncertainty; best practices explicitly recommend using a large number and wide energy range of gamma lines for detector characterization and, by extension, robust analysis. citeturn29view1turn14view0

### Grounding examples for RAFM steels and related fusion materials

Published post-irradiation gamma spectra of steels and EUROFER-like RAFM materials repeatedly show characteristic activation products such as ^60Co, ^58Co, ^54Mn, ^51Cr, ^59Fe, ^65Zn, and others, and document that trace impurities (e.g., Ta in some steels) can create prominent activation products and many sum peaks. citeturn6view0turn6view1turn6view2 In one specific EUROFER-97 example spectrum, ^182Ta is highlighted as comparatively high (present by design), with numerous sum peaks mostly originating from ^182Ta, and the paper notes that ^182Ta can contribute significantly to overall gamma dose on certain timescales. citeturn6view0

Time dependence matters: a study on irradiated iron-based structures reports ^56Mn as dominating dose rate at EOI (short-lived, rapid decay) while ^54Mn and ^60Co dominate intermediate and long cooldown, illustrating why “cooldown choice” is inseparable from “what is dose-relevant.” citeturn4view4 Benchmark work used for validating activation calculations similarly reports half-lives and dominance windows for nuclides like ^51Cr (27.70 d), ^54Mn (312.16 d), and ^58Co (70.87 d), with ^58Co dominating an extended post-irradiation period in that benchmark context. citeturn6view3

### Masking and interference metrics

A workflow should compute *predictive* masking metrics for candidate schedules \(d\), not only retrospective flags after measurement.

**Line-overlap metric (energy-resolution aware).** For a line \((i,\ell)\) at energy \(E_{i\ell}\), define overlap with a potentially interfering line \((j,m)\):
\[
\text{Overlap}_{(i\ell),(jm)} = \exp\!\left(-\frac{(E_{i\ell}-E_{jm})^2}{2\sigma_E(E)^2}\right),
\]
where \(\sigma_E\) is derived from the detector FWHM model at energy \(E\). The practical need arises because complex spectra contain multiple overlapping peaks and peak-shape assumptions matter for accurate area extraction. citeturn7view3turn6view4

**Interference-to-signal ratio (ISR) at time \(t\).** Predict:
\[
\text{ISR}_{i\ell}(t;d)=\frac{\sum_{j\neq i}\sum_m \mathbb{E}[C_{jm}(t;d)]\,\text{Overlap}_{(i\ell),(jm)} + \mathbb{E}[B_{i\ell}(t;d)]}
{\mathbb{E}[C_{i\ell}(t;d)]},
\]
where \(B\) includes Compton continuum contributions (modeled via full-spectrum response or local background). Gamma spectrometry guidance explicitly calls out “gamma-ray interferences,” “low-abundance gamma rays of high-activity nuclides,” and “limited sensitivity due to background” as routine issues. citeturn6view4

**Sum-peak / coincidence-summing risk.** Use a binary or continuous risk flag for nuclides with complex cascades in close geometry. Coincidence summing and summing-in/out are described in procedural manuals and can create additional peaks at summed energies; extended-source corrections can require advanced/Monte Carlo procedures. citeturn9view1turn28view0turn6view0

**Masking centrality and schedule sensitivity.** Compute how masking changes with \(t\): short-lived high-dose nuclides may mask early, while longer-lived nuclides dominate later. The “dominant nuclides over time” framing is directly supported by activation/dose studies and by code outputs that sort nuclides by dose rate. citeturn8view1turn4view4turn6view3

## Recommended workflow outputs and uncertainty reporting

A robust experiment-planning and analysis workflow should produce *auditably explainable artifacts* that can be used both for day-to-day planning and for publishable methods/results.

**Schedule package (irradiation, cooldown, counting).**  
- Recommended \(t_{\text{irr}}\) with saturation/threshold rationale (short-lived vs long-lived targets), and explicit inclusion of initial inventory for second irradiations. citeturn4view1turn8view3  
- Recommended cooldown/count sequence \(\{(t_k,T_k)\}\) with predicted dominant nuclides and predicted masking metrics at each \(t_k\). citeturn8view1turn4view4turn6view4

**Multi-line inference report.**  
- Estimated \(A_i^{\text{EOI}}\), \(A_i(t)\) for user-selected times, and daughter ingrowth when relevant (explicit Bateman-chain treatment). Inventory-code capabilities and k0-NAA formalisms emphasize physics-model-based conversion from measured counts to quantities of interest. citeturn4view1turn29view0turn13view0  
- A line-consistency diagnostic: activities inferred from different gamma energies should be statistically consistent after corrections; this is aligned with k0-NAA’s emphasis on combining results across gamma rays and measurement modes and recognizing correlations. citeturn29view3turn29view1

**Characteristic limits and detectability.**  
Report decision thresholds/detection limits consistent with ISO-11929-style characteristic limits, and explicitly note where overlapping peaks or peaked backgrounds inflate thresholds. citeturn5view4turn7view1

**Detector correction traceability.**  
- Pile-up/dead-time characterization curves and validation ranges; NIST procedures describe practical methods to fit loss in photopeak rate versus total count rate and to verify energy dependence. citeturn4view3  
- Coincidence/summing correction approach statement: whether using empirical factors, deterministic tools (e.g., EFFTRAN-style approaches), or Monte Carlo; deterministic coincidence-summing approaches have been verified against full Monte Carlo with small average differences in at least one reported context. citeturn28view0turn9view1

**Library and calibration metadata.**  
Track which gamma lines were used, efficiency curve parameterization, and geometry transfer assumptions; these are known to materially affect results dispersion in standardized NAA contexts. citeturn29view1turn4view2

## Validation and publishable novelty positioning

A publishable contribution in this space is strongest if it demonstrates: (i) improved inference quality for EOI activities and time-dependent dose-relevant nuclides, (ii) explicit, quantitative masking predictions that match observed failure modes (missed peaks, false positives), and (iii) robustness across first and second irradiations.

A credible validation ladder is:

**Synthetic truth studies (end-to-end).** Generate synthetic inventories and spectra using physically motivated libraries and detector response; then compare schedule choices under (a) classic single-line heuristics vs (b) Baseline A/B/C vs (c) Novel A/B. Bayesian “virtual measurement” concepts and regularized ML inversion frameworks support doing this with calibrated uncertainty intervals rather than only point estimates. citeturn5view3turn25view0

**Benchmark against published complex activation spectra.** Published post-irradiation spectra in ITER/fusion materials programs provide concrete examples of the nuclides present (including multi-line nuclides and sum peaks) and show how impurities (e.g., Ta) create prominent activation products and additional spectral structure; these are excellent test cases for masking-aware inference. citeturn6view0turn6view1turn6view2

**Dose-time dominance validation.** Validate that schedule recommendations align with known dominance windows, such as early dominance of short-lived products (e.g., ^56Mn) and later dominance of longer-lived products (e.g., ^54Mn, ^60Co) in iron-based materials, and that these shifts drive the optimizer’s cooldown choices. citeturn4view4turn6view3

**What to implement first vs later (clear separation).**  
- Implement **Baselines A/B/C first** because they are transparent, physics-based, and easy to defend in QA contexts (explicit likelihood/Poisson statistics, explicit characteristic limits, explicit correction models). citeturn17view1turn5view4turn14view0  
- Implement **Novel A/B later** because they require more computational machinery and more careful validation (posterior predictive simulation, mutual information estimation, differentiable pipelines), but they offer the clearest path to publishable novelty: adaptive scheduling and end-to-end dose-weighted, masking-aware optimization with calibrated uncertainty. citeturn20view0turn5view3turn25view0