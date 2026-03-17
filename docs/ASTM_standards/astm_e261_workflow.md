# ASTM E261 Workflow

FluxForge includes a built-in ASTM E261 reactor dosimetry workflow for converting measured activation data into:

- end-of-irradiation activity,
- reaction rate,
- target-nuclei inventory,
- fluence rate, and
- integrated fluence.

The implementation is intentionally CLI-first and GUI-accessible. The GUI Standards tab calls the same backend used by the `astm-e261` terminal command.

## Scope

The workflow is designed for ASTM E261-style radioactivation reduction where the user already has an effective spectrum-averaged monitor cross section for each dosimeter reaction.

FluxForge reuses existing activation primitives instead of duplicating the math:

- `fluxforge.physics.activation.GammaLineMeasurement`
- `fluxforge.physics.activation.irradiation_buildup_factor()`
- `fluxforge.physics.activation.activation_study_metrics()`

## Input plan

The workflow consumes a JSON plan with schema `fluxforge.astm_e261_plan.v1`.

A minimal example is provided in [examples/astm_e261_plan.json](../examples/astm_e261_plan.json).

### Irradiation block

You may specify either:

- `irradiation.segments`: a list of `{duration_s, relative_power}` entries, or
- `irradiation.duration_s`: a single constant-power irradiation.

The power-weighted duration is used when reporting total fluence:

$$
\Phi = \dot{\Phi} \sum_i t_i p_i
$$

where $t_i$ is the segment duration and $p_i$ is the relative power for that segment.

### Measurement block

Each measurement should include:

- count data: `net_counts`, `live_time_s`
- counting corrections: `cooling_time_s`, optional `dead_time_fraction`
- detector terms: `efficiency`, `gamma_intensity`
- decay term: `half_life_s`
- target inventory terms: `sample_mass_g`, `atomic_mass_g_mol`, optional `isotopic_abundance`, `mass_fraction`, `sample_purity`, `atoms_per_formula_unit`
- dosimetry term: `effective_cross_section_barn`

Optional correction factors are supported:

- `self_shielding_factor`
- `cover_correction_factor`
- `geometry_factor`
- `astm_correction_factor`

These are multiplied together and applied in the fluence denominator.

## Implemented relationships

### 1. Activity at the reference time

FluxForge uses the existing counted-line activity model:

$$
A_{ref} = \frac{C}{\varepsilon I_\gamma \left(1-e^{-\lambda t_c}\right)/\lambda} e^{\lambda t_d}
$$

where:

- $C$ is dead-time-corrected net counts,
- $\varepsilon$ is full-energy peak efficiency,
- $I_\gamma$ is gamma emission probability,
- $t_c$ is counting live time, and
- $t_d$ is cooling time from EOI to count start.

### 2. Reaction rate

Using the irradiation buildup factor $B$:

$$
R = \frac{A_{EOI}}{B}
$$

For segmented irradiation histories, FluxForge evaluates $B$ with the same multi-segment helper already used by the existing activation workflow.

### 3. Number of target nuclei

$$
N = \frac{m w p \theta}{M} N_A n
$$

where:

- $m$ is monitor mass,
- $w$ is analyte mass fraction,
- $p$ is sample purity,
- $\theta$ is isotopic abundance,
- $M$ is molar mass, and
- $n$ is target atoms per formula unit.

### 4. Fluence rate

$$
\dot{\Phi} = \frac{R}{N \bar{\sigma} C}
$$

where $\bar{\sigma}$ is the effective spectrum-averaged cross section and $C$ is the combined correction factor.

### 5. Total fluence

$$
\Phi = \dot{\Phi} t_{eq}
$$

with $t_{eq}$ taken from the power-weighted irradiation duration unless `irradiation.fluence_duration_s` is explicitly provided.

## Command-line usage

Example:

- input plan: [examples/astm_e261_plan.json](../examples/astm_e261_plan.json)
- command: `fluxforge astm-e261 --plan-file examples/astm_e261_plan.json --output astm_e261.json`

The output is a JSON bundle with schema `fluxforge.astm_e261_result.v1`.

## GUI usage

Open the Standards tab and use the **ASTM E261 workflow** panel:

1. choose a plan file,
2. choose an output artifact path,
3. run the workflow, and
4. load the preview.

The GUI preview summarizes reaction IDs, fluence rates, and integrated fluences.

## Notes

- This workflow expects effective reaction cross sections as inputs; it does not replace the response-matrix and unfolding tools.
- Use the existing IRDFF/ASTM dosimetry browser and unfolding paths when the spectrum-averaged cross sections must be derived from a spectrum solution.
- Use the existing cover-correction and self-shielding modules when those corrections need to be computed upstream.
