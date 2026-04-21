"""Shared CLI command catalog metadata and renderers."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandFamily:
    """High-level grouping used for CLI help and generated docs."""

    key: str
    title: str
    description: str


@dataclass(frozen=True)
class CommandMetadata:
    """Non-parser metadata for a FluxForge CLI command."""

    family: str
    use_case: str
    example: str


@dataclass(frozen=True)
class CommandCatalogEntry:
    """Resolved CLI catalog entry merged with parser help text."""

    name: str
    family: CommandFamily
    purpose: str
    use_case: str
    example: str


COMMAND_FAMILIES: tuple[CommandFamily, ...] = (
    CommandFamily(
        key="setup",
        title="Setup and Discovery",
        description="Discover installed entrypoints, grouped command families, and next-step help.",
    ),
    CommandFamily(
        key="spectrum",
        title="Spectrum Analysis",
        description="Ingest spectra, generate plots, and perform peak and ROI analysis workflows.",
    ),
    CommandFamily(
        key="review",
        title="Artifact Review and Comparison",
        description="Query artifact trees and compare structured outputs during review work.",
    ),
    CommandFamily(
        key="validation",
        title="Validation and Governance",
        description="Run parity, crosswalk, GUI acceptance, and release-gate checks.",
    ),
    CommandFamily(
        key="activation",
        title="Activation, Inventory, and Libraries",
        description="Manage libraries and convert peak outputs into activity, rate, and inventory artifacts.",
    ),
    CommandFamily(
        key="planning",
        title="Planning and Optimization",
        description="Rank isotopes, review masking, optimize schedules, and package planning bundles.",
    ),
    CommandFamily(
        key="dosimetry",
        title="Dosimetry and Standards",
        description="Run ASTM-style dosimetry workflows and browse governed dosimetry reactions.",
    ),
    CommandFamily(
        key="reference-workflows",
        title="Reference Workflows",
        description="Replay bundled RAFM validation, benchmark, and planning-oriented worked examples.",
    ),
    CommandFamily(
        key="unfolding",
        title="Unfolding and Reporting",
        description="Build response matrices, unfold spectra, compare results, and generate plots or reports.",
    ),
    CommandFamily(
        key="k0",
        title="k0-NAA",
        description="Normalize observations, characterize detector and facility state, analyze, aggregate, and report.",
    ),
    CommandFamily(
        key="gui",
        title="GUI and Visualization",
        description="Launch the desktop GUI and generate headless plot bundles.",
    ),
)

COMMAND_FAMILY_BY_KEY = {family.key: family for family in COMMAND_FAMILIES}


COMMAND_METADATA: "OrderedDict[str, CommandMetadata]" = OrderedDict(
    [
        (
            "commands",
            CommandMetadata(
                family="setup",
                use_case="Browse the installed FluxForge command surface by family before choosing a workflow.",
                example="fluxforge commands --family spectrum",
            ),
        ),
        (
            "ingest",
            CommandMetadata(
                family="spectrum",
                use_case="Convert one measured spectrum into a normalized FluxForge artifact before downstream analysis.",
                example=(
                    "fluxforge ingest --input "
                    "examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC "
                    "--profile rafm_25cm --output rafm4_b_ingest.json"
                ),
            ),
        ),
        (
            "ingest-batch",
            CommandMetadata(
                family="spectrum",
                use_case="Ingest a directory tree of measured spectra using one shared detector/background profile.",
                example=(
                    "fluxforge ingest-batch --input-dir "
                    "examples/RAFM_irradiation/raw_gamma_spec/RAFM4 "
                    "--profile rafm_25cm --output-dir artifacts/rafm4_ingest"
                ),
            ),
        ),
        (
            "spectrum-plot",
            CommandMetadata(
                family="spectrum",
                use_case="Save a calibrated spectrum plot, optionally with background subtraction and manual ROI overlays.",
                example=(
                    "fluxforge spectrum-plot --input "
                    "examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC "
                    "--profile rafm_25cm --background-subtracted "
                    "--manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv "
                    "--output Ti-RAFM-1a_25cm_manual.png"
                ),
            ),
        ),
        (
            "peaks",
            CommandMetadata(
                family="spectrum",
                use_case="Detect peaks from an ingested spectrum artifact or compute manual ROI counts from an overlay file.",
                example="fluxforge peaks --spectrum-file rafm4_b_ingest.json --output rafm4_b_peaks.json",
            ),
        ),
        (
            "roi-analyze",
            CommandMetadata(
                family="spectrum",
                use_case="Analyze one explicit ROI with sideband or SNIP background handling.",
                example=(
                    "fluxforge roi-analyze --input rafm4_b_ingest.json "
                    "--left-keV 1170 --right-keV 1180 --output roi_analysis.json"
                ),
            ),
        ),
        (
            "roi-statistics",
            CommandMetadata(
                family="spectrum",
                use_case="Apply the same ROI definition across many spectra and summarize detector-consistency statistics.",
                example=(
                    "fluxforge roi-statistics --inputs rafm4_a_ingest.json rafm4_b_ingest.json "
                    "--left-keV 1170 --right-keV 1180 --output roi_statistics.json"
                ),
            ),
        ),
        (
            "file-query",
            CommandMetadata(
                family="review",
                use_case="Search an artifact tree for reviewable files during archive or result audits.",
                example=(
                    "fluxforge file-query --root examples/RAFM_irradiation/results "
                    "--contains validation --format json --output file_query.json"
                ),
            ),
        ),
        (
            "batch-compare",
            CommandMetadata(
                family="review",
                use_case="Compare baseline and candidate JSON/CSV tables by shared key columns and numeric deltas.",
                example=(
                    "fluxforge batch-compare --baseline baseline.csv --candidate candidate.csv "
                    "--keys sample_id,line_energy_keV --output batch_compare.json"
                ),
            ),
        ),
        (
            "parity-check",
            CommandMetadata(
                family="validation",
                use_case="Run algorithm and workflow parity fixtures against committed reference cases.",
                example="fluxforge parity-check --scope all --output parity_check.json",
            ),
        ),
        (
            "phase5-crosswalk-report",
            CommandMetadata(
                family="validation",
                use_case="Validate and summarize the Phase 5 writeup crosswalk with optional parity coverage.",
                example=(
                    "fluxforge phase5-crosswalk-report --include-parity-summary "
                    "--output phase5_crosswalk_report.json --markdown-output phase5_crosswalk_report.md"
                ),
            ),
        ),
        (
            "gui-acceptance-check",
            CommandMetadata(
                family="validation",
                use_case="Check that the GUI release checklist and probe artifacts are present and consistent.",
                example="fluxforge gui-acceptance-check --output gui_acceptance_check.json",
            ),
        ),
        (
            "phase5-release-gate",
            CommandMetadata(
                family="validation",
                use_case="Run the strict Phase 5 release-gate bundle across parity, tests, GUI evidence, and docs.",
                example="fluxforge phase5-release-gate --output phase5_release_gate.json",
            ),
        ),
        (
            "library-list",
            CommandMetadata(
                family="activation",
                use_case="Inspect bundled and user-registered nuclear-data sources before choosing a library for analysis.",
                example="fluxforge library-list --capability peak-identification",
            ),
        ),
        (
            "library-register",
            CommandMetadata(
                family="activation",
                use_case="Register a user-supplied gamma-line library under a governed alias.",
                example=(
                    "fluxforge library-register --alias my_lines "
                    "--locator /path/to/library.csv --description \"User review library\""
                ),
            ),
        ),
        (
            "library-remove",
            CommandMetadata(
                family="activation",
                use_case="Remove a previously registered governed library entry.",
                example="fluxforge library-remove --source-id user:my_lines",
            ),
        ),
        (
            "activity",
            CommandMetadata(
                family="activation",
                use_case="Convert a peak report into line-level activities for one isotope or reaction.",
                example=(
                    "fluxforge activity --peaks-file rafm4_b_peaks.json "
                    "--live-time-s 3600 --output activities.json"
                ),
            ),
        ),
        (
            "activity-review",
            CommandMetadata(
                family="activation",
                use_case="Review matched isotopes and gamma lines from one peak report and export line/isotope tables.",
                example=(
                    "fluxforge activity-review --peaks-file rafm4_b_peaks.json "
                    "--output activity_review.json --line-csv-output activity_lines.csv "
                    "--isotope-csv-output activity_isotopes.csv"
                ),
            ),
        ),
        (
            "inventory-review",
            CommandMetadata(
                family="activation",
                use_case="Propagate an activity-review inventory to arbitrary times for decay, atoms, mass, or dose review.",
                example=(
                    "fluxforge inventory-review --activity-review-file activity_review.json "
                    "--time-stop-s 86400 --plot-output inventory_plot.png --output inventory_review.json"
                ),
            ),
        ),
        (
            "second-irradiation-plan",
            CommandMetadata(
                family="planning",
                use_case="Rank second-irradiation candidates from an inventory seed, schedule, and candidate definition.",
                example=(
                    "fluxforge second-irradiation-plan --inventory-file inventory_review.json "
                    "--schedule-file schedule.json --candidates-file candidates.json "
                    "--output second_irradiation_plan.json"
                ),
            ),
        ),
        (
            "ffexp-export",
            CommandMetadata(
                family="planning",
                use_case="Package Phase 6 review artifacts into a portable `.ffexp` benchmark bundle.",
                example=(
                    "fluxforge ffexp-export --activity-review-file activity_review.json "
                    "--inventory-review-file inventory_review.json --output benchmark_bundle.ffexp"
                ),
            ),
        ),
        (
            "isotope-priority",
            CommandMetadata(
                family="planning",
                use_case="Rank isotopes of interest from an activity-review bundle before planning follow-on irradiations.",
                example=(
                    "fluxforge isotope-priority --activity-review-file activity_review.json "
                    "--csv-output isotope_priority.csv --output isotope_priority.json"
                ),
            ),
        ),
        (
            "masking-review",
            CommandMetadata(
                family="planning",
                use_case="Rank masking interactions and alternate-line guidance from activity-review outputs.",
                example=(
                    "fluxforge masking-review --activity-review-file activity_review.json "
                    "--csv-output masking_lines.csv --output masking_review.json"
                ),
            ),
        ),
        (
            "optimization-sweep",
            CommandMetadata(
                family="planning",
                use_case="Score candidate irradiation schedules using DI-FOM, FIM, MWDCS, or advanced objectives.",
                example=(
                    "fluxforge optimization-sweep --activity-review-file activity_review.json "
                    "--objective di-fom --output optimization_sweep.json"
                ),
            ),
        ),
        (
            "rates",
            CommandMetadata(
                family="activation",
                use_case="Convert line activities into reaction-rate estimates using irradiation history information.",
                example=(
                    "fluxforge rates --lines-file activity_lines.json --duration-s 3600 "
                    "--output rates.json"
                ),
            ),
        ),
        (
            "astm-e2005",
            CommandMetadata(
                family="dosimetry",
                use_case="Run the ASTM E2005 reactor dosimetry workflow from a structured plan file.",
                example="fluxforge astm-e2005 --plan-file my_astm_e2005_plan.json --output astm_e2005.json",
            ),
        ),
        (
            "astm-e261",
            CommandMetadata(
                family="dosimetry",
                use_case="Run the ASTM E261 reactor dosimetry workflow from the shipped plan format.",
                example="fluxforge astm-e261 --plan-file examples/astm_e261_plan.json --output astm_e261.json",
            ),
        ),
        (
            "astm-e262",
            CommandMetadata(
                family="dosimetry",
                use_case="Run the ASTM E262 thermal neutron fluence workflow from a plan file.",
                example="fluxforge astm-e262 --plan-file my_astm_e262_plan.json --output astm_e262.json",
            ),
        ),
        (
            "astm-e3376",
            CommandMetadata(
                family="dosimetry",
                use_case="Run the ASTM E3376 HPGe detection workflow from a plan file.",
                example="fluxforge astm-e3376 --plan-file my_astm_e3376_plan.json --output astm_e3376.json",
            ),
        ),
        (
            "rafm-validate",
            CommandMetadata(
                family="reference-workflows",
                use_case="Replay the committed RAFM raw-spectrum validation workflow against bundled reference assets.",
                example=(
                    "fluxforge rafm-validate --example-root examples/RAFM_irradiation "
                    "--results-root /tmp/rafm_validation --no-fail"
                ),
            ),
        ),
        (
            "rafm-qg-benchmark",
            CommandMetadata(
                family="reference-workflows",
                use_case="Process committed Quantum Gold RAFM data into reaction-rate and unfolding outputs.",
                example=(
                    "fluxforge rafm-qg-benchmark --example-root examples/RAFM_irradiation "
                    "--results-root /tmp/rafm_qg"
                ),
            ),
        ),
        (
            "rafm-compare-branches",
            CommandMetadata(
                family="reference-workflows",
                use_case="Compare completed raw-branch and Quantum Gold RAFM result trees.",
                example=(
                    "fluxforge rafm-compare-branches --raw-results-root raw_results "
                    "--qg-results-root qg_results --output-root branch_compare"
                ),
            ),
        ),
        (
            "phase6-ldrd-worked-example",
            CommandMetadata(
                family="reference-workflows",
                use_case="Generate the bundled Phase 6 worked example outputs for one committed RAFM sample.",
                example=(
                    "fluxforge phase6-ldrd-worked-example --sample-id RAFM4-C_15dEOI "
                    "--output-root /tmp/phase6_ldrd_worked_example"
                ),
            ),
        ),
        (
            "phase6-ldrd-second-irradiation-repo",
            CommandMetadata(
                family="reference-workflows",
                use_case="Build the Phase 6 second-irradiation decision repository bundle for one RAFM sample.",
                example=(
                    "fluxforge phase6-ldrd-second-irradiation-repo --sample-id RAFM4-C_15dEOI "
                    "--output-root /tmp/phase6_second_irradiation_repo"
                ),
            ),
        ),
        (
            "response",
            CommandMetadata(
                family="unfolding",
                use_case="Build a response matrix from cross sections, number densities, and group boundaries.",
                example=(
                    "fluxforge response --cross-section-file cross_sections.json "
                    "--number-densities-file number_densities.json --boundaries-file boundaries.json "
                    "--output response.json"
                ),
            ),
        ),
        (
            "unfold",
            CommandMetadata(
                family="unfolding",
                use_case="Infer a neutron spectrum using GLS, GRAVEL, MLEM, MAXED, RMLE, or ML-seeded workflows.",
                example=(
                    "fluxforge unfold --rates-file rates.json --response-file response.json "
                    "--method gravel --output spectrum.json"
                ),
            ),
        ),
        (
            "compare",
            CommandMetadata(
                family="unfolding",
                use_case="Compare an unfolded spectrum against a trusted reference spectrum.",
                example=(
                    "fluxforge compare --unfold-file spectrum.json --truth-flux-file truth_flux.json "
                    "--output validation.json"
                ),
            ),
        ),
        (
            "report",
            CommandMetadata(
                family="unfolding",
                use_case="Assemble a report bundle from one or more analysis artifacts.",
                example=(
                    "fluxforge report --spectrum-file spectrum.json --peaks-file peaks.json "
                    "--output report.json"
                ),
            ),
        ),
        (
            "k0-normalize",
            CommandMetadata(
                family="k0",
                use_case="Convert a peak report into normalized k0 peak observations.",
                example=(
                    "fluxforge k0-normalize --peaks-file rafm4_b_peaks.json "
                    "--output k0_observations.json"
                ),
            ),
        ),
        (
            "k0-detector",
            CommandMetadata(
                family="k0",
                use_case="Fit a reusable detector-characterization artifact from calibration points.",
                example=(
                    "fluxforge k0-detector --points-file detector_points.csv --detector-id hpge_demo "
                    "--reference-position-mm 250 --output detector_characterization.json"
                ),
            ),
        ),
        (
            "k0-facility",
            CommandMetadata(
                family="k0",
                use_case="Characterize a thermal irradiation facility from a bare triple-monitor dataset.",
                example="fluxforge k0-facility --input facility_input.json --output facility_characterization.json",
            ),
        ),
        (
            "k0-analyze",
            CommandMetadata(
                family="k0",
                use_case="Run a first-pass k0 analysis from normalized observations and a characterized facility.",
                example=(
                    "fluxforge k0-analyze --observations-file k0_observations.json "
                    "--facility-file facility_characterization.json --sample-mass-g 0.5 "
                    "--output k0_analysis.json"
                ),
            ),
        ),
        (
            "k0-aggregate",
            CommandMetadata(
                family="k0",
                use_case="Aggregate multiple k0 analysis bundles across measurements or irradiations.",
                example=(
                    "fluxforge k0-aggregate --analysis-files sample_a.json sample_b.json "
                    "--output k0_aggregation.json"
                ),
            ),
        ),
        (
            "k0-qaqc",
            CommandMetadata(
                family="k0",
                use_case="Evaluate blank and CRM QA/QC performance for a k0 analysis plan.",
                example="fluxforge k0-qaqc --plan-file k0_qaqc_plan.json --output k0_qaqc.json",
            ),
        ),
        (
            "k0-report",
            CommandMetadata(
                family="k0",
                use_case="Build a richer k0 report bundle with optional aggregation and QA/QC context.",
                example="fluxforge k0-report --analysis-file k0_analysis.json --output k0_report.json",
            ),
        ),
        (
            "k0-import-kayzero",
            CommandMetadata(
                family="k0",
                use_case="Import a user-supplied Kayzero folder or archive into governed FluxForge JSON.",
                example=(
                    "fluxforge k0-import-kayzero --input /path/to/kayzero_library "
                    "--output kayzero_k0_library.json"
                ),
            ),
        ),
        (
            "reactions",
            CommandMetadata(
                family="dosimetry",
                use_case="Browse bundled IRDFF-II dosimetry reactions by category or target nuclide.",
                example="fluxforge reactions --category thermal --format table",
            ),
        ),
        (
            "gui",
            CommandMetadata(
                family="gui",
                use_case="Launch the desktop GUI for interactive spectrum review and ROI editing.",
                example="fluxforge gui --project-dir .",
            ),
        ),
        (
            "plots",
            CommandMetadata(
                family="gui",
                use_case="Generate headless plot bundles from analysis artifacts or bundled example inputs.",
                example="fluxforge plots --example --output-dir output/plots",
            ),
        ),
    ]
)


def _get_subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("FluxForge CLI parser is missing subcommands")


def build_command_catalog(parser: argparse.ArgumentParser) -> list[CommandCatalogEntry]:
    """Build the resolved command catalog from parser help and static metadata."""

    subparsers = _get_subparsers_action(parser)
    parser_names = [choice.dest for choice in subparsers._choices_actions]
    parser_name_set = set(parser_names)
    metadata_name_set = set(COMMAND_METADATA)
    if parser_name_set != metadata_name_set:
        missing_metadata = sorted(parser_name_set - metadata_name_set)
        missing_parser = sorted(metadata_name_set - parser_name_set)
        raise RuntimeError(
            "Command catalog metadata is out of sync with parser commands: "
            f"missing metadata={missing_metadata}, missing parser entries={missing_parser}"
        )
    catalog: list[CommandCatalogEntry] = []
    for choice in subparsers._choices_actions:
        metadata = COMMAND_METADATA[choice.dest]
        family = COMMAND_FAMILY_BY_KEY[metadata.family]
        catalog.append(
            CommandCatalogEntry(
                name=choice.dest,
                family=family,
                purpose=(choice.help or "").strip(),
                use_case=metadata.use_case,
                example=metadata.example,
            )
        )
    return catalog


def filter_command_catalog(
    entries: list[CommandCatalogEntry], family: str | None = None
) -> list[CommandCatalogEntry]:
    """Return only entries from one family when requested."""

    if family is None:
        return list(entries)
    return [entry for entry in entries if entry.family.key == family]


def group_command_catalog(
    entries: list[CommandCatalogEntry],
) -> list[tuple[CommandFamily, list[CommandCatalogEntry]]]:
    """Group catalog entries by family while preserving family order."""

    grouped: list[tuple[CommandFamily, list[CommandCatalogEntry]]] = []
    for family in COMMAND_FAMILIES:
        family_entries = [entry for entry in entries if entry.family.key == family.key]
        if family_entries:
            grouped.append((family, family_entries))
    return grouped


def render_command_catalog_text(
    entries: list[CommandCatalogEntry], family: str | None = None
) -> str:
    """Render the grouped command catalog as plain text for CLI output."""

    filtered = filter_command_catalog(entries, family=family)
    lines = [
        "FluxForge command catalog",
        "",
        "Run `fluxforge <command> --help` for command-specific flags and argument details.",
        "",
    ]
    for command_family, family_entries in group_command_catalog(filtered):
        lines.append(f"{command_family.title} [{command_family.key}]")
        lines.append(command_family.description)
        for entry in family_entries:
            lines.append(f"  {entry.name}")
            lines.append(f"    Purpose : {entry.purpose}")
            lines.append(f"    Use case: {entry.use_case}")
            lines.append(f"    Example : {entry.example}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_command_catalog_markdown(
    entries: list[CommandCatalogEntry], family: str | None = None
) -> str:
    """Render the grouped command catalog as committed Markdown."""

    filtered = filter_command_catalog(entries, family=family)
    lines = [
        "# FluxForge CLI Reference",
        "",
        "This reference is generated from the FluxForge CLI parser metadata.",
        "Refresh it with `python tools/generate_cli_reference.py` after updating CLI commands or catalog metadata.",
        "",
        "Use `fluxforge commands` for the terminal view and `fluxforge <command> --help` for full flag details.",
        "",
    ]
    for command_family, family_entries in group_command_catalog(filtered):
        lines.extend(
            [
                f"## {command_family.title}",
                "",
                command_family.description,
                "",
            ]
        )
        for entry in family_entries:
            lines.extend(
                [
                    f"### `{entry.name}`",
                    "",
                    f"- Purpose: {entry.purpose}",
                    f"- Common use case: {entry.use_case}",
                    f"- Detailed help: `fluxforge {entry.name} --help`",
                    "",
                    "```bash",
                    entry.example,
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_top_level_family_help(entries: list[CommandCatalogEntry]) -> str:
    """Render a grouped family summary for the root `fluxforge --help` output."""

    blocks: list[str] = []
    for command_family, family_entries in group_command_catalog(entries):
        command_list = ", ".join(entry.name for entry in family_entries)
        blocks.append(
            f"{command_family.title} [{command_family.key}]\n"
            f"  {command_family.description}\n"
            f"  Commands: {command_list}"
        )
    return "\n\n".join(blocks)
