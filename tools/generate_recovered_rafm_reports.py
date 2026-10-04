"""Generate the five recovered samples' reports/plots from physical raw analysis.

Existing sample outputs are never replaced. Reference counts/activity overlays
are prohibited; Quantum Gold reports enter only comparison/reporting stages.
"""

import argparse
import hashlib
import json
from pathlib import Path

from fluxforge.analysis import flux_wire_analysis
from fluxforge.examples import rafm_workflow as workflow
from fluxforge.io.flux_wire import read_raw_asc


SAMPLES = {
    "Cu-Cd-RAFM-1_25cm",
    "Fe-Cd-RAFM-1_0cm",
    "RAFM3-A_24hrEOI",
    "RAFM3-A_300sEOI",
    "RAFM3-A_4dEOI",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    root = args.root.resolve()
    paths = workflow.default_paths(root / "examples/RAFM_irradiation")
    metadata = workflow.load_rafm_example_metadata(paths.example_root)
    metadata.config["flux_wire_counting_method"] = "iec_tiered"
    metadata.config["generic_targeted_counting_method"] = "iec_tiered"
    files = workflow.discover_input_files(paths)
    pairs, _, _ = workflow.pair_input_files(
        files["raw"], files["qg"], metadata.pairing_aliases
    )
    selected = [(r, q, k) for r, q, k in pairs if r.stem in SAMPLES]
    if {r.stem for r, _, _ in selected} != SAMPLES or any(
        q is None for _, q, _ in selected
    ):
        raise ValueError("Recovered raw/report pairs incomplete")
    for raw, _, _ in selected:
        for relative in (
            f"analysis_json/{raw.stem}.json",
            f"reports/{raw.stem}_comparison.txt",
            f"plots/comparisons/{raw.stem}_vs_qg.png",
        ):
            if (paths.results_root / relative).exists():
                raise FileExistsError(paths.results_root / relative)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Raw report generation cannot substitute QG counts or activities"
        )

    workflow.apply_generic_qg_report_parity = forbidden
    workflow.reference_isotope_payload = forbidden
    flux_wire_analysis.apply_qg_report_parity = forbidden
    tree = workflow.ensure_results_tree(paths.results_root)
    library, half_lives = workflow.build_generic_gamma_library(metadata)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=workflow.workflow_profile_energy_calibration(
            metadata.config
        ),
        profile_name=metadata.config["profile_name"],
    ).spectrum
    receipt = dict(
        scientific_admission=False,
        source_sha256={
            p.relative_to(root).as_posix(): sha(p)
            for p in (root / "src/fluxforge").rglob("*.py")
        },
        counting_method="iec_tiered",
        samples=[],
        limits=[
            "Existing common profile retained; Fe-Cd near-contact efficiency and shared calibration/efficiency covariance are not qualified by this replay."
        ],
    )
    for raw, qg, key in selected:
        if raw.parent.name == "flux_wires":
            artifact = workflow.analyze_flux_wire_sample(
                raw, metadata, paths, tree, background, qg, key
            )
        else:
            artifact = workflow.analyze_generic_sample(
                raw, metadata, paths, tree, library, half_lives, background, qg
            )
        assert artifact["validation"]["reference_used_for_analysis"] is False
        assert artifact["validation"]["comparison_basis"] == "raw_estimate_vs_report"
        receipt["samples"].append(
            dict(
                sample=raw.stem,
                raw_sha256=sha(raw),
                qg_sha256=sha(qg),
                validation=artifact["validation"],
                outputs_sha256={
                    p.relative_to(root).as_posix(): sha(p)
                    for directory in tree.values()
                    if directory.is_dir()
                    for p in directory.glob(raw.stem + "*")
                    if p.is_file()
                },
            )
        )
        print(raw.stem, "reports and plots generated", flush=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
